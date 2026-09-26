"""How much does "new-fire lift" depend on how "new" is defined?

A fire pixel is counted as new if no fire was detected within radius r (px,
exact euclidean distance) of it in the history window: the union of the last W
y_fire_3d slices that end on the forecast issue day (W slices cover W+2 days of
detections), land-masked. The script sweeps r in RADII and W in WINDOWS and
reports, for each cell, the share of fire pixels that still qualify and the
model's lift at several top-k fractions.

Uses the same patch protocol as evaluate.py (seed, fire-density threshold,
land fraction). Note that evaluate.py's headline "new fire" uses a different,
narrower reference -- one history slice and a 3-iteration cross-shaped
dilation (taxicab radius 3) -- so its headline new-fire lift is not exactly a
cell of this grid (nearest cell: r=3 px, W=1).

Usage:
    SMOLDER_DATA=/path/to/cubes python -m smolder.evaluation.newfire_definition_sweep
"""
import os
import sys

import numpy as np
import torch
import zarr
from scipy import ndimage

from smolder.models.conv_lstm_lit_dual import ConvLSTMLitDual
from smolder.data.io import daily_cube, open_zarr_root
from smolder.data.zarr_dual_datamodule import DualPatchConfig, DualWindowDataset

# CKPTS may list several checkpoints (colon-separated); each is scored on the
# same patches. Default: the released model.
CKPTS = [p for p in os.environ.get("CKPTS", os.environ.get("CKPT", "checkpoints/smolder_swa.ckpt")).split(":") if p.strip()]
CKPT = CKPTS[0]
PATCH = int(os.environ.get("PATCH", 384))
N_PATCH = int(os.environ.get("N_PATCH", 600))
EVAL_YEAR = os.environ.get("EVAL_YEAR", "2020")
Y_KEY = os.environ.get("Y_KEY", "y_fire_3d")
_BASE_PATCH, _BASE_MIN_POS = 256, 20
MIN_POS = max(1, round(_BASE_MIN_POS * (PATCH / _BASE_PATCH) ** 2))
OFFS = {"2015": 0, "2016": 365, "2017": 731, "2018": 1096, "2019": 1461, "2020": 1826}

# px (= km). 3 matches evaluate.py; 0 means "same pixel only".
RADII = [0, 1, 3, 5, 10, 20, 40]
# number of y_fire_3d history slices ending on the issue day; 1 matches evaluate.py.
WINDOWS = [1, 3, 7, 14, 30, 90]
TOPK = [0.001, 0.002, 0.005, 0.01, 0.02]
REPORT_K = 0.005     # headline operating point (where new-fire lift peaks)


def lift_at(prob, truth, land, f):
    v = prob[land]
    if v.size == 0 or truth.sum() == 0:
        return np.nan, np.nan
    k = max(1, int(round(f * v.size)))
    thr = np.partition(v, -k)[-k]
    sel = land & (prob >= thr)
    tp = float((truth & sel).sum())
    tpr = tp / max(float(truth.sum()), 1)
    prec = tp / max(float(sel.sum()), 1)
    base = float(truth.sum()) / float(land.sum())
    return tpr, (prec / base if base > 0 else np.nan)


def main():
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = open_zarr_root(daily_cube(EVAL_YEAR))
    yarr = g[Y_KEY]
    T_max = yarr.shape[0]

    models = []
    for cp in CKPTS:
        mm = ConvLSTMLitDual.load_from_checkpoint(cp, map_location=dev)
        mm.eval()
        mm.to(dev)
        models.append(mm)
        print(f"[info] loaded {os.path.basename(cp)}", flush=True)
    # Each member gets its own dataset because the specialist was trained
    # WITHOUT fire-history channels -- feeding it the main model's input would
    # be a channel mismatch. Datasets are built with identical seed/patch/
    # min_pos, so ds[i] refers to the same patch for every member.
    names = [f"m{i}:{os.path.basename(c)[:28]}" for i, c in enumerate(CKPTS)]
    if len(models) > 1:
        names.append("prob-mean")
    print(f"[info] {len(models)} model(s) on {dev}, patch={PATCH}, "
          f"min_pos={MIN_POS}, n={N_PATCH}", flush=True)

    def mk_ds(fh):
        """fh=False builds the specialist's fire-history-free input."""
        return DualWindowDataset(DualPatchConfig(
            zarr_paths=(daily_cube(EVAL_YEAR),),
            stats_path="channel_stats_2015_2018.json",
            slow_cube_path="cube_slow_8day.zarr", day_offset=OFFS[EVAL_YEAR],
            patch_size=PATCH, samples_per_epoch=N_PATCH * 3, seed=21,
            min_pos_pixels=MIN_POS, pos_frac=1.0, deterministic=True,
            fire_history=fh, fire_history_lags=(3, 4, 5), fire_history_distance=fh,
            use_lightning=os.environ.get("USE_LIGHTNING", "0") == "1" and not fh,
            use_slope_aspect=os.environ.get("USE_SLOPE_ASPECT", "0") == "1" and not fh,
            use_wind_dir=os.environ.get("USE_WIND_DIR", "0") == "1" and not fh))

    # NO_FIREHIST lists which members were trained without fire history
    # (colon-separated indices), so each gets the input layout it expects.
    nofh = {int(x) for x in os.environ.get("NO_FIREHIST", "").split(":") if x.strip()}
    ds_fh = mk_ds(True)
    ds_nofh = mk_ds(False) if nofh else None
    ds = ds_fh

    # accumulators keyed by (radius, window); lifts additionally per member
    n_new = {(r, w): 0 for r in RADII for w in WINDOWS}
    lifts = {nm: {(r, w): {f: [] for f in TOPK} for r in RADII for w in WINDOWS}
             for nm in names}
    tot_fire = 0
    kept = 0

    for i in range(N_PATCH * 3):
        if kept >= N_PATCH:
            break
        b = ds[i]
        land = b["mask"].numpy() > 0.5
        truth = (b["y"][-1].numpy() > 0) & land
        if truth.sum() < MIN_POS or land.mean() < 0.5:
            continue
        y0, x0, t_end = int(b["y0"]), int(b["x0"]), int(b["t_end"])

        b_nofh = ds_nofh[i] if ds_nofh is not None else None
        probs = {}
        with torch.no_grad():
            for mi, mm in enumerate(models):
                bb = b_nofh if mi in nofh else b
                probs[names[mi]] = torch.sigmoid(mm.forward_seq(
                    bb["x_slow"].unsqueeze(0).to(dev),
                    bb["x_fast"].unsqueeze(0).to(dev),
                    bb["x_cat"].unsqueeze(0).to(dev))[:, -1])[0].cpu().numpy()
        if len(models) > 1:
            probs["prob-mean"] = np.mean([probs[names[mi]] for mi in range(len(models))], axis=0)

        # The last-step target is y_fire_3d[t_end-1] (fire on days t_end..t_end+2).
        # History slices end at y_fire_3d[t_end-4] (fire up to day t_end-1, the
        # issue day). A window of W slices covers the W+2 days up to that day.
        t_hi = t_end - 1 - 3
        hist_cache = {}
        for w in WINDOWS:
            t_lo = max(0, t_hi - w + 1)
            if t_hi < 0:
                hist_cache[w] = np.zeros_like(land)
                continue
            span = np.asarray(yarr[t_lo:t_hi + 1, y0:y0 + PATCH, x0:x0 + PATCH]) > 0
            # ocean is labelled fire=1 in y_fire_3d -- must mask before any
            # distance transform, or coastal "nearest fire" collapses to the sea
            hist_cache[w] = span.any(axis=0) & land

        for w in WINDOWS:
            recent = hist_cache[w]
            if recent.any():
                dist = ndimage.distance_transform_edt(~recent)
            else:
                dist = np.full(land.shape, np.inf, np.float32)
            for r in RADII:
                known = dist <= r
                new_fire = truth & ~known
                cnt = int(new_fire.sum())
                n_new[(r, w)] += cnt
                if cnt >= 3:
                    for nm in names:
                        for f in TOPK:
                            _, lf = lift_at(probs[nm], new_fire, land, f)
                            lifts[nm][(r, w)][f].append(lf)

        tot_fire += int(truth.sum())
        kept += 1
        if kept % 50 == 0:
            print(f"  {kept}...", flush=True)

    print(f"\n*** {kept} patches, {tot_fire:,} fire px total "
          f"({tot_fire/kept:.0f}/patch) ***")

    print("\n=== share of fire pixels still counted as NEW (%) ===")
    print("            " + "".join(f"{'W='+str(w):>9}" for w in WINDOWS))
    for r in RADII:
        cells = "".join(f"{100.0*n_new[(r,w)]/max(tot_fire,1):8.1f} " for w in WINDOWS)
        tag = "  <-- headline radius" if r == 3 else ""
        print(f"  r={r:<3}px  {cells}{tag}")
    print("            " + "".join(f"{'^headline' if w==1 else '':>9}" for w in WINDOWS))

    for nm in names:
        print(f"\n=== NEW-fire lift @ top-{100*REPORT_K:g}%  --  {nm} ===")
        print("            " + "".join(f"{'W='+str(w):>9}" for w in WINDOWS))
        for r in RADII:
            cells = ""
            for w in WINDOWS:
                v = lifts[nm][(r, w)][REPORT_K]
                cells += f"{np.nanmean(v):8.2f} " if v else f"{'--':>8} "
            print(f"  r={r:<3}px  {cells}")

    if len(names) > 1:
        print(f"\n=== FAR-FIELD verdict (r>=10px = genuinely new ignition) ===")
        print(f"  {'model':<34} " + "".join(f"r={r}px".rjust(9) for r in (10, 20, 40)))
        for nm in names:
            cells = "".join(f"{np.nanmean(lifts[nm][(r,1)][REPORT_K]):9.2f}" for r in (10, 20, 40))
            print(f"  {nm:<34} {cells}")
        print("  (lift <1.0 = WORSE than random; the main model scores ~0.1x here)")

    print(f"\n=== mean NEW-fire px per patch ===")
    print("            " + "".join(f"{'W='+str(w):>9}" for w in WINDOWS))
    for r in RADII:
        print(f"  r={r:<3}px  " + "".join(f"{n_new[(r,w)]/max(kept,1):8.1f} " for w in WINDOWS))

    import json
    out = dict(eval_year=int(EVAL_YEAR), n_patches=kept, fire_px=tot_fire, report_k=REPORT_K,
               radii_px=RADII, windows=WINDOWS,
               share_new_pct={str(w): [100.0 * n_new[(r, w)] / max(tot_fire, 1) for r in RADII] for w in WINDOWS},
               lift={nm: {str(w): [float(np.nanmean(lifts[nm][(r, w)][REPORT_K])) if lifts[nm][(r, w)][REPORT_K] else None
                                   for r in RADII] for w in WINDOWS} for nm in names})
    with open(f"newfire_sweep_{EVAL_YEAR}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote newfire_sweep_{EVAL_YEAR}.json")

    cur = 100.0 * n_new[(3, 1)] / max(tot_fire, 1)
    strict = 100.0 * n_new[(20, 30)] / max(tot_fire, 1)
    print(f"\n  headline rule (r=3px, W=1) counts {cur:.1f}% of fire px as NEW")
    print(f"  strict rule   (r=20px, W=30) counts {strict:.1f}%")
    if cur > 0:
        print(f"  => {100.0*(1-strict/cur):.0f}% of what the headline rule calls 'new fire' is "
              f"within 20km of, or a re-detection of, fire from the past 30 days")


if __name__ == "__main__":
    main()
