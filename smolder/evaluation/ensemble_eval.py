"""Evaluate an ENSEMBLE of dual-branch checkpoints on the 2020 test year.

Same protocol/metric definitions as operational_stats_dual.py (pooled AUC-PR /
ROC-AUC, plus TPR/FPR/lift vs top-k for ALL fire and for NEW fire only), but
runs several checkpoints over the SAME patches and combines their scores.

Why ensembling is worth a shot here specifically: this session established that
val_ap on 2019 does not reliably predict 2020 test AUC-PR (`trimmed` won val_ap
by 4% and lost on real test), i.e. a big chunk of the run-to-run difference is
variance, not skill. Averaging decorrelated members is the standard fix and
needs no retraining.

TWO combination rules are reported, because they are NOT equivalent here:

  * prob-mean  -- plain mean of sigmoid outputs. Sensitive to calibration
    differences between members, and members trained with different pos_weight
    schedules / recipes genuinely have different raw output distributions
    (CLAUDE.md documents that recalibration is per-checkpoint for exactly this
    reason: ps384_final's raw mean|fire 0.87 vs the annealed runs' 0.70/0.66).
    A more confident member can dominate the mean regardless of being better.

  * rank-mean  -- map each member's scores through its OWN empirical CDF to
    percentiles, then average. This is calibration-invariant, so members
    contribute equally by their ranking quality alone. Usually the right
    default when members disagree on calibration, which is the case here.

Pooled metrics use a GLOBAL quantile transform (fit on a subsample of that
member's pooled scores) so scores stay comparable across patches. The per-patch
top-k table uses WITHIN-patch ranks, which is the natural unit there since
top-k is itself computed within a patch.

All checkpoints must share one input-feature config (the script builds a single
dataset). Mixing e.g. `trimmed` (slope/aspect/fuel_age/wind_dir) with
ps384_final (none of those) would need two datasets -- the script fails loudly
on a channel mismatch rather than silently feeding wrong inputs.

Usage:
    CKPTS="a.ckpt:b.ckpt" PATCH=384 EVAL_YEAR=2020 N_PATCH=1500 \
      OUT_TAG=_ens python ensemble_eval.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import zarr
from scipy import ndimage
from scipy.stats import rankdata

from conv_lstm_lit_dual import ConvLSTMLitDual
from zarr_dual_datamodule import DualPatchConfig, DualWindowDataset

CKPTS = [p for p in os.environ["CKPTS"].split(":") if p.strip()]
N_PATCH = int(os.environ.get("N_PATCH", 200))
PATCH = int(os.environ.get("PATCH", 384))
# Density-scaled fire filter, same convention as operational_stats_dual.py --
# a fixed absolute pixel count makes different patch sizes non-comparable.
_BASE_PATCH, _BASE_MIN_POS = 256, 20
MIN_POS = max(1, round(_BASE_MIN_POS * (PATCH / _BASE_PATCH) ** 2))
DILATE = int(os.environ.get("DILATE", 3))
KS = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]
EVAL_YEAR = os.environ.get("EVAL_YEAR", "2020")
OUT_TAG = os.environ.get("OUT_TAG", "_ensemble")
QUANTILE_SAMPLE = int(os.environ.get("QUANTILE_SAMPLE", 5_000_000))
LAT0, PX = -9.005000113999998, 0.01
SEAS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
        6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def tpr_fpr(prob, truth, land, f):
    v = prob[land]
    if v.size == 0 or truth.sum() == 0:
        return (np.nan,) * 3
    k = max(1, int(round(f * v.size)))
    thr = np.partition(v, -k)[-k]
    sel = land & (prob >= thr)
    tp = float((truth & sel).sum())
    fn = float((truth & ~sel).sum())
    neg = land & ~truth
    fp = float((neg & sel).sum())
    tn = float((neg & ~sel).sum())
    prec = tp / max(float(sel.sum()), 1)
    base = float(truth.sum()) / float(land.sum())
    return tp / max(tp + fn, 1), fp / max(fp + tn, 1), (prec / base if base > 0 else np.nan)


def rank01(a):
    """Within-array percentile rank in [0,1] with ties AVERAGED.

    Ties must be averaged, not broken arbitrarily: saturated sigmoid outputs
    and flat low-risk regions produce many exactly-equal scores, and a stable
    argsort would hand those an arbitrary strict ordering -- which then leaks
    index order into the ensemble as if it were signal, and breaks the
    calibration-invariance this whole combination rule exists to provide.
    """
    if a.size == 0:
        return a
    r = rankdata(a, method="average") - 1.0
    return (r / max(a.size - 1, 1)).astype(np.float32)


def global_quantile_transform(x, grid):
    """Map scores to [0,1] via a precomputed sorted subsample of the SAME
    member's pooled scores -- calibration-invariant and memory-cheap."""
    idx = np.searchsorted(grid, x, side="left")
    return (idx / float(max(len(grid) - 1, 1))).astype(np.float32)


def topk_table(df, label):
    print(f"\n=== {label} | n={len(df)} patches | mean fire {df.n_fire.mean():.0f}, "
          f"NEW {df.n_new.mean():.0f} ({100*df.n_new.sum()/max(df.n_fire.sum(),1):.0f}%) ===")
    print(f'\n{"top-k":>7} {"TPR(all)":>9} {"FPR":>8} {"lift(all)":>10} {"TPR(new)":>9} {"lift(new)":>10}')
    for f in KS:
        k = f"{f:g}"
        tn = df[f"tprNEW_{k}"].mean() if f"tprNEW_{k}" in df else np.nan
        ln = df[f"liftNEW_{k}"].mean() if f"liftNEW_{k}" in df else np.nan
        print(f"{100*f:6.1f}% {df[f'tpr_{k}'].mean():9.3f} {df[f'fpr_{k}'].mean():8.4f} "
              f"{df[f'lift_{k}'].mean():10.1f} {tn:9.3f} {ln:10.1f}")
    ba = max(KS, key=lambda f: np.nanmean(df[f"lift_{f:g}"]))
    cand = [f for f in KS if f"liftNEW_{f:g}" in df]
    print(f"\n  ALL-fire enrichment peaks at top-{100*ba:g}%  ({np.nanmean(df[f'lift_{ba:g}']):.1f}x)")
    if cand:
        bn = max(cand, key=lambda f: np.nanmean(df[f"liftNEW_{f:g}"]))
        print(f"  NEW-fire enrichment peaks at top-{100*bn:g}%  ({np.nanmean(df[f'liftNEW_{bn:g}']):.1f}x)")


def main():
    torch.set_num_threads(8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = zarr.open_group(f"cube_daily_smgrid_{EVAL_YEAR}.zarr", mode="r")
    TIMES = list(g.attrs.get("time", []))

    models = []
    for c in CKPTS:
        m = ConvLSTMLitDual.load_from_checkpoint(c, map_location=device)
        m.eval()
        m.to(device)
        models.append(m)
        print(f"[info] loaded {os.path.basename(c)}", flush=True)
    M = len(models)
    if M < 2:
        sys.exit("[error] need >=2 checkpoints for an ensemble")
    print(f"[info] {M} members on {device}, patch_size={PATCH}, "
          f"min_pos_pixels={MIN_POS} (density {100*MIN_POS/PATCH**2:.4f}%)", flush=True)

    ds = DualWindowDataset(DualPatchConfig(
        zarr_paths=(f"cube_daily_smgrid_{EVAL_YEAR}.zarr",),
        stats_path="channel_stats_2015_2018.json",
        slow_cube_path="cube_slow_8day.zarr",
        day_offset={"2015": 0, "2016": 365, "2017": 731, "2018": 1096,
                    "2019": 1461, "2020": 1826}[EVAL_YEAR],
        patch_size=PATCH, samples_per_epoch=N_PATCH * 3, seed=21,
        min_pos_pixels=MIN_POS, pos_frac=1.0, deterministic=True,
        fire_history=True, fire_history_lags=(3, 4, 5), fire_history_distance=True,
        use_lightning=os.environ.get("USE_LIGHTNING", "0") == "1",
        use_elevation=os.environ.get("USE_ELEVATION", "0") == "1",
        use_slope_aspect=os.environ.get("USE_SLOPE_ASPECT", "0") == "1",
        use_fuel_age=os.environ.get("USE_FUEL_AGE", "0") == "1",
        use_wind_dir=os.environ.get("USE_WIND_DIR", "0") == "1"))

    # rows_* accumulate per-patch metrics for each combination rule + member
    rows = {f"member{i}": [] for i in range(M)}
    rows["prob-mean"] = []
    rows["rank-mean"] = []
    pooled = {f"member{i}": [] for i in range(M)}
    pooled["prob-mean"] = []
    pooled_y = []

    n_kept = 0
    for i in range(N_PATCH * 3):
        if n_kept >= N_PATCH:
            break
        b = ds[i]
        land = b["mask"].numpy() > 0.5
        truth = (b["y"][-1].numpy() > 0) & land
        if truth.sum() < MIN_POS or land.mean() < 0.5:
            continue
        recent = b["x_fast"][-1, :, :, 7].numpy() > 0.5
        known = ndimage.binary_dilation(recent, iterations=DILATE)
        new_fire = truth & ~known

        xs = b["x_slow"].unsqueeze(0).to(device)
        xf = b["x_fast"].unsqueeze(0).to(device)
        xc = b["x_cat"].unsqueeze(0).to(device)
        ps = []
        with torch.no_grad():
            for mi, m in enumerate(models):
                try:
                    p = torch.sigmoid(m.forward_seq(xs, xf, xc)[:, -1])[0].cpu().numpy()
                except RuntimeError as e:
                    sys.exit(f"[error] member {mi} ({os.path.basename(CKPTS[mi])}) failed on "
                             f"the shared dataset -- do all checkpoints use the same feature "
                             f"flags? Underlying error:\n{e}")
                ps.append(p)

        combos = {f"member{mi}": ps[mi] for mi in range(M)}
        combos["prob-mean"] = np.mean(ps, axis=0)
        # within-patch rank mean (top-k is a within-patch operation)
        rk = np.zeros_like(ps[0], dtype=np.float32)
        for p in ps:
            flat = np.zeros(p.size, np.float32)
            flat[land.ravel()] = rank01(p.ravel()[land.ravel()])
            rk += flat.reshape(p.shape)
        combos["rank-mean"] = rk / M

        tt = int(b["t_end"])
        date = TIMES[tt] if tt < len(TIMES) else ""
        mth = int(date[5:7]) if date else 0
        lat = LAT0 - (int(b["y0"]) + PATCH / 2) * PX
        meta = dict(n_fire=int(truth.sum()), n_new=int(new_fire.sum()), date=date,
                    season=SEAS.get(mth, "?"),
                    region="Tropical N" if lat > -20 else ("Temperate S" if lat < -30 else "Central"))

        for name, prob in combos.items():
            r = dict(meta)
            for f in KS:
                k = f"{f:g}"
                t_, f_, l_ = tpr_fpr(prob, truth, land, f)
                r[f"tpr_{k}"], r[f"fpr_{k}"], r[f"lift_{k}"] = t_, f_, l_
                if new_fire.sum() >= 3:
                    tn_, _, ln_ = tpr_fpr(prob, new_fire, land, f)
                    r[f"tprNEW_{k}"], r[f"liftNEW_{k}"] = tn_, ln_
            rows[name].append(r)

        for mi in range(M):
            pooled[f"member{mi}"].append(ps[mi][land].astype(np.float32))
        pooled["prob-mean"].append(combos["prob-mean"][land].astype(np.float32))
        pooled_y.append(truth[land].astype(np.int8))

        n_kept += 1
        if n_kept % 50 == 0:
            print(f"  {n_kept}...", flush=True)

    from sklearn.metrics import average_precision_score, roc_auc_score

    Y = np.concatenate(pooled_y)
    print(f"\n*** POOLED {EVAL_YEAR} TEST METRICS "
          f"(n={len(Y):,} land px, base rate {Y.mean():.5f}) ***")

    member_scores = [np.concatenate(pooled[f"member{mi}"]) for mi in range(M)]

    # global rank-mean: transform each member through its own empirical CDF
    rng = np.random.default_rng(0)
    rank_sum = np.zeros(len(Y), np.float32)
    for s in member_scores:
        sub = s if s.size <= QUANTILE_SAMPLE else rng.choice(s, QUANTILE_SAMPLE, replace=False)
        grid = np.sort(sub)
        rank_sum += global_quantile_transform(s, grid)
    rank_mean_pooled = rank_sum / M

    results = {}
    for mi in range(M):
        results[f"member{mi}"] = member_scores[mi]
    results["prob-mean"] = np.concatenate(pooled["prob-mean"])
    results["rank-mean"] = rank_mean_pooled

    print(f"\n{'combination':<14} {'AUC-PR':>9} {'ROC-AUC':>9}")
    summary = {}
    for name, s in results.items():
        ap = average_precision_score(Y, s)
        roc = roc_auc_score(Y, s)
        summary[name] = (ap, roc)
        tag = os.path.basename(CKPTS[int(name[6:])]) if name.startswith("member") else ""
        print(f"{name:<14} {ap:9.4f} {roc:9.4f}  {tag}")

    best_member = max((summary[f"member{mi}"][0], f"member{mi}") for mi in range(M))
    print(f"\n  best single member: {best_member[1]} (AUC-PR {best_member[0]:.4f})")
    for rule in ("prob-mean", "rank-mean"):
        d = 100.0 * (summary[rule][0] - best_member[0]) / best_member[0]
        verdict = "BEATS" if d > 0 else "loses to"
        print(f"  {rule:<10} {verdict} best member by {abs(d):.2f}%")

    for name in list(rows.keys()):
        df = pd.DataFrame(rows[name])
        topk_table(df, name)
        if name in ("prob-mean", "rank-mean"):
            df.to_csv(f"ensemble_stats_{EVAL_YEAR}{OUT_TAG}_{name}.csv", index=False)
    print(f"\nwrote ensemble_stats_{EVAL_YEAR}{OUT_TAG}_*.csv")


if __name__ == "__main__":
    main()
