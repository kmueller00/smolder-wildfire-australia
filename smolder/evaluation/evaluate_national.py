"""National evaluation of SMOLDER: every land pixel of Australia, every day.

The patch evaluation (evaluate.py) measures ranking skill inside fire-active
384 px windows. A national risk product has to rank all ~7.7 M land pixels
of the continent against each other on every day, including the large areas
where nothing burns. This script produces that evaluation.

Inference
  For each issue day D the continent is covered by 384 x 384 px tiles (the
  training patch size; required, because the fire-history distance feature is
  computed inside each window) with stride STRIDE (default 288, i.e. 96 px
  overlap). Tile predictions of the last fast time step (fire on days
  D+1..D+3) are blended with a weight that ramps linearly to zero over the
  outer 96 px of each tile, so every pixel is dominated by tiles in which it
  lies away from the edge. Tiles with less than MIN_LAND land are skipped.

Metrics (all on land pixels; target y_fire_3d[D])
  1. per day, national: AUC-PR, ROC-AUC, and at national top-k budgets
     (fraction of Australia's land flagged) TPR, precision and lift, for all
     fire and for new fire (fire more than 3 px, taxicab, from any fire
     detected on days D-2..D).
  2. pooled over the year: AUC-PR and ROC-AUC on a stratified sample (every
     fire pixel, NEG_FRAC of the non-fire pixels, weighted back), overall and
     by Koppen climate group, latitude band and season.
  3. 25 km cells (CELL x CELL px): cell risk = mean pixel risk over land, cell
     target = any fire. Per-day and pooled AUC-PR, lift at top-k cells.
  4. calibration (optional, CALIB_YEAR): the same national inference on a
     calibration year (every CALIB_STRIDE-th day) fits an isotonic map from
     raw score to fire probability; reliability of the calibrated 2020
     probabilities is reported by bin, overall and per region, with the
     expected calibration error.

Outputs (in the working directory)
  national_<year>.json             summary of everything above
  national_<year>_daily.csv        one row per issue day
  national_<year>_maps.npz         annual mean risk, fire frequency, example days
  national_<year>_calibration.pkl  isotonic map (if CALIB_YEAR is set)
  $SAVE_SCORES (optional)          float32 (n_days, n_land) scores of the land
                                   pixels in row-major order, one row per issue
                                   day, plus <SAVE_SCORES>.days.npy with the days

Usage
  SMOLDER_DATA=/path/to/cubes CALIB_YEAR=2019 \
      python -m smolder.evaluation.evaluate_national
Needs the daily cube of EVAL_YEAR (and CALIB_YEAR), cube_slow_8day.zarr and
the model. A GPU is strongly recommended (about 150 tiles per day).
"""
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

from smolder.data.io import daily_cube, open_zarr_root
from smolder.data.zarr_dual_datamodule import check_checkpoint_inputs, DualPatchConfig, DualWindowDataset
from smolder.models.conv_lstm_lit_dual import ConvLSTMLitDual

CKPT = os.environ.get("CKPT", "checkpoints/smolder_swa.ckpt")
EVAL_YEAR = int(os.environ.get("EVAL_YEAR", 2020))
CALIB_YEAR = os.environ.get("CALIB_YEAR", "")
CALIB_STRIDE = int(os.environ.get("CALIB_STRIDE", 5))
DAY_STRIDE = int(os.environ.get("DAY_STRIDE", 1))
PATCH = int(os.environ.get("PATCH", 384))
STRIDE = int(os.environ.get("STRIDE", 288))
RAMP = int(os.environ.get("RAMP", 96))
MIN_LAND = float(os.environ.get("MIN_LAND", 0.01))
CELL = int(os.environ.get("CELL", 25))
NEG_FRAC = float(os.environ.get("NEG_FRAC", 0.002))
BATCH = int(os.environ.get("BATCH", 4))
WORKERS = int(os.environ.get("WORKERS", 8))
SAVE_SCORES = os.environ.get("SAVE_SCORES", "")
DILATE = 3
KS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]
OFFSETS = {2015: 0, 2016: 365, 2017: 731, 2018: 1096, 2019: 1461, 2020: 1826}
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
          6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
LAT0, PX = -9.005000113999998, 0.01
N_EXAMPLE_DAYS = 4
MAP_DOWN = 4


# ---------------------------------------------------------------- metrics
def ap_auc(score, label, weight=None):
    """Average precision and ROC-AUC from one sort; ties are grouped, so the
    result does not depend on array order. Optional sample weights."""
    score = np.asarray(score, np.float64)
    label = np.asarray(label, bool)
    w = np.ones_like(score) if weight is None else np.asarray(weight, np.float64)
    pos_w = w * label
    neg_w = w * ~label
    P, N = pos_w.sum(), neg_w.sum()
    if P == 0 or N == 0:
        return np.nan, np.nan
    order = np.argsort(-score, kind="stable")
    s = score[order]
    tp = np.cumsum(pos_w[order])
    fp = np.cumsum(neg_w[order])
    last = np.r_[np.nonzero(np.diff(s))[0], s.size - 1]       # end of each tie group
    tp, fp = tp[last], fp[last]
    prec = tp / (tp + fp)
    rec = tp / P
    ap = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
    fpr = np.r_[0.0, fp / N]
    tpr = np.r_[0.0, rec]
    auc = float(np.trapz(tpr, fpr))
    return ap, auc


def topk_stats(score, label, frac):
    """Flag the top `frac` of pixels; return TPR, precision and lift."""
    n = score.size
    k = max(1, int(round(frac * n)))
    thr = np.partition(score, n - k)[n - k]
    sel = score >= thr
    n_sel = int(sel.sum())
    n_pos = int(label.sum())
    hit = int((sel & label).sum())
    if n_pos == 0:
        return np.nan, np.nan, np.nan
    prec = hit / n_sel
    return hit / n_pos, prec, prec / (n_pos / n)


# ---------------------------------------------------------------- inference
def tile_origins(H, W):
    ys = sorted(set(list(range(0, H - PATCH + 1, STRIDE)) + [H - PATCH]))
    xs = sorted(set(list(range(0, W - PATCH + 1, STRIDE)) + [W - PATCH]))
    return ys, xs


def blend_weight():
    r = np.arange(PATCH, dtype=np.float32)
    ramp = np.minimum(1.0, np.minimum(r + 0.5, PATCH - r - 0.5) / RAMP)
    return np.outer(ramp, ramp).astype(np.float32)


class Tiles(Dataset):
    def __init__(self, ds, items):
        self.ds, self.items = ds, items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        D, y0, x0 = self.items[i]
        b = self.ds.sample_at(D + 1, y0, x0)        # last fast step forecasts D+1..D+3
        return b["x_slow"], b["x_fast"], b["x_cat"], torch.tensor([D, y0, x0])


def env_flag(name):
    return os.environ.get(name, "0") == "1"


def national_days(model, year, days, device, on_day):
    """Run national inference for `days` of `year`; call on_day(D, prob) for
    each finished day with the blended land probability map (NaN at sea)."""
    cube = daily_cube(year)
    ds = DualWindowDataset(DualPatchConfig(
        zarr_paths=(cube,), stats_path="channel_stats_2015_2018.json",
        slow_cube_path="cube_slow_8day.zarr", day_offset=OFFSETS[year],
        patch_size=PATCH, samples_per_epoch=1, seed=0, deterministic=True,
        fire_history=True, fire_history_lags=(3, 4, 5), fire_history_distance=True,
        use_lightning=env_flag("USE_LIGHTNING"), use_elevation=env_flag("USE_ELEVATION"),
        use_slope_aspect=env_flag("USE_SLOPE_ASPECT"), use_fuel_age=env_flag("USE_FUEL_AGE"),
        use_wind_dir=env_flag("USE_WIND_DIR"), use_ffdi=env_flag("USE_FFDI"),
        use_fmc=env_flag("USE_FMC"), fmc_store=os.environ.get("FMC_STORE", "fmc_weekly_ff.zarr")))
    land = ds.landmask > 0
    H, W = land.shape
    ys, xs = tile_origins(H, W)
    tiles = [(y0, x0) for y0 in ys for x0 in xs
             if land[y0:y0 + PATCH, x0:x0 + PATCH].mean() >= MIN_LAND]
    items = [(D, y0, x0) for D in days for (y0, x0) in tiles]
    print(f"[national] {year}: {len(days)} days x {len(tiles)} tiles = {len(items)} forward passes",
          flush=True)
    wtile = torch.from_numpy(blend_weight()).to(device)
    loader = DataLoader(Tiles(ds, items), batch_size=BATCH, num_workers=WORKERS, shuffle=False,
                        persistent_workers=False, pin_memory=device == "cuda")
    acc = torch.zeros((H, W), device=device)
    wsum = torch.zeros((H, W), device=device)
    cur, t0, done = None, time.time(), 0
    with torch.no_grad():
        for xs_, xf_, xc_, meta in loader:
            logits = model.forward_seq(xs_.to(device), xf_.to(device), xc_.to(device))[:, -1]
            p = torch.sigmoid(logits)
            for i in range(p.shape[0]):
                D, y0, x0 = (int(v) for v in meta[i])
                if cur is not None and D != cur:
                    prob = (acc / wsum.clamp_min(1e-6)).cpu().numpy()
                    prob[~land] = np.nan
                    on_day(cur, prob)
                    acc.zero_(); wsum.zero_(); done += 1
                    if done % 20 == 0:
                        el = time.time() - t0
                        print(f"  {done}/{len(days)} days, {el/done:.1f} s/day", flush=True)
                cur = D
                acc[y0:y0 + PATCH, x0:x0 + PATCH] += p[i] * wtile
                wsum[y0:y0 + PATCH, x0:x0 + PATCH] += wtile
    if cur is not None:
        prob = (acc / wsum.clamp_min(1e-6)).cpu().numpy()
        prob[~land] = np.nan
        on_day(cur, prob)
    return land


def valid_days(year, stride):
    g = open_zarr_root(daily_cube(year))
    valid = np.asarray(g["y_fire_3d_valid"][...]) > 0 if "y_fire_3d_valid" in g else np.ones(g["X"].shape[0], bool)
    T = g["X"].shape[0]
    # issue day D needs 14 fast days (D-13..D) inside the cube and a valid target
    return [D for D in range(13, T) if valid[D]][::stride], g


# ---------------------------------------------------------------- main
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(8)
    check_checkpoint_inputs(CKPT)
    model = ConvLSTMLitDual.load_from_checkpoint(CKPT, map_location=device).eval().to(device)
    print(f"[national] model {os.path.basename(CKPT)} on {device}", flush=True)
    rng = np.random.default_rng(0)

    # ---- optional calibration year: pooled stratified sample -> isotonic map
    iso = None
    if CALIB_YEAR:
        cyear = int(CALIB_YEAR)
        cdays, cg = valid_days(cyear, CALIB_STRIDE)
        cy = cg["y_fire_3d"]
        cal_s, cal_y, cal_w = [], [], []

        def on_cal(D, prob):
            lm = np.isfinite(prob)
            y = (np.asarray(cy[D]) > 0) & lm
            neg = lm & ~y & (rng.random(lm.shape) < NEG_FRAC)
            cal_s.append(np.r_[prob[y], prob[neg]])
            cal_y.append(np.r_[np.ones(y.sum(), bool), np.zeros(neg.sum(), bool)])
            cal_w.append(np.r_[np.ones(y.sum()), np.full(neg.sum(), 1.0 / NEG_FRAC)])

        national_days(model, cyear, cdays, device, on_cal)
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(np.concatenate(cal_s), np.concatenate(cal_y).astype(float), sample_weight=np.concatenate(cal_w))
        with open(f"national_{EVAL_YEAR}_calibration.pkl", "wb") as fh:
            pickle.dump(iso, fh)
        print(f"[national] isotonic calibration fitted on {cyear} ({len(cdays)} days)", flush=True)

    # ---- evaluation year
    days, g = valid_days(EVAL_YEAR, DAY_STRIDE)
    times = list(g.attrs["time"])
    yarr = g["y_fire_3d"]
    kg = np.asarray(g["koppen_geiger"][...])
    H, W = kg.shape
    kg_group = np.full((H, W), "other", object)
    kg_group[(kg >= 1) & (kg <= 3)] = "tropical"
    kg_group[(kg >= 4) & (kg <= 7)] = "arid"
    kg_group[(kg >= 8) & (kg <= 16)] = "temperate"
    lat = LAT0 - (np.arange(H) + 0.5) * PX
    band_row = np.where(lat > -20, "north (>20S)", np.where(lat < -30, "south (<30S)", "central"))
    band = np.broadcast_to(band_row[:, None], (H, W))
    Hc, Wc = int(np.ceil(H / CELL)), int(np.ceil(W / CELL))

    def cells(a, how):
        pad = np.full((Hc * CELL, Wc * CELL), np.nan if how == "mean" else 0, np.float32)
        pad[:H, :W] = a
        b = pad.reshape(Hc, CELL, Wc, CELL)
        return np.nanmean(b, axis=(1, 3)) if how == "mean" else b.max(axis=(1, 3))

    rows = []
    pool = dict(s=[], y=[], w=[], new=[], kg=[], band=[], season=[], cal=[])
    cpool = dict(s=[], y=[])
    risk_sum = np.zeros((H, W), np.float64)
    fire_cnt = np.zeros((H, W), np.int32)
    n_days = [0]
    examples = []        # (n_fire, D, prob_down, fire_down)
    dump = None
    if SAVE_SCORES:
        n_land = int((np.asarray(g["landmask"][...]) > 0).sum())
        dump = np.lib.format.open_memmap(SAVE_SCORES, mode="w+", dtype=np.float32, shape=(len(days), n_land))
        np.save(SAVE_SCORES + ".days.npy", np.asarray(days))
        row_of = {D: i for i, D in enumerate(days)}

    def on_day(D, prob):
        lm = np.isfinite(prob)
        if dump is not None:
            dump[row_of[D]] = prob[lm]
        y = (np.asarray(yarr[D]) > 0) & lm
        recent = (np.asarray(yarr[D - 3]) > 0) & lm if D >= 3 else np.zeros_like(lm)
        new = y & ~ndimage.binary_dilation(recent, iterations=DILATE)
        s = prob[lm]; yl = y[lm]; nl = new[lm]
        date = str(times[D]); month = int(date[5:7])
        r = dict(date=date, day=D, n_land=int(lm.sum()), n_fire=int(yl.sum()), n_new=int(nl.sum()),
                 base_rate=float(yl.mean()))
        r["auc_pr"], r["roc_auc"] = ap_auc(s, yl)
        r["auc_pr_new"], _ = ap_auc(s, nl) if nl.any() else (np.nan, np.nan)
        for k in KS:
            t, pr, li = topk_stats(s, yl, k)
            tn, _, ln = topk_stats(s, nl, k) if nl.any() else (np.nan, np.nan, np.nan)
            r[f"tpr_{k:g}"], r[f"prec_{k:g}"], r[f"lift_{k:g}"] = t, pr, li
            r[f"tpr_new_{k:g}"], r[f"lift_new_{k:g}"] = tn, ln
        # 25 km cells
        cs = cells(np.where(lm, prob, np.nan), "mean")
        cy_ = cells(y.astype(np.float32), "max") > 0
        cm = np.isfinite(cs)
        r["cell_auc_pr"], r["cell_roc_auc"] = ap_auc(cs[cm], cy_[cm])
        for k in (0.01, 0.05, 0.10):
            r[f"cell_tpr_{k:g}"], _, r[f"cell_lift_{k:g}"] = topk_stats(cs[cm], cy_[cm], k)
        cpool["s"].append(cs[cm]); cpool["y"].append(cy_[cm])
        rows.append(r)
        # pooled stratified sample
        neg = lm & ~y & (rng.random(lm.shape) < NEG_FRAC)
        keep = y | neg
        pool["s"].append(prob[keep]); pool["y"].append(y[keep]); pool["new"].append(new[keep])
        pool["w"].append(np.where(y[keep], 1.0, 1.0 / NEG_FRAC))
        pool["kg"].append(kg_group[keep]); pool["band"].append(band[keep])
        pool["season"].append(np.full(int(keep.sum()), SEASON[month], object))
        # annual maps and example days
        risk_sum[lm] += prob[lm]; fire_cnt[y] += 1; n_days[0] += 1
        if len(examples) < N_EXAMPLE_DAYS or r["n_fire"] > examples[-1][0]:
            d = MAP_DOWN
            pdn = prob[:H // d * d, :W // d * d].reshape(H // d, d, W // d, d)
            fdn = y[:H // d * d, :W // d * d].reshape(H // d, d, W // d, d).any(axis=(1, 3))
            examples.append((r["n_fire"], date, np.nanmean(pdn, axis=(1, 3)).astype(np.float16), fdn))
            examples.sort(key=lambda e: -e[0])
            del examples[N_EXAMPLE_DAYS:]

    land = national_days(model, EVAL_YEAR, days, device, on_day)
    if dump is not None:
        dump.flush()

    df = pd.DataFrame(rows)
    df.to_csv(f"national_{EVAL_YEAR}_daily.csv", index=False)

    # ---- pooled metrics
    P = {k: np.concatenate(v) for k, v in pool.items() if v and k != "cal"}
    out = dict(checkpoint=os.path.basename(CKPT), eval_year=EVAL_YEAR, n_days=int(len(df)),
               patch=PATCH, stride=STRIDE, ramp=RAMP, neg_sample_frac=NEG_FRAC,
               n_land_px=int(land.sum()), fire_px_total=int(df.n_fire.sum()),
               base_rate=float(df.n_fire.sum() / df.n_land.sum()))
    out["pooled_auc_pr"], out["pooled_roc_auc"] = ap_auc(P["s"], P["y"], P["w"])
    newmask = P["new"] | ~P["y"]            # new fire vs non-fire (old fire excluded)
    out["pooled_auc_pr_new"], out["pooled_roc_auc_new"] = ap_auc(P["s"][newmask], P["new"][newmask], P["w"][newmask])
    out["daily"] = {c: dict(mean=float(np.nanmean(df[c])), median=float(np.nanmedian(df[c])))
                    for c in df.columns if c not in ("date", "day")}
    out["topk_national"] = [dict(k=k, tpr=float(np.nanmean(df[f"tpr_{k:g}"])),
                                 precision=float(np.nanmean(df[f"prec_{k:g}"])),
                                 lift=float(np.nanmean(df[f"lift_{k:g}"])),
                                 tpr_new=float(np.nanmean(df[f"tpr_new_{k:g}"])),
                                 lift_new=float(np.nanmean(df[f"lift_new_{k:g}"]))) for k in KS]
    by = {}
    for key in ("kg", "band", "season"):
        by[key] = {}
        for grp in sorted(set(P[key])):
            m = P[key] == grp
            if P["y"][m].sum() < 50:
                continue
            ap_, auc_ = ap_auc(P["s"][m], P["y"][m], P["w"][m])
            by[key][str(grp)] = dict(auc_pr=ap_, roc_auc=auc_, fire_px=int(P["y"][m].sum()),
                                     base_rate=float(P["y"][m].sum() / P["w"][m].sum()))
    out["by_group"] = by
    CS, CY = np.concatenate(cpool["s"]), np.concatenate(cpool["y"])
    out["cells"] = dict(cell_km=CELL, n_cell_days=int(CS.size), base_rate=float(CY.mean()))
    out["cells"]["pooled_auc_pr"], out["cells"]["pooled_roc_auc"] = ap_auc(CS, CY)
    out["cells"]["topk"] = [dict(k=k, tpr=float(np.nanmean(df[f"cell_tpr_{k:g}"])),
                                 lift=float(np.nanmean(df[f"cell_lift_{k:g}"]))) for k in (0.01, 0.05, 0.10)]

    # ---- calibration / reliability
    if iso is not None:
        pc = iso.predict(P["s"])
        edges = np.array([0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0])
        rel = []
        ece = 0.0
        for lo_, hi_ in zip(edges[:-1], edges[1:]):
            m = (pc >= lo_) & (pc < hi_ if hi_ < 1 else pc <= hi_)
            if not m.any():
                continue
            wsum = P["w"][m].sum()
            pred = float(np.sum(pc[m] * P["w"][m]) / wsum)
            obs = float(np.sum(P["y"][m] * P["w"][m]) / wsum)
            rel.append(dict(bin_lo=float(lo_), bin_hi=float(hi_), mean_predicted=pred,
                            observed_rate=obs, weight_share=float(wsum / P["w"].sum())))
            ece += (wsum / P["w"].sum()) * abs(pred - obs)
        out["calibration"] = dict(fit_year=int(CALIB_YEAR), fit_day_stride=CALIB_STRIDE,
                                  reliability=rel, expected_calibration_error=float(ece))
        out["calibration"]["by_kg"] = {}
        for grp in sorted(set(P["kg"])):
            m = P["kg"] == grp
            if P["y"][m].sum() < 50:
                continue
            wsum = P["w"][m].sum()
            out["calibration"]["by_kg"][str(grp)] = dict(
                mean_predicted=float(np.sum(pc[m] * P["w"][m]) / wsum),
                observed_rate=float(np.sum(P["y"][m] * P["w"][m]) / wsum))

    with open(f"national_{EVAL_YEAR}.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)

    d = MAP_DOWN
    np.savez_compressed(
        f"national_{EVAL_YEAR}_maps.npz",
        mean_risk=(risk_sum / max(n_days[0], 1))[:H // d * d, :W // d * d].reshape(H // d, d, W // d, d)
        .mean(axis=(1, 3)).astype(np.float16),
        fire_days=fire_cnt[:H // d * d, :W // d * d].reshape(H // d, d, W // d, d).max(axis=(1, 3)).astype(np.int16),
        land=land[:H // d * d, :W // d * d].reshape(H // d, d, W // d, d).mean(axis=(1, 3)) > 0.5,
        example_dates=np.array([e[1] for e in examples]),
        example_risk=np.stack([e[2] for e in examples]),
        example_fire=np.stack([e[3] for e in examples]),
        down=d)

    print(f"\n*** NATIONAL {EVAL_YEAR}: {len(df)} days, {out['n_land_px']:,} land px, "
          f"base rate {out['base_rate']:.5f} ***")
    print(f"  pooled AUC-PR {out['pooled_auc_pr']:.4f}  ROC-AUC {out['pooled_roc_auc']:.4f}  "
          f"(new fire: {out['pooled_auc_pr_new']:.4f} / {out['pooled_roc_auc_new']:.4f})")
    print(f"  daily AUC-PR mean {out['daily']['auc_pr']['mean']:.4f}  median {out['daily']['auc_pr']['median']:.4f}")
    print(f"  {'top-k':>7} {'TPR':>7} {'prec':>8} {'lift':>8} {'TPRnew':>7} {'liftnew':>8}")
    for t in out["topk_national"]:
        print(f"  {100*t['k']:6.1f}% {t['tpr']:7.3f} {t['precision']:8.4f} {t['lift']:8.1f} "
              f"{t['tpr_new']:7.3f} {t['lift_new']:8.1f}")
    print(f"  25 km cells: pooled AUC-PR {out['cells']['pooled_auc_pr']:.4f} ROC-AUC "
          f"{out['cells']['pooled_roc_auc']:.4f} (cell base rate {out['cells']['base_rate']:.4f})")
    for key, grp in by.items():
        print(f"  by {key}: " + ", ".join(f"{g_} AP {v['auc_pr']:.3f}" for g_, v in grp.items()))
    if iso is not None:
        print(f"  calibration fitted on {CALIB_YEAR}: ECE {out['calibration']['expected_calibration_error']:.5f}")
    print(f"wrote national_{EVAL_YEAR}.json, _daily.csv, _maps.npz", flush=True)


if __name__ == "__main__":
    main()
