"""Persistence baseline for the national evaluation (no model, no training).

Score per issue day D and land pixel: closeness to the nearest fire pixel
detected on days D-2..D (y_fire_3d[D-3], the newest fire-history window of
SMOLDER at its last fast step). The distance is a Euclidean distance
transform over the whole continent, not per tile. Pixels are ranked by
increasing distance; equal distances (for example all burning pixels at
distance 0) are ordered at random with a fixed seed per day.

The score is -(d + 1e-5 u) with d the distance in pixels and u ~ U(0, 1).
Distinct distances on this grid differ by at least ~9e-5 px, so the jitter
only breaks ties and never reorders different distances; the same score is
comparable across days, which the pooled metrics need. A 25 km cell's score
is the mean pixel score over its land, i.e. minus the mean distance, exactly
as SMOLDER's cell risk is the mean pixel risk. Days without any recent fire
give every pixel the same distance, so their ranking is random.

Everything else (issue days, land mask, target, top-k shares, new-fire
definition, pooled stratified sample, 25 km cells, groups) is taken from
evaluate_national.py, so the two results are directly comparable. Only the
random negative sample of the pooled metrics differs (seeded per day here).

Outputs (in the working directory)
  national_<year>_persistence.json        same structure as national_<year>.json
  national_<year>_persistence_daily.csv   same columns as national_<year>_daily.csv

Usage
  SMOLDER_DATA=/path/to/cubes python -m smolder.evaluation.evaluate_persistence
Needs only the daily cube of EVAL_YEAR. CPU only (WORKERS processes).
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import ndimage

from smolder.evaluation.evaluate_national import (CELL, DILATE, KS, LAT0, NEG_FRAC, PX, SEASON,
                                                  ap_auc, topk_stats, valid_days)
from smolder.data.io import daily_cube, open_zarr_root

EVAL_YEAR = int(os.environ.get("EVAL_YEAR", 2020))
DAY_STRIDE = int(os.environ.get("DAY_STRIDE", 1))
WORKERS = int(os.environ.get("WORKERS", 12))
SEED = int(os.environ.get("SEED", 0))
JITTER = 1e-5
FAR = 1e4                      # distance on days without recent fire (> grid diagonal)

G = {}                          # per-process state


def _init():
    g = open_zarr_root(daily_cube(EVAL_YEAR))
    G["y"] = g["y_fire_3d"]
    G["land"] = np.asarray(g["landmask"][...]) > 0
    kg = np.asarray(g["koppen_geiger"][...])
    H, W = kg.shape
    kg_group = np.full((H, W), "other", object)
    kg_group[(kg >= 1) & (kg <= 3)] = "tropical"
    kg_group[(kg >= 4) & (kg <= 7)] = "arid"
    kg_group[(kg >= 8) & (kg <= 16)] = "temperate"
    lat = LAT0 - (np.arange(H) + 0.5) * PX
    band_row = np.where(lat > -20, "north (>20S)", np.where(lat < -30, "south (<30S)", "central"))
    G["kg"] = kg_group
    G["band"] = np.broadcast_to(band_row[:, None], (H, W))
    G["times"] = list(g.attrs["time"])


def cells(a, how):
    H, W = a.shape
    Hc, Wc = int(np.ceil(H / CELL)), int(np.ceil(W / CELL))
    pad = np.full((Hc * CELL, Wc * CELL), np.nan if how == "mean" else 0, np.float32)
    pad[:H, :W] = a
    b = pad.reshape(Hc, CELL, Wc, CELL)
    return np.nanmean(b, axis=(1, 3)) if how == "mean" else b.max(axis=(1, 3))


def persistence_score(recent, rng):
    d = ndimage.distance_transform_edt(~recent) if recent.any() else np.full(recent.shape, FAR)
    return -(d + JITTER * rng.random(recent.shape)), d


def one_day(D):
    land = G["land"]
    rng = np.random.default_rng([SEED, D])
    y = (np.asarray(G["y"][D]) > 0) & land
    recent = (np.asarray(G["y"][D - 3]) > 0) & land
    score, dist = persistence_score(recent, rng)
    prob = np.where(land, score, np.nan)
    lm = land
    new = y & ~ndimage.binary_dilation(recent, iterations=DILATE)
    s = prob[lm]; yl = y[lm]; nl = new[lm]
    date = str(G["times"][D]); month = int(date[5:7])
    r = dict(date=date, day=D, n_land=int(lm.sum()), n_fire=int(yl.sum()), n_new=int(nl.sum()),
             base_rate=float(yl.mean()))
    r["auc_pr"], r["roc_auc"] = ap_auc(s, yl)
    r["auc_pr_new"], _ = ap_auc(s, nl) if nl.any() else (np.nan, np.nan)
    for k in KS:
        t, pr, li = topk_stats(s, yl, k)
        tn, _, ln = topk_stats(s, nl, k) if nl.any() else (np.nan, np.nan, np.nan)
        r[f"tpr_{k:g}"], r[f"prec_{k:g}"], r[f"lift_{k:g}"] = t, pr, li
        r[f"tpr_new_{k:g}"], r[f"lift_new_{k:g}"] = tn, ln
    cs = cells(np.where(lm, prob, np.nan).astype(np.float32), "mean")
    cy_ = cells(y.astype(np.float32), "max") > 0
    cm = np.isfinite(cs)
    r["cell_auc_pr"], r["cell_roc_auc"] = ap_auc(cs[cm], cy_[cm])
    for k in (0.01, 0.05, 0.10):
        r[f"cell_tpr_{k:g}"], _, r[f"cell_lift_{k:g}"] = topk_stats(cs[cm], cy_[cm], k)
    neg = lm & ~y & (rng.random(lm.shape) < NEG_FRAC)
    keep = y | neg
    pool = dict(s=prob[keep], y=y[keep], new=new[keep], w=np.where(y[keep], 1.0, 1.0 / NEG_FRAC),
                kg=G["kg"][keep], band=G["band"][keep],
                season=np.full(int(keep.sum()), SEASON[month], object))
    r["median_dist_fire_px"] = float(np.median(dist[y])) if y.any() else np.nan
    return r, pool, (cs[cm], cy_[cm])


def main():
    days, _ = valid_days(EVAL_YEAR, DAY_STRIDE)
    print(f"[persistence] {EVAL_YEAR}: {len(days)} issue days, {WORKERS} workers", flush=True)
    rows, pools, cpool = [], [], []
    t0 = time.time()
    with Pool(WORKERS, initializer=_init) as pool:
        for i, (r, p, c) in enumerate(pool.imap(one_day, days), 1):
            rows.append(r); pools.append(p); cpool.append(c)
            if i % 25 == 0:
                print(f"  {i}/{len(days)} days, {(time.time() - t0) / i:.1f} s/day", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"national_{EVAL_YEAR}_persistence_daily.csv", index=False)
    P = {k: np.concatenate([p[k] for p in pools]) for k in pools[0]}
    out = dict(baseline="persistence: distance to nearest fire on days D-2..D, continent-wide, "
                        "random tie-break", seed=SEED, eval_year=EVAL_YEAR, n_days=int(len(df)),
               neg_sample_frac=NEG_FRAC, n_land_px=int(df.n_land.iloc[0]),
               fire_px_total=int(df.n_fire.sum()), base_rate=float(df.n_fire.sum() / df.n_land.sum()))
    out["pooled_auc_pr"], out["pooled_roc_auc"] = ap_auc(P["s"], P["y"], P["w"])
    newmask = P["new"] | ~P["y"]
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
    CS = np.concatenate([c[0] for c in cpool]); CY = np.concatenate([c[1] for c in cpool])
    out["cells"] = dict(cell_km=CELL, n_cell_days=int(CS.size), base_rate=float(CY.mean()))
    out["cells"]["pooled_auc_pr"], out["cells"]["pooled_roc_auc"] = ap_auc(CS, CY)
    out["cells"]["topk"] = [dict(k=k, tpr=float(np.nanmean(df[f"cell_tpr_{k:g}"])),
                                 lift=float(np.nanmean(df[f"cell_lift_{k:g}"]))) for k in (0.01, 0.05, 0.10)]
    with open(f"national_{EVAL_YEAR}_persistence.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)

    print(f"\n*** PERSISTENCE {EVAL_YEAR}: {len(df)} days, base rate {out['base_rate']:.5f} ***")
    print(f"  pooled AUC-PR {out['pooled_auc_pr']:.4f}  ROC-AUC {out['pooled_roc_auc']:.4f}")
    print(f"  25 km cells: pooled AUC-PR {out['cells']['pooled_auc_pr']:.4f}")
    print(f"wrote national_{EVAL_YEAR}_persistence.json, _persistence_daily.csv", flush=True)


if __name__ == "__main__":
    main()
