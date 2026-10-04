"""NDVI (MODIS, 500 m) against LAI (HiQ-LAI, 5 km) on the 1 km grid, 2019.

1. Spatial detail: mean standard deviation of each variable inside 5 x 5 px
   blocks of land on a few days (how much they vary between neighbouring
   pixels).
2. ROC-AUC alone for fire in the next three days on the case-control sample
   of anomaly_feature_diagnostic.py (every 2019 fire pixel + N_NEG non-fire
   pixel-days, same seed): value on the issue day and its 144-day mean
   (D-143..D), all pixels and pixels > 10 km from fire of the last 32 days,
   by land cover. Oriented, max(AUC, 1 - AUC).

Near existing fire, what NDVI adds on top of SMOLDER is measured with
frp_gate.py (EXTRA_VEG=1). Output: ndvi_lai_check_2019.json (OUT).
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from sklearn.metrics import roc_auc_score

from smolder.data.io import daily_cube, open_zarr_root
from smolder.evaluation.anomaly_feature_diagnostic import FAR_PX, LANDCOVER, LANDCOVERS, OFFSETS, sample

OUT = os.environ.get("OUT", "ndvi_lai_check_2019.json")
WORKERS = int(os.environ.get("WORKERS", 8))
NDVI, LAI = 5, 6
G = {}


def _init():
    G["X"] = {y: open_zarr_root(daily_cube(y))["X"] for y in (2018, 2019)}


def day_values(args):
    D, rr, cc = args
    out = {}
    g = D + OFFSETS[2019]
    vals = {NDVI: [], LAI: []}
    for gg in range(g - 143, g + 1, 8):                    # 18 days spread over the window (cube chunks are 7 days)
        y = 2019 if gg >= OFFSETS[2019] else 2018
        x = np.asarray(G["X"][y][gg - OFFSETS[y], :, :, [NDVI, LAI]], np.float32)
        for k, ch in enumerate((NDVI, LAI)):
            vals[ch].append(x[rr, cc, k])
    for k, ch in enumerate((NDVI, LAI)):
        out[ch] = (vals[ch][-1], np.nanmean(np.stack(vals[ch]), 0))
    return D, out


def auc(y, v):
    ok = np.isfinite(v)
    if y[ok].sum() < 20 or (~y[ok]).sum() < 20:
        return None
    a = float(roc_auc_score(y[ok], v[ok]))
    return round(max(a, 1 - a), 4)


def main():
    t0 = time.time()
    g = open_zarr_root(daily_cube(2019))
    land = np.asarray(g["landmask"][...]) > 0
    detail = {}
    for D in (60, 180, 300):
        x = np.asarray(g["X"][D, :, :, [NDVI, LAI]], np.float32)
        H, W = land.shape
        h, w = H // 5 * 5, W // 5 * 5
        for k, name in enumerate(("ndvi", "lai")):
            a = np.where(land, x[..., k], np.nan)[:h, :w].reshape(h // 5, 5, w // 5, 5)
            with np.errstate(all="ignore"):
                sd = np.nanstd(a, (1, 3)); m = np.nanmean(a, (1, 3))
            ok = np.isfinite(sd) & (np.isfinite(a).mean((1, 3)) > 0.8)
            detail.setdefault(name, []).append(dict(day=D, within_5px_sd=float(np.mean(sd[ok])),
                                                    within_5px_cv=float(np.mean(sd[ok] / np.maximum(np.abs(m[ok]), 1e-6)))))
    S, _ = sample()
    n = S["y"].size
    vals = {k: np.full(n, np.nan) for k in ("ndvi", "ndvi_144d", "lai", "lai_144d")}
    jobs = [(int(D), S["r"][S["D"] == D], S["c"][S["D"] == D]) for D in np.unique(S["D"])]
    idx = {int(D): np.flatnonzero(S["D"] == D) for D in np.unique(S["D"])}
    with Pool(WORKERS, initializer=_init) as pool:
        for k, (D, o) in enumerate(pool.imap_unordered(day_values, jobs), 1):
            i = idx[D]
            vals["ndvi"][i], vals["ndvi_144d"][i] = o[NDVI]
            vals["lai"][i], vals["lai_144d"][i] = o[LAI]
            if k % 50 == 0:
                print(f"  {k}/{len(jobs)} days ({time.time() - t0:.0f} s)", flush=True)
    far = S["d32"] > FAR_PX
    lcn = np.array([LANDCOVER.get(int(v), "other") for v in S["lc"]])
    tabs = {}
    for sel_name, sel in (("a_all", np.ones(n, bool)), ("b_far_10km", far)):
        tabs[sel_name] = {"overall": {k: auc(S["y"][sel], v[sel]) for k, v in vals.items()},
                          "by_landcover": {lc: {k: auc(S["y"][sel & (lcn == lc)], v[sel & (lcn == lc)])
                                                for k, v in vals.items()} for lc in LANDCOVERS}}
    out = dict(eval_year=2019, spatial_detail=detail, n_fire=int(S["y"].sum()), n_nofire=int((~S["y"]).sum()),
               tables=tabs, note="oriented ROC-AUC; 144d = mean of 18 days spread over D-143..D")
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(detail, indent=0))
    for k in ("a_all", "b_far_10km"):
        print(k, tabs[k]["overall"])
        print("   ", {lc: tabs[k]["by_landcover"][lc] for lc in LANDCOVERS})


if __name__ == "__main__":
    main()
