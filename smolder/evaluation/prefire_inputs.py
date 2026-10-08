"""Inputs before fire by forecast outcome, from the STORED national scores (the
model is not run). Same 1500 fire-active 384 px patches as explain_topk.py
(same data module, same seed); in each patch every land pixel is classed with
the stored score of the issue day (SCORES, written by evaluate_national) at the
best-F2 threshold of THRESHOLDS_FROM (2019):
  hit          flagged, burned in the next three days
  false_alarm  flagged, no fire
  miss         burned, not flagged
  background   neither
Per patch up to PER_CLASS pixels of each class are drawn (fixed seed per
patch), with cls_n = the class's pixels in the patch for weighting, and the
inputs as the model reads them on the issue day, back in native units:
  sm_bXX, ppt_bXX, lai_bXX  the 18 8-day averages of the slow branch
  vpd_dXX, tmax_dXX         the 14 days of the fast branch (BARRA-C2, kPa, deg C)
  ndvi_dXX                  the 14 days of the fast branch (newest complete
                            MOD09A1 composite of each day; NaN where missing)

Writes OUT (csv.gz). Usage:
  SMOLDER_DATA=... EVAL_YEAR=2020 SCORES=... THRESHOLDS_FROM=... OUT=... python -m smolder.evaluation.prefire_inputs
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from smolder.data.io import CHANNEL_STATS, daily_cube, open_zarr_root
from smolder.data.zarr_dual_datamodule import CH, SLOW_CHANNELS, DualPatchConfig, DualWindowDataset

EVAL_YEAR = int(os.environ.get("EVAL_YEAR", 2020))
OFFSETS = {2019: 1461, 2020: 1826}
SCORES = os.environ["SCORES"]
OUT = os.environ["OUT"]
THR = json.load(open(os.environ["THRESHOLDS_FROM"]))["adaptive_best"]["SMOLDER"]["f2"]["threshold"]
P_THR = 1.0 / (1.0 + np.exp(-THR))
N_PATCH = int(os.environ.get("N_PATCH", 1500))
PER_CLASS = int(os.environ.get("PER_CLASS", 60))
WORKERS = int(os.environ.get("WORKERS", 16))
PATCH, SEED = 384, 21                                   # as explain_topk.py
CLIMATE = {**{k: "tropical" for k in (1, 2, 3)}, **{k: "arid" for k in (4, 5, 6, 7)},
           **{k: "temperate" for k in range(8, 30)}}
G = {}


def _init():
    cube = daily_cube(EVAL_YEAR)
    G["times"] = list(open_zarr_root(cube).attrs["time"])
    G["ds"] = DualWindowDataset(DualPatchConfig(          # the configuration of explain_topk.py
        zarr_paths=(cube,), stats_path="channel_stats_2015_2018.json",
        slow_cube_path="cube_slow_8day.zarr", day_offset=OFFSETS[EVAL_YEAR], patch_size=PATCH,
        samples_per_epoch=N_PATCH * 3, seed=SEED, min_pos_pixels=45, pos_frac=1.0,
        deterministic=True, fire_history=True, fire_history_lags=(3, 4, 5), fire_history_distance=True,
        use_lightning=os.environ.get("USE_LIGHTNING", "0") == "1",
        use_elevation=os.environ.get("USE_ELEVATION", "0") == "1",
        use_slope_aspect=os.environ.get("USE_SLOPE_ASPECT", "0") == "1"))
    G["land"] = np.asarray(open_zarr_root(cube)["landmask"][...]) > 0
    G["scores"] = np.load(SCORES, mmap_mode="r")
    G["row"] = {int(D): i for i, D in enumerate(np.load(SCORES + ".days.npy"))}
    stats = json.loads(CHANNEL_STATS.read_text())
    G["ndvi"] = (float(stats["x_mean"][CH["NDVI"]]), float(stats["x_std"][CH["NDVI"]]))


def one_patch(i):
    ds = G["ds"]
    b = ds[i]
    land = b["mask"].numpy() > 0.5
    y = (b["y"][-1].numpy() > 0) & land
    if y.sum() < 45 or land.mean() < 0.5:                 # the acceptance rule of explain_topk.py
        return i, None
    t_end, y0, x0 = int(b["t_end"]), int(b["y0"]), int(b["x0"])
    D = t_end - 1                                          # issue day (day index of the year)
    if D not in G["row"]:
        return i, None
    full = np.full(G["land"].shape, np.nan, np.float32)
    full[G["land"]] = G["scores"][G["row"][D]]
    p = full[y0:y0 + PATCH, x0:x0 + PATCH]
    flag = land & (p >= P_THR)
    classes = {"hit": flag & y, "false_alarm": flag & ~y, "miss": y & ~flag, "background": land & ~y & ~flag}
    lay = ds.channel_layout()["fast"]
    slow, fast = b["x_slow"].numpy(), b["x_fast"].numpy()
    kg = b["x_cat"][..., 1].numpy()
    bm, bs = ds.barra_stats["vpd"]
    tm, ts = ds.barra_stats["tasmax"]
    nm, ns = G["ndvi"]
    iv, it, ind = lay["VPD"][0], lay["TMAX"][0], lay["NDVI (fast)"][0]
    sl = {c: SLOW_CHANNELS.index(c) for c in ("SM", "PPT", "LAI")}
    rng = np.random.default_rng([SEED, i])
    rows = []
    for cname, m in classes.items():
        yy, xx = np.nonzero(m)
        if yy.size == 0:
            continue
        for j in rng.choice(yy.size, min(PER_CLASS, yy.size), replace=False):
            a, c = yy[j], xx[j]
            ndvi = fast[:, a, c, ind] * ns + nm
            ndvi = np.where(np.abs(ndvi) < 1e-6, np.nan, ndvi)          # missing composites enter as 0
            row = dict(cls=cname, cls_n=int(yy.size), patch=i, date=G["times"][D],
                       climate=CLIMATE.get(int(kg[a, c]), "other"), risk=float(p[a, c]))
            for name, key in (("sm", "SM"), ("ppt", "PPT"), ("lai", "LAI")):
                k = sl[key]
                v = slow[:, a, c, k] * ds.slow_std[k] + ds.slow_mean[k]
                row.update({f"{name}_b{q:02d}": float(x) for q, x in enumerate(v)})
            row.update({f"vpd_d{q:02d}": float(x) for q, x in enumerate(fast[:, a, c, iv] * bs + bm)})
            row.update({f"tmax_d{q:02d}": float(x) for q, x in enumerate(fast[:, a, c, it] * ts + tm)})
            row.update({f"ndvi_d{q:02d}": float(x) for q, x in enumerate(ndvi)})
            rows.append(row)
    return i, rows


def main():
    t0 = time.time()
    kept, rows = 0, []
    with Pool(WORKERS, initializer=_init) as pool:
        for i, r in pool.imap(one_patch, range(N_PATCH * 3), chunksize=4):
            if r is None:
                continue
            kept += 1
            for row in r:
                row["patch"] = kept - 1                     # patch number in the order of explain_topk.py
            rows.extend(r)
            if kept % 100 == 0:
                print(f"  {kept}/{N_PATCH} patches ({time.time() - t0:.0f} s)", flush=True)
            if kept >= N_PATCH:
                break
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {kept} patches, {len(df)} pixels; classes {df.cls.value_counts().to_dict()}; "
          f"NDVI missing {100 * df.filter(like='ndvi_d').isna().values.mean():.1f} %", flush=True)


if __name__ == "__main__":
    main()
