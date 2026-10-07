"""Skill of single predictors for fire in the next three days as a function of
lag, with the inputs as the final model reads them (causal_inputs).

Method of the July 2026 analysis (firecastnet/lagged_skill_extended.py): for
each year and lag, DAYS_PER_CELL random target days t and N_SAMPLES random land
pixels in all; the predictor at day t - lag against the target y_fire_3d[t]
(fire on days t+1..t+3); ROC-AUC (not oriented: below 0.5 means lower values
precede fire) averaged over the years. Changes:
  - years 2015-2019 only (no test year), target days whose window D+1..D+3
    stays inside the year, t - lag >= 31 so that every variable exists;
  - inputs as in causal_inputs: LAI = HiQ-LAI 500 m of the newest composite
    starting on or before day - 31; NDVI = the cube's NDVI of day - 7 (the
    newest composite that has ended); VPD and maximum air temperature from
    BARRA-C2 (nearest 0.04 deg cell); soil moisture, precipitation (daily
    total) and wind speed from the daily cubes;
  - "fire on the issue day": any VIIRS detection at the pixel on day t, the
    day before the target window (firms_daily.zarr), same samples.
Writes results/lag_analysis_causal.json.

  SMOLDER_DATA=... WORKERS=8 python -m smolder.evaluation.lag_analysis_causal
"""
import datetime as dt
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
import rasterio
from sklearn.metrics import roc_auc_score

from smolder.data.io import open_zarr_root

YEARS = (2015, 2016, 2017, 2018, 2019)
LAGS = [0, 1, 2, 3, 5, 7, 10, 14, 21, 28, 30, 35, 42, 56, 70, 90, 110, 130, 150, 180]
N_SAMPLES = int(os.environ.get("N_SAMPLES", 400_000))
DAYS_PER_CELL = int(os.environ.get("DAYS_PER_CELL", 40))
LAI_DIR = os.environ.get("LAI_DIR", "/home/saturn/gwgi/gwgi107h/wildfire_data/lai/HiQ_LAI_1km")
LAI_LAG, NDVI_LAG = 31, 7
START = dt.date(2015, 1, 1)
OFF = {y: (dt.date(y, 1, 1) - START).days for y in range(2015, 2021)}
VARS = ["LAI", "NDVI", "VPD", "TMAX", "SM", "PPT", "WIND"]
LAT0, LON0, PX = -9.005000114, 112.904998779, 0.009997121616580312
G = {}


def _init():
    G["cube"] = {y: open_zarr_root(f"cube_daily_smgrid_{y}.zarr") for y in YEARS}
    G["ch"] = {n: i for i, n in enumerate(G["cube"][2019].attrs["dyn_vars"])}
    G["barra"] = open_zarr_root("barra_c2_daily.zarr")
    G["blat"] = np.asarray(G["barra"]["lat"][...]); G["blon"] = np.asarray(G["barra"]["lon"][...])
    G["firms"] = open_zarr_root("firms_daily.zarr")["n_det"]
    comps = []
    for p in sorted(glob.glob(f"{LAI_DIR}/hiq_lai_1km_*.tif")):
        with rasterio.open(p) as r:
            names = list(r.descriptions)
            year = int(p.rsplit("_", 1)[1][:4])
            for b in range(1, r.count + 1):
                d = names[b - 1]
                start = (dt.date(int(d[:4]), int(d[4:6]), int(d[6:])) if d and len(d) >= 8 and d[:8].isdigit()
                         else dt.date(year, 1, 1) + dt.timedelta(days=8 * (b - 1)))
                comps.append(((start - START).days, p, b))
    comps.sort()
    G["lai_starts"] = np.array([c[0] for c in comps]); G["lai_comps"] = comps
    G["lai_cache"] = {}


def _day(g):
    d = START + dt.timedelta(days=int(g))
    return d.year, d.timetuple().tm_yday - 1


def _X(g, yy, xx, ch):
    y, i = _day(g)
    return np.asarray(G["cube"][y]["X"][i, :, :, ch], np.float32)[yy, xx]


def _lai(g, yy, xx):
    k = int(np.searchsorted(G["lai_starts"], g - LAI_LAG, side="right")) - 1
    assert k >= 0 and G["lai_starts"][k] + LAI_LAG <= g
    if k not in G["lai_cache"]:
        G["lai_cache"].clear()
        _, p, b = G["lai_comps"][k]
        with rasterio.open(p) as r:
            a = r.read(b).astype(np.float32)
        a[a == 65535] = np.nan
        G["lai_cache"][k] = a / 100.0
    return G["lai_cache"][k][yy, xx]


def _barra(var, g, yy, xx):
    lat = LAT0 - (yy + 0.5) * PX; lon = LON0 + (xx + 0.5) * 0.009997566018978103
    bi = np.clip(np.searchsorted(G["blat"], lat), 1, len(G["blat"]) - 1)
    bi -= (np.abs(G["blat"][bi - 1] - lat) < np.abs(G["blat"][bi] - lat))
    bj = np.clip(np.searchsorted(G["blon"], lon), 1, len(G["blon"]) - 1)
    bj -= (np.abs(G["blon"][bj - 1] - lon) < np.abs(G["blon"][bj] - lon))
    a = np.asarray(G["barra"][var][g], np.float32)
    return a[bi, bj]


def cell(args):
    year, lag, seed = args
    rng = np.random.default_rng(seed)
    g0 = OFF[year]
    n_days = OFF.get(year + 1, g0 + 365) - g0
    lm = np.asarray(G["cube"][year]["landmask"][...]) > 0
    land = np.argwhere(lm)
    t_lo = max(g0, lag + LAI_LAG + NDVI_LAG)                     # every variable exists
    t_hi = g0 + n_days - 3                                       # window D+1..D+3 inside the year
    days = rng.choice(np.arange(t_lo, t_hi), size=min(DAYS_PER_CELL, t_hi - t_lo), replace=False)
    per = max(1, N_SAMPLES // len(days))
    lab, fire_d, vals = [], [], {v: [] for v in VARS}
    for t in days:
        idx = rng.integers(0, len(land), per); yy, xx = land[idx, 0], land[idx, 1]
        y, i = _day(t)
        lab.append(np.asarray(G["cube"][y]["y_fire_3d"][i], np.uint8)[yy, xx] > 0)
        fire_d.append(np.asarray(G["firms"][t], np.uint8)[yy, xx] > 0)
        s = int(t) - lag
        X = G["cube"][_day(s)[0]]["X"][_day(s)[1]]
        X = np.asarray(X, np.float32)
        vals["SM"].append(X[yy, xx, G["ch"]["sm"]]); vals["PPT"].append(X[yy, xx, G["ch"]["precip"]])
        vals["WIND"].append(X[yy, xx, G["ch"]["wind"]])
        vals["NDVI"].append(_X(s - NDVI_LAG, yy, xx, G["ch"]["ndvi"]))
        vals["LAI"].append(_lai(s, yy, xx))
        vals["VPD"].append(_barra("vpd", s, yy, xx)); vals["TMAX"].append(_barra("tasmax", s, yy, xx))
    lab = np.concatenate(lab); out = {}
    for v in VARS:
        x = np.concatenate(vals[v]); m = np.isfinite(x)
        out[v] = dict(roc_auc=float(roc_auc_score(lab[m], x[m])), n=int(m.sum()), fire_frac=float(lab[m].mean()))
    if lag == 0:
        f = np.concatenate(fire_d).astype(np.float32)
        out["fire on the issue day"] = dict(roc_auc=float(roc_auc_score(lab, f)), n=int(len(lab)),
                                            fire_frac=float(lab.mean()))
    return year, lag, out


def main():
    jobs = [(y, l, 1000 * y + l) for y in YEARS for l in LAGS]
    res = {}
    with Pool(int(os.environ.get("WORKERS", 8)), initializer=_init) as pool:
        for year, lag, out in pool.imap_unordered(cell, jobs):
            for v, d in out.items():
                res.setdefault(v, {}).setdefault(str(lag), {})[str(year)] = d
            print(f"[lag] {year} lag {lag}", flush=True)
    summary = {}
    for v, by_lag in res.items():
        mean = {int(l): float(np.mean([d["roc_auc"] for d in yrs.values()])) for l, yrs in by_lag.items()}
        lags = sorted(mean)
        best = max(lags, key=lambda l: abs(mean[l] - 0.5))
        summary[v] = dict(roc_auc_by_lag={str(l): mean[l] for l in lags}, best_lag=best, roc_auc_best=mean[best],
                          roc_auc_lag0=mean.get(0), roc_auc_lag30=mean.get(30),
                          note="best lag = largest distance of the ROC-AUC from 0.5 (values below 0.5: lower values precede fire)")
    out = dict(method=__doc__.split("Writes")[0].strip(), years=list(YEARS), lags=LAGS, n_samples=N_SAMPLES,
               days_per_cell=DAYS_PER_CELL, summary=summary, per_year=res)
    p = os.path.join(os.path.dirname(__file__), "..", "..", "results", "lag_analysis_causal.json")
    json.dump(out, open(p, "w"), indent=1)
    for v, s in summary.items():
        print(f"{v:24s} best lag {s['best_lag']:4d}  AUC {s['roc_auc_best']:.4f}  lag0 {s['roc_auc_lag0']:.4f}  "
              f"lag30 {s['roc_auc_lag30'] if s['roc_auc_lag30'] is not None else float('nan'):.4f}")


if __name__ == "__main__":
    main()
