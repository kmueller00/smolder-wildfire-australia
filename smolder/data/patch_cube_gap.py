"""Fill days of a daily cube whose SM, wind, VPD, precipitation and LAI are empty.

The 2016 cube (cube_daily_smgrid_2016.zarr) holds no soil moisture, wind,
VPD, precipitation or LAI on days 163-365 (12 June to 31 December 2016);
LST, NDVI, the statics and the fire target are complete. The source
GeoTIFFs exist and are already on the cube grid, and on complete days the
cube equals them exactly: raw value, nodata -> NaN, LAI as the most recent
8-day HiQ-LAI composite starting on or before the day (checked on 2016 days
100 and 150, all land pixels, max difference 0).

The script first rebuilds VALIDATE_DAYS (complete days) from the sources and
requires an exact match with the cube, then writes the gap days. Work is
split by the cube's 7-day time chunks so no two processes write one chunk.

Usage
  YEAR=2016 DAY_FROM=163 DAY_TO=365 python -m smolder.data.patch_cube_gap
  (DRY=1 validates only)
"""
import datetime as dt
import glob
import os
import time
from multiprocessing import Pool

import numpy as np
import rasterio
import zarr

ROOT = os.environ.get("ROOT", "/home/saturn/gwgi/gwgi107h/wildfire_data")
YEAR = int(os.environ.get("YEAR", 2016))
DAY_FROM, DAY_TO = int(os.environ.get("DAY_FROM", 163)), int(os.environ.get("DAY_TO", 365))
VALIDATE_DAYS = [int(d) for d in os.environ.get("VALIDATE_DAYS", "40,100,150,162").split(",")]
DRY = os.environ.get("DRY", "0") == "1"
WORKERS = int(os.environ.get("WORKERS", 12))
CUBE = os.environ.get("CUBE", f"{ROOT}/firecastnet/cube_daily_smgrid_{YEAR}.zarr")
CH = {"sm": 0, "wind": 1, "vpd": 2, "precip": 3, "lai": 6}


def daily_path(var, day):
    s = day.strftime("%Y%m%d")
    return {"sm": f"{ROOT}/sm/daily/{day.year}/smips_smi_perc_{s}.tif",
            "wind": f"{ROOT}/wind/Wind_wgs84_1km/{day.year}/wind10m_speed_{s}.tif",
            "vpd": f"{ROOT}/vpd/VPD_wgs84_1km/{day.year}/ERA5_VPD_at_Tmax_{day.year}_VPD_{s}.tif",
            "precip": f"{ROOT}/precipitation/{day.year}/anucilm_rain_1km_{s}.tif"}[var]


LAI = sorted(glob.glob(f"{ROOT}/lai/LAI_wgs84_1km - Kopie/{YEAR}/HiQ_LAI_AUS_8d_*.tif"))
LAI_DATES = [dt.datetime.strptime(p.rsplit("_", 1)[1][:8], "%Y%m%d").date() for p in LAI]


def read(path):
    with rasterio.open(path) as r:
        a = r.read(1).astype(np.float32)
        nd = r.nodata
    if nd is not None:
        a[a == nd] = np.nan
    return a


def source_day(t):
    """The five channels of day index t, as the cube builder wrote them."""
    day = dt.date(YEAR, 1, 1) + dt.timedelta(days=t)
    out = {v: read(daily_path(v, day)) for v in ("sm", "wind", "vpd", "precip")}
    k = max(i for i, d in enumerate(LAI_DATES) if d <= day)          # LOCF of the 8-day composite
    out["lai"] = read(LAI[k])
    return out


def validate(t):
    X = zarr.open_group(CUBE, mode="r")["X"]
    x = np.asarray(X[t])
    src = source_day(t)
    bad = {}
    for v, ch in CH.items():
        a, c = src[v], x[..., ch]
        same = (a == c) | (np.isnan(a) & np.isnan(c))
        if not same.all():
            bad[v] = int((~same).sum())
    return t, bad


def patch_chunk(t_lo):
    X = zarr.open_group(CUBE, mode="r+")["X"]
    T = X.shape[0]
    t_hi = min(t_lo + X.chunks[0], T)
    blk = np.asarray(X[t_lo:t_hi])
    for t in range(max(t_lo, DAY_FROM), min(t_hi, DAY_TO + 1)):
        src = source_day(t)
        for v, ch in CH.items():
            blk[t - t_lo, ..., ch] = src[v]
    X[t_lo:t_hi] = blk
    return t_lo, t_hi


def main():
    t0 = time.time()
    with Pool(min(WORKERS, len(VALIDATE_DAYS))) as pool:
        res = pool.map(validate, VALIDATE_DAYS)
    for t, bad in res:
        print(f"[validate] day {t}: {'exact' if not bad else 'MISMATCH ' + str(bad)}", flush=True)
    if any(bad for _, bad in res):
        raise SystemExit("sources do not reproduce the complete days; not patching")
    if DRY:
        return
    X = zarr.open_group(CUBE, mode="r")["X"]
    step = X.chunks[0]
    chunks = sorted({(t // step) * step for t in range(DAY_FROM, DAY_TO + 1)})
    with Pool(WORKERS) as pool:
        for i, (lo, hi) in enumerate(pool.imap_unordered(patch_chunk, chunks), 1):
            print(f"  wrote days {lo}-{hi - 1} ({i}/{len(chunks)}, {time.time() - t0:.0f} s)", flush=True)
    with Pool(min(WORKERS, 4)) as pool:
        res = pool.map(validate, [DAY_FROM, (DAY_FROM + DAY_TO) // 2, DAY_TO])
    for t, bad in res:
        print(f"[check] patched day {t}: {'exact' if not bad else 'MISMATCH ' + str(bad)}", flush=True)


if __name__ == "__main__":
    main()
