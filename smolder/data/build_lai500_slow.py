"""HiQ-LAI 500 m (averaged to the 1 km grid) as 8-day slow-branch bins.

Input: the yearly GeoTIFFs written by gee_export_hiq_lai_500m.js
(hiq_lai_1km_YYYY.tif, 2015-2020, on the cube grid): one band per 8-day
composite named by its start date (YYYYMMDD), LAI x 100 as uint16, 65535 =
no valid 500 m pixel. If a file carries no band names, composite k of a year
is taken to start on day of year 1 + 8k (the MODIS 8-day calendar).

Daily values are built the way the cube's 5 km LAI was (checked on 2016:
identical to the source): each day takes the newest composite that starts
on or before it. These are averaged over the slow cube's 8-day bins and
written to cube_slow_8day_lai500.zarr (X_slow (n_bins, H, W, 1), same
bin_start_day as cube_slow_8day.zarr), used by SLOW_VEG=lai500.

Usage
  LAI_DIR=.../lai/HiQ_LAI_1km SMOLDER_DATA=... python -m smolder.data.build_lai500_slow
  CHECK_ONLY=1 ... only reads the files and prints bands, dates, value ranges
"""
import datetime as dt
import glob
import os

import numpy as np
import rasterio
import zarr

from smolder.data.io import open_zarr_root, resolve

LAI_DIR = os.environ.get("LAI_DIR", "/home/saturn/gwgi/gwgi107h/wildfire_data/lai/HiQ_LAI_1km")
OUT = os.environ.get("OUT", "cube_slow_8day_lai500.zarr")
START = dt.date(2015, 1, 1)
GT = (112.904998779, 0.009997566018978103, -9.005000113999998, -0.009997121616580312)
H, W = 3474, 4110


def composites():
    """[(start date, file, band index)] over all yearly files, sorted."""
    out = []
    for p in sorted(glob.glob(f"{LAI_DIR}/hiq_lai_1km_*.tif")):
        year = int(os.path.basename(p)[12:16])
        with rasterio.open(p) as r:
            assert (r.height, r.width) == (H, W), (p, r.shape)
            t = r.transform
            assert np.allclose([t.c, t.a, t.f, t.e], GT, atol=1e-9), (p, t)
            for b in range(1, r.count + 1):
                d = r.descriptions[b - 1]
                if d and len(d) == 8 and d.isdigit():
                    start = dt.date(int(d[:4]), int(d[4:6]), int(d[6:]))
                else:
                    start = dt.date(year, 1, 1) + dt.timedelta(days=8 * (b - 1))
                out.append((start, p, b))
    return sorted(out)


def read(p, b):
    with rasterio.open(p) as r:
        a = r.read(b).astype(np.float32)
    a[a == 65535] = np.nan
    return a / 100.0


def main():
    comp = composites()
    years = sorted({c[0].year for c in comp})
    print(f"[lai500] {len(comp)} composites, years {years}, first {comp[0][0]}, last {comp[-1][0]}")
    for y in years:
        n = sum(c[0].year == y for c in comp)
        print(f"  {y}: {n} composites")
    if os.environ.get("CHECK_ONLY", "0") == "1":
        a = read(comp[len(comp) // 2][1], comp[len(comp) // 2][2])
        print("  middle composite:", comp[len(comp) // 2][0], "valid share", float(np.isfinite(a).mean()),
              "p5/50/95", np.nanpercentile(a, [5, 50, 95]))
        return
    sc = open_zarr_root("cube_slow_8day.zarr")
    starts = np.asarray(sc["bin_start_day"][...])
    n_bins, bin_days = starts.size, int(sc.attrs["bin_days"])
    root = zarr.open_group(str(resolve(OUT)), mode="w")
    z = root.create_dataset("X_slow", shape=(n_bins, H, W, 1), chunks=(1, 512, 512, 1), dtype="f4")
    root["bin_start_day"] = starts
    root.attrs.update(channels=["LAI500"], agg=["mean"], bin_days=bin_days, source="HiQ-LAI 500 m via GEE, mean to 1 km")
    cstarts = [(c[0] - START).days for c in comp]
    cache = {}
    for b, s in enumerate(starts):
        days = range(int(s), int(s) + bin_days)
        acc = np.zeros((H, W), np.float64)
        for g in days:
            k = int(np.searchsorted(cstarts, g, side="right")) - 1     # newest composite starting <= day
            if k < 0:
                continue
            if k not in cache:
                cache.clear()
                cache[k] = read(comp[k][1], comp[k][2])
            acc += np.nan_to_num(cache[k])                             # as build_slow_cube: missing day = 0
        z[b] = (acc / bin_days).astype(np.float32)[..., None]
        if b % 20 == 0:
            print(f"  bin {b}/{n_bins}", flush=True)
    print(f"[lai500] wrote {OUT}")


if __name__ == "__main__":
    main()
