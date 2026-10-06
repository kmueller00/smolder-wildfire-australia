"""Stores for the causal inputs (CAUSAL_INPUTS=1, see zarr_dual_datamodule).

STEP=tasmax
  Adds tasmax (daily maximum 2 m air temperature of the UTC day, deg C) to
  barra_c2_daily.zarr from the downloaded BARRA-C2 day files, and its mean and
  standard deviation over the days of 2015-2018 to the store's
  stats_2015_2018. It replaces the gap-filled MODIS LST, whose cloudy days
  are filled with a smoothing spline over the whole year (Zhang et al. 2022)
  and so carry later days. Rebuild barra_c2_fast.zarr afterwards
  (build_fast_stores STEP=barra).

STEP=agb
  agb_yearly.zarr: the ESA CCI biomass maps of 2010 and 2015-2020 on the model
  grid, agb (n_years, H, W) float32 (NaN = no data) and years. The datamodule
  gives an issue day in year Y the map of Y-1 (2010 for 2015). The daily
  cubes hold the mean of the 2015-2020 maps, which reaches past every issue
  day. attrs mean/std: over land and the maps the training issue days use
  (2010, 2015, 2016, 2017), each map weighted equally.

STEP=landcover
  landcover_yearly.zarr: Copernicus LC100 v3.0.1 of 2015 (base) and
  2016-2019 (2016-2018 consolidated, 2019 near real time), landcover
  (n_years, H, W) uint8 (255 = no data) and years. Every LC100 map is made
  from three years of input; the base (2015) and consolidated (2016-2018)
  maps are centred on their year, so the map of X uses data up to X+1. The
  datamodule gives an issue day in year Y the map of Y-2, at least 2015: the
  evaluation years see nothing after Y-1 (2019: map 2017; 2020: map 2018).
  Training issue days in 2015 and 2016 get the 2015 base map, which uses
  data up to 2016 (no earlier map exists). The daily cubes hold the most
  frequent class of the 2015-2020 maps.

Usage
  SMOLDER_DATA=... STEP=tasmax|agb|landcover|all WORKERS=8 python -m smolder.data.build_causal_stores
"""
import datetime as dt
import glob
import os
import time
from multiprocessing import Pool

import netCDF4
import numpy as np
import rasterio
import zarr
from numcodecs import Blosc

from smolder.data.io import open_zarr_root, resolve

ROOT = os.environ.get("RAW_ROOT", "/home/saturn/gwgi/gwgi107h/wildfire_data")
BARRA_DIR = os.environ.get("BARRA_DIR", f"{ROOT}/barra_c2/day")
WORKERS = int(os.environ.get("WORKERS", 8))
START = dt.date(2015, 1, 1)
H, W = 3474, 4110
GT = (0.009997566, 0.0, 112.904998779, 0.0, -0.009997122, -9.005000114)
COMP = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


def _tasmax_month(path):
    out = zarr.open_group(str(resolve("barra_c2_daily.zarr")), mode="r+")
    with netCDF4.Dataset(path) as d:
        t = d.variables["time"]
        days = netCDF4.num2date(t[:], t.units, only_use_cftime_datetimes=False)
        x = np.ma.filled(d.variables["tasmax"][:].astype(np.float32), np.nan) - 273.15
        lat = d.variables["lat"][:]
    g = np.array([(dt.date(v.year, v.month, v.day) - START).days for v in days])
    assert np.all(np.diff(lat) > 0)
    keep = (g >= 0) & (g < out["tasmax"].shape[0])
    for k in np.nonzero(keep)[0]:
        out["tasmax"][g[k]] = x[k].astype(np.float16)
    return path, int(g.min()), int(g.max())


def build_tasmax():
    root = zarr.open_group(str(resolve("barra_c2_daily.zarr")), mode="r+")
    n, nl, nk = root["vpd"].shape
    root.create_dataset("tasmax", shape=(n, nl, nk), chunks=(1, 256, 256), dtype=np.float16,
                        fill_value=np.nan, compressor=COMP, overwrite=True)
    files = sorted(glob.glob(f"{BARRA_DIR}/tasmax/*.nc"))
    t0 = time.time()
    with Pool(WORKERS) as pool:
        for i, (p, lo, hi) in enumerate(pool.imap_unordered(_tasmax_month, files), 1):
            if i % 12 == 0:
                print(f"  tasmax {i}/{len(files)} ({time.time() - t0:.0f} s)", flush=True)
    end = (dt.date(2019, 1, 1) - START).days
    rng = np.random.default_rng(0)
    vals = []
    for g in rng.choice(end, 120, replace=False):
        a = np.asarray(root["tasmax"][int(g)], np.float32)
        vals.append(a[np.isfinite(a)][::7])
    v = np.concatenate(vals)
    st = dict(root.attrs["stats_2015_2018"])
    st["tasmax"] = [float(v.mean()), float(v.std())]
    variables = list(root.attrs["variables"])
    root.attrs.update(stats_2015_2018=st, variables=variables + (["tasmax"] if "tasmax" not in variables else []),
                      tasmax="daily maximum 2 m air temperature of the UTC day (deg C), BARRA-C2 tasmax")
    filled = sum(int(np.isfinite(float(root["tasmax"][g, 500, 600]))) for g in range(n))
    print(f"[tasmax] done, stats {st['tasmax']}, {filled}/{n} days filled at a test pixel", flush=True)


def _read_tif(path, nodata_to):
    with rasterio.open(path) as r:
        assert r.shape == (H, W), (path, r.shape)
        assert np.allclose(tuple(r.transform)[:6], GT, atol=1e-8), (path, r.transform)
        a = r.read(1)
        nd = r.nodata
    if nodata_to is None:
        return a, nd
    a = a.astype(np.float32)
    a[a == nd] = nodata_to
    return a, nd


def build_agb():
    files = {int(p.split("/")[-2]): p for p in glob.glob(f"{ROOT}/CCI_ABG/*/*.tif")}
    years = sorted(files)
    assert years == [2010, 2015, 2016, 2017, 2018, 2019, 2020], years
    root = zarr.open_group(str(resolve("agb_yearly.zarr")), mode="w")
    z = root.create_dataset("agb", shape=(len(years), H, W), chunks=(1, 512, 512), dtype=np.float32,
                            fill_value=np.nan, compressor=COMP)
    lm = np.asarray(open_zarr_root("cube_daily_smgrid_2019.zarr")["landmask"][...]) > 0
    cube_mean = np.asarray(open_zarr_root("cube_daily_smgrid_2019.zarr")["agb"][...], np.float32)
    maps = {}
    for i, y in enumerate(years):
        a, _ = _read_tif(files[y], np.nan)
        z[i] = a
        maps[y] = a
    # the cubes' agb is the mean of the 2015-2020 maps: check that these are the same maps
    m = np.nanmean(np.stack([maps[y] for y in range(2015, 2021)]), 0)
    ok = lm & np.isfinite(m) & np.isfinite(cube_mean)
    assert np.abs(m[ok] - cube_mean[ok]).max() < 1e-3, np.abs(m[ok] - cube_mean[ok]).max()
    train = np.concatenate([maps[y][lm & np.isfinite(maps[y])] for y in (2010, 2015, 2016, 2017)])
    root["years"] = np.asarray(years)
    root.attrs.update(rule="issue day in year Y: map of Y-1 (2010 for 2015)", units="Mg/ha",
                      mean=float(train.mean()), std=float(train.std()),
                      stats_from="land pixels of the 2010, 2015, 2016 and 2017 maps (used by training issue days 2015-2018)",
                      source="ESA CCI Biomass, CCI_ABG/<year>/*smgrid.tif")
    print(f"[agb] years {years}, mean {train.mean():.3f}, std {train.std():.3f}", flush=True)


def build_landcover():
    files = {int(p.split("/")[-2]): p for p in glob.glob(f"{ROOT}/landcover/*/*.tif")}
    years = [y for y in sorted(files) if y <= 2019]           # LC100 v3 ends in 2019 (the 2020 folder repeats it)
    assert years == [2015, 2016, 2017, 2018, 2019], years
    root = zarr.open_group(str(resolve("landcover_yearly.zarr")), mode="w")
    z = root.create_dataset("landcover", shape=(len(years), H, W), chunks=(1, 512, 512), dtype=np.uint8,
                            fill_value=255, compressor=COMP)
    for i, y in enumerate(years):
        a, nd = _read_tif(files[y], None)
        assert a.dtype == np.uint8 and nd == 255, (y, a.dtype, nd)
        z[i] = a
    root["years"] = np.asarray(years)
    root.attrs.update(rule="issue day in year Y: map of max(Y-2, 2015)",
                      source="Copernicus LC100 v3.0.1 (2015 base, 2016-2018 conso, 2019 nrt), landcover/<year>/*.tif")
    print(f"[landcover] years {years}", flush=True)


def main():
    step = os.environ.get("STEP", "all")
    if step in ("tasmax", "all"):
        build_tasmax()
    if step in ("agb", "all"):
        build_agb()
    if step in ("landcover", "all"):
        build_landcover()


if __name__ == "__main__":
    main()
