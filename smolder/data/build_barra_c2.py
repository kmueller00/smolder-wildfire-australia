"""BARRA-C2 daily fields on their native 0.04 deg grid, one store.

Reads the monthly BARRA-C2 (AUST-04, 4.4 km) daily files downloaded from
NCI THREDDS (ob53, product BARRA-C2 v1, frequency day) and writes
barra_c2_daily.zarr with, per global day (0 = 2015-01-01, through
2021-01-31; daily values are UTC days, like the VIIRS fire days):

  sfcWindmax  daily maximum 10 m wind speed (m/s)
  uas, vas    daily mean 10 m wind components (m/s, eastward / northward)
  vpd         vapour pressure deficit at the daily maximum temperature (kPa):
              es(tasmax) - e, with es from the FAO-56 Tetens form and the
              actual vapour pressure e = q p / (0.622 + 0.378 q) from the
              daily mean specific humidity q and surface pressure p

float16, chunks (1, 256, 256), plus lat (ascending) and lon. The model grid
is filled by bilinear interpolation per patch (zarr_dual_datamodule).
Licence CC BY 4.0; cite the dataset (doi 10.25914/1X6G-2V48) and Su et al.
(2025), JSHESS 75, ES25032.

Usage
  BARRA_DIR=.../barra_c2/day SMOLDER_DATA=... python -m smolder.data.build_barra_c2
"""
import datetime as dt
import glob
import os
import time
from multiprocessing import Pool

import netCDF4
import numpy as np
import zarr
from numcodecs import Blosc

from smolder.data.io import resolve

BARRA_DIR = os.environ.get("BARRA_DIR", "/home/saturn/gwgi/gwgi107h/wildfire_data/barra_c2/day")
OUT = os.environ.get("OUT", "barra_c2_daily.zarr")
WORKERS = int(os.environ.get("WORKERS", 24))
START = dt.date(2015, 1, 1)
N_DAYS = (dt.date(2021, 2, 1) - START).days        # 2223
VARS = ("sfcWindmax", "uas", "vas", "vpd")


def es_kpa(t_c):
    return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))


def _read(var, ym):
    path = f"{BARRA_DIR}/{var}/{var}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_day_{ym}-{ym}.nc"
    with netCDF4.Dataset(path) as d:
        t = d.variables["time"]
        days = netCDF4.num2date(t[:], t.units, only_use_cftime_datetimes=False)
        x = np.ma.filled(d.variables[var][:].astype(np.float32), np.nan)
    gidx = np.array([(dt.date(v.year, v.month, v.day) - START).days for v in days])
    return gidx, x


def month(ym):
    out = zarr.open_group(str(resolve(OUT)), mode="r+")
    for var in ("sfcWindmax", "uas", "vas"):
        gidx, x = _read(var, ym)
        for k, g in enumerate(gidx):
            out[var][g] = x[k].astype(np.float16)
    gidx, tmax = _read("tasmax", ym)
    g2, q = _read("huss", ym)
    g3, p = _read("ps", ym)
    assert np.array_equal(gidx, g2) and np.array_equal(gidx, g3), ym
    e = q * p / (0.622 + 0.378 * q) / 1000.0
    vpd = np.maximum(es_kpa(tmax - 273.15) - e, 0.0)
    for k, g in enumerate(gidx):
        out["vpd"][g] = vpd[k].astype(np.float16)
    return ym, gidx.min(), gidx.max()


def main():
    files = sorted(glob.glob(f"{BARRA_DIR}/tasmax/*.nc"))
    yms = [f.rsplit("_", 1)[1][:6] for f in files]
    with netCDF4.Dataset(files[0]) as d:
        lat = d.variables["lat"][:].astype(np.float64)
        lon = d.variables["lon"][:].astype(np.float64)
    assert np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0)
    root = zarr.open_group(str(resolve(OUT)), mode="w")
    comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    for v in VARS:
        root.create_dataset(v, shape=(N_DAYS, lat.size, lon.size), chunks=(1, 256, 256), dtype=np.float16,
                            fill_value=np.nan, compressor=comp)
    root["lat"] = lat
    root["lon"] = lon
    root.attrs.update(start_date=START.isoformat(), n_days=N_DAYS, day="UTC day",
                      source="BARRA-C2 v1 AUST-04 day (NCI ob53)", variables=list(VARS),
                      vpd="es(tasmax) - q p / (0.622 + 0.378 q), kPa, FAO-56 Tetens es")
    t0 = time.time()
    with Pool(WORKERS) as pool:
        for i, (ym, lo, hi) in enumerate(pool.imap_unordered(month, yms), 1):
            print(f"  {ym} days {lo}-{hi} ({i}/{len(yms)}, {time.time() - t0:.0f} s)", flush=True)
    filled = [int(np.isfinite(np.asarray(root["vpd"][g, 500, 600], np.float32))) for g in range(N_DAYS)]
    print(f"[barra] {sum(filled)}/{N_DAYS} days filled at a test pixel")


if __name__ == "__main__":
    main()
