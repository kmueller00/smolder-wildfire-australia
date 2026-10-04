"""BARRA-C2 afternoon wind (05 UTC) into one store on the native grid.

Reads the monthly files of hourly uas and vas at 05 UTC only (15:00 AEST,
13:00 AWST) downloaded with the NCI THREDDS subset service and writes
barra_c2_pm_wind.zarr: uas_05, vas_05 (global day 0 = 2015-01-01 through
2021-01-31, UTC days, float16, chunks (1, 256, 256)), plus lat and lon.

Usage
  PM_DIR=.../barra_c2/1hr_05utc SMOLDER_DATA=... python -m smolder.data.build_barra_pm_wind
"""
import datetime as dt
import glob
import os

import netCDF4
import numpy as np
import zarr
from numcodecs import Blosc

from smolder.data.io import resolve

PM_DIR = os.environ.get("PM_DIR", "/home/saturn/gwgi/gwgi107h/wildfire_data/barra_c2/1hr_05utc")
OUT = os.environ.get("OUT", "barra_c2_pm_wind.zarr")
START = dt.date(2015, 1, 1)
N_DAYS = (dt.date(2021, 2, 1) - START).days


def main():
    root = None
    for var in ("uas", "vas"):
        files = sorted(glob.glob(f"{PM_DIR}/{var}/{var}_05utc_*.nc"))
        for p in files:
            with netCDF4.Dataset(p) as d:
                if root is None:
                    lat = np.asarray(d.variables["lat"][:], np.float64)
                    lon = np.asarray(d.variables["lon"][:], np.float64)
                    root = zarr.open_group(str(resolve(OUT)), mode="w")
                    comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
                    for v in ("uas_05", "vas_05"):
                        root.create_dataset(v, shape=(N_DAYS, lat.size, lon.size), chunks=(1, 256, 256),
                                            dtype=np.float16, fill_value=np.nan, compressor=comp)
                    root["lat"], root["lon"] = lat, lon
                t = d.variables["time"]
                days = netCDF4.num2date(t[:], t.units, only_use_cftime_datetimes=False)
                x = np.ma.filled(d.variables[var][:].astype(np.float32), np.nan)
            for k, v in enumerate(days):
                assert v.hour == 5, (p, v)
                root[f"{var}_05"][(dt.date(v.year, v.month, v.day) - START).days] = x[k].astype(np.float16)
        print(f"[pm-wind] {var}: {len(files)} monthly files", flush=True)
    root.attrs.update(start_date=START.isoformat(), hour_utc=5, source="BARRA-C2 v1 AUST-04 1hr (NCI ob53), NCSS")
    filled = sum(np.isfinite(np.asarray(root["uas_05"][g, 500, 600], np.float32)) for g in range(N_DAYS))
    print(f"[pm-wind] {filled}/{N_DAYS} days filled at a test cell")


if __name__ == "__main__":
    main()
