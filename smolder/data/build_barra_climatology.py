"""Per-cell climatology of BARRA-C2 VPD, 2015-2018 only.

Same recipe as the cube climatology of anomaly_feature_diagnostic.py: mean
and standard deviation for every 8th day of year (anchors at day 1, 9, ...,
361), each over all days within +-15 days of the anchor (circular in day of
year) in 2015-2018, on the native 0.04 deg grid. Written into
barra_c2_daily.zarr as vpd_clim_mean and vpd_clim_std (46, lat, lon) float32;
the datamodule interpolates linearly between anchors and bilinearly to the
patch.

Usage
  SMOLDER_DATA=... python -m smolder.data.build_barra_climatology
"""
import datetime as dt
import os

import numpy as np
import zarr

from smolder.data.io import resolve

STORE = os.environ.get("BARRA_STORE", "barra_c2_daily.zarr")
ANCHORS = np.arange(1, 366, 8)
HALF = 15
N_CLIM = (dt.date(2019, 1, 1) - dt.date(2015, 1, 1)).days        # 1461: 2015-2018


def main():
    root = zarr.open_group(str(resolve(STORE)), mode="r+")
    vpd = np.asarray(root["vpd"][:N_CLIM], np.float32)              # global days 0..1460 only
    doy = np.array([min((dt.date(2015, 1, 1) + dt.timedelta(days=g)).timetuple().tm_yday, 365)
                    for g in range(N_CLIM)])
    mu = np.empty((ANCHORS.size,) + vpd.shape[1:], np.float32)
    sd = np.empty_like(mu)
    for k, a in enumerate(ANCHORS):
        d = np.abs(doy - a)
        sel = np.minimum(d, 365 - d) <= HALF
        with np.errstate(all="ignore"):
            mu[k] = np.nanmean(vpd[sel], 0)
            sd[k] = np.nanstd(vpd[sel], 0)
    for name, arr in (("vpd_clim_mean", mu), ("vpd_clim_std", sd)):
        root.create_dataset(name, data=arr, chunks=(ANCHORS.size, 256, 256), overwrite=True)
    root.attrs["vpd_climatology"] = dict(years=[2015, 2016, 2017, 2018], anchors_doy=ANCHORS.tolist(),
                                         half_window_days=HALF)
    print("[barra-clim] written; median std over land-ish cells:", float(np.nanmedian(sd)))


if __name__ == "__main__":
    main()
