"""Daily VIIRS fire radiative power on the model grid, 2015-2020.

From the FIRMS archive of VIIRS S-NPP 375 m Collection 2 active fires
(DL_FIRE_SV-C2 csv, converted to firms_viirs_snpp_c2_2015_2020.npz), with the
rule that reproduces the binary fire labels (VIIRS_bin_veg_*.tif, checked on
40 random days: >= 99.9 % of pixels identical): nominal or high confidence,
type 0 (vegetation fire), UTC acquisition date, row/column from the label
rasters' geotransform (pixel size 0.0099976 x 0.0099971 deg, not 0.01).

Output firms_daily.zarr, per global day (0 = 2015-01-01) and pixel:
  frp_sum   summed FRP of all detections (MW), float32
  n_det     number of detections, uint8 (capped at 255)
  n_night   number of night detections, uint8
chunks (1, 512, 512).

Usage
  SMOLDER_DATA=... python -m smolder.data.build_firms_daily
"""
import os
import time
from multiprocessing import Pool

import numpy as np
import rasterio
import zarr
from numcodecs import Blosc

from smolder.data.io import resolve

FIRMS = os.environ.get("FIRMS", "/home/saturn/gwgi/gwgi107h/wildfire_data/y_fire/firms_viirs_snpp/"
                                "firms_viirs_snpp_c2_2015_2020.npz")
LABEL_TIF = os.environ.get("LABEL_TIF", "/home/saturn/gwgi/gwgi107h/wildfire_data/y_fire/"
                                        "VIIRS_binary_veg_nominal_high/2019/VIIRS_bin_veg_2019_100.tif")
OUT = os.environ.get("OUT", "firms_daily.zarr")
N_DAYS, H, W = 2192, 3474, 4110
WORKERS = int(os.environ.get("WORKERS", 12))
G = {}


def _init():
    z = np.load(FIRMS)
    keep = np.isin(z["conf"], ["n", "h"]) & (z["type"] == 0)
    with rasterio.open(LABEL_TIF) as r:
        T = r.transform
    row = np.floor((z["lat"][keep].astype(np.float64) - T.f) / T.e).astype(np.int64)
    col = np.floor((z["lon"][keep].astype(np.float64) - T.c) / T.a).astype(np.int64)
    ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    day = (z["utc_minutes"][keep].astype(np.int64) // 1440)[ok]
    order = np.argsort(day, kind="stable")
    G.update(day=day[order], pix=(row * W + col)[ok][order], frp=z["frp"][keep][ok][order].astype(np.float64),
             night=(z["daynight"][keep] == "N")[ok][order])
    G["out"] = zarr.open_group(str(resolve(OUT)), mode="r+")


def one(g):
    i0, i1 = np.searchsorted(G["day"], g, "left"), np.searchsorted(G["day"], g, "right")
    pix = G["pix"][i0:i1]
    frp = np.bincount(pix, weights=G["frp"][i0:i1], minlength=H * W).reshape(H, W)
    n = np.minimum(np.bincount(pix, minlength=H * W), 255).reshape(H, W)
    nn = np.minimum(np.bincount(pix, weights=G["night"][i0:i1].astype(np.float64), minlength=H * W), 255).reshape(H, W)
    G["out"]["frp_sum"][g] = frp.astype(np.float32)
    G["out"]["n_det"][g] = n.astype(np.uint8)
    G["out"]["n_night"][g] = nn.astype(np.uint8)
    return g, i1 - i0


def main():
    root = zarr.open_group(str(resolve(OUT)), mode="w")
    comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    for name, dt in (("frp_sum", np.float32), ("n_det", np.uint8), ("n_night", np.uint8)):
        root.create_dataset(name, shape=(N_DAYS, H, W), chunks=(1, 512, 512), dtype=dt, fill_value=0, compressor=comp)
    root.attrs.update(start_date="2015-01-01", day="UTC acquisition date", source="FIRMS VIIRS S-NPP C2 archive",
                      filter="confidence n/h, type 0", grid="label raster geotransform")
    t0, tot = time.time(), 0
    with Pool(WORKERS, initializer=_init) as pool:
        for k, (g, n) in enumerate(pool.imap_unordered(one, range(N_DAYS)), 1):
            tot += n
            if k % 200 == 0:
                print(f"  {k}/{N_DAYS} days, {tot:,} detections, {time.time() - t0:.0f} s", flush=True)
    print(f"[firms] {tot:,} detections written to {OUT}")


if __name__ == "__main__":
    main()
