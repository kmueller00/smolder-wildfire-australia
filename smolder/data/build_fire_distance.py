"""Continent-wide distance to recent fire, for every day 2015-2020.

For each day t of the continuous 2015-2020 axis (global index = year offset
+ day of year, as in cube_slow_8day.zarr), stores the Euclidean distance in
pixels from every pixel to the nearest land pixel with fire in the union of
y_fire_3d[t-WINDOW+1 .. t]. WINDOW=1 (default) is the newest fire-history map
(fire on days t+1..t+3); WINDOW=30 covers fire on days t-28..t+3, i.e. read
at t = D-3 it is the fire of the 32 days D-31..D, the "last 32 days" of
newfire_definition_sweep.py (W=30). The datamodule and the diagnostics read
index D-3 (or s-L with L >= 3), so no detection later than the issue day is
ever used. At the start of 2015 the window is cut to the days available.

The distance runs over the whole grid (not cut at tile or patch edges);
fire over the ocean is ignored, as in the fire-history channels. uint16,
clipped at 65534; 65535 marks a day without any fire on the continent.

Output: fire_dist_continental.zarr (WINDOW=1) or fire_dist<W>_continental.zarr with
  dist_px   (2192, H, W) uint16, chunks (1, 512, 512)
  attrs     start_date, years, day_offsets, nodata, window

Usage
  SMOLDER_DATA=/path/to/cubes python -m smolder.data.build_fire_distance
"""
import os
import time
from multiprocessing import Pool

import numpy as np
import zarr
from numcodecs import Blosc
from scipy import ndimage

from smolder.data.io import daily_cube, open_zarr_root, resolve

YEARS = [2015, 2016, 2017, 2018, 2019, 2020]
OFFSETS = {2015: 0, 2016: 365, 2017: 731, 2018: 1096, 2019: 1461, 2020: 1826}
N_DAYS = 2192
NODATA = 65535
WINDOW = int(os.environ.get("WINDOW", 1))
OUT = os.environ.get("OUT", "fire_dist_continental.zarr" if WINDOW == 1 else f"fire_dist{WINDOW}_continental.zarr")
WORKERS = int(os.environ.get("WORKERS", 12))

G = {}


def _init():
    G["y"] = {y: open_zarr_root(daily_cube(y))["y_fire_3d"] for y in YEARS}
    G["land"] = np.asarray(open_zarr_root(daily_cube(2020))["landmask"][...]) > 0
    G["out"] = zarr.open_group(str(resolve(OUT)), mode="r+")["dist_px"]


def _fire(gday):
    year = max(y for y in YEARS if OFFSETS[y] <= gday)
    return (np.asarray(G["y"][year][gday - OFFSETS[year]]) > 0) & G["land"]


def one(block):
    """Days block[0]..block[1]-1 in order, with a rolling count of fire over
    the last WINDOW maps so each map is read about once."""
    lo, hi = block
    count = np.zeros(G["land"].shape, np.uint8)
    for t in range(max(0, lo - WINDOW + 1), lo):
        count += _fire(t)
    for gday in range(lo, hi):
        count += _fire(gday)
        fire = count > 0
        if fire.any():
            d = ndimage.distance_transform_edt(~fire)
            G["out"][gday] = np.minimum(np.rint(d), NODATA - 1).astype(np.uint16)
        else:
            G["out"][gday] = np.full(fire.shape, NODATA, np.uint16)
        if gday - WINDOW + 1 >= 0:
            count -= _fire(gday - WINDOW + 1)
    return block


def main():
    land = np.asarray(open_zarr_root(daily_cube(2020))["landmask"][...])
    H, W = land.shape
    path = str(resolve(OUT))
    root = zarr.open_group(path, mode="a")
    if "dist_px" not in root:
        root.create_dataset("dist_px", shape=(N_DAYS, H, W), chunks=(1, 512, 512), dtype=np.uint16,
                            fill_value=0, compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE))
        root.attrs.update(start_date="2015-01-01", years=YEARS, day_offsets=OFFSETS, nodata=NODATA,
                          window=WINDOW,
                          description="Euclidean distance (px, ~1 km) to the nearest land pixel with "
                                      f"y_fire_3d > 0 in any of the {WINDOW} slices ending at t, whole grid")
    assert int(root.attrs.get("window", 1)) == WINDOW, "store was built with another WINDOW"
    done = set(root.attrs.get("done", []))
    todo = [d for d in range(N_DAYS) if d not in done]
    print(f"[fire_dist] window {WINDOW}: {len(todo)} of {N_DAYS} days to build -> {path}", flush=True)
    step = max(1, -(-len(todo) // (4 * WORKERS)))
    blocks = []                                   # contiguous runs of todo days, split into chunks
    for d in todo:
        if blocks and d == blocks[-1][1] and blocks[-1][1] - blocks[-1][0] < step:
            blocks[-1][1] += 1
        else:
            blocks.append([d, d + 1])
    t0 = time.time()
    with Pool(WORKERS, initializer=_init) as pool:
        for i, (lo, hi) in enumerate(pool.imap_unordered(one, [tuple(b) for b in blocks]), 1):
            done.update(range(lo, hi))
            root.attrs["done"] = sorted(done)
            print(f"  {len(done)}/{N_DAYS} days, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
