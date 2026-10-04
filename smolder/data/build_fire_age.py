"""Days since each pixel last burned, for every day 2015-2020.

age_px[t] = t - (last day index u <= t with y_fire_3d[u] > 0 at the pixel),
i.e. the time since the pixel's newest fire-history map that showed fire.
Read at t = D - 3 (as the fire-history channels), it uses no detection after
the issue day D. uint16, capped at 65534; 65535 = no fire since 2015-01-01
(history starts there, so ages are censored at the elapsed time).

Output fire_age_continental.zarr: age_px (2192, H, W), chunks (1, 512, 512).

Usage
  SMOLDER_DATA=... python -m smolder.data.build_fire_age
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import zarr
from numcodecs import Blosc

from smolder.data.io import daily_cube, open_zarr_root, resolve

YEARS = [2015, 2016, 2017, 2018, 2019, 2020]
OFFSETS = {2015: 0, 2016: 365, 2017: 731, 2018: 1096, 2019: 1461, 2020: 1826}
N_DAYS = 2192
NODATA = 65535
OUT = os.environ.get("OUT", "fire_age_continental.zarr")


def main():
    ys = {y: open_zarr_root(daily_cube(y))["y_fire_3d"] for y in YEARS}
    land = np.asarray(open_zarr_root(daily_cube(2020))["landmask"][...]) > 0
    H, W = land.shape
    root = zarr.open_group(str(resolve(OUT)), mode="w")
    z = root.create_dataset("age_px", shape=(N_DAYS, H, W), chunks=(1, 512, 512), dtype=np.uint16,
                            fill_value=NODATA, compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE))
    root.attrs.update(start_date="2015-01-01", nodata=NODATA,
                      description="days since the newest y_fire_3d map with fire at the pixel (land only)")
    last = np.full((H, W), -1, np.int32)
    t0 = time.time()

    def read(g):
        y = max(v for v in YEARS if OFFSETS[v] <= g)
        return (np.asarray(ys[y][g - OFFSETS[y]]) > 0) & land

    with ThreadPoolExecutor(8) as ex:
        nxt = ex.submit(read, 0)
        writes = []
        for g in range(N_DAYS):
            fire = nxt.result()
            if g + 1 < N_DAYS:
                nxt = ex.submit(read, g + 1)
            last[fire] = g
            age = np.where(last >= 0, np.minimum(g - last, NODATA - 1), NODATA).astype(np.uint16)
            writes.append(ex.submit(z.__setitem__, g, age))
            if g % 200 == 0:
                print(f"  day {g}/{N_DAYS} ({time.time() - t0:.0f} s)", flush=True)
        for w in writes:
            w.result()
    print(f"[fire_age] wrote {OUT} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
