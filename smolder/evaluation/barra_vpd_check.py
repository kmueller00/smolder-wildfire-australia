"""Check BARRA-C2 VPD against the ERA5-based product, and store input statistics.

1. Normalization statistics (mean, standard deviation) of sfcWindmax, uas,
   vas and vpd over model land pixels, from every 7th day of 2015-2018 only,
   written to the attrs of barra_c2_daily.zarr (stats_2015_2018).
2. VPD comparison on N_DAYS days drawn from 2015-2019 (no 2020): BARRA-C2
   VPD interpolated to the 1 km model grid against the cube's VPD channel
   (ERA5-based VPD at Tmax, Montes et al. 2021, resampled from about 31 km),
   on land: Pearson r, mean difference and RMSE at 1 km, and the same after
   averaging both to blocks of 31 x 31 px (about the ERA5 product's
   resolution). Also the share of within-block variance BARRA adds: the
   1 km standard deviation inside each 31 px block, for both products.

Output: barra_vpd_check.json (OUT).
"""
import datetime as dt
import json
import os

import numpy as np
import zarr
from scipy.ndimage import map_coordinates

from smolder.data.io import daily_cube, open_zarr_root, resolve

STORE = os.environ.get("BARRA_STORE", "barra_c2_daily.zarr")
OUT = os.environ.get("OUT", "barra_vpd_check.json")
N_DAYS = int(os.environ.get("N_DAYS", 24))
GT = (112.904998779, 0.009997566018978103, -9.005000113999998, -0.009997121616580312)
BLOCK = 31
VARS = ("sfcWindmax", "uas", "vas", "vpd")


def grid_coords(lat, lon, H, W):
    c0, a, f0, e = GT
    la = f0 + (np.arange(H) + 0.5) * e
    lo = c0 + (np.arange(W) + 0.5) * a
    fi = (la - lat[0]) / (lat[1] - lat[0])
    fj = (lo - lon[0]) / (lon[1] - lon[0])
    return np.meshgrid(fi, fj, indexing="ij")


def blocks(a, land):
    H, W = a.shape
    h, w = H // BLOCK * BLOCK, W // BLOCK * BLOCK
    x = np.where(land, a, np.nan)[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK)
    with np.errstate(all="ignore"):
        return np.nanmean(x, (1, 3)), np.nanstd(x, (1, 3)), np.isfinite(x).mean((1, 3))


def main():
    root = zarr.open_group(str(resolve(STORE)), mode="r+")
    lat, lon = np.asarray(root["lat"]), np.asarray(root["lon"])
    land = np.asarray(open_zarr_root(daily_cube(2019))["landmask"][...]) > 0
    H, W = land.shape
    FI, FJ = grid_coords(lat, lon, H, W)
    # native cells that some model land pixel falls into, for the statistics
    cells = np.unique(np.rint(FI[land]).astype(int) * lon.size + np.rint(FJ[land]).astype(int))
    stats = {}
    for v in VARS:
        vals = []
        for g in range(0, 1461, 7):                                  # 2015-2018 only
            x = np.asarray(root[v][g], np.float32).ravel()[cells]
            vals.append(x[np.isfinite(x)])
        x = np.concatenate(vals)
        stats[v] = [float(x.mean()), float(x.std())]
    root.attrs["stats_2015_2018"] = stats
    print("[stats]", stats, flush=True)

    rng = np.random.default_rng(0)
    gdays = np.sort(rng.choice(np.arange(13, 1826), N_DAYS, replace=False))  # 2015-2019
    res = []
    for g in gdays:
        d = dt.date(2015, 1, 1) + dt.timedelta(days=int(g))
        cube = open_zarr_root(daily_cube(d.year))
        names = cube.attrs.get("channels")
        ich = list(names).index("vpd") if names else 2
        era = np.asarray(cube["X"][d.timetuple().tm_yday - 1, :, :, ich], np.float32)
        bar = map_coordinates(np.asarray(root["vpd"][int(g)], np.float32), [FI, FJ], order=1, mode="nearest")
        m = land & np.isfinite(era) & np.isfinite(bar)
        r1 = float(np.corrcoef(era[m], bar[m])[0, 1])
        bm_e, bs_e, fr = blocks(era, m)
        bm_b, bs_b, _ = blocks(bar, m)
        k = fr > 0.5
        res.append(dict(date=d.isoformat(), r_1km=r1, mean_era=float(era[m].mean()), mean_barra=float(bar[m].mean()),
                        bias_barra_minus_era=float((bar[m] - era[m]).mean()),
                        rmse=float(np.sqrt(((bar[m] - era[m]) ** 2).mean())),
                        r_31px=float(np.corrcoef(bm_e[k], bm_b[k])[0, 1]),
                        within_block_sd_era=float(np.nanmean(bs_e[k])), within_block_sd_barra=float(np.nanmean(bs_b[k]))))
        print(res[-1], flush=True)
    keys = [k for k in res[0] if k != "date"]
    out = dict(store=STORE, stats_2015_2018=stats, n_days=len(res), days=res,
               mean={k: float(np.nanmean([r[k] for r in res])) for k in keys},
               n_days_valid=int(np.sum([np.isfinite(r["r_1km"]) for r in res])))
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print("[mean]", out["mean"])


if __name__ == "__main__":
    main()
