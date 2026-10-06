"""Training-input stores laid out for fast patch reads (FAST_STORES=1).

The original stores keep one day per chunk file, so a 384 px training sample
opened about 560 files (~6.5 ms each on NFS) and read 433 MB to keep about
70 MB. These copies hold the same values in blocks of 16 days (32 for BARRA),
cutting a sample to about 50 file reads. Nothing is recomputed except the FRP
inputs, which are stored as the per-day features the model uses; the
datamodule reads them instead of summing three days per step.

  fire_inputs_continental.zarr   global day axis 2015-01-01 .. 2020-12-31
    y_fire_3d (2192, H, W) uint8, chunks (16, 256, 256)
        the daily cubes' y_fire_3d, all years on one axis
    age_px    (2192, H, W) uint16, chunks (16, 256, 256)
        fire_age_continental.zarr, rechunked
    frp_feat  (2192, H, W, 3) float32, chunks (16, 256, 256, 3)
        features over days d-2..d of firms_daily.zarr, exactly as the
        datamodule computed them: log1p(sum FRP)/5, log1p(sum detections)/3,
        night share (0 without detections). Days 0-1 use the days available.
  barra_c2_fast.zarr
    x (2223, lat, lon, 4) float16 [vpd, uas, vas, tasmax], chunks (32, 128, 128, 4)
        barra_c2_daily.zarr, the fast-branch variables in one array
  cube_slow_8day_lai500m_lag31.zarr (SLOW_STORE)
    X_slow (274, H, W, 3) float32 [LAI, SM, PPT], chunks (16, 256, 256, 3)
        cube_slow_8day.zarr with LAI replaced by LAI_STORE
        (cube_slow_8day_lai500_lag31.zarr: HiQ-LAI 500 m, 31-day lag)

Every block is written whole and then read back and compared with its source
(VERIFY=1, the default).

Usage
  SMOLDER_DATA=... STEP=fire|barra|slow|all WORKERS=8 python -m smolder.data.build_fast_stores
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
TB = 16                                  # days (bins) per chunk
WORKERS = int(os.environ.get("WORKERS", 8))
VERIFY = os.environ.get("VERIFY", "1") == "1"
INT_COMP = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
FLT_COMP = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)

FIRE_STORE = "fire_inputs_continental.zarr"
BARRA_STORE = os.environ.get("BARRA_STORE", "barra_c2_fast_tmax.zarr")
SLOW_STORE = os.environ.get("SLOW_STORE", "cube_slow_8day_lai500m_lag31.zarr")
LAI_STORE = os.environ.get("LAI_STORE", "cube_slow_8day_lai500_lag31.zarr")
BARRA_VARS = ["vpd", "uas", "vas", "tasmax"]


def frp_features(frp, nd, nn):
    """(n, H, W) float32 daily sums of the 3-day windows ending on each day of
    frp[2:], as zarr_dual_datamodule computed them per fast step."""
    out = np.empty((frp.shape[0] - 2,) + frp.shape[1:] + (3,), np.float32)
    for j in range(out.shape[0]):
        w = slice(j, j + 3)
        f_, n_, m_ = frp[w].sum(0), nd[w].sum(0), nn[w].sum(0)
        out[j, ..., 0] = np.log1p(f_) / 5.0
        out[j, ..., 1] = np.log1p(n_) / 3.0
        out[j, ..., 2] = np.where(n_ > 0, m_ / np.maximum(n_, 1), 0.0)
    return out


def run_blocks(name, n, fn):
    t0 = time.time()
    starts = list(range(0, n, TB))
    with ThreadPoolExecutor(WORKERS) as ex:
        for k, _ in enumerate(ex.map(fn, starts)):
            if k % 10 == 0:
                print(f"  [{name}] block {k + 1}/{len(starts)} ({time.time() - t0:.0f} s)", flush=True)
    print(f"[{name}] done ({time.time() - t0:.0f} s)", flush=True)


def build_fire():
    ys = {y: open_zarr_root(daily_cube(y))["y_fire_3d"] for y in YEARS}
    H, W = ys[2015].shape[1:]
    for y in YEARS:
        assert ys[y].shape[0] == (OFFSETS[y + 1] if y < 2020 else N_DAYS) - OFFSETS[y], y
    age = open_zarr_root("fire_age_continental.zarr")["age_px"]
    firms = open_zarr_root("firms_daily.zarr")
    fs, fd, fn_ = firms["frp_sum"], firms["n_det"], firms["n_night"]
    assert age.shape == (N_DAYS, H, W) and fs.shape == (N_DAYS, H, W), (age.shape, fs.shape)

    root = zarr.open_group(str(resolve(FIRE_STORE)), mode="w")
    zy = root.create_dataset("y_fire_3d", shape=(N_DAYS, H, W), chunks=(TB, 256, 256), dtype=np.uint8,
                             fill_value=int(ys[2015].fill_value or 0), compressor=INT_COMP)
    za = root.create_dataset("age_px", shape=(N_DAYS, H, W), chunks=(TB, 256, 256), dtype=np.uint16,
                             fill_value=age.fill_value, compressor=INT_COMP)
    zf = root.create_dataset("frp_feat", shape=(N_DAYS, H, W, 3), chunks=(TB, 256, 256, 3), dtype=np.float32,
                             fill_value=0.0, compressor=FLT_COMP)
    root.attrs.update(start_date="2015-01-01", n_days=N_DAYS, block_days=TB,
                      frp_feat=["log1p(sum frp_sum d-2..d)/5", "log1p(sum n_det)/3", "sum n_night / sum n_det"],
                      sources=["cube_daily_smgrid_YYYY.zarr:y_fire_3d", "fire_age_continental.zarr:age_px",
                               "firms_daily.zarr"])

    def y_days(g0, g1):
        parts = []
        g = g0
        while g < g1:
            y = max(v for v in YEARS if OFFSETS[v] <= g)
            end = min(g1, OFFSETS[y] + ys[y].shape[0])
            parts.append(np.asarray(ys[y][g - OFFSETS[y]:end - OFFSETS[y]]))
            g = end
        return np.concatenate(parts)

    def block(b0):
        b1 = min(b0 + TB, N_DAYS)
        yb = y_days(b0, b1)
        zy[b0:b1] = yb
        ab = np.asarray(age[b0:b1])
        za[b0:b1] = ab
        lo = max(b0 - 2, 0)
        frp = np.asarray(fs[lo:b1], np.float32)
        nd = np.asarray(fd[lo:b1], np.float32)
        nn = np.asarray(fn_[lo:b1], np.float32)
        pad = 2 - (b0 - lo)                     # days 0-1: windows with the days available
        if pad:
            z = np.zeros((pad,) + frp.shape[1:], np.float32)
            frp, nd, nn = (np.concatenate([z, a]) for a in (frp, nd, nn))
        fb = frp_features(frp, nd, nn)
        zf[b0:b1] = fb
        if VERIFY:
            assert np.array_equal(np.asarray(zy[b0:b1]), yb), ("y", b0)
            assert np.array_equal(np.asarray(za[b0:b1]), ab), ("age", b0)
            assert np.array_equal(np.asarray(zf[b0:b1]), fb, equal_nan=True), ("frp", b0)

    run_blocks("fire", N_DAYS, block)


def build_barra():
    src = open_zarr_root("barra_c2_daily.zarr")
    n, nl, nk = src["vpd"].shape
    root = zarr.open_group(str(resolve(BARRA_STORE)), mode="w")
    z = root.create_dataset("x", shape=(n, nl, nk, len(BARRA_VARS)), chunks=(32, 128, 128, len(BARRA_VARS)), dtype=np.float16,
                            fill_value=np.nan, compressor=INT_COMP)
    root["lat"] = np.asarray(src["lat"][...])
    root["lon"] = np.asarray(src["lon"][...])
    root.attrs.update({k: v for k, v in src.attrs.items() if k != "vpd_climatology"})
    root.attrs.update(variables=BARRA_VARS, source_store="barra_c2_daily.zarr")

    def block(b0):
        b1 = min(b0 + 32, n)
        x = np.stack([np.asarray(src[v][b0:b1]) for v in BARRA_VARS], axis=-1)
        assert x.dtype == np.float16, x.dtype
        z[b0:b1] = x
        if VERIFY:
            assert np.array_equal(np.asarray(z[b0:b1]), x, equal_nan=True), ("barra", b0)

    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(block, range(0, n, 32)))
    print(f"[barra] done ({time.time() - t0:.0f} s)", flush=True)


def build_slow():
    sc = open_zarr_root("cube_slow_8day.zarr")
    lg = open_zarr_root(LAI_STORE)
    starts = np.asarray(sc["bin_start_day"][...])
    assert np.array_equal(np.asarray(lg["bin_start_day"][...]), starts)
    assert list(sc.attrs["channels"]) == ["LAI", "SM", "PPT"], sc.attrs["channels"]
    src, lai = sc["X_slow"], lg["X_slow"]
    n, H, W, _ = src.shape
    root = zarr.open_group(str(resolve(SLOW_STORE)), mode="w")
    z = root.create_dataset("X_slow", shape=(n, H, W, 3), chunks=(TB, 256, 256, 3), dtype=np.float32,
                            fill_value=0.0, compressor=FLT_COMP)
    root["bin_start_day"] = starts
    root["year_of_bin"] = np.asarray(sc["year_of_bin"][...])
    root.attrs.update(dict(sc.attrs))
    root.attrs.update(lai_source=f"{LAI_STORE} (HiQ-LAI 500 m, mean to 1 km)", sm_ppt_source="cube_slow_8day.zarr",
                      lag_days=int(lg.attrs.get("lag_days", 0)), first_full_bin=int(lg.attrs.get("first_full_bin", 0)))

    def block(b0):
        b1 = min(b0 + TB, n)
        x = np.asarray(src[b0:b1], np.float32)
        x[..., 0] = np.asarray(lai[b0:b1, ..., 0], np.float32)
        z[b0:b1] = x
        if VERIFY:
            assert np.array_equal(np.asarray(z[b0:b1]), x, equal_nan=True), ("slow", b0)

    run_blocks("slow", n, block)


def main():
    step = os.environ.get("STEP", "all")
    if step in ("barra", "all"):
        build_barra()
    if step in ("slow", "all"):
        build_slow()
    if step in ("fire", "all"):
        build_fire()


if __name__ == "__main__":
    main()
