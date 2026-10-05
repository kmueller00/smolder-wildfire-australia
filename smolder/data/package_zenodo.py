"""Zenodo archives for the SMOLDER model (fast stores, NDVI, 500 m LAI, FRP, fuel age,
BARRA-C2), written next to the existing archives without changing them.

STEP=ndvi    ndvi_composites.zarr: NDVI of the daily cubes from global day
             NDVI_FROM to the end of 2020, one plane per distinct 8-day
             composite, and the composite of every day. The archived 2020
             cube has no NDVI channel; with this store it does not need one.
STEP=barra   barra_c2_fast.zarr cut to the days a 2020 evaluation reads
             (global day >= BARRA_FROM): the chunk files of those days are
             copied byte for byte; attribute days_present records the range.
STEP=check   2020 samples of the model's inputs built from the archives alone
             (the archived 2020 cube without NDVI, plus the stores above)
             (a directory holding only them, with the fast stores) equal those
             built from the full data with the original stores. Inputs only;
             no model is run.
STEP=tar     one tar per store (the slow cube as on disk, i.e. with the
             repaired 2016 bins) and SHA256SUMS.
STEP=all     ndvi, barra, check and tar.

Usage
  SMOLDER_DATA=... OUT_DIR=.../zenodo_new STEP=all python -m smolder.data.package_zenodo
"""
import hashlib
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import zarr

from numcodecs import Blosc

from smolder.data.io import resolve

OUT_DIR = os.environ["OUT_DIR"]
YEAR = int(os.environ.get("YEAR", 2020))
WORKERS = int(os.environ.get("WORKERS", 8))
BARRA_FROM = int(os.environ.get("BARRA_FROM", 1824))        # 2019-12-31; must be a multiple of the chunk length
NDVI_FROM = int(os.environ.get("NDVI_FROM", 1824))
CHANNELS = ["sm", "wind", "vpd", "precip", "lst_day", "ndvi", "lai"]
TARS = {"cube_slow_8day.tar": ["@cube_slow_8day.zarr"],
        "cube_slow_8day_lai500m.tar": ["@cube_slow_8day_lai500m.zarr"],
        "fire_inputs_continental.tar": ["@fire_inputs_continental.zarr"],
        "barra_c2_fast_2020.tar": ["barra_c2_fast.zarr"],     # "@": taken from SMOLDER_DATA
        "ndvi_composites_2020.tar": ["ndvi_composites.zarr"]}


def build_ndvi():
    """NDVI of the daily cubes for global days NDVI_FROM..end of 2020, stored once
    per distinct 8-day composite (each day of the cube repeats the newest
    composite) plus day_to_comp, the composite each global day shows (-1 = not
    stored). Every day is assigned by exact comparison with the previous day."""
    T_END = 2192
    out = zarr.open_group(os.path.join(OUT_DIR, "ndvi_composites.zarr"), mode="w")
    H, W = 3474, 4110
    nd = out.create_dataset("ndvi", shape=(0, H, W), chunks=(1, 512, 512), dtype=np.float32,
                            fill_value=np.nan, compressor=Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE))
    d2c = np.full(T_END, -1, np.int32)
    offsets = {2019: 1461, 2020: 1826}
    prev, n, t0 = None, 0, time.time()
    for year, off in offsets.items():
        X = zarr.open_group(str(resolve(f"cube_daily_smgrid_{year}.zarr")), mode="r")["X"]
        assert X.shape[-1] == len(CHANNELS)
        lo = max(NDVI_FROM - off, 0)
        for b0 in range(lo, X.shape[0], X.chunks[0]):
            blk = np.asarray(X[b0:min(b0 + X.chunks[0], X.shape[0]), :, :, CHANNELS.index("ndvi")], np.float32)
            for i, plane in enumerate(blk):
                g = off + b0 + i
                if g < NDVI_FROM:
                    continue
                if prev is None or not np.array_equal(plane, prev, equal_nan=True):
                    nd.append(plane[None], axis=0)
                    prev, n = plane, n + 1
                d2c[g] = n - 1
        print(f"  [ndvi] {year}: {n} composites so far ({time.time() - t0:.0f} s)", flush=True)
    out["day_to_comp"] = d2c
    out.attrs.update(days_present=[NDVI_FROM, T_END], source="MODIS MOD09A1 NDVI of the daily cubes (channel ndvi)",
                     note="ndvi[day_to_comp[g]] is the NDVI of global day g (days since 2015-01-01); "
                          "-1 = day not stored. Missing = NaN.")
    print(f"[ndvi] {n} composites for days {NDVI_FROM}..{T_END - 1} ({time.time() - t0:.0f} s)", flush=True)


def build_barra():
    src_path = str(resolve("barra_c2_fast.zarr"))
    src = zarr.open_group(src_path, mode="r")
    x = src["x"]
    tc = x.chunks[0]
    assert BARRA_FROM % tc == 0, (BARRA_FROM, tc)
    dst = os.path.join(OUT_DIR, "barra_c2_fast.zarr")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(os.path.join(dst, "x"))
    for f in (".zgroup", ".zattrs"):
        shutil.copy2(os.path.join(src_path, f), dst)
    for name in ("lat", "lon"):
        shutil.copytree(os.path.join(src_path, name), os.path.join(dst, name))
    shutil.copy2(os.path.join(src_path, "x", ".zarray"), os.path.join(dst, "x"))
    first = BARRA_FROM // tc
    n = 0
    for f in os.listdir(os.path.join(src_path, "x")):
        if f.startswith(".") or int(f.split(".")[0]) < first:
            continue
        shutil.copy2(os.path.join(src_path, "x", f), os.path.join(dst, "x", f))
        n += 1
    g = zarr.open_group(dst, mode="r+")
    g.attrs.update(days_present=[BARRA_FROM, int(x.shape[0])],
                   note=f"Cut to global days {BARRA_FROM}..{x.shape[0] - 1} (from 2015-01-01), enough for a "
                        "2020 evaluation; other days read as missing (NaN).")
    a = zarr.open_group(dst, mode="r")["x"]
    for g0 in (BARRA_FROM, x.shape[0] - 5):
        assert np.array_equal(np.asarray(a[g0:g0 + 5]), np.asarray(x[g0:g0 + 5]), equal_nan=True)
    print(f"[barra] {n} chunk files, days {BARRA_FROM}..{x.shape[0] - 1}", flush=True)


def check(n=6):
    import torch
    from smolder.data.zarr_dual_datamodule import DualPatchConfig, DualWindowDataset
    full_dir = os.environ["SMOLDER_DATA"]
    arch = os.path.join(OUT_DIR, "check_data")
    os.makedirs(arch, exist_ok=True)
    links = {"barra_c2_fast.zarr": os.path.join(OUT_DIR, "barra_c2_fast.zarr"),
             "ndvi_composites.zarr": os.path.join(OUT_DIR, "ndvi_composites.zarr")}
    for m in (f"cube_{YEAR}_zenodo.zarr", "cube_slow_8day.zarr", "cube_slow_8day_lai500m.zarr",
              "fire_inputs_continental.zarr"):
        links[m] = os.path.join(full_dir, m)
    for name, target in links.items():
        p = os.path.join(arch, name)
        if not os.path.islink(p):
            os.symlink(target, p)
    os.chdir(arch)
    kw = dict(stats_path="channel_stats_2015_2018.json", slow_cube_path="cube_slow_8day.zarr",
              day_offset=1826, patch_size=384, samples_per_epoch=1, seed=0, deterministic=True,
              fire_history=True, fire_history_lags=(3, 4, 5), fire_history_distance=True,
              use_elevation=True, use_slope_aspect=True, use_fuel_age=True, fuel_age_lookback=1095,
              vpd_source="barra", use_frp=True, use_barra_uv=True, slow_veg="lai500", use_fast_ndvi=True,
              perfect_forecast=False, use_vpd_anomaly=False)
    ref = DualWindowDataset(DualPatchConfig(zarr_paths=(os.path.join(full_dir, f"cube_daily_smgrid_{YEAR}.zarr"),),
                                            fast_stores=False, **kw))
    os.environ["SMOLDER_DATA"] = arch
    new = DualWindowDataset(DualPatchConfig(zarr_paths=(f"cube_{YEAR}_zenodo.zarr",), fast_stores=True, **kw))
    os.environ["SMOLDER_DATA"] = full_dir
    rng = np.random.default_rng(0)
    ts = [int(ref.targets[0]), int(ref.targets[-1])] + [int(t) for t in rng.choice(ref.targets, n - 2)]
    for t in ts:
        y0, x0 = int(rng.integers(0, ref.H - 384)), int(rng.integers(0, ref.W - 384))
        a, b = ref.sample_at(t, y0, x0), new.sample_at(t, y0, x0)
        assert set(a) == set(b)
        for k in a:
            assert torch.equal(a[k], b[k]), (k, t, y0, x0, float((a[k].float() - b[k].float()).abs().max()))
        print(f"  [check] t_end {t} ({y0}, {x0}): all {len(a)} tensors identical", flush=True)
    print(f"[check] {len(ts)} samples from the archives alone equal the full-data samples", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def build_tars():
    sums = []
    for tar, members in TARS.items():
        dst = os.path.join(OUT_DIR, tar)
        for m in members:
            base = str(resolve(m[1:])) if m.startswith("@") else os.path.join(OUT_DIR, m)
            t0 = time.time()
            subprocess.run(["tar", "-cf", dst, "-C", os.path.dirname(base), os.path.basename(base)], check=True)
            print(f"[tar] {tar} {os.path.getsize(dst) / 1e9:.1f} GB ({time.time() - t0:.0f} s)", flush=True)
        sums.append(f"{sha256(dst)}  {tar}")
    open(os.path.join(OUT_DIR, "SHA256SUMS"), "w").write("\n".join(sums) + "\n")
    print("\n".join(sums))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    step = os.environ.get("STEP", "all")
    if step in ("ndvi", "all"):
        build_ndvi()
    if step in ("barra", "all"):
        build_barra()
    if step in ("check", "all"):
        check()
    if step in ("tar", "all"):
        build_tars()


if __name__ == "__main__":
    main()
