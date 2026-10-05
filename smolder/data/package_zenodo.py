"""Zenodo archives for the v2 model (fast stores, NDVI, 500 m LAI, FRP, fuel age,
BARRA-C2), written next to the existing archives without changing them.

STEP=cube    cube_2020_zenodo.zarr with NDVI: every array except X is copied
             unchanged from the archived cube; X gets NDVI back at its original
             position ([sm, wind, vpd, precip, lst_day, ndvi, lai], missing = 0
             as in the archive). Each block is checked: the six archived
             channels must equal the archived X exactly.
STEP=barra   barra_c2_fast.zarr cut to the days a 2020 evaluation reads
             (global day >= BARRA_FROM): the chunk files of those days are
             copied byte for byte; attribute days_present records the range.
STEP=tar     one tar per store (the slow cube as on disk, i.e. with the
             repaired 2016 bins) and SHA256SUMS.
STEP=all     all three.

Usage
  SMOLDER_DATA=... OUT_DIR=.../zenodo_v2 STEP=all python -m smolder.data.package_zenodo
"""
import hashlib
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import zarr

from smolder.data.io import resolve

OUT_DIR = os.environ["OUT_DIR"]
YEAR = int(os.environ.get("YEAR", 2020))
WORKERS = int(os.environ.get("WORKERS", 8))
BARRA_FROM = int(os.environ.get("BARRA_FROM", 1824))        # 2019-12-31; must be a multiple of the chunk length
CHANNELS = ["sm", "wind", "vpd", "precip", "lst_day", "ndvi", "lai"]
TARS = {"cube_2020_zenodo.tar": ["cube_2020_zenodo.zarr"],
        "cube_slow_8day.tar": ["@cube_slow_8day.zarr"],
        "cube_slow_8day_lai500m.tar": ["@cube_slow_8day_lai500m.zarr"],
        "fire_inputs_continental.tar": ["@fire_inputs_continental.zarr"],
        "barra_c2_fast_2020.tar": ["barra_c2_fast.zarr"]}     # "@": taken from SMOLDER_DATA


def build_cube():
    src = zarr.open_group(str(resolve(f"cube_daily_smgrid_{YEAR}.zarr")), mode="r")
    old = zarr.open_group(str(resolve(f"cube_{YEAR}_zenodo.zarr")), mode="r")
    assert list(src.attrs["dyn_vars"]) == CHANNELS, src.attrs["dyn_vars"]
    keep = [CHANNELS.index(c) for c in old.attrs["channels"]]
    out = zarr.open_group(os.path.join(OUT_DIR, f"cube_{YEAR}_zenodo.zarr"), mode="w")
    for name, a in old.arrays():
        if name != "X":
            zarr.copy(a, out, name=name)
    attrs = dict(old.attrs)
    attrs.update(channels=CHANNELS, note="Channels [sm, wind, vpd, precip, lst_day, ndvi, lai] in native units, "
                                         "missing = 0. NDVI (channel 5) added back in v2 of this archive: the v2 "
                                         "model reads it on the fast branch.")
    out.attrs.update(attrs)
    ox = old["X"]
    T, H, W, _ = ox.shape
    X = out.create_dataset("X", shape=(T, H, W, len(CHANNELS)), chunks=ox.chunks[:3] + (len(CHANNELS),),
                           dtype=np.float32, fill_value=0.0, compressor=ox.compressor)
    tb = ox.chunks[0]
    t0 = time.time()

    def block(b0):
        b1 = min(b0 + tb, T)
        x = np.nan_to_num(np.asarray(src["X"][b0:b1], np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        ref = np.asarray(ox[b0:b1])
        if not np.array_equal(x[..., keep], ref):
            bad = int((x[..., keep] != ref).sum())
            raise AssertionError(f"days {b0}-{b1}: {bad} values differ from the archived X")
        X[b0:b1] = x
        assert np.array_equal(np.asarray(X[b0:b1]), x)

    with ThreadPoolExecutor(WORKERS) as ex:
        for i, _ in enumerate(ex.map(block, range(0, T, tb))):
            if i % 10 == 0:
                print(f"  [cube] block {i + 1}/{-(-T // tb)} ({time.time() - t0:.0f} s)", flush=True)
    print(f"[cube] done, six archived channels identical ({time.time() - t0:.0f} s)", flush=True)


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
    if step in ("cube", "all"):
        build_cube()
    if step in ("barra", "all"):
        build_barra()
    if step in ("tar", "all"):
        build_tars()


if __name__ == "__main__":
    main()
