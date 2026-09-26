"""Pre-aggregate the slow predictors into 8-day bins, once, instead of per sample.

The problem this solves. The dual-branch slow window reads 144 days per sample.
Measured against the raw cube (chunks (7, 512, 512, 7)) that costs ~2.4s/sample
and pulls ~1079 MB to keep ~14 MB -- a 76x read amplification:

  temporal : 144 days read vs 18 bins kept    -> 8.0x
  spatial  : 512x512 chunk vs 256x256 patch   -> 4.0x
  channels : 7 channels in chunk vs 3 used    -> 2.3x

More dataloader workers cannot fix this: the reads are bandwidth-bound on NFS,
not latency-bound, so workers just queue behind the same pipe. Pre-aggregating
kills the temporal and channel terms and does the binning once rather than every
sample of every epoch.

Result: 273 bins x 3 channels = ~47 GB for 2015-2020, vs 875 GB for the full X.
That fits in the node's page cache (476 GB free), so after the first epoch the
slow branch reads from RAM instead of NFS.

Bins are on a FIXED GLOBAL GRID (day 0, 8, 16, ... of the continuous 2015-2020
axis), not relative to each target day. This is what makes them shareable across
targets -- target-relative bins would need a different alignment per sample and
could not be precomputed. Cost: the most recent slow bin can be up to 7 days
stale, which is irrelevant for variables peaking at lag 130-150.

Aggregation is per-variable: PPT is SUMMED (cumulative rainfall is the physical
quantity; striding would drop a heavy rain day), LAI/SM are MEANED.

Writes cube_slow_8day.zarr with:
  X_slow (n_bins, H, W, 3)  float32, channels [LAI, SM, PPT]
  bin_start_day (n_bins,)   int32, first day of each bin on the continuous axis
  year_of_bin  (n_bins,)    int32, source year (for provenance)

Usage:  SMOLDER_DATA=/path/to/cubes python -m smolder.data.build_slow_cube            # all years
        YEARS=2015,2016 SMOLDER_DATA=/path/to/cubes python -m smolder.data.build_slow_cube
"""
import os
from pathlib import Path

import numpy as np
import zarr

from smolder.data.io import data_dir

SCRIPT_DIR = data_dir()   # cubes are read from / written to $SMOLDER_DATA
OUT = SCRIPT_DIR / "cube_slow_8day.zarr"

# X channel order: [sm, wind, vpd, precip, lst_day, ndvi, lai]
CH = {"SM": 0, "WIND": 1, "VPD": 2, "PPT": 3, "LST": 4, "NDVI": 5, "LAI": 6}
# NDVI excluded: correlates 0.795 with LAI and adds nothing for the target
# (LAI alone 0.771 AUC, NDVI+LAI 0.770, all three 0.765).
SLOW_CHANNELS = ("LAI", "SM", "PPT")
SLOW_AGG = {"LAI": "mean", "SM": "mean", "PPT": "sum"}
BIN = 8  # native cadence of LAI (8-day composites)


def main():
    years = tuple(int(y) for y in os.environ.get("YEARS", "2015,2016,2017,2018,2019,2020").split(","))
    bin_days = int(os.environ.get("BIN", BIN))

    groups, offsets, off = [], [], 0
    for y in years:
        g = zarr.open_group(str(SCRIPT_DIR / f"cube_daily_smgrid_{y}.zarr"), mode="r")
        groups.append((y, g))
        offsets.append(off)
        off += g["X"].shape[0]
    T_total = off
    _, H, W, _ = groups[0][1]["X"].shape
    n_bins = T_total // bin_days

    ch_idx = [CH[c] for c in SLOW_CHANNELS]
    sum_mask = np.array([SLOW_AGG[c] == "sum" for c in SLOW_CHANNELS], dtype=bool)

    print(f"[info] {T_total} days across {years} -> {n_bins} bins of {bin_days} days")
    print(f"[info] channels {SLOW_CHANNELS} (agg: {[SLOW_AGG[c] for c in SLOW_CHANNELS]})")
    print(f"[info] output {OUT}  (~{n_bins*H*W*len(ch_idx)*4/1e9:.1f} GB)")

    root = zarr.open_group(str(OUT), mode="w")
    # chunk one bin x 512x512 x all-3-channels: a 256x256 patch then pulls one
    # chunk per bin instead of 21 fat time-chunks of the raw cube.
    z = root.create_dataset(
        "X_slow", shape=(n_bins, H, W, len(ch_idx)), chunks=(1, 512, 512, len(ch_idx)),
        dtype="f4", overwrite=True,
    )

    def locate(t):
        i = int(np.searchsorted(np.asarray(offsets), t, side="right") - 1)
        return i, t - offsets[i]

    bin_start = np.zeros(n_bins, dtype=np.int32)
    year_of = np.zeros(n_bins, dtype=np.int32)

    for b in range(n_bins):
        t0 = b * bin_days
        bin_start[b] = t0
        ci, _ = locate(t0)
        year_of[b] = groups[ci][0]

        # read the bin's days, crossing cubes if needed
        parts, t = [], t0
        t_end = t0 + bin_days
        while t < t_end:
            ci, lt = locate(t)
            g = groups[ci][1]
            cube_end = offsets[ci] + g["X"].shape[0]
            take = min(t_end, cube_end) - t
            arr = np.asarray(g["X"][lt:lt + take, :, :, :], dtype=np.float32)[..., ch_idx]
            parts.append(arr)
            t += take
        stack = np.concatenate(parts, axis=0)
        stack = np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)

        agg = np.where(sum_mask[None, None, :], stack.sum(axis=0), stack.mean(axis=0))
        z[b] = agg.astype(np.float32)
        if b % 10 == 0:
            print(f"[bin] {b}/{n_bins}  day {t0}  year {year_of[b]}", flush=True)

    root.create_dataset("bin_start_day", data=bin_start, overwrite=True)
    root.create_dataset("year_of_bin", data=year_of, overwrite=True)
    root.attrs["bin_days"] = bin_days
    root.attrs["channels"] = list(SLOW_CHANNELS)
    root.attrs["agg"] = [SLOW_AGG[c] for c in SLOW_CHANNELS]
    root.attrs["years"] = list(years)
    root.attrs["source_channel_idx"] = ch_idx
    print(f"[done] wrote {OUT}")


if __name__ == "__main__":
    main()
