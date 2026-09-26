"""Compute per-channel mean/std of the dynamic X channels (+ AGB) over the
training years (2015-2018) by random patch sampling.
Writes channel_stats_2015_2018.json.
"""
import json
import numpy as np
from pathlib import Path
import zarr

from smolder.data.io import data_dir

SCRIPT_DIR = data_dir()   # cubes are read from / written to $SMOLDER_DATA
YEARS = [2015, 2016, 2017, 2018]
PATCH = 256
PATCHES_PER_YEAR = 100
SEED = 7


def main():
    rng = np.random.default_rng(SEED)
    s = s2 = None
    n = 0
    agb_vals = []

    for y in YEARS:
        g = zarr.open_group(str(SCRIPT_DIR / f"cube_daily_smgrid_{y}.zarr"), mode="r")
        X = g["X"]  # (T, H, W, C)
        T, H, W, C = X.shape
        if s is None:
            s = np.zeros(C, dtype=np.float64)
            s2 = np.zeros(C, dtype=np.float64)
        agb = g["agb"][...].astype(np.float64)

        for i in range(PATCHES_PER_YEAR):
            t = int(rng.integers(0, T))
            y0 = int(rng.integers(0, H - PATCH + 1))
            x0 = int(rng.integers(0, W - PATCH + 1))
            p = X[t, y0:y0 + PATCH, x0:x0 + PATCH, :].astype(np.float64)
            p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
            flat = p.reshape(-1, C)
            s += flat.sum(axis=0)
            s2 += (flat ** 2).sum(axis=0)
            n += flat.shape[0]
            a = np.nan_to_num(agb[y0:y0 + PATCH, x0:x0 + PATCH], nan=0.0)
            agb_vals.append(a.ravel())
        print(f"[stats] {y}: done ({PATCHES_PER_YEAR} patches, X shape {X.shape})", flush=True)

    mean = s / n
    var = np.maximum(s2 / n - mean ** 2, 0.0)
    std = np.sqrt(var)
    std = np.where(std < 1e-6, 1.0, std)

    agb_all = np.concatenate(agb_vals)
    agb_mean = float(agb_all.mean())
    agb_std = float(agb_all.std()) or 1.0

    out = {
        "n_pixels": int(n),
        "x_mean": mean.tolist(),
        "x_std": std.tolist(),
        "agb_mean": agb_mean,
        "agb_std": agb_std,
    }
    out_path = SCRIPT_DIR / "channel_stats_2015_2018.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[stats] wrote {out_path}")
    print("x_mean:", np.round(mean, 4))
    print("x_std :", np.round(std, 4))
    print("agb   : mean=%.3f std=%.3f" % (agb_mean, agb_std))


if __name__ == "__main__":
    main()
