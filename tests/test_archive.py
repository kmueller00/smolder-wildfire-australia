"""The published data record is enough to run the model on 2020.

ARCHIVE_DIR holds only the extracted archives of the Zenodo record
(cube_2020_zenodo.zarr, cube_slow_8day_lai500m.zarr, fire_inputs_continental.zarr,
barra_c2_fast.zarr, ndvi_composites.zarr). For every valid 2020 issue day one
patch (random position, seed 0) is built twice: from the archive alone with the
fast stores, and from the full data (SMOLDER_DATA) with the original stores.
Every tensor must be identical; for N_FORWARD of the days the model (CKPT)
is run on both and its outputs must be identical too. No labels are scored.

Usage
  SMOLDER_DATA=<full data> ARCHIVE_DIR=<extracted record> CKPT=<swa.ckpt> python tests/test_archive.py
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smolder.data.zarr_dual_datamodule import DualPatchConfig, DualWindowDataset  # noqa: E402

YEAR, OFFSET, PATCH = 2020, 1826, 384
FULL = dict(fire_history=True, fire_history_lags=(3, 4, 5), fire_history_distance=True,
            use_elevation=True, use_slope_aspect=True, use_fuel_age=True, fuel_age_lookback=1095,
            vpd_source="barra", use_frp=True, use_barra_uv=True, slow_veg="lai500", use_fast_ndvi=True,
            perfect_forecast=False, use_vpd_anomaly=False, use_wind_align=False)


def build(data_dir, cube, fast):
    os.environ["SMOLDER_DATA"] = data_dir
    return DualWindowDataset(DualPatchConfig(
        zarr_paths=(cube,), stats_path="channel_stats_2015_2018.json", slow_cube_path="cube_slow_8day.zarr",
        day_offset=OFFSET, patch_size=PATCH, samples_per_epoch=1, seed=0, deterministic=True,
        fast_stores=fast, **FULL))


def main():
    full_dir, arch_dir = os.environ["SMOLDER_DATA"], os.environ["ARCHIVE_DIR"]
    os.chdir(arch_dir)                                    # nothing else resolvable from the cwd
    assert sorted(p for p in os.listdir(arch_dir) if p.endswith(".zarr")) == sorted(
        ["cube_2020_zenodo.zarr", "cube_slow_8day_lai500m.zarr", "fire_inputs_continental.zarr",
         "barra_c2_fast.zarr", "ndvi_composites.zarr"]), os.listdir(arch_dir)
    ref = build(full_dir, os.path.join(full_dir, f"cube_daily_smgrid_{YEAR}.zarr"), False)
    arc = build(arch_dir, f"cube_{YEAR}_zenodo.zarr", True)
    os.environ["SMOLDER_DATA"] = arch_dir
    assert np.array_equal(ref.targets, arc.targets)
    model = None
    n_fwd = int(os.environ.get("N_FORWARD", 24))
    if os.environ.get("CKPT"):
        from smolder.models.conv_lstm_lit_dual import ConvLSTMLitDual
        model = ConvLSTMLitDual.load_from_checkpoint(os.environ["CKPT"], map_location="cpu").eval()
    rng = np.random.default_rng(0)
    fwd_days = set(rng.choice(ref.targets, min(n_fwd, len(ref.targets)), replace=False).tolist())
    t0, n_fwd_done, worst = time.time(), 0, 0.0
    for i, t in enumerate(ref.targets):
        for _ in range(50):                               # a patch with land
            y0, x0 = int(rng.integers(0, ref.H - PATCH)), int(rng.integers(0, ref.W - PATCH))
            if ref._patch_ok(y0, x0):
                break
        a, b = ref.sample_at(int(t), y0, x0), arc.sample_at(int(t), y0, x0)
        assert set(a) == set(b), (t, set(a) ^ set(b))
        for k in a:
            assert torch.equal(a[k], b[k]), (int(t), y0, x0, k)
        if model is not None and int(t) in fwd_days:
            with torch.no_grad():
                pa = model.forward_seq(a["x_slow"][None], a["x_fast"][None], a["x_cat"][None], a["x_static"][None])
                pb = model.forward_seq(b["x_slow"][None], b["x_fast"][None], b["x_cat"][None], b["x_static"][None])
            assert torch.equal(pa, pb), (int(t), float((pa - pb).abs().max()))
            n_fwd_done += 1
        if i % 50 == 0:
            print(f"  day {i + 1}/{len(ref.targets)} ({time.time() - t0:.0f} s)", flush=True)
    print(f"[archive] {len(ref.targets)} issue days: all input tensors identical; "
          f"model output identical on {n_fwd_done} of them", flush=True)


if __name__ == "__main__":
    main()
