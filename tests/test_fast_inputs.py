"""The fast input path gives the same samples as the original one.

  compact_statics: statics and day of year inserted by expand_compact equal
    the channels the dataset used to concatenate itself.
  fast_stores: samples read from build_fast_stores' copies equal those read
    from the original stores, exactly for every channel except the BARRA ones,
    which are interpolated by matrix products instead of map_coordinates.

Needs the data cubes (SMOLDER_DATA) and, for the fast_stores tests, the
stores of smolder.data.build_fast_stores (skipped when missing). Runs with
pytest or directly: python tests/test_fast_inputs.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smolder.data.io import resolve                                           # noqa: E402
from smolder.data.zarr_dual_datamodule import (DualPatchConfig, DualWindowDataset,  # noqa: E402
                                               expand_compact)

OFFSETS = {2015: 0, 2019: 1461}
# the full model of submit_full_model.sh
FULL = dict(fire_history=True, fire_history_lags=(3, 4, 5), fire_history_distance=True,
            use_elevation=True, use_slope_aspect=True, use_fuel_age=True, fuel_age_lookback=1095,
            vpd_source="barra", use_frp=True, use_barra_uv=True, slow_veg="lai500", use_fast_ndvi=True,
            perfect_forecast=False, use_vpd_anomaly=False)
STORES = ("fire_inputs_continental.zarr", "barra_c2_fast.zarr", "cube_slow_8day_lai500m.zarr")


def _ds(year=2019, train=False, **kw):
    paths = ("cube_daily_smgrid_2015.zarr", "cube_daily_smgrid_2016.zarr") if year == 2015 \
        else ("cube_daily_smgrid_2019.zarr",)
    cfg = dict(zarr_paths=paths, stats_path="channel_stats_2015_2018.json",
               slow_cube_path="cube_slow_8day.zarr", day_offset=OFFSETS[year], patch_size=384,
               samples_per_epoch=8, seed=7, deterministic=not train, pos_frac=0.5 if train else 0.0,
               min_pos_pixels=45 if train else 0, new_fire_frac=0.3 if train else 0.0,
               fire_history_dropout_prob=0.3 if train else 0.0, fast_stores=False, **FULL)
    cfg.update(kw)
    return DualWindowDataset(DualPatchConfig(**cfg))


def _collate(samples):
    return {k: torch.stack([s[k] for s in samples]) for k in samples[0]}


def _have_stores():
    return all(resolve(p).exists() for p in STORES)


def test_linear_weights_match_map_coordinates():
    from scipy.ndimage import map_coordinates
    rng = np.random.default_rng(0)
    f = rng.normal(size=(23, 31)).astype(np.float32)
    ci = np.sort(rng.uniform(0.0, 22.0, 50)); cj = np.sort(rng.uniform(0.0, 30.0, 40))
    ci[0], cj[-1] = 0.0, 30.0
    FI, FJ = np.meshgrid(ci, cj, indexing="ij")
    ref = map_coordinates(f, [FI, FJ], order=1, mode="nearest")
    Ar = DualWindowDataset._linear_weights(ci, 23); Ak = DualWindowDataset._linear_weights(cj, 31)
    got = (Ar @ f.astype(np.float64) @ Ak.T).astype(np.float32)
    assert np.allclose(got, ref, rtol=0, atol=2e-6), np.abs(got - ref).max()


def test_compact_statics_expand_to_full_layout():
    for year in (2019, 2015):
        full, comp = _ds(year), _ds(year, compact_statics=True)
        rng = np.random.default_rng(year)
        samples_f, samples_c = [], []
        for _ in range(2):
            t = int(rng.choice(full.targets)); y0 = int(rng.integers(1000, 2600)); x0 = int(rng.integers(1000, 3300))
            samples_f.append(full.sample_at(t, y0, x0)); samples_c.append(comp.sample_at(t, y0, x0))
        bf, bc = _collate(samples_f), expand_compact(_collate(samples_c))
        assert set(bf) == set(bc), (set(bf) ^ set(bc))
        for k in bf:
            assert bf[k].shape == bc[k].shape and torch.equal(bf[k], bc[k]), k
        # far less data per sample
        n_full = sum(s["x_fast"].numel() + s["x_slow"].numel() for s in samples_f)
        n_comp = sum(s["x_fast"].numel() + s["x_slow"].numel() for s in samples_c)
        print(f"  compact {year}: {n_comp / n_full:.2f} of the full input volume")


def test_compact_statics_training_draws_with_dropout():
    # random draws incl. fire-history dropout: same RNG stream, same blanked channels
    full, comp = _ds(2015, train=True), _ds(2015, train=True, compact_statics=True)
    for i in range(4):
        a, b = full[i], comp[i]
        bb = expand_compact(_collate([b]))
        for k in a:
            assert torch.equal(a[k][None] if k not in ("t_end", "y0", "x0") else a[k][None], bb[k]), k


def test_fast_stores_equal_original():
    if not _have_stores():
        print("  skipped: build_fast_stores output missing")
        return
    rng = np.random.default_rng(1)
    worst = 0.0
    for year in (2019, 2015):
        old, new = _ds(year), _ds(year, fast_stores=True)
        assert new.fire_y is not None and new.barra_fast is not None and new.frp_feat is not None
        lay = old.channel_layout()["fast"]
        barra = set(lay.get("wind u/v", [])) | {0}               # VPD (BARRA) is fast channel 0
        ts = [int(old.targets[0]), int(old.targets[-1])] + [int(t) for t in rng.choice(old.targets, 4)]
        for t in ts:                                             # first target: zero-padded history
            y0 = int(rng.integers(0, old.H - 384)); x0 = int(rng.integers(0, old.W - 384))
            a, b = old.sample_at(t, y0, x0), new.sample_at(t, y0, x0)
            assert set(a) == set(b)
            for k in a:
                if k != "x_fast":
                    assert torch.equal(a[k], b[k]), (k, year, t)
            for c in range(a["x_fast"].shape[-1]):
                xa, xb = a["x_fast"][..., c], b["x_fast"][..., c]
                if c in barra:
                    d = float((xa - xb).abs().max())
                    worst = max(worst, d)
                    assert d < 1e-5, (c, d, year, t)
                else:
                    assert torch.equal(xa, xb), (c, year, t, float((xa - xb).abs().max()))
    print(f"  BARRA channels: max |difference| {worst:.2e} (normalized units)")


def test_fast_stores_same_training_draws():
    if not _have_stores():
        print("  skipped: build_fast_stores output missing")
        return
    old, new = _ds(2015, train=True), _ds(2015, train=True, fast_stores=True)
    for i in range(4):
        a, b = old[i], new[i]
        for k in ("t_end", "y0", "x0", "y", "x_slow", "mask"):
            assert torch.equal(a[k], b[k]), (k, i)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
