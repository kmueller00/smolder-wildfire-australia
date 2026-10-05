"""Downwind-alignment input (use_wind_align).

Checked against a brute-force computation on a fire-active 2019 patch: the
cosine between the day's BARRA-C2 wind at the pixel and the direction from the
nearest fire of y_fire_3d[s-3] to it, for pixels whose nearest fire pixel is
unique. Needs the data cubes (SMOLDER_DATA) and the fast stores. Runs with
pytest or directly: python tests/test_wind_align.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smolder.data.io import open_zarr_root                                    # noqa: E402
from smolder.data.zarr_dual_datamodule import DualPatchConfig, DualWindowDataset  # noqa: E402

OFF_2019 = 1461
T_END, Y0, X0 = 316, 448, 2900          # issue day 2019-11-12, savanna fires in Cape York
FULL = dict(fire_history=True, fire_history_lags=(3, 4, 5), fire_history_distance=True,
            use_elevation=True, use_slope_aspect=True, use_fuel_age=True, fuel_age_lookback=1095,
            vpd_source="barra", use_frp=True, use_barra_uv=True, slow_veg="lai500", use_fast_ndvi=True,
            perfect_forecast=False, use_vpd_anomaly=False, fast_stores=True)


def _ds(**kw):
    cfg = dict(zarr_paths=("cube_daily_smgrid_2019.zarr",), stats_path="channel_stats_2015_2018.json",
               slow_cube_path="cube_slow_8day.zarr", day_offset=OFF_2019, patch_size=384, samples_per_epoch=4,
               seed=3, deterministic=True, **FULL)
    cfg.update(kw)
    return DualWindowDataset(DualPatchConfig(**cfg))


def test_alignment_matches_brute_force():
    ds = _ds(use_wind_align=True)
    base = _ds(use_wind_align=False)
    a, b = ds.sample_at(T_END, Y0, X0), base.sample_at(T_END, Y0, X0)
    assert ds.channel_layout()["fast"]["downwind alignment"] == [a["x_fast"].shape[-1] - 1]
    assert torch.equal(a["x_fast"][..., :-1], b["x_fast"])            # no other channel moves
    wa = a["x_fast"][..., -1].numpy()
    assert np.abs(wa).max() <= 1.0 + 1e-6
    y = open_zarr_root("cube_daily_smgrid_2019.zarr")["y_fire_3d"]
    land = np.asarray(open_zarr_root("cube_daily_smgrid_2019.zarr")["landmask"][Y0:Y0 + 384, X0:X0 + 384]) > 0
    T = wa.shape[0]
    g0 = T_END - T + OFF_2019
    u = np.nan_to_num(ds._barra_patch("uas", range(g0, g0 + T), Y0, X0))
    v = np.nan_to_num(ds._barra_patch("vas", range(g0, g0 + T), Y0, X0))
    rng = np.random.default_rng(0)
    checked = 0
    for j in (0, T // 2, T - 1):
        s_j = T_END - T + j
        fire = (np.asarray(y[s_j - 3, Y0:Y0 + 384, X0:X0 + 384]) > 0) & land
        assert fire.any(), "test patch needs recent fire"
        assert np.all(wa[j][fire] == 0)                               # burning pixels
        fr, fc = np.nonzero(fire)
        cand = np.argwhere(land & ~fire)
        for r, c in cand[rng.choice(len(cand), 60, replace=False)]:
            d2 = (fr - r) ** 2 + (fc - c) ** 2
            k = np.argsort(d2)[:2]
            if d2[k[0]] == d2[k[1]]:
                continue                                              # nearest fire pixel not unique
            de, dn = c - fc[k[0]], fr[k[0]] - r
            exp = (u[j, r, c] * de + v[j, r, c] * dn) / (np.hypot(de, dn) * np.hypot(u[j, r, c], v[j, r, c]))
            assert abs(wa[j, r, c] - exp) < 1e-4, (j, r, c, wa[j, r, c], exp)
            checked += 1
    assert checked >= 50, checked
    print(f"  {checked} pixels match the brute-force alignment")


def test_dropout_blanks_alignment():
    ds = _ds(use_wind_align=True, deterministic=False, fire_history_dropout_prob=1.0, pos_frac=1.0,
             min_pos_pixels=45)
    for i in range(2):
        out = ds[i]
        assert float(out["x_fast"][..., -1].abs().max()) == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
