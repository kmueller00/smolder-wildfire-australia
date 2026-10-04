"""Checks for the BARRA-C2 inputs and the perfect_forecast channels.

perfect_forecast deliberately uses the weather of the three days after each
fast step's issue day (an upper bound, never an operational model). These
tests make sure it uses exactly those days, reads no fire or target data,
and leaves every other input unchanged. Needs the data (SMOLDER_DATA) and
barra_c2_daily.zarr with its stats. Runs with pytest or
python tests/test_perfect_forecast.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smolder.data.io import open_zarr_root                                       # noqa: E402
from smolder.data.zarr_dual_datamodule import DualPatchConfig, DualWindowDataset  # noqa: E402

OFF = 1461
Y0, X0, T_END = 1600, 2600, 200


def _ds(**kw):
    return DualWindowDataset(DualPatchConfig(
        zarr_paths=("cube_daily_smgrid_2019.zarr",), stats_path="channel_stats_2015_2018.json",
        slow_cube_path="cube_slow_8day.zarr", day_offset=OFF, patch_size=64, samples_per_epoch=1,
        seed=0, deterministic=True, fire_history=True, fire_history_lags=(3, 4, 5),
        fire_history_distance=True, **kw))


class _NoFire:
    """Stands in for a cube group: any fire or target array access fails."""
    def __init__(self, g):
        self.g = g

    def __getitem__(self, k):
        if k.startswith("y_"):
            raise AssertionError(f"perfect_forecast read {k}")
        return self.g[k]

    def __getattr__(self, k):
        return getattr(self.g, k)


def _bilinear(arr, fi, fj):
    i0, j0 = int(np.floor(fi)), int(np.floor(fj))
    a, b = fi - i0, fj - j0
    return ((1 - a) * (1 - b) * arr[i0, j0] + (1 - a) * b * arr[i0, j0 + 1]
            + a * (1 - b) * arr[i0 + 1, j0] + a * b * arr[i0 + 1, j0 + 1])


def test_barra_interpolation():
    ds = _ds(vpd_source="barra")
    z = open_zarr_root("barra_c2_daily.zarr")
    c0, a, f0, e = ds._GT
    g = T_END + OFF
    got = ds._barra_patch("vpd", [g], Y0, X0)[0]
    for r, c in ((0, 0), (31, 17), (63, 63)):
        fi = (f0 + (Y0 + r + 0.5) * e - ds.barra_lat[0]) / 0.04
        fj = (c0 + (X0 + c + 0.5) * a - ds.barra_lon[0]) / 0.04
        ref = _bilinear(np.asarray(z["vpd"][g], np.float32), fi, fj)
        assert abs(got[r, c] - ref) < 1e-4, (r, c, got[r, c], ref)


def test_vpd_source_barra_replaces_only_vpd():
    base, ds = _ds(), _ds(vpd_source="barra")
    b0, b1 = base.sample_at(T_END, Y0, X0), ds.sample_at(T_END, Y0, X0)
    assert not torch.equal(b0["x_fast"][..., 0], b1["x_fast"][..., 0])     # VPD is channel 0
    assert torch.equal(b0["x_fast"][..., 1:], b1["x_fast"][..., 1:])
    assert torch.equal(b0["x_slow"], b1["x_slow"]) and torch.equal(b0["y"], b1["y"])
    T = b1["x_fast"].shape[0]
    raw = ds._barra_patch("vpd", range(T_END - T + OFF, T_END + OFF), Y0, X0)
    m, s = ds.barra_stats["vpd"]
    assert np.allclose(b1["x_fast"][..., 0].numpy(), (raw - m) / s, atol=1e-4)


def test_perfect_forecast_no_fire_and_exact_days():
    base, ds = _ds(), _ds(perfect_forecast=True)
    b0, b1 = base.sample_at(T_END, Y0, X0), ds.sample_at(T_END, Y0, X0)
    assert b1["x_fast"].shape[-1] == b0["x_fast"].shape[-1] + 5
    assert torch.equal(b1["x_fast"][..., :-5], b0["x_fast"])              # fire history etc. unchanged
    assert torch.equal(b1["y"], b0["y"])
    # the channels never touch fire or target arrays
    groups, ds.groups = ds.groups, [_NoFire(g) for g in ds.groups]
    try:
        pf = ds._perfect_forecast(T_END, Y0, X0)
    finally:
        ds.groups = groups
    assert np.array_equal(pf, b1["x_fast"][..., -5:].numpy())
    # step j uses exactly days s_j+1..s_j+3
    T = pf.shape[0]
    cube = open_zarr_root("cube_daily_smgrid_2019.zarr")
    st = ds.barra_stats
    for j in (0, T - 1):
        s_j = T_END - T + j                                                # local issue day of step j
        days = [s_j + 1, s_j + 2, s_j + 3]
        wmax = ds._barra_patch("sfcWindmax", [d + OFF for d in days], Y0, X0).max(0)
        assert np.allclose(pf[j, ..., 0], (wmax - st["sfcWindmax"][0]) / st["sfcWindmax"][1], atol=1e-4)
        ppt = sum(np.asarray(cube["X"][d, Y0:Y0 + 64, X0:X0 + 64, 3], np.float32) for d in days)
        ip = 3
        ref = (np.nan_to_num(ppt) - 3 * ds.x_mean_all[ip]) / (np.sqrt(3) * ds.x_std_all[ip])
        assert np.allclose(pf[j, ..., 4], ref, atol=1e-4)
        # the step's target is y_fire_3d[s_j], i.e. fire on exactly those days s_j+1..s_j+3
        ref_y = (np.asarray(cube["y_fire_3d"][s_j, Y0:Y0 + 64, X0:X0 + 64]) > 0).astype(np.float32)
        assert np.array_equal(b1["y"][j].numpy() > 0, ref_y > 0)


def test_barra_vpd_anomaly_and_forecast_vpd():
    """With vpd_source barra, the anomaly uses BARRA VPD and its own 2015-2018
    climatology, and the perfect-forecast VPD is BARRA's."""
    ds = _ds(vpd_source="barra", use_vpd_anomaly=True, perfect_forecast=True)
    b = ds.sample_at(T_END, Y0, X0)
    T = b["x_fast"].shape[0]
    z = open_zarr_root("barra_c2_daily.zarr")
    a = np.r_[np.asarray(dict(z.attrs["vpd_climatology"])["anchors_doy"], float), 366.0]
    mu, sd = ds._barra_static("vpd_clim_mean", Y0, X0), ds._barra_static("vpd_clim_std", Y0, X0)
    import datetime as dt
    for j in (0, T - 1):
        t = T_END - T + j
        vpd = ds._barra_patch("vpd", [t + OFF], Y0, X0)[0]
        doy = (dt.date(2019, 1, 1) + dt.timedelta(days=t)).timetuple().tm_yday
        k0 = int(np.searchsorted(a, doy, side="right") - 1); f = (doy - a[k0]) / (a[k0 + 1] - a[k0])
        k1 = (k0 + 1) % 46
        ref = (vpd - (mu[k0] * (1 - f) + mu[k1] * f)) / np.maximum(sd[k0] * (1 - f) + sd[k1] * f, 0.01)
        assert np.allclose(b["x_fast"][j, ..., -6].numpy(), np.clip(ref, -6, 6), atol=1e-4)   # anomaly before PF
        s_j = T_END - T + j
        fut = ds._barra_patch("vpd", [s_j + 1 + OFF, s_j + 2 + OFF, s_j + 3 + OFF], Y0, X0).max(0)
        m_, s_ = ds.barra_stats["vpd"]
        assert np.allclose(b["x_fast"][j, ..., -2].numpy(), (fut - m_) / s_, atol=1e-4)


def test_frp_and_barra_uv_channels():
    base, ds = _ds(), _ds(use_frp=True, use_barra_uv=True)
    b0, b1 = base.sample_at(T_END, Y0, X0), ds.sample_at(T_END, Y0, X0)
    C0 = b0["x_fast"].shape[-1]
    assert b1["x_fast"].shape[-1] == C0 + 5 and torch.equal(b1["x_fast"][..., :C0], b0["x_fast"])
    T = b1["x_fast"].shape[0]
    f = open_zarr_root("firms_daily.zarr")
    land = np.asarray(open_zarr_root("cube_daily_smgrid_2019.zarr")["landmask"][Y0:Y0 + 64, X0:X0 + 64]) > 0
    for j in (0, T - 1):
        s_j = T_END - T + j + OFF                                   # global issue day of step j
        days = slice(s_j - 2, s_j + 1)                              # s_j-2 .. s_j, nothing later
        frp = np.asarray(f["frp_sum"][days, Y0:Y0 + 64, X0:X0 + 64], np.float32).sum(0)
        n = np.asarray(f["n_det"][days, Y0:Y0 + 64, X0:X0 + 64], np.float32).sum(0)
        assert np.allclose(b1["x_fast"][j, ..., C0].numpy(), np.log1p(frp) / 5.0 * land, atol=1e-5)
        assert np.allclose(b1["x_fast"][j, ..., C0 + 1].numpy(), np.log1p(n) / 3.0 * land, atol=1e-5)
        st = ds.barra_stats
        u = ds._barra_patch("uas", [T_END - T + j + OFF], Y0, X0)[0]
        assert np.allclose(b1["x_fast"][j, ..., C0 + 3].numpy(), (u - st["uas"][0]) / st["uas"][1], atol=1e-4)


def test_fire_history_dropout_blanks_frp():
    ds = DualWindowDataset(DualPatchConfig(
        zarr_paths=("cube_daily_smgrid_2019.zarr",), stats_path="channel_stats_2015_2018.json",
        slow_cube_path="cube_slow_8day.zarr", day_offset=OFF, patch_size=64, samples_per_epoch=1,
        seed=0, deterministic=False, fire_history=True, fire_history_lags=(3, 4, 5),
        fire_history_distance=True, use_frp=True, fire_history_dropout_prob=1.0))
    b = ds.sample_at(T_END, Y0, X0)
    i0 = ds.fire_hist_start_idx
    assert torch.all(b["x_fast"][..., i0:i0 + ds.n_fire_hist_channels] == 0)
    assert torch.all(b["x_fast"][..., -3:] == 0)                    # FRP channels blanked too


def test_fast_ndvi_channel_and_position():
    import json
    st = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "smolder", "data", "channel_stats_2015_2018.json")))
    cube = open_zarr_root("cube_daily_smgrid_2019.zarr")
    for kw in (dict(use_fast_ndvi=True), dict(use_fast_ndvi=True, use_frp=True, use_barra_uv=True,
                                               use_vpd_anomaly=True, vpd_source="barra")):
        base = _ds(**{k: v for k, v in kw.items() if k != "use_fast_ndvi"})
        ds = _ds(**kw)
        b0, b1 = base.sample_at(T_END, Y0, X0), ds.sample_at(T_END, Y0, X0)
        i = ds.fast_ndvi_idx
        assert b1["x_fast"].shape[-1] == b0["x_fast"].shape[-1] + 1
        assert torch.equal(b1["x_fast"][..., :i], b0["x_fast"][..., :i])          # before it: unchanged
        assert torch.equal(b1["x_fast"][..., i + 1:], b0["x_fast"][..., i:])      # after it: shifted by one
        T = b1["x_fast"].shape[0]
        for j in (0, T - 1):
            t = T_END - T + j
            raw = np.nan_to_num(np.asarray(cube["X"][t, Y0:Y0 + 64, X0:X0 + 64, 5], np.float32))
            assert np.allclose(b1["x_fast"][j, ..., i].numpy(), (raw - st["x_mean"][5]) / st["x_std"][5], atol=1e-5)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
