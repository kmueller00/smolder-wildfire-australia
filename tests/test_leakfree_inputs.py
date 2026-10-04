"""Leak checks for the slow-branch window and the past-fire loss weight.

Needs the data cubes (SMOLDER_DATA), cube_slow_8day.zarr,
fire_dist30_continental.zarr and checkpoints/smolder_swa.ckpt. Runs with
pytest or directly: python tests/test_leakfree_inputs.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smolder.data.io import open_zarr_root                                    # noqa: E402
from smolder.data.zarr_dual_datamodule import DualPatchConfig, DualWindowDataset  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFF_2019 = 1461


def _ds(legacy=False, store=None):
    return DualWindowDataset(DualPatchConfig(
        zarr_paths=("cube_daily_smgrid_2019.zarr",), stats_path="channel_stats_2015_2018.json",
        slow_cube_path="cube_slow_8day.zarr", day_offset=OFF_2019, patch_size=128, samples_per_epoch=1,
        seed=0, deterministic=True, fire_history=True, fire_history_lags=(3, 4, 5),
        fire_history_distance=True, slow_window_legacy=legacy, past_fire_dist_store=store))


def test_slow_window_ends_on_issue_day():
    ds = _ds()
    starts, b = ds.slow_bin_start, ds.cfg.slow_bin
    for t_end in range(14, 366):
        D = t_end - 1 + OFF_2019                                  # issue day, global
        e = ds._slow_bin_end(t_end)
        newest_last_day = starts[e - 1] + b - 1
        assert newest_last_day <= D, (t_end, newest_last_day, D)  # nothing after the issue day
        assert newest_last_day >= D - (b - 1)                     # at most slow_bin-1 days old
        assert e - ds.t_slow >= 0


def test_legacy_window_reproduces_released_alignment():
    ds = _ds(legacy=True)
    for t_end in range(14, 366):
        assert ds._slow_bin_end(t_end) == int(np.searchsorted(ds.slow_bin_start, t_end + OFF_2019, side="right"))


def test_past_dist_index():
    ds = _ds(store="fire_dist30_continental.zarr")
    store = open_zarr_root("fire_dist30_continental.zarr")["dist_px"]
    y0, x0 = 1600, 2600
    for t_end in (20, 150, 300):
        b = ds.sample_at(t_end, y0, x0)
        T = b["past_dist"].shape[0]
        for j in (0, T - 1):
            s_j = t_end - T + j                                   # issue day of fast step j
            ref = np.asarray(store[s_j - 3 + OFF_2019, y0:y0 + 128, x0:x0 + 128], np.float32)
            assert np.array_equal(b["past_dist"][j].numpy(), ref)
        # the target of step j is y_fire_3d[s_j]; its history must end before it starts
        assert b["y"].shape[0] == T


def test_past_fire_weight_off_is_identity_and_on_is_correct():
    from smolder.models.conv_lstm_lit_dual import ConvLSTMLitDual
    m = ConvLSTMLitDual.load_from_checkpoint(os.path.join(REPO, "checkpoints", "smolder_swa.ckpt"),
                                             map_location="cpu").eval()
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 32, 32, generator=g)
    y = (torch.rand(2, 32, 32, generator=g) < 0.05).float()
    d = torch.randint(0, 40, (2, 32, 32), generator=g).float()
    mask = torch.ones(2, 32, 32, dtype=torch.bool)
    assert m._past_fire_weight(y, d) is None                       # default a = 0
    base, _ = m._compute_loss(logits, y, mask=mask)
    off, _ = m._compute_loss(logits, y, mask=mask, extra_weight=m._past_fire_weight(y, d))
    assert torch.equal(base, off)
    m.past_fire_weight_a = 10.0
    w = m._past_fire_weight(y, d)
    exp = torch.where(y > 0.5, 1.0 + 10.0 * torch.clamp(d / 10.0, max=1.0), torch.ones_like(d))
    assert torch.equal(w, exp)
    assert torch.all(w[y < 0.5] == 1.0)                            # background keeps weight 1
    assert torch.all(w[(y > 0.5) & (d == 0)] == 1.0)              # fire at recent fire: no extra weight
    assert torch.all(w[(y > 0.5) & (d >= 10)] == 11.0)            # saturates at 1 + a beyond 10 px
    on, _ = m._compute_loss(logits, y, mask=mask, extra_weight=w)
    assert not torch.equal(base, on)


def test_vpd_anomaly_channel():
    import datetime as dt
    base = _ds()
    ds = DualWindowDataset(DualPatchConfig(**{**base.cfg.__dict__, "use_vpd_anomaly": True}))
    y0, x0, t_end = 1600, 2600, 200
    b0, b1 = base.sample_at(t_end, y0, x0), ds.sample_at(t_end, y0, x0)
    assert b1["x_fast"].shape[-1] == b0["x_fast"].shape[-1] + 1        # one extra channel, last
    assert torch.equal(b1["x_fast"][..., :-1], b0["x_fast"])            # everything else unchanged
    g = open_zarr_root("cube_daily_smgrid_2019.zarr")
    clim = open_zarr_root("climatology_2015_2018.zarr")
    a = np.r_[np.asarray(clim.attrs["anchors_doy"], float), 366.0]
    T = b1["x_fast"].shape[0]
    for j, (r, c) in ((T - 1, (10, 20)), (0, (100, 50))):
        t = t_end - T + j                                               # local day of fast step j
        vpd = float(g["X"][t, y0 + r, x0 + c, 2])
        doy = (dt.date(2019, 1, 1) + dt.timedelta(days=t)).timetuple().tm_yday
        k0 = int(np.searchsorted(a, doy, side="right") - 1); f = (doy - a[k0]) / (a[k0 + 1] - a[k0])
        k1 = (k0 + 1) % 46
        mu = clim["vpd_mean"][:, y0 + r, x0 + c].astype(np.float32); sd = clim["vpd_std"][:, y0 + r, x0 + c].astype(np.float32)
        ref = (vpd - (mu[k0] * (1 - f) + mu[k1] * f)) / max(sd[k0] * (1 - f) + sd[k1] * f, 0.01)
        assert abs(float(b1["x_fast"][j, r, c, -1]) - np.clip(ref, -6, 6)) < 1e-4, (j, ref)


def test_slow_veg_ndvi():
    import json
    base = _ds()
    rep = DualWindowDataset(DualPatchConfig(**{**base.cfg.__dict__, "slow_veg": "ndvi"}))
    add = DualWindowDataset(DualPatchConfig(**{**base.cfg.__dict__, "slow_veg": "lai+ndvi"}))
    y0, x0, t_end = 1600, 2600, 200
    b0, br, ba = (d.sample_at(t_end, y0, x0) for d in (base, rep, add))
    assert torch.equal(br["x_slow"][..., 1:], b0["x_slow"][..., 1:])          # only LAI (channel 0) replaced
    assert not torch.equal(br["x_slow"][..., 0], b0["x_slow"][..., 0])
    assert ba["x_slow"].shape[-1] == b0["x_slow"].shape[-1] + 1
    assert torch.equal(ba["x_slow"][..., :-1], b0["x_slow"])                 # NDVI appended last
    assert torch.equal(ba["x_slow"][..., -1], br["x_slow"][..., 0])
    # newest NDVI bin = mean of the cube's NDVI over that bin's 8 days, which end on or before the issue day
    e = base._slow_bin_end(t_end)
    start = int(base.slow_bin_start[e - 1])
    assert start + 7 <= t_end - 1 + OFF_2019
    cube = open_zarr_root("cube_daily_smgrid_2019.zarr")
    days = [start + k - OFF_2019 for k in range(8)]
    nd = np.mean([np.nan_to_num(np.asarray(cube["X"][d, y0:y0 + 128, x0:x0 + 128, 5], np.float32)) for d in days], 0)
    st = json.load(open(os.path.join(REPO, "smolder", "data", "channel_stats_2015_2018.json")))
    ref = (nd - st["x_mean"][5]) / st["x_std"][5]
    assert np.allclose(br["x_slow"][-1, ..., 0].numpy(), ref, atol=1e-4)


def test_fuel_age_store_matches_direct_computation():
    import time
    base = _ds()
    kw = {**base.cfg.__dict__, "use_fuel_age": True, "fuel_age_lookback": 250}
    direct = DualWindowDataset(DualPatchConfig(**{**kw, "fuel_age_store": None}))
    stored = DualWindowDataset(DualPatchConfig(**kw))
    assert direct.fire_age is None and stored.fire_age is not None
    y0, x0, t_end = 1600, 2600, 300            # 250-day history of every step lies inside 2019
    a, b = direct.sample_at(t_end, y0, x0), stored.sample_at(t_end, y0, x0)
    assert torch.equal(a["x_fast"], b["x_fast"])
    long = DualWindowDataset(DualPatchConfig(**{**kw, "fuel_age_lookback": 730}))
    t0 = time.time(); c = long.sample_at(t_end, y0, x0); dt_ = time.time() - t0
    i = long.fire_hist_start_idx + long.n_fire_hist_channels             # fuel age follows the fire history
    ages = (c["x_fast"][..., i].numpy() + 1.0) * 365.0                   # undo (age - LB/2) / (LB/2)
    assert ages.min() >= -1e-3 and ages.max() <= 730 + 1e-3 and (ages > 250).any()
    assert dt_ < 30, dt_


def test_data_loader_workers_draw_different_samples():
    from torch.utils.data import DataLoader
    from smolder.data.zarr_dual_datamodule import seed_worker

    def batches():
        ds = DualWindowDataset(DualPatchConfig(
            zarr_paths=("cube_daily_smgrid_2017.zarr",), stats_path="channel_stats_2015_2018.json",
            slow_cube_path="cube_slow_8day.zarr", day_offset=731, patch_size=64, samples_per_epoch=32, seed=123,
            min_pos_pixels=1, pos_frac=0.5, fire_history=True, fire_history_lags=(3, 4, 5),
            fire_history_distance=True))
        dl = DataLoader(ds, batch_size=2, num_workers=4, worker_init_fn=seed_worker)
        return [tuple((int(t), int(y), int(x)) for t, y, x in zip(b["t_end"], b["y0"], b["x0"]))
                for _, b in zip(range(8), dl)]
    a = batches()
    assert len(set(a)) == len(a)                      # every batch distinct (it was 1 per 4 workers before)
    assert batches() == a                             # reproducible for a given seed


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
