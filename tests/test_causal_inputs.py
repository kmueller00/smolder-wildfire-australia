"""Exact index checks of the causal inputs (causal_inputs=True), against the
raw stores, for random 2019 and 2020 samples of model B's configuration:

  NDVI (fast, step s)     = the daily cube's NDVI of day s - 7 (the composite shown
                            then has ended by day s)
  temperature (step s)    = BARRA-C2 tasmax of UTC day s (bilinear, as VPD)
  biomass                 = CCI map of year Y - 1 (2010 for 2015), Y = issue year
  land cover (x_cat[0])   = LC100 map of year max(Y - 2, 2015)
  target window           = inside the split's own days
  LAI store               = lag_days >= 31 (the builder asserts the rule per day)

Run: SMOLDER_DATA=... python tests/test_causal_inputs.py
"""
import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smolder.data.io import open_zarr_root                                        # noqa: E402
from smolder.data.zarr_dual_datamodule import DualPatchConfig, DualWindowDataset  # noqa: E402

B = dict(fire_history=True, fire_history_lags=(3, 4, 5), fire_history_distance=True,
         use_elevation=True, use_slope_aspect=True, use_fuel_age=True, fuel_age_lookback=1095,
         vpd_source="barra", use_frp=True, use_barra_uv=True, slow_veg="lai500", use_fast_ndvi=True,
         perfect_forecast=False, use_vpd_anomaly=False, use_wind_align=True)
OFF = {2019: 1461, 2020: 1826}


def gdate(g):
    return dt.date(2015, 1, 1) + dt.timedelta(days=int(g))


def main():
    rng = np.random.default_rng(3)
    agb = open_zarr_root("agb_yearly.zarr"); ay = [int(v) for v in agb["years"][...]]
    lc = open_zarr_root("landcover_yearly.zarr"); ly = [int(v) for v in lc["years"][...]]
    lai = open_zarr_root("cube_slow_8day_lai500_lag31.zarr")
    assert int(lai.attrs["lag_days"]) >= 31
    for year, off in OFF.items():
        kw = dict(zarr_paths=(f"cube_daily_smgrid_{year}.zarr",), stats_path="channel_stats_2015_2018.json",
                  slow_cube_path="cube_slow_8day.zarr", day_offset=off, patch_size=128, samples_per_epoch=1,
                  seed=0, deterministic=True, causal_inputs=True, **B)
        ds = DualWindowDataset(DualPatchConfig(fast_stores=True, **kw))
        slow_ds = DualWindowDataset(DualPatchConfig(fast_stores=False, **kw))   # map_coordinates path
        assert (ds.targets + 2 <= ds.T_total - 1).all()
        lay = ds.channel_layout()
        cube = {y: open_zarr_root(f"cube_daily_smgrid_{y}.zarr") for y in (year - 1, year)}
        for _ in range(12):
            t_end = int(rng.choice(ds.targets)); y0 = int(rng.integers(1000, 2400)); x0 = int(rng.integers(1500, 3800))
            b = ds.sample_at(t_end, y0, x0)
            xf, xs, xc = b["x_fast"].numpy(), b["x_slow"].numpy(), b["x_cat"].numpy()
            T = xf.shape[0]
            Y = gdate(t_end - 1 + off).year
            for j in (0, T - 1):
                g = t_end - T + j + off
                d7 = gdate(g - 7)
                raw = np.asarray(cube[d7.year]["X"][d7.timetuple().tm_yday - 1, y0:y0 + 128, x0:x0 + 128, 5], np.float32)
                i = lay["fast"]["NDVI (fast)"][0]
                ref = (np.nan_to_num(raw) - ds.x_mean_all[5]) / ds.x_std_all[5]
                assert np.allclose(xf[j, ..., i], ref, atol=1e-5), ("ndvi", t_end, j)
                bt = slow_ds._barra_patch("tasmax", [g], y0, x0)[0]
                it = lay["fast"]["TMAX"][0]
                st = ds.barra_stats["tasmax"]
                ref_t = (np.nan_to_num(bt, nan=st[0]) - st[0]) / st[1]
                assert np.abs(xf[j, ..., it] - ref_t).max() < 1e-3, ("tmax", t_end, j, np.abs(xf[j, ..., it] - ref_t).max())
            ya = Y - 1 if (Y - 1) in ay else max(v for v in ay if v < Y - 1)
            a = np.asarray(agb["agb"][ay.index(ya), y0:y0 + 128, x0:x0 + 128], np.float32)
            ref_a = np.nan_to_num((a - ds.agb_mean) / ds.agb_std)
            assert np.allclose(xs[0, ..., lay["slow"]["biomass"][0]], ref_a, atol=1e-5), ("agb", Y)
            yl = max(Y - 2, ly[0])
            assert np.array_equal(xc[..., 0], np.asarray(lc["landcover"][ly.index(yl), y0:y0 + 128, x0:x0 + 128])), ("lc", Y)
        print(f"{year}: NDVI = cube day s-7, TMAX = BARRA tasmax day s, biomass map {ya}, land cover {yl}: all equal")
    print("causal input checks passed")


if __name__ == "__main__":
    main()
