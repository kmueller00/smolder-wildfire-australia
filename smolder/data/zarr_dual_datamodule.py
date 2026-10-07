"""SMOLDER data pipeline: patches with a slow and a fast temporal window.

For a target day t and a PATCH x PATCH window, each sample holds
  x_slow  (18, H, W, C)  LAI (HiQ-LAI 500 m), soil moisture, precipitation
                         over 144 days in 8-day bins, plus broadcast statics
                         and day-of-year (sin, cos)
  x_fast  (14, H, W, C)  VPD, daily maximum air temperature (with
                         causal_inputs; land-surface temperature without),
                         wind over 14 days,
                         plus the same statics, day-of-year, and fire history
                         (fire at t-3, t-4, t-5 and exp(-distance/5 px) to fire
                         at t-3; nothing later than t-3 is ever read)
  x_cat   (H, W, 2)      land cover and Koppen-Geiger class (embedded in-model)
  y       (14, H, W)     next-3-day fire occurrence for every fast-window day
  mask    (H, W)         land mask

All cubes are indexed on one continuous 2015-2020 day axis, so windows can
cross year boundaries. Inputs are standardised with channel_stats_2015_2018.json
(training years only).

causal_inputs (default on) restricts every input to what is known on the
issue day; see DualPatchConfig.causal_inputs and tests/leak_audit_*.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl

from smolder.data.io import daily_cube, data_dir, open_zarr_root, resolve, resolve_stats

# Channel order of the 7-channel training cubes and of channel_stats_2015_2018.json:
# [sm, wind, vpd, precip, lst_day, ndvi, lai]. Cubes that carry a `channels`
# attribute (e.g. the NDVI-free Zenodo archive) are read by name instead.
CH = {"SM": 0, "WIND": 1, "VPD": 2, "PPT": 3, "LST": 4, "NDVI": 5, "LAI": 6}

# NDVI (5) deliberately absent: redundant with LAI, see module docstring.
SLOW_CHANNELS = ("LAI", "SM", "PPT")
FAST_CHANNELS = ("VPD", "LST", "WIND")
# variables of barra_c2_fast.zarr's x, in channel order (build_fast_stores)
BARRA_FAST_VARS = ["vpd", "uas", "vas"]

# PPT accumulates (a single heavy rain day matters and would be lost by
# striding); the others are averaged over each bin.
SLOW_AGG = {"LAI": "mean", "SM": "mean", "PPT": "sum"}

# smips grid georeferencing (EPSG:4326, 0.01deg pixels), used only for the
# wind-direction feature's patch-center lat/lon lookup into the BARRA2 cache.
GRID_LON0, GRID_LAT0, GRID_PX = 112.904998779, -9.005000113999998, 0.01


@dataclass
class DualPatchConfig:
    zarr_paths: Tuple[str, ...]
    stats_path: str
    y_key: str = "y_fire_3d"
    valid_t_key: Optional[str] = "y_fire_3d_valid"

    # Pre-binned slow cube from build_slow_cube.py. Strongly preferred: reading
    # 144 raw days per sample costs ~2.3s and pulls 1079MB to keep 14MB (76x
    # amplification, bandwidth-bound on NFS -- more workers do not help). The
    # pre-binned cube serves the same window in ~0.10s (22x faster) and is 47GB,
    # small enough to sit in page cache. When None, falls back to aggregating raw
    # days on the fly (correct but slow; used by tests).
    slow_cube_path: Optional[str] = None
    # Day index of this split's first day on the slow cube's GLOBAL 2015-2020
    # axis. Each dataset numbers its own days from 0, but the slow cube is built
    # once over all years -- without this the val split (2019) would look up bins
    # near day 0 and silently read 2015's data.
    day_offset: int = 0

    slow_days: int = 144          # inside the measured 130-150 saturation band
    slow_bin: int = 8             # native cadence of LAI (8-day composites)
    fast_days: int = 14

    patch_size: int = 256
    samples_per_epoch: int = 2000
    seed: int = 0

    min_valid_frac: float = 0.45
    min_pos_pixels: int = 20
    max_tries: int = 1000
    pos_frac: float = 0.0
    # Fraction of TRAINING samples forced to contain at least min_new_fire_
    # pixels genuinely NEW fire pixels (target fire not present at t_end-3 at
    # the same pixel -- a cheap, non-dilated proxy for the val_ap_newfire/
    # operational_stats_dual.py "new fire" definition, good enough for a
    # sampling bias). Unlike isolation_gamma (reweights the LOSS after a
    # patch is drawn) or fire_history_dropout_prob (masks a feature), this
    # changes which patches the model trains ON -- over-representing the
    # rare hard case directly in the training distribution, not just at loss
    # time. Independent of pos_frac (both can be active on the same draw).
    new_fire_frac: float = 0.0
    min_new_fire_pixels: int = 1

    add_doy: bool = True
    deterministic: bool = False
    # leakage-free past-fire channels on the fast branch (lags must be >=3)
    fire_history: bool = False
    fire_history_lag: int = 3
    fire_history_lags: Optional[Tuple[int, ...]] = None
    fire_history_distance: bool = False
    # ---- third, YEAR-SCALE branch (accumulated drought) ----
    # Measured on identical fire patches: year-scale accumulation beats both the
    # single lag and the 144d window for every slow variable --
    #   SM  0.477 (single@150) -> 0.515 (144d) -> 0.602 (432d)
    #   PPT 0.501 -> 0.577 -> 0.616 ; LAI 0.672 -> 0.679 -> 0.711
    # i.e. CUMULATIVE moisture deficit carries signal no single lag does, which
    # is why SM had ~zero permutation importance in the 30-day model.
    # 432d / 24d bins = 18 steps. 24 = 3x8 so it reuses the existing 8-day cube.
    vslow_days: int = 0          # 0 = branch disabled
    vslow_bin: int = 24

    # Lightning flash-density climatology (LIS/OTD HRFC), appended to the static
    # stack alongside agb/landmask. Passed a two-regime XGBoost gate (+7.4% test
    # AP, roads/population did NOT: -4 to -9%) -- see aux_xgb_check.py.
    use_lightning: bool = False
    # Elevation (ETOPO1). Passed the same XGBoost gate: +5.3% test AP, ranked
    # 8/15 by SHAP -- see aux_xgb_check.py.
    use_elevation: bool = False
    # Slope + aspect (sin/cos), derived from the ETOPO1 DEM. + fuel_age (days
    # since last fire, leakage-free). Passed the gate TOGETHER: +22.3% AP on
    # top of elevation alone -- see aux_xgb_check.py.
    use_slope_aspect: bool = False
    use_fuel_age: bool = field(default_factory=lambda: os.environ.get("USE_FUEL_AGE", "0") == "1")
    fuel_age_lookback: int = field(default_factory=lambda: int(os.environ.get("FUEL_AGE_LOOKBACK", 250)))
    # Precomputed days since the last fire per pixel and day (build_fire_age,
    # continuous 2015-2020). Same values as computing from y_fire_3d inside the
    # sample, but reads 14 slices instead of fuel_age_lookback, and is not cut
    # at the first day of a split's own cubes. Used when the store exists.
    fuel_age_store: Optional[str] = "fire_age_continental.zarr"
    # Downwind-of-recent-fire wind alignment (BARRA2 AUS-11). +2.4% AP on top
    # of the above, ranked 12/19 by SHAP -- see aux_xgb_check_wind.py.
    use_wind_dir: bool = False
    # McArthur FFDI and Griffiths drought factor from SILO (ffdi_<year>.zarr),
    # and Sentinel-2 live fuel moisture (DEA ga_s2_fmc_3_v1) on the model grid.
    use_ffdi: bool = False
    use_fmc: bool = False
    fmc_store: str = "fmc_weekly_ff.zarr"
    # Random flip (east-west mirror) + k*90deg rotation, TRAIN ONLY (val_ds
    # uses deterministic=True, unaffected regardless of this flag). Was never
    # implemented for this datamodule (only the old v1/v2 single-branch one
    # had it) -- implemented with care because of the
    # wind u/v / aspect correctness trap _augment below handles explicitly.
    augment: bool = False
    # Probability of blanking the fire_hist_t-3/4/5 + fire_dist channels to 0
    # (a valid "no recent fire nearby" state, not a corruption -- fire_hist is
    # 0/1 and fire_dist is exp(-dist/5) which already ->0 far from any fire)
    # for a training sample, forcing the network to predict from weather/
    # terrain alone on those samples. Targets the fire-proximity shortcut
    # directly (permutation importance: fire_dist+fire_hist_t-3 = ~78% of the
    # model's decisions) rather than just reweighting the same loss the way
    # isolation_gamma does. TRAIN ONLY, 0.0 = disabled (old behaviour).
    fire_history_dropout_prob: float = 0.0
    # Slow-branch window. Fixed (default): only 8-day bins that END on or
    # before the issue day D = t_end - 1 are used, so the newest bin is 0-7
    # days old. Legacy: every bin that STARTS on or before t_end, the
    # behaviour the first released checkpoint (smolder_v1_swa.ckpt) was trained with;
    # its newest bin always reaches past D, by 1-8 days and over the whole
    # target window on 75 % of issue days (results/slow_window_leak_check_2019.json).
    # Set SLOW_WINDOW_LEGACY=1 only to reproduce that checkpoint's results.
    slow_window_legacy: bool = field(
        default_factory=lambda: os.environ.get("SLOW_WINDOW_LEGACY", "0") == "1")
    # Optional continent-wide distance to past fire for a loss weight, read
    # from a store built by smolder.data.build_fire_distance (WINDOW=30: fire
    # of the last 32 days). Returned as batch["past_dist"] (T, H, W) in px,
    # for fast step s at index s - 3, so it never includes fire after that
    # step's own issue day. Not a model input. None = not returned.
    past_fire_dist_store: Optional[str] = None
    # Optional standardized VPD anomaly as one extra fast-branch channel, the
    # LAST channel of x_fast: (VPD_t - mean) / max(sd, 0.01 kPa) for every fast
    # day t, with the per-pixel climatology of climatology_2015_2018.zarr
    # (2015-2018 only, +-15 days, anchors every 8 days, interpolated), from
    # smolder.evaluation.anomaly_feature_diagnostic. Clipped to +-6.
    use_vpd_anomaly: bool = field(
        default_factory=lambda: os.environ.get("USE_VPD_ANOMALY", "0") == "1")
    clim_store: str = "climatology_2015_2018.zarr"
    # BARRA-C2 (4.4 km) daily fields from smolder.data.build_barra_c2, read on
    # their native grid and interpolated bilinearly to the patch.
    #   vpd_source "barra": the fast-branch VPD channel is BARRA-C2 VPD at
    #     tasmax instead of the 31 km ERA5-based product (normalized with the
    #     2015-2018 statistics stored in the BARRA store).
    #   perfect_forecast: 5 extra fast-branch channels (appended last) with the
    #     weather of the three days AFTER each fast step's issue day s:
    #     max sfcWindmax, mean uas, mean vas (BARRA-C2), max VPD (cube) and
    #     precipitation sum (cube) over s+1..s+3; with vpd_source "barra" the
    #     VPD of both the perfect_forecast channels and the VPD anomaly comes
    #     from BARRA-C2 (anomaly against vpd_clim_* of the BARRA store). This
    #     uses future weather on
    #     purpose: an upper bound for perfect weather forecasts, never an
    #     operational model. No fire information is read for these channels.
    vpd_source: str = field(default_factory=lambda: os.environ.get("VPD_SOURCE", "montes"))
    perfect_forecast: bool = field(
        default_factory=lambda: os.environ.get("PERFECT_FORECAST", "0") == "1")
    barra_store: str = "barra_c2_daily.zarr"
    # VIIRS fire radiative power (firms_daily.zarr, smolder.data.build_firms_daily):
    # 3 fast-branch channels per step s over the days s-2..s, the same days as
    # the newest fire-history map: log(1 + summed FRP) / 5, log(1 + number of
    # detections) / 3, share of night detections. Blanked together with the
    # fire history by fire_history_dropout_prob.
    use_frp: bool = field(default_factory=lambda: os.environ.get("USE_FRP", "0") == "1")
    firms_store: str = "firms_daily.zarr"
    # BARRA-C2 daily mean 10 m wind components (eastward, northward) of every
    # fast day as 2 fast-branch channels, normalized with the store's
    # 2015-2018 statistics. Absolute directions: incompatible with augment.
    use_barra_uv: bool = field(default_factory=lambda: os.environ.get("USE_BARRA_UV", "0") == "1")
    # Downwind alignment as the LAST fast-branch channel: for every fast step s
    # and pixel, the cosine of the angle between that day's BARRA-C2 10 m mean
    # wind at the pixel and the direction from the nearest fire of y_fire_3d[s-3]
    # (days s-2..s, the newest fire history) to the pixel: +1 downwind, -1
    # upwind, 0 crosswind; 0 at burning pixels and without fire in the patch.
    # Uses no information after day s. Blanked by fire_history_dropout_prob.
    use_wind_align: bool = field(default_factory=lambda: os.environ.get("USE_WIND_ALIGN", "0") == "1")
    # Vegetation in the slow branch: HiQ-LAI 500 m averaged to 1 km
    # (build_lai500_slow). The 5 km LAI of the cubes and the NDVI variants are
    # no longer supported: both reach past the issue day.
    slow_veg: str = field(default_factory=lambda: os.environ.get("SLOW_VEG", "lai500"))
    # MODIS 500 m NDVI from the daily cubes as one fast-branch channel (value of
    # each fast day: the newest 8-day composite), next to LAI in the slow
    # branch. Placed after the BARRA wind channels and before the VPD anomaly
    # and perfect-forecast channels; its position is self.fast_ndvi_idx.
    use_fast_ndvi: bool = field(default_factory=lambda: os.environ.get("USE_FAST_NDVI", "0") == "1")
    lai500_slow_store: str = "cube_slow_8day_lai500.zarr"   # SLOW_VEG=lai500 (build_lai500_slow)
    # Read y_fire_3d, fuel age, FRP, BARRA vpd/uas/vas and (SLOW_VEG=lai500)
    # the slow bins from the time-blocked copies of smolder.data.build_fast_stores:
    # the same values in ~50 instead of ~560 chunk files per sample. FRP comes
    # as precomputed per-day features, BARRA is interpolated with two matrix
    # products instead of map_coordinates (same bilinear weights, float64).
    fast_stores: bool = field(default_factory=lambda: os.environ.get("FAST_STORES", "0") == "1")
    fire_store: str = "fire_inputs_continental.zarr"
    barra_fast_store: str = "barra_c2_fast.zarr"
    slow_lai500m_store: str = "cube_slow_8day_lai500m.zarr"
    # NDVI for use_fast_ndvi when the daily cube has no NDVI channel (the
    # archived cubes): the distinct 8-day composites and, per global day, the
    # composite it shows (package_zenodo STEP=ndvi). Same values as the cube.
    ndvi_store: str = "ndvi_composites.zarr"
    # Leave the statics and day-of-year channels out of x_slow/x_fast and return
    # them once per sample ("x_static", "doy_slow", "doy_fast"); expand_compact()
    # inserts them again on the GPU at the same channel positions. Training only
    # (DualDataModule); a dataset used directly keeps the full layout.
    compact_statics: bool = False
    # Inputs restricted to what is known on the issue day (leak audit of
    # 2026-10-06, tests/leak_audit_fire.py and tests/leak_audit_event_study.py):
    #   NDVI (fast)   the newest 8-day composite that has ended: the one shown on
    #                 day s - 7 (a MOD09A1 composite is labelled by its first day)
    #   temperature   BARRA-C2 daily maximum air temperature of day s in place of
    #                 the MODIS LST, whose cloudy days are filled with a spline
    #                 over the whole year (Zhang et al. 2022)
    #   LAI (slow)    lai500_lag31_store: composites starting <= d - 31, past
    #                 HiQ-LAI's smoothing over +-3 composites (Yan et al. 2024)
    #   biomass       the CCI map of year Y-1 (2010 for 2015) for an issue day in
    #                 year Y (agb_store); the cubes hold the 2015-2020 mean
    #   land cover    the LC100 map of year max(Y-2, 2015) (landcover_store);
    #                 each map uses three years of data, centred on its year
    #   target        issue days whose 3-day target window leaves the split's
    #                 own days are dropped (no labels of the next split)
    # Off reproduces the earlier inputs, for checkpoints trained with them.
    causal_inputs: bool = field(default_factory=lambda: os.environ.get("CAUSAL_INPUTS", "1") == "1")
    ndvi_lag_days: int = 7
    lai500_lag31_store: str = "cube_slow_8day_lai500_lag31.zarr"
    slow_lai500m_lag31_store: str = "cube_slow_8day_lai500m_lag31.zarr"
    barra_fast_tmax_store: str = "barra_c2_fast_tmax.zarr"           # vpd, uas, vas, tasmax
    agb_store: str = "agb_yearly.zarr"
    landcover_store: str = "landcover_yearly.zarr"


def expand_compact(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Insert the statics and day-of-year channels that compact_statics left
    out, on whatever device the batch is on: x_static is broadcast over the
    time steps and doy_slow/doy_fast over the patch, right after the dynamic
    variables -- the positions _append_statics puts them at. A batch without
    doy_fast is returned unchanged."""
    if "doy_fast" not in batch:
        return batch
    st = batch["x_static"]                                   # (B, H, W, S)
    for key, dkey, n_dyn in (("x_slow", "doy_slow", len(SLOW_CHANNELS)),
                             ("x_fast", "doy_fast", len(FAST_CHANNELS))):
        x, d = batch[key], batch.pop(dkey)                   # (B, T, H, W, C), (B, T, 2)
        B, T, H, W, _ = x.shape
        batch[key] = torch.cat([x[..., :n_dyn],
                                st[:, None].to(x.dtype).expand(B, T, H, W, st.shape[-1]),
                                d[:, :, None, None, :].to(x.dtype).expand(B, T, H, W, d.shape[-1]),
                                x[..., n_dyn:]], dim=-1)
    return batch


def limit_worker_threads(n: int = 1) -> None:
    """One thread per numerical library in a data worker. Forked workers
    inherit numpy's and scipy's OpenBLAS pools (64 threads each), an OpenMP
    pool (72) and Blosc's decompression threads (8); torch limits only its
    own. With 16 workers on 32 cores they oversubscribed the node: workers
    ran 34 threads each and the GPU waited about half the time (job 1831064).
    Thread counts do not change any value (tests/test_fast_inputs.py)."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(n)
    except ImportError:
        pass
    from numcodecs import blosc
    blosc.use_threads = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: give every worker its own random stream.

    Training samples are drawn from the dataset's own generator, created once
    from cfg.seed. Forked workers each got a copy of that same generator, so
    with N workers every batch came out N times and an epoch held only
    1/N distinct samples (8 workers: 250 of 2000). Each worker now draws from
    (seed, worker_id). Deterministic datasets reseed per index and are not
    affected. LEGACY_WORKER_RNG=1 keeps the old behaviour (to reproduce the
    released model's training exactly)."""
    limit_worker_threads()
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    if os.environ.get("LEGACY_WORKER_RNG", "0") == "1" or getattr(ds.cfg, "deterministic", False):
        return
    ds.rng = np.random.default_rng([int(ds.cfg.seed), int(worker_id)])


def check_checkpoint_inputs(ckpt_path: str) -> None:
    """Stop if the input settings of this process (VPD_SOURCE, USE_VPD_ANOMALY,
    PERFECT_FORECAST, read by DualPatchConfig) differ from those the checkpoint
    was trained with (its stored datamodule hyperparameters). Checkpoints from
    before these options existed count as trained with the defaults."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    trained = ck.get("datamodule_hyper_parameters") or {}
    now = DualPatchConfig(zarr_paths=(), stats_path="")
    diffs = []
    for key, default in (("vpd_source", "montes"), ("use_vpd_anomaly", False), ("perfect_forecast", False),
                         ("use_frp", False), ("use_barra_uv", False), ("slow_veg", "lai"),
                         ("use_fast_ndvi", False), ("use_fuel_age", False), ("fuel_age_lookback", 250),
                         ("use_wind_align", False), ("causal_inputs", False)):
        t = trained.get(key)
        t = default if t is None else t
        if key == "fuel_age_lookback" and not (trained.get("use_fuel_age") or False):
            continue                                    # only matters when the model uses fuel age
        if t != getattr(now, key):
            diffs.append(f"{key}: checkpoint {t!r}, this run {getattr(now, key)!r}")
    # statics and other inputs each script passes from these environment variables
    for key, env in (("use_lightning", "USE_LIGHTNING"), ("use_elevation", "USE_ELEVATION"),
                     ("use_slope_aspect", "USE_SLOPE_ASPECT"), ("use_wind_dir", "USE_WIND_DIR"),
                     ("use_ffdi", "USE_FFDI"), ("use_fmc", "USE_FMC")):
        t = bool(trained.get(key) or False)
        n = os.environ.get(env, "0") == "1"
        if t != n:
            diffs.append(f"{key}: checkpoint {t!r}, this run {env}={int(n)}")
    if diffs:
        raise ValueError(f"{ckpt_path} was trained with other inputs ({'; '.join(diffs)}); "
                         "set VPD_SOURCE / USE_VPD_ANOMALY / PERFECT_FORECAST / USE_FRP / USE_BARRA_UV / SLOW_VEG / "
                         "USE_FAST_NDVI / USE_FUEL_AGE / FUEL_AGE_LOOKBACK / USE_WIND_ALIGN / CAUSAL_INPUTS / USE_ELEVATION / USE_SLOPE_ASPECT / "
                         "USE_LIGHTNING / USE_WIND_DIR / USE_FFDI / USE_FMC to match")


class DualWindowDataset(Dataset):
    """Samples one (slow window, fast window, target) triple per __getitem__."""

    def __init__(self, cfg: DualPatchConfig):
        self.cfg = cfg
        if cfg.slow_days % cfg.slow_bin != 0:
            raise ValueError(f"slow_days {cfg.slow_days} must divide by slow_bin {cfg.slow_bin}")
        self.t_slow = cfg.slow_days // cfg.slow_bin
        self.t_fast = int(cfg.fast_days)
        self.ph = self.pw = int(cfg.patch_size)

        # ---- open every cube, build one continuous day axis ----
        self.groups, self.offsets = [], []
        off = 0
        for p in cfg.zarr_paths:
            g = open_zarr_root(p)
            self.groups.append(g)
            self.offsets.append(off)
            off += g["X"].shape[0]
        self.T_total = off
        self.offsets = np.asarray(self.offsets)

        # ---- optional pre-binned slow cube ----
        self.slow_cube = None
        if cfg.slow_cube_path:
            sc_path = cfg.slow_cube_path
            merged = (cfg.slow_lai500m_lag31_store if getattr(cfg, "causal_inputs", False)
                      else cfg.slow_lai500m_store)
            if getattr(cfg, "fast_stores", False) and not resolve(sc_path).exists():
                sc_path = merged                       # same bins and attributes, LAI at 500 m
            sc = open_zarr_root(sc_path)
            # Fail loudly on any mismatch: a silently wrong bin size or channel
            # order would corrupt training without ever raising.
            if int(sc.attrs["bin_days"]) != int(cfg.slow_bin):
                raise ValueError(
                    f"slow cube bin_days={sc.attrs['bin_days']} != cfg.slow_bin={cfg.slow_bin}")
            if tuple(sc.attrs["channels"]) != tuple(SLOW_CHANNELS):
                raise ValueError(
                    f"slow cube channels {tuple(sc.attrs['channels'])} != {SLOW_CHANNELS}")
            self.slow_cube = sc["X_slow"]
            self.slow_bin_start = np.asarray(sc["bin_start_day"][...])
            n_bins_needed = cfg.slow_days // cfg.slow_bin
            if self.slow_cube.shape[0] < n_bins_needed:
                raise ValueError(f"slow cube has {self.slow_cube.shape[0]} bins, need {n_bins_needed}")
            print(f"[dual-dataset] using pre-binned slow cube: {self.slow_cube.shape} "
                  f"({sc.attrs['channels']}, agg {sc.attrs['agg']})")

        g0 = self.groups[0]
        _, self.H, self.W, self.C = g0["X"].shape
        if self.ph > self.H or self.pw > self.W:
            raise ValueError(f"patch_size {self.ph} too large for grid {(self.H, self.W)}")

        # ---- wind (BARRA2 AUS-11, ~11km, BOM regional reanalysis) for the
        # downwind-of-recent-fire feature. Passed the gate: +2.4% AP on top of
        # the 18-feature model (elevation+slope+aspect+fuel_age), ranked 12/19
        # by SHAP -- smaller than fuel_age/elevation but real. See
        # aux_xgb_check_wind.py / firecastnet-aux-statics memory. Loaded per
        # cube-year from aux_rasters/wind_dir/barra_uv_{year}.npz (one npz per
        # calendar year, day-0 aligned with that cube's day-0).
        self.use_wind_dir = bool(getattr(cfg, "use_wind_dir", False))
        self.use_ffdi = bool(getattr(cfg, "use_ffdi", False))
        self.use_fmc = bool(getattr(cfg, "use_fmc", False))
        self.cube_year: Dict[int, int] = {
            ci: int(str(g.attrs["time"][0])[:4]) for ci, g in enumerate(self.groups)}
        self.wind_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        if self.use_wind_dir:
            for year in set(self.cube_year.values()):
                d = np.load(data_dir() / "aux_rasters" / "wind_dir" / f"barra_uv_{year}.npz")
                self.wind_cache[year] = (d["u"], d["v"], d["lat"], d["lon"])
        self.ffdi_groups = {}
        if self.use_ffdi:
            for year in set(self.cube_year.values()):
                self.ffdi_groups[year] = open_zarr_root(f"ffdi_{year}.zarr")
        self.vpd_source = str(getattr(cfg, "vpd_source", "montes"))
        self.perfect_forecast = bool(getattr(cfg, "perfect_forecast", False))
        assert self.vpd_source in ("montes", "barra"), self.vpd_source
        self.slow_veg = str(getattr(cfg, "slow_veg", "lai500"))
        if self.slow_veg != "lai500":
            raise ValueError(f"slow_veg={self.slow_veg!r}: only 'lai500' is supported (the 5 km LAI and the "
                             "NDVI variants are no longer available)")
        self.causal = bool(getattr(cfg, "causal_inputs", False))
        fast = bool(getattr(cfg, "fast_stores", False))   # inputs from build_fast_stores only
        # HiQ-LAI 500 m averaged to 1 km (build_lai500_slow) in place of the cube's LAI
        assert self.slow_cube is not None, "slow_veg needs the pre-binned slow cube"
        self.lai500_slow = None                             # fast stores: merged slow store, opened below
        self.lai_first_full_bin = 0
        lai_store = cfg.lai500_lag31_store if self.causal else cfg.lai500_slow_store
        if fast:                       # the merged slow store holds the LAI and its attributes
            lai_store = cfg.slow_lai500m_lag31_store if self.causal else cfg.slow_lai500m_store
        lg = open_zarr_root(lai_store)
        assert np.array_equal(np.asarray(lg["bin_start_day"][...]), self.slow_bin_start), "LAI500 bins differ"
        if self.causal:
            assert int(lg.attrs.get("lag_days", 0)) >= 31, (lai_store, lg.attrs.get("lag_days"))
            self.lai_first_full_bin = int(lg.attrs["first_full_bin"])
        if not fast:
            self.lai500_slow = lg["X_slow"]
        self.use_frp = bool(getattr(cfg, "use_frp", False))
        self.use_barra_uv = bool(getattr(cfg, "use_barra_uv", False))
        self.use_wind_align = bool(getattr(cfg, "use_wind_align", False))
        if self.use_frp and not fast:                        # fast stores: frp_feat of the fire store
            self.firms = open_zarr_root(getattr(cfg, "firms_store", "firms_daily.zarr"))
        if self.causal and self.vpd_source != "barra":
            raise ValueError("causal_inputs needs vpd_source='barra'")
        if self.vpd_source == "barra" or self.perfect_forecast or self.use_barra_uv or self.use_wind_align or self.causal:
            # fast stores: grid and statistics from barra_c2_fast.zarr; the full store is
            # only needed for the variables it alone holds (perfect forecast, VPD climatology)
            only_fast = fast and not (self.perfect_forecast or getattr(cfg, "use_vpd_anomaly", False))
            self.barra_fast_path = cfg.barra_fast_tmax_store if self.causal else cfg.barra_fast_store
            self.barra = open_zarr_root(self.barra_fast_path if only_fast
                                        else getattr(cfg, "barra_store", "barra_c2_daily.zarr"))
            self.barra_lat = np.asarray(self.barra["lat"][...])
            self.barra_lon = np.asarray(self.barra["lon"][...])
            self.barra_stats = dict(self.barra.attrs["stats_2015_2018"])
            if self.perfect_forecast and getattr(cfg, "augment", False):
                raise ValueError("perfect_forecast carries wind u/v, which flips/rotations do not correct")
        if self.perfect_forecast:
            alias = {"sm": "SM", "wind": "WIND", "vpd": "VPD", "precip": "PPT",
                     "lst_day": "LST", "ndvi": "NDVI", "lai": "LAI"}
            self.x_by_year = {}
            for y in range(2015, 2021):
                if resolve(daily_cube(y)).exists():
                    gy = open_zarr_root(daily_cube(y))
                    names = gy.attrs.get("channels")
                    chy = {alias[n]: i for i, n in enumerate(names)} if names else CH
                    self.x_by_year[y] = (gy["X"], chy)
        self.use_vpd_anomaly = bool(getattr(cfg, "use_vpd_anomaly", False))
        if self.use_vpd_anomaly and self.vpd_source == "barra":
            # anomaly of BARRA-C2 VPD against its own 2015-2018 climatology (build_barra_climatology)
            info = dict(self.barra.attrs["vpd_climatology"])
            assert [int(y) for y in info["years"]] == [2015, 2016, 2017, 2018], info["years"]
            self.clim_vpd = (self.barra["vpd_clim_mean"], self.barra["vpd_clim_std"])
            self.clim_anchors = np.asarray(info["anchors_doy"], np.float64)
        elif self.use_vpd_anomaly:
            cg = open_zarr_root(getattr(cfg, "clim_store", "climatology_2015_2018.zarr"))
            assert [int(y) for y in cg.attrs["years"]] == [2015, 2016, 2017, 2018], cg.attrs["years"]
            self.clim_vpd = (cg["vpd_mean"], cg["vpd_std"])
            self.clim_anchors = np.asarray(cg.attrs["anchors_doy"], np.float64)
        self.fmc_group = None
        if self.use_fmc:
            self.fmc_group = open_zarr_root(str(getattr(cfg, "fmc_store", "fmc_weekly_ff.zarr")))
            self.fmc_bin_days = int(self.fmc_group.attrs["days_per_bin"])
            self.fmc_epoch = str(self.fmc_group.attrs["epoch"])

        # ---- normalization ----
        stats = json.loads(resolve_stats(cfg.stats_path).read_text())
        x_mean = np.asarray(stats["x_mean"], dtype=np.float32)
        x_std = np.asarray(stats["x_std"], dtype=np.float32)
        x_std = np.where(x_std < 1e-6, 1.0, x_std)
        self.slow_idx = [CH[c] for c in SLOW_CHANNELS]
        self.fast_idx = [CH[c] for c in FAST_CHANNELS]
        self.slow_mean, self.slow_std = x_mean[self.slow_idx], x_std[self.slow_idx]
        self.x_mean_all, self.x_std_all = x_mean, x_std
        self.fast_mean, self.fast_std = x_mean[self.fast_idx], x_std[self.fast_idx]
        # Positions to READ from the cube's X array. Identical to the stats
        # positions for 7-channel cubes; looked up by name when the cube says
        # which channels it holds.
        names = g0.attrs.get("channels")
        if names:
            alias = {"sm": "SM", "wind": "WIND", "vpd": "VPD", "precip": "PPT",
                     "lst_day": "LST", "ndvi": "NDVI", "lai": "LAI"}
            cube_ch = {alias[n]: i for i, n in enumerate(names)}
        else:
            cube_ch = CH
        # A cube may lack channels the configured model does not read (the archived
        # lean cube holds only LST and wind speed): the slow channels when the
        # pre-binned slow cube is used, and VPD when it comes from BARRA-C2.
        # Missing channels are read as 0 and must be replaced later.
        self.read_slow_idx = [cube_ch.get(c) for c in SLOW_CHANNELS]
        self.read_fast_idx = [cube_ch.get(c) for c in FAST_CHANNELS]
        missing = [c for c, i in zip(FAST_CHANNELS, self.read_fast_idx) if i is None]
        replaced = ({"VPD"} if self.vpd_source == "barra" else set()) | ({"LST"} if self.causal else set())
        if missing and not set(missing) <= replaced:          # VPD and (causal) LST come from BARRA-C2
            raise ValueError(f"cube lacks fast-branch channels {missing}")
        if None in self.read_slow_idx and self.slow_cube is None:
            raise ValueError("cube lacks slow-branch channels and no pre-binned slow cube is given")
        self.use_fast_ndvi = bool(getattr(cfg, "use_fast_ndvi", False))
        self.read_ndvi_idx = cube_ch.get("NDVI")
        self.ndvi_comp = None
        if self.use_fast_ndvi and self.read_ndvi_idx is None:
            store = getattr(cfg, "ndvi_store", "ndvi_composites.zarr")
            if not resolve(store).exists():
                raise ValueError(f"use_fast_ndvi needs a cube with an NDVI channel or {store}")
            ng = open_zarr_root(store)
            self.ndvi_comp = ng["ndvi"]
            self.ndvi_day_to_comp = np.asarray(ng["day_to_comp"][...])
        self.agb_mean = float(stats["agb_mean"])
        self.agb_std = float(stats["agb_std"]) or 1.0

        # PPT is summed over slow_bin days, so its normalization must scale too:
        # a bin sum has ~slow_bin x the mean of a single day.
        self.slow_sum_mask = np.array(
            [SLOW_AGG[c] == "sum" for c in SLOW_CHANNELS], dtype=bool
        )
        self.slow_mean = self.slow_mean.copy()
        self.slow_std = self.slow_std.copy()
        self.slow_mean[self.slow_sum_mask] *= cfg.slow_bin
        self.slow_std[self.slow_sum_mask] *= np.sqrt(cfg.slow_bin)

        # ---- statics ----
        # landcover/koppen_geiger identical across years -> take from cube 0.
        # agb differs per year -> resolved per-sample in __getitem__.
        self.landmask = g0["landmask"][...].astype(np.uint8)
        self.landmask_f = self.landmask.astype(np.float32)
        if self.causal:                       # land cover per year from landcover_store (below)
            self.cat_stack = np.stack([np.zeros(g0["koppen_geiger"].shape, np.int64),
                                       g0["koppen_geiger"][...].astype(np.int64)], axis=-1)
        else:
            self.cat_stack = np.stack(
                [g0["landcover"][...].astype(np.int64), g0["koppen_geiger"][...].astype(np.int64)],
                axis=-1,
            )
        self._agb_cache: Dict[int, np.ndarray] = {}
        if self.causal:
            ag = open_zarr_root(cfg.agb_store)
            self.agb_years = [int(y) for y in np.asarray(ag["years"][...])]
            self.agb_maps = ag["agb"]
            self.agb_mean, self.agb_std = float(ag.attrs["mean"]), float(ag.attrs["std"]) or 1.0
            lc = open_zarr_root(cfg.landcover_store)
            self.lc_years = [int(y) for y in np.asarray(lc["years"][...])]
            self.lc_maps = lc["landcover"]
            self.kg = self.cat_stack[..., 1].copy()
            self.cat_stack = None

        # Lightning flash-density climatology (LIS/OTD HRFC, flashes/km^2/yr),
        # identical across years like landcover/koppen -> take from cube 0.
        # Added because it passed a two-regime XGBoost gate (+7.4% test AP,
        # 2nd-ranked SHAP feature) while distance-to-road/population FAILED the
        # same gate (-4 to -9% AP) -- those look like a human-ignition-density
        # shortcut that doesn't generalize; lightning is real ignition physics.
        # See aux_xgb_check.py / firecastnet-aux-statics memory.
        self.use_lightning = bool(getattr(cfg, "use_lightning", False))
        if self.use_lightning:
            lt = np.asarray(g0["lightning"], dtype=np.float32)
            lt = np.log1p(np.clip(lt, 0, None))
            self.lightning_mean = float(lt.mean())
            self.lightning_std = float(lt.std()) or 1.0
            self.lightning = ((lt - self.lightning_mean) / self.lightning_std).astype(np.float32)

        # Elevation (m, ETOPO1 1 arc-min, bilinear-resampled to grid), identical
        # across years -> take from cube 0. Passed the same XGBoost gate as
        # lightning: +5.3% test AP, ranked 8/15 by SHAP (above sm, agb, wind,
        # lai, landcover, precip). See aux_xgb_check.py / firecastnet-aux-statics
        # memory. No log transform -- elevation isn't heavy-tailed like flash
        # density, and can be negative (bathymetry over ocean, masked at use time
        # like agb).
        self.use_elevation = bool(getattr(cfg, "use_elevation", False))
        if self.use_elevation:
            el = np.asarray(g0["elevation"], dtype=np.float32)
            self.elevation_mean = float(el.mean())
            self.elevation_std = float(el.std()) or 1.0
            self.elevation = ((el - self.elevation_mean) / self.elevation_std).astype(np.float32)

        # Slope (deg) + aspect (sin/cos of compass bearing), derived from the
        # same ETOPO1 DEM as elevation -- no new external source. Passed the
        # XGBoost gate TOGETHER with fuel_age: +22.3% AP on top of elevation
        # alone. aspect_cos ("northness", solar-exposure proxy) mattered ~5x
        # more than aspect_sin, consistent with southern-hemisphere solar
        # geometry. See aux_xgb_check.py / firecastnet-aux-statics memory.
        # aspect_sin/cos are already circular-encoded at build time, no further
        # normalization needed; slope gets the usual z-score.
        self.use_slope_aspect = bool(getattr(cfg, "use_slope_aspect", False))
        if self.use_slope_aspect:
            sl = np.asarray(g0["slope"], dtype=np.float32)
            self.slope_mean = float(sl.mean())
            self.slope_std = float(sl.std()) or 1.0
            self.slope = ((sl - self.slope_mean) / self.slope_std).astype(np.float32)
            self.aspect_sin = np.asarray(g0["aspect_sin"], dtype=np.float32)
            self.aspect_cos = np.asarray(g0["aspect_cos"], dtype=np.float32)
            if self.causal:
                # The cubes' aspect_cos is the SOUTH component of the downhill direction
                # (build_aux_statics: bearing = atan2(-dZ/deast, -dZ/dsouth)); its negative is the
                # north component, so (aspect_sin, aspect_cos) is the downhill (east, north) vector
                # (cosine 0.997 with the elevation gradient; 0.148 before). Needed for the vector
                # turns of _augment, and it makes the input what its name says.
                self.aspect_cos = -self.aspect_cos

        # fuel_age: days since this pixel last had fire, leakage-free (window
        # ends at t_end-3, same cutoff as fire_hist_t-3), censored at
        # fuel_age_lookback if no fire found. DYNAMIC (depends on t_end), so it
        # lives on the fast branch alongside fire_history, not in the static
        # stack. Ranked 9th of 18 by SHAP in the gate -- right behind elevation,
        # ahead of NDVI/LAI/landcover/AGB/precip. lookback=250 matches what was
        # actually validated in aux_xgb_check.py (which stayed within one
        # cube); this datamodule CAN cross cube boundaries via _read_y_span, so
        # a longer lookback could be tried later without re-deriving the feature.
        self.use_fuel_age = bool(getattr(cfg, "use_fuel_age", False))
        self.fuel_age_lookback = int(getattr(cfg, "fuel_age_lookback", 250))
        self.fire_age = None
        fa_store = getattr(cfg, "fuel_age_store", None)
        if self.use_fuel_age and fa_store and resolve(fa_store).exists():
            self.fire_age = open_zarr_root(fa_store)["age_px"]

        # Channel index of aspect_sin within x_slow/x_fast (aspect_cos is the
        # next channel), needed by _augment to correct the ABSOLUTE compass
        # bearing under rotation/flip. Same index in both branches: both go
        # through the identical static-block-then-doy append order. Static
        # order is [agb, lm, (lightning), (elevation), (slope, aspect_sin,
        # aspect_cos)], preceded by len(slow_idx)==len(fast_idx) dynamic vars.
        self.aspect_idx = (len(self.slow_idx) + 2 + int(self.use_lightning)
                           + int(self.use_elevation) + 1) if self.use_slope_aspect else None

        # Same aspect_sin index but within the STANDALONE static stack (x_static,
        # [agb, lm, (lightning), (elevation), (slope, aspect_sin, aspect_cos)] with
        # no dynamic-var/doy prefix) -- used by the static head (permutation
        # importance during development: elevation/lightning/aspect_sin
        # contribute ~0 through the recurrent branches, most likely because they're
        # constant-over-time values diluted by broadcasting through 14-18 recurrent
        # steps, or because they're spatially near-flat within a patch and a small
        # conv kernel has little local contrast to key on. This feeds the same raw
        # statics through a separate pointwise (1x1) path straight to the fusion
        # head instead, in ADDITION to (not replacing) the existing broadcast path.
        self.static_aspect_idx = (2 + int(self.use_lightning) + int(self.use_elevation) + 1
                                   ) if self.use_slope_aspect else None

        # Channel range [fire_hist_start_idx, +n_fire_hist_channels) within
        # x_fast holding the leakage-free fire-history lags (+ distance) --
        # used by fire_history_dropout_prob (see DualPatchConfig) to blank
        # this specific shortcut signal for a fraction of TRAINING samples.
        # Same static-block-then-doy prefix width as aspect_idx above, since
        # fire-history is appended to x_fast right after that block.
        self.fire_hist_start_idx = None
        self.n_fire_hist_channels = 0
        if getattr(cfg, "fire_history", False):
            static_width = 2 + int(self.use_lightning) + int(self.use_elevation) + 3 * int(self.use_slope_aspect)
            doy_width = 2 if getattr(cfg, "add_doy", True) else 0
            self.fire_hist_start_idx = len(self.fast_idx) + static_width + doy_width
            n_lags = len(getattr(cfg, "fire_history_lags", None) or [int(getattr(cfg, "fire_history_lag", 3))])
            self.n_fire_hist_channels = n_lags + int(getattr(cfg, "fire_history_distance", False))
        # position of the fast NDVI channel: everything appended before it, in build order
        self.fast_ndvi_idx = None
        if self.use_fast_ndvi:
            self.fast_ndvi_idx = (len(self.fast_idx) + static_width + doy_width + self.n_fire_hist_channels
                                  + int(self.use_fuel_age) + int(self.use_wind_dir) + 2 * int(self.use_ffdi)
                                  + 2 * int(self.use_fmc) + 3 * int(self.use_frp) + 2 * int(self.use_barra_uv))
        self.fire_history_dropout_prob = float(getattr(cfg, "fire_history_dropout_prob", 0.0))

        # ---- compact statics: channels left out of x_slow/x_fast per sample ----
        self.compact = bool(getattr(cfg, "compact_statics", False))
        self.static_doy_width = (2 + int(self.use_lightning) + int(self.use_elevation)
                                 + 3 * int(self.use_slope_aspect) + (2 if getattr(cfg, "add_doy", True) else 0))
        if self.compact and getattr(cfg, "vslow_days", 0):
            raise ValueError("compact_statics does not support the very-slow branch")

        # ---- time-blocked input stores (build_fast_stores) ----
        self.fire_y = self.frp_feat = self.barra_fast = None
        self._bcache_key = None
        if getattr(cfg, "fast_stores", False):
            import datetime as _dt
            for ci in range(len(self.groups)):      # local day + day_offset = global day
                g_start = (_dt.date(self.cube_year[ci], 1, 1) - _dt.date(2015, 1, 1)).days
                assert int(self.offsets[ci]) + int(cfg.day_offset) == g_start, (ci, self.offsets, cfg.day_offset)
            fg = open_zarr_root(cfg.fire_store)
            if cfg.y_key == "y_fire_3d":
                self.fire_y = fg["y_fire_3d"]
            if self.use_fuel_age:
                self.fire_age = fg["age_px"]
            if self.use_frp:
                self.frp_feat = fg["frp_feat"]
            if hasattr(self, "barra"):
                bf = open_zarr_root(self.barra_fast_path)
                self.barra_fast_vars = list(bf.attrs["variables"])
                need = BARRA_FAST_VARS + (["tasmax"] if self.causal else [])
                assert set(need) <= set(self.barra_fast_vars), (need, self.barra_fast_vars)
                assert np.array_equal(np.asarray(bf["lat"][...]), self.barra_lat)
                assert np.array_equal(np.asarray(bf["lon"][...]), self.barra_lon)
                self.barra_fast = bf["x"]
                # a store cut to part of the period records the global days it holds
                self._barra_days = tuple(bf.attrs.get("days_present", (0, self.barra_fast.shape[0])))
            if self.slow_veg == "lai500":            # LAI500, SM, PPT in one array
                sg = open_zarr_root(cfg.slow_lai500m_lag31_store if self.causal else cfg.slow_lai500m_store)
                if self.causal:
                    assert int(sg.attrs.get("lag_days", 0)) >= 31, sg.attrs.get("lag_days")
                assert np.array_equal(np.asarray(sg["bin_start_day"][...]), self.slow_bin_start)
                assert tuple(sg.attrs["channels"]) == tuple(SLOW_CHANNELS), sg.attrs["channels"]
                self.slow_cube = sg["X_slow"]
                self.lai500_slow = None

        # ---- valid target days ----
        # Need slow_days of history behind the target. With a day_offset the
        # history may live *before* this split's first day (e.g. the 2019 val
        # split is preceded by 2018 in the slow cube), so only require local
        # history for what the offset does not already cover.
        slow_need = max(cfg.slow_days, int(getattr(cfg,'vslow_days',0))) + (cfg.slow_bin if self.slow_cube is not None else 0)
        if self.slow_cube is not None:
            slow_need = max(0, slow_need - int(cfg.day_offset))
        span = max(slow_need, cfg.fast_days)
        targets = np.arange(span, self.T_total)
        if cfg.valid_t_key is not None:
            ok = []
            for t in targets:
                ci, lt = self._locate(int(t))
                vt = self.groups[ci][cfg.valid_t_key]
                if vt[lt] > 0:
                    ok.append(t)
            targets = np.asarray(ok)
        if self.causal:
            # the target of issue day t_end - 1 covers days t_end .. t_end + 2: keep it inside
            # this split's own days, so no label of the next split enters training or validation
            targets = targets[targets + 2 <= self.T_total - 1]
            # the 144-day window must not reach LAI averages without a composite old enough
            if self.slow_cube is not None and self.lai_first_full_bin > 0:
                ends = np.array([self._slow_bin_end(int(t)) for t in targets])
                targets = targets[ends - self.t_slow >= self.lai_first_full_bin]
        if targets.size == 0:
            raise ValueError("No valid target days")
        self.targets = targets
        print(f"[dual-dataset] {self.T_total} days across {len(self.groups)} cubes; "
              f"{len(self.targets)} valid targets; slow {cfg.slow_days}d/{self.t_slow} steps, "
              f"fast {self.t_fast}d")

        self.rng = np.random.default_rng(cfg.seed)

    # ------------------------------------------------------------------

    def _locate(self, t: int) -> Tuple[int, int]:
        """Global day -> (cube index, local day)."""
        i = int(np.searchsorted(self.offsets, t, side="right") - 1)
        return i, t - int(self.offsets[i])

    def _issue_year(self, t_end: int) -> int:
        import datetime as _dt
        return (_dt.date(2015, 1, 1) + _dt.timedelta(days=int(t_end - 1 + self.cfg.day_offset))).year

    def _agb_causal(self, t_end: int, y0: int, x0: int) -> np.ndarray:
        """Normalized biomass map of the year before the issue day's year (2010 for 2015)."""
        y = self._issue_year(t_end) - 1
        if y not in self.agb_years:
            y = max(v for v in self.agb_years if v < y)
        a = np.asarray(self.agb_maps[self.agb_years.index(y), y0:y0 + self.ph, x0:x0 + self.pw], np.float32)
        return np.nan_to_num((a - self.agb_mean) / self.agb_std, nan=0.0, posinf=0.0, neginf=0.0)

    def _cat_causal(self, t_end: int, y0: int, x0: int) -> np.ndarray:
        """Land cover of year max(Y-2, 2015) for an issue day in year Y, and the Koppen class."""
        y = max(self._issue_year(t_end) - 2, self.lc_years[0])
        lc = np.asarray(self.lc_maps[self.lc_years.index(y), y0:y0 + self.ph, x0:x0 + self.pw]).astype(np.int64)
        return np.stack([lc, self.kg[y0:y0 + self.ph, x0:x0 + self.pw]], axis=-1)

    def _ndvi_causal(self, t_end: int, y0: int, x0: int) -> np.ndarray:
        """Raw NDVI of fast steps s = t_end - T .. t_end - 1 as shown on day s - ndvi_lag_days:
        the newest composite whose 8 days have all passed by day s. Read from the daily cube
        of whichever year holds that day."""
        lag = int(self.cfg.ndvi_lag_days)
        g = np.arange(t_end - self.t_fast, t_end) + int(self.cfg.day_offset) - lag
        if self.ndvi_comp is not None:
            comp = self.ndvi_day_to_comp[g]
            if (comp < 0).any():
                raise ValueError(f"no NDVI composite for global days {g[comp < 0].tolist()}")
            planes = {c: np.asarray(self.ndvi_comp[c, y0:y0 + self.ph, x0:x0 + self.pw], np.float32)
                      for c in np.unique(comp)}
            return np.stack([planes[c] for c in comp])
        import datetime as _dt
        if not hasattr(self, "_ndvi_cubes"):
            self._ndvi_cubes = {}
        out = np.zeros((len(g), self.ph, self.pw), np.float32)
        by_year = {}
        for k, gd in enumerate(g):
            d = _dt.date(2015, 1, 1) + _dt.timedelta(days=int(gd))
            by_year.setdefault(d.year, []).append((k, d.timetuple().tm_yday - 1))
        for year, kd in by_year.items():
            if year not in self._ndvi_cubes:
                gy = open_zarr_root(daily_cube(year))
                names = gy.attrs.get("channels")
                ch = list(names).index("ndvi") if names else CH["NDVI"]
                self._ndvi_cubes[year] = (gy["X"], ch)
            X, ch = self._ndvi_cubes[year]
            lo, hi = min(d for _, d in kd), max(d for _, d in kd)
            blk = np.asarray(X[lo:hi + 1, y0:y0 + self.ph, x0:x0 + self.pw, ch], np.float32)
            for k, d in kd:
                out[k] = blk[d - lo]
        return out

    def _agb(self, ci: int) -> np.ndarray:
        """Normalized agb for the cube owning the target day (agb is per-year)."""
        if ci not in self._agb_cache:
            a = self.groups[ci]["agb"][...].astype(np.float32)
            a = (a - self.agb_mean) / self.agb_std
            # agb is NaN over ocean (51.26% of the grid) plus ~791 constant land
            # holes. Left raw these propagate NaN through the whole batch -- the
            # v2 datamodule sanitises agb but this one did not, so every dual run
            # before 2026-07-20 trained on NaN-contaminated agb.
            self._agb_cache[ci] = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        return self._agb_cache[ci]

    def _read_span(self, t_end: int, n_days: int, ch_idx: List[int]) -> np.ndarray:
        """Read [t_end-n_days, t_end) for the given channels, crossing cubes.

        Returns (n_days, ph, pw, len(ch_idx)) -- caller slices spatially first via
        the y0/x0 closure args passed through self._cur_yx.
        """
        y0, x0 = self._cur_yx
        t_lo = t_end - n_days
        parts = []
        t = t_lo
        while t < t_end:
            ci, lt = self._locate(t)
            cube_end = int(self.offsets[ci]) + self.groups[ci]["X"].shape[0]
            take = min(t_end, cube_end) - t
            arr = np.asarray(self.groups[ci]["X"][lt:lt + take, y0:y0 + self.ph, x0:x0 + self.pw, :], np.float32)
            if None in ch_idx:                                   # channel not in this cube: 0
                part = np.zeros(arr.shape[:-1] + (len(ch_idx),), np.float32)
                for k, c in enumerate(ch_idx):
                    if c is not None:
                        part[..., k] = arr[..., c]
                parts.append(part)
            else:
                parts.append(arr[..., ch_idx])
            t += take
        return np.concatenate(parts, axis=0)

    def _read_y_span(self, t_end: int, n_days: int, key: str) -> np.ndarray:
        """Read [t_end-n_days, t_end) of a single-channel target array (e.g.
        y_fire_3d) for the current patch, crossing cube boundaries like
        _read_span. Zero-pads (no fire history) if t_end-n_days < 0 -- the very
        start of the full 2015-2020 axis has no earlier data to look at, same
        convention as fire_history's tg>=0 guard.
        """
        y0, x0 = self._cur_yx
        t_lo = t_end - n_days
        pad_before = max(0, -t_lo)
        t = max(t_lo, 0)
        parts = []
        if self.fire_y is not None and key == self.cfg.y_key and t < t_end:
            off = int(self.cfg.day_offset)          # days before the split stay zero, as below
            parts.append(np.asarray(self.fire_y[t + off:t_end + off, y0:y0 + self.ph, x0:x0 + self.pw]))
            t = t_end
        while t < t_end:
            ci, lt = self._locate(t)
            cube_end = int(self.offsets[ci]) + self.groups[ci][key].shape[0]
            take = min(t_end, cube_end) - t
            arr = self.groups[ci][key][lt:lt + take, y0:y0 + self.ph, x0:x0 + self.pw]
            parts.append(np.asarray(arr))
            t += take
        out = np.concatenate(parts, axis=0) if parts else np.zeros((0, self.ph, self.pw), np.uint8)
        pad_before = n_days - out.shape[0]      # days before the start of the data: no fire
        if pad_before > 0:
            pad = np.zeros((pad_before, self.ph, self.pw), dtype=out.dtype)
            out = np.concatenate([pad, out], axis=0)
        return out

    def _y_patch(self, t: int, y0: int, x0: int) -> np.ndarray:
        """y_key of local day t for the patch at (y0, x0)."""
        if self.fire_y is not None:
            return np.asarray(self.fire_y[t + int(self.cfg.day_offset), y0:y0 + self.ph, x0:x0 + self.pw])
        ci, lt = self._locate(t)
        return np.asarray(self.groups[ci][self.cfg.y_key][lt, y0:y0 + self.ph, x0:x0 + self.pw])

    def _patch_ok(self, y0: int, x0: int) -> bool:
        m = self.landmask[y0:y0 + self.ph, x0:x0 + self.pw]
        return float(m.mean()) >= float(self.cfg.min_valid_frac)

    def __len__(self) -> int:
        return int(self.cfg.samples_per_epoch)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        if cfg.deterministic:
            self.rng = np.random.default_rng((int(cfg.seed) << 20) ^ int(idx))
        force_pos = (cfg.pos_frac > 0.0) and (self.rng.random() < cfg.pos_frac)
        force_new_fire = (getattr(cfg, "new_fire_frac", 0.0) > 0.0) and (self.rng.random() < cfg.new_fire_frac)
        min_pos = int(cfg.min_pos_pixels)
        min_new_fire = int(getattr(cfg, "min_new_fire_pixels", 1))

        t_end = int(self.rng.choice(self.targets))
        y0 = x0 = 0
        for _ in range(int(cfg.max_tries)):
            t_end = int(self.rng.choice(self.targets))
            found = False
            for _m in range(20):
                y0 = int(self.rng.integers(0, self.H - self.ph + 1))
                x0 = int(self.rng.integers(0, self.W - self.pw + 1))
                if self._patch_ok(y0, x0):
                    found = True
                    break
            if not found:
                continue
            if not force_pos and not force_new_fire:
                break
            y_chk = self._y_patch(t_end - 1, y0, x0)      # last-step target: y_fire_3d[t_end-1]
            # Count LAND fire only: y_fire_3d labels ocean as y=1 (98.6% of ocean
            # pixels), so a raw (y>0).sum() accepts pure seawater as a "fire"
            # patch -- measured at 83% of accepted positives in the v2 sampler.
            lm_p = self.landmask[y0:y0 + self.ph, x0:x0 + self.pw]
            y_land = (y_chk > 0) & (lm_p > 0)
            if force_pos and int(y_land.sum()) < max(1, min_pos):
                continue
            if force_new_fire:
                tg3 = t_end - 1 - 3                # newest fire-history window of the last step
                if tg3 < 0:
                    continue
                recent = self._y_patch(tg3, y0, x0) > 0
                if int((y_land & ~recent).sum()) < max(1, min_new_fire):
                    continue
            break

        return self._build_sample(t_end, y0, x0)

    def sample_at(self, t_end: int, y0: int, x0: int) -> Dict[str, torch.Tensor]:
        """Sample for a given t_end (continuous day axis; the last fast step
        forecasts days t_end..t_end+2) and patch corner (y0, x0)."""
        return self._build_sample(int(t_end), int(y0), int(x0))

    # exact model-grid geotransform (label rasters; pixel size is not exactly 0.01 deg)
    _GT = (112.904998779, 0.009997566018978103, -9.005000113999998, -0.009997121616580312)

    def _barra_patch(self, var: str, gdays, y0: int, x0: int) -> np.ndarray:
        """BARRA-C2 `var` for global days `gdays`, bilinearly interpolated to the
        patch, (len(gdays), H, W) float32."""
        if self.barra_fast is not None and var in getattr(self, "barra_fast_vars", BARRA_FAST_VARS):
            return self._barra_patch_fast(var, gdays, y0, x0)
        from scipy.ndimage import map_coordinates
        c0, a, f0, e = self._GT
        lat = f0 + (np.arange(y0, y0 + self.ph) + 0.5) * e
        lon = c0 + (np.arange(x0, x0 + self.pw) + 0.5) * a
        fi = (lat - self.barra_lat[0]) / (self.barra_lat[1] - self.barra_lat[0])
        fj = (lon - self.barra_lon[0]) / (self.barra_lon[1] - self.barra_lon[0])
        r0, r1 = int(np.floor(fi.min())), int(np.ceil(fi.max())) + 1
        k0, k1 = int(np.floor(fj.min())), int(np.ceil(fj.max())) + 1
        gdays = list(gdays)
        win = np.asarray(self.barra[var][gdays[0]:gdays[-1] + 1, r0:r1 + 1, k0:k1 + 1], np.float32)
        win = win[[g - gdays[0] for g in gdays]]
        FI, FJ = np.meshgrid(fi - r0, fj - k0, indexing="ij")
        return np.stack([map_coordinates(w, [FI, FJ], order=1, mode="nearest") for w in win])

    @staticmethod
    def _linear_weights(c: np.ndarray, n: int) -> np.ndarray:
        """(len(c), n) matrix of 1-D linear interpolation weights at the
        fractional positions c, edges extended (map_coordinates order=1,
        mode="nearest")."""
        A = np.zeros((c.size, n), np.float64)
        if n == 1:
            A[:, 0] = 1.0
            return A
        c = np.clip(c, 0.0, n - 1.0)
        i0 = np.minimum(np.floor(c).astype(np.int64), n - 2)
        w = c - i0
        rows = np.arange(c.size)
        A[rows, i0] = 1.0 - w
        A[rows, i0 + 1] += w
        return A

    def _barra_patch_fast(self, var: str, gdays, y0: int, x0: int) -> np.ndarray:
        """_barra_patch from barra_c2_fast.zarr: one read of vpd, uas and vas
        for the day range (kept for the next variable of the same sample), and
        bilinear interpolation as A_lat @ field @ A_lon^T."""
        gdays = list(gdays)
        lo, hi = self._barra_days
        if not (lo <= gdays[0] and gdays[-1] < hi):
            raise ValueError(f"BARRA days {gdays[0]}..{gdays[-1]} outside {self.cfg.barra_fast_store} "
                             f"(holds global days {lo}..{hi - 1})")
        key = (gdays[0], gdays[-1], y0, x0)
        if self._bcache_key != key:
            c0, a, f0, e = self._GT
            fi = (f0 + (np.arange(y0, y0 + self.ph) + 0.5) * e - self.barra_lat[0]) / (self.barra_lat[1] - self.barra_lat[0])
            fj = (c0 + (np.arange(x0, x0 + self.pw) + 0.5) * a - self.barra_lon[0]) / (self.barra_lon[1] - self.barra_lon[0])
            r0, r1 = int(np.floor(fi.min())), int(np.ceil(fi.max())) + 1
            k0, k1 = int(np.floor(fj.min())), int(np.ceil(fj.max())) + 1
            win = np.asarray(self.barra_fast[gdays[0]:gdays[-1] + 1, r0:r1 + 1, k0:k1 + 1, :], np.float32)
            self._bcache = (win, self._linear_weights(fi - r0, win.shape[1]),
                            self._linear_weights(fj - k0, win.shape[2]), fi - r0, fj - k0)
            self._bcache_key = key
        win, Ar, Ak, fi, fj = self._bcache
        vi = (self.barra_fast_vars if hasattr(self, "barra_fast_vars") else BARRA_FAST_VARS).index(var)
        w = win[[g - gdays[0] for g in gdays], :, :, vi]
        if not np.isfinite(w).all():                 # a product would spread a NaN over the whole patch
            from scipy.ndimage import map_coordinates
            FI, FJ = np.meshgrid(fi, fj, indexing="ij")
            return np.stack([map_coordinates(d, [FI, FJ], order=1, mode="nearest") for d in w])
        return (Ar[None] @ w.astype(np.float64) @ Ak.T[None]).astype(np.float32)

    def _wind_align(self, t_end, y0, x0, steps, recent_fire, lm_p) -> np.ndarray:
        """(T, H, W, 1): cos(angle between the day's 10 m wind and the direction
        from the nearest recent fire to the pixel), see DualPatchConfig.use_wind_align."""
        from scipy import ndimage as _ndi
        T = len(steps)
        g0 = t_end - T + int(self.cfg.day_offset)
        u = np.nan_to_num(self._barra_patch("uas", range(g0, g0 + T), y0, x0))     # eastward
        v = np.nan_to_num(self._barra_patch("vas", range(g0, g0 + T), y0, x0))     # northward
        rr, cc = np.meshgrid(np.arange(self.ph), np.arange(self.pw), indexing="ij")
        out = np.zeros((T, self.ph, self.pw, 1), np.float32)
        for j, s_j in enumerate(steps):
            fire = recent_fire(s_j)
            if not fire.any():
                continue
            dist, (nr, nc) = _ndi.distance_transform_edt(~fire, return_indices=True)
            de = (cc - nc).astype(np.float32)                   # fire -> pixel, east
            dn = (nr - rr).astype(np.float32)                   # fire -> pixel, north (rows grow southward)
            norm = np.hypot(de, dn) * np.hypot(u[j], v[j])
            a = (u[j] * de + v[j] * dn) / np.maximum(norm, 1e-6)
            out[j, :, :, 0] = np.where((dist > 0) & (norm > 1e-6), a, 0.0) * (lm_p > 0)
        return out

    def _barra_static(self, name: str, y0: int, x0: int) -> np.ndarray:
        """A (k, lat, lon) array of the BARRA store, bilinearly interpolated to the patch."""
        from scipy.ndimage import map_coordinates
        c0, a, f0, e = self._GT
        fi = (f0 + (np.arange(y0, y0 + self.ph) + 0.5) * e - self.barra_lat[0]) / (self.barra_lat[1] - self.barra_lat[0])
        fj = (c0 + (np.arange(x0, x0 + self.pw) + 0.5) * a - self.barra_lon[0]) / (self.barra_lon[1] - self.barra_lon[0])
        r0, r1 = int(np.floor(fi.min())), int(np.ceil(fi.max())) + 1
        k0, k1 = int(np.floor(fj.min())), int(np.ceil(fj.max())) + 1
        win = np.asarray(self.barra[name][:, r0:r1 + 1, k0:k1 + 1], np.float32)
        FI, FJ = np.meshgrid(fi - r0, fj - k0, indexing="ij")
        return np.stack([map_coordinates(w, [FI, FJ], order=1, mode="nearest") for w in win])

    def _cube_days(self, ch: str, gdays, y0: int, x0: int) -> np.ndarray:
        """Cube channel `ch` (name, raw units) for global days, any year; zeros where no cube."""
        import datetime as _dt
        out = np.zeros((len(gdays), self.ph, self.pw), np.float32)
        by_year = {}                                     # one read per year: each chunk decompressed once
        for k, g in enumerate(gdays):
            d = _dt.date(2015, 1, 1) + _dt.timedelta(days=int(g))
            by_year.setdefault(d.year, []).append((k, d.timetuple().tm_yday - 1))
        for year, kd in by_year.items():
            if year not in self.x_by_year:
                continue
            X, chy = self.x_by_year[year]
            lo, hi = min(d for _, d in kd), max(d for _, d in kd)
            blk = np.asarray(X[lo:hi + 1, y0:y0 + self.ph, x0:x0 + self.pw, chy[ch]], np.float32)
            for k, d in kd:
                out[k] = blk[d - lo]
        return np.nan_to_num(out)

    def _perfect_forecast(self, t_end: int, y0: int, x0: int) -> np.ndarray:
        """(T, H, W, 5): weather of days s+1..s+3 for every fast step s (see
        DualPatchConfig.perfect_forecast). Reads only weather stores."""
        T = self.t_fast
        g_first = t_end - T + int(self.cfg.day_offset)            # issue day of step 0
        gdays = list(range(g_first + 1, g_first + T + 3))         # s+1 of step 0 .. s+3 of step T-1
        st = self.barra_stats
        wmax = np.nan_to_num(self._barra_patch("sfcWindmax", gdays, y0, x0))
        u = np.nan_to_num(self._barra_patch("uas", gdays, y0, x0))
        v = np.nan_to_num(self._barra_patch("vas", gdays, y0, x0))
        if self.vpd_source == "barra":
            vpd = np.nan_to_num(self._barra_patch("vpd", gdays, y0, x0))
        else:
            vpd = self._cube_days("VPD", gdays, y0, x0)
        ppt = self._cube_days("PPT", gdays, y0, x0)
        iv, ip = CH["VPD"], CH["PPT"]
        out = np.empty((T, self.ph, self.pw, 5), np.float32)
        for j in range(T):
            w = slice(j, j + 3)                                   # days s_j+1 .. s_j+3
            out[j, ..., 0] = (wmax[w].max(0) - st["sfcWindmax"][0]) / st["sfcWindmax"][1]
            out[j, ..., 1] = (u[w].mean(0) - st["uas"][0]) / st["uas"][1]
            out[j, ..., 2] = (v[w].mean(0) - st["vas"][0]) / st["vas"][1]
            if self.vpd_source == "barra":
                out[j, ..., 3] = (vpd[w].max(0) - st["vpd"][0]) / st["vpd"][1]
            else:
                out[j, ..., 3] = (vpd[w].max(0) - self.x_mean_all[iv]) / self.x_std_all[iv]
            out[j, ..., 4] = (ppt[w].sum(0) - 3 * self.x_mean_all[ip]) / (np.sqrt(3) * self.x_std_all[ip])
        return out

    def _vpd_anomaly(self, vpd_raw: np.ndarray, t_end: int, y0: int, x0: int) -> np.ndarray:
        """(T, H, W) standardized VPD anomaly of fast days t_end-T .. t_end-1."""
        import datetime as _dt
        sl = (slice(None), slice(y0, y0 + self.ph), slice(x0, x0 + self.pw))
        if self.vpd_source == "barra":                           # native grid -> patch
            mu, sd = self._barra_static("vpd_clim_mean", y0, x0), self._barra_static("vpd_clim_std", y0, x0)
        else:
            mu = np.asarray(self.clim_vpd[0][sl], np.float32)
            sd = np.asarray(self.clim_vpd[1][sl], np.float32)
        a = np.r_[self.clim_anchors, 366.0]
        out = np.empty(vpd_raw.shape, np.float32)
        for j in range(vpd_raw.shape[0]):
            g = t_end - vpd_raw.shape[0] + j + int(self.cfg.day_offset)
            doy = min((_dt.date(2015, 1, 1) + _dt.timedelta(days=int(g))).timetuple().tm_yday, 365)
            k0 = int(np.searchsorted(a, doy, side="right") - 1)
            f = (doy - a[k0]) / (a[k0 + 1] - a[k0])
            k1 = (k0 + 1) % len(self.clim_anchors)
            m = mu[k0] * (1 - f) + mu[k1] * f
            s = np.maximum(sd[k0] * (1 - f) + sd[k1] * f, 0.01)
            out[j] = (vpd_raw[j] - m) / s
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -6.0, 6.0)

    def channel_layout(self) -> Dict[str, Dict[str, List[int]]]:
        """Channel indices of every input group in x_slow and x_fast, in the
        order _build_sample appends them (checked against each sample)."""
        cfg = self.cfg
        stat = ["biomass", "landmask"] + (["lightning"] if self.use_lightning else []) \
            + (["elevation"] if self.use_elevation else []) \
            + (["slope", "aspect", "aspect"] if self.use_slope_aspect else [])
        doy = ["day of year"] * (2 if getattr(cfg, "add_doy", True) else 0)
        slow = list(SLOW_CHANNELS) + stat + doy
        fast = [("TMAX" if (c == "LST" and self.causal) else c) for c in FAST_CHANNELS] + stat + doy
        if getattr(cfg, "fire_history", False):
            fast += ["fire history"] * self.n_fire_hist_channels
        fast += ["fuel age"] * int(self.use_fuel_age) + ["wind direction (BARRA2)"] * int(self.use_wind_dir) \
            + ["FFDI"] * (2 * int(self.use_ffdi)) + ["FMC"] * (2 * int(self.use_fmc)) \
            + ["fire radiative power"] * (3 * int(self.use_frp)) + ["wind u/v"] * (2 * int(self.use_barra_uv)) \
            + ["NDVI (fast)"] * int(self.use_fast_ndvi) + ["VPD anomaly"] * int(self.use_vpd_anomaly) \
            + ["perfect forecast"] * (5 * int(self.perfect_forecast)) \
            + ["downwind alignment"] * int(self.use_wind_align)
        out = {}
        for key, names in (("slow", slow), ("fast", fast)):
            d: Dict[str, List[int]] = {}
            for i, nm in enumerate(names):
                d.setdefault(nm, []).append(i)
            out[key] = d
        out["n"] = {"slow": len(slow), "fast": len(fast)}
        return out

    def _slow_bin_end(self, t_end: int) -> int:
        """Exclusive index of the newest slow bin used for a sample ending at
        t_end (issue day t_end - 1). Fixed: bins that end on or before the
        issue day. Legacy: bins that start on or before t_end (reaches 1-8
        days past the issue day)."""
        t_global = t_end + int(self.cfg.day_offset)
        if getattr(self.cfg, "slow_window_legacy", False):
            return int(np.searchsorted(self.slow_bin_start, t_global, side="right"))
        return int(np.searchsorted(self.slow_bin_start, t_global - self.cfg.slow_bin, side="right"))

    def _build_sample(self, t_end: int, y0: int, x0: int) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        self._cur_yx = (y0, x0)
        ci_t, lt_t = self._locate(t_end)

        # ---- slow branch ----
        if self.slow_cube is not None:
            # Pre-binned on a FIXED GLOBAL grid: take the t_slow bins that end on
            # or before the issue day D = t_end - 1 (bin b covers days
            # start_b .. start_b + slow_bin - 1). The newest bin can be up to
            # slow_bin-1 days stale, which is irrelevant for variables peaking at
            # lag 130-150 -- and it is what lets bins be shared across targets.
            # map this split's local day onto the cube's global 2015-2020 axis
            t_global = t_end + int(cfg.day_offset)
            b_end = self._slow_bin_end(t_end)
            b_lo = b_end - self.t_slow
            if b_lo < 0:
                raise IndexError(
                    f"target day {t_end} (global {t_global}) has only {b_end} slow bins of history")
            x_slow = np.asarray(
                self.slow_cube[b_lo:b_end, y0:y0 + self.ph, x0:x0 + self.pw, :], dtype=np.float32
            )
        else:
            # Fallback: aggregate raw days on the fly (correct, ~22x slower).
            raw_slow = self._read_span(t_end, cfg.slow_days, self.read_slow_idx)
            raw_slow = np.nan_to_num(raw_slow, nan=0.0, posinf=0.0, neginf=0.0)
            binned = raw_slow.reshape(self.t_slow, cfg.slow_bin, self.ph, self.pw, len(self.slow_idx))
            x_slow = np.where(
                self.slow_sum_mask[None, None, None, :],
                binned.sum(axis=1),
                binned.mean(axis=1),
            ).astype(np.float32)
        x_slow = np.nan_to_num(x_slow, nan=0.0, posinf=0.0, neginf=0.0)
        x_slow = (x_slow - self.slow_mean[None, None, None, :]) / self.slow_std[None, None, None, :]
        if self.lai500_slow is not None:                            # same bins, same LAI normalization
            il = SLOW_CHANNELS.index("LAI")                         # (fast stores: already channel 0)
            lv = np.asarray(self.lai500_slow[b_lo:b_end, y0:y0 + self.ph, x0:x0 + self.pw, 0], np.float32)
            x_slow[..., il] = (np.nan_to_num(lv) - self.slow_mean[il]) / self.slow_std[il]

        # ---- very-slow branch: year-scale accumulated drought ----
        x_vslow = None
        if getattr(cfg, "vslow_days", 0) and self.slow_cube is not None:
            grp = int(cfg.vslow_bin // cfg.slow_bin)      # 24/8 = 3 source bins per step
            n_v = int(cfg.vslow_days // cfg.vslow_bin)    # 432/24 = 18 steps
            vb_end = b_end
            vb_lo = vb_end - n_v*grp
            if vb_lo < 0:
                raise IndexError(f"target day {t_end} lacks {cfg.vslow_days}d of history")
            raw = np.asarray(self.slow_cube[vb_lo:vb_end, y0:y0+self.ph, x0:x0+self.pw, :],
                             dtype=np.float32)
            raw = raw.reshape(n_v, grp, self.ph, self.pw, raw.shape[-1])
            # PPT bins are sums -> sum again; others are means -> mean again
            x_vslow = np.where(self.slow_sum_mask[None,None,None,:],
                               raw.sum(axis=1), raw.mean(axis=1)).astype(np.float32)
            x_vslow = np.nan_to_num(x_vslow, nan=0.0, posinf=0.0, neginf=0.0)
            vm = self.slow_mean.copy(); vs = self.slow_std.copy()
            vm[self.slow_sum_mask] *= grp; vs[self.slow_sum_mask] *= np.sqrt(grp)
            x_vslow = (x_vslow - vm[None,None,None,:]) / vs[None,None,None,:]

        # ---- fast branch: daily ----
        ndvi_raw = None
        if self.use_fast_ndvi and self.causal:                    # completed composites only
            x_fast = self._read_span(t_end, cfg.fast_days, self.read_fast_idx)
            ndvi_raw = self._ndvi_causal(t_end, y0, x0)
        elif self.use_fast_ndvi and self.ndvi_comp is None:       # NDVI in the same pass over the cube
            xr = self._read_span(t_end, cfg.fast_days, self.read_fast_idx + [self.read_ndvi_idx])
            x_fast, ndvi_raw = xr[..., :-1], xr[..., -1]
        elif self.use_fast_ndvi:                                   # cube without NDVI: composite store
            x_fast = self._read_span(t_end, cfg.fast_days, self.read_fast_idx)
            g = np.arange(t_end - cfg.fast_days, t_end) + int(cfg.day_offset)
            comp = self.ndvi_day_to_comp[g]
            if (comp < 0).any():
                raise ValueError(f"no NDVI composite for global days {g[comp < 0].tolist()} in {cfg.ndvi_store}")
            planes = {c: np.asarray(self.ndvi_comp[c, y0:y0 + self.ph, x0:x0 + self.pw], np.float32)
                      for c in np.unique(comp)}
            ndvi_raw = np.stack([planes[c] for c in comp])
        else:
            x_fast = self._read_span(t_end, cfg.fast_days, self.read_fast_idx)
        x_fast = np.nan_to_num(x_fast, nan=0.0, posinf=0.0, neginf=0.0)
        vpd_raw = x_fast[..., FAST_CHANNELS.index("VPD")].copy() if self.use_vpd_anomaly else None
        if self.vpd_source == "barra":
            iv = FAST_CHANNELS.index("VPD")
            g0 = t_end - self.t_fast + int(cfg.day_offset)
            bv = self._barra_patch("vpd", range(g0, g0 + self.t_fast), y0, x0)
            x_fast[..., iv] = (np.nan_to_num(bv) - self.barra_stats["vpd"][0]) / self.barra_stats["vpd"][1] \
                * self.fast_std[iv] + self.fast_mean[iv]          # undone by the normalization below
            if self.use_vpd_anomaly:
                vpd_raw = np.nan_to_num(bv)
        if self.causal:                                           # BARRA-C2 tmax of day s in the LST slot
            it = FAST_CHANNELS.index("LST")
            g0 = t_end - self.t_fast + int(cfg.day_offset)
            bt = self._barra_patch("tasmax", range(g0, g0 + self.t_fast), y0, x0)
            x_fast[..., it] = (np.nan_to_num(bt, nan=self.barra_stats["tasmax"][0]) - self.barra_stats["tasmax"][0]) \
                / self.barra_stats["tasmax"][1] * self.fast_std[it] + self.fast_mean[it]   # undone below
        x_fast = (x_fast - self.fast_mean[None, None, None, :]) / self.fast_std[None, None, None, :]

        # ---- target: fast branch is daily, so y aligns with its axis ----
        if self.fire_y is not None:
            y = self._read_y_span(t_end, self.t_fast, cfg.y_key).astype(np.float32)
        else:
            y = np.empty((self.t_fast, self.ph, self.pw), dtype=np.float32)
            for j, t in enumerate(range(t_end - self.t_fast, t_end)):
                ci, lt = self._locate(t)
                y[j] = self.groups[ci][cfg.y_key][lt, y0:y0 + self.ph, x0:x0 + self.pw]

        # ---- statics appended to BOTH branches (agb from the target's year) ----
        agb_p = self._agb_causal(t_end, y0, x0) if self.causal else self._agb(ci_t)[y0:y0 + self.ph, x0:x0 + self.pw]
        lm_p = self.landmask_f[y0:y0 + self.ph, x0:x0 + self.pw]
        static_list = [agb_p, lm_p]
        if self.use_lightning:
            static_list.append(self.lightning[y0:y0 + self.ph, x0:x0 + self.pw])
        if self.use_elevation:
            static_list.append(self.elevation[y0:y0 + self.ph, x0:x0 + self.pw])
        if self.use_slope_aspect:
            static_list.append(self.slope[y0:y0 + self.ph, x0:x0 + self.pw])
            static_list.append(self.aspect_sin[y0:y0 + self.ph, x0:x0 + self.pw])
            static_list.append(self.aspect_cos[y0:y0 + self.ph, x0:x0 + self.pw])
        st = np.stack(static_list, axis=-1)  # (ph,pw,2..7) depending on optional statics

        doy_sc = {}

        def _append_statics(x, n_steps, day_starts):
            sc = np.zeros((n_steps, 0), np.float32)
            if cfg.add_doy:
                doy = (np.asarray(day_starts, dtype=np.float32) + 1.0) / 365.25
                ang = 2.0 * np.pi * doy
                sc = np.stack([np.sin(ang), np.cos(ang)], axis=-1).astype(np.float32)
            if self.compact:                   # inserted on the GPU by expand_compact
                doy_sc[len(doy_sc)] = sc
                return x
            extra = [np.repeat(st[None], n_steps, axis=0)]
            if cfg.add_doy:
                sc = np.broadcast_to(sc[:, None, None, :], (n_steps, self.ph, self.pw, 2))
                extra.append(np.ascontiguousarray(sc))
            return np.concatenate([x] + extra, axis=-1)

        # DOY uses each step's representative day (bin start for slow, day for fast).
        # With the fixed grid the slow steps are the actual bin start days, not a
        # target-relative arange -- they differ by up to slow_bin-1 days.
        if self.slow_cube is not None:
            slow_days_axis = self.slow_bin_start[b_lo:b_end].astype(np.float32)
        else:
            slow_days_axis = np.arange(t_end - cfg.slow_days, t_end, cfg.slow_bin)
        fast_days_axis = np.arange(t_end - cfg.fast_days, t_end)
        x_slow = _append_statics(x_slow, self.t_slow, slow_days_axis)
        if x_vslow is not None:
            n_v = x_vslow.shape[0]
            v_axis = self.slow_bin_start[b_end-n_v*int(cfg.vslow_bin//cfg.slow_bin):b_end:
                                         int(cfg.vslow_bin//cfg.slow_bin)].astype(np.float32)
            x_vslow = _append_statics(x_vslow, n_v, v_axis)
        x_fast = _append_statics(x_fast, self.t_fast, fast_days_axis)

        # ---- fire history on the FAST branch, per timestep ----
        # y_fire_3d[s] = fire on days s+1..s+3, and fast step j is trained on
        # target y_fire_3d[s_j] with s_j = t_end - t_fast + j. Its history uses
        # y_fire_3d[s_j - L] for L in fire_history_lags; L >= 3 means the newest
        # history window ends on day s_j, the day before that step's target
        # window starts, so no step ever sees fire from inside its own target.
        if getattr(cfg, "fire_history", False):
            lags = list(getattr(cfg, "fire_history_lags", None) or [int(getattr(cfg, "fire_history_lag", 3))])
            if min(lags) < 3:
                raise ValueError(f"fire_history_lags must all be >= 3, got {lags}")
            T = self.t_fast
            first = t_end - T - max(lags)            # oldest history index needed (step 0)
            span_end = t_end - 1 - min(lags) + 1     # exclusive end (last step, smallest lag)
            span = (self._read_y_span(span_end, span_end - first, cfg.y_key) > 0).astype(np.float32)
            span *= lm_p[None]
            fh = np.empty((T, self.ph, self.pw, len(lags)), np.float32)
            for j in range(T):
                s_j = t_end - T + j
                for k, lg in enumerate(lags):
                    fh[j, :, :, k] = span[s_j - lg - first]
            chans = [fh]
            if getattr(cfg, "fire_history_distance", False):
                from scipy import ndimage
                dist = np.zeros((T, self.ph, self.pw, 1), np.float32)
                for j in range(T):
                    recent = fh[j, :, :, 0] > 0.5
                    if recent.any():
                        dist[j, :, :, 0] = np.exp(-ndimage.distance_transform_edt(~recent) / 5.0)
                chans.append(dist)
            x_fast = np.concatenate([x_fast] + chans, axis=-1).astype(np.float32)

        # ---- optional per-step features --------------------------------------
        # Fast step j is issued on day s_j = t_end - t_fast + j and trained on
        # y_fire_3d[s_j] (fire on days s_j+1..s_j+3). Every feature below uses
        # only information up to day s_j: fire from y_fire_3d[<= s_j-3], wind
        # and FFDI on day s_j, fuel moisture from the last complete week.
        T = self.t_fast
        steps = [t_end - T + j for j in range(T)]

        fuel_age_idx = None

        def _recent_fire(s_j):
            if s_j - 3 < 0:
                return np.zeros((self.ph, self.pw), bool)
            return (self._y_patch(s_j - 3, y0, x0) > 0) & (lm_p > 0)

        if self.use_fuel_age and self.fire_age is not None:
            LB = self.fuel_age_lookback
            g0 = steps[0] - 3 + int(cfg.day_offset)            # newest history index of step 0, global
            ages = np.asarray(self.fire_age[g0:g0 + T, y0:y0 + self.ph, x0:x0 + self.pw], np.float32)
            ages = np.minimum(np.where(ages == 65535, LB, ages), LB)        # never burned / older: capped
            fuel_age_idx = x_fast.shape[-1]
            x_fast = np.concatenate([x_fast, ((ages - LB / 2.0) / (LB / 2.0))[..., None]], axis=-1)
        elif self.use_fuel_age:
            LB = self.fuel_age_lookback
            e0 = steps[0] - 3                                  # newest history index of step 0
            hist = (self._read_y_span(e0 + 1, LB, cfg.y_key) > 0) & (lm_p > 0)[None]   # y_fire_3d[e0-LB+1 .. e0], land only
            idx = np.arange(LB)[:, None, None]
            last = np.where(hist, idx, -10 ** 6).max(axis=0) + (e0 - LB + 1)   # absolute index
            fa = np.empty((T, self.ph, self.pw, 1), np.float32)
            for j, s_j in enumerate(steps):
                e = s_j - 3
                if j > 0:
                    last = np.where(_recent_fire(s_j), e, last)
                age = np.minimum(e - last, LB).astype(np.float32)
                fa[j, :, :, 0] = (age - LB / 2.0) / (LB / 2.0)
            fuel_age_idx = x_fast.shape[-1]
            x_fast = np.concatenate([x_fast, fa], axis=-1)

        if self.use_wind_dir:
            from scipy import ndimage as _ndi
            lat_c = GRID_LAT0 - (y0 + self.ph / 2) * GRID_PX
            lon_c = GRID_LON0 + (x0 + self.pw / 2) * GRID_PX
            rr, cc = np.meshgrid(np.arange(self.ph), np.arange(self.pw), indexing="ij")
            wd = np.zeros((T, self.ph, self.pw, 1), np.float32)
            for j, s_j in enumerate(steps):
                fire = _recent_fire(s_j)
                if not fire.any():
                    continue
                dist_px, (near_r, near_c) = _ndi.distance_transform_edt(~fire, return_indices=True)
                vs = (rr - near_r).astype(np.float32); ve = (cc - near_c).astype(np.float32)
                vn = np.sqrt(vs ** 2 + ve ** 2) + 1e-6
                ci_w, lt_w = self._locate(s_j)
                u_all, v_all, wlat, wlon = self.wind_cache[self.cube_year[ci_w]]
                li = int(np.abs(wlat - lat_c).argmin()); lj = int(np.abs(wlon - lon_c).argmin())
                uw = float(u_all[lt_w, li, lj]); vw = float(v_all[lt_w, li, lj])
                wn = (uw ** 2 + vw ** 2) ** 0.5 + 1e-6
                wd[j, :, :, 0] = ((-vw / wn) * vs / vn + (uw / wn) * ve / vn) * (dist_px > 0)
            x_fast = np.concatenate([x_fast, wd], axis=-1)

        if self.use_ffdi:
            fd = np.zeros((T, self.ph, self.pw, 2), np.float32)
            for j, s_j in enumerate(steps):
                ci_f, lt_f = self._locate(s_j)
                gfd = self.ffdi_groups[self.cube_year[ci_f]]
                if lt_f < gfd["ffdi"].shape[0]:
                    fd[j, :, :, 0] = np.asarray(gfd["ffdi"][lt_f, y0:y0 + self.ph, x0:x0 + self.pw], np.float32) / 50.0
                    fd[j, :, :, 1] = np.asarray(gfd["df"][lt_f, y0:y0 + self.ph, x0:x0 + self.pw], np.float32) / 250.0
            x_fast = np.concatenate([x_fast, np.nan_to_num(fd)], axis=-1)

        if self.use_fmc:
            import datetime as _dt
            ep0 = _dt.date.fromisoformat(self.fmc_epoch)
            fm = np.zeros((T, self.ph, self.pw, 2), np.float32)
            cache = {}
            for j, s_j in enumerate(steps):
                ci_f, lt_f = self._locate(s_j)
                day = _dt.date(self.cube_year[ci_f], 1, 1) + _dt.timedelta(days=int(lt_f))
                mi = (day - ep0).days // self.fmc_bin_days - 1     # last COMPLETE week before day s_j
                if not 0 <= mi < self.fmc_group["fmc"].shape[0]:
                    continue
                if mi not in cache:
                    raw = np.asarray(self.fmc_group["fmc"][mi, y0:y0 + self.ph, x0:x0 + self.pw])
                    seen = raw != 255
                    stale = np.asarray(self.fmc_group["staleness"][mi, y0:y0 + self.ph, x0:x0 + self.pw]).astype(np.float32)
                    cache[mi] = (np.where(seen, raw / 100.0, 0.0),
                                 np.where(seen, 1.0 - np.minimum(stale, 12.0) / 12.0, 0.0))
                fm[j, :, :, 0], fm[j, :, :, 1] = cache[mi]
            x_fast = np.concatenate([x_fast, fm], axis=-1)

        frp_slice = None
        if self.use_frp and self.frp_feat is not None:      # per-day features over s_j-2..s_j
            off = int(cfg.day_offset)
            fr = np.asarray(self.frp_feat[t_end - T + off:t_end + off, y0:y0 + self.ph, x0:x0 + self.pw], np.float32)
            frp_slice = slice(x_fast.shape[-1], x_fast.shape[-1] + 3)
            x_fast = np.concatenate([x_fast, fr * lm_p[None, :, :, None]], axis=-1)
        elif self.use_frp:
            g_lo = t_end - T - 2 + int(cfg.day_offset)            # step 0 needs s_0-2 .. s_0
            g_hi = t_end + int(cfg.day_offset)                    # exclusive: step T-1 ends on t_end-1
            sl = (slice(g_lo, g_hi), slice(y0, y0 + self.ph), slice(x0, x0 + self.pw))
            frp = np.asarray(self.firms["frp_sum"][sl], np.float32)
            nd = np.asarray(self.firms["n_det"][sl], np.float32)
            nn = np.asarray(self.firms["n_night"][sl], np.float32)
            fr = np.empty((T, self.ph, self.pw, 3), np.float32)
            for j in range(T):
                w = slice(j, j + 3)                               # days s_j-2 .. s_j
                f_, n_, m_ = frp[w].sum(0), nd[w].sum(0), nn[w].sum(0)
                fr[j, ..., 0] = np.log1p(f_) / 5.0
                fr[j, ..., 1] = np.log1p(n_) / 3.0
                fr[j, ..., 2] = np.where(n_ > 0, m_ / np.maximum(n_, 1), 0.0)
            frp_slice = slice(x_fast.shape[-1], x_fast.shape[-1] + 3)
            x_fast = np.concatenate([x_fast, fr * lm_p[None, :, :, None]], axis=-1)

        if self.use_barra_uv:
            g0 = t_end - T + int(cfg.day_offset)
            bst = self.barra_stats                                # not `st`: that name holds the statics
            u = np.nan_to_num(self._barra_patch("uas", range(g0, g0 + T), y0, x0))
            v = np.nan_to_num(self._barra_patch("vas", range(g0, g0 + T), y0, x0))
            x_fast = np.concatenate([x_fast, ((u - bst["uas"][0]) / bst["uas"][1])[..., None],
                                     ((v - bst["vas"][0]) / bst["vas"][1])[..., None]], axis=-1)

        if self.use_fast_ndvi:
            nd = np.nan_to_num(ndvi_raw)
            nd = (nd - self.x_mean_all[CH["NDVI"]]) / self.x_std_all[CH["NDVI"]]
            assert x_fast.shape[-1] == self.fast_ndvi_idx - self.compact * self.static_doy_width, \
                (x_fast.shape, self.fast_ndvi_idx)
            x_fast = np.concatenate([x_fast, nd[..., None].astype(np.float32)], axis=-1)

        if self.use_vpd_anomaly:
            x_fast = np.concatenate([x_fast, self._vpd_anomaly(vpd_raw, t_end, y0, x0)[..., None]], axis=-1)

        if self.perfect_forecast:
            x_fast = np.concatenate([x_fast, self._perfect_forecast(t_end, y0, x0)], axis=-1)

        wind_align_idx = None
        if self.use_wind_align:
            wind_align_idx = x_fast.shape[-1]
            x_fast = np.concatenate([x_fast, self._wind_align(t_end, y0, x0, steps, _recent_fire, lm_p)], axis=-1)

        # Fire-history feature dropout (TRAIN only): blank fire_hist_t-3/4/5 +
        # fire_dist to 0 for this sample with probability fire_history_dropout_
        # prob, forcing the network to predict from weather/terrain alone when
        # it triggers. 0 is a physically valid "no recent fire nearby" state
        # for both channel types (fire_hist is 0/1, fire_dist = exp(-dist/5)
        # already ->0 far from any fire), not a corruption. See DualPatchConfig.
        if (self.fire_history_dropout_prob > 0.0 and self.fire_hist_start_idx is not None
                and not cfg.deterministic and self.rng.random() < self.fire_history_dropout_prob):
            i0 = self.fire_hist_start_idx - self.compact * self.static_doy_width
            x_fast[..., i0:i0 + self.n_fire_hist_channels] = 0.0
            if frp_slice is not None:
                x_fast[..., frp_slice] = 0.0
            if fuel_age_idx is not None:                         # ages of a few days are fire history too
                x_fast[..., fuel_age_idx] = 1.0                   # = the cap: "long unburned"
            if wind_align_idx is not None:                       # derived from the fire history
                x_fast[..., wind_align_idx] = 0.0

        if not hasattr(self, "_layout_n"):
            self._layout_n = self.channel_layout()["n"]
        cw = self.compact * self.static_doy_width
        assert (x_slow.shape[-1], x_fast.shape[-1]) == (self._layout_n["slow"] - cw, self._layout_n["fast"] - cw), \
            (x_slow.shape, x_fast.shape, self._layout_n, cw)
        out = {
            "x_slow": torch.from_numpy(np.ascontiguousarray(x_slow)),
            "x_fast": torch.from_numpy(np.ascontiguousarray(x_fast)),
            "y": torch.from_numpy(y),
            "x_cat": torch.from_numpy(np.ascontiguousarray(
                self._cat_causal(t_end, y0, x0) if self.causal
                else self.cat_stack[y0:y0 + self.ph, x0:x0 + self.pw, :])),
            "mask": torch.from_numpy(lm_p.copy()),
            "x_static": torch.from_numpy(np.ascontiguousarray(st)),
            "t_end": torch.tensor(t_end, dtype=torch.int64),
            "y0": torch.tensor(y0, dtype=torch.int64),
            "x0": torch.tensor(x0, dtype=torch.int64),
        }
        if self.compact:
            out["doy_slow"] = torch.from_numpy(doy_sc[0])
            out["doy_fast"] = torch.from_numpy(doy_sc[1])
        if x_vslow is not None:
            out["x_vslow"] = torch.from_numpy(np.ascontiguousarray(x_vslow))
        if getattr(cfg, "past_fire_dist_store", None):
            if not hasattr(self, "_past_dist"):
                self._past_dist = open_zarr_root(cfg.past_fire_dist_store)["dist_px"]
            # step j has issue day s_j = t_end - t_fast + j and reads index s_j - 3
            g_hi = t_end - 3 + int(cfg.day_offset)          # exclusive end: step T-1 reads t_end - 4
            pd_ = np.asarray(self._past_dist[g_hi - self.t_fast:g_hi, y0:y0 + self.ph, x0:x0 + self.pw],
                             dtype=np.float32)
            out["past_dist"] = torch.from_numpy(np.ascontiguousarray(pd_))

        if getattr(cfg, "augment", False) and not cfg.deterministic:
            self._augment(out)
        return out

    def _augment(self, out: Dict[str, torch.Tensor]) -> None:
        """In-place random flip (east-west mirror) + k*90deg rotation of every
        spatial tensor, mirroring zarr_daily_datamodule_v2.py's _augment.

        Two features need explicit handling beyond the plain spatial rot90/
        flip, because a naive transform would silently teach the model wrong
        wind/terrain associations (a known risk for directional features):

        - aspect_sin/aspect_cos (east and north component of the direction the
          slope faces) and the BARRA-C2 wind components u (east), v (north) are
          vectors: they turn with the patch. torch.rot90(k=1) over (H, W) turns
          the image counter-clockwise (rows grow southward), so each rotation
          maps (east, north) -> (-north, east); the east-west mirror maps
          east -> -east. Composed in the order of the array transform (k
          rotations, then the flip); checked against the gradient of a rotated
          field in tests/test_augment_vectors.py. (Before 2026-10-06 the aspect
          was turned clockwise, wrong for k = 1 and 3; no model since then used
          augmentation.)
        - downwind_align is deliberately NOT corrected: it's already a
          RELATIVE angle (cos of the angle between the wind vector and the
          fire->pixel vector), and both of those vectors live in the patch's
          local frame -- they rotate/reflect together under any orthogonal
          transform applied identically to both, so their relative angle
          (and therefore downwind_align) is invariant by construction. The
          plain spatial rot90/flip on the whole tensor already moves the
          VALUE to the correct pixel; no value-level correction is needed.
        """
        k = int(self.rng.integers(0, 4))
        do_flip = bool(self.rng.integers(0, 2))

        def rot_flip(t: torch.Tensor, dims: Tuple[int, int]) -> torch.Tensor:
            t = torch.rot90(t, k, dims=dims)
            if do_flip:
                t = torch.flip(t, dims=(dims[1],))
            return t.contiguous()

        out["x_slow"] = rot_flip(out["x_slow"], (1, 2))       # (T,H,W,C)
        out["x_fast"] = rot_flip(out["x_fast"], (1, 2))
        if "x_vslow" in out:
            out["x_vslow"] = rot_flip(out["x_vslow"], (1, 2))
        out["y"] = rot_flip(out["y"], (-2, -1))                # (T,H,W)
        if "past_dist" in out:
            out["past_dist"] = rot_flip(out["past_dist"], (-2, -1))
        out["mask"] = rot_flip(out["mask"], (-2, -1))          # (H,W)
        out["x_cat"] = rot_flip(out["x_cat"], (0, 1))          # (H,W,ncat)
        out["x_static"] = rot_flip(out["x_static"], (0, 1))   # (H,W,C_static)

        def turn(e, n):
            """(east, north) components after k counter-clockwise quarter turns, then the mirror."""
            for _ in range(k):
                e, n = -n, e
            if do_flip:
                e = -e
            return e, n

        if self.aspect_idx is not None and not self.compact:   # compact: statics only in x_static
            i = self.aspect_idx
            for key in ("x_slow", "x_fast"):
                e, n = turn(out[key][..., i].clone(), out[key][..., i + 1].clone())
                out[key][..., i] = e; out[key][..., i + 1] = n

        if self.static_aspect_idx is not None:
            i = self.static_aspect_idx
            e, n = turn(out["x_static"][..., i].clone(), out["x_static"][..., i + 1].clone())
            out["x_static"][..., i] = e; out["x_static"][..., i + 1] = n

        if self.use_barra_uv:                                    # normalized u, v: turn the physical vector
            iu = self.channel_layout()["fast"]["wind u/v"][0] - self.compact * self.static_doy_width
            bst = self.barra_stats
            u = out["x_fast"][..., iu] * bst["uas"][1] + bst["uas"][0]
            v = out["x_fast"][..., iu + 1] * bst["vas"][1] + bst["vas"][0]
            u, v = turn(u, v)
            out["x_fast"][..., iu] = (u - bst["uas"][0]) / bst["uas"][1]
            out["x_fast"][..., iu + 1] = (v - bst["vas"][0]) / bst["vas"][1]


class DualDataModule(pl.LightningDataModule):
    """Train/val split by *year*, mirroring DailyDataModuleV2 (train 2015-2018,
    val 2019), but each split gets its own continuous axis so slow windows can
    cross year boundaries within a split."""

    def __init__(
        self,
        train_paths: List[str],
        val_paths: List[str],
        stats_path: str,
        slow_cube_path: Optional[str] = None,
        y_key: str = "y_fire_3d",
        valid_key: Optional[str] = "y_fire_3d_valid",
        slow_days: int = 144,
        slow_bin: int = 8,
        fast_days: int = 14,
        patch_size: int = 256,
        samples_per_epoch: int = 2000,
        batch_size: int = 2,
        num_workers: int = 8,
        min_valid_frac: float = 0.45,
        fire_history: bool = False,
        fire_history_lags: Optional[Tuple[int, ...]] = None,
        fire_history_distance: bool = False,
        vslow_days: int = 0,
        vslow_bin: int = 24,
        use_lightning: bool = False,
        use_elevation: bool = False,
        use_slope_aspect: bool = False,
        use_fuel_age: bool = False,
        fuel_age_lookback: int = 250,
        use_wind_dir: bool = False,
        use_ffdi: bool = False,
        use_fmc: bool = False,
        fmc_store: str = "fmc_weekly_ff.zarr",
        augment: bool = False,
        fire_history_dropout_prob: float = 0.0,
        new_fire_frac: float = 0.0,
        min_new_fire_pixels: int = 1,
        past_fire_dist_store: Optional[str] = None,
        use_vpd_anomaly: bool = False,
        use_frp: bool = False,
        use_barra_uv: bool = False,
        use_wind_align: bool = False,
        slow_veg: str = "lai500",
        use_fast_ndvi: bool = False,
        causal_inputs: bool = True,
        vpd_source: str = "montes",
        perfect_forecast: bool = False,
        compact_statics: bool = False,   # statics/day of year inserted on the GPU (expand_compact)
        min_pos_pixels: int = 20,   # scale with patch area to hold fire-DENSITY fixed
                                     # across patch-size experiments: 20/256^2 = 0.0305%
        train_seed: int = 123,      # override for multi-seed noise-floor checks;
                                     # val_ds seed stays fixed at 999 so the val set
                                     # itself never changes across seed reruns.
    ):
        super().__init__()
        self.save_hyperparameters()
        self.h = self.hparams

    def _day_offset(self, paths) -> int:
        """First day of `paths` on the slow cube's global axis.

        The slow cube is built once over every year in build_slow_cube.py's YEARS
        order, but each split only opens its own cubes and numbers days from 0.
        Derive the offset from the years preceding this split rather than
        hardcoding it, so it stays correct if the split changes.
        """
        if not self.h.slow_cube_path:
            return 0
        sc = open_zarr_root(self.h.slow_cube_path)
        cube_years = [int(y) for y in sc.attrs["years"]]
        first = int(Path(str(paths[0])).stem.split("_")[-1])
        off = 0
        for y in cube_years:
            if y == first:
                return off
            off += 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
        raise ValueError(f"year {first} not in slow cube years {cube_years}")

    def setup(self, stage: Optional[str] = None):
        h = self.h
        self.train_ds = DualWindowDataset(DualPatchConfig(
            zarr_paths=tuple(h.train_paths), stats_path=h.stats_path,
            slow_cube_path=h.slow_cube_path,
            day_offset=self._day_offset(h.train_paths),
            y_key=h.y_key, valid_t_key=h.valid_key,
            slow_days=h.slow_days, slow_bin=h.slow_bin, fast_days=h.fast_days,
            patch_size=h.patch_size, samples_per_epoch=h.samples_per_epoch,
            seed=h.train_seed, min_valid_frac=h.min_valid_frac,
            min_pos_pixels=h.min_pos_pixels, max_tries=1000, pos_frac=0.5,
            new_fire_frac=h.new_fire_frac, min_new_fire_pixels=h.min_new_fire_pixels,
            fire_history=h.fire_history, fire_history_lags=h.fire_history_lags,
            fire_history_distance=h.fire_history_distance,
            vslow_days=h.vslow_days, vslow_bin=h.vslow_bin,
            use_lightning=h.use_lightning,
            use_elevation=h.use_elevation,
            use_slope_aspect=h.use_slope_aspect,
            use_fuel_age=h.use_fuel_age,
            fuel_age_lookback=h.fuel_age_lookback,
            use_wind_dir=h.use_wind_dir,
            use_ffdi=h.use_ffdi, use_fmc=h.use_fmc, fmc_store=h.fmc_store,
            augment=h.augment,
            fire_history_dropout_prob=h.fire_history_dropout_prob,
            past_fire_dist_store=h.past_fire_dist_store,
            use_vpd_anomaly=h.use_vpd_anomaly,
            vpd_source=h.vpd_source, perfect_forecast=h.perfect_forecast,
            use_frp=h.use_frp, use_barra_uv=h.use_barra_uv, use_wind_align=h.use_wind_align,
            slow_veg=h.slow_veg, use_fast_ndvi=h.use_fast_ndvi,
            causal_inputs=h.get("causal_inputs", False),
            compact_statics=h.compact_statics,
        ))
        self.val_ds = DualWindowDataset(DualPatchConfig(
            zarr_paths=tuple(h.val_paths), stats_path=h.stats_path,
            slow_cube_path=h.slow_cube_path,
            day_offset=self._day_offset(h.val_paths),
            y_key=h.y_key, valid_t_key=h.valid_key,
            slow_days=h.slow_days, slow_bin=h.slow_bin, fast_days=h.fast_days,
            patch_size=h.patch_size,
            samples_per_epoch=max(1024, h.samples_per_epoch // 4),
            seed=999, min_valid_frac=h.min_valid_frac,
            min_pos_pixels=0, max_tries=2000, pos_frac=0.0,
            deterministic=True,
            fire_history=h.fire_history, fire_history_lags=h.fire_history_lags,
            fire_history_distance=h.fire_history_distance,
            vslow_days=h.vslow_days, vslow_bin=h.vslow_bin,
            use_lightning=h.use_lightning,
            use_elevation=h.use_elevation,
            use_slope_aspect=h.use_slope_aspect,
            use_fuel_age=h.use_fuel_age,
            fuel_age_lookback=h.fuel_age_lookback,
            use_wind_dir=h.use_wind_dir,
            use_ffdi=h.use_ffdi, use_fmc=h.use_fmc, fmc_store=h.fmc_store,
            augment=False,   # val is always deterministic/unaugmented, regardless of h.augment
            use_vpd_anomaly=h.use_vpd_anomaly,
            vpd_source=h.vpd_source, perfect_forecast=h.perfect_forecast,
            use_frp=h.use_frp, use_barra_uv=h.use_barra_uv, use_wind_align=h.use_wind_align,
            slow_veg=h.slow_veg, use_fast_ndvi=h.use_fast_ndvi,
            causal_inputs=h.get("causal_inputs", False),
            fire_history_dropout_prob=0.0,  # val must see the real fire-history signal always
            compact_statics=h.compact_statics,
        ))

    def _dl(self, ds, shuffle=False):
        return DataLoader(
            ds, batch_size=self.h.batch_size, shuffle=shuffle,
            num_workers=self.h.num_workers, pin_memory=True,
            persistent_workers=self.h.num_workers > 0, worker_init_fn=seed_worker,
        )

    def train_dataloader(self):
        return self._dl(self.train_ds)

    def val_dataloader(self):
        return self._dl(self.val_ds)
