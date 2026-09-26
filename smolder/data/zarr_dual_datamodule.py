"""SMOLDER data pipeline: patches with a slow and a fast temporal window.

For a target day t and a PATCH x PATCH window, each sample holds
  x_slow  (18, H, W, C)  LAI, soil moisture, precipitation over 144 days in
                         8-day bins (from the pre-binned cube_slow_8day.zarr),
                         plus broadcast statics and day-of-year (sin, cos)
  x_fast  (14, H, W, C)  VPD, land-surface temperature, wind over 14 days,
                         plus the same statics, day-of-year, and fire history
                         (fire at t-3, t-4, t-5 and exp(-distance/5 px) to fire
                         at t-3; nothing later than t-3 is ever read)
  x_cat   (H, W, 2)      land cover and Koppen-Geiger class (embedded in-model)
  y       (14, H, W)     next-3-day fire occurrence for every fast-window day
  mask    (H, W)         land mask

All cubes are indexed on one continuous 2015-2020 day axis, so windows can
cross year boundaries. Inputs are standardised with channel_stats_2015_2018.json
(training years only). NDVI is present in the full cubes but not read.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl

from smolder.data.io import data_dir, open_zarr_root, resolve_stats

# Channel order of the 7-channel training cubes and of channel_stats_2015_2018.json:
# [sm, wind, vpd, precip, lst_day, ndvi, lai]. Cubes that carry a `channels`
# attribute (e.g. the NDVI-free Zenodo archive) are read by name instead.
CH = {"SM": 0, "WIND": 1, "VPD": 2, "PPT": 3, "LST": 4, "NDVI": 5, "LAI": 6}

# NDVI (5) deliberately absent: redundant with LAI, see module docstring.
SLOW_CHANNELS = ("LAI", "SM", "PPT")
FAST_CHANNELS = ("VPD", "LST", "WIND")

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
    use_fuel_age: bool = False
    fuel_age_lookback: int = 250
    # Downwind-of-recent-fire wind alignment (BARRA2 AUS-11). +2.4% AP on top
    # of the above, ranked 12/19 by SHAP -- see aux_xgb_check_wind.py.
    use_wind_dir: bool = False
    # Random flip (east-west mirror) + k*90deg rotation, TRAIN ONLY (val_ds
    # uses deterministic=True, unaffected regardless of this flag). Was never
    # implemented for this datamodule (only the old v1/v2 single-branch one
    # had it) -- flagged in CLAUDE.md's backlog specifically because of the
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
            sc = open_zarr_root(cfg.slow_cube_path)
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
        self.wind_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self.cube_year: Dict[int, int] = {}
        if self.use_wind_dir:
            wind_dir = str(data_dir() / "aux_rasters" / "wind_dir")
            for ci, p in enumerate(cfg.zarr_paths):
                year = int(Path(str(p)).stem.split("_")[-1])
                self.cube_year[ci] = year
                if year not in self.wind_cache:
                    npz_path = os.path.join(wind_dir, f"barra_uv_{year}.npz")
                    d = np.load(npz_path)
                    self.wind_cache[year] = (d["u"], d["v"], d["lat"], d["lon"])

        # ---- normalization ----
        stats = json.loads(resolve_stats(cfg.stats_path).read_text())
        x_mean = np.asarray(stats["x_mean"], dtype=np.float32)
        x_std = np.asarray(stats["x_std"], dtype=np.float32)
        x_std = np.where(x_std < 1e-6, 1.0, x_std)
        self.slow_idx = [CH[c] for c in SLOW_CHANNELS]
        self.fast_idx = [CH[c] for c in FAST_CHANNELS]
        self.slow_mean, self.slow_std = x_mean[self.slow_idx], x_std[self.slow_idx]
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
        self.read_slow_idx = [cube_ch[c] for c in SLOW_CHANNELS]
        self.read_fast_idx = [cube_ch[c] for c in FAST_CHANNELS]
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
        self.cat_stack = np.stack(
            [g0["landcover"][...].astype(np.int64), g0["koppen_geiger"][...].astype(np.int64)],
            axis=-1,
        )
        self._agb_cache: Dict[int, np.ndarray] = {}

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
        # no dynamic-var/doy prefix) -- used by the static head (see CLAUDE.md
        # 2026-07-25 permutation-importance finding: elevation/lightning/aspect_sin
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
        self.fire_history_dropout_prob = float(getattr(cfg, "fire_history_dropout_prob", 0.0))

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
            arr = self.groups[ci]["X"][lt:lt + take, y0:y0 + self.ph, x0:x0 + self.pw, :]
            parts.append(np.asarray(arr, dtype=np.float32)[..., ch_idx])
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
        while t < t_end:
            ci, lt = self._locate(t)
            cube_end = int(self.offsets[ci]) + self.groups[ci][key].shape[0]
            take = min(t_end, cube_end) - t
            arr = self.groups[ci][key][lt:lt + take, y0:y0 + self.ph, x0:x0 + self.pw]
            parts.append(np.asarray(arr))
            t += take
        out = np.concatenate(parts, axis=0) if parts else np.zeros((0, self.ph, self.pw))
        if pad_before > 0:
            pad = np.zeros((pad_before, self.ph, self.pw), dtype=out.dtype)
            out = np.concatenate([pad, out], axis=0)
        return out

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
            ci, lt = self._locate(t_end - 1)      # last-step target: y_fire_3d[t_end-1]
            y_chk = self.groups[ci][cfg.y_key][lt, y0:y0 + self.ph, x0:x0 + self.pw]
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
                ci3, lt3 = self._locate(tg3)
                recent = np.asarray(self.groups[ci3][cfg.y_key][lt3, y0:y0 + self.ph, x0:x0 + self.pw]) > 0
                if int((y_land & ~recent).sum()) < max(1, min_new_fire):
                    continue
            break

        self._cur_yx = (y0, x0)
        ci_t, lt_t = self._locate(t_end)

        # ---- slow branch ----
        if self.slow_cube is not None:
            # Pre-binned on a FIXED GLOBAL grid: take the t_slow bins ending at or
            # before t_end. The newest bin can be up to slow_bin-1 days stale,
            # which is irrelevant for variables peaking at lag 130-150 -- and it
            # is what lets bins be shared across targets (and thus precomputed).
            # map this split's local day onto the cube's global 2015-2020 axis
            t_global = t_end + int(cfg.day_offset)
            b_end = int(np.searchsorted(self.slow_bin_start, t_global, side="right"))
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
        x_fast = self._read_span(t_end, cfg.fast_days, self.read_fast_idx)
        x_fast = np.nan_to_num(x_fast, nan=0.0, posinf=0.0, neginf=0.0)
        x_fast = (x_fast - self.fast_mean[None, None, None, :]) / self.fast_std[None, None, None, :]

        # ---- target: fast branch is daily, so y aligns with its axis ----
        y = np.empty((self.t_fast, self.ph, self.pw), dtype=np.float32)
        for j, t in enumerate(range(t_end - self.t_fast, t_end)):
            ci, lt = self._locate(t)
            y[j] = self.groups[ci][cfg.y_key][lt, y0:y0 + self.ph, x0:x0 + self.pw]

        # ---- statics appended to BOTH branches (agb from the target's year) ----
        agb_p = self._agb(ci_t)[y0:y0 + self.ph, x0:x0 + self.pw]
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

        def _append_statics(x, n_steps, day_starts):
            extra = [np.repeat(st[None], n_steps, axis=0)]
            if cfg.add_doy:
                doy = (np.asarray(day_starts, dtype=np.float32) + 1.0) / 365.25
                ang = 2.0 * np.pi * doy
                sc = np.stack([np.sin(ang), np.cos(ang)], axis=-1)
                sc = np.broadcast_to(sc[:, None, None, :], (n_steps, self.ph, self.pw, 2))
                extra.append(np.ascontiguousarray(sc.astype(np.float32)))
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

        # ---- fuel age: days since this pixel last had fire (leakage-free) ----
        # Different from fire_history above (short-range, 3-5 day proximity-in-
        # time): this is a long-range fuel-accumulation proxy. Window ends at
        # t_end-3 (same leakage cutoff), scans fuel_age_lookback days back,
        # censored at the lookback if no fire found. Passed the XGBoost gate
        # together with slope/aspect: +22.3% AP on top of elevation alone;
        # ranked 9/18 by SHAP, right behind elevation. See aux_xgb_check.py.
        if self.use_fuel_age:
            LB = self.fuel_age_lookback
            hist = self._read_y_span(t_end - 3, LB, cfg.y_key) > 0   # ends at y_fire_3d[t_end-4]
            idx = np.arange(hist.shape[0])[:, None, None]
            last_idx = np.where(hist, idx, -1).max(axis=0)
            fuel_age = np.where(last_idx >= 0, (hist.shape[0] - 1 - last_idx), LB).astype(np.float32)
            fuel_age_n = (fuel_age - LB / 2.0) / (LB / 2.0)           # roughly [-1, 1]
            fa_fast = np.broadcast_to(fuel_age_n[None, :, :, None],
                                       (self.t_fast, self.ph, self.pw, 1))
            x_fast = np.concatenate([x_fast, np.ascontiguousarray(fa_fast.astype(np.float32))], axis=-1)

        # ---- downwind-of-recent-fire wind alignment ----
        # +2.4% AP on top of the 18-feature model (elevation+slope+aspect+
        # fuel_age), ranked 12/19 by SHAP -- smaller than fuel_age/elevation
        # but real. Uses BARRA2 AUS-11 (~11km BOM regional reanalysis) wind at
        # t_end and the same lag-3 "recent fire" mask fire_history uses.
        # downwind_align in [-1,1]: +1 = wind blows FROM recent fire directly
        # TOWARD this pixel. See aux_xgb_check_wind.py.
        if self.use_wind_dir:
            tg3 = t_end - 1 - 3
            ci_h3, lt_h3 = self._locate(tg3)
            fh3_wd = (np.asarray(self.groups[ci_h3][cfg.y_key][lt_h3, y0:y0 + self.ph, x0:x0 + self.pw]) > 0)
            from scipy import ndimage as _ndi
            if fh3_wd.any():
                dist_px, (near_r, near_c) = _ndi.distance_transform_edt(~fh3_wd, return_indices=True)
                rr, cc = np.meshgrid(np.arange(self.ph), np.arange(self.pw), indexing="ij")
                vec_south = (rr - near_r).astype(np.float32)
                vec_east = (cc - near_c).astype(np.float32)
                vnorm = np.sqrt(vec_south ** 2 + vec_east ** 2) + 1e-6
                vec_south /= vnorm; vec_east /= vnorm
            else:
                vec_south = np.zeros((self.ph, self.pw), np.float32)
                vec_east = np.zeros((self.ph, self.pw), np.float32)
                dist_px = np.full((self.ph, self.pw), 999.0, np.float32)

            year_t = self.cube_year[ci_t]
            u_all, v_all, wlat, wlon = self.wind_cache[year_t]
            lat_c = GRID_LAT0 - (y0 + self.ph / 2) * GRID_PX
            lon_c = GRID_LON0 + (x0 + self.pw / 2) * GRID_PX
            li = int(np.abs(wlat - lat_c).argmin()); lj = int(np.abs(wlon - lon_c).argmin())
            uw = float(u_all[lt_t, li, lj]); vw = float(v_all[lt_t, li, lj])   # eastward, northward m/s
            wnorm = (uw ** 2 + vw ** 2) ** 0.5 + 1e-6
            wind_south = -vw / wnorm; wind_east = uw / wnorm   # met v=northward -> south = -v
            downwind_align = (wind_south * vec_south + wind_east * vec_east).astype(np.float32)
            downwind_align = np.where(dist_px < 900, downwind_align, 0.0)
            wd_fast = np.broadcast_to(downwind_align[None, :, :, None],
                                       (self.t_fast, self.ph, self.pw, 1))
            x_fast = np.concatenate([x_fast, np.ascontiguousarray(wd_fast.astype(np.float32))], axis=-1)

        # Fire-history feature dropout (TRAIN only): blank fire_hist_t-3/4/5 +
        # fire_dist to 0 for this sample with probability fire_history_dropout_
        # prob, forcing the network to predict from weather/terrain alone when
        # it triggers. 0 is a physically valid "no recent fire nearby" state
        # for both channel types (fire_hist is 0/1, fire_dist = exp(-dist/5)
        # already ->0 far from any fire), not a corruption. See DualPatchConfig.
        if (self.fire_history_dropout_prob > 0.0 and self.fire_hist_start_idx is not None
                and not cfg.deterministic and self.rng.random() < self.fire_history_dropout_prob):
            i0 = self.fire_hist_start_idx
            x_fast[..., i0:i0 + self.n_fire_hist_channels] = 0.0

        out = {
            "x_slow": torch.from_numpy(np.ascontiguousarray(x_slow)),
            "x_fast": torch.from_numpy(np.ascontiguousarray(x_fast)),
            "y": torch.from_numpy(y),
            "x_cat": torch.from_numpy(
                np.ascontiguousarray(self.cat_stack[y0:y0 + self.ph, x0:x0 + self.pw, :])
            ),
            "mask": torch.from_numpy(lm_p.copy()),
            "x_static": torch.from_numpy(np.ascontiguousarray(st)),
            "t_end": torch.tensor(t_end, dtype=torch.int64),
            "y0": torch.tensor(y0, dtype=torch.int64),
            "x0": torch.tensor(x0, dtype=torch.int64),
        }
        if x_vslow is not None:
            out["x_vslow"] = torch.from_numpy(np.ascontiguousarray(x_vslow))

        if getattr(cfg, "augment", False) and not cfg.deterministic:
            self._augment(out)
        return out

    def _augment(self, out: Dict[str, torch.Tensor]) -> None:
        """In-place random flip (east-west mirror) + k*90deg rotation of every
        spatial tensor, mirroring zarr_daily_datamodule_v2.py's _augment.

        Two features need explicit handling beyond the plain spatial rot90/
        flip, because a naive transform would silently teach the model wrong
        wind/terrain associations (the exact risk CLAUDE.md's backlog flagged):

        - aspect_sin/aspect_cos encode an ABSOLUTE compass bearing (direction
          the slope faces relative to true North, which does NOT rotate with
          the patch) -- these get an explicit sin/cos correction on top of the
          spatial rearrangement, composed in the SAME order as the array
          transform (k rotations, then the flip).
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
        out["mask"] = rot_flip(out["mask"], (-2, -1))          # (H,W)
        out["x_cat"] = rot_flip(out["x_cat"], (0, 1))          # (H,W,ncat)
        out["x_static"] = rot_flip(out["x_static"], (0, 1))   # (H,W,C_static)

        if self.aspect_idx is not None:
            i = self.aspect_idx
            for key in ("x_slow", "x_fast"):
                s = out[key][..., i].clone(); c = out[key][..., i + 1].clone()
                for _ in range(k):
                    s, c = c, -s                # +90deg: sin(t+90)=cos(t), cos(t+90)=-sin(t)
                if do_flip:
                    s = -s                      # east-west mirror: sin(-t)=-sin(t), cos(-t)=cos(t)
                out[key][..., i] = s; out[key][..., i + 1] = c

        if self.static_aspect_idx is not None:
            i = self.static_aspect_idx
            s = out["x_static"][..., i].clone(); c = out["x_static"][..., i + 1].clone()
            for _ in range(k):
                s, c = c, -s
            if do_flip:
                s = -s
            out["x_static"][..., i] = s; out["x_static"][..., i + 1] = c


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
        augment: bool = False,
        fire_history_dropout_prob: float = 0.0,
        new_fire_frac: float = 0.0,
        min_new_fire_pixels: int = 1,
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
            augment=h.augment,
            fire_history_dropout_prob=h.fire_history_dropout_prob,
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
            augment=False,   # val is always deterministic/unaugmented, regardless of h.augment
            fire_history_dropout_prob=0.0,  # val must see the real fire-history signal always
        ))

    def _dl(self, ds, shuffle=False):
        return DataLoader(
            ds, batch_size=self.h.batch_size, shuffle=shuffle,
            num_workers=self.h.num_workers, pin_memory=True,
            persistent_workers=self.h.num_workers > 0,
        )

    def train_dataloader(self):
        return self._dl(self.train_ds)

    def val_dataloader(self):
        return self._dl(self.val_ds)
