# FireCastNet — wildfire risk forecasting (ConvLSTM)

Australia-wide per-pixel wildfire forecasting: predict fire occurrence in the
**next 3 days** (`y_fire_3d`) from daily predictor maps. Grid 3474×4110
(~1 km, smips grid, EPSG:4326, 0.01° px, origin lon 112.904998779 /
lat -9.005000114).

**Current architecture (dual-branch, unified across landcover):**
`ConvLSTMLitDual` runs two separate ConvLSTM encoders — a **slow** branch
(LAI/SM/PPT, 144d @ 8-day bins, from a pre-binned `cube_slow_8day.zarr`) and a
**fast** branch (VPD/LST/WIND, 14d daily) — fused via FireSenseNet-style
cross-attention (`fuse="cross_attn"`). Landcover is fed in as an embedding
(not a per-class split — that was the old v1/v2 approach, now abandoned: a
single unified model beats 5 separate per-landcover models). Model code:
`seasfire/backbones/conv_lstm.py` (`ConvLSTMSegDual`), `conv_lstm_lit_dual.py`.
Datamodule: `zarr_dual_datamodule.py`. Training: `train_convlstm_dual_fastproto.py`
(despite the name, this is now the production training script — see State
2026-07-25 below for why "fastproto" stuck as the filename).

The older single-branch, 30-day-window, per-landcover-split pipeline
(`conv_lstm_lit.py`/`_v2.py`, `zarr_daily_datamodule_v2.py`,
`train_convlstm_30day_*.py`) still exists and still works, but is no longer
the model being improved — see State 2026-07-14 for its history.

## Layout
- Data cubes: `cube_daily_smgrid_YYYY.zarr` (2015–2020) in this dir.
  `X (T,H,W,7)` dyn channels `[sm, wind, vpd, precip, lst_day, ndvi, lai]`.
  Statics: `agb, landmask, landcover, koppen_geiger` (original) +
  `lightning, elevation, slope, aspect_sin, aspect_cos` (added 2026-07-25, see
  below). Targets `y_fire_3d(+_valid)`, `y_fire_8d`. Train 2015–2018, val 2019,
  **test 2020** (the true holdout — 2019 is validation/model-selection, not test).
- Pre-binned slow cube: `cube_slow_8day.zarr` (`build_slow_cube.py`) — 8-day
  LAI/SM/PPT bins on a fixed global 2015–2020 day axis, 22× faster to read
  than aggregating 144 raw days per sample on the fly.
- Aux static rasters: `aux_rasters/` (`.npy` on the model grid) +
  `aux_rasters/wind_dir/` (`barra_uv_{year}.npz`, dynamic). Built by
  `build_aux_statics.py`. See firecastnet-aux-statics memory for provenance
  and the two rejected features (roads, population).
- Model: `seasfire/backbones/conv_lstm.py`, `conv_lstm_lit.py` (base Lightning
  module: loss schedules, isolation-weighted loss, pos_weight annealing),
  `conv_lstm_lit_v2.py` (deep supervision + cosine LR), `conv_lstm_lit_dual.py`
  (dual-branch, inherits v2).
- Datamodules: `zarr_daily_datamodule_v2.py` (old single-branch v1/v2),
  `zarr_dual_datamodule.py` (current dual-branch: per-channel normalization
  from `channel_stats_2015_2018.json`, DOY sin/cos, fire-history lags +
  distance, fuel_age, flip/rot augmentation with directional-feature
  correctness, all 6 new statics/features, continuous 2015–2020 day axis).
- Training: `train_convlstm_dual_fastproto.py` (current, despite the name —
  supports full production scale via `SAMPLES_PER_EPOCH`/`MAX_EPOCHS` env
  overrides), `train_convlstm_dual.py` (older, no-focal/anneal/isolation
  overrides not wired), `train_convlstm_30day_v2.py` (old single-branch).
  Checkpoints → `/home/saturn/gwgi/gwgi107h/wildfire_data/checkpoints/<run_tag>/job_<id>/`,
  metrics CSV in `lightning_logs/`. **Selection metric: `val_ap` (Average
  Precision)**, not `val_topk_f1_composite_cond` — that composite sits at the
  noise floor under this data's extreme imbalance and kept near-random
  checkpoints early on; see Pitfalls.
- Python: `/home/hpc/gwgi/gwgi107h/.conda/envs/firecastnet_py310/bin/python`.
- GPU jobs: `ssh tinyx.nhr.fau.de "sbatch ..."` (TinyGPU, partition a100; ~3-job
  concurrency in practice). Current submit script: `submit_train_dual_fastproto.sbatch`
  (env: `VARIANT=fixed|baseline`, `PATCH_SIZE`, `SAMPLES_PER_EPOCH`, `MAX_EPOCHS`,
  `BATCH_SIZE`/`ACCUM`, `FOCAL_START_EPOCH`, `RESUME_CKPT`).

## State as of 2026-07-25

### Best checkpoints (2020 test-set numbers, not val)
| Tag | Patch | val_ap (2019) | 2020 test AUC-PR | 2020 test ROC-AUC | New-fire peak lift |
|---|---|---|---|---|---|
| `dual_fh_attn` job_1755283 (old, superseded) | 256px | 0.2790 | 0.3664 | 0.8604 | 4.4x @2% |
| `dual_fh_attn_ps128_final` job_1760963 | 128px | 0.4383 (converged, see below) | 0.4020 (density-fixed eval) | 0.8814 | 5.6x @1% |
| `dual_fh_attn_ps384_final` job_1760964 (superseded 2026-07-29) | 384px | 0.4589 | 0.4230 (density-fixed eval) | 0.9026 | 11.9x @0.5% |
| `..._newfiresample_fhdropout_combo` job_1764635 ep15 | 384px | 0.5046 | 0.4368 | 0.8955 | 12.6x @0.5% |
| `SWA(combo ep15,17,18)` job_1764635 | 384px | n/a (averaged) | 0.4429 | 0.9011 | 12.6x @0.5% |
| **`SWA(resume ep28,29,30)` job_1764897** | **384px** | **n/a (averaged)** | **0.4485** | **0.8983** | **14.5x @0.5%** |

**CURRENT BEST MODEL (2026-07-29):**
`dual_fh_attn_ps384_final_newfiresample_fhdropout_combo/job_1764897/
swa_resume_ep28_29_30.ckpt` -- a WEIGHT AVERAGE of epochs 28/29/30 from the
resumed combo run. 2020 test AUC-PR=0.4485, ROC-AUC=0.8983, new-fire lift
**14.5x@0.5%** (TPR-new 0.073). Versus the day's starting baseline
`ps384_final` (0.4230 / 11.9x) that is **+6.0% AUC-PR and +21.8% new-fire
lift**. Built with `average_checkpoints.py`; needs no special inference path
(it is an ordinary Lightning checkpoint).

The former best, `job_1764635/.../best-epoch=15-val_ap=0.5046.ckpt` (combines
`fire_history_dropout_prob=0.3` + `new_fire_frac=0.3` on top of
`ps384_final`'s exact recipe — see State 2026-07-27/29 below for the full
new-fire-mechanism investigation this came out of). 128px was independently
re-run with a stretched epoch budget (25→45, resumed from its own
`last.ckpt`) to check whether it was just epoch-capped: it peaked at ep26
(0.4388, barely above its original 0.438) then **early-stopped normally** —
confirmed genuinely converged, not cut off. 384px's edge over 128px is real,
not an artifact of unequal training budgets.

### The root-cause finding: focal-loss-schedule crash
Every run all session (9+ seeds/configs, both `dual_fh_attn`'s original 256px
and the full 128/256/384px patch-size sweep) peaked at epoch 6–7 then dropped
hard — e.g. 256px seed789 ep7=0.213→ep8=0.081, 384px seed123 ep7=0.376→ep8=0.177.
This lines up exactly with `loss_schedule="bce_dice_bce_focal"`'s switch from
BCE to **focal loss at `focal_start_epoch=8`** (the default in every training
script). Confirmed causally with a controlled test (`dual_fastproto_nofocal`,
`FOCAL_START_EPOCH=999`): identical to the focal-switching baseline through
epoch 7 (both 0.189), then diverges immediately — baseline dips to 0.166,
nofocal climbs to 0.209 and keeps going to 0.243 (vs baseline's capped 0.219).
**Fix: disable the focal-loss switch** (`FOCAL_START_EPOCH` pushed past
`max_epochs`). This is now standard for every training run; `focal_start_epoch`
staying at 8 anywhere is a bug, not a choice.

### Two more validated training-time changes (combined with no-focal, all
production runs now use all three)
- **pos_weight annealing** (`pos_weight_start=100, pos_weight_end=20,
  pos_weight_anneal_epochs=8`, `conv_lstm_lit.py::on_train_epoch_start`):
  `pos_weight=100` buys ranking under extreme imbalance at the cost of
  calibration (loss-optimal output for true-p is `w·p/(w·p+(1-p))`, which
  pushes even low-p pixels toward ~0.5+ at w=100). Annealing down over
  training gave a consistent edge in the fastproto A/B (0.2302 vs 0.2191 at
  matching epochs) and produced a MUCH less compressed raw output
  distribution on the new checkpoint (raw mean|fire 0.70 vs mean|bg 0.30 --
  compare the old checkpoint's raw 0.87 vs 0.69, barely separated at all).
- **Isolation-weighted loss** (`isolation_gamma=2.0`, `conv_lstm_lit.py::
  _isolation_weight`): upweights loss on positive pixels that are spatially
  isolated from other fire in the CURRENT target (box-filter local density),
  targeting the model's fire-proximity shortcut (permutation importance
  showed distance-to-recent-fire dominates by far, weather channels near
  zero -- the model has learned "fire near fire" and has ~no signal for
  genuinely new, isolated ignitions).

### Recalibration is PER-CHECKPOINT, not reusable
`pos_weight=100` (constant) vs annealed compresses the raw sigmoid output
very differently, so an isotonic map fit on one checkpoint is WRONG for
another (`fit_recalibration.py`, env-overridable `CKPT`/`PATCH`/`OUT`). Always
refit before treating a new checkpoint's continent-wide probabilities as
calibrated. Current: `recal_isotonic_dual_fh_attn_ps384_final.pkl`.

### Six new features added, validated via a cheap XGBoost gate BEFORE any
ConvLSTM compute (`aux_xgb_check.py`, `aux_xgb_check_wind.py` — always run
this gate on a new candidate feature before wiring it into the datamodule)
| Feature | 2020 test AP delta | Verdict |
|---|---|---|
| lightning (LIS/OTD HRFC, via APDRC OPeNDAP mirror, no auth) | +7.4%, SHAP rank 2/14 | **kept** |
| distance-to-road (GRIP4 **Region 7** = Oceania — NOT 6 or 5, see Pitfalls) | −4 to −9% (2 regimes) | **rejected** |
| population (WorldPop 1km) | tested with roads, same negative result | **rejected** |
| elevation (ETOPO1, via NOAA ERDDAP griddap, no auth) | +5.3% (+9.1% in the combined run below) | **kept** |
| slope + aspect_sin + aspect_cos (derived from the same ETOPO1 DEM, free) | +22.3% combined with fuel_age, on top of elevation | **kept** |
| fuel_age (days since last fire, leakage-free, derived from `y_fire_3d`, free) | dominant part of the above; SHAP rank 9/18 | **kept** |
| downwind_align (BARRA2 AUS-11 wind vs recent-fire direction) | +2.4%, SHAP rank 12/19 (smaller, but real) | **kept** |

Roads/population encode "where fire has always been" (human-density proxy),
not real risk — confirmed negative in two different sampling regimes. Every
kept feature is now wired into `zarr_dual_datamodule.py` behind its own flag
(`use_lightning`, `use_elevation`, `use_slope_aspect`, `use_fuel_age`,
`use_wind_dir`) — none were in the checkpoints listed above, since they were
only wired in after `ps384_final` finished training. **The next training run
should enable all five flags together** — that hasn't happened yet.

`downwind_align` is deliberately NOT flip/rotation-corrected in `_augment`
(see below): it's a relative angle between two co-rotating vectors (wind
bearing, fire→pixel direction), invariant under any orthogonal transform
applied identically to both. `aspect_sin`/`aspect_cos` DO need explicit
correction (absolute compass bearing, doesn't rotate with the patch) — this
is handled and pixel-level-verified in `_augment`.

### Flip/rotation augmentation — implemented for the first time on the dual
model (`zarr_dual_datamodule.py::_augment`, `cfg.augment`, train-only)
The old single-branch v1/v2 pipeline had flip+90°-rotation augmentation; the
dual-branch pipeline (everything trained this whole session) never did. Adding
it naively would have been wrong given the two new absolute-direction
features — exactly the risk flagged in the old backlog note below
("flip/rot augmentation (careful: wind u/v components)"). Implemented with
explicit sin/cos rotation correction for `aspect_sin/cos`; `downwind_align`
needs no correction (see above). Verified pixel-exact against hand-computed
expected values, not just "doesn't crash."

### Full-Australia GeoTIFF pipeline (`make_australia_geotiff.py`,
`make_australia_groundtruth_geotiff.py`)
- **PATCH must equal the checkpoint's training patch size.** Tiling at 128px
  for a 256px-trained checkpoint caused two confirmed, measured artifacts:
  coastal risk inflation (~3x on non-fire land pixels near the coast) and
  tile-boundary seams (+25% pixel-jump at stride boundaries) — root cause is
  the leakage-free fire-history distance-transform feature (model's #1
  feature by permutation importance) being computed fresh inside whatever
  window it's given, so a mismatched tile size reintroduces a real
  train/inference distribution shift. Fixed by setting `PS`/`STRIDE` to match
  the checkpoint (currently 384/288).
- Output format (changed 2026-07-25, per-file not bundled): one single-band
  GeoTIFF per date per product — `firerisk_prob_<date>.tif`,
  `firerisk_top1pct_<date>.tif`, `firerisk_top0.5pct_<date>.tif`,
  `firerisk_top0.1pct_<date>.tif`; ground truth similarly split into
  `groundtruth_fire_<date>.tif` (one per date) + `landmask_static.tif`
  (written once, identical across dates).
- `operational_stats_dual.py`'s fire-density filter (`min_pos_pixels`) must be
  SCALED BY PATCH AREA (`BASE_PATCH=256, BASE_MIN_POS=20`, density 0.0305%)
  the same way training's sampler is — a fixed absolute pixel count made the
  original 128px-vs-384px lift comparison (5.4x vs 21.2x) not apples-to-apples;
  384px's larger window could satisfy the same absolute threshold at much
  lower relative fire density, mechanically inflating its lift numbers. Fixed;
  honest comparison is 5.6x vs 11.9x.

## State as of 2026-07-27

### Architecture/scale sweep on top of `dual_fh_attn_ps384_fullfeat` — all four lost
Ran four variants of the full-feature (6 new features + augmentation) recipe,
each changing exactly one thing vs `dual_fh_attn_ps384_fullfeat` (job 1761503,
best 0.4436 @ep15 — itself already slightly below the pre-feature baseline
`ps384_final`, 0.4589 @ep14):

| Variant | Change | Best val_ap | Verdict |
|---|---|---|---|
| kernel=9×9 | `kernel_size=(9,9)`, params 1.2M→3.7M | 0.2049 @ep7, collapsed | **rejected** |
| 2-layer | `hidden_layers=2`, params 1.2M→2.8M | 0.4194 @ep6, collapsed | **rejected** |
| 512px patch | `patch_size=512`, `batch_size=1` | 0.3014 @ep11 | **rejected** |
| static head | pointwise path for constant-over-time statics (see below) | 0.3685 @ep7, collapsed | **rejected** |

None beat even `main_fullfeat`, let alone the pre-feature baseline. **The 512px
result forced a correction** to earlier reasoning: `fire_dist` isn't raw pixel
distance, it's `exp(-distance_px/5.0)` — a 5-pixel length scale, already ≈0
beyond ~25-30px regardless of patch size. The truncation-frequency table used
to justify going bigger (128px→384px measured real gains) was measuring the
wrong representation; 512px's extra area barely changes `fire_dist`'s actual
value and instead just adds noise (smaller effective batch, less-diverse
sampling per epoch at a fixed `SAMPLES_PER_EPOCH`).

### The static head (implemented, tested, net negative)
Permutation importance on `dual_fh_attn_ps384_fullfeat`'s checkpoint (24
patches @ 192px, `convlstm_importance_dual.py`) showed `fire_dist` (43.3%) +
`fire_hist_t-3` (34.7%) dominate ~78% of the model's decisions; of the 6 new
features, only **slope, fuel_age, downwind_align, aspect_cos** register real
importance — **elevation, lightning, aspect_sin sit at ~0**, despite elevation
and lightning both passing their own standalone XGBoost gates earlier. Added a
`static_head` (`ConvLSTMSegDual`, `static_dim` param): a pointwise (1×1 conv)
path for the 7 constant-over-time statics (agb, landmask, lightning,
elevation, slope, aspect_sin, aspect_cos), concatenated into the fusion
representation alongside the cross-attention output — additive, not a
replacement for the existing broadcast-through-the-recurrent-branches path.
Hypothesis: a value that's identical at every one of 14-18 recurrent steps
gives the ConvLSTM no temporal signal to key gradient on. Result: **didn't
help** (0.3685 @ep7, see table above) — early promise (briefly ahead of
`main_fullfeat` through ep7) that then collapsed like the other three variants.

### The early-peak-then-collapse pattern — found a real cause, then partially refuted it
All four variants above share a suspicious shape: peak around epoch 6-8, then
decline, vs the untouched `ps384_final` baseline's steady climb to ep14.
Checked `val_p_mean` (mean predicted probability) across epoch 7→8 for all
five full-feature runs plus the original no-feature baseline:

| Run | val_p_mean ep7→ep8 | val_ap ep7→ep8 |
|---|---|---|
| baseline (no features/augment) | 0.387→0.295 | 0.363→0.377 (climbing) |
| main_fullfeat | 0.390→0.037 | 0.365→0.312 (dips) |
| kernel9 / 2-layer / static-head | similar ~10-25x val_p_mean collapse | all dip |

Every full-feature+augmentation run shows a 10-25x collapse in mean predicted
probability exactly at epoch 8 that the untouched baseline doesn't show at
all — `main_fullfeat` recovers from it (climbs on to 0.4436@ep15), the three
architecture variants don't recover in time and get killed by
`EarlyStopping(patience=6)` before reaching the epoch range where recovery
happens. **Hypothesis tested: is this the `pos_weight` anneal (100→20 over 8
epochs) completing?** Reran static-head and 2-layer with
`POS_WEIGHT_ANNEAL_EPOCHS=14` (anneal now only ~57% done at epoch 8, not
complete) — **the dip still happened at exactly epoch 7-8, at nearly
identical magnitude** (static-head: 0.3685→0.3669 best val_ap, same epoch;
2-layer: 0.4194→0.4056, same epoch). **This refutes the anneal-timing
hypothesis** — stretching the schedule didn't move the dip at all. Leading
alternative explanation, not yet tested: every one of these runs shares the
same fixed training seed (`SEED=123`, never overridden), so they all see an
*identical* sequence of training patches in identical order — a
seed-diagnostic rerun of `main_fullfeat` with `SEED=456` is running now
(`dual_fh_attn_ps384_fullfeat_seed456`) to check whether the dip moves or
disappears with a different seed.

### Three code changes made in response (all additive, none gated behind a
flag that changes default behavior except where noted)
- **`static_head`** (`ConvLSTMSegDual`/`ConvLSTMLitDual`, `static_dim` param,
  `USE_STATIC_HEAD` env in the training script): see above. Tested, rejected,
  but the code path stays available for future retries (e.g. with a lower LR
  specific to the new head) since the underlying diagnosis (dead-weight
  features via the recurrent path) is still valid even though this fix didn't
  pan out.
- **`dilation`** (`ConvLSTMSegDual`, `DILATION` env, default 1): grows the
  whole network's receptive field ((k-1)·dilation+1 per conv, same-padding)
  at **zero extra parameter cost** — unlike the kernel=9×9 attempt, which
  tripled params (1.2M→3.7M) and collapsed hardest of all four variants.
  Verified dilation=1/2/3 all give identical param count (1,159,681). A
  `dual_fh_attn_ps384_dilation2` run is queued to test this cheaper way of
  addressing the same "9px effective receptive field" observation that
  motivated the failed kernel=9×9 attempt.
- **Multi-scale isolation-weighted loss** (`conv_lstm_lit.py::
  _isolation_weight`, always active when `isolation_gamma>0`, no flag): the
  old version used a single `avg_pool2d(kernel=9)` box-filter density, which
  saturates at its max weight for anything with no other fire within ~4px —
  it cannot tell "isolated by 20px" from "isolated by 200px", but the actual
  new-fire lift curve (below) peaks at top-0.5% and falls on both sides, i.e.
  isolation strength is graded, not binary. Naively averaging raw density
  across wider scales doesn't fix this either (dividing by an 81×81 window's
  6561 pixels dilutes 1-2 nearby fire pixels' contribution below 0.001 —
  measured directly). Fixed by using per-scale PRESENCE (any *other* fire
  pixel within radius, via a pooled count with the center pixel's own
  contribution subtracted off) at 3 scales (`isolation_kernel`×1/3/9),
  averaged into 4 clean, well-separated tiers instead of one saturating
  radius. Verified: fully-isolated (nothing within 81px) → weight 3.0;
  fire 40px away → 2.33; fire 5px away → 1.67; background → 1.0 (gamma=2).
  Since this changes loss dynamics, it applies retroactively to ALL runs
  submitted after this change regardless of what else they're testing —
  `dual_fh_attn_ps384_isoweight_v2` (identical to `main_fullfeat`'s recipe
  otherwise, same seed) is the one clean, dedicated comparison against
  `main_fullfeat`'s 0.4436.
- **`val_ap_newfire`** (`conv_lstm_lit.py::validation_step`/
  `on_validation_epoch_end`, always on, pure logging, no training-dynamics
  effect): pooled AP restricted to genuinely NEW ignitions (fire pixels not
  within ~3px of the `fire_hist_t-3` mask) as the positive label, logged every
  epoch alongside `val_ap`. Previously this was only visible via a separate
  offline script (`operational_stats_dual.py`) after training finished, so
  checkpoint selection was blind to new-fire skill specifically — `val_ap` is
  dominated by the easy persistence/growth cases. Not used for selection yet,
  just now visible.

### Full-year lift-vs-topk curve (regenerated with a much larger, representative
sample: n=1500 patches, not the earlier n=200)
`dual_fh_attn_ps384_final`, `figures/fig_lift_vs_topk_2020_full_year.png`:
all-fire lift climbs monotonically as top-k tightens (920.9x @0.01%, dominated
by easy persistence/growth pixels); new-fire lift peaks in the MIDDLE
(11.9x @0.5%) and falls off at both extremes — genuinely new ignitions are
not the model's most-confident predictions, confirming that an ultra-tight
top-k threshold biases toward known-fire persistence rather than fresh starts.

### XGBoost final comparison (`compare_xgb_convlstm_australia.py`,
`xgb_australia_grid.py`, `figures/make_xgb_vs_convlstm_comparison.py`)
Trained the best-possible XGBoost model combining ALL 6 validated new features
together for the first time (lightning + elevation + slope/aspect + fuel_age
+ downwind_align, 21 features total, pooled 2015-2018 train / 2020 test, now
that BARRA2 wind is cached for all 6 years) — this had never been tested as
one combined XGBoost run before (lightning was gated standalone; the other 5
were gated together but never with lightning). **2020 test AUC-PR: 0.0611**,
vs ConvLSTM's 0.4230 — roughly 7x worse. XGBoost's SHAP importance is spread
thin (vpd 12.9%, fuel_age 9.3%, lst_day 8.3%, fire_hist_t-3 8.1%, sm 7.7%,
...) vs ConvLSTM's concentration on fire-history/distance (~78% combined),
consistent with the AUC-PR gap. On the most fire-active 2020 date
(2020-11-15, 5884 ground-truth pixels, northern-Australia savanna fires):
ConvLSTM caught 4726/5884 in its top-5% vs XGBoost's 826/5884 — and XGBoost's
hit count barely changes from top-5% to top-0.5% (826→693), meaning most of
what a tighter threshold drops for XGBoost is false alarms, not real fires,
i.e. its ranking has many near-ties rather than confident separation.

**Bug found and fixed during this**: `y_fire_3d` labels ocean as fire=1 (the
same issue flagged for validation metrics in `conv_lstm_lit.py`) was ALSO
uncorrected in `aux_xgb_check_wind.py`'s `fh3` variable, used for the
`downwind_align` distance transform — every coastal pixel's "nearest fire"
was collapsing to the nearest ocean pixel. Harmless-ish in small random
training patches (only some are coastal) but glaring at full-Australia map
scale. Fixed in `compare_xgb_convlstm_australia.py` and `xgb_australia_grid.py`
(`fh3 = (... > 0) & lm`); the pre-existing `aux_xgb_check_wind.py` still has
the bug (low priority to fix, already served its purpose as a gate).

### Results of the seed456/trimmed/dilation2/isoweight_v2 sweep — new best model found
All four completed (early-stopped at ep13, patience=6). Every one still hit
the SAME epoch-7→8 dip regardless of seed, feature set, receptive field, or
loss formula — none of them recovered past their own epoch-7 peak afterward:

| Run | Best val_ap | Epoch | vs `ps384_final` (0.4589) |
|---|---|---|---|
| **`dual_fh_attn_ps384_trimmed`** | **0.4773** | 7 | **new best, +4%** |
| `dual_fh_attn_ps384_isoweight_v2` | 0.4426 | 7 | below |
| `dual_fh_attn_ps384_fullfeat_seed456` | 0.4339 | 7 | below |
| `dual_fh_attn_ps384_dilation2` | 0.4136 | 7 | below |

**`dual_fh_attn_ps384_trimmed/job_1762760/.../best-epoch=7-val_ap=0.4773.ckpt`
is the new best checkpoint for the whole session**, beating `ps384_final`'s
0.4589 by ~4%. It drops `use_lightning`/`use_elevation` (both showed ~0
permutation importance, see above) while keeping
`use_slope_aspect`/`use_fuel_age`/`use_wind_dir` + the multiscale
isolation-loss fix. Note it does NOT avoid the epoch-8 dip (0.477→0.340,
same as everything else) — it wins purely because its pre-dip trajectory is
so much stronger than the others', not because it dodged the instability.
`isoweight_v2`'s comparatively modest gain (0.443, close to `main_fullfeat`'s
original 0.4436) suggests the isolation-loss fix alone isn't the dominant
factor — dropping the two dead-weight features looks like the bigger lever.

Two seed-diagnostic conclusions from `fullfeat_seed456`: it also dipped at
epoch 7→8 (0.434→0.331) — **refutes seed-specificity** as a sole cause,
same as the anneal-stretch and isolation-formula tests before it. The
epoch-7/8 dip mechanism remains genuinely unexplained after four ruled-out
hypotheses (anneal timing, seed, isolation-loss formula, feature set/RF);
current live hypothesis is augmentation itself (the one remaining untested
variable — `ps384_final`, the only run that never dipped, also never used
augmentation) — `dual_fh_attn_ps384_noaugment` is testing this.

### Augmentation refuted too -- the epoch-7/8 dip is now unexplained after SIX hypotheses
`dual_fh_attn_ps384_noaugment` (AUGMENT=0, the one remaining untested variable
since `ps384_final` was the only run that never dipped AND the only one
without augmentation): still dipped. ep7=0.452 (a new high) -> ep8=0.374, a
17.3% relative drop -- squarely inside the range seen across every
WITH-augmentation run this session (12%-40%). This refutes augmentation as
the cause, joining anneal-timing, training seed, isolation-loss formula,
feature set, and receptive field as ruled-out hypotheses -- SIX tested, SIX
refuted. Decision: stop dedicated root-cause investigation here
(diminishing returns) and treat the dip as an unexplained but apparently
harmless characteristic of training in this regime -- every run that reaches
a strong-enough pre-dip peak (`trimmed`'s 0.4773) still ends up as the best
result even without understanding why the dip happens or recovering past it.
Focus going forward: keep testing recipes empirically, not the dip itself.

### noaugment / trimmed_noaugment results -- augmentation is a net positive, not a cause of the dip
`dual_fh_attn_ps384_noaugment` completed: best 0.4521 @ep7 (between `main_
fullfeat`'s 0.4436 and `ps384_final`'s 0.4589 -- decent, not a contender).
`dual_fh_attn_ps384_trimmed_noaugment` (trimmed's feature set, no augment)
completed: peaked 0.4575 @ep7, still hit the same dip (0.323 @ep8), never
recovered past its peak -- and that peak is BELOW `trimmed`'s own 0.4773 at
the identical epoch (0.409/0.458 vs 0.441/0.477 at ep6/ep7). So removing
augmentation didn't just fail to help `trimmed`'s recipe, it measurably
*hurt* it. Augmentation appears to be a net positive here, the opposite of
the original concern that it was destabilizing training.

**`dual_fh_attn_ps384_trimmed` (0.4773 @ep7) stands as the best model.**
Now validating it on the real 2020 test set (`submit_ps384_trimmed_products.
sbatch`: refit recalibration + `operational_stats_dual.py`, density-scaled,
n=1500, PATCH=384 matching training) to get a real AUC-PR number comparable
to `ps384_final`'s known 0.4230, rather than relying on `val_ap` alone.

### IMPORTANT CORRECTION: trimmed's real 2020 test AUC-PR is WORSE than ps384_final's, despite a higher val_ap
`fit_recalibration.py` and `operational_stats_dual.py` both had a bug: their
`DualPatchConfig` construction never passed `use_lightning`/`use_elevation`/
`use_slope_aspect`/`use_fuel_age`/`use_wind_dir`, so they always built the
OLD 7-channel input regardless of what the checkpoint being evaluated
actually needs -- crashed immediately on any checkpoint trained with the new
features (channel-count mismatch, e.g. 7 given vs 10 expected). Fixed by
adding env-var flags (`USE_LIGHTNING`/`USE_ELEVATION`/`USE_SLOPE_ASPECT`/
`USE_FUEL_AGE`/`USE_WIND_DIR`) to both scripts, matching the pattern already
used in `train_convlstm_dual_fastproto.py`. `make_australia_geotiff.py` does
NOT share this bug -- it builds its features directly rather than through
`DualPatchConfig`, so it needs a separate (larger, not yet done) update
before it can render maps for any of this session's new-feature checkpoints.

With the fix applied, ran the real validation pipeline
(`submit_ps384_trimmed_products.sbatch`, `USE_SLOPE_ASPECT=1 USE_FUEL_AGE=1
USE_WIND_DIR=1` matching `trimmed`'s actual recipe) on `dual_fh_attn_ps384_
trimmed`'s checkpoint (`best-epoch=7-val_ap=0.4773.ckpt`, n=1500 patches,
density-corrected):

**2020 test AUC-PR = 0.4017, ROC-AUC = 0.8658** — this is *worse* than
`ps384_final`'s known 0.4230, despite `trimmed`'s higher validation val_ap
(0.4773 vs 0.4589). The validation-set win did NOT transfer to the true
holdout year. Leading explanation: `trimmed` peaked at epoch 7 (far fewer
total training steps than `ps384_final`'s epoch-14 checkpoint) -- an
early-epoch checkpoint that scores well on 2019 validation may not have had
enough training to consolidate a representation that generalizes to a
*different* year as robustly as a more mature checkpoint does, even with a
lower raw val_ap. This is a real methodological caveat for the whole
session: **val_ap on 2019 is not always a reliable predictor of 2020 test
performance**, especially for early-epoch checkpoints.

Also notable: `trimmed`'s new-fire lift curve has a completely different
SHAPE from `ps384_final`'s -- it peaks at the tightest threshold (826.7x
@0.01%) and falls monotonically as top-k widens, unlike `ps384_final`'s
middle-peaking curve (11.9x @0.5%, falling off at BOTH extremes). A genuine
behavioral difference between the two models, not an artifact.

Testing now: does `trimmed`'s own LATER checkpoint (`best-epoch=13-
val_ap=0.4463.ckpt`, lower val_ap but more training steps) generalize better
to 2020 test than its epoch-7 peak did? (job 1764021). If so, that would
directly confirm the "needs more steps to generalize" hypothesis and argue
for selecting checkpoints with some minimum training maturity, not purely by
peak val_ap.

### Epoch-13 test settles it: neither of trimmed's checkpoints beats ps384_final on real test data
`trimmed`'s epoch-13 checkpoint (val_ap=0.4463, ~2x the training steps of its
epoch-7 peak): 2020 test AUC-PR = **0.3982**, ROC-AUC = 0.9025. LOWER AUC-PR
than epoch-7's 0.4017, not higher -- the "needs more training steps to
generalize" hypothesis is refuted too. Full picture:

| Checkpoint | val_ap (2019) | 2020 test AUC-PR |
|---|---|---|
| **`ps384_final`** (ep14) | 0.4589 | **0.4230 -- still the best** |
| `trimmed` ep7 | 0.4773 | 0.4017 |
| `trimmed` ep13 | 0.4463 | 0.3982 |

**`ps384_final` remains the best validated model.** Nothing this session has
beaten it on the real 2020 test set, despite `trimmed` looking better on
val_ap. This session's actual output: two real bugs found+fixed (ocean-
labeled-as-fire in `downwind_align`'s distance transform; the checkpoint-
eval feature-flag mismatch above), six ruled-out hypotheses for the
epoch-7/8 training dip, and a documented warning that val_ap on 2019 does
not reliably predict 2020 test AUC-PR for this task -- checkpoint selection
by val_ap alone is not sufficient; a candidate new-best should always be
confirmed on 2020 test before being treated as an actual improvement.

### New lever tested: training the model specifically for new-fire pixels
User question: can the model be trained to specifically get better at NEW
fire (not just reweighted at loss time, which `isolation_gamma` already
does)? Implemented **fire-history feature dropout**
(`zarr_dual_datamodule.py`: `DualPatchConfig.fire_history_dropout_prob`,
computed `fire_hist_start_idx`/`n_fire_hist_channels`, applied in
`__getitem__` train-only) -- randomly blanks `fire_hist_t-3/4/5` +
`fire_dist` to 0 (a valid "no recent fire nearby" state for both channel
types, not a corruption) for a fraction of TRAINING samples, forcing the
network to predict from weather/terrain alone on those samples. Targets the
actual root cause directly: permutation importance showed fire-history/
distance account for ~78% of the model's decisions, i.e. it has learned a
shortcut that only helps with persistence/growth of already-known fire, not
genuinely new ignitions.

`dual_fh_attn_ps384_final_fhdropout` (job 1764064, `FIRE_HISTORY_DROPOUT_
PROB=0.3`, otherwise `ps384_final`'s exact recipe): `val_ap_newfire`/`val_ap`
ratio jumped to **15.4% at epoch 4** (0.0556/0.361) -- already closing in on
`ps384_final`'s own CONVERGED ratio of **17.1%** (val_ap=0.466, val_ap_
newfire=0.0798, measured via a one-off `trainer.validate()` pass since this
metric didn't exist during its original training -- see `quick_valnewfire_
check.py`), and fhdropout is only 4 epochs in vs `ps384_final`'s full 14.
Promising but not yet conclusive -- the real test is whether it EXCEEDS
17.1% once it also converges, not just approaches it early.

Also running: `dual_fh_attn_ps384_final_isoweight` (job 1764059,
`ps384_final`'s exact recipe + the multiscale isolation-loss fix ONLY, the
cleanest single-variable comparison available) -- 9 epochs, peaked
**0.427@ep7** (vs `ps384_final`'s own 0.363@ep7), and its epoch-8 dip was
much milder than every other run this session (0.427→0.396, ~7% drop vs the
usual 15-40%). Still needs the same real-2020-test verification as
`trimmed` before any val_ap lead is trusted (see the correction above).

### isoweight result: the clean isolation-loss-only test made the FINAL result worse, despite an early lead
`dual_fh_attn_ps384_final_isoweight` (job 1764059) completed: best val_ap =
**0.4271 @ep7**, early-stopped ep13. Below `ps384_final`'s 0.4589. Despite
leading early (0.427 vs `ps384_final`'s own 0.363 at ep7), it never climbed
back to match `ps384_final`'s eventual rise to 0.4589@ep14 -- peaked early,
plateaued lower. **In the one truly clean, single-variable comparison run
this session, the new multiscale isolation-loss fix made the final result
WORSE, not better**, correcting the earlier read (which was based on an
early, not-yet-converged lead). The fix may still be worth keeping for its
effect on new-fire ranking specifically (not measured for this run, `val_ap_
newfire` was added after this job started) but it is NOT a clean overall win
on `val_ap` alone.

### fhdropout result: essentially a tie, not a win -- session conclusion
`dual_fh_attn_ps384_final_fhdropout` (job 1764064) completed: best val_ap =
**0.4595 @ep7**, early-stopped ep13. Technically above `ps384_final`'s
0.4589 but by +0.13% relative -- noise, not a real improvement. New-fire
picture is mixed: final epoch (13)'s val_ap_newfire/val_ap RATIO hit 19.6%
(above `ps384_final`'s 17.1% baseline), but the ABSOLUTE val_ap_newfire
(0.0743) is slightly BELOW baseline's 0.0798 -- the higher ratio just
reflects a much lower overall score (0.379) being divided more evenly, not
genuinely better new-fire detection. Given the margin is coin-flip-sized and
this session already showed much larger val_ap gaps (`trimmed`'s +4%) don't
reliably predict 2020 test performance, no real-test validation was run for
this result -- not worth the GPU time to resolve a difference this small.

**Session conclusion on "train specifically for new fire pixels"**: tested
two mechanistically-sound approaches (multiscale isolation-loss reweighting;
fire-history feature dropout forcing weather/terrain-only prediction on 30%
of samples). Both changed measurable training behavior but neither produced
a checkpoint clearly better than `ps384_final`. The `fire_history_dropout_
prob` mechanism (`zarr_dual_datamodule.py`) remains available for future
attempts (e.g. a different dropout probability, or combined with a longer
training budget) even though this specific test didn't win.

**Overall session conclusion**: after testing training seed, feature
trimming, dilation, kernel size, depth, static-head routing, augmentation
on/off, the isolation-loss fix, and fire-history dropout -- `ps384_final`
(job 1760964, val_ap=0.4589@ep14, 2020 test AUC-PR=0.4230) remains the best
model on real 2020 test data. Real output of the session: two genuine bugs
found+fixed (ocean-labeled-as-fire in `downwind_align`; the checkpoint-eval
feature-flag mismatch), the multiscale isolation-loss and `val_ap_newfire`
logging improvements (kept even though this round didn't win with them), and
a validated warning that `val_ap` on 2019 does not reliably predict 2020
test performance for this task -- always confirm a candidate improvement on
real test data before trusting it.

### Fourth lever: biasing the SAMPLER toward new-fire patches (not just loss/features)
User follow-up: what about biasing which patches the model trains on, not
just reweighting the loss (`isolation_gamma`) or masking a feature
(`fire_history_dropout_prob`)? Implemented **`new_fire_frac`**
(`DualPatchConfig.new_fire_frac`/`min_new_fire_pixels`, `zarr_dual_
datamodule.py::__getitem__`'s patch-search/rejection loop, independent of
`pos_frac`) -- forces a fraction of TRAINING draws to contain at least
`min_new_fire_pixels` pixels where the target has fire but the same pixel
did NOT at `t_end-3` (a cheap non-dilated proxy for "genuinely new
ignition", good enough for a sampling bias though not identical to
`val_ap_newfire`'s dilated definition). Verified directly: with `new_fire_
frac=0.0`, 38/90 (42%) of sampled "positive" patches had ZERO new-fire
pixels (pure persistence/growth); with `new_fire_frac=1.0`, 0/90 did.
`dual_fh_attn_ps384_final_newfiresample` (job 1764413, `NEW_FIRE_FRAC=0.3`,
otherwise `ps384_final`'s exact recipe) is running to test whether changing
the training DATA DISTRIBUTION itself (not just loss weight or feature
availability) moves the needle where the other two new-fire-targeted
attempts (isolation-loss fix, fire-history dropout at two doses) didn't.

### fhdropout=0.5 result: clear dose-response, MORE dropout hurts more
`dual_fh_attn_ps384_final_fhdropout50` (job 1764271, `FIRE_HISTORY_DROPOUT_
PROB=0.5`, double the earlier 0.3 dose) completed: best val_ap = **0.3683
@ep6**, early-stopped ep12. Clearly worse than both `ps384_final` (0.4589)
and the 0.3-dose version (0.4595) -- confirms a real dose-response: some
fire-history dropout is roughly neutral, more is actively harmful. Makes
sense in hindsight: blanking the dominant signal (78% of the model's
decisions) for half of training samples removes too much for the network to
compensate for with weather/terrain alone in the time available.

### newfiresample: briefly looked dip-free through ep7, but ep8 refuted it -- milder dip, not no dip
`dual_fh_attn_ps384_final_newfiresample` (job 1764413, `NEW_FIRE_FRAC=0.3`)
held flat ep6->ep7 (0.4293->0.4286, val_ap_newfire ratio jumping to 24.7%,
clearly above `ps384_final`'s converged 17.1%) which looked like the first
run all session to avoid the epoch-7/8 dip entirely. Epoch 8 landed and
refuted that: **val_ap 0.4286->0.3808, val_ap_newfire 0.1059->0.0500** -- a
real dip, just milder than the usual 15-40% (this one ~11%). So the sampler-
bias mechanism doesn't avoid the dip, it just softens it somewhat. The
epoch-7 peak (0.4286) is still a solid number, close to (if still below)
`ps384_final`'s 0.4589 -- open question now is the same one every other run
answered differently: does it recover past 0.4286, or plateau below it like
`isoweight`/`trimmed` did. `dual_fh_attn_ps384_final_newfiresample70`
(`NEW_FIRE_FRAC=0.7`) is only at epoch 4 (0.372@ep3, 0.366@ep4, both clearly
above baseline's own trajectory at these epochs) -- too early to compare
doses yet, hasn't reached the epoch-7/8 dip point. Epoch 9: **0.4210**,
recovering from the ep8 dip but not yet back past the ep7 peak (0.4286),
let alone `ps384_final`'s 0.4589 -- still open whether it eventually clears
its own peak like `ps384_final` did (0.363@ep7 -> eventual 0.4589@ep14) or
plateaus below it like `isoweight_v2`/`trimmed` did.
`newfiresample70` epoch 6: 0.375 (oscillating 0.372/0.366/0.398/0.375 since
ep3, no sharp dip yet, but no clear climb either) -- still too early to call.

### newfiresample30/70 final results -- third new-fire mechanism also fails to beat baseline, plus a checkpoint-pruning gap found
Both completed. `newfiresample30` (job 1764413, `NEW_FIRE_FRAC=0.3`)
early-stopped at ep16: best **0.4360 @ep10** -- recovered off the ep8 dip but
plateaued below `ps384_final`'s 0.4589, the same outcome as `isoweight_v2`/
`trimmed`. `newfiresample70` (job 1764414, `NEW_FIRE_FRAC=0.7`) early-stopped
at ep13: best **0.4447 @ep7** -- slightly ahead of the 0.3 dose but still
below baseline. So across THREE distinct new-fire-targeted mechanisms
(isolation-loss reweighting, fire-history dropout at 2 doses, sampler bias
at 2 doses -- 5 configs total), none has produced a checkpoint that beats
`ps384_final` on `val_ap`, let alone real 2020 test data.

**Checkpoint-pruning gap found**: `newfiresample30`'s single best epoch for
new-fire detection specifically was ep7 (`val_ap_newfire`=0.1059, ratio
24.7% -- the best new-fire score of the entire session), but
`ModelCheckpoint(monitor="val_ap", save_top_k=3)` only tracks plain `val_ap`
-- once ep10 scored higher overall, ep7's checkpoint file was silently
deleted, so it can never be checked against real 2020 test data now. Fixed
for future runs: added a second `ModelCheckpoint(monitor="val_ap_newfire",
save_top_k=1)` callback in `train_convlstm_dual_fastproto.py` so the
best-by-new-fire epoch always survives independently of the best-by-overall-AP
one, per the existing `save_top_k` pitfall already documented below --
tracking two different selection criteria needs two separate callbacks, one
callback can't safely serve both.

### Fifth attempt: combining the two new-fire mechanisms that came closest to baseline
Of the 5 new-fire-targeted configs tested, `fire_history_dropout_prob=0.3`
(0.4595, essentially tying `ps384_final`) and `new_fire_frac=0.3` (0.4360,
the better of the two sampler-bias doses) came closest. They intervene at
different points in the pipeline -- dropout masks a feature for a fraction
of samples, sampler bias changes which patches get drawn in the first place
-- so they aren't obviously redundant. `dual_fh_attn_ps384_final_
newfiresample_fhdropout_combo` (job 1764635, both flags set together,
otherwise `ps384_final`'s exact recipe) is running to test whether they
compound or just plateau like every other new-fire attempt this session. As of this
check, job 1764635 is still PENDING with an estimated start time ~20+ hours
out (`squeue` shows it as the ONLY job in the entire a100 queue, yet
`sshare` shows this account's `EffectvUsage=0.95` -- i.e. it has burned far
more than its fairshare allocation from this session's long back-to-back
run of 3-6h A100 jobs). This is a structural scheduling delay, not a
transient queue blip -- don't expect it to clear on its own in the next
poll cycle. Future sessions: if `squeue -p a100` shows an empty queue but a
submitted job still gets a many-hour estimated start time, check
`sshare -u <user>` for `EffectvUsage` before assuming something is wrong
with the submission itself.

### CONFIRMED NEW BEST MODEL: fhdropout0.3 + new_fire_frac0.3 combo beats ps384_final on REAL 2020 test data
Job 1764635 hit its 6h wall-clock limit (`TIMEOUT`) at ep19, cut off before
`EarlyStopping` could trigger naturally -- so this is a truncated, not a
converged, result; more budget might go even higher (see Next steps). Best
checkpoint: **ep15, val_ap=0.5046**, the strongest trajectory of the entire
session, sustaining a 0.497-0.505 plateau through ep19 (not a brief spike),
~10% above `ps384_final`'s 0.4589. Its epoch-13 dip (0.4747->0.4259) was
much milder than the usual 15-40% collapse and it recovered past its prior
peak afterward, unlike `isoweight_v2`/`trimmed`/`newfiresample30/70` which
all plateaued below their pre-dip highs.

Regenerated the 4-panel XGBoost-vs-ConvLSTM prediction map figure
(`make_prediction_maps_dual.py`, `CKPT`=this checkpoint) for a qualitative
look ahead of the real-test numbers:
`figures/fig_pred_maps_DUAL_TEST2020_newfiresample_fhdropout_combo.png`.
Across the same 4 fire-rich 2020 test-year patches used for earlier
figures, this checkpoint visually dominates XGBoost: 2020-11-22 94% vs 31%
caught, 2020-04-24 91% vs 77%, 2020-11-23 99% vs 60%, 2020-11-17 77% vs 45%.

**Real-2020-test validation (job 1764871, `fit_recalibration.py` +
`operational_stats_dual.py`, PATCH=384, N_PATCH=1500) CONFIRMS a genuine
improvement** -- unlike `trimmed`, this one holds up on every real-test
metric, not just val_ap:

| Metric | `ps384_final` (prior best) | This checkpoint | Change |
|---|---|---|---|
| val_ap (2019) | 0.4589 | 0.5046 | +10.0% |
| 2020 test AUC-PR | 0.4230 | **0.4368** | **+3.3%** |
| 2020 test ROC-AUC | 0.9026 | 0.8955 | **-0.8%** |
| New-fire lift peak | 11.9x @0.5% | **12.6x @0.5%** | **+5.9%** |

**CORRECTION (2026-07-29)**: an earlier version of this table listed
`ps384_final`'s 2020 ROC-AUC as 0.8814 and therefore claimed the combo model
improved ROC-AUC by +1.6%. That was wrong -- 0.8814 is `ps128_final`'s value,
copied into a cell that was originally "-". `logs/opstats_ps384final_fixed_
1761391.log` gives `ps384_final` ROC-AUC=**0.9026**, independently reproduced
as member0 of the job-1764905 ensemble run (AUC-PR 0.4230 matched exactly,
confirming the same eval protocol). So the combo model trades a small
ROC-AUC loss (-0.8%) for its AUC-PR and new-fire-lift gains. AUC-PR is the
selection metric under this imbalance, so the trade is still favourable, but
it is a trade, not a clean sweep.

Full topk table (pooled n=1500 patches, 188.6M land px, base rate 0.175%):

| top-k | TPR(all) | lift(all) | TPR(new) | lift(new) |
|---|---|---|---|---|
| 0.01% | 0.101 | 999.2x | 0.000 | 0.2x |
| 0.1% | 0.347 | 347.0x | 0.006 | 6.4x |
| 0.2% | 0.423 | 211.3x | 0.020 | 10.2x |
| 0.5% | 0.509 | 101.8x | 0.063 | **12.6x (peak)** |
| 1.0% | 0.563 | 56.3x | 0.117 | 11.7x |
| 2.0% | 0.609 | 30.4x | 0.184 | 9.2x |
| 5.0% | 0.662 | 13.2x | 0.292 | 5.8x |
| 10.0% | 0.704 | 7.0x | 0.390 | 3.9x |

Same shape as `ps384_final` (all-fire lift peaks at the tightest threshold,
new-fire lift peaks mid-range) but with a higher new-fire peak. By season:
SON/spring (Sept-Nov, the main Australian fire season) strongest
(TPR-new=0.279, lift=13.9x); by region, Tropical North leads
(TPR-new=0.228, lift=11.4x) -- consistent with the known northern-savanna
fire pattern. mean fire pixels/patch=220, 39% of which are genuinely new
ignitions (not persistence). Recalibration succeeded too: raw AP=0.4786,
mean|fire=0.66 vs mean|bg=0.29 (well-separated, similar to `ps384_final`'s
annealed-pos_weight distribution) ->
`recal_isotonic_dual_fh_attn_ps384_newfiresample_fhdropout_combo.pkl`.

**`dual_fh_attn_ps384_final_newfiresample_fhdropout_combo/job_1764635/.../
best-epoch=15-val_ap=0.5046.ckpt` is now the best model of the session,
superseding `ps384_final`.** This is also the first confirmed win from the
"train specifically for new fire" investigation -- neither of the two
mechanisms won alone (fhdropout0.3 alone: 0.4595 val_ap, essentially tied;
new_fire_frac0.3 alone: 0.4360, below baseline) but combined they compound
into a real, validated improvement.

**Correction to the backlog**: the "landmask into the loss (currently only
masks metrics, not gradients)" item was checked directly in `_compute_loss`
and its callers (`conv_lstm_lit.py::training_step`/`validation_step`,
`conv_lstm_lit_dual.py::training_step`) -- the landmask (`batch["mask"]`,
populated in `zarr_dual_datamodule.py`) IS already passed through
`_get_lc_mask` into `_compute_loss` for both the main and auxiliary
deep-supervision losses, i.e. it already masks gradients, not just eval
metrics. This backlog item was stale/incorrect and is removed below.

## State as of 2026-07-14 (updated ~19:30)

### Generation history
| Tag | Jobs | seq_len | Notes | Status |
|-----|------|---------|-------|--------|
| `focal_*` | — | 16 | seq16 baseline, 5 real channels (lst/ndvi were zero) | Done |
| `focal30_*` | 1745052–56 | 30 | recency-weight bugfix, but still on broken cubes | Cancelled at ep2-3 |
| `focal30v2_*` | TBD | 30 | fixed cubes + DOY sin/cos + deep supervision + cosine LR | Pending (submitting after patch) |

### Baseline results (composite cond. F1 = 0.5·F1@1% + 0.3·F1@5% + 0.2·F1@15%)
- seq16 (`focal_*`, last epoch): shrubland 0.114, closed_forest 0.092, open_forest 0.088, grassland 0.029, cropland 0.029
- v1 (`focal30_*`, best ep, cancelled early): open_forest 0.104@ep0, cropland 0.094@ep0, shrubland 0.080@ep1, closed_forest 0.077@ep2, grassland 0.066@ep0

### Data bug — FIXED (patching in progress)
X channels 4 (`lst_day`) and 5 (`ndvi`) were ALL-ZERO in every cube — builder never
filled them. All models before today trained on only 5 real predictors. Fix:
- `patch_lst_ndvi.py` patches both channels in-place from source GeoTIFFs
- 2015: **PATCHED** ✓ (lst=48.58%, ndvi=48.23%)
- 2016–2020: **patching in parallel** (5 concurrent processes, pids 89215–89219)
  - logs: `logs/patch_2016.log` … `logs/patch_2020.log`
  - LST writes confirmed through day 200+ (actual) / 180 (logged) as of ~19:20
  - zarr chunks: (7, 512, 512, 7) — NFS-bound but progressing

### v2 improvements
- Per-channel normalization from `channel_stats_2015_2018.json` (recomputed after patch)
- DOY sin/cos appended as channels 8–9 in `zarr_daily_datamodule_v2.py` (lines 296-300) → 11 total inputs
- Deep supervision + cosine LR in `conv_lstm_lit_v2.py`
- 25 epochs with early stopping

### Next steps as of 2026-07-14 (historical — all superseded by the dual-branch
pipeline above; kept for the record, not actionable)
1. Wait for parallel patch (pids 89215–89219) to print `PATCHED` for all years.
2. Recompute stats: `python -u compute_channel_stats.py 2>&1 | tee logs/channel_stats_v2.log`
3. Verify patch: `python check_channels.py` (lst_day/ndvi should be ~47% nonzero like lai).
4. Submit v2: `ssh tinyx.nhr.fau.de bash /home/saturn/gwgi/gwgi107h/wildfire_data/firecastnet/submit_all_seq30_v2.sh`
5. When v2 finishes: compare metrics CSVs across `focal_*` (seq16), `focal30_*` (v1), `focal30v2_*` (v2) — best checkpoint, not last epoch.
6. Open improvement backlog: temporal attention pooling instead of last-hidden-only;
   2 layers / hidden 128; flip/rot augmentation (careful: wind u/v components);
   landmask into the loss; PR-AUC for model selection; percentile calibration in
   `combine_lc_predictions`; investigate partially-empty 2016 slices.

## Next steps
1. **New best model, confirmed on real 2020 test data**:
   `dual_fh_attn_ps384_final_newfiresample_fhdropout_combo/job_1764635/.../
   best-epoch=15-val_ap=0.5046.ckpt` (combining `fire_history_dropout_prob=
   0.3` + `new_fire_frac=0.3`) — 2020 test AUC-PR=0.4368, ROC-AUC=0.8955,
   new-fire lift 12.6x@0.5%, all beating `ps384_final`'s 0.4230/0.8814/11.9x.
   This is the first mechanistically-targeted new-fire intervention to
   actually win after 6 configs tried. **[RUNNING]** job 1764897 resumes
   training from this checkpoint's `last.ckpt` (epoch 19, where the original
   run got cut off by its 6h wall-clock limit, not by convergence) with
   `MAX_EPOCHS=40` (was 25) and an 8h time limit — checking whether more
   budget pushes val_ap even higher. If it does, re-run real-2020-test
   validation on the new checkpoint the same way (`fit_recalibration.py` +
   `operational_stats_dual.py`, PATCH=384, no USE_* flags) before trusting
   any further val_ap gain.
2. Fixed for future runs: a second `ModelCheckpoint(monitor="val_ap_newfire")`
   callback now runs alongside the `val_ap` one in
   `train_convlstm_dual_fastproto.py`, so the best-by-new-fire epoch survives
   independently instead of being silently pruned by `save_top_k=3` on plain
   `val_ap` (this already happened once — see the `newfiresample30` note
   above).
3. Backlog (not yet started, roughly in priority order): ranking-style
   auxiliary loss that directly targets the top-0.5%-1% operating point where
   new-fire lift peaks (BCE/dice/focal all optimize per-pixel correctness, none
   of them know the actual metric of interest is a ranking one); use
   `val_ap_newfire` as a secondary/tie-breaking checkpoint-selection signal for
   future runs (now that it survives pruning, see above); percentile
   calibration in `combine_lc_predictions`-equivalent for the dual model;
   temporal attention pooling instead of last-hidden-only in the slow branch;
   investigate whether the 2020 test set has any of the same
   partially-empty-slice issues flagged for 2016 back on 2026-07-14; update
   `make_australia_geotiff.py` to support the 5 new-feature flags plus
   `fire_history_dropout_prob`/`new_fire_frac` (it builds features directly
   rather than through `DualPatchConfig`, so it didn't inherit the fix
   already applied to `fit_recalibration.py`/`operational_stats_dual.py`) --
   now more urgent since the new best model needs continent-wide maps too.

## Tier-1 improvements implemented 2026-07-29 (variance + objective-alignment)

Three changes aimed at the two weaknesses this session actually demonstrated:
(a) `val_ap` is noisy and doesn't predict 2020 test, i.e. a lot of run-to-run
difference is VARIANCE, and (b) the loss optimizes per-pixel correctness while
the metric is a RANKING one at top-0.5%-1%.

### 1. OHEM (online hard example mining) -- `conv_lstm_lit.py`
`ohem_frac` / `ohem_min_negatives` / `ohem_rescale_negatives` (env:
`OHEM_FRAC`/`OHEM_MIN_NEG`/`OHEM_RESCALE`). Keeps ALL positives + only the
hardest `ohem_frac` of negatives, so gradient goes to the negatives that
actually compete with fire pixels for top-k slots rather than the ~99.8% of
trivially-easy background. `ohem_frac=0.0` (default) is the byte-identical
old path -- verified `torch.equal` against the plain BCE reference.

Verified numerically (not just "doesn't crash"): exactly 10.0% of negatives
receive gradient at frac=0.1; all positives retain gradient; the kept
negatives are exactly the highest-loss ones (min kept 2.63018 >= max dropped
2.63017); `frac=1.0` collapses back to the plain mean.

**`ohem_rescale_negatives` renormalizes to preserve total negative LOSS mass**
(measured ratio 1.0000 vs the plain mean at every frac). The obvious
alternative -- scaling by `n_neg/k` -- is WRONG and was caught by the test:
the hardest 10% already carry ~31% of negative mass, so that factor lands at
**3.1x** the original loss, silently acting as a second `pos_weight`. Caveat
measured and documented in-code: loss mass and GRADIENT mass can't both be
preserved; pos:neg gradient ratio still shifts 0.134 -> 0.232 at frac=0.1, so
`ohem_frac` is mostly- but not perfectly-orthogonal to `pos_weight`.

### 2. Ensembling -- `average_checkpoints.py`, `ensemble_eval.py`
Safe here because the architecture has NO running-stats normalization (only
`nn.LayerNorm`) -- checked before writing, so weight averaging needs no
BatchNorm re-estimation pass.
- `average_checkpoints.py`: weight-averages checkpoints from ONE run (SWA).
  Built `swa_ep15_17_18.ckpt` from the combo run's three near-equal late
  checkpoints (0.5046/0.5035/0.5042 -- indistinguishable, classic variance).
- `ensemble_eval.py`: runs N checkpoints over the SAME patches, reports the
  full `operational_stats_dual.py`-style topk tables for each member plus two
  combination rules. **prob-mean vs rank-mean are NOT equivalent here**:
  members have genuinely different raw output distributions (that's why
  recalibration is per-checkpoint), and a measured spearman of 0.94 between
  prob-mean before/after a monotone rescale of ONE member confirms plain
  probability averaging is distorted by calibration. rank-mean is
  calibration-invariant and is the better-motivated default.
  `rank01` averages ties deliberately -- saturated sigmoids and flat regions
  produce many equal scores, and a stable argsort would leak array index order
  into the ensemble as if it were signal.

### 3. Grouped (region x season) isotonic recalibration -- `fit_grouped_recalibration.py`
The key point: **a global isotonic map provably CANNOT change AUC-PR** (it's
monotone; `fit_recalibration.py`'s own log says "AP unchanged is EXPECTED").
Per-group maps CAN, because two different monotone maps reorder scores across
groups. Motivated by the ~3x skill spread across groups on 2020 test (SON
new-fire lift 13.9x vs MAM 4.1x; Tropical N 11.4x vs Temperate S 4.7x) -- a
single global threshold spends its top-k budget suboptimally when a 0.6 in a
weak group and a 0.6 in a strong group carry different evidence.
Deliberately isotonic (raw score -> observed fire RATE per group), NOT
percentile-normalization, which would flatten away genuine base-rate
differences (Tropical N really does burn more) and destroy signal. Fit on
2019, applied to 2020, groups with <`MIN_POS_PER_GROUP` positives fall back
to the global map. The script reports raw vs global vs grouped so the global
row acts as a built-in sanity check (must come out ~0%).

### RESULTS of the three Tier-1 validations (all completed 2026-07-29)

| Job | What | Outcome |
|---|---|---|
| 1764905 | ensemble + grouped recal | **SWA WINS: 0.4429 AUC-PR** (new best). prob-mean/rank-mean both LOSE to it. Grouped recal: no gain on AUC-PR. |
| 1764897 | resume combo, MAX_EPOCHS=40 | **cut WAS premature**: best val_ap 0.5144@ep30 (vs 0.5046@ep15), then declined and early-stopped ep36 -- genuinely converged this time. |
| 1764906 | combo + `OHEM_FRAC=0.1` | **best val_ap of the project: 0.5242@ep10**, and it got there in 10 epochs vs the combo's 15-30. Real-test pending. |

#### 1. Weight averaging (SWA) is a real, retraining-free win -- plain ensembling is NOT
2020 test, n=1500 patches, pooled 188.6M land px (job 1764905):

| member / rule | AUC-PR | ROC-AUC |
|---|---|---|
| `ps384_final` ep14 | 0.4230 | 0.9026 |
| combo ep15 | 0.4368 | 0.8955 |
| **SWA(combo ep15,17,18)** | **0.4429** | 0.9011 |
| prob-mean of the 3 | 0.4407 | 0.9034 |
| rank-mean of the 3 | 0.4387 | 0.9019 |

**SWA beat the best individual member (+1.4% over combo ep15, +4.7% over
`ps384_final`) for zero extra training** -- just averaging three near-equal
late checkpoints (0.5046/0.5035/0.5042) that val_ap could not distinguish.
That is exactly the variance-reduction this was aimed at.

**Both cross-run ensembling rules LOST to the best single member** (prob-mean
-0.49%, rank-mean -0.94%). Cause is not the combination rule but the member
pool: `ps384_final` at 0.4230 is materially weaker than the other two, and
averaging drags toward it. Lesson for future ensembles here: only pool members
of comparable strength; a 3-way average with one weak member is worse than
just taking the strong one. (rank-mean also came in slightly below prob-mean,
so calibration mismatch was NOT the binding problem in this particular pool.)

#### 2. Grouped (region x season) recalibration does not improve ranking
All 12 region x season groups had enough positives to fit (0% of test pixels
needed the global fallback). 2020 test, 900 patches:

| transform | AUC-PR | ROC-AUC |
|---|---|---|
| raw (no calibration) | 0.4402 | 0.9018 |
| global isotonic | 0.4370 | 0.9026 |
| GROUPED isotonic | 0.4382 | **0.9124** |

Grouped beats global (+0.28pp AUC-PR) but both sit BELOW raw, so **grouped
calibration is not a ranking improvement** -- raw scores rank best for AUC-PR.
It does give the best ROC-AUC of the three (0.9124, +1.2% over raw), i.e. it
improves discrimination across the whole score range while very slightly
hurting the top-heavy AP. Recalibration stays worth doing for its original
purpose (meaningful probability values on maps), not as an AP lever.

#### 3. FINAL Tier-1 verdict (job 1765197: 5 candidates, one shared 1500-patch pass)
Anchor check passed exactly -- member0 (SWA-combo) reproduced 0.4429 to 4dp
against job 1764905, so this eval harness is deterministic and the deltas
below are real, not sampling noise.

| candidate | val_ap (2019) | 2020 AUC-PR | ROC-AUC | new-fire lift |
|---|---|---|---|---|
| SWA(combo ep15,17,18) [anchor] | n/a | 0.4429 | 0.9011 | 12.6x |
| resume ep30 | 0.5144 | 0.4459 | 0.8959 | 14.1x |
| **SWA(resume ep28,29,30)** | n/a | **0.4485** | 0.8983 | **14.5x** |
| OHEM ep10 | **0.5242** (best val_ap ever) | 0.4314 | 0.8374 | 9.2x |
| SWA(OHEM ep10,15,16) | n/a | 0.4461 | 0.8468 | 11.9x |
| prob-mean of all 5 | -- | **0.4516** | 0.9000 | 13.4x |
| rank-mean of all 5 | -- | 0.4469 | 0.8846 | 12.4x |

**SWA confirmed a SECOND time, independently**: it lifted the resume run
(0.4459 -> 0.4485) exactly as it lifted the combo run (0.4368 -> 0.4429), and
it even partly rescued OHEM (0.4314 -> 0.4461, +3.4%). Weight-averaging late
checkpoints is now the single most reliable free gain found on this task --
always do it before declaring a run's best number.

**OHEM is REJECTED -- the sharpest val_ap-vs-test inversion of the whole
project.** It produced the highest `val_ap` ever measured here (0.5242, in
only 10 epochs) and the WORST real-test AUC-PR of the five (0.4314), with a
badly degraded ROC-AUC (0.8374 vs ~0.90) and new-fire lift collapsing to 9.2x
(vs 12.6x baseline). Worse than `trimmed`'s inversion. Likely mechanism is the
gradient-balance shift measured during implementation (pos:neg 0.134 -> 0.232
at frac=0.1): OHEM partly acts as a `pos_weight` increase, and `pos_weight` is
already known here to buy validation ranking at the cost of calibration and
generalization. A dose-response sweep was deliberately NOT run -- sweeping
doses of a lever that inverts on real test is not worth the GPU time.
`ohem_frac=0.0` (default) remains off; the code stays for reference.

**Ensembling: the earlier failure WAS the member pool, as predicted.** In job
1764905 prob-mean lost because `ps384_final` (0.4230) dragged the average
down. With 5 comparable-strength members, prob-mean now WINS outright
(0.4516, +0.68% over the best single member) -- a correctly-predicted result.
BUT it is not the recommended deliverable: it needs 5 forward passes per
inference (5x cost) and its new-fire lift (13.4x) is WORSE than the single
SWA(resume) model's 14.5x. **SWA(resume) is the pick**: one model, best
new-fire lift, within 0.7% of the ensemble on pooled AP.

**rank-mean lost to prob-mean both times** (0.4469 vs 0.4516 here; 0.4387 vs
0.4407 before). The original argument for preferring rank-mean -- that
members' differing calibration would distort probability averaging -- has now
been contradicted twice on real data. Calibration mismatch is evidently not
the binding constraint in these pools; prefer prob-mean here.

#### IMPORTANT correction to this script's own sanity check
`fit_grouped_recalibration.py` prints "global isotonic vs raw: must be ~0 --
monotone maps cannot reorder". It came out **-0.723%**, not ~0. That is not a
harness bug -- the assumption behind the check was subtly wrong. Isotonic
regression is *weakly* monotone: it is a piecewise-CONSTANT step function, so
it maps many distinct raw scores onto the same fitted value. It cannot
REORDER, but it can and does CREATE TIES, and ties genuinely change AP
(within-plateau ranking information is destroyed). So a small negative delta
is the expected floor for any isotonic map, and the right reading is
grouped-vs-global (+0.28pp), not grouped-vs-raw. Fix the printed wording
before relying on that line again.

## THE NEW-FIRE METRIC IS MEASURING NEAR-FIELD SPREAD, NOT NEW IGNITION (2026-07-30)

`newfire_definition_sweep.py` (job 1765273, 600 patches, 2020 test, model =
SWA(resume ep28,29,30)) swept the two hardcoded choices behind every
"new-fire lift" number in this project -- spatial radius and history window --
and the result reframes all of them.

### Share of fire px still counted NEW (%)
|  | 3d | 7d | 14d | 30d | 90d |
|---|---|---|---|---|---|
| r=0px | 59.7 | 58.9 | 58.2 | 57.2 | 56.1 |
| r=1px | 48.2 | 46.6 | 44.9 | 42.6 | 40.1 |
| **r=3px (current)** | **35.7** | 33.2 | 29.8 | 26.0 | 22.8 |
| r=5px | 29.6 | 26.5 | 22.4 | 18.3 | 15.3 |
| r=10px | 20.9 | 17.2 | 13.1 | 9.8 | 7.6 |
| r=20px | 11.5 | 8.6 | 5.8 | **3.9** | 2.7 |
| r=40px | 4.8 | 3.1 | 2.0 | 1.5 | 0.8 |

### NEW-fire lift @ top-0.5% -- THE KEY TABLE
|  | 3d | 7d | 14d | 30d | 90d |
|---|---|---|---|---|---|
| r=0px | 53.0 | 53.1 | 53.2 | 53.3 | 53.7 |
| r=1px | 33.0 | 33.4 | 33.7 | 34.1 | 34.5 |
| **r=3px (current)** | **11.1** | 11.5 | 11.7 | 12.1 | 13.2 |
| r=5px | 3.4 | 3.6 | 3.7 | 3.7 | 4.4 |
| r=10px | **0.1** | 0.1 | 0.1 | 0.1 | 0.2 |
| r=20px | 0.1 | 0.1 | 0.2 | 0.2 | 0.4 |
| r=40px | 0.2 | 0.3 | 0.5 | 0.8 | 0.2 |

### What this means
1. **89% of what the current metric calls "new fire" is within 20 km of, or a
   re-detection of, fire from the past 30 days** (35.7% -> 3.9% of fire px).
2. **Skill collapses with distance and is BELOW RANDOM past ~10 km**:
   11.1x @3px -> 3.4x @5px -> **0.1x @>=10px**. A lift of 0.1 means precision
   is 10x WORSE than the base rate -- the model's top-0.5% is actively
   anti-correlated with genuinely isolated ignitions, because it spends that
   budget near known fire. Not a small-sample artifact: 46 new-fire px/patch
   still qualify at r=10px/3d.
3. **Spatial radius dominates; the history window is secondary.** Widening
   3d->90d at fixed radius moves the share modestly (35.7->22.8 at r=3) and
   barely moves lift. The intermittent-VIIRS-re-detection concern is real but
   second-order; the spread-front contamination is the big one.
4. r=0px (i.e. "any pixel that did not itself burn at t-3", including adjacent
   ones) scores 53x -- adjacency to existing fire is trivially predictable,
   which is the same fire-proximity shortcut permutation importance found
   (fire_dist + fire_hist = ~78% of decisions).

### Consequences
- Past model-vs-model comparisons at r=3px stay VALID as RELATIVE rankings
  (same definition throughout, and the SWA/combo gains were real). They are
  just MISLABELLED in absolute terms: "14.5x new-fire lift" means
  **near-field spread**, not new ignition.
- Small numeric note: this sweep uses an exact euclidean disk, whereas
  `operational_stats_dual.py` uses `binary_dilation(iterations=3)`, whose
  default cross structuring element is a TAXICAB diamond -- strictly smaller
  than a euclidean r=3 disk. That is why r=3 here reads 11.1x vs the headline
  14.5x. Self-consistent, but the shipped metric is not actually a 3px radius.
- **Report two numbers going forward**: near-field spread lift (r=3px) and
  true new-ignition lift (r>=10px). The second is currently ~0.1x, i.e. the
  model has essentially NO skill at genuinely new isolated ignitions.
- That 0.1x is the real headroom. Fire-history features cannot help there by
  construction (there is no nearby fire to key on), which is precisely the
  regime a no-fire-history specialist model targets -- and it means the
  weather/terrain/lightning channels, currently near-zero in permutation
  importance, are the only signals available.

## `val_ap_newfire` is NOT a usable selection metric (job 1765273 stage 2)

The second `ModelCheckpoint(monitor="val_ap_newfire")` callback was added so
the best-by-new-fire epoch would survive `save_top_k` pruning. Those saved
checkpoints have now been scored on real 2020 test data for the first time,
with SWA(resume ep28,29,30) as a reproducibility anchor (it re-hit 0.4485 /
14.5x exactly -- harness deterministic for the third time).

| member | val_ap_newfire | 2020 AUC-PR | ROC-AUC | new-fire lift @0.5% |
|---|---|---|---|---|
| SWA(resume 28,29,30) [anchor, selected on val_ap] | -- | **0.4485** | 0.8983 | **14.5x** |
| SWA(resume 29,30,33) [avg over best-NEWFIRE epochs] | -- | 0.4485 | 0.8997 | 14.3x |
| bestnewfire ep33 resume | 0.0920 | 0.4397 | 0.9009 | 13.4x |
| bestnewfire ep8 combo | 0.0894 | 0.3970 | 0.8578 | 8.3x |
| bestnewfire ep8 OHEM | **0.1088 (highest ever)** | 0.4145 | 0.8113 | **8.8x** |
| prob-mean of the 5 | -- | 0.4448 | 0.8950 | 12.7x |
| rank-mean of the 5 | -- | 0.4348 | 0.8885 | 11.0x |

**The control fired.** Ranking the three `bestnewfire` checkpoints by the
metric that selected them (0.1088 > 0.0920 > 0.0894) gives real new-fire lifts
of 8.8x, 13.4x, 8.3x -- no monotone relationship, and the HIGHEST
`val_ap_newfire` ever recorded produced nearly the WORST real lift. Meanwhile
the two checkpoints selected on plain `val_ap` + SWA took the top two lift
slots (14.5x / 14.3x). So **`val_ap_newfire` does not predict real new-fire
lift and must be treated as a diagnostic only, never a selection criterion.**
Keep the callback (it costs nothing and preserves epochs for inspection) but
do not choose checkpoints with it. Selection stays: `val_ap` -> SWA the top
epochs -> confirm on real 2020 test.

Corollary: averaging over "best new-fire epochs" (SWA 29,30,33) was no better
than averaging over "best val_ap epochs" (SWA 28,29,30) -- 0.4485 both, lift
14.3x vs 14.5x. No advantage.

**Ensembling lost again (prob-mean 0.4448 < 0.4485)**, and again the pool
contained weak members (0.3970 and 0.4145). That is now 2 losses with mixed
pools vs 1 win with a comparable-strength pool -- the "only pool members of
similar strength" rule has held up three times.

**`SWA(resume ep28,29,30)` therefore still stands as the best model**:
0.4485 AUC-PR, 14.5x near-field lift. Nothing in this round beat it.

## FINAL RESULT: the no-fire-history specialist DOES carry far-field signal (job 1766529)

The specialist (`FIRE_HISTORY=0`, weather/terrain/lightning only) looked dead on
`val_ap`: 0.0116 peak vs the main model's 0.51, a ~50x collapse, and it
early-stopped at ep8 with a flat, trendless curve. Judged on `val_ap` it is a
total failure. **But `val_ap` is the wrong metric for it by construction** --
pooled AP is dominated by near-field pixels the specialist cannot see. Scored
where it was actually designed to work (r>=10px, genuinely isolated ignition):

### NEW-fire lift @ top-0.5%, far field
| model | r=10px | r=20px | r=40px |
|---|---|---|---|
| main model SWA(resume) | 0.11 | 0.12 | 0.20 |
| **specialist (no fire history)** | **0.60** | **0.43** | **1.55** |
| prob-mean of the two | 0.07 | 0.06 | 0.23 |

Specialist full grid at r=40px across windows: 1.55 / 2.33 / **3.44** / 1.98 /
2.87 (3d/7d/14d/30d/90d) -- **above random**, the only configuration in the
whole project to clear 1.0x on genuinely isolated ignitions.

### Three conclusions
1. **The far-field signal is real but weak.** 0.60 vs 0.11 at r=10px (46
   new-fire px/patch, so not a small-sample artifact) is a robust ~5x
   advantage. The r=40px numbers (1.55-3.44x) sit on 2-11 px/patch and are
   correspondingly noisy -- treat the direction as real, the magnitude as
   uncertain.
2. **Naive prob-mean DESTROYS it** (0.07, worse than either member). The main
   model's confident near-field predictions dominate the average and swamp the
   specialist's weak far-field signal. Combining these two requires a
   REGIME-SWITCHED rule -- use the specialist only where no recent fire is
   within ~10km, the main model elsewhere -- not a blend. This is a different
   failure mode from the earlier "weak member drags the pool" cases: here the
   members are good at DISJOINT regimes, so averaging is the wrong operator
   entirely.
3. **This vindicates the earlier caution about judging it on `val_ap`.** A
   model can be 50x worse on the pooled metric and still be the only thing
   that works in the regime that matters. Same lesson as `val_ap_newfire` and
   the OHEM inversion, in the opposite direction.

### Corrected claim
An earlier draft of the repo README stated the specialist "failed to learn at
all ... indicating the remaining predictors do not contain sufficient signal
for isolated ignition at this resolution." **That is wrong and was corrected
before publication.** Weather/terrain/lightning DO contain far-field ignition
signal; it is simply far too weak to register in a pooled metric dominated by
near-field spread.

### Next step if this work continues
Build the regime-switched combiner (specialist where `fire_dist` indicates no
fire within ~10km, main model otherwise) and evaluate at both radii. That is
the single most promising untested idea left, and it needs no new training --
both checkpoints already exist and are archived.

## Regime-switched combiner: REJECTED (job 1767172)

Following the specialist result, the obvious next step was to switch rather
than blend -- assign each pixel to a regime by distance-to-recent-fire and
score it with whichever model owns that regime, using within-regime percentile
ranks so the two models' very different output distributions stay
commensurable. `beta` controlled how much of the top-k budget the far field
could claim. Swept 3 switch radii x 6 betas (`regime_switch_eval.py`).

**No favourable trade exists.** Every setting either left the far field alone
or destroyed everything else:

| switch | beta | AUC-PR | all-fire | near r=3 | FAR r=10 |
|---|---|---|---|---|---|
| baseline (main only) | -- | **0.4556** | 104.7x | 13.48x | **2.51x** |
| 10px | 0.25 | 0.3505 | 104.0 | 11.98 | 0.02 |
| 10px | 0.75 | 0.3486 | 103.3 | 10.46 | 0.19 |
| 10px | 1.0 | 0.0331 | 45.7 | 1.27 | 1.92 |
| 10px | 1.5 | 0.0021 | 0.6 | 1.36 | 2.10 |

At beta<=0.75 the far field never gets enough budget to matter; at beta>=1.0
it does, and AUC-PR collapses by an order of magnitude (0.456 -> 0.033 ->
0.002). **Nothing beat the baseline's own far-field 2.51x.** Top-k is a fixed
budget and the specialist's far-field ranking is not good enough to justify
spending any of it -- the pixels it promotes are not the right ones often
enough. Note also that the percentile-rank construction alone costs AUC-PR
(0.4556 -> ~0.37 even at beta=0), because ranking within regimes discards
cross-regime magnitude information.

Conclusion: the specialist's far-field advantage over the main model is real
but too weak to exploit through score combination. Exploiting it would need a
genuinely better far-field model, not a better way of mixing this one.

### METHODOLOGICAL WARNING: far-field lift is definition-sensitive across scripts
The baseline far-field lift reads **2.51x** here but **0.11x** in
`newfire_definition_sweep.py` on the same model, patches and radius. The cause
is a real difference in how "recent fire" is built:
- `newfire_definition_sweep.py`: UNION of `y_fire_3d` over a W-day window read
  from the zarr (line ~175, `span.any(axis=0) & land`).
- `regime_switch_eval.py`: the SINGLE `fire_hist_t-3` channel (line 111).

The union marks more area as "known", so fewer pixels qualify as far-field new
fire and the survivors are genuinely more isolated -- hence much lower lift.
Both are internally consistent and each sweep's relative comparisons are valid,
but **far-field numbers must not be compared across the two scripts**. Any
published far-field figure has to state which definition produced it. The
README's 0.11x quotes the sweep (union) definition.

## Pitfalls
- v1 `conv_lstm_lit.py` had a recency-weights bug (never applied on channels-first
  input) — fixed 2026-07-14; backup at `conv_lstm_lit.py.bak_pre_recencyfix`.
- `zarr_8day_indexed_datamodule.py` is a stale broken stub — ignore it.
- Duplicated sampling block in `zarr_8day_datamodule.py::__getitem__` (v1 only;
  v2 datamodule is clean).
- NFS between fritz and tinyx can lag a few seconds after writes.
- LST tif nodata is −3.4e38 (mask with `< -1e30`), NDVI nodata −9999.
- **`focal_start_epoch` must never be reachable within `max_epochs`.** Every
  run this session that left it at its default of 8 crashed at epoch 7→8, no
  exceptions observed. Set `FOCAL_START_EPOCH` past `max_epochs` (env override
  in `train_convlstm_dual_fastproto.py`) for every new run.
- **A different, still-unexplained epoch-7/8 instability affects every
  full-feature+augmentation run, regardless of architecture.** Don't assume
  `POS_WEIGHT_ANNEAL_EPOCHS` timing is the cause — tested directly
  (stretched 8→14), the dip didn't move. Current leading suspect is the
  shared fixed training seed (`SEED=123`, never varied across any run this
  session) — see State 2026-07-27's seed-diagnostic job. Until resolved,
  don't conclude an architecture change is "bad" from one seed's result alone
  if it shows this same collapse-around-epoch-7 shape.
- **`fire_dist` is `exp(-distance_px/5.0)`, not raw distance** — it's already
  ≈0 beyond ~25-30px regardless of window/patch size. Don't reason about
  "truncation" for this specific feature the way it's valid to reason about
  raw fire-history distance transforms elsewhere; the 512px patch-size
  experiment failed partly because this assumption was wrong (see State
  2026-07-27).
- **`y_fire_3d` labels ocean as fire=1** in `aux_xgb_check_wind.py`'s `fh3`
  too (used for `downwind_align`'s distance transform), not just in the
  ConvLSTM validation path already documented above — fixed in
  `compare_xgb_convlstm_australia.py`/`xgb_australia_grid.py`
  (`fh3 = (... > 0) & lm`), NOT fixed in the original `aux_xgb_check_wind.py`
  (low priority, it already served its purpose as a one-off gate). Check for
  this pattern (`y_fire_3d[...] > 0` without a landmask multiply feeding a
  distance transform) before trusting any new script built on this data.
- **GeoTIFF/eval tile size must match the checkpoint's training patch size**,
  not be chosen independently — a mismatch reintroduces a real distribution
  shift in the fire-history distance feature (coastal inflation + tile seams).
  Check `PS`/`STRIDE` in `make_australia_geotiff.py` and `PATCH` in
  `operational_stats_dual.py` against the checkpoint's `job_<id>` training
  config before trusting any new checkpoint's maps or metrics.
- **`min_pos_pixels`-style density filters must scale with patch area**
  (`BASE_PATCH=256, BASE_MIN_POS=20` convention in `operational_stats_dual.py`),
  or cross-patch-size comparisons (128 vs 384px) silently favor the larger
  patch by sampling it from a lower relative fire density.
- **GRIP4 region numbering is not what the folder name implies**: Australia is
  **Region 7**, not Region 6 (Asia) or Region 5 (N. Hemisphere) — verify by
  checking the shapefile's actual lat bounds, not by trusting a filename or a
  prior assumption.
- **Isotonic recalibration is per-checkpoint.** `pos_weight` schedule changes
  (constant-100 vs annealed-100→20) change the raw sigmoid distribution enough
  that an old `.pkl` is silently wrong on a new checkpoint — always refit via
  `fit_recalibration.py` (env `CKPT`/`PATCH`/`OUT`) after any retrain.
- **`ModelCheckpoint(save_top_k=3)` prunes old bests as training continues** —
  a hardcoded checkpoint filename/epoch in a script (e.g. old default in
  `operational_stats.py`) silently goes stale and 404s or picks up a
  leftover file from a different run. Prefer env-var checkpoint paths (as
  `operational_stats_dual.py`, `fit_recalibration.py` now do) over hardcoding.
- **Absolute-bearing features need explicit augmentation correction; relative-
  angle features don't.** `aspect_sin/cos` (compass bearing) must be rotated
  by `_augment`'s k/flip; `downwind_align` (angle between two co-rotating
  vectors) is already invariant — don't "fix" it, that would double-correct
  and break it.
