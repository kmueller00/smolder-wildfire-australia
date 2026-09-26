# SMOLDER

**S**low-**M**emory **O**perator with **L**atent **D**ual-attention for
**E**stimating fire **R**isk: a per-pixel forecast of wildfire occurrence over
the next three days for continental Australia at 0.01° (~1 km) resolution.

Fire needs fuel that is dry enough and weather that lets it ignite and spread.
The two change on different time scales: fuel state over months, fire weather
over days. SMOLDER therefore reads two input streams, 144 days of vegetation,
soil moisture and rainfall and 14 days of fire weather and recent fire. Each
stream has its own ConvLSTM encoder, and the two are fused by per-pixel
cross-attention.

![architecture](figures/fig_smolder_architecture.png)

## Task

For an issue day *D* and every land pixel, SMOLDER outputs a risk score for
**at least one VIIRS active-fire detection on day D+1, D+2 or D+3**. Inputs
contain no information later than *D*. The output is a ranking score, not a
calibrated probability (see *Limitations*).

- Grid: 3474 × 4110 px, EPSG:4326, 0.01° pixels, origin 112.905° E / 9.005° S
- Split by year: train 2015–2018, validation 2019 (checkpoint selection),
  test 2020 (used only for the numbers below)

## Inputs

![input channels](figures/fig_inputs.png)

| Variable | Source | Native resolution | Used as |
|---|---|---|---|
| Soil moisture index | CSIRO SMIPS (Soil Moisture Integration and Prediction System); also defines the model grid | 0.01°, daily | slow branch |
| Precipitation | ANUClimate 2.0 daily rainfall (ANU / NCI) | 0.01°, daily | slow branch (8-day sums) |
| Leaf area index | HiQ-LAI, reprocessed MODIS LAI (Yan et al., 2024) | 8-day | slow branch |
| Vapour pressure deficit | ERA5 (Hersbach et al., 2020), at daily maximum temperature | 0.25°, daily | fast branch |
| Land-surface temperature | Gap-filled daily MODIS LST (Zhang et al., 2022) | 1 km, daily | fast branch |
| 10 m wind speed | BARRA2, Bureau of Meteorology regional reanalysis | ~12 km, daily | fast branch |
| Active fire (inputs and target) | VIIRS 375 m active fire (VNP14IMG, NASA FIRMS; Schroeder et al., 2014); vegetation fires, nominal and high confidence | 375 m, daily | fire history, target |
| Above-ground biomass | ESA Climate Change Initiative Biomass | 100 m | static |
| Land cover | Copernicus Global Land Service LC100 v3.0.1 (Buchhorn et al., 2020) | 100 m | embedding |
| Climate zone | Köppen–Geiger 1991–2020 (Beck et al., 2023) | 1 km | embedding |

All fields are resampled to the SMIPS grid. Gaps in the 8-day LAI are filled
by carrying the last observation forward. The original cubes also hold NDVI
(MODIS MOD09A1), which the model does not read.

## Training

| Setting | Value |
|---|---|
| Encoders | ConvLSTM, 1 layer, 64 hidden channels, 5 × 5 kernels |
| Fusion | multi-head cross-attention (4 heads); the fast state queries the slow state per pixel |
| Parameters | 1.09 M |
| Patches | 384 × 384 px, 2000 per epoch; half of the draws must contain ≥ 45 fire pixels |
| Batch | 2, gradient accumulation 4 |
| Optimiser | AdamW, lr 3·10⁻⁴, weight decay 0.01, cosine schedule with a 25-epoch period, up to 40 epochs, early stopping on validation AP (patience 6) |
| Loss | BCE on soft labels (fire 0.9, background 0.02), plus 0.3 · Dice for epochs 0–2; positive weight annealed 100 → 20 over 8 epochs; isolated fire pixels up-weighted (γ = 2); auxiliary loss on every fast time step (weight 0.3) |
| Sampling | 30 % of patches must contain fire absent from the fire history; fire-history channels zeroed for 30 % of samples |
| Released weights | average of the three best checkpoints by validation AP |

The exact configuration is in [configs/smolder.env](configs/smolder.env).

![training convergence](figures/fig_convergence.png)

## Results, 2020 hold-out year

> **Pending.** The model is being retrained after the correction described
> below. AUC-PR, ROC-AUC, lift tables and the figures in this section will be
> added when training and evaluation have finished.

![lift curve](figures/fig_lift_curve.png)

**Evaluation protocol.** 1500 patches of 384 × 384 px are drawn with a
fixed seed from fire-active scenes (≥ 45 fire pixels and ≥ 50 % land per
patch). The results therefore measure how well SMOLDER ranks pixels *where
fire occurs*, not the continent-wide false-alarm rate. AUC-PR and ROC-AUC are
pooled over all land pixels of all patches. TPR and lift are computed per patch
(top-k of that patch's land pixels) and averaged. *New fire* is target fire
more than 3 px (taxicab distance) from any fire in the newest fire-history
window, i.e. fire detected on days D−2 to D. Reproduce with
`python -m smolder.evaluation.evaluate`.

## Observed fire and predicted risk

Four 2020 forecasts. Left: fire detected in the three days after the issue
date. Right: SMOLDER's risk map issued on that date, as a within-patch
percentile, with the top-1 % area outlined.

![observed vs predicted](figures/fig_gt_vs_pred_2020.png)

## What "new-fire lift" measures

Whether a fire pixel counts as "new" depends on a distance threshold and on
how much fire history is considered, and the resulting lift depends on both.

![new-fire distance dependence](figures/fig_newfire_distance_decay.png)

Results for the retrained model will be added here.

## Limitations

- **Skill depends on fire that is already burning.** During development,
  skill was concentrated near recent fire: spread, flare-ups and
  re-detection. The previous model ranked isolated new ignitions (more than
  10 km from recent fire) no better than random; this will be re-measured
  for the retrained model. The immediate cause of
  an isolated ignition (a lightning strike, a spark) is a point event with no
  precursor in any 1 km daily field, so part of this limit is probably
  irreducible at this resolution.
- **Additional 1 km predictors did not help.** During development, lightning
  climatology, terrain (elevation, slope, aspect), fuel age, downwind
  alignment, road distance, population, a McArthur forest fire danger index
  from SILO reanalysis, and Sentinel-2 live fuel moisture aggregated to 1 km
  were each tested. None improved hold-out skill beyond evaluation noise, so
  none is used. These tests were run before the fire-history correction
  described below, while the model still had access to first-day detections.
  They should be repeated before the plateau is taken as settled. Likely
  reasons for a genuine plateau: sub-kilometre fuel continuity and ignition
  sources are averaged away at 1 km, and weather at 12–25 km resolution
  varies little between neighbouring pixels.
- **Labels are satellite detections.** VIIRS misses fires under cloud or
  canopy, small or short-lived fires, and fires between overpasses. Missed
  detections enter as negatives, both as targets and in the fire history.
- **One test year.** 2020 includes the end of the 2019–20 Black Summer fire
  season. Performance in other years has not been measured.
- **Scores are not probabilities.** The positive-class weighting compresses
  the raw output. `smolder.evaluation.fit_recalibration` fits an isotonic map
  on the 2019 validation year. This changes the probability values but not
  the ranking.

## Correction to earlier versions

Versions of this repository before v1.0 reported a 2020 AUC-PR of 0.4485 and
a new-fire lift of 14.5×. These results are withdrawn. In that model the
most recent fire-history input covered detections up to and including the
first day of the target window, a one-day overlap between input and label:
30 % of target fire pixels were visible in the input only through this
overlap. The released model is retrained with fire history that ends on the
issue day for every time step. Results for the corrected model will be
added to this README when its training has finished.

## Data

| Zenodo DOI | Content | Size |
|---|---|---|
| [10.5281/zenodo.22115979](https://doi.org/10.5281/zenodo.22115979) | `cube_2020_zenodo.tar` → `cube_2020_zenodo.zarr`: daily cube for the 2020 test year | 35 GB |
| [10.5281/zenodo.21749290](https://doi.org/10.5281/zenodo.21749290) | `cube_slow_8day.tar` → `cube_slow_8day.zarr`: LAI, soil moisture and precipitation in 8-day bins, 2015–2020; `aux_rasters.tar`: static rasters not used by the released model | 16 GB + 2.8 GB |

Download both records and extract the archives into one directory, for
example `tar -xf cube_2020_zenodo.tar && tar -xf cube_slow_8day.tar`; set
`SMOLDER_DATA` to that directory. The two records are sufficient to reproduce
every result in this README. The
2015–2019 daily cubes (~250 GB) are not archived because they exceed the
record size limit.

**Daily cube** (`zarr` v2, one store per year):

| Array | Shape | Content |
|---|---|---|
| `X` | (days, 3474, 4110, 6) | channels listed in the attribute `channels`: soil moisture, wind, VPD, precipitation, LST, LAI (native units; missing = 0) |
| `y_fire_3d` | (days, 3474, 4110) | 1 if VIIRS detected fire at the pixel on day t+1, t+2 or t+3 |
| `y_fire_3d_valid` | (days,) | per-day validity flag of `y_fire_3d` |
| `landmask` | (3474, 4110) | 1 = land |
| `agb`, `landcover`, `koppen_geiger` | (3474, 4110) | static layers read by the model |
| `elevation`, `slope`, `aspect_sin`, `aspect_cos`, `lightning` | (3474, 4110) | static layers not read by the released model |
| other `y_fire_*` | | alternative targets, not used |

Attribute `time` lists the ISO date of each day. **`y_fire_3d` is 1 over the
ocean**; apply `landmask` before any metric or distance computation.

## Installation and use

```bash
git clone https://github.com/kmueller00/smolder-wildfire-australia
cd smolder-wildfire-australia
pip install -e .

export SMOLDER_DATA=/path/to/zarr/stores   # holds cube_2020_zenodo.zarr and cube_slow_8day.zarr

python -m smolder.evaluation.evaluate                   # headline metrics (needs checkpoints/smolder_swa.ckpt)
python -m smolder.evaluation.newfire_definition_sweep   # distance dependence of new-fire lift

# training (needs the 2015-2019 cubes)
set -a; source configs/smolder.env; set +a
python -m smolder.training.train
```

Figures are regenerated from `results/` without data or GPU:
`cd figures && python make_<name>.py`.

```
smolder/
  data/         data pipeline, slow-cube builder, channel statistics
  models/       ConvLSTM backbone and Lightning modules
  training/     training entry point
  evaluation/   evaluation, distance sweep, checkpoint averaging, recalibration
configs/        training configuration of the released model
checkpoints/    released weights (smolder_swa.ckpt, added after retraining)
results/        evaluation outputs behind every figure and table
figures/        figures and the scripts that draw them
```

## References

- Beck, H. E. et al. (2023). High-resolution (1 km) Köppen–Geiger maps for 1901–2099 based on constrained CMIP6 projections. *Scientific Data* 10, 724.
- Buchhorn, M. et al. (2020). Copernicus Global Land Service: Land Cover 100 m, collection 3.
- Hersbach, H. et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society* 146, 1999–2049.
- Schroeder, W. et al. (2014). The New VIIRS 375 m active fire detection data product. *Remote Sensing of Environment* 143, 85–96.
- Yan, K. et al. (2024). HiQ-LAI: a high-quality reprocessed MODIS leaf area index dataset with better spatiotemporal consistency from 2000 to 2022. *Earth System Science Data* 16.
- Zhang, T., Zhou, Y., Zhu, Z., Li, X., Asrar, G. R. (2022). A global seamless 1 km resolution daily land surface temperature dataset (2003–2020). *Earth System Science Data* 14, 651–664.

Each input dataset remains under its provider's licence.

## Citation

Please cite the software (metadata in [CITATION.cff](CITATION.cff)) and the
data records above.

```bibtex
@software{mueller_smolder_2026,
  author  = {Müller, Korbinian},
  title   = {SMOLDER: next-3-day wildfire risk for continental Australia at 1 km},
  year    = {2026},
  version = {1.0},
  url     = {https://github.com/kmueller00/smolder-wildfire-australia}
}
```

## License

MIT for the code (see LICENSE).
