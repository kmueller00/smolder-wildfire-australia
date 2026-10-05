# Handoff: anomaly and trend features for new fire (Stage 0)

Question: do anomaly and trend features of soil moisture, LAI, VPD, LST and
precipitation separate fire from no fire better than the raw variables, in
particular far from existing fire? No model is involved; this is a data
diagnostic on the selection year 2019. 2020 was not used.

Code: commit `c05ed8c` (`smolder/evaluation/anomaly_feature_diagnostic.py`,
`smolder/data/build_fire_distance.py`, `figures/make_explain_figures.py`).
The tables under Results are from the first run (`results/anomaly_feature_diagnostic_2019_gap2016.json`); the current numbers are in the next section.

## Rerun after the repair of the 2016 cube (current numbers)

The 2016 daily cube had empty days 163 to 365 (repaired in commit d7192f0), which also entered the 2015-2018 climatology. The diagnostic was rerun on the repaired cube; `results/anomaly_feature_diagnostic_2019.json` holds the rerun, `results/anomaly_feature_diagnostic_2019_gap2016.json` the first run that the tables further below still show. The conclusion is unchanged: `z_vpd` is the only marked feature.

| Feature | Raw variable | (a) all: feature | (a) all: raw | (b) > 10 km: feature | (b) > 10 km: raw | Gain in (b) | (b) feature, first run |
|---|---|---|---|---|---|---|---|
| z_sm | sm | 0.578 | 0.510 | 0.567 | 0.546 | +0.021 | 0.582 |
| z_lai | lai | 0.564 | 0.783 | 0.545 | 0.715 | -0.170 | 0.558 |
| z_vpd | vpd | 0.583 | 0.534 | 0.618 | 0.581 | +0.037 | 0.625 |
| z_lst | lst | 0.518 | 0.503 | 0.570 | 0.558 | +0.013 | 0.570 |
| sm_slope_144d | sm | 0.663 | 0.510 | 0.587 | 0.546 | +0.040 | 0.587 |
| lai_slope_144d | lai | 0.611 | 0.783 | 0.505 | 0.715 | -0.210 | 0.505 |
| vpd_change_14d | vpd | 0.549 | 0.534 | 0.557 | 0.581 | -0.023 | 0.557 |
| ppt_pct_144d | ppt_144d | 0.507 | 0.607 | 0.540 | 0.591 | -0.051 | 0.508 |

## Method

- Climatology: per pixel, mean and standard deviation of SM, LAI, VPD and LST
  for every 8th day of year, each over the days within +-15 days of that day
  in 2015-2018 only, interpolated linearly in between. Stored in
  `climatology_2015_2018.zarr` (data folder) for a later training run.
- Features at issue day D (no value after D is used):
  standardized anomaly `z_sm`, `z_lai`, `z_vpd`, `z_lst`; OLS slope of SM and
  LAI over D-143..D; VPD change over D-13..D (OLS slope times 13 days);
  144-day precipitation sum as a percentile of the pixel's own 2015-2018
  144-day sums ending within +-15 days of the same day of year.
- Sample: every fire pixel of the 2019 issue days (1,345,605 pixel-days, target
  fire on D+1..D+3) and 399,802 random land pixel-days without fire. ROC-AUC
  does not depend on the class ratio, so this estimates the same ROC-AUC as
  random sampling with far fewer negatives.
- (b) keeps pixels more than 10 km from any fire of the last 32 days
  (D-31..D, `fire_dist30_continental.zarr`): 237,611 fire pixel-days.
- ROC-AUC is shown oriented, max(AUC, 1 - AUC); the raw values with their
  direction are in the JSON. Marked: oriented ROC-AUC in (b) >= 0.60 and
  >= 0.03 above the raw variable.

## Results

Oriented ROC-AUC for fire in the next three days, 2019:

| Feature | Raw variable | (a) all: feature | (a) all: raw | (b) > 10 km: feature | (b) > 10 km: raw | Gain in (b) | Marked |
|---|---|---|---|---|---|---|---|
| z_sm | sm | 0.603 | 0.510 | 0.582 | 0.546 | +0.036 |  |
| z_lai | lai | 0.577 | 0.783 | 0.558 | 0.715 | -0.157 |  |
| z_vpd | vpd | 0.594 | 0.534 | 0.625 | 0.581 | +0.045 | **yes** |
| z_lst | lst | 0.518 | 0.503 | 0.570 | 0.558 | +0.013 |  |
| sm_slope_144d | sm | 0.663 | 0.510 | 0.587 | 0.546 | +0.040 |  |
| lai_slope_144d | lai | 0.611 | 0.783 | 0.505 | 0.715 | -0.210 |  |
| vpd_change_14d | vpd | 0.549 | 0.534 | 0.557 | 0.581 | -0.023 |  |
| ppt_pct_144d | ppt_144d | 0.519 | 0.607 | 0.508 | 0.591 | -0.083 |  |

(b) by land cover, oriented ROC-AUC:

| Feature | grassland | shrubland | open forest | closed forest | cropland |
|---|---|---|---|---|---|
| z_sm | 0.542 | 0.607 | 0.553 | 0.653 | 0.508 |
| z_lai | 0.552 | 0.553 | 0.545 | 0.602 | 0.501 |
| z_vpd | 0.617 | 0.651 | 0.607 | 0.681 | 0.518 |
| z_lst | 0.576 | 0.587 | 0.549 | 0.612 | 0.540 |
| sm_slope_144d | 0.505 | 0.589 | 0.537 | 0.654 | 0.515 |
| lai_slope_144d | 0.511 | 0.543 | 0.513 | 0.524 | 0.559 |
| vpd_change_14d | 0.534 | 0.601 | 0.558 | 0.597 | 0.589 |
| ppt_pct_144d | 0.505 | 0.521 | 0.507 | 0.567 | 0.567 |
| sm | 0.571 | 0.584 | 0.505 | 0.586 | 0.549 |
| lai | 0.657 | 0.637 | 0.624 | 0.610 | 0.592 |
| vpd | 0.610 | 0.703 | 0.630 | 0.665 | 0.515 |
| lst | 0.602 | 0.675 | 0.610 | 0.581 | 0.503 |
| ppt_144d | 0.567 | 0.504 | 0.510 | 0.506 | 0.625 |

Figure: `figures/fig_explain_prefire_newfire.png`, median inputs before fire
for pixels more than 10 km from fire of the last 32 days, burned against not
burned, by land cover. Unlike the original figure (model slow bins, 2020
patches), its 8-day bins end on the issue day and the pixels are drawn
from all of Australia in 2019.

## Reading

- One feature is marked: `z_vpd`, the VPD anomaly (0.625 against
  0.581 for raw VPD far from fire).
- The gain comes mostly from pooling across Australia. Within single land
  covers the VPD anomaly beats raw VPD only in closed forest and grassland and
  is worse in shrubland and open forest. Raw VPD mixes regions with very
  different normal VPD; the anomaly removes that. The model already receives
  land cover, climate zone and the spatial context of its patch, so its gain
  from `z_vpd` will likely be smaller than the pooled +0.045.
- The soil-moisture slope over 144 days is not marked (0.587 overall) but is
  the largest single gain within a land cover: closed forest
  0.654 against 0.586 for raw SM, matching the steady drying in the figure.
- Raw LAI remains the strongest single variable far from fire (0.715).
- The precipitation percentile carries almost nothing on 2019 (0.508). 2019
  was Australia's driest year on record, so most 144-day sums lie below the
  2015-2018 range and the percentile is compressed near zero.

## Commands

```
# 32-day distance store (once)
SMOLDER_DATA=<cubes> WINDOW=30 python -m smolder.data.build_fire_distance
# diagnostic (whole node, about 20 min with 8 workers; writes the climatology too)
SMOLDER_DATA=<cubes> WORKERS=8 OUT=results/anomaly_feature_diagnostic_2019.json \
    python -m smolder.evaluation.anomaly_feature_diagnostic
# figure
cd figures && python -c "import make_explain_figures as m; m.prefire_newfire()"
```

## Stage 1 (not started, needs confirmation)

Add `z_vpd` as a daily fast-branch channel behind a switch (default off),
train one run with it on the leak-free pipeline, and compare with the leak-free
control that is already training (`pfw_a0_seed123`, same seed and settings),
then rerun permutation importance.
