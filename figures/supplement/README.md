# Supplementary figures

Figures that accompany the paper but are not part of it. The six result
figures (`fig_*`) currently show the provisional model B (results in
`results/experiments/smolder/full_model_windalign_seed123/`); they will be
replaced by the final model's versions under the same file names. The maps of
the input data (`map_*`) describe the data and do not depend on the model.
Scripts: `figures/make_national_figures.py`, `make_gt_vs_pred.py`,
`make_convergence_figure.py`, `make_explain_figures.py`, `make_input_maps.py`.

## Results

**fig_national_example.png.** National forecast issued on 15 November 2020,
the most fire-active issue day of the test year. Colours show the predicted
risk of fire on 16 to 18 November as a percentile across Australia's land; the
1 % of the land with the highest risk is hatched. Observed fire of the three
days is shown in green inside this area and in purple outside it. Displayed
at 0.04 deg.

**fig_national_maps.png.** Annual picture for 2020. (a) Mean predicted risk
score over all issue days. (b) Number of issue days with fire in the three-day
target window.

**fig_gt_vs_pred_2020.png.** Observed fire and predicted risk for four
forecasts of 2020 at full resolution (384 x 384 pixels each), chosen by a
deterministic rule (fire-active patches, one per month, the four with the most
fire). For each forecast, the left panel shows the fire detected in the three
days after the issue day, with the location of the patch in Australia (inset,
red outline), and the right panel the risk map issued on that day as a
percentile within the patch, with the 1 % of the patch with the highest risk
hatched. Observed fire is green inside this area and purple outside it; black
lines are the coastline.

**fig_convergence.png.** Training convergence. (a) Training and validation
loss. (b) Average precision on the 1024 validation patches of 2019 at every
epoch; the orange markers show the three epochs whose weights were averaged
into the model.

**fig_explain_prefire.png.** Median of the inputs over the model's look-back
window for pixels that burned in the next three days (orange) and for
randomly drawn pixels that did not burn (blue, dashed), by land cover, 2020:
soil moisture index, precipitation per 8 days and leaf area index over the
144 days of the slow branch, vapour pressure deficit over the 14 days of the
fast branch.

**fig_explain_conditions.png.** Conditions on the issue day by forecast
outcome, 2020, by climate zone: pixels in the national top 1 % that burned
(true positive), in the top 1 % without fire (false positive), burned outside
the top 1 % (false negative) and neither (true negative). Each violin is cut at
its group's 1st and 99th percentile; the thick line is the median, the thin
lines the interquartile range.

## Input data, 2015 to 2020

**map_landcover.png.** Land cover 2018 (Copernicus Global Land Cover LC100
v3.0.1, 100 m; the map the final model uses for issue days in 2020), grouped
into nine classes.

**map_koppen_geiger.png.** Köppen-Geiger climate zones 1991 to 2020 (Beck et
al. 2023, 1 km).

**map_agb_mean.png.** Above-ground biomass, mean of the ESA CCI Biomass maps
2015 to 2020 (Mg ha-1). The model uses the map of the year before the issue day.

**map_fire_days.png.** Number of days with a VIIRS active-fire detection
(S-NPP, 375 m, nominal and high confidence vegetation fires), 2015 to 2020,
maximum over 2 x 2 km blocks for display.

**map_sm_mean.png.** Mean SMIPS soil moisture index (fraction of the 90 cm
soil store that is filled), mean of the 8-day averages 2015 to 2020.

**map_precip_annual.png.** Mean annual precipitation 2015 to 2020 (ANUClimate
2.0, mm per year).

**map_lai_mean.png.** Mean leaf area index 2015 to 2020 (HiQ-LAI 500 m,
averaged to the 1 km grid; mean of the 8-day averages that the model reads,
with the 31-day delay).

**map_ndvi_mean.png.** Mean NDVI 2015 to 2020 (MODIS MOD09A1 8-day composites,
500 m).

**map_wind_mean.png.** Mean 10 m wind speed 2015 to 2020 (BARRA-C2 daily
means on the 1 km grid).

**map_vpd_mean.png.** Mean vapour pressure deficit at the daily maximum
temperature 2015 to 2020 (BARRA-C2, 4.4 km; es(Tmax) - e of the UTC day).

**map_tmax_mean.png.** Mean daily maximum 2 m air temperature 2015 to 2020
(BARRA-C2, 4.4 km).
