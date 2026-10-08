# Supplementary figures

Figures that accompany the paper but are not part of it. The result figures
(`fig_*`) show the final model (results in
`results/experiments/smolder/causal_b_50ep_seed123/test_2020/` and
`results/experiments/final/causal_b_50ep_seed123/`). The maps of the input
data (`map_*`) describe the data and do not depend on the model.
Scripts: `figures/make_national_figures.py`, `make_gt_vs_pred.py`,
`make_convergence_figure.py`, `make_explain_figures.py`, `make_input_maps.py`.

## Results

**fig_national_maps.png.** Annual picture for 2020. (a) Mean predicted risk
score over all issue days. (b) Number of issue days with fire in the three-day
target window.

**fig_gt_vs_pred_2020.png.** Observed fire and predicted risk for four
forecasts of 2020 at full resolution (384 x 384 pixels each), chosen by a
deterministic rule (fire-active patches, one per month, the four with the most
fire). For each forecast, the left panel shows the fire detected in the three
days after the issue day, with the location of the patch in Australia (inset,
red outline), and the right panel the risk map issued on that day as a
percentile within the patch. Hatched: the pixels whose score lies above the
threshold chosen on 2019 (best F2), the same threshold for every forecast, so
the flagged area varies with the situation. Observed fire is green inside this
area and purple outside it; black lines are the coastline.

**fig_convergence.png.** Training convergence. (a) Training and validation
loss. (b) Average precision on the 1024 validation patches of 2019 at every
epoch; the orange markers show the three epochs whose weights were averaged
into the model.

**fig_explain_prefire.png.** How the pixels of each forecast outcome in 2020
differed beforehand from other land of the same patch and issue day: fire
SMOLDER caught, fire it missed and its false alarms (threshold chosen on 2019,
best F2; outcomes from the stored national scores), for (a) soil moisture
index, (b) rain per 8 days, (c) leaf area index, (d) vapour pressure deficit,
(e) maximum air temperature and (f) NDVI. Lines: mean difference over 1500
fire-active patches, weighted by the outcome's pixels in each patch; bands:
95 % bootstrap interval over patches. Slow inputs are 8-day periods drawn at
their centre; leaf area index is drawn at the date it was observed, 31 days
before the model receives it, and NDVI at the date of the composite the model
reads, 7 days before.

**fig_explain_conditions.png.** Where the forecast outcomes of 2020 lie
relative to fire already burning, with the threshold chosen on 2019 (best
F2): share of the fire that SMOLDER caught, of the fire it missed, of its
false alarms and of all other land, by distance to the nearest fire detected
on the issue day or the two days before. Pixels are weighted to the true size
of their outcome in each patch.

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
