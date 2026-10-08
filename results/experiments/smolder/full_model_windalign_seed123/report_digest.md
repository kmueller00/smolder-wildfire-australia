
### Training

Source: `results/experiments/smolder/full_model_windalign_seed123/training_curve.csv` (columns epoch, val_ap, in_swa)

| quantity | value |
|---|---|
| epochs trained | 0 to 39 (40) |
| highest validation AP (epoch) | 0.2076 (32) |
| epochs averaged | 30, 31, 32 |
| validation AP of the averaged epochs | 0.2047, 0.2045, 0.2076 |

### Validation year 2019 (model selection)

Source: each run's `national_2019.json`: pooled_auc_pr, pooled_auc_pr_new, pooled_roc_auc, daily.auc_pr.mean, n_days

| run | pooled AUC-PR | AUC-PR new fire | ROC-AUC | mean daily AUC-PR | issue days | file |
|---|---|---|---|---|---|---|
| B (this run) | 0.2068 | 0.0096 | 0.9204 | 0.1804 | 352 | `results/experiments/smolder/full_model_windalign_seed123/national_2019.json` |
| A (40-epoch schedule) | 0.2062 | 0.0093 | 0.9322 | 0.1796 | 352 | `results/experiments/smolder/full_model_cos40_seed123/national_2019.json` |
| reference (25-epoch schedule) | 0.2008 | 0.0085 | 0.9254 | 0.1755 | 352 | `results/experiments/smolder/full_model_seed123/national_2019.json` |

### 2019 best adaptive threshold (thresholds that 2020 uses)

Source: `results/experiments/smolder/full_model_windalign_seed123/adaptive_budget_2019.json`: adaptive_best.<ranking>.<f1|f2>

| ranking | criterion | F1 | F2 | recall | precision | mean daily area | threshold (logit) |
|---|---|---|---|---|---|---|---|
| persistence | F1 | 0.1601 | 0.2163 | 0.2824 | 0.1117 | 0.140 % | -0.881 |
| persistence | F2 | 0.1494 | 0.2238 | 0.3350 | 0.0961 | 0.193 % | -1.098 |
| SMOLDER | F1 | 0.3007 | 0.3008 | 0.3009 | 0.3005 | 0.056 % | 0.956 |
| SMOLDER | F2 | 0.2673 | 0.3346 | 0.4021 | 0.2002 | 0.112 % | 0.357 |

### Test year 2020: pooled measures

Source: `results/experiments/smolder/full_model_windalign_seed123/test_2020/national_2020.json` and `results/national_2020_persistence.json`: pooled_auc_pr, pooled_auc_pr_new, pooled_roc_auc, pooled_roc_auc_new, daily.auc_pr.mean, base_rate, cells.pooled_auc_pr, n_days, fire_px_total, n_land_px

| measure | SMOLDER | persistence |
|---|---|---|
| pooled AUC-PR | 0.1167 | 0.0361 |
| pooled AUC-PR, new fire | 0.0058 | 0.0022 |
| pooled ROC-AUC | 0.9209 | 0.9358 |
| pooled ROC-AUC, new fire | 0.8755 | 0.8981 |
| mean daily AUC-PR | 0.1122 | 0.0377 |
| AUC-PR on 25 km cells | 0.3653 | 0.3691 |
| base rate | 0.0317 % | 0.0317 % |
| issue days | 350 | 350 |
| fire pixels (all days) | 762836 | 762836 |
| land pixels per day | 6885750 | 6885750 |

### Test year 2020: fixed share of land flagged each day (mean over days)

Source: topk_national[k].tpr, .precision, .lift, .tpr_new, .lift_new of both files

| land flagged | fire caught S | fire caught P | precision S | precision P | lift S | lift P | new fire caught S | new fire caught P |
|---|---|---|---|---|---|---|---|---|
| 0.1 % | 30.9 % | 21.7 % | 0.0989 | 0.0645 | 308.6 | 217.0 | 6.3 % | 0.1 % |
| 0.2 % | 37.5 % | 29.8 % | 0.0611 | 0.0472 | 187.3 | 149.0 | 11.2 % | 1.1 % |
| 0.5 % | 46.0 % | 39.3 % | 0.0304 | 0.0260 | 92.0 | 78.5 | 20.9 % | 8.3 % |
| 1.0 % | 53.0 % | 45.7 % | 0.0177 | 0.0154 | 53.0 | 45.7 | 30.8 % | 18.6 % |
| 2.0 % | 60.7 % | 52.6 % | 0.0103 | 0.0089 | 30.4 | 26.3 | 42.0 % | 29.5 % |
| 5.0 % | 70.8 % | 64.3 % | 0.0048 | 0.0043 | 14.2 | 12.9 | 56.6 % | 47.4 % |
| 10.0 % | 77.5 % | 75.3 % | 0.0026 | 0.0025 | 7.7 | 7.5 | 66.5 % | 63.7 % |

### Test year 2020 by season

Source: by_group.season.<class>.auc_pr, .roc_auc, .fire_px, .base_rate of both files

| class | AUC-PR S | AUC-PR P | ratio | ROC-AUC S | fire pixels | base rate |
|---|---|---|---|---|---|---|
| DJF | 0.1306 | 0.0563 | 2.32 | 0.8805 | 83745 | 0.0162 % |
| JJA | 0.1095 | 0.0327 | 3.35 | 0.9427 | 252349 | 0.0398 % |
| MAM | 0.0540 | 0.0168 | 3.21 | 0.9052 | 197421 | 0.0311 % |
| SON | 0.1742 | 0.0582 | 2.99 | 0.9192 | 229321 | 0.0366 % |

### Test year 2020 by climate zone

Source: by_group.kg.<class>.auc_pr, .roc_auc, .fire_px, .base_rate of both files

| class | AUC-PR S | AUC-PR P | ratio | ROC-AUC S | fire pixels | base rate |
|---|---|---|---|---|---|---|
| arid | 0.0849 | 0.0245 | 3.46 | 0.8453 | 178542 | 0.0091 % |
| other | 0.3132 | 0.0513 | 6.10 | 0.9042 | 1048 | 0.0311 % |
| temperate | 0.1134 | 0.0514 | 2.21 | 0.8532 | 129000 | 0.0581 % |
| tropical | 0.1331 | 0.0423 | 3.15 | 0.9138 | 454246 | 0.1925 % |

### Test year 2020 by latitude band

Source: by_group.band.<class>.auc_pr, .roc_auc, .fire_px, .base_rate of both files

| class | AUC-PR S | AUC-PR P | ratio | ROC-AUC S | fire pixels | base rate |
|---|---|---|---|---|---|---|
| central | 0.1096 | 0.0411 | 2.66 | 0.8675 | 154922 | 0.0119 % |
| north (>20S) | 0.1290 | 0.0398 | 3.24 | 0.9278 | 497288 | 0.1040 % |
| south (<30S) | 0.0831 | 0.0366 | 2.27 | 0.8588 | 110626 | 0.0176 % |

### Test year 2020: day by day against persistence

Source: `results/experiments/smolder/full_model_windalign_seed123/test_2020/comparison_2020.json`: vs_reference.SMOLDER.<metric>.days_won, .mean_diff, .ci95 (2000 bootstrap resamples of days)

| daily measure | days SMOLDER higher | mean difference | 95 % interval |
|---|---|---|---|
| auc_pr | 99.1 % | +0.0744 | [+0.0686, +0.0804] |
| lift_0.005 | 98.3 % | +13.4918 | [+12.5142, +14.5619] |
| lift_0.01 | 94.9 % | +7.3451 | [+6.7673, +8.0054] |
| tpr_new_0.01 | 95.7 % | +0.1223 | [+0.1145, +0.1305] |

### Test year 2020: false alarms

Source: false_alarms.<run>.fp_per_tp_<k>, .budget_for_ref_capture_<k>, .fp_change_at_ref_capture_<k>

| land flagged | false alarms per hit P | false alarms per hit S | S needs for P's catch | change in false alarms |
|---|---|---|---|---|
| 0.1 % | 14.5 | 9.1 | not defined (below the smallest share evaluated) |  |
| 0.5 % | 37.4 | 31.8 | 0.242 % | -52.8 % |
| 1.0 % | 64.0 | 55.3 | 0.484 % | -52.3 % |
| 5.0 % | 229.7 | 207.3 | 2.775 % | -44.7 % |
| 10.0 % | 399.2 | 382.3 | 8.023 % | -19.8 % |

### Test year 2020: one score threshold for all days, fixed on 2019 (adaptive daily area)

Source: `results/experiments/smolder/full_model_windalign_seed123/test_2020/adaptive_budget_2020.json` (thresholds_from = adaptive_budget_2019.json): adaptive_best.<ranking>.<f1|f2>; TP = recall x fire pixels, FN = fire pixels - TP, FP = fp_per_tp x TP

| ranking | 2019 criterion | F1 | F2 | recall | precision | mean daily area | false alarms per hit | TP | FN | FP | new fire caught | caught 0-3 / 3-10 / > 10 km |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| persistence | F1 | 0.1135 | 0.1629 | 0.2296 | 0.0754 | 0.096 % | 12.3 | 175153 | 587683 | 2148441 | 0.0 % | 59.6 % / 0.0 % / 0.0 % |
| persistence | F2 | 0.1027 | 0.1651 | 0.2777 | 0.0630 | 0.140 % | 14.9 | 211805 | 551031 | 3150757 | 0.0 % | 72.0 % / 0.0 % / 0.0 % |
| SMOLDER | F1 | 0.2232 | 0.2171 | 0.2132 | 0.2342 | 0.029 % | 3.3 | 162613 | 600223 | 531745 | 2.3 % | 52.2 % / 5.8 % / 0.1 % |
| SMOLDER | F2 | 0.1913 | 0.2449 | 0.3011 | 0.1402 | 0.068 % | 6.1 | 229706 | 533130 | 1408558 | 5.1 % | 70.9 % / 13.0 % / 0.5 % |

### Test year 2020: one fixed share of land for all days, chosen on 2019

Source: `results/experiments/smolder/full_model_windalign_seed123/operating_point_2019.json` (share with the highest F1/F2) read off in `results/experiments/smolder/full_model_windalign_seed123/test_2020/operating_point_2020.json`: rankings.<ranking>.<k|f1|f2|recall|precision|fp_per_tp>[i]

| ranking | 2019 criterion | share chosen on 2019 | F1 | F2 | recall | precision | false alarms per hit |
|---|---|---|---|---|---|---|---|
| persistence | F1 | 0.130 % | 0.0937 | 0.1476 | 0.2395 | 0.0582 | 16.2 |
| persistence | F2 | 0.234 % | 0.0760 | 0.1400 | 0.3188 | 0.0432 | 22.2 |
| B | F1 | 0.057 % | 0.1795 | 0.2172 | 0.2525 | 0.1393 | 6.2 |
| B | F2 | 0.146 % | 0.1256 | 0.2048 | 0.3532 | 0.0764 | 12.1 |

### Test year 2020: skill by distance to fire of the last 32 days

Source: `results/experiments/smolder/full_model_windalign_seed123/test_2020/distance_band_diagnostic_2020.json`: bands.<band>.fire_share, .auc_pr, .roc_auc, .base_rate, .capture_0.005, .lift_0.005, .capture_0.01, .lift_0.01

| band | share of fire | AUC-PR | ROC-AUC | base rate | caught at 0.5 % | lift at 0.5 % | caught at 1 % | lift at 1 % |
|---|---|---|---|---|---|---|---|---|
| 0-3 km | 58.6 % | 0.1987 | 0.8902 | 0.6921 % | 22.1 % | 44.1 | 31.9 % | 31.9 |
| 3-10 km | 23.3 % | 0.0229 | 0.8267 | 0.0919 % | 13.6 % | 27.3 | 18.8 % | 18.8 |
| > 10 km | 18.1 % | 0.0020 | 0.7676 | 0.0064 % | 12.0 % | 24.0 | 17.3 % | 17.3 |

### Test year 2020: fire caught by distance to fire of days D-2..D

Source: `results/experiments/smolder/full_model_windalign_seed123/test_2020/fire_distance_2020.json`: all_days.<smolder|persistence>.bands.<band>.fire_share, .captured_0.005, .captured_0.01

| band | share of fire | caught at 0.5 % S | caught at 0.5 % P | caught at 1 % S | caught at 1 % P |
|---|---|---|---|---|---|
| 0 km | 10.4 % | 98.6 % | 100.0 % | 99.3 % | 100.0 % |
| 1-3 km | 28.1 % | 92.8 % | 97.9 % | 96.6 % | 100.0 % |
| 3-10 km | 19.8 % | 46.0 % | 15.9 % | 64.6 % | 50.5 % |
| 10-25 km | 18.0 % | 7.2 % | 0.0 % | 19.4 % | 0.5 % |
| > 25 km | 23.6 % | 5.6 % | 0.0 % | 9.7 % | 0.0 % |

### Test year 2020: permutation importance

Source: `results/experiments/smolder/full_model_windalign_seed123/test_2020/explain_2020.json`: groups.<input>.retention (displaced = 1 - retention), .ap_drop; 1500 fire-active patches, top 1.0 %, base AUC-PR 0.1369 (base_auc_pr)

| input | top-1 % pixels displaced | relative fall of AUC-PR |
|---|---|---|
| fire history | 70.4 % | 56.9 % |
| fuel age | 17.0 % | 5.6 % |
| NDVI (fast branch) | 14.5 % | 6.2 % |
| fire radiative power | 13.8 % | 31.9 % |
| biomass | 9.5 % | 1.6 % |
| downwind alignment | 8.8 % | 1.1 % |
| aspect | 8.8 % | 1.3 % |
| vapour pressure deficit | 8.6 % | 1.0 % |
| slope | 8.1 % | 1.5 % |
| leaf area index (500 m) | 7.1 % | 2.1 % |
| wind direction (u, v) | 6.2 % | 0.0 % |
| land cover | 6.0 % | 0.1 % |
| climate zone | 5.9 % | -0.2 % |
| soil moisture | 4.5 % | -0.1 % |
| precipitation | 4.3 % | 0.3 % |
| land surface temperature | 3.6 % | 0.7 % |
| wind speed | 2.9 % | 0.1 % |
| elevation | 1.7 % | 0.0 % |

### Test year 2020: the event maps

Source: `results/experiments/smolder/full_model_windalign_seed123/test_2020/event_maps_2020.json`: events.<event>.date, .fire_px, .<ranking>.caught, .false_alarms; budget 0.130 % per day; adaptive = 2019 threshold 0.238

| event | date | fire pixels | caught P | false alarms P | caught S | false alarms S | caught S adaptive | false alarms S adaptive |
|---|---|---|---|---|---|---|---|---|
| well 1 | 2020-11-21 | 526 | 18.8 % | 554 | 76.6 % | 355 | 84.6 % | 539 |
| well 2 | 2020-08-28 | 418 | 23.2 % | 870 | 83.7 % | 911 | 88.3 % | 1233 |
| well 3 | 2020-07-14 | 609 | 12.2 % | 176 | 57.8 % | 582 | 57.8 % | 580 |
| well 4 | 2020-11-16 | 477 | 19.5 % | 214 | 52.0 % | 284 | 65.2 % | 707 |
| poorly 1 | 2020-06-02 | 1504 | 14.0 % | 1531 | 25.4 % | 2264 | 23.1 % | 1719 |
| poorly 2 | 2020-07-27 | 1366 | 11.5 % | 1159 | 13.1 % | 1063 | 13.3 % | 1114 |
| poorly 3 | 2020-02-06 | 1065 | 4.5 % | 112 | 14.6 % | 386 | 2.1 % | 90 |
| poorly 4 | 2020-05-09 | 983 | 7.2 % | 1232 | 14.1 % | 1257 | 11.7 % | 919 |
