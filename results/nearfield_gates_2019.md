Near-field gates, 2019 (gradient-boosted trees on pixels within 25 px of fire of days D-2..D, fitted on odd and scored on even issue days; S = with the released model's score as a feature). Gains are differences in AUC-PR, relative gain in brackets, with the day-bootstrap 95 % interval.

## frp_gate_2019.json

Feature sets:

- A: log_dist
- C: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth
- B: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, log_frp, n_det, night_share, log_cluster_frp
- S: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, smolder_logit
- SF: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, log_frp, n_det, night_share, log_cluster_frp, smolder_logit

### 0-3 km: 314730 fire pixels, base rate 0.06094

| set | AUC-PR | ROC-AUC |
|---|---|---|
| A | 0.1091 | 0.685 |
| C | 0.2348 | 0.807 |
| B | 0.2624 | 0.820 |
| S | 0.2567 | 0.838 |
| SF | 0.2899 | 0.836 |

| gain | d AUC-PR | relative | 95 % interval |
|---|---|---|---|
| B-C | +0.0275 | +12 % | [+0.0174, +0.0387] |
| SF-S | +0.0332 | +13 % | [+0.0244, +0.0459] |

### 3-10 km: 141620 fire pixels, base rate 0.00589

| set | AUC-PR | ROC-AUC |
|---|---|---|
| A | 0.0113 | 0.666 |
| C | 0.0385 | 0.772 |
| B | 0.0394 | 0.777 |
| S | 0.0439 | 0.797 |
| SF | 0.0473 | 0.794 |

| gain | d AUC-PR | relative | 95 % interval |
|---|---|---|---|
| B-C | +0.0009 | +2 % | [-0.0003, +0.0024] |
| SF-S | +0.0034 | +8 % | [+0.0011, +0.0062] |

### 10-25 km: 89955 fire pixels, base rate 0.00115

| set | AUC-PR | ROC-AUC |
|---|---|---|
| A | 0.0017 | 0.609 |
| C | 0.0033 | 0.655 |
| B | 0.0034 | 0.664 |
| S | 0.0049 | 0.724 |
| SF | 0.0048 | 0.701 |

| gain | d AUC-PR | relative | 95 % interval |
|---|---|---|---|
| B-C | +0.0001 | +4 % | [-0.0001, +0.0005] |
| SF-S | -0.0001 | -2 % | [-0.0005, +0.0005] |

## nearfield_gate_2019.json

Feature sets:

- A: log_dist
- C: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth
- B: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, log_frp, n_det, night_share, log_cluster_frp
- CW: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, wind_speed_q, wind_align, wind_push, gust_push, upslope, slope_p, aspect_align
- S: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, smolder_logit
- SF: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, log_frp, n_det, night_share, log_cluster_frp, smolder_logit
- SW: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, wind_speed_q, wind_align, wind_push, gust_push, upslope, slope_p, aspect_align, smolder_logit
- SWF: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, wind_speed_q, wind_align, wind_push, gust_push, upslope, slope_p, aspect_align, f_wind_align, f_wind_push, smolder_logit
- SFW: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, log_frp, n_det, night_share, log_cluster_frp, wind_speed_q, wind_align, wind_push, gust_push, upslope, slope_p, aspect_align, smolder_logit

### 0-3 km: 314730 fire pixels, base rate 0.06094

| set | AUC-PR | ROC-AUC |
|---|---|---|
| A | 0.1091 | 0.685 |
| C | 0.2348 | 0.807 |
| B | 0.2624 | 0.820 |
| CW | 0.2355 | 0.809 |
| S | 0.2567 | 0.838 |
| SF | 0.2899 | 0.836 |
| SW | 0.2753 | 0.843 |
| SWF | 0.2733 | 0.842 |
| SFW | 0.2988 | 0.843 |

| gain | d AUC-PR | relative | 95 % interval |
|---|---|---|---|
| B-C | +0.0275 | +12 % | [+0.0174, +0.0387] |
| SF-S | +0.0332 | +13 % | [+0.0244, +0.0459] |
| CW-C | +0.0007 | +0 % | [-0.0056, +0.0081] |
| SW-S | +0.0186 | +7 % | [+0.0105, +0.0273] |
| SWF-S | +0.0166 | +6 % | [+0.0092, +0.0267] |
| SFW-SF | +0.0089 | +3 % | [-0.0016, +0.0198] |

### 3-10 km: 141620 fire pixels, base rate 0.00589

| set | AUC-PR | ROC-AUC |
|---|---|---|
| A | 0.0113 | 0.666 |
| C | 0.0385 | 0.772 |
| B | 0.0394 | 0.777 |
| CW | 0.0368 | 0.768 |
| S | 0.0439 | 0.797 |
| SF | 0.0473 | 0.794 |
| SW | 0.0424 | 0.805 |
| SWF | 0.0430 | 0.805 |
| SFW | 0.0460 | 0.801 |

| gain | d AUC-PR | relative | 95 % interval |
|---|---|---|---|
| B-C | +0.0009 | +2 % | [-0.0003, +0.0024] |
| SF-S | +0.0034 | +8 % | [+0.0011, +0.0062] |
| CW-C | -0.0018 | -5 % | [-0.0024, -0.0009] |
| SW-S | -0.0015 | -4 % | [-0.0035, +0.0008] |
| SWF-S | -0.0009 | -2 % | [-0.0028, +0.0012] |
| SFW-SF | -0.0013 | -3 % | [-0.0043, +0.0011] |

### 10-25 km: 89955 fire pixels, base rate 0.00115

| set | AUC-PR | ROC-AUC |
|---|---|---|
| A | 0.0017 | 0.609 |
| C | 0.0033 | 0.655 |
| B | 0.0034 | 0.664 |
| CW | 0.0031 | 0.651 |
| S | 0.0049 | 0.724 |
| SF | 0.0048 | 0.701 |
| SW | 0.0054 | 0.749 |
| SWF | 0.0055 | 0.749 |
| SFW | 0.0053 | 0.739 |

| gain | d AUC-PR | relative | 95 % interval |
|---|---|---|---|
| B-C | +0.0001 | +4 % | [-0.0001, +0.0005] |
| SF-S | -0.0001 | -2 % | [-0.0005, +0.0005] |
| CW-C | -0.0002 | -7 % | [-0.0004, -0.0001] |
| SW-S | +0.0005 | +9 % | [-0.0001, +0.0015] |
| SWF-S | +0.0006 | +12 % | [+0.0002, +0.0010] |
| SFW-SF | +0.0005 | +9 % | [-0.0004, +0.0009] |

## nearfield_gate_pm_2019.json

Feature sets:

- A: log_dist
- C: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth
- B: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, log_frp, n_det, night_share, log_cluster_frp
- CW: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, wind_speed_q, wind_align, wind_push, gust_push, upslope, slope_p, aspect_align
- S: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, smolder_logit
- SF: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, log_frp, n_det, night_share, log_cluster_frp, smolder_logit
- SW: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, wind_speed_q, wind_align, wind_push, gust_push, upslope, slope_p, aspect_align, smolder_logit
- SWF: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, wind_speed_q, wind_align, wind_push, gust_push, upslope, slope_p, aspect_align, f_wind_align, f_wind_push, smolder_logit
- SFW: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, log_frp, n_det, night_share, log_cluster_frp, wind_speed_q, wind_align, wind_push, gust_push, upslope, slope_p, aspect_align, smolder_logit
- SN: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, ndvi_p, smolder_logit
- SLAI: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, lai_p, smolder_logit
- SPM: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, upslope, slope_p, aspect_align, pm_speed_q, pm_wind_align, pm_wind_push, smolder_logit

### 0-3 km: 314730 fire pixels, base rate 0.06094

| set | AUC-PR | ROC-AUC |
|---|---|---|
| A | 0.1091 | 0.685 |
| C | 0.2348 | 0.807 |
| B | 0.2624 | 0.820 |
| CW | 0.2355 | 0.809 |
| S | 0.2567 | 0.838 |
| SF | 0.2899 | 0.836 |
| SW | 0.2753 | 0.843 |
| SWF | 0.2733 | 0.842 |
| SFW | 0.2988 | 0.843 |
| SN | 0.2758 | 0.841 |
| SLAI | 0.2872 | 0.841 |
| SPM | 0.2743 | 0.841 |

| gain | d AUC-PR | relative | 95 % interval |
|---|---|---|---|
| B-C | +0.0275 | +12 % | [+0.0174, +0.0387] |
| SF-S | +0.0332 | +13 % | [+0.0244, +0.0459] |
| CW-C | +0.0007 | +0 % | [-0.0056, +0.0081] |
| SW-S | +0.0186 | +7 % | [+0.0105, +0.0273] |
| SWF-S | +0.0166 | +6 % | [+0.0092, +0.0267] |
| SFW-SF | +0.0089 | +3 % | [-0.0016, +0.0198] |
| SN-S | +0.0191 | +7 % | [+0.0122, +0.0270] |
| SLAI-S | +0.0304 | +12 % | [+0.0221, +0.0403] |
| SN-SLAI | -0.0114 | -4 % | [-0.0192, -0.0031] |
| SPM-S | +0.0176 | +7 % | [+0.0106, +0.0264] |
| SPM-SW | -0.0010 | -0 % | [-0.0077, +0.0052] |

### 3-10 km: 141620 fire pixels, base rate 0.00589

| set | AUC-PR | ROC-AUC |
|---|---|---|
| A | 0.0113 | 0.666 |
| C | 0.0385 | 0.772 |
| B | 0.0394 | 0.777 |
| CW | 0.0368 | 0.768 |
| S | 0.0439 | 0.797 |
| SF | 0.0473 | 0.794 |
| SW | 0.0424 | 0.805 |
| SWF | 0.0430 | 0.805 |
| SFW | 0.0460 | 0.801 |
| SN | 0.0471 | 0.806 |
| SLAI | 0.0450 | 0.803 |
| SPM | 0.0422 | 0.802 |

| gain | d AUC-PR | relative | 95 % interval |
|---|---|---|---|
| B-C | +0.0009 | +2 % | [-0.0003, +0.0024] |
| SF-S | +0.0034 | +8 % | [+0.0011, +0.0062] |
| CW-C | -0.0018 | -5 % | [-0.0024, -0.0009] |
| SW-S | -0.0015 | -4 % | [-0.0035, +0.0008] |
| SWF-S | -0.0009 | -2 % | [-0.0028, +0.0012] |
| SFW-SF | -0.0013 | -3 % | [-0.0043, +0.0011] |
| SN-S | +0.0031 | +7 % | [+0.0009, +0.0065] |
| SLAI-S | +0.0010 | +2 % | [-0.0005, +0.0025] |
| SN-SLAI | +0.0021 | +5 % | [+0.0005, +0.0055] |
| SPM-S | -0.0017 | -4 % | [-0.0033, +0.0006] |
| SPM-SW | -0.0002 | -0 % | [-0.0016, +0.0011] |

### 10-25 km: 89955 fire pixels, base rate 0.00115

| set | AUC-PR | ROC-AUC |
|---|---|---|
| A | 0.0017 | 0.609 |
| C | 0.0033 | 0.655 |
| B | 0.0034 | 0.664 |
| CW | 0.0031 | 0.651 |
| S | 0.0049 | 0.724 |
| SF | 0.0048 | 0.701 |
| SW | 0.0054 | 0.749 |
| SWF | 0.0055 | 0.749 |
| SFW | 0.0053 | 0.739 |
| SN | 0.0066 | 0.743 |
| SLAI | 0.0062 | 0.721 |
| SPM | 0.0053 | 0.741 |

| gain | d AUC-PR | relative | 95 % interval |
|---|---|---|---|
| B-C | +0.0001 | +4 % | [-0.0001, +0.0005] |
| SF-S | -0.0001 | -2 % | [-0.0005, +0.0005] |
| CW-C | -0.0002 | -7 % | [-0.0004, -0.0001] |
| SW-S | +0.0005 | +9 % | [-0.0001, +0.0015] |
| SWF-S | +0.0006 | +12 % | [+0.0002, +0.0010] |
| SFW-SF | +0.0005 | +9 % | [-0.0004, +0.0009] |
| SN-S | +0.0017 | +35 % | [+0.0002, +0.0053] |
| SLAI-S | +0.0013 | +26 % | [-0.0001, +0.0061] |
| SN-SLAI | +0.0004 | +7 % | [-0.0021, +0.0018] |
| SPM-S | +0.0004 | +8 % | [-0.0001, +0.0007] |
| SPM-SW | -0.0001 | -1 % | [-0.0010, +0.0004] |

## nearfield_gate_fuelage_2019.json

Feature sets:

- A: log_dist
- C: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth
- B: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, log_frp, n_det, night_share, log_cluster_frp
- CW: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, wind_speed_q, wind_align, wind_push, gust_push, upslope, slope_p, aspect_align
- S: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, smolder_logit
- SF: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, log_frp, n_det, night_share, log_cluster_frp, smolder_logit
- SW: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, wind_speed_q, wind_align, wind_push, gust_push, upslope, slope_p, aspect_align, smolder_logit
- SWF: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, wind_speed_q, wind_align, wind_push, gust_push, upslope, slope_p, aspect_align, f_wind_align, f_wind_push, smolder_logit
- SFW: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, log_frp, n_det, night_share, log_cluster_frp, wind_speed_q, wind_align, wind_push, gust_push, upslope, slope_p, aspect_align, smolder_logit
- SN: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, ndvi_p, smolder_logit
- SLAI: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, lai_p, smolder_logit
- CA: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, fuel_age_p
- SA: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, fuel_age_p, smolder_logit
- SLA: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, lai_p, fuel_age_p, smolder_logit
- SNA: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, ndvi_p, fuel_age_p, smolder_logit
- SPM: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, upslope, slope_p, aspect_align, pm_speed_q, pm_wind_align, pm_wind_push, smolder_logit

### 0-3 km: 314730 fire pixels, base rate 0.06094

| set | AUC-PR | ROC-AUC |
|---|---|---|
| A | 0.1091 | 0.685 |
| C | 0.2348 | 0.807 |
| B | 0.2624 | 0.820 |
| CW | 0.2355 | 0.809 |
| S | 0.2567 | 0.838 |
| SF | 0.2899 | 0.836 |
| SW | 0.2753 | 0.843 |
| SWF | 0.2733 | 0.842 |
| SFW | 0.2988 | 0.843 |
| SN | 0.2758 | 0.841 |
| SLAI | 0.2872 | 0.841 |
| CA | 0.2415 | 0.814 |
| SA | 0.2653 | 0.841 |
| SLA | 0.2705 | 0.830 |
| SNA | 0.2762 | 0.843 |
| SPM | 0.2743 | 0.841 |

| gain | d AUC-PR | relative | 95 % interval |
|---|---|---|---|
| B-C | +0.0275 | +12 % | [+0.0174, +0.0387] |
| SF-S | +0.0332 | +13 % | [+0.0244, +0.0459] |
| CW-C | +0.0007 | +0 % | [-0.0056, +0.0081] |
| SW-S | +0.0186 | +7 % | [+0.0105, +0.0273] |
| SWF-S | +0.0166 | +6 % | [+0.0092, +0.0267] |
| SFW-SF | +0.0089 | +3 % | [-0.0016, +0.0198] |
| SN-S | +0.0191 | +7 % | [+0.0122, +0.0270] |
| SLAI-S | +0.0304 | +12 % | [+0.0221, +0.0403] |
| SN-SLAI | -0.0114 | -4 % | [-0.0192, -0.0031] |
| SLA-SLAI | -0.0167 | -6 % | [-0.0256, -0.0069] |
| SNA-SN | +0.0004 | +0 % | [-0.0077, +0.0080] |
| CA-C | +0.0066 | +3 % | [+0.0006, +0.0134] |
| SA-S | +0.0086 | +3 % | [+0.0031, +0.0152] |
| SPM-S | +0.0176 | +7 % | [+0.0106, +0.0264] |
| SPM-SW | -0.0010 | -0 % | [-0.0077, +0.0052] |

### 3-10 km: 141620 fire pixels, base rate 0.00589

| set | AUC-PR | ROC-AUC |
|---|---|---|
| A | 0.0113 | 0.666 |
| C | 0.0385 | 0.772 |
| B | 0.0394 | 0.777 |
| CW | 0.0368 | 0.768 |
| S | 0.0439 | 0.797 |
| SF | 0.0473 | 0.794 |
| SW | 0.0424 | 0.805 |
| SWF | 0.0430 | 0.805 |
| SFW | 0.0460 | 0.801 |
| SN | 0.0471 | 0.806 |
| SLAI | 0.0450 | 0.803 |
| CA | 0.0429 | 0.804 |
| SA | 0.0489 | 0.816 |
| SLA | 0.0491 | 0.808 |
| SNA | 0.0528 | 0.816 |
| SPM | 0.0422 | 0.802 |

| gain | d AUC-PR | relative | 95 % interval |
|---|---|---|---|
| B-C | +0.0009 | +2 % | [-0.0003, +0.0024] |
| SF-S | +0.0034 | +8 % | [+0.0011, +0.0062] |
| CW-C | -0.0018 | -5 % | [-0.0024, -0.0009] |
| SW-S | -0.0015 | -4 % | [-0.0035, +0.0008] |
| SWF-S | -0.0009 | -2 % | [-0.0028, +0.0012] |
| SFW-SF | -0.0013 | -3 % | [-0.0043, +0.0011] |
| SN-S | +0.0031 | +7 % | [+0.0009, +0.0065] |
| SLAI-S | +0.0010 | +2 % | [-0.0005, +0.0025] |
| SN-SLAI | +0.0021 | +5 % | [+0.0005, +0.0055] |
| SLA-SLAI | +0.0041 | +9 % | [+0.0023, +0.0062] |
| SNA-SN | +0.0057 | +12 % | [+0.0017, +0.0104] |
| CA-C | +0.0044 | +11 % | [+0.0024, +0.0069] |
| SA-S | +0.0050 | +11 % | [+0.0018, +0.0104] |
| SPM-S | -0.0017 | -4 % | [-0.0033, +0.0006] |
| SPM-SW | -0.0002 | -0 % | [-0.0016, +0.0011] |

### 10-25 km: 89955 fire pixels, base rate 0.00115

| set | AUC-PR | ROC-AUC |
|---|---|---|
| A | 0.0017 | 0.609 |
| C | 0.0033 | 0.655 |
| B | 0.0034 | 0.664 |
| CW | 0.0031 | 0.651 |
| S | 0.0049 | 0.724 |
| SF | 0.0048 | 0.701 |
| SW | 0.0054 | 0.749 |
| SWF | 0.0055 | 0.749 |
| SFW | 0.0053 | 0.739 |
| SN | 0.0066 | 0.743 |
| SLAI | 0.0062 | 0.721 |
| CA | 0.0039 | 0.711 |
| SA | 0.0066 | 0.766 |
| SLA | 0.0088 | 0.744 |
| SNA | 0.0077 | 0.753 |
| SPM | 0.0053 | 0.741 |

| gain | d AUC-PR | relative | 95 % interval |
|---|---|---|---|
| B-C | +0.0001 | +4 % | [-0.0001, +0.0005] |
| SF-S | -0.0001 | -2 % | [-0.0005, +0.0005] |
| CW-C | -0.0002 | -7 % | [-0.0004, -0.0001] |
| SW-S | +0.0005 | +9 % | [-0.0001, +0.0015] |
| SWF-S | +0.0006 | +12 % | [+0.0002, +0.0010] |
| SFW-SF | +0.0005 | +9 % | [-0.0004, +0.0009] |
| SN-S | +0.0017 | +35 % | [+0.0002, +0.0053] |
| SLAI-S | +0.0013 | +26 % | [-0.0001, +0.0061] |
| SN-SLAI | +0.0004 | +7 % | [-0.0021, +0.0018] |
| SLA-SLAI | +0.0026 | +41 % | [+0.0005, +0.0051] |
| SNA-SN | +0.0011 | +17 % | [+0.0002, +0.0024] |
| CA-C | +0.0006 | +18 % | [-0.0000, +0.0018] |
| SA-S | +0.0017 | +34 % | [+0.0006, +0.0030] |
| SPM-S | +0.0004 | +8 % | [-0.0001, +0.0007] |
| SPM-SW | -0.0001 | -1 % | [-0.0010, +0.0004] |

## nearfield_gate_fuelage_caps_2019.json

Feature sets:

- S: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, smolder_logit
- SA365: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, fuel_age_p, smolder_logit
- SA730: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, fuel_age_p, smolder_logit
- SA1095: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, fuel_age_p, smolder_logit
- SA1460: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, fuel_age_p, smolder_logit
- SAnocap: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, fuel_age_p, smolder_logit
- SPM: log_dist, hist_near, hist_own, log_cluster_size, cluster_growth, upslope, slope_p, aspect_align, pm_speed_q, pm_wind_align, pm_wind_push, smolder_logit

### 0-3 km: 314730 fire pixels, base rate 0.06094

| set | AUC-PR | ROC-AUC |
|---|---|---|
| S | 0.2567 | 0.838 |
| SA365 | 0.2572 | 0.832 |
| SA730 | 0.2642 | 0.838 |
| SA1095 | 0.2660 | 0.841 |
| SA1460 | 0.2650 | 0.839 |
| SAnocap | 0.2653 | 0.841 |
| SPM | 0.2743 | 0.841 |

| gain | d AUC-PR | relative | 95 % interval |
|---|---|---|---|
| SA365-S | +0.0005 | +0 % | [-0.0043, +0.0050] |
| SA730-S | +0.0075 | +3 % | [+0.0018, +0.0144] |
| SA1095-S | +0.0093 | +4 % | [+0.0043, +0.0157] |
| SA1460-S | +0.0083 | +3 % | [+0.0023, +0.0148] |
| SAnocap-S | +0.0086 | +3 % | [+0.0031, +0.0152] |
| SPM-S | +0.0176 | +7 % | [+0.0106, +0.0264] |

### 3-10 km: 141620 fire pixels, base rate 0.00589

| set | AUC-PR | ROC-AUC |
|---|---|---|
| S | 0.0439 | 0.797 |
| SA365 | 0.0464 | 0.797 |
| SA730 | 0.0479 | 0.808 |
| SA1095 | 0.0487 | 0.812 |
| SA1460 | 0.0509 | 0.812 |
| SAnocap | 0.0489 | 0.816 |
| SPM | 0.0422 | 0.802 |

| gain | d AUC-PR | relative | 95 % interval |
|---|---|---|---|
| SA365-S | +0.0024 | +6 % | [+0.0011, +0.0043] |
| SA730-S | +0.0039 | +9 % | [+0.0022, +0.0058] |
| SA1095-S | +0.0048 | +11 % | [+0.0031, +0.0070] |
| SA1460-S | +0.0069 | +16 % | [+0.0052, +0.0093] |
| SAnocap-S | +0.0050 | +11 % | [+0.0018, +0.0104] |
| SPM-S | -0.0017 | -4 % | [-0.0033, +0.0006] |

### 10-25 km: 89955 fire pixels, base rate 0.00115

| set | AUC-PR | ROC-AUC |
|---|---|---|
| S | 0.0049 | 0.724 |
| SA365 | 0.0050 | 0.714 |
| SA730 | 0.0054 | 0.751 |
| SA1095 | 0.0056 | 0.759 |
| SA1460 | 0.0055 | 0.760 |
| SAnocap | 0.0066 | 0.766 |
| SPM | 0.0053 | 0.741 |

| gain | d AUC-PR | relative | 95 % interval |
|---|---|---|---|
| SA365-S | +0.0001 | +2 % | [-0.0001, +0.0008] |
| SA730-S | +0.0005 | +9 % | [-0.0000, +0.0007] |
| SA1095-S | +0.0007 | +14 % | [+0.0001, +0.0014] |
| SA1460-S | +0.0006 | +11 % | [+0.0000, +0.0009] |
| SAnocap-S | +0.0017 | +34 % | [+0.0006, +0.0030] |
| SPM-S | +0.0004 | +8 % | [-0.0001, +0.0007] |

