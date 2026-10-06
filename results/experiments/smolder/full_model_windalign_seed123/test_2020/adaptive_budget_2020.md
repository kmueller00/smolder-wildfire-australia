Thresholds fixed in results/experiments/smolder/full_model_windalign_seed123/adaptive_budget_2019.json, applied unchanged; 'mean budget' of the adaptive rows is the area they flag in this year. 
Fixed vs adaptive daily budget, 2020 (350 days). Adaptive: one score threshold for all days.

| ranking | mean budget | rule | fire caught | precision | false alarms per hit | F2 | new-fire caught | caught 0-3 km | caught 3-10 km | caught > 10 km | daily area p5 / p50 / p95 / max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| persistence | 0.06 % | fixed | 0.145 | 0.076 | 12.1 | 0.123 | 0.000 | 0.376 | 0.000 | 0.000 | constant |
| persistence | 0.0321 % | adaptive | 0.104 | 0.103 | 8.7 | 0.104 | 0.000 | 0.271 | 0.000 | 0.000 | 0.00468 % / 0.0293 % / 0.0704 % / 0.0855 % |
| persistence | 0.13 % | fixed | 0.239 | 0.058 | 16.2 | 0.148 | 0.001 | 0.620 | 0.002 | 0.000 | constant |
| persistence | 0.0964 % | adaptive | 0.230 | 0.075 | 12.3 | 0.163 | 0.000 | 0.596 | 0.000 | 0.000 | 0.0175 % / 0.0881 % / 0.19 % / 0.246 % |
| persistence | 0.23 % | fixed | 0.317 | 0.044 | 21.9 | 0.141 | 0.009 | 0.813 | 0.016 | 0.000 | constant |
| persistence | 0.187 % | adaptive | 0.316 | 0.053 | 17.7 | 0.159 | 0.000 | 0.821 | 0.000 | 0.000 | 0.0383 % / 0.168 % / 0.356 % / 0.464 % |
| persistence | 0.5 % | fixed | 0.411 | 0.026 | 37.4 | 0.104 | 0.068 | 0.985 | 0.159 | 0.000 | constant |
| persistence | 0.395 % | adaptive | 0.404 | 0.032 | 29.9 | 0.123 | 0.048 | 1.000 | 0.091 | 0.000 | 0.0942 % / 0.359 % / 0.739 % / 0.918 % |
| persistence | 1 % | fixed | 0.486 | 0.015 | 64.0 | 0.068 | 0.180 | 1.000 | 0.505 | 0.002 | constant |
| persistence | 0.822 % | adaptive | 0.481 | 0.019 | 53.0 | 0.080 | 0.171 | 1.000 | 0.482 | 0.000 | 0.229 % / 0.781 % / 1.49 % / 1.77 % |
| SMOLDER | 0.06 % | fixed | 0.257 | 0.136 | 6.4 | 0.218 | 0.032 | 0.623 | 0.072 | 0.007 | constant |
| SMOLDER | 0.0316 % | adaptive | 0.223 | 0.223 | 3.5 | 0.223 | 0.025 | 0.543 | 0.065 | 0.001 | 0.00216 % / 0.0256 % / 0.0842 % / 0.158 % |
| SMOLDER | 0.13 % | fixed | 0.341 | 0.083 | 11.1 | 0.210 | 0.071 | 0.783 | 0.165 | 0.015 | constant |
| SMOLDER | 0.082 % | adaptive | 0.320 | 0.124 | 7.1 | 0.243 | 0.059 | 0.746 | 0.150 | 0.006 | 0.00742 % / 0.0732 % / 0.202 % / 0.341 % |
| SMOLDER | 0.23 % | fixed | 0.400 | 0.055 | 17.1 | 0.178 | 0.114 | 0.872 | 0.270 | 0.027 | constant |
| SMOLDER | 0.162 % | adaptive | 0.387 | 0.076 | 12.2 | 0.212 | 0.099 | 0.861 | 0.249 | 0.015 | 0.0188 % / 0.147 % / 0.369 % / 0.569 % |
| SMOLDER | 0.5 % | fixed | 0.481 | 0.030 | 31.8 | 0.121 | 0.202 | 0.944 | 0.460 | 0.063 | constant |
| SMOLDER | 0.403 % | adaptive | 0.482 | 0.038 | 25.5 | 0.144 | 0.201 | 0.947 | 0.464 | 0.060 | 0.0551 % / 0.344 % / 0.864 % / 1.27 % |
| SMOLDER | 1 % | fixed | 0.561 | 0.018 | 55.3 | 0.079 | 0.313 | 0.973 | 0.646 | 0.139 | constant |
| SMOLDER | 0.873 % | adaptive | 0.577 | 0.021 | 46.8 | 0.091 | 0.337 | 0.977 | 0.669 | 0.163 | 0.119 % / 0.701 % / 2.02 % / 2.63 % |

Best adaptive threshold:

| ranking | criterion | mean budget | fire caught | precision | false alarms per hit | daily area p5 / p50 / p95 / max | rank corr. daily area vs daily fire |
|---|---|---|---|---|---|---|---|
| persistence | F1 | 0.0964 % | 0.230 | 0.075 | 12.3 | 0.0175 % / 0.0881 % / 0.19 % / 0.246 % | 0.75 |
| persistence | F2 | 0.14 % | 0.278 | 0.063 | 14.9 | 0.0273 % / 0.126 % / 0.266 % / 0.352 % | 0.74 |
| SMOLDER | F1 | 0.0288 % | 0.213 | 0.234 | 3.3 | 0.00188 % / 0.0233 % / 0.0769 % / 0.145 % | 0.72 |
| SMOLDER | F2 | 0.068 % | 0.301 | 0.140 | 6.1 | 0.00596 % / 0.0593 % / 0.169 % / 0.295 % | 0.75 |
