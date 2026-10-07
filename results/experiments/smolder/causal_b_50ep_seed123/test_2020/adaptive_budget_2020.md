Thresholds fixed in results/experiments/final/causal_b_50ep_seed123/adaptive_budget_2019.json, applied unchanged; 'mean budget' of the adaptive rows is the area they flag in this year. 
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
| SMOLDER | 0.06 % | fixed | 0.257 | 0.135 | 6.4 | 0.218 | 0.031 | 0.624 | 0.065 | 0.007 | constant |
| SMOLDER | 0.0333 % | adaptive | 0.227 | 0.216 | 3.6 | 0.225 | 0.024 | 0.558 | 0.058 | 0.002 | 0.00203 % / 0.0272 % / 0.0895 % / 0.138 % |
| SMOLDER | 0.13 % | fixed | 0.339 | 0.083 | 11.1 | 0.209 | 0.068 | 0.784 | 0.156 | 0.015 | constant |
| SMOLDER | 0.0852 % | adaptive | 0.324 | 0.120 | 7.3 | 0.242 | 0.059 | 0.757 | 0.147 | 0.007 | 0.00725 % / 0.0775 % / 0.206 % / 0.302 % |
| SMOLDER | 0.23 % | fixed | 0.399 | 0.055 | 17.2 | 0.177 | 0.111 | 0.873 | 0.262 | 0.025 | constant |
| SMOLDER | 0.169 % | adaptive | 0.391 | 0.073 | 12.6 | 0.210 | 0.101 | 0.869 | 0.250 | 0.016 | 0.0177 % / 0.156 % / 0.383 % / 0.522 % |
| SMOLDER | 0.5 % | fixed | 0.480 | 0.030 | 31.9 | 0.121 | 0.200 | 0.946 | 0.466 | 0.056 | constant |
| SMOLDER | 0.422 % | adaptive | 0.488 | 0.037 | 26.4 | 0.141 | 0.209 | 0.951 | 0.493 | 0.056 | 0.0486 % / 0.36 % / 0.925 % / 1.48 % |
| SMOLDER | 1 % | fixed | 0.562 | 0.018 | 55.2 | 0.079 | 0.314 | 0.975 | 0.673 | 0.127 | constant |
| SMOLDER | 0.9 % | adaptive | 0.584 | 0.021 | 47.7 | 0.090 | 0.348 | 0.979 | 0.710 | 0.159 | 0.112 % / 0.719 % / 2.02 % / 2.84 % |

Best adaptive threshold:

| ranking | criterion | mean budget | fire caught | precision | false alarms per hit | daily area p5 / p50 / p95 / max | rank corr. daily area vs daily fire |
|---|---|---|---|---|---|---|---|
| persistence | F1 | 0.0964 % | 0.230 | 0.075 | 12.3 | 0.0175 % / 0.0881 % / 0.19 % / 0.246 % | 0.75 |
| persistence | F2 | 0.14 % | 0.278 | 0.063 | 14.9 | 0.0273 % / 0.126 % / 0.266 % / 0.352 % | 0.74 |
| SMOLDER | F1 | 0.028 % | 0.210 | 0.237 | 3.2 | 0.00172 % / 0.0228 % / 0.0762 % / 0.118 % | 0.75 |
| SMOLDER | F2 | 0.0707 % | 0.305 | 0.136 | 6.3 | 0.00558 % / 0.0632 % / 0.175 % / 0.261 % | 0.78 |
