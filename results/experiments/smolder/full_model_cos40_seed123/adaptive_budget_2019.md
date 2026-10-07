
Fixed vs adaptive daily budget, 2019 (352 days). Adaptive: one score threshold for all days, set so the mean flagged area equals the fixed budget.

| ranking | mean budget | rule | fire caught | precision | false alarms per hit | F2 | new-fire caught | caught 0-3 km | caught 3-10 km | caught > 10 km | daily area p5 / p50 / p95 / max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| persistence | 0.06 % | fixed | 0.125 | 0.116 | 7.6 | 0.123 | 0.000 | 0.269 | 0.000 | 0.000 | constant |
| persistence | 0.0542 % | adaptive | 0.139 | 0.142 | 6.0 | 0.140 | 0.000 | 0.300 | 0.000 | 0.000 | 0.0169 % / 0.0458 % / 0.12 % / 0.243 % |
| persistence | 0.13 % | fixed | 0.232 | 0.099 | 9.1 | 0.183 | 0.001 | 0.500 | 0.001 | 0.000 | constant |
| persistence | 0.14 % | adaptive | 0.282 | 0.112 | 7.9 | 0.216 | 0.000 | 0.608 | 0.000 | 0.000 | 0.0462 % / 0.131 % / 0.255 % / 0.441 % |
| persistence | 0.23 % | fixed | 0.334 | 0.081 | 11.4 | 0.205 | 0.005 | 0.717 | 0.007 | 0.000 | constant |
| persistence | 0.253 % | adaptive | 0.382 | 0.084 | 10.9 | 0.223 | 0.000 | 0.823 | 0.000 | 0.000 | 0.0916 % / 0.235 % / 0.431 % / 0.644 % |
| persistence | 0.5 % | fixed | 0.470 | 0.052 | 18.2 | 0.181 | 0.067 | 0.955 | 0.125 | 0.000 | constant |
| persistence | 0.5 % | adaptive | 0.485 | 0.054 | 17.6 | 0.187 | 0.063 | 1.000 | 0.098 | 0.000 | 0.186 % / 0.48 % / 0.82 % / 1.06 % |
| persistence | 1 % | fixed | 0.569 | 0.032 | 30.6 | 0.129 | 0.217 | 1.000 | 0.497 | 0.001 | constant |
| persistence | 1 % | adaptive | 0.576 | 0.032 | 30.3 | 0.131 | 0.229 | 1.000 | 0.530 | 0.000 | 0.384 % / 0.966 % / 1.6 % / 1.91 % |
| SMOLDER | 0.06 % | fixed | 0.264 | 0.245 | 3.1 | 0.260 | 0.025 | 0.545 | 0.047 | 0.004 | constant |
| SMOLDER | 0.0601 % | adaptive | 0.311 | 0.287 | 2.5 | 0.306 | 0.033 | 0.639 | 0.065 | 0.002 | 0.01 % / 0.0452 % / 0.174 % / 0.32 % |
| SMOLDER | 0.13 % | fixed | 0.375 | 0.160 | 5.2 | 0.296 | 0.068 | 0.740 | 0.136 | 0.009 | constant |
| SMOLDER | 0.13 % | adaptive | 0.421 | 0.180 | 4.6 | 0.332 | 0.088 | 0.818 | 0.186 | 0.006 | 0.0273 % / 0.106 % / 0.314 % / 0.506 % |
| SMOLDER | 0.23 % | fixed | 0.458 | 0.111 | 8.0 | 0.281 | 0.123 | 0.860 | 0.256 | 0.016 | constant |
| SMOLDER | 0.229 % | adaptive | 0.493 | 0.119 | 7.4 | 0.303 | 0.149 | 0.907 | 0.321 | 0.013 | 0.0551 % / 0.2 % / 0.519 % / 0.787 % |
| SMOLDER | 0.5 % | fixed | 0.558 | 0.062 | 15.1 | 0.215 | 0.233 | 0.951 | 0.486 | 0.043 | constant |
| SMOLDER | 0.503 % | adaptive | 0.578 | 0.064 | 14.7 | 0.221 | 0.257 | 0.965 | 0.541 | 0.047 | 0.117 % / 0.488 % / 0.956 % / 1.38 % |
| SMOLDER | 1 % | fixed | 0.633 | 0.035 | 27.5 | 0.144 | 0.347 | 0.981 | 0.689 | 0.100 | constant |
| SMOLDER | 0.997 % | adaptive | 0.644 | 0.036 | 26.9 | 0.147 | 0.365 | 0.984 | 0.712 | 0.116 | 0.229 % / 0.961 % / 1.81 % / 2.03 % |

Best adaptive threshold:

| ranking | criterion | mean budget | fire caught | precision | false alarms per hit | daily area p5 / p50 / p95 / max | rank corr. daily area vs daily fire |
|---|---|---|---|---|---|---|---|
| persistence | F1 | 0.14 % | 0.282 | 0.112 | 7.9 | 0.0462 % / 0.131 % / 0.255 % / 0.441 % | 0.69 |
| persistence | F2 | 0.193 % | 0.335 | 0.096 | 9.4 | 0.0672 % / 0.182 % / 0.336 % / 0.534 % | 0.66 |
| SMOLDER | F1 | 0.0545 % | 0.297 | 0.302 | 2.3 | 0.00882 % / 0.041 % / 0.158 % / 0.304 % | 0.75 |
| SMOLDER | F2 | 0.115 % | 0.405 | 0.195 | 4.1 | 0.0225 % / 0.0934 % / 0.283 % / 0.46 % | 0.78 |
