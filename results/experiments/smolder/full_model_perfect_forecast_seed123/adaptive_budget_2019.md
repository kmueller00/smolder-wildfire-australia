
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
| SMOLDER | 0.06 % | fixed | 0.270 | 0.250 | 3.0 | 0.266 | 0.031 | 0.553 | 0.060 | 0.004 | constant |
| SMOLDER | 0.0599 % | adaptive | 0.320 | 0.297 | 2.4 | 0.315 | 0.040 | 0.651 | 0.083 | 0.001 | 0.00798 % / 0.0412 % / 0.196 % / 0.375 % |
| SMOLDER | 0.13 % | fixed | 0.383 | 0.164 | 5.1 | 0.302 | 0.078 | 0.747 | 0.160 | 0.009 | constant |
| SMOLDER | 0.13 % | adaptive | 0.433 | 0.185 | 4.4 | 0.342 | 0.104 | 0.828 | 0.224 | 0.006 | 0.0251 % / 0.0987 % / 0.348 % / 0.599 % |
| SMOLDER | 0.23 % | fixed | 0.468 | 0.113 | 7.9 | 0.287 | 0.137 | 0.865 | 0.287 | 0.018 | constant |
| SMOLDER | 0.229 % | adaptive | 0.505 | 0.122 | 7.2 | 0.311 | 0.168 | 0.911 | 0.366 | 0.015 | 0.0557 % / 0.191 % / 0.544 % / 0.922 % |
| SMOLDER | 0.5 % | fixed | 0.568 | 0.063 | 14.9 | 0.218 | 0.250 | 0.953 | 0.518 | 0.051 | constant |
| SMOLDER | 0.501 % | adaptive | 0.590 | 0.065 | 14.3 | 0.227 | 0.279 | 0.968 | 0.584 | 0.055 | 0.144 % / 0.465 % / 1.1 % / 1.58 % |
| SMOLDER | 1 % | fixed | 0.643 | 0.036 | 27.0 | 0.146 | 0.364 | 0.982 | 0.713 | 0.113 | constant |
| SMOLDER | 1 % | adaptive | 0.660 | 0.037 | 26.4 | 0.150 | 0.391 | 0.987 | 0.752 | 0.132 | 0.259 % / 0.926 % / 1.97 % / 2.58 % |

Best adaptive threshold:

| ranking | criterion | mean budget | fire caught | precision | false alarms per hit | daily area p5 / p50 / p95 / max | rank corr. daily area vs daily fire |
|---|---|---|---|---|---|---|---|
| persistence | F1 | 0.14 % | 0.282 | 0.112 | 7.9 | 0.0462 % / 0.131 % / 0.255 % / 0.441 % | 0.69 |
| persistence | F2 | 0.193 % | 0.335 | 0.096 | 9.4 | 0.0672 % / 0.182 % / 0.336 % / 0.534 % | 0.66 |
| SMOLDER | F1 | 0.0551 % | 0.307 | 0.310 | 2.2 | 0.00711 % / 0.0374 % / 0.186 % / 0.361 % | 0.77 |
| SMOLDER | F2 | 0.112 % | 0.414 | 0.204 | 3.9 | 0.0207 % / 0.084 % / 0.311 % / 0.534 % | 0.80 |
