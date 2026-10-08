
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
| SMOLDER | 0.06 % | fixed | 0.265 | 0.246 | 3.1 | 0.261 | 0.028 | 0.545 | 0.052 | 0.004 | constant |
| SMOLDER | 0.0599 % | adaptive | 0.312 | 0.289 | 2.5 | 0.307 | 0.036 | 0.638 | 0.072 | 0.002 | 0.00936 % / 0.0444 % / 0.176 % / 0.313 % |
| SMOLDER | 0.13 % | fixed | 0.377 | 0.161 | 5.2 | 0.297 | 0.072 | 0.739 | 0.147 | 0.010 | constant |
| SMOLDER | 0.13 % | adaptive | 0.423 | 0.180 | 4.5 | 0.333 | 0.093 | 0.817 | 0.197 | 0.007 | 0.0263 % / 0.105 % / 0.328 % / 0.527 % |
| SMOLDER | 0.23 % | fixed | 0.460 | 0.111 | 8.0 | 0.283 | 0.129 | 0.857 | 0.268 | 0.018 | constant |
| SMOLDER | 0.23 % | adaptive | 0.495 | 0.119 | 7.4 | 0.304 | 0.154 | 0.905 | 0.332 | 0.015 | 0.0533 % / 0.2 % / 0.535 % / 0.848 % |
| SMOLDER | 0.5 % | fixed | 0.559 | 0.062 | 15.1 | 0.215 | 0.236 | 0.950 | 0.491 | 0.046 | constant |
| SMOLDER | 0.5 % | adaptive | 0.579 | 0.064 | 14.6 | 0.223 | 0.260 | 0.964 | 0.546 | 0.049 | 0.124 % / 0.475 % / 0.962 % / 1.51 % |
| SMOLDER | 1 % | fixed | 0.635 | 0.035 | 27.4 | 0.144 | 0.351 | 0.980 | 0.692 | 0.105 | constant |
| SMOLDER | 1 % | adaptive | 0.649 | 0.036 | 26.9 | 0.147 | 0.373 | 0.985 | 0.719 | 0.125 | 0.248 % / 0.955 % / 1.84 % / 2.11 % |

Best adaptive threshold:

| ranking | criterion | mean budget | fire caught | precision | false alarms per hit | daily area p5 / p50 / p95 / max | rank corr. daily area vs daily fire |
|---|---|---|---|---|---|---|---|
| persistence | F1 | 0.14 % | 0.282 | 0.112 | 7.9 | 0.0462 % / 0.131 % / 0.255 % / 0.441 % | 0.69 |
| persistence | F2 | 0.193 % | 0.335 | 0.096 | 9.4 | 0.0672 % / 0.182 % / 0.336 % / 0.534 % | 0.66 |
| SMOLDER | F1 | 0.0556 % | 0.301 | 0.300 | 2.3 | 0.00819 % / 0.0409 % / 0.167 % / 0.301 % | 0.76 |
| SMOLDER | F2 | 0.112 % | 0.402 | 0.200 | 4.0 | 0.021 % / 0.0879 % / 0.283 % / 0.46 % | 0.78 |
