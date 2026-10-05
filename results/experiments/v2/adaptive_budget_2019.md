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
| SMOLDER | 0.06 % | fixed | 0.261 | 0.241 | 3.1 | 0.257 | 0.021 | 0.543 | 0.038 | 0.003 | constant |
| SMOLDER | 0.0599 % | adaptive | 0.305 | 0.283 | 2.5 | 0.300 | 0.027 | 0.633 | 0.052 | 0.001 | 0.00964 % / 0.0428 % / 0.184 % / 0.324 % |
| SMOLDER | 0.13 % | fixed | 0.371 | 0.158 | 5.3 | 0.292 | 0.061 | 0.739 | 0.120 | 0.007 | constant |
| SMOLDER | 0.13 % | adaptive | 0.414 | 0.177 | 4.6 | 0.327 | 0.078 | 0.816 | 0.163 | 0.004 | 0.0257 % / 0.101 % / 0.33 % / 0.565 % |
| SMOLDER | 0.23 % | fixed | 0.453 | 0.109 | 8.1 | 0.278 | 0.114 | 0.859 | 0.234 | 0.015 | constant |
| SMOLDER | 0.23 % | adaptive | 0.487 | 0.118 | 7.5 | 0.299 | 0.139 | 0.906 | 0.300 | 0.011 | 0.0564 % / 0.191 % / 0.548 % / 0.968 % |
| SMOLDER | 0.5 % | fixed | 0.552 | 0.061 | 15.3 | 0.212 | 0.221 | 0.952 | 0.465 | 0.037 | constant |
| SMOLDER | 0.502 % | adaptive | 0.573 | 0.063 | 14.8 | 0.220 | 0.247 | 0.968 | 0.530 | 0.036 | 0.131 % / 0.487 % / 0.986 % / 1.67 % |
| SMOLDER | 1 % | fixed | 0.627 | 0.035 | 27.7 | 0.142 | 0.335 | 0.982 | 0.679 | 0.087 | constant |
| SMOLDER | 1.01 % | adaptive | 0.641 | 0.035 | 27.3 | 0.145 | 0.358 | 0.986 | 0.712 | 0.103 | 0.26 % / 1.03 % / 1.68 % / 2.4 % |

Best adaptive threshold:

| ranking | criterion | mean budget | fire caught | precision | false alarms per hit | daily area p5 / p50 / p95 / max | rank corr. daily area vs daily fire |
|---|---|---|---|---|---|---|---|
| persistence | F1 | 0.14 % | 0.282 | 0.112 | 7.9 | 0.0462 % / 0.131 % / 0.255 % / 0.441 % | 0.69 |
| persistence | F2 | 0.193 % | 0.335 | 0.096 | 9.4 | 0.0672 % / 0.182 % / 0.336 % / 0.534 % | 0.66 |
| SMOLDER | F1 | 0.0534 % | 0.289 | 0.300 | 2.3 | 0.00824 % / 0.0376 % / 0.165 % / 0.302 % | 0.74 |
| SMOLDER | F2 | 0.116 % | 0.399 | 0.191 | 4.2 | 0.0215 % / 0.0892 % / 0.305 % / 0.498 % | 0.77 |
