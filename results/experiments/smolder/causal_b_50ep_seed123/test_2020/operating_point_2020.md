Operating points 2020 (350 days; budget = top k % of land pixels per day)

| ranking | criterion | budget | recall | precision | false alarms per fire px | new-fire recall | recall 0-3 km | recall 3-10 km | recall > 10 km | false alarms 0-3 km | 3-10 km | > 10 km |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| persistence | best F1 | 0.0725 % | 0.166 | 0.0723 | 12.8 | 0.000 | 0.429 | 0.000 | 0.000 | 98 % | 2 % | 0 % |
| persistence | best F2 | 0.146 % | 0.256 | 0.0553 | 17.1 | 0.002 | 0.662 | 0.003 | 0.000 | 94 % | 6 % | 0 % |
| causal_B | best F1 | 0.0319 % | 0.192 | 0.1907 | 4.2 | 0.014 | 0.481 | 0.027 | 0.004 | 88 % | 6 % | 6 % |
| causal_B | best F2 | 0.0815 % | 0.289 | 0.1124 | 7.9 | 0.043 | 0.691 | 0.095 | 0.010 | 77 % | 15 % | 8 % |

Recall and false alarms per fire pixel caught at selected budgets:

| budget | persistence recall | persistence FA/TP | causal_B recall | causal_B FA/TP |
|---|---|---|---|---|
| 0.051 % | 0.129 | 11.5 | 0.240 | 5.7 |
| 0.103 % | 0.208 | 14.7 | 0.315 | 9.3 |
| 0.208 % | 0.303 | 20.7 | 0.389 | 15.9 |
| 0.472 % | 0.405 | 35.9 | 0.474 | 30.4 |
| 0.953 % | 0.481 | 61.6 | 0.556 | 53.1 |
| 1.92 % | 0.559 | 107.8 | 0.647 | 92.9 |
| 4.91 % | 0.682 | 226.4 | 0.762 | 202.5 |
| 9.91 % | 0.788 | 396.2 | 0.832 | 375.1 |
