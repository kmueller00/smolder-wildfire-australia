Operating points 2019 (352 days; budget = top k % of land pixels per day)

| ranking | criterion | budget | recall | precision | false alarms per fire px | new-fire recall | recall 0-3 km | recall 3-10 km | recall > 10 km | false alarms 0-3 km | 3-10 km | > 10 km |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| persistence | best F1 | 0.13 % | 0.233 | 0.0992 | 9.1 | 0.001 | 0.500 | 0.001 | 0.000 | 99 % | 1 % | 0 % |
| persistence | best F2 | 0.234 % | 0.337 | 0.0801 | 11.5 | 0.005 | 0.723 | 0.007 | 0.000 | 96 % | 4 % | 0 % |
| released | best F1 | 0.0725 % | 0.242 | 0.1854 | 4.4 | 0.006 | 0.518 | 0.008 | 0.000 | 98 % | 2 % | 0 % |
| released | best F2 | 0.165 % | 0.364 | 0.1229 | 7.1 | 0.034 | 0.755 | 0.063 | 0.001 | 90 % | 10 % | 0 % |
| full_s123 | best F1 | 0.0574 % | 0.255 | 0.2468 | 3.1 | 0.020 | 0.532 | 0.035 | 0.002 | 92 % | 6 % | 2 % |
| full_s123 | best F2 | 0.13 % | 0.371 | 0.1582 | 5.3 | 0.061 | 0.739 | 0.120 | 0.007 | 82 % | 15 % | 3 % |

Recall and false alarms per fire pixel caught at selected budgets:

| budget | persistence recall | persistence FA/TP | released recall | released FA/TP | full_s123 recall | full_s123 FA/TP |
|---|---|---|---|---|---|---|
| 0.051 % | 0.109 | 7.4 | 0.195 | 3.7 | 0.240 | 2.8 |
| 0.103 % | 0.195 | 8.5 | 0.293 | 5.3 | 0.337 | 4.5 |
| 0.208 % | 0.316 | 10.9 | 0.400 | 8.4 | 0.439 | 7.5 |
| 0.472 % | 0.460 | 17.5 | 0.513 | 15.6 | 0.545 | 14.6 |
| 0.953 % | 0.563 | 29.5 | 0.595 | 27.8 | 0.622 | 26.6 |
| 1.92 % | 0.645 | 52.7 | 0.673 | 50.5 | 0.690 | 49.2 |
| 4.91 % | 0.740 | 118.4 | 0.755 | 116.0 | 0.774 | 113.2 |
| 9.91 % | 0.812 | 218.6 | 0.800 | 222.1 | 0.831 | 213.6 |
