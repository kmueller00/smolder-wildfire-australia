| metric | persistence | SMOLDER |
|---|---|---|
| pooled AUC-PR | 0.0361 | 0.1167 |
| pooled ROC-AUC | 0.936 | 0.921 |
| pooled AUC-PR new fire | 0.0022 | 0.0058 |
| mean daily AUC-PR | 0.0377 | 0.1122 |
| capture top 0.1 % | 0.217 | 0.309 |
| lift top 0.1 % | 217.0 | 308.6 |
| new-fire capture top 0.1 % | 0.001 | 0.063 |
| capture top 0.5 % | 0.393 | 0.460 |
| lift top 0.5 % | 78.5 | 92.0 |
| new-fire capture top 0.5 % | 0.083 | 0.209 |
| capture top 1 % | 0.457 | 0.530 |
| lift top 1 % | 45.7 | 53.0 |
| new-fire capture top 1 % | 0.186 | 0.308 |
| capture top 5 % | 0.643 | 0.708 |
| lift top 5 % | 12.9 | 14.2 |
| new-fire capture top 5 % | 0.474 | 0.566 |
| capture top 10 % | 0.753 | 0.775 |
| lift top 10 % | 7.5 | 7.7 |
| new-fire capture top 10 % | 0.637 | 0.665 |

Against persistence (per day; share of days won, mean difference, day-bootstrap 95 % CI):

| run | metric | days won | mean diff | 95 % CI |
|---|---|---|---|---|
| SMOLDER | auc_pr | 99 % | +0.07445 | [+0.06863, +0.08036] |
| SMOLDER | lift_0.005 | 98 % | +13.49 | [+12.51, +14.56] |
| SMOLDER | lift_0.01 | 95 % | +7.345 | [+6.767, +8.005] |
| SMOLDER | tpr_new_0.01 | 96 % | +0.1223 | [+0.1145, +0.1305] |

False alarms per detected fire pixel at each budget (top k % of land pixels):

| budget | persistence | SMOLDER |
|---|---|---|
| top 0.1 % | 14.5 | 9.1 |
| top 0.5 % | 37.4 | 31.8 |
| top 1 % | 64.0 | 55.3 |
| top 5 % | 229.7 | 207.3 |
| top 10 % | 399.2 | 382.3 |

Budget needed to capture as much fire as persistence does with the top k % (TPR interpolated linearly in log k between evaluated budgets), and the change in false alarms:

| run | ref budget | ref capture | budget needed | false alarms |
|---|---|---|---|---|
| SMOLDER | 0.5 % | 0.393 | 0.242 % | -53 % |
| SMOLDER | 1 % | 0.457 | 0.484 % | -52 % |
| SMOLDER | 5 % | 0.643 | 2.77 % | -45 % |
| SMOLDER | 10 % | 0.753 | 8.02 % | -20 % |
