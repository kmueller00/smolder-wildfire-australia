| metric | persistence | SMOLDER |
|---|---|---|
| pooled AUC-PR | 0.0361 | 0.1206 |
| pooled ROC-AUC | 0.936 | 0.930 |
| pooled AUC-PR new fire | 0.0022 | 0.0058 |
| mean daily AUC-PR | 0.0377 | 0.1127 |
| capture top 0.1 % | 0.217 | 0.308 |
| lift top 0.1 % | 217.0 | 308.2 |
| new-fire capture top 0.1 % | 0.001 | 0.062 |
| capture top 0.5 % | 0.393 | 0.459 |
| lift top 0.5 % | 78.5 | 91.8 |
| new-fire capture top 0.5 % | 0.083 | 0.207 |
| capture top 1 % | 0.457 | 0.531 |
| lift top 1 % | 45.7 | 53.1 |
| new-fire capture top 1 % | 0.186 | 0.307 |
| capture top 5 % | 0.643 | 0.714 |
| lift top 5 % | 12.9 | 14.3 |
| new-fire capture top 5 % | 0.474 | 0.576 |
| capture top 10 % | 0.753 | 0.786 |
| lift top 10 % | 7.5 | 7.9 |
| new-fire capture top 10 % | 0.637 | 0.682 |

Against persistence (per day; share of days won, mean difference, day-bootstrap 95 % CI):

| run | metric | days won | mean diff | 95 % CI |
|---|---|---|---|---|
| SMOLDER | auc_pr | 99 % | +0.07495 | [+0.06926, +0.08074] |
| SMOLDER | lift_0.005 | 98 % | +13.26 | [+12.35, +14.27] |
| SMOLDER | lift_0.01 | 96 % | +7.36 | [+6.802, +8.004] |
| SMOLDER | tpr_new_0.01 | 96 % | +0.1219 | [+0.1144, +0.1298] |

False alarms per detected fire pixel at each budget (top k % of land pixels):

| budget | persistence | SMOLDER |
|---|---|---|
| top 0.1 % | 14.5 | 9.1 |
| top 0.5 % | 37.4 | 31.9 |
| top 1 % | 64.0 | 55.2 |
| top 5 % | 229.7 | 205.7 |
| top 10 % | 399.2 | 378.2 |

Budget needed to capture as much fire as persistence does with the top k % (TPR interpolated linearly in log k between evaluated budgets), and the change in false alarms:

| run | ref budget | ref capture | budget needed | false alarms |
|---|---|---|---|---|
| SMOLDER | 0.5 % | 0.393 | 0.246 % | -52 % |
| SMOLDER | 1 % | 0.457 | 0.491 % | -52 % |
| SMOLDER | 5 % | 0.643 | 2.68 % | -47 % |
| SMOLDER | 10 % | 0.753 | 7.28 % | -27 % |
