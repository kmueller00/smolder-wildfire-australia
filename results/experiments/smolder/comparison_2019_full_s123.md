| metric | persistence | released | full_s123 |
|---|---|---|---|
| pooled AUC-PR | 0.0619 | 0.1405 | 0.2008 |
| pooled ROC-AUC | 0.936 | 0.907 | 0.925 |
| pooled AUC-PR new fire | 0.0036 | 0.0053 | 0.0085 |
| mean daily AUC-PR | 0.0601 | 0.1241 | 0.1755 |
| capture top 0.1 % | 0.213 | 0.306 | 0.347 |
| lift top 0.1 % | 212.9 | 306.2 | 346.7 |
| new-fire capture top 0.1 % | 0.001 | 0.019 | 0.057 |
| capture top 0.5 % | 0.462 | 0.508 | 0.537 |
| lift top 0.5 % | 92.5 | 101.6 | 107.4 |
| new-fire capture top 0.5 % | 0.087 | 0.175 | 0.238 |
| capture top 1 % | 0.548 | 0.581 | 0.605 |
| lift top 1 % | 54.8 | 58.1 | 60.5 |
| new-fire capture top 1 % | 0.230 | 0.294 | 0.344 |
| capture top 5 % | 0.712 | 0.735 | 0.754 |
| lift top 5 % | 14.2 | 14.7 | 15.1 |
| new-fire capture top 5 % | 0.522 | 0.555 | 0.586 |
| capture top 10 % | 0.790 | 0.784 | 0.813 |
| lift top 10 % | 7.9 | 7.8 | 8.1 |
| new-fire capture top 10 % | 0.650 | 0.634 | 0.684 |

Against persistence (per day; share of days won, mean difference, day-bootstrap 95 % CI):

| run | metric | days won | mean diff | 95 % CI |
|---|---|---|---|---|
| released | auc_pr | 100 % | +0.06398 | [+0.06025, +0.06799] |
| released | lift_0.005 | 92 % | +9.118 | [+8.51, +9.727] |
| released | lift_0.01 | 87 % | +3.33 | [+2.987, +3.664] |
| released | tpr_new_0.01 | 89 % | +0.06329 | [+0.05779, +0.06882] |
| full_s123 | auc_pr | 100 % | +0.1154 | [+0.1094, +0.1215] |
| full_s123 | lift_0.005 | 98 % | +14.94 | [+14.13, +15.73] |
| full_s123 | lift_0.01 | 93 % | +5.76 | [+5.35, +6.174] |
| full_s123 | tpr_new_0.01 | 94 % | +0.1143 | [+0.1069, +0.1215] |

False alarms per detected fire pixel at each budget (top k % of land pixels):

| budget | persistence | released | full_s123 |
|---|---|---|---|
| top 0.1 % | 8.5 | 5.2 | 4.4 |
| top 0.5 % | 18.2 | 16.3 | 15.3 |
| top 1 % | 30.6 | 29.0 | 27.7 |
| top 5 % | 120.3 | 118.0 | 115.1 |
| top 10 % | 220.4 | 224.0 | 215.5 |

Budget needed to capture as much fire as persistence does with the top k % (TPR interpolated linearly in log k between evaluated budgets), and the change in false alarms:

| run | ref budget | ref capture | budget needed | false alarms |
|---|---|---|---|---|
| released | 0.5 % | 0.462 | 0.341 % | -34 % |
| released | 1 % | 0.548 | 0.729 % | -28 % |
| released | 5 % | 0.712 | 3.87 % | -23 % |
| full_s123 | 0.5 % | 0.462 | 0.256 % | -51 % |
| full_s123 | 1 % | 0.548 | 0.557 % | -46 % |
| full_s123 | 5 % | 0.712 | 3.18 % | -37 % |
| full_s123 | 10 % | 0.790 | 7.6 % | -24 % |
