| metric | persistence | SMOLDER |
|---|---|---|
| pooled AUC-PR | 0.0609 | 0.2091 |
| pooled ROC-AUC | 0.935 | 0.930 |
| pooled AUC-PR new fire | 0.0036 | 0.0093 |
| mean daily AUC-PR | 0.0595 | 0.1808 |
| capture top 0.1 % | 0.214 | 0.353 |
| lift top 0.1 % | 213.7 | 352.6 |
| new-fire capture top 0.1 % | 0.001 | 0.066 |
| capture top 0.5 % | 0.462 | 0.541 |
| lift top 0.5 % | 92.5 | 108.2 |
| new-fire capture top 0.5 % | 0.087 | 0.249 |
| capture top 1 % | 0.547 | 0.611 |
| lift top 1 % | 54.7 | 61.1 |
| new-fire capture top 1 % | 0.231 | 0.357 |
| capture top 5 % | 0.712 | 0.763 |
| lift top 5 % | 14.2 | 15.3 |
| new-fire capture top 5 % | 0.521 | 0.602 |
| capture top 10 % | 0.789 | 0.821 |
| lift top 10 % | 7.9 | 8.2 |
| new-fire capture top 10 % | 0.650 | 0.698 |

Against persistence (per day; share of days won, mean difference, day-bootstrap 95 % CI):

| run | metric | days won | mean diff | 95 % CI |
|---|---|---|---|---|
| SMOLDER | auc_pr | 100 % | +0.1212 | [+0.1153, +0.1273] |
| SMOLDER | lift_0.005 | 97 % | +15.67 | [+14.87, +16.49] |
| SMOLDER | lift_0.01 | 95 % | +6.388 | [+5.931, +6.826] |
| SMOLDER | tpr_new_0.01 | 95 % | +0.1254 | [+0.1178, +0.1325] |

False alarms per detected fire pixel at each budget (top k % of land pixels):

| budget | persistence | SMOLDER |
|---|---|---|
| top 0.1 % | 8.5 | 4.4 |
| top 0.5 % | 18.5 | 15.5 |
| top 1 % | 31.3 | 28.0 |
| top 5 % | 123.1 | 116.5 |
| top 10 % | 225.5 | 218.8 |

Budget needed to capture as much fire as persistence does with the top k % (TPR interpolated linearly in log k between evaluated budgets), and the change in false alarms:

| run | ref budget | ref capture | budget needed | false alarms |
|---|---|---|---|---|
| SMOLDER | 0.5 % | 0.462 | 0.246 % | -54 % |
| SMOLDER | 1 % | 0.547 | 0.533 % | -48 % |
| SMOLDER | 5 % | 0.712 | 2.85 % | -43 % |
| SMOLDER | 10 % | 0.789 | 6.82 % | -32 % |
