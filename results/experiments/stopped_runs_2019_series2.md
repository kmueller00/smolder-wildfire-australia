Validation metrics (1024 fixed 2019 patches) of runs stopped before a national evaluation. 'first 9' = best over epochs 0 to 8; lifts at that epoch.

| run | epochs | best val_ap (epoch) | best val_ap, first 9 (epoch) | val_ap_newfire | lift top 1 % | lift top 5 % | lift top 15 % |
|---|---|---|---|---|---|---|---|
| without the new inputs | 14 | 0.1335 (10) | 0.1218 (8) | 0.0030 | 24.77 | 6.42 | 2.58 |
| without the new inputs and old loader | 9 | 0.0817 (5) | 0.0817 (5) | 0.0018 | 22.68 | 5.68 | 2.31 |
| with the new inputs | 30 | 0.1997 (23) | 0.1718 (7) | 0.0030 | 24.87 | 6.34 | 2.57 |
