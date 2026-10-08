Validation metrics (1024 fixed 2019 patches) of runs stopped before a national evaluation. 'first 10' = best over epochs 0 to 9; lifts at that epoch.

| run | epochs | best val_ap (epoch) | best val_ap, first 10 (epoch) | val_ap_newfire | lift top 1 % | lift top 5 % | lift top 15 % |
|---|---|---|---|---|---|---|---|
| reference (a = 0) | 15 | 0.1042 (13) | 0.0898 (9) | 0.0023 | 22.33 | 5.80 | 2.35 |
| distance weight a = 5 | 10 | 0.0736 (3) | 0.0736 (3) | 0.0018 | 22.24 | 5.56 | 2.23 |
| distance weight a = 10 | 10 | 0.0722 (3) | 0.0722 (3) | 0.0019 | 22.05 | 5.56 | 2.23 |
| distance weight a = 20 | 12 | 0.0730 (5) | 0.0730 (5) | 0.0020 | 21.97 | 5.72 | 2.32 |
| distance weight a = 40 | 12 | 0.0593 (5) | 0.0593 (5) | 0.0023 | 21.41 | 5.58 | 2.29 |
| distance weight a = 80 | 14 | 0.0583 (7) | 0.0583 (7) | 0.0022 | 21.46 | 5.58 | 2.31 |
| VPD anomaly | 15 | 0.0867 (13) | 0.0795 (9) | 0.0019 | 22.53 | 5.76 | 2.36 |
| VPD from the ERA5-based product | 10 | 0.0794 (3) | 0.0794 (3) | 0.0014 | 21.97 | 5.44 | 2.20 |
| future weather (stopped after 5 epochs) | 5 | 0.0855 (2) | 0.0855 (2) | 0.0022 | 22.17 | 5.66 | 2.28 |
