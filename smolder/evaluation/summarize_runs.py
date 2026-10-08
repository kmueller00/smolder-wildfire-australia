"""Validation metrics of training runs that were stopped before a national
evaluation, from their Lightning metrics.csv, as one comparable table.

All runs are scored by the training code on the same 1024 deterministic 2019
validation patches (384 px, seed 999). Per run: epochs completed; the highest
val_ap and its epoch, over all epochs and over the first MATCH epochs only
(so runs of different length compare at the same budget); at the epoch of
that best-within-MATCH val_ap: val_ap_newfire and the lift at the top 1, 5
and 15 % of the validation pixels (val_top{k}p0_recall / k).

Usage
  RUNS="name=path/to/metrics.csv,..." MATCH=10 OUT=results/experiments/x python -m smolder.evaluation.summarize_runs
"""
import json
import os

import pandas as pd

KS = (1, 5, 15)


def main():
    runs = [r.rsplit("=", 1) for r in os.environ["RUNS"].split(",")]
    match = int(os.environ.get("MATCH", 10))
    out = os.environ["OUT"]
    res = {"match_epochs": match, "runs": {}}
    for name, path in runs:
        m = pd.read_csv(path)
        v = m.dropna(subset=["val_ap"]).groupby("epoch").last()
        w = v.loc[v.index < match]
        e = int(w.val_ap.idxmax())
        r = dict(metrics=path, epochs=int(len(v)), best_val_ap=float(v.val_ap.max()),
                 best_epoch=int(v.val_ap.idxmax()), best_val_ap_first=float(w.val_ap.max()), best_epoch_first=e,
                 val_ap_newfire=float(v.loc[e, "val_ap_newfire"]) if "val_ap_newfire" in v else None)
        for k in KS:
            r[f"lift_top{k}"] = float(v.loc[e, f"val_top{k}p0_recall"] / (k / 100))
        res["runs"][name] = r
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(res, open(out + ".json", "w"), indent=1)
    L = [f"Validation metrics (1024 fixed 2019 patches) of runs stopped before a national evaluation. "
         f"'first {match}' = best over epochs 0 to {match - 1}; lifts at that epoch.", "",
         f"| run | epochs | best val_ap (epoch) | best val_ap, first {match} (epoch) | val_ap_newfire | "
         "lift top 1 % | lift top 5 % | lift top 15 % |", "|---|---|---|---|---|---|---|---|"]
    for name, r in res["runs"].items():
        nf = f"{r['val_ap_newfire']:.4f}" if r["val_ap_newfire"] is not None else ""
        L.append(f"| {name} | {r['epochs']} | {r['best_val_ap']:.4f} ({r['best_epoch']}) | "
                 f"{r['best_val_ap_first']:.4f} ({r['best_epoch_first']}) | {nf} | "
                 + " | ".join(f"{r[f'lift_top{k}']:.2f}" for k in KS) + " |")
    open(out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
