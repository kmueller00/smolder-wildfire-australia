"""Compare national evaluations (evaluate_national / evaluate_persistence) of
several runs on one year, against a reference (persistence by default).

Reads each run's national_<year>*.json and the matching *_daily.csv and writes
  - pooled AUC-PR / ROC-AUC (all fire and new fire)
  - pooled capture (TPR) and lift at the top 0.1/0.5/1/5/10 % of land pixels
  - mean daily AUC-PR and lift
  - per run against the reference: share of days won (daily AUC-PR, lift at
    0.5 % and 1 %, new-fire capture at 1 %) and a day-bootstrap 95 % interval
    of the mean daily difference
  - false alarms per detected fire pixel at each budget, and the budget (and
    change in false alarms) each run needs to capture as much fire as the
    reference does at that budget.

Usage
  RUNS="persistence=results/.../national_2019_persistence.json,full=results/.../national_2019.json" \
  REF=persistence OUT=results/experiments/smolder/comparison_2019 python -m smolder.evaluation.compare_national
"""
import json
import os

import numpy as np
import pandas as pd

KS = [0.001, 0.005, 0.01, 0.05, 0.1]
DAILY = ["auc_pr", "lift_0.005", "lift_0.01", "tpr_new_0.01"]
N_BOOT = 2000


def load(path):
    d = json.load(open(path))
    daily = pd.read_csv(path[:-len(".json")] + "_daily.csv")
    return d, daily


def topk(d, k):
    return next(r for r in d["topk_national"] if abs(r["k"] - k) < 1e-12)


def main():
    runs = [r.split("=", 1) for r in os.environ["RUNS"].split(",")]
    ref = os.environ.get("REF", runs[0][0])
    out = os.environ.get("OUT", "comparison")
    data = {n: load(p) for n, p in runs}
    rng = np.random.default_rng(0)
    res = {"runs": dict(runs), "reference": ref, "metrics": {}, "vs_reference": {}}
    for n, (d, daily) in data.items():
        m = dict(pooled_auc_pr=d["pooled_auc_pr"], pooled_roc_auc=d["pooled_roc_auc"],
                 pooled_auc_pr_new=d["pooled_auc_pr_new"], pooled_roc_auc_new=d["pooled_roc_auc_new"],
                 daily_auc_pr_mean=float(daily["auc_pr"].mean()), n_days=int(len(daily)))
        for k in KS:
            t = topk(d, k)
            m[f"tpr_{k}"], m[f"lift_{k}"] = t["tpr"], t["lift"]
            m[f"tpr_new_{k}"], m[f"lift_new_{k}"] = t["tpr_new"], t["lift_new"]
        res["metrics"][n] = m
    dref = data[ref][1].set_index("date")
    for n, (_, daily) in data.items():
        if n == ref:
            continue
        a = daily.set_index("date").join(dref, rsuffix="_ref", how="inner")
        v = {"n_days": int(len(a))}
        for c in DAILY:
            diff = (a[c] - a[c + "_ref"]).to_numpy(float)
            diff = diff[np.isfinite(diff)]
            boot = [rng.choice(diff, diff.size).mean() for _ in range(N_BOOT)]
            v[c] = dict(days_won=float((diff > 0).mean()), mean_diff=float(diff.mean()),
                        ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))])
        res["vs_reference"][n] = v
    # false alarms: per detected fire pixel at each budget, and at the reference's capture
    br = data[ref][0]["base_rate"]
    res["false_alarms"] = {}
    for n, (d, _) in data.items():
        tk = sorted(d["topk_national"], key=lambda r: r["k"])
        lk, tp = np.log([r["k"] for r in tk]), np.array([r["tpr"] for r in tk])
        fa = {}
        for k in KS:
            t = topk(d, k)
            fa[f"fp_per_tp_{k}"] = (1.0 - t["precision"]) / t["precision"]
            if n != ref:
                c = topk(data[ref][0], k)["tpr"]                   # reference capture at k
                if tp[0] <= c <= tp[-1]:
                    kn = float(np.exp(np.interp(c, tp, lk)))       # budget this run needs for it
                    fa[f"budget_for_ref_capture_{k}"] = kn
                    fa[f"fp_change_at_ref_capture_{k}"] = (kn - c * br) / (k - c * br) - 1.0
        res["false_alarms"][n] = fa
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(res, open(out + ".json", "w"), indent=1)

    names = [n for n, _ in runs]
    rows = [("pooled AUC-PR", "pooled_auc_pr", 4), ("pooled ROC-AUC", "pooled_roc_auc", 3),
            ("pooled AUC-PR new fire", "pooled_auc_pr_new", 4), ("mean daily AUC-PR", "daily_auc_pr_mean", 4)]
    for k in KS:
        rows += [(f"capture top {100 * k:g} %", f"tpr_{k}", 3), (f"lift top {100 * k:g} %", f"lift_{k}", 1),
                 (f"new-fire capture top {100 * k:g} %", f"tpr_new_{k}", 3)]
    lines = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for label, key, nd in rows:
        lines.append(f"| {label} | " + " | ".join(f"{res['metrics'][n][key]:.{nd}f}" for n in names) + " |")
    lines += ["", f"Against {ref} (per day; share of days won, mean difference, day-bootstrap 95 % CI):", "",
              "| run | metric | days won | mean diff | 95 % CI |", "|---|---|---|---|---|"]
    for n, v in res["vs_reference"].items():
        for c in DAILY:
            s = v[c]
            lines.append(f"| {n} | {c} | {100 * s['days_won']:.0f} % | {s['mean_diff']:+.4g} | "
                         f"[{s['ci95'][0]:+.4g}, {s['ci95'][1]:+.4g}] |")
    lines += ["", "False alarms per detected fire pixel at each budget (top k % of land pixels):", "",
              "| budget | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for k in KS:
        lines.append(f"| top {100 * k:g} % | " + " | ".join(
            f"{res['false_alarms'][n][f'fp_per_tp_{k}']:.1f}" for n in names) + " |")
    lines += ["", f"Budget needed to capture as much fire as {ref} does with the top k % "
              "(TPR interpolated linearly in log k between evaluated budgets), and the change in false alarms:", "",
              "| run | ref budget | ref capture | budget needed | false alarms |", "|---|---|---|---|---|"]
    for n in names:
        if n == ref:
            continue
        for k in KS:
            fa = res["false_alarms"][n]
            if f"budget_for_ref_capture_{k}" in fa:
                lines.append(f"| {n} | {100 * k:g} % | {topk(data[ref][0], k)['tpr']:.3f} | "
                             f"{100 * fa[f'budget_for_ref_capture_{k}']:.3g} % | "
                             f"{100 * fa[f'fp_change_at_ref_capture_{k}']:+.0f} % |")
    open(out + ".md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
