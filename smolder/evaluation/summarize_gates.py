"""Readable summary of the near-field gate files (frp_gate.py output) in one
markdown file: per file and distance band the number of fire pixels, the base
rate, the AUC-PR of every feature set and every gain with its day-bootstrap
95 % interval, plus the feature list of each set.

Usage
  FILES="results/frp_gate_2019.json,results/nearfield_gate_pm_2019.json,..." \\
      OUT=results/nearfield_gates_2019.md python -m smolder.evaluation.summarize_gates
"""
import json
import os


def main():
    files = os.environ["FILES"].split(",")
    L = ["Near-field gates, 2019 (gradient-boosted trees on pixels within 25 px of fire of days D-2..D, "
         "fitted on odd and scored on even issue days; S = with the released model's score as a feature). "
         "Gains are differences in AUC-PR, relative gain in brackets, with the day-bootstrap 95 % interval.", ""]
    for f in files:
        d = json.load(open(f))
        L += [f"## {os.path.basename(f)}", "", "Feature sets:", ""]
        for k, v in d["models"].items():
            L.append(f"- {k}: {', '.join(v)}")
        for band, b in d["bands"].items():
            L += ["", f"### {band}: {b['n_fire']} fire pixels, base rate {b['base_rate']:.5f}", "",
                  "| set | AUC-PR | ROC-AUC |", "|---|---|---|"]
            for k in b["auc_pr"]:
                L.append(f"| {k} | {b['auc_pr'][k]:.4f} | {b['roc_auc'][k]:.3f} |")
            L += ["", "| gain | d AUC-PR | relative | 95 % interval |", "|---|---|---|---|"]
            for k, g in b["gains"].items():
                lo, hi = g["d_ap_ci95"]
                L.append(f"| {k} | {g['d_ap']:+.4f} | {100 * g['d_ap_relative']:+.0f} % | [{lo:+.4f}, {hi:+.4f}] |")
        L.append("")
    open(os.environ["OUT"], "w").write("\n".join(L) + "\n")
    print(f"wrote {os.environ['OUT']}")


if __name__ == "__main__":
    main()
