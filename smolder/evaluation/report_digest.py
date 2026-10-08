"""All numbers of one run for the report, as Markdown tables, each table with
the results file and keys it is read from. Nothing is computed here except
1 - retention (share of the top-1 % pixels displaced) and percentages.

Rounding: AUC-PR, ROC-AUC, F1, F2 and precision to 4 decimals; shares in % to
1 decimal, daily areas flagged in % to 3 decimals; lift and false alarms per hit to 1 decimal; counts as integers.

  RUN=results/experiments/smolder/full_model_windalign_seed123 \\
  PERSIST_2020=results/national_2020_persistence.json \\
  OTHERS_2019="A=results/.../full_model_cos40_seed123/national_2019.json,reference=results/.../full_model_seed123/national_2019.json" \\
  python -m smolder.evaluation.report_digest > .../report_digest.md
"""
import json
import os

import numpy as np
import pandas as pd

RUN = os.environ["RUN"]
T = os.path.join(RUN, "test_2020")
J = lambda p: json.load(open(p))                    # noqa: E731
f4 = lambda v: f"{v:.4f}"                            # noqa: E731
pc = lambda v: f"{100 * v:.1f} %"                    # noqa: E731
f1d = lambda v: f"{v:.1f}"                           # noqa: E731
pa = lambda v: f"{100 * v:.3f} %"                    # noqa: E731  areas below 1 %
L = []


def table(title, src, head, rows):
    L.extend(["", f"### {title}", "", f"Source: {src}", "", "| " + " | ".join(head) + " |",
              "|" + "---|" * len(head)])
    L.extend("| " + " | ".join(str(c) for c in r) + " |" for r in rows)


def main():
    rel = lambda p: os.path.relpath(p)               # noqa: E731
    # ---------------- training ----------------
    tc = pd.read_csv(os.path.join(RUN, "training_curve.csv"))
    best = tc.loc[tc.val_ap.idxmax()]
    table("Training", f"`{rel(os.path.join(RUN, 'training_curve.csv'))}` (columns epoch, val_ap, in_swa)",
          ["quantity", "value"],
          [["epochs trained", f"{int(tc.epoch.min())} to {int(tc.epoch.max())} ({len(tc)})"],
           ["highest validation AP (epoch)", f"{f4(best.val_ap)} ({int(best.epoch)})"],
           ["epochs averaged", ", ".join(str(int(e)) for e in tc.epoch[tc.in_swa == 1])],
           ["validation AP of the averaged epochs", ", ".join(f4(v) for v in tc.val_ap[tc.in_swa == 1])]])
    # ---------------- 2019 selection ----------------
    rows = []
    others = [o.split("=", 1) for o in os.environ.get("OTHERS_2019", "").split(",") if o]
    for name, p in [("B (this run)", os.path.join(RUN, "national_2019.json"))] + others:
        d = J(p)
        rows.append([name, f4(d["pooled_auc_pr"]), f4(d["pooled_auc_pr_new"]), f4(d["pooled_roc_auc"]),
                     f4(d["daily"]["auc_pr"]["mean"]), d["n_days"], f"`{rel(p)}`"])
    table("Validation year 2019 (model selection)", "each run's `national_2019.json`: pooled_auc_pr, pooled_auc_pr_new, "
          "pooled_roc_auc, daily.auc_pr.mean, n_days",
          ["run", "pooled AUC-PR", "AUC-PR new fire", "ROC-AUC", "mean daily AUC-PR", "issue days", "file"], rows)
    ab19 = J(os.path.join(RUN, "adaptive_budget_2019.json"))["adaptive_best"]
    table("2019 best adaptive threshold (thresholds that 2020 uses)", f"`{rel(os.path.join(RUN, 'adaptive_budget_2019.json'))}`: "
          "adaptive_best.<ranking>.<f1|f2>",
          ["ranking", "criterion", "F1", "F2", "recall", "precision", "mean daily area", "threshold (logit)"],
          [[n, c.upper(), f4(x["f1"]), f4(x["f2"]), f4(x["recall"]), f4(x["precision"]), pa(x["mean_share"]),
            f"{x['threshold']:.3f}"]
           for n in ("persistence", "SMOLDER") for c in ("f1", "f2") for x in [ab19[n][c]]])
    # ---------------- 2020 national ----------------
    S = J(os.path.join(T, "national_2020.json")); P = J(os.environ["PERSIST_2020"])
    table("Test year 2020: pooled measures", f"`{rel(os.path.join(T, 'national_2020.json'))}` and "
          f"`{rel(os.environ['PERSIST_2020'])}`: pooled_auc_pr, pooled_auc_pr_new, pooled_roc_auc, "
          "pooled_roc_auc_new, daily.auc_pr.mean, base_rate, cells.pooled_auc_pr, n_days, fire_px_total, n_land_px",
          ["measure", "SMOLDER", "persistence"],
          [["pooled AUC-PR", f4(S["pooled_auc_pr"]), f4(P["pooled_auc_pr"])],
           ["pooled AUC-PR, new fire", f4(S["pooled_auc_pr_new"]), f4(P["pooled_auc_pr_new"])],
           ["pooled ROC-AUC", f4(S["pooled_roc_auc"]), f4(P["pooled_roc_auc"])],
           ["pooled ROC-AUC, new fire", f4(S["pooled_roc_auc_new"]), f4(P["pooled_roc_auc_new"])],
           ["mean daily AUC-PR", f4(S["daily"]["auc_pr"]["mean"]), f4(P["daily"]["auc_pr"]["mean"])],
           ["AUC-PR on 25 km cells", f4(S["cells"]["pooled_auc_pr"]), f4(P["cells"]["pooled_auc_pr"]) if "cells" in P else ""],
           ["base rate", f"{100 * S['base_rate']:.4f} %", f"{100 * P['base_rate']:.4f} %"],
           ["issue days", S["n_days"], P["n_days"]],
           ["fire pixels (all days)", S["fire_px_total"], P.get("fire_px_total", "")],
           ["land pixels per day", S["n_land_px"], P.get("n_land_px", "")]])
    tk = {round(x["k"], 4): x for x in S["topk_national"]}; tp = {round(x["k"], 4): x for x in P["topk_national"]}
    table("Test year 2020: fixed share of land flagged each day (mean over days)",
          "topk_national[k].tpr, .precision, .lift, .tpr_new, .lift_new of both files",
          ["land flagged", "fire caught S", "fire caught P", "precision S", "precision P", "lift S", "lift P",
           "new fire caught S", "new fire caught P"],
          [[pc(k), pc(tk[k]["tpr"]), pc(tp[k]["tpr"]), f4(tk[k]["precision"]), f4(tp[k]["precision"]),
            f1d(tk[k]["lift"]), f1d(tp[k]["lift"]), pc(tk[k]["tpr_new"]), pc(tp[k]["tpr_new"])] for k in sorted(tk)])
    for grp, lab in (("season", "season"), ("kg", "climate zone"), ("band", "latitude band")):
        table(f"Test year 2020 by {lab}", f"by_group.{grp}.<class>.auc_pr, .roc_auc, .fire_px, .base_rate of both files",
              ["class", "AUC-PR S", "AUC-PR P", "ratio", "ROC-AUC S", "fire pixels", "base rate"],
              [[c, f4(v["auc_pr"]), f4(P["by_group"][grp][c]["auc_pr"]), f"{v['auc_pr'] / P['by_group'][grp][c]['auc_pr']:.2f}",
                f4(v["roc_auc"]), v["fire_px"], f"{100 * v['base_rate']:.4f} %"]
               for c, v in S["by_group"][grp].items()])
    # ---------------- comparison ----------------
    C = J(os.path.join(T, "comparison_2020.json"))
    vs = C["vs_reference"]["SMOLDER"]
    table("Test year 2020: day by day against persistence", f"`{rel(os.path.join(T, 'comparison_2020.json'))}`: "
          "vs_reference.SMOLDER.<metric>.days_won, .mean_diff, .ci95 (2000 bootstrap resamples of days)",
          ["daily measure", "days SMOLDER higher", "mean difference", "95 % interval"],
          [[m, pc(vs[m]["days_won"]), f"{vs[m]['mean_diff']:+.4f}", f"[{vs[m]['ci95'][0]:+.4f}, {vs[m]['ci95'][1]:+.4f}]"]
           for m in ("auc_pr", "lift_0.005", "lift_0.01", "tpr_new_0.01")])
    fa = C["false_alarms"]
    table("Test year 2020: false alarms", "false_alarms.<run>.fp_per_tp_<k>, .budget_for_ref_capture_<k>, "
          ".fp_change_at_ref_capture_<k>",
          ["land flagged", "false alarms per hit P", "false alarms per hit S", "S needs for P's catch",
           "change in false alarms"],
          [[pc(float(k)), f1d(fa["persistence"][f"fp_per_tp_{k}"]), f1d(fa["SMOLDER"][f"fp_per_tp_{k}"]),
            pa(fa["SMOLDER"][f"budget_for_ref_capture_{k}"]) if f"budget_for_ref_capture_{k}" in fa["SMOLDER"]
            else "not defined (below the smallest share evaluated)",
            pc(fa["SMOLDER"][f"fp_change_at_ref_capture_{k}"]) if f"fp_change_at_ref_capture_{k}" in fa["SMOLDER"] else ""]
           for k in ("0.001", "0.005", "0.01", "0.05", "0.1")])
    # ---------------- operating points ----------------
    A20 = J(os.path.join(T, "adaptive_budget_2020.json"))
    fire = S["fire_px_total"]
    rows = []
    for n in ("persistence", "SMOLDER"):
        for c in ("f1", "f2"):
            x = A20["adaptive_best"][n][c]; tpv = x["recall"] * fire
            rows.append([n, c.upper(), f4(x["f1"]), f4(x["f2"]), f4(x["recall"]), f4(x["precision"]),
                         pa(x["mean_share"]), f1d(x["fp_per_tp"]), int(round(tpv)), int(round(fire - tpv)),
                         int(round(x["fp_per_tp"] * tpv)), pc(x["recall_new"]),
                         " / ".join(pc(x[f"recall {b}"]) for b in ("0-3 km", "3-10 km", "> 10 km"))])
    table("Test year 2020: one score threshold for all days, fixed on 2019 (adaptive daily area)",
          f"`{rel(os.path.join(T, 'adaptive_budget_2020.json'))}` (thresholds_from = adaptive_budget_2019.json): "
          "adaptive_best.<ranking>.<f1|f2>; TP = recall x fire pixels, FN = fire pixels - TP, FP = fp_per_tp x TP",
          ["ranking", "2019 criterion", "F1", "F2", "recall", "precision", "mean daily area", "false alarms per hit",
           "TP", "FN", "FP", "new fire caught", "caught 0-3 / 3-10 / > 10 km"], rows)
    O19 = J(os.path.join(RUN, "operating_point_2019.json")); O20 = J(os.path.join(T, "operating_point_2020.json"))
    rows = []
    for n in ("persistence", "B"):
        for c in ("f1", "f2"):
            r19 = O19["rankings"][n]; i = int(np.argmax(r19[c])); k = r19["k"][i]
            r20 = O20["rankings"][n]; j = int(np.argmin(np.abs(np.array(r20["k"]) - k)))
            rows.append([n, c.upper(), pa(k), f4(r20["f1"][j]), f4(r20["f2"][j]), f4(r20["recall"][j]),
                         f4(r20["precision"][j]), f1d(r20["fp_per_tp"][j])])
    table("Test year 2020: one fixed share of land for all days, chosen on 2019",
          f"`{rel(os.path.join(RUN, 'operating_point_2019.json'))}` (share with the highest F1/F2) read off in "
          f"`{rel(os.path.join(T, 'operating_point_2020.json'))}`: rankings.<ranking>.<k|f1|f2|recall|precision|fp_per_tp>[i]",
          ["ranking", "2019 criterion", "share chosen on 2019", "F1", "F2", "recall", "precision", "false alarms per hit"], rows)
    # ---------------- distance ----------------
    DB = J(os.path.join(T, "distance_band_diagnostic_2020.json"))["bands"]
    table("Test year 2020: skill by distance to fire of the last 32 days",
          f"`{rel(os.path.join(T, 'distance_band_diagnostic_2020.json'))}`: bands.<band>.fire_share, .auc_pr, .roc_auc, "
          ".base_rate, .capture_0.005, .lift_0.005, .capture_0.01, .lift_0.01",
          ["band", "share of fire", "AUC-PR", "ROC-AUC", "base rate", "caught at 0.5 %", "lift at 0.5 %",
           "caught at 1 %", "lift at 1 %"],
          [[b, pc(v["fire_share"]), f4(v["auc_pr"]), f4(v["roc_auc"]), f"{100 * v['base_rate']:.4f} %",
            pc(v["capture_0.005"]), f1d(v["lift_0.005"]), pc(v["capture_0.01"]), f1d(v["lift_0.01"])]
           for b, v in DB.items()])
    FD = J(os.path.join(T, "fire_distance_2020.json"))["all_days"]
    table("Test year 2020: fire caught by distance to fire of days D-2..D",
          f"`{rel(os.path.join(T, 'fire_distance_2020.json'))}`: all_days.<smolder|persistence>.bands.<band>.fire_share, "
          ".captured_0.005, .captured_0.01",
          ["band", "share of fire", "caught at 0.5 % S", "caught at 0.5 % P", "caught at 1 % S", "caught at 1 % P"],
          [[b, pc(v["fire_share"]), pc(v["captured_0.005"]), pc(FD["persistence"]["bands"][b]["captured_0.005"]),
            pc(v["captured_0.01"]), pc(FD["persistence"]["bands"][b]["captured_0.01"])]
           for b, v in FD["smolder"]["bands"].items()])
    # ---------------- importance ----------------
    E = J(os.path.join(T, "explain_2020.json"))
    g = sorted(E["groups"].items(), key=lambda kv: kv[1]["retention"])
    table("Test year 2020: permutation importance",
          f"`{rel(os.path.join(T, 'explain_2020.json'))}`: groups.<input>.retention (displaced = 1 - retention), "
          f".ap_drop; {E['n_patches']} fire-active patches, top {pc(E['top_fraction'])}, base AUC-PR "
          f"{f4(E['base_auc_pr'])} (base_auc_pr)",
          ["input", "top-1 % pixels displaced", "relative fall of AUC-PR"],
          [[k, pc(1 - v["retention"]), pc(v["ap_drop"])] for k, v in g])
    # ---------------- events ----------------
    EV = J(os.path.join(T, "event_maps_2020.json"))
    table("Test year 2020: the event maps",
          f"`{rel(os.path.join(T, 'event_maps_2020.json'))}`: events.<event>.date, .fire_px, .<ranking>.caught, "
          f".false_alarms; budget {pa(EV['budget'])} per day; adaptive = 2019 threshold {EV['adaptive_logit_threshold']:.3f}",
          ["event", "date", "fire pixels", "caught P", "false alarms P", "caught S", "false alarms S",
           "caught S adaptive", "false alarms S adaptive"],
          [[k, v["date"], v["fire_px"], pc(v["persistence"]["caught"]), v["persistence"]["false_alarms"],
            pc(v["full"]["caught"]), v["full"]["false_alarms"], pc(v["full_adaptive"]["caught"]),
            v["full_adaptive"]["false_alarms"]] for k, v in EV["events"].items()])
    print("\n".join(L))


if __name__ == "__main__":
    main()
