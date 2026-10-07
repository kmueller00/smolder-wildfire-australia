"""Numbers the report derives from other result files, written to a file so that
each one has a file and key.

For YEAR (default 2019) and the run RUN (default the final model):
  caught_by_band   at the best-F1 and best-F2 thresholds (adaptive_budget_<YEAR>.json,
                   adaptive_best): the share of the fire caught that lies 0-3 km,
                   3-10 km and more than 10 km from recent fire, from the recall per
                   band and the share of all fire per band (operating_point_<YEAR>.json,
                   bands.fire_share)
  ratios           SMOLDER over persistence: pooled AUC-PR (national_<YEAR>.json and the
                   persistence file), precision and flagged area at the best-F2 thresholds;
                   pooled AUC-PR over the base rate of the year, for both
  common_days      (YEAR=2019 only) daily AUC-PR of the final model and of model B
                   (earlier inputs) on the issue days both were evaluated on: mean of
                   each, share of days on which the final model is higher

  RUN_DIR=... PERSIST=... OUT=... python -m smolder.evaluation.report_derived
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results", "experiments")
YEAR = int(os.environ.get("YEAR", 2019))
RUN_DIR = os.environ.get("RUN_DIR", os.path.join(RES, "final", "causal_b_50ep_seed123")
                         + ("" if YEAR == 2019 else "/../../smolder/causal_b_50ep_seed123/test_2020"))
PERSIST = os.environ.get("PERSIST", os.path.join(RES, "final", f"persistence_{YEAR}",
                                                 f"national_{YEAR}_persistence.json"))
MODEL_B = os.path.join(RES, "smolder", "full_model_windalign_seed123", f"national_{YEAR}_daily.csv")
OUT = os.environ.get("OUT", os.path.join(RUN_DIR, f"report_derived_{YEAR}.json"))
BANDS = ["0-3 km", "3-10 km", "> 10 km"]


def main():
    ab = json.load(open(os.path.join(RUN_DIR, f"adaptive_budget_{YEAR}.json")))["adaptive_best"]
    share = json.load(open(os.path.join(RUN_DIR, f"operating_point_{YEAR}.json")))["bands"]["fire_share"]
    out = dict(year=YEAR, run_dir=os.path.relpath(RUN_DIR, os.path.join(HERE, "..", "..")),
               sources=dict(adaptive=f"adaptive_budget_{YEAR}.json: adaptive_best",
                            bands=f"operating_point_{YEAR}.json: bands.fire_share",
                            national=f"national_{YEAR}.json", persistence=os.path.basename(PERSIST)),
               fire_share_by_band=dict(zip(BANDS, share)), caught_by_band={})
    for name in ("SMOLDER", "persistence"):
        for crit in ("f1", "f2"):
            b = ab[name][crit]
            caught = np.array([b[f"recall {k}"] * s for k, s in zip(BANDS, share)])
            out["caught_by_band"][f"{name}.{crit}"] = dict(
                recall=b["recall"], recall_from_bands=float(caught.sum()),
                share_of_caught=dict(zip(BANDS, (caught / caught.sum()).tolist())))
    S = json.load(open(os.path.join(RUN_DIR, f"national_{YEAR}.json")))
    P = json.load(open(PERSIST))
    out["ratios"] = dict(pooled_auc_pr=S["pooled_auc_pr"] / P["pooled_auc_pr"],
                         precision_f2=ab["SMOLDER"]["f2"]["precision"] / ab["persistence"]["f2"]["precision"],
                         auc_pr_over_base_rate_SMOLDER=S["pooled_auc_pr"] / S["base_rate"],
                         auc_pr_over_base_rate_persistence=P["pooled_auc_pr"] / P["base_rate"],
                         area_f2_SMOLDER_over_persistence=ab["SMOLDER"]["f2"]["mean_share"]
                         / ab["persistence"]["f2"]["mean_share"])
    if YEAR == 2019 and os.path.exists(MODEL_B):
        a = pd.read_csv(os.path.join(RUN_DIR, f"national_{YEAR}_daily.csv"))[["date", "auc_pr"]]
        b = pd.read_csv(MODEL_B)[["date", "auc_pr"]]
        j = a.merge(b, on="date", suffixes=("_final", "_model_B")).dropna()
        out["common_days"] = dict(n_days=int(len(j)), n_days_final=int(len(a)), n_days_model_B=int(len(b)),
                                  daily_auc_pr_mean_final=float(j.auc_pr_final.mean()),
                                  daily_auc_pr_mean_model_B=float(j.auc_pr_model_B.mean()),
                                  share_days_final_higher=float((j.auc_pr_final > j.auc_pr_model_B).mean()))
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
