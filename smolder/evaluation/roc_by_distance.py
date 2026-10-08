"""ROC-AUC and AUC-PR of SMOLDER and persistence by distance to the nearest fire
of days D-2..D, from the STORED national scores (the model is not run).
Persistence is recomputed per day as in adaptive_budget.py.

For every band (0-3, 3-10, 10-25, > 25 km, and > 15 km) two views:
  within      fire and non-fire pixels of the band only: how well each ranking
              orders the land of that band
  vs_all      fire pixels of the band against ALL non-fire pixels: the band's
              share in the pooled ROC-AUC (which compares every fire pixel with
              every non-fire pixel)
plus all pixels together (a check against national_<year>.json, which uses a
subsample of the non-fire pixels).

Computed exactly from histograms over the NB score bins of adaptive_budget.py
(ties within a bin count half for ROC-AUC; for AUC-PR a bin is one threshold
step).

  SMOLDER_DATA=... EVAL_YEAR=2020 SCORES=... OUT=... python -m smolder.evaluation.roc_by_distance
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from smolder.evaluation import adaptive_budget as ab
from smolder.evaluation.evaluate_persistence import persistence_score

OUT = os.environ["OUT"]
WORKERS = int(os.environ.get("WORKERS", 16))
BANDS = {"0-3 km": (-1, 3), "3-10 km": (3, 10), "10-25 km": (10, 25), "> 25 km": (25, np.inf),
         "> 15 km": (15, np.inf), "all": (-1, np.inf)}
G = ab.G


def one_day(args):
    D, row = args
    land = G["land"]
    y = (np.asarray(G["y"][D]) > 0) & land
    recent = (np.asarray(G["y"][D - 3]) > 0) & land
    sb, d = persistence_score(recent, np.random.default_rng([ab.SEED, D]))
    yl, dl = y[land], d[land]
    res = {}
    for name, s in (("persistence", sb[land]), ("SMOLDER", np.asarray(G["scores"][row], np.float32))):
        idx = np.clip(np.searchsorted(ab.EDGES[name], ab.transform(name, s), side="right") - 1, 0, ab.NB - 1)
        res[name] = {}
        for band, (lo, hi) in BANDS.items():
            m = (dl > lo) & (dl <= hi)
            res[name][band] = (np.bincount(idx[m & yl], minlength=ab.NB), np.bincount(idx[m & ~yl], minlength=ab.NB))
    return res


def roc_ap(f, n):
    """ROC-AUC and AUC-PR from fire and non-fire counts per score bin (ascending)."""
    F, N = f.sum(), n.sum()
    if F == 0 or N == 0:
        return None, None
    n_below = np.concatenate([[0], np.cumsum(n)[:-1]])
    roc = float((f * (n_below + 0.5 * n)).sum() / (F * N))
    cf, ca = np.cumsum(f[::-1]), np.cumsum((f + n)[::-1])          # from the highest score down
    fr = f[::-1]
    ap = float((fr / F * np.where(ca > 0, cf / np.maximum(ca, 1), 0)).sum())
    return roc, ap


def decompose(path):
    """The pooled ROC-AUC is the fire-weighted mean of the bands' vs_all ROC-AUC over
    the disjoint bands (0-3, 3-10, 10-25, > 25 km). Adds each band's weight, its term
    and the difference persistence minus SMOLDER to the file (key pooled_decomposition)."""
    d = json.load(open(path))
    disj = ["0-3 km", "3-10 km", "10-25 km", "> 25 km"]
    R = d["results"]
    F = sum(R["SMOLDER"][b]["n_fire"] for b in disj)
    dec = {}
    for b in disj:
        w = R["SMOLDER"][b]["n_fire"] / F
        ts, tp = w * R["SMOLDER"][b]["vs_all_nonfire"]["roc_auc"], w * R["persistence"][b]["vs_all_nonfire"]["roc_auc"]
        dec[b] = dict(weight=w, term_SMOLDER=ts, term_persistence=tp, persistence_minus_SMOLDER=tp - ts)
    dec["sum"] = {k: sum(dec[b][k] for b in disj) for k in ("term_SMOLDER", "term_persistence",
                                                             "persistence_minus_SMOLDER")}
    d["pooled_decomposition"] = dec
    json.dump(d, open(path, "w"), indent=1)
    print(json.dumps(dec, indent=1))


def main():
    days = np.load(ab.SCORES + ".days.npy")
    t0 = time.time()
    with Pool(WORKERS, initializer=ab._init) as pool:
        res = list(pool.imap(one_day, [(int(D), i) for i, D in enumerate(days)], chunksize=2))
    print(f"[roc] {len(res)} days in {time.time() - t0:.0f} s", flush=True)
    out = dict(eval_year=ab.EVAL_YEAR, n_days=len(res), scores=ab.SCORES, bands_km=list(BANDS), results={})
    for name in ("SMOLDER", "persistence"):
        H = {b: (np.sum([r[name][b][0] for r in res], axis=0), np.sum([r[name][b][1] for r in res], axis=0))
             for b in BANDS}
        n_all = H["all"][1]
        out["results"][name] = {}
        for b in BANDS:
            f, n = H[b]
            roc_w, ap_w = roc_ap(f, n)
            roc_a, ap_a = roc_ap(f, n_all)
            out["results"][name][b] = dict(n_fire=int(f.sum()), n_nonfire=int(n.sum()),
                                           base_rate=float(f.sum() / max(f.sum() + n.sum(), 1)),
                                           within=dict(roc_auc=roc_w, auc_pr=ap_w),
                                           vs_all_nonfire=dict(roc_auc=roc_a, auc_pr=ap_a))
    json.dump(out, open(OUT, "w"), indent=1)
    for b in BANDS:
        s_, p_ = out["results"]["SMOLDER"][b], out["results"]["persistence"][b]
        print(f"{b:9s} fire {s_['n_fire']:8d} base {100 * s_['base_rate']:.4f} %  within ROC S {s_['within']['roc_auc']:.4f} "
              f"P {p_['within']['roc_auc']:.4f}  AP S {s_['within']['auc_pr']:.4f} P {p_['within']['auc_pr']:.4f} | "
              f"vs all ROC S {s_['vs_all_nonfire']['roc_auc']:.4f} P {p_['vs_all_nonfire']['roc_auc']:.4f}")
    print("wrote", OUT)


if __name__ == "__main__":
    import sys
    decompose(sys.argv[2]) if "--decompose" in sys.argv else main()
