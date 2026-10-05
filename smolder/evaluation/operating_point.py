"""Which daily alert budget (top k % of land pixels) to pick: fire caught
against false alarms, for several rankings on one year.

Every issue day each ranking flags its top k % of land pixels, for 60 budgets
from 0.02 % to 20 % (log-spaced). Counts are pooled over the days:
  caught      flagged pixels with fire in the target window (TP)
  false alarm flagged pixels without fire (FP)
  recall      TP / all fire, precision TP / flagged, F1 and F2 (recall
              weighted 2x) from them; new-fire recall as in evaluate_national
and split by the distance of the pixel to the nearest fire of days D-2..D
(the persistence history): 0-3, 3-10 and > 10 px (about km), so false alarms
near fire can be compared directly. The best budget of each ranking is the
one with the highest pooled F1 (and F2).

Rankings: persistence (evaluate_persistence, same seed) and any number of
score files written by evaluate_national with SAVE_SCORES. Pixels are taken
in score order; ties are broken by position (scores are continuous, and the
persistence score carries its own random tie-break).

Usage
  SMOLDER_DATA=... EVAL_YEAR=2019 SCORES="released=/p/a.npy,full=/p/b.npy" \\
      OUT=results/experiments/smolder/operating_point_2019 python -m smolder.evaluation.operating_point
CPU only.
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy import ndimage

from smolder.data.io import daily_cube, open_zarr_root
from smolder.evaluation.evaluate_national import DILATE
from smolder.evaluation.evaluate_persistence import persistence_score

EVAL_YEAR = int(os.environ.get("EVAL_YEAR", 2019))
SCORES = [s.split("=", 1) for s in os.environ["SCORES"].split(",")]
OUT = os.environ.get("OUT", f"operating_point_{EVAL_YEAR}")
WORKERS = int(os.environ.get("WORKERS", 16))
SEED = int(os.environ.get("SEED", 0))                     # must match evaluate_persistence.py
KS = np.unique(np.round(np.logspace(np.log10(0.0002), np.log10(0.2), 60), 6))
BAND_EDGES = [3, 10]                                      # px: 0-3 | 3-10 | > 10
BANDS = ["0-3 km", "3-10 km", "> 10 km"]
G = {}


def _init():
    g = open_zarr_root(daily_cube(EVAL_YEAR))
    G["y"] = g["y_fire_3d"]
    G["land"] = np.asarray(g["landmask"][...]) > 0
    G["scores"] = {n: np.load(p, mmap_mode="r") for n, p in SCORES}


def counts(score, yl, nl, band):
    """Cumulative counts at each budget for one ranking on one day."""
    n = score.size
    order = np.argsort(-score, kind="stable")
    cut = np.maximum(1, np.round(KS * n).astype(np.int64))
    ys, ns, bs = yl[order], nl[order], band[order]
    out = dict(tp=np.cumsum(ys)[cut - 1], tp_new=np.cumsum(ns)[cut - 1], flagged=cut)
    for b in range(len(BANDS)):
        inb = bs == b
        out[f"tp_b{b}"] = np.cumsum(ys & inb)[cut - 1]
        out[f"fp_b{b}"] = np.cumsum(~ys & inb)[cut - 1]
    return out


def one_day(args):
    D, row = args
    land = G["land"]
    y = (np.asarray(G["y"][D]) > 0) & land
    recent = (np.asarray(G["y"][D - 3]) > 0) & land
    sb, d = persistence_score(recent, np.random.default_rng([SEED, D]))
    new = y & ~ndimage.binary_dilation(recent, iterations=DILATE)
    yl, nl = y[land], new[land]
    band = np.digitize(d[land], BAND_EDGES, right=True)
    res = dict(fire=int(yl.sum()), new=int(nl.sum()),
               fire_b=np.bincount(band[yl], minlength=len(BANDS)),
               land_b=np.bincount(band, minlength=len(BANDS)),
               rank={"persistence": counts(sb[land], yl, nl, band)})
    for n, _ in SCORES:
        res["rank"][n] = counts(np.asarray(G["scores"][n][row], np.float32), yl, nl, band)
    return res


def summarize(res, name):
    tot = lambda k: np.sum([r["rank"][name][k] for r in res], axis=0).astype(np.float64)
    fire, new = sum(r["fire"] for r in res), sum(r["new"] for r in res)
    fire_b = np.sum([r["fire_b"] for r in res], axis=0)
    tp, flagged = tot("tp"), tot("flagged")
    rec, prec = tp / fire, tp / flagged
    f1 = 2 * prec * rec / (prec + rec)
    f2 = 5 * prec * rec / (4 * prec + rec)
    s = dict(k=KS.tolist(), recall=rec.tolist(), precision=prec.tolist(), f1=f1.tolist(), f2=f2.tolist(),
             fp_per_tp=((flagged - tp) / tp).tolist(), recall_new=(tot("tp_new") / new).tolist(),
             false_alarms=(flagged - tp).tolist())
    for b, bn in enumerate(BANDS):
        s[f"recall {bn}"] = (tot(f"tp_b{b}") / max(fire_b[b], 1)).tolist()
        s[f"false_alarms {bn}"] = tot(f"fp_b{b}").tolist()
    for crit, arr in (("f1", f1), ("f2", f2)):
        i = int(np.nanargmax(arr))
        s[f"best_{crit}"] = dict(k=float(KS[i]), recall=float(rec[i]), precision=float(prec[i]),
                                 f1=float(f1[i]), f2=float(f2[i]), fp_per_tp=float((flagged[i] - tp[i]) / tp[i]),
                                 recall_new=float(s["recall_new"][i]),
                                 **{f"recall {bn}": float(s[f"recall {bn}"][i]) for bn in BANDS},
                                 **{f"false_alarm_share {bn}": float(s[f"false_alarms {bn}"][i] / (flagged[i] - tp[i]))
                                    for bn in BANDS})
    return s


def main():
    days = None
    for n, p in SCORES:
        dd = np.load(p + ".days.npy")
        assert days is None or np.array_equal(dd, days), f"{n}: other days"
        days = dd
    if os.environ.get("N_DAYS"):                          # quick test on the first days
        days = days[:int(os.environ["N_DAYS"])]
    t0 = time.time()
    with Pool(WORKERS, initializer=_init) as pool:
        res = list(pool.imap(one_day, [(int(D), i) for i, D in enumerate(days)], chunksize=2))
    print(f"[op] {len(days)} days in {time.time() - t0:.0f} s", flush=True)
    names = ["persistence"] + [n for n, _ in SCORES]
    fire_b = np.sum([r["fire_b"] for r in res], axis=0)
    land_b = np.sum([r["land_b"] for r in res], axis=0)
    out = dict(eval_year=EVAL_YEAR, n_days=int(len(days)), scores=dict(SCORES), seed=SEED,
               budget="top k % of land pixels per issue day, counts pooled over days",
               bands=dict(edges_px=BAND_EDGES, names=BANDS, fire_share=(fire_b / fire_b.sum()).tolist(),
                          land_share=(land_b / land_b.sum()).tolist()),
               rankings={n: summarize(res, n) for n in names})
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    json.dump(out, open(OUT + ".json", "w"), indent=1)

    R = out["rankings"]
    L = [f"Operating points {EVAL_YEAR} ({len(days)} days; budget = top k % of land pixels per day)", "",
         "| ranking | criterion | budget | recall | precision | false alarms per fire px | new-fire recall | "
         "recall 0-3 km | recall 3-10 km | recall > 10 km | false alarms 0-3 km | 3-10 km | > 10 km |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        for crit in ("f1", "f2"):
            b = R[n][f"best_{crit}"]
            L.append(f"| {n} | best {crit.upper()} | {100 * b['k']:.3g} % | {b['recall']:.3f} | {b['precision']:.4f} | "
                     f"{b['fp_per_tp']:.1f} | {b['recall_new']:.3f} | "
                     + " | ".join(f"{b[f'recall {bn}']:.3f}" for bn in BANDS) + " | "
                     + " | ".join(f"{100 * b[f'false_alarm_share {bn}']:.0f} %" for bn in BANDS) + " |")
    L += ["", "Recall and false alarms per fire pixel caught at selected budgets:", "",
          "| budget | " + " | ".join(f"{n} recall | {n} FA/TP" for n in names) + " |",
          "|---|" + "---|---|" * len(names)]
    for kk in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1):
        i = int(np.argmin(np.abs(KS - kk)))
        L.append(f"| {100 * KS[i]:.3g} % | " + " | ".join(
            f"{R[n]['recall'][i]:.3f} | {R[n]['fp_per_tp'][i]:.1f}" for n in names) + " |")
    open(OUT + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
