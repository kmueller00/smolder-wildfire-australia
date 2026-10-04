"""Where the fire is, relative to recent fire: SMOLDER vs persistence, and a blend.

Reads the per-pixel national scores written by evaluate_national.py with
SAVE_SCORES, and for every issue day D computes the continent-wide distance
of each land pixel to the nearest fire detected on days D-2..D
(y_fire_3d[D-3]), as in evaluate_persistence.py.

  1. Distance bands. Target fire pixels are split by that distance
     (0, 1-3, 3-10, 10-25, > 25 px; 1 px is about 1 km). Per band: share of
     all fire pixels, and the share captured in the national top 0.5 % and
     1 % of land by each ranking (pooled over days).
  2. Rankings compared: SMOLDER, the persistence baseline (identical
     ranking to evaluate_persistence.py, same seed), and optionally (BLEND=1)
     a logistic blend of logit(SMOLDER score) and log(1 + distance in px).
     The blend is fitted on the odd-numbered issue days of the year (a
     stratified pixel sample: every fire pixel and NEG_FRAC of the others,
     weighted back) and evaluated on the even-numbered days only. All three
     rankings are reported on those even days so the comparison is fair.
  3. For each ranking: capture and lift at the national top-k shares, for
     all fire and new fire (> 3 px taxicab from fire of D-2..D, as in
     evaluate_national.py), mean daily AUC-PR and pooled AUC-PR.

Usage
  SMOLDER_DATA=... EVAL_YEAR=2019 SCORES=/path/scores_2019.npy BLEND=1 \\
      OUT=stage0_2019.json python -m smolder.evaluation.fire_distance_diagnostics
CPU only.
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy import ndimage

from smolder.data.io import daily_cube, open_zarr_root
from smolder.evaluation.evaluate_national import DILATE, NEG_FRAC, ap_auc, topk_stats
from smolder.evaluation.evaluate_persistence import persistence_score

EVAL_YEAR = int(os.environ.get("EVAL_YEAR", 2019))
SCORES = os.environ["SCORES"]
OUT = os.environ.get("OUT", f"fire_distance_{EVAL_YEAR}.json")
BLEND = os.environ.get("BLEND", "0") == "1"
WORKERS = int(os.environ.get("WORKERS", 12))
SEED = int(os.environ.get("SEED", 0))           # must match evaluate_persistence.py
KS = [0.001, 0.005, 0.01, 0.05, 0.10]
BAND_EDGES = [0, 3, 10, 25]                     # px; bands 0 | (0,3] | (3,10] | (10,25] | > 25
BAND_NAMES = ["0 km", "1-3 km", "3-10 km", "10-25 km", "> 25 km"]
BAND_KS = [0.005, 0.01]
EPS = 1e-6

G = {}


def _init():
    g = open_zarr_root(daily_cube(EVAL_YEAR))
    G["y"] = g["y_fire_3d"]
    G["land"] = np.asarray(g["landmask"][...]) > 0
    G["scores"] = np.load(SCORES, mmap_mode="r")


def blend_features(s, d):
    s = np.clip(np.asarray(s, np.float64), EPS, 1 - EPS)
    return np.stack([np.log(s / (1 - s)), np.log1p(d)], axis=1)


def rank_metrics(s, yl, nl, band):
    r = {}
    r["auc_pr"], _ = ap_auc(s, yl)
    for k in KS:
        t, _, li = topk_stats(s, yl, k)
        tn, _, ln = topk_stats(s, nl, k) if nl.any() else (np.nan, np.nan, np.nan)
        r[f"tpr_{k:g}"], r[f"lift_{k:g}"], r[f"tpr_new_{k:g}"], r[f"lift_new_{k:g}"] = t, li, tn, ln
    n = s.size
    for k in BAND_KS:
        kk = max(1, int(round(k * n)))
        thr = np.partition(s, n - kk)[n - kk]
        hit = (s >= thr) & yl
        r[f"band_hit_{k:g}"] = np.bincount(band[hit], minlength=len(BAND_NAMES))
    return r


def one_day(args):
    D, row, coef = args
    land = G["land"]
    rng = np.random.default_rng([SEED, D])
    y = (np.asarray(G["y"][D]) > 0) & land
    recent = (np.asarray(G["y"][D - 3]) > 0) & land
    sb, d = persistence_score(recent, rng)
    new = y & ~ndimage.binary_dilation(recent, iterations=DILATE)
    yl, nl, dl = y[land], new[land], d[land]
    sm = np.asarray(G["scores"][row], np.float32)
    band = np.digitize(dl, BAND_EDGES, right=True)
    out = dict(day=D, row=row, n_fire=int(yl.sum()),
               band_fire=np.bincount(band[yl], minlength=len(BAND_NAMES)), rank={})
    if coef is None:
        out["rank"]["smolder"] = rank_metrics(sm, yl, nl, band)
        out["rank"]["persistence"] = rank_metrics(sb[land], yl, nl, band)
        samp = yl | (np.random.default_rng([SEED, D, 1]).random(yl.size) < NEG_FRAC)
        out["sample"] = dict(sm=sm[samp], sb=sb[land][samp], d=dl[samp], y=yl[samp], new=nl[samp],
                             w=np.where(yl[samp], 1.0, 1.0 / NEG_FRAC), row=np.full(int(samp.sum()), row))
    else:
        out["rank"]["blend"] = rank_metrics(blend_features(sm, dl) @ coef, yl, nl, band)
    return out


def summarize(days_out, name, pooled_ap):
    rk = [o["rank"][name] for o in days_out]
    s = dict(daily_auc_pr_mean=float(np.nanmean([r["auc_pr"] for r in rk])), pooled_auc_pr=pooled_ap)
    s["topk"] = [dict(k=k, tpr=float(np.nanmean([r[f"tpr_{k:g}"] for r in rk])),
                      lift=float(np.nanmean([r[f"lift_{k:g}"] for r in rk])),
                      tpr_new=float(np.nanmean([r[f"tpr_new_{k:g}"] for r in rk])),
                      lift_new=float(np.nanmean([r[f"lift_new_{k:g}"] for r in rk]))) for k in KS]
    fire = np.sum([o["band_fire"] for o in days_out], axis=0)
    s["bands"] = {b: dict(fire_share=float(fire[i] / fire.sum()),
                          **{f"captured_{k:g}": float(np.sum([r[f"band_hit_{k:g}"][i] for r in rk]) / max(fire[i], 1))
                             for k in BAND_KS})
                  for i, b in enumerate(BAND_NAMES)}
    return s


def pooled(sample, score, mask):
    return float(ap_auc(score[mask], sample["y"][mask], sample["w"][mask])[0])


def main():
    days = np.load(SCORES + ".days.npy")
    print(f"[diag] {EVAL_YEAR}: {len(days)} days, blend={BLEND}", flush=True)
    t0 = time.time()
    with Pool(WORKERS, initializer=_init) as pool:
        res = list(pool.imap(one_day, [(int(D), i, None) for i, D in enumerate(days)]))
        print(f"  pass 1 done in {time.time() - t0:.0f} s", flush=True)
        S = {k: np.concatenate([o["sample"][k] for o in res]) for k in res[0]["sample"]}
        out = dict(eval_year=EVAL_YEAR, scores=os.path.basename(SCORES), n_days=len(days),
                   band_edges_px=BAND_EDGES, seed=SEED)
        allm = np.ones(S["y"].size, bool)
        out["all_days"] = {n: summarize(res, n, pooled(S, S[c], allm))
                           for n, c in (("smolder", "sm"), ("persistence", "sb"))}
        if BLEND:
            from sklearn.linear_model import LogisticRegression
            fit = S["row"] % 2 == 1
            ev = ~fit
            X = blend_features(S["sm"], S["d"])
            lr = LogisticRegression(C=1e4, max_iter=1000).fit(X[fit], S["y"][fit], sample_weight=S["w"][fit])
            coef = lr.coef_[0].astype(np.float64)
            res2 = list(pool.imap(one_day, [(int(D), i, coef) for i, D in enumerate(days) if i % 2 == 0]))
            even = [o for o in res if o["row"] % 2 == 0]
            for o, o2 in zip(even, res2):
                o["rank"]["blend"] = o2["rank"]["blend"]
            out["blend"] = dict(features=["logit(SMOLDER score)", "log(1 + distance px)"],
                                coef=coef.tolist(), intercept=float(lr.intercept_[0]),
                                fit_days="odd-numbered issue days", eval_days="even-numbered issue days",
                                n_eval_days=len(even))
            out["even_days"] = {n: summarize(even, n, pooled(S, S[c] if c else X @ coef, ev))
                                for n, c in (("smolder", "sm"), ("persistence", "sb"), ("blend", None))}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"wrote {OUT} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
