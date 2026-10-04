"""Skill of a national forecast by distance to past fire.

All land pixels of every issue day D are split into bands by their distance
to the nearest fire detected in the last 32 days, D-31..D (union of
y_fire_3d[D-32..D-3], newfire_definition_sweep.py W=30; read from
fire_dist30_continental.zarr built by smolder.data.build_fire_distance with
WINDOW=30). 1 px is about 1 km. Per band:

  fire_share      share of all target fire pixels (pooled over days)
  roc_auc,
  auc_pr,
  base_rate       pooled over days on a stratified sample: every fire pixel
                  and NEG_FRAC of the others, weighted back (as in
                  evaluate_national.py)
  lift / capture  at 0.5 % and 1 % of the band's pixels, computed exactly
                  per day and averaged over the days with fire in the band

Scores come from evaluate_national.py run with SAVE_SCORES.

Usage
  SMOLDER_DATA=... EVAL_YEAR=2019 SCORES=/path/scores_2019.npy \\
      OUT=results/distance_band_diagnostic_2019.json \\
      python -m smolder.evaluation.distance_band_diagnostic
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from smolder.data.io import daily_cube, open_zarr_root
from smolder.evaluation.evaluate_national import NEG_FRAC, OFFSETS, ap_auc

EVAL_YEAR = int(os.environ.get("EVAL_YEAR", 2019))
SCORES = os.environ["SCORES"]
DIST_STORE = os.environ.get("DIST_STORE", "fire_dist30_continental.zarr")
OUT = os.environ.get("OUT", f"distance_band_diagnostic_{EVAL_YEAR}.json")
WORKERS = int(os.environ.get("WORKERS", 12))
SEED = int(os.environ.get("SEED", 0))
BANDS = [("0-3 km", 0, 3), ("3-10 km", 3, 10), ("> 10 km", 10, np.inf)]   # (lo, hi] in px; 0 included in the first
KS = [0.005, 0.01]

G = {}


def _init():
    g = open_zarr_root(daily_cube(EVAL_YEAR))
    G["y"] = g["y_fire_3d"]
    G["land"] = np.asarray(g["landmask"][...]) > 0
    G["dist"] = open_zarr_root(DIST_STORE)["dist_px"]
    G["scores"] = np.load(SCORES, mmap_mode="r")


def band_of(d):
    b = np.full(d.shape, 2, np.int8)
    b[d <= 10] = 1
    b[d <= 3] = 0
    return b


def one_day(args):
    D, row = args
    land = G["land"]
    y = ((np.asarray(G["y"][D]) > 0) & land)[land]
    d = np.asarray(G["dist"][OFFSETS[EVAL_YEAR] + D - 3])[land].astype(np.float32)
    s = np.asarray(G["scores"][row], np.float32)
    b = band_of(d)
    rng = np.random.default_rng([SEED, D])
    keep = y | (rng.random(y.size) < NEG_FRAC)
    out = dict(day=D, band=[], sample=dict(s=s[keep], y=y[keep], b=b[keep]))
    for i in range(len(BANDS)):
        m = b == i
        sb, yb = s[m], y[m]
        r = dict(n_px=int(m.sum()), n_fire=int(yb.sum()))
        for k in KS:
            if r["n_fire"] == 0:
                r[f"capture_{k:g}"] = np.nan
                continue
            kk = max(1, int(round(k * sb.size)))
            thr = np.partition(sb, sb.size - kk)[sb.size - kk]
            sel = sb >= thr
            r[f"capture_{k:g}"] = float((sel & yb).sum() / yb.sum())
            r[f"lift_{k:g}"] = float(((sel & yb).sum() / sel.sum()) / (yb.sum() / yb.size))
        out["band"].append(r)
    return out


def main():
    days = np.load(SCORES + ".days.npy")
    t0 = time.time()
    with Pool(WORKERS, initializer=_init) as pool:
        res = list(pool.imap(one_day, [(int(D), i) for i, D in enumerate(days)]))
    S = {k: np.concatenate([o["sample"][k] for o in res]) for k in ("s", "y", "b")}
    w = np.where(S["y"], 1.0, 1.0 / NEG_FRAC)
    tot_fire = sum(r["n_fire"] for o in res for r in o["band"])
    out = dict(eval_year=EVAL_YEAR, scores=os.path.basename(SCORES), n_days=len(days),
               history="fire detected on days D-31..D (union of y_fire_3d[D-32..D-3])",
               distance="continent-wide Euclidean distance in px (about 1 km)",
               neg_sample_frac=NEG_FRAC, seed=SEED, bands={})
    for i, (name, lo, hi) in enumerate(BANDS):
        rows = [o["band"][i] for o in res]
        m = S["b"] == i
        auc_pr, roc = ap_auc(S["s"][m], S["y"][m], w[m])
        nf = sum(r["n_fire"] for r in rows)
        e = dict(fire_share=nf / tot_fire, fire_px=nf, land_px_share=sum(r["n_px"] for r in rows) / sum(
            r["n_px"] for o in res for r in o["band"]), roc_auc=roc, auc_pr=auc_pr,
            base_rate=float(S["y"][m].sum() / w[m].sum()),
            n_days_with_fire=int(sum(r["n_fire"] > 0 for r in rows)))
        for k in KS:
            e[f"capture_{k:g}"] = float(np.nanmean([r[f"capture_{k:g}"] for r in rows]))
            e[f"lift_{k:g}"] = float(np.nanmean([r.get(f"lift_{k:g}", np.nan) for r in rows]))
        out["bands"][name] = e
    auc_pr, roc = ap_auc(S["s"], S["y"], w)
    out["all_land"] = dict(roc_auc=roc, auc_pr=auc_pr, base_rate=float(S["y"].sum() / w.sum()))
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"wrote {OUT} ({time.time() - t0:.0f} s)")
    print(f"{'band':10} {'fire%':>6} {'ROC-AUC':>8} {'AUC-PR':>8} {'base':>9} {'lift0.5%':>9} {'lift1%':>8}")
    for name, e in out["bands"].items():
        print(f"{name:10} {100*e['fire_share']:6.1f} {e['roc_auc']:8.3f} {e['auc_pr']:8.4f} {e['base_rate']:9.6f} "
              f"{e['lift_0.005']:9.1f} {e['lift_0.01']:8.1f}")


if __name__ == "__main__":
    main()
