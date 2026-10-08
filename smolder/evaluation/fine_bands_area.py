"""Fire caught by distance in fine bands, and the area needed to catch a given
share of the fire. Reads the stored national scores of a year (SCORES, written
by evaluate_national); the model is not run. Persistence is recomputed per day
exactly as in adaptive_budget.py (same distance, same seed for the jitter).

1. fine_bands  Distance of every land pixel to the nearest fire of days D-2..D
               (VIIRS, the fire-history mask), in bands 0 km (the pixel itself
               burned), up to 1, 1-2, 2-3, 3-5, 5-10, 10-25, > 25 km (1 px is
               about 1 km). For SMOLDER and persistence at the best-F2
               threshold of THRESHOLDS_FROM (2019): the share of all fire in
               each band and the share of it that is caught. The bands 0-3, 3-10
               and > 10 km of adaptive_budget.py are unions of these.
2. area_for_recall  The mean daily share of the land that has to be flagged to
               catch 50, 75, 90 and 95 % of the fire, (a) with one score
               threshold for all days (sweep over the NB bins of
               adaptive_budget.py, down to the whole land) and (b) with the
               same top share of land every day (sweep over 400 shares from
               0.001 % to 100 %). Interpolated linearly in the log of the area
               between the two sweep points around the target.

  SMOLDER_DATA=... EVAL_YEAR=2020 SCORES=... THRESHOLDS_FROM=.../adaptive_budget_2019.json \\
      OUT=.../fine_bands_area_2020.json python -m smolder.evaluation.fine_bands_area
CPU only.
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from smolder.evaluation import adaptive_budget as ab                 # NB, EDGES, SEED, transform, _init
from smolder.evaluation.evaluate_persistence import persistence_score

EVAL_YEAR = ab.EVAL_YEAR
SCORES = ab.SCORES
OUT = os.environ["OUT"]
WORKERS = int(os.environ.get("WORKERS", 16))
REF = json.load(open(os.environ["THRESHOLDS_FROM"]))["adaptive_best"]
THR = {name: REF[name]["f2"]["threshold"] for name in ("SMOLDER", "persistence")}
FINE_EDGES = [0, 1, 2, 3, 5, 10, 25]                  # right-closed: d <= 0, 0 < d <= 1, ...
FINE = ["0 km", "up to 1 km", "1-2 km", "2-3 km", "3-5 km", "5-10 km", "10-25 km", "> 25 km"]
KS = np.unique(np.concatenate([np.logspace(-5, 0, 400), [1.0]]))
TARGETS = [0.5, 0.75, 0.9, 0.95]
G = ab.G


def one_day(args):
    D, row = args
    land = G["land"]
    y = (np.asarray(G["y"][D]) > 0) & land
    recent = (np.asarray(G["y"][D - 3]) > 0) & land
    sb, d = persistence_score(recent, np.random.default_rng([ab.SEED, D]))
    yl = y[land]
    band = np.digitize(d[land], FINE_EDGES, right=True)
    res = dict(n=int(yl.size), fire=int(yl.sum()), fire_band=np.bincount(band[yl], minlength=len(FINE)),
               land_band=np.bincount(band, minlength=len(FINE)))
    for name, s in (("persistence", sb[land]), ("SMOLDER", np.asarray(G["scores"][row], np.float32))):
        t = ab.transform(name, s)
        flag = t >= THR[name]
        res[name] = dict(
            flagged=int(flag.sum()),
            caught_band=np.bincount(band[flag & yl], minlength=len(FINE)),
            hist_n=np.bincount(np.clip(np.searchsorted(ab.EDGES[name], t, side="right") - 1, 0, ab.NB - 1),
                               minlength=ab.NB),
            hist_fire=np.bincount(np.clip(np.searchsorted(ab.EDGES[name], t[yl], side="right") - 1, 0, ab.NB - 1),
                                  minlength=ab.NB))
        order = np.argsort(-s, kind="stable")                       # same tie rule as adaptive_budget's fixed rule
        cum = np.concatenate([[0], np.cumsum(yl[order])])
        res[name]["fixed_caught"] = cum[np.maximum(1, np.round(KS * s.size).astype(int))]
    return res


def area_for(share, recall, target):
    """Smallest area at which recall reaches target, log-linear between sweep points
    (share increasing). None if never reached."""
    i = int(np.searchsorted(recall, target, side="left"))
    if i >= len(recall):
        return None
    if i == 0 or recall[i] == target:
        return float(share[i])
    r0, r1, a0, a1 = recall[i - 1], recall[i], np.log(share[i - 1]), np.log(share[i])
    return float(np.exp(a0 + (target - r0) / (r1 - r0) * (a1 - a0)))


def main():
    days = np.load(SCORES + ".days.npy")
    if os.environ.get("N_DAYS"):                       # quick test on the first days
        days = days[:int(os.environ["N_DAYS"])]
    t0 = time.time()
    with Pool(WORKERS, initializer=ab._init) as pool:
        res = list(pool.imap(one_day, [(int(D), i) for i, D in enumerate(days)], chunksize=2))
    print(f"[fine] {len(res)} days in {time.time() - t0:.0f} s", flush=True)
    nd, n_tot = len(res), sum(r["n"] for r in res)
    fire = sum(r["fire"] for r in res)
    fire_band = np.sum([r["fire_band"] for r in res], axis=0)
    out = dict(eval_year=EVAL_YEAR, n_days=nd, scores=SCORES, thresholds_from=os.environ["THRESHOLDS_FROM"],
               thresholds_f2=THR, bands=FINE, band_edges_px=FINE_EDGES, n_fire=int(fire),
               fine_bands=dict(fire_share=(fire_band / fire).tolist(), n_fire=fire_band.tolist(),
                               land_share=(np.sum([r["land_band"] for r in res], axis=0) / n_tot).tolist()),
               area_for_recall={}, curves={})
    for name in ("SMOLDER", "persistence"):
        caught = np.sum([r[name]["caught_band"] for r in res], axis=0)
        out["fine_bands"][name] = dict(recall=(caught / np.maximum(fire_band, 1)).tolist(),
                                       recall_all=float(caught.sum() / fire),
                                       mean_share_flagged=float(sum(r[name]["flagged"] for r in res) / n_tot))
        # (a) one threshold for all days: cumulative from the highest bin down to the whole land
        hn = np.sum([r[name]["hist_n"] for r in res], axis=0)[::-1]
        hf = np.sum([r[name]["hist_fire"] for r in res], axis=0)[::-1]
        keep = hn > 0
        share_a, rec_a = np.cumsum(hn)[keep] / n_tot, np.cumsum(hf)[keep] / fire
        # (b) the same top share of land every day
        rec_b = np.sum([r[name]["fixed_caught"] for r in res], axis=0) / fire
        out["area_for_recall"][name] = {
            rule: {f"{int(100 * t)} %": area_for(sh, rc, t) for t in TARGETS}
            | dict(max_recall=float(rc[-1]), share_at_max=float(sh[-1]))
            for rule, sh, rc in (("one threshold", share_a, rec_a), ("same area every day", KS, rec_b))}
        sel = np.unique(np.round(np.linspace(0, len(share_a) - 1, min(1000, len(share_a)))).astype(int))
        out["curves"][name] = {"one threshold": dict(mean_share=share_a[sel].tolist(), recall=rec_a[sel].tolist()),
                               "same area every day": dict(mean_share=KS.tolist(), recall=rec_b.tolist())}
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(f"fire share by band: {np.round(100 * fire_band / fire, 1)}")
    for name in ("SMOLDER", "persistence"):
        fb = out["fine_bands"][name]
        print(f"{name}: flagged {100 * fb['mean_share_flagged']:.3f} %, caught {100 * fb['recall_all']:.1f} %, "
              f"by band {np.round(100 * np.array(fb['recall']), 1)}")
        for rule, v in out["area_for_recall"][name].items():
            print(f"   {rule}: " + ", ".join(f"{k} {100 * x:.3f} %" if isinstance(x, float) and k.endswith('%')
                                             else f"{k} {x}" for k, x in v.items()))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
