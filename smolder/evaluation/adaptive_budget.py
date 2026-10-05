"""Fixed against adaptive daily alert budgets, for SMOLDER and persistence.

  fixed     every issue day the top k % of land pixels is flagged
            (as in operating_point)
  adaptive  every pixel above one score threshold is flagged, the same
            threshold on every day, so the flagged area follows the day's
            fire danger as the ranking sees it: for SMOLDER its score, for
            persistence the distance to the nearest fire of days D-2..D
            (everything within r km). The threshold is set so that the
            flagged area AVERAGED over the year equals the fixed budget,
            so both rules flag the same total area.

Counts pooled over the days: fire caught (recall), precision, false alarms per
fire pixel caught, F1/F2, new-fire recall, recall and false alarms by
distance to the nearest fire of D-2..D (0-3, 3-10, > 10 px); for the adaptive
rule also the spread of the daily flagged area and its rank correlation with
the day's amount of fire. The best adaptive threshold by F1 and F2 is
reported too, and the whole threshold sweep (adaptive_curve) for plotting.

Scores are binned per day (NB bins: logit of the SMOLDER score,
-log(1 + distance) for persistence), so the adaptive thresholds sit on bin
edges and the realised mean area is reported next to the target.

Usage
  SMOLDER_DATA=... EVAL_YEAR=2019 SCORES=/p/full_model_seed123_2019.npy \\
      OUT=results/experiments/v2/adaptive_budget_2019 python -m smolder.evaluation.adaptive_budget
CPU only.
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy import ndimage
from scipy.stats import spearmanr

from smolder.data.io import daily_cube, open_zarr_root
from smolder.evaluation.evaluate_national import DILATE
from smolder.evaluation.evaluate_persistence import persistence_score

EVAL_YEAR = int(os.environ.get("EVAL_YEAR", 2019))
SCORES = os.environ["SCORES"]
OUT = os.environ.get("OUT", f"adaptive_budget_{EVAL_YEAR}")
WORKERS = int(os.environ.get("WORKERS", 16))
SEED = int(os.environ.get("SEED", 0))                     # must match evaluate_persistence.py
BUDGETS = [0.0006, 0.0013, 0.0023, 0.005, 0.01]
NB = 8000
EDGES = {"SMOLDER": np.linspace(-17.0, 17.0, NB + 1), "persistence": np.linspace(-9.0, 0.5, NB + 1)}
BAND_EDGES = [3, 10]
BANDS = ["0-3 km", "3-10 km", "> 10 km"]
G = {}


def _init():
    g = open_zarr_root(daily_cube(EVAL_YEAR))
    G["y"] = g["y_fire_3d"]
    G["land"] = np.asarray(g["landmask"][...]) > 0
    G["scores"] = np.load(SCORES, mmap_mode="r")


def transform(name, s):
    if name == "persistence":                       # s = -(distance + jitter)
        return -np.log1p(-s.astype(np.float64))
    s = np.clip(s.astype(np.float64), 1e-7, 1 - 1e-7)
    return np.log(s / (1 - s))


def fixed_counts(s, yl, nl, band):
    n = s.size
    order = np.argsort(-s, kind="stable")
    ys, ns, bs = yl[order], nl[order], band[order]
    out = {}
    for k in BUDGETS:
        c = max(1, int(round(k * n)))
        r = dict(flagged=c, tp=int(ys[:c].sum()), tp_new=int(ns[:c].sum()))
        for b in range(len(BANDS)):
            inb = bs[:c] == b
            r[f"tp_b{b}"] = int((ys[:c] & inb).sum())
            r[f"fp_b{b}"] = int((~ys[:c] & inb).sum())
        out[k] = r
    return out


def hists(name, s, yl, nl, band):
    idx = np.clip(np.searchsorted(EDGES[name], transform(name, s), side="right") - 1, 0, NB - 1)
    h = dict(n=np.bincount(idx, minlength=NB), fire=np.bincount(idx[yl], minlength=NB),
             new=np.bincount(idx[nl], minlength=NB))
    for b in range(len(BANDS)):
        inb = band == b
        h[f"fire_b{b}"] = np.bincount(idx[yl & inb], minlength=NB)
        h[f"land_b{b}"] = np.bincount(idx[inb], minlength=NB)
    return h


def one_day(args):
    D, row = args
    land = G["land"]
    y = (np.asarray(G["y"][D]) > 0) & land
    recent = (np.asarray(G["y"][D - 3]) > 0) & land
    sb, d = persistence_score(recent, np.random.default_rng([SEED, D]))
    new = y & ~ndimage.binary_dilation(recent, iterations=DILATE)
    yl, nl = y[land], new[land]
    band = np.digitize(d[land], BAND_EDGES, right=True)
    res = dict(fire=int(yl.sum()), new=int(nl.sum()), n=int(yl.size),
               fire_b=np.bincount(band[yl], minlength=len(BANDS)), fixed={}, hist={})
    for name, s in (("persistence", sb[land]), ("SMOLDER", np.asarray(G["scores"][row], np.float32))):
        res["fixed"][name] = fixed_counts(s, yl, nl, band)
        res["hist"][name] = hists(name, s, yl, nl, band)
    return res


def metrics(flagged, tp, tp_new, fire, new, tp_b, fp_b, fire_b):
    rec, prec = tp / fire, tp / max(flagged, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    f2 = 5 * prec * rec / max(4 * prec + rec, 1e-12)
    m = dict(recall=rec, precision=prec, fp_per_tp=(flagged - tp) / max(tp, 1), f1=f1, f2=f2,
             recall_new=tp_new / new)
    for b, bn in enumerate(BANDS):
        m[f"recall {bn}"] = tp_b[b] / max(fire_b[b], 1)
        m[f"false_alarms_per_day {bn}"] = fp_b[b]
    return m


def main():
    days = np.load(SCORES + ".days.npy")
    if os.environ.get("N_DAYS"):
        days = days[:int(os.environ["N_DAYS"])]
    t0 = time.time()
    with Pool(WORKERS, initializer=_init) as pool:
        res = list(pool.imap(one_day, [(int(D), i) for i, D in enumerate(days)], chunksize=2))
    print(f"[adaptive] {len(days)} days in {time.time() - t0:.0f} s", flush=True)
    nd = len(res)
    fire, new = sum(r["fire"] for r in res), sum(r["new"] for r in res)
    fire_b = np.sum([r["fire_b"] for r in res], axis=0)
    n_tot = sum(r["n"] for r in res)
    daily_fire = np.array([r["fire"] for r in res])
    out = dict(eval_year=EVAL_YEAR, n_days=nd, scores=SCORES, budgets=BUDGETS, bins=NB,
               fixed={}, adaptive={}, adaptive_best={})
    for name in ("persistence", "SMOLDER"):
        out["fixed"][name], out["adaptive"][name] = {}, {}
        for k in BUDGETS:
            t = lambda key: sum(r["fixed"][name][k][key] for r in res)
            out["fixed"][name][str(k)] = dict(mean_share=k, **metrics(
                t("flagged"), t("tp"), t("tp_new"), fire, new,
                [t(f"tp_b{b}") for b in range(3)], [t(f"fp_b{b}") / nd for b in range(3)], fire_b))
        H = {key: np.sum([r["hist"][name][key] for r in res], axis=0) for key in res[0]["hist"][name]}
        C = {key: np.cumsum(v[::-1])[::-1] for key, v in H.items()}          # flagged at threshold = bin i
        daily_n = np.cumsum(np.array([r["hist"][name]["n"] for r in res])[:, ::-1], axis=1)[:, ::-1]
        share = C["n"] / n_tot

        def at(i):
            tp_b = [C[f"fire_b{b}"][i] for b in range(3)]
            fp_b = [(C[f"land_b{b}"][i] - C[f"fire_b{b}"][i]) / nd for b in range(3)]
            m = metrics(C["n"][i], C["fire"][i], C["new"][i], fire, new, tp_b, fp_b, fire_b)
            ds = daily_n[:, i] / np.array([r["n"] for r in res])
            m.update(mean_share=float(share[i]), threshold=float(EDGES[name][i]),
                     daily_share_p5_p50_p95_max=[float(np.percentile(ds, q)) for q in (5, 50, 95)] + [float(ds.max())],
                     daily_share_vs_daily_fire_spearman=float(spearmanr(ds, daily_fire)[0]))
            return m
        for k in BUDGETS:
            i = int(np.argmin(np.abs(share - k)))
            out["adaptive"][name][str(k)] = at(i)
        # whole threshold sweep (mean flagged area 0.01 % to 20 %), for plotting
        sel = np.nonzero((share >= 1e-4) & (share <= 0.2))[0]
        sel = sel[np.unique(np.round(np.linspace(0, len(sel) - 1, min(400, len(sel)))).astype(int))]
        out.setdefault("adaptive_curve", {})[name] = dict(
            mean_share=share[sel].tolist(), recall=(C["fire"][sel] / fire).tolist(),
            fp_per_tp=((C["n"][sel] - C["fire"][sel]) / np.maximum(C["fire"][sel], 1)).tolist(),
            **{f"recall {bn}": (C[f"fire_b{b}"][sel] / max(fire_b[b], 1)).tolist() for b, bn in enumerate(BANDS)})
        tp_all, fl_all = C["fire"].astype(float), np.maximum(C["n"], 1).astype(float)
        rec, prec = tp_all / fire, tp_all / fl_all
        ok = share > 1e-5
        for crit, beta in (("f1", 1.0), ("f2", 2.0)):
            f = np.where(ok, (1 + beta ** 2) * prec * rec / np.maximum(beta ** 2 * prec + rec, 1e-12), 0)
            out["adaptive_best"].setdefault(name, {})[crit] = at(int(np.argmax(f)))
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    json.dump(out, open(OUT + ".json", "w"), indent=1, default=float)

    L = [f"Fixed vs adaptive daily budget, {EVAL_YEAR} ({nd} days). Adaptive: one score threshold for all days, "
         "set so the mean flagged area equals the fixed budget.", "",
         "| ranking | mean budget | rule | fire caught | precision | false alarms per hit | F2 | new-fire caught | "
         "caught 0-3 km | caught 3-10 km | caught > 10 km | daily area p5 / p50 / p95 / max |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("persistence", "SMOLDER"):
        for k in BUDGETS:
            for rule in ("fixed", "adaptive"):
                m = out[rule][name][str(k)]
                spread = (" / ".join(f"{100 * v:.3g} %" for v in m["daily_share_p5_p50_p95_max"])
                          if rule == "adaptive" else "constant")
                L.append(f"| {name} | {100 * m['mean_share']:.3g} % | {rule} | {m['recall']:.3f} | {m['precision']:.3f} | "
                         f"{m['fp_per_tp']:.1f} | {m['f2']:.3f} | {m['recall_new']:.3f} | "
                         + " | ".join(f"{m[f'recall {bn}']:.3f}" for bn in BANDS) + f" | {spread} |")
    L += ["", "Best adaptive threshold:", "", "| ranking | criterion | mean budget | fire caught | precision | "
          "false alarms per hit | daily area p5 / p50 / p95 / max | rank corr. daily area vs daily fire |",
          "|---|---|---|---|---|---|---|---|"]
    for name in ("persistence", "SMOLDER"):
        for crit in ("f1", "f2"):
            m = out["adaptive_best"][name][crit]
            L.append(f"| {name} | {crit.upper()} | {100 * m['mean_share']:.3g} % | {m['recall']:.3f} | {m['precision']:.3f} | "
                     f"{m['fp_per_tp']:.1f} | " + " / ".join(f"{100 * v:.3g} %" for v in m["daily_share_p5_p50_p95_max"])
                     + f" | {m['daily_share_vs_daily_fire_spearman']:.2f} |")
    open(OUT + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
