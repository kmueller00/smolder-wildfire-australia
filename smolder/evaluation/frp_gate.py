"""Does fire intensity add to distance near existing fire? (2019, no model)

A cheap gate before adding VIIRS fire radiative power (FRP) inputs. For every
2019 issue day D, pixels within 25 px (about 25 km) of fire detected on days
D-2..D (y_fire_3d[D-3], the model's newest fire-history map) are scored for
fire on D+1..D+3 (y_fire_3d[D]) by gradient-boosted tree models on
feature sets that separate what the model already sees from what FRP adds:

  A   distance to that fire only: log(1 + d)
  C   A + binary fire information the model already receives or can infer:
      number of the model's three history windows (y_fire_3d[D-3], [D-4],
      [D-5]) with fire at the nearest fire pixel and at the pixel itself,
      log(1 + size) of the connected fire cluster (8-neighbour) and its
      growth log(1 + size on D-2..D) - log(1 + size on D-5..D-3)
  B   C + information only the detections carry, from the FIRMS VIIRS S-NPP
      Collection 2 archive (nominal/high confidence, type 0, UTC date,
      gridded with the label rasters' geotransform, which reproduces the
      binary labels): log(1 + summed FRP), number of detections and share at
      night at the nearest fire pixel (D-2..D), log(1 + cluster FRP)
  S   C + the logit of a SMOLDER national score (SCORES, from
      evaluate_national.py SAVE_SCORES), if given
  SF  S + the FRP information of B

The gains that matter are B - C (FRP over the binary fire maps) and SF - S
(FRP over the model's own forecast).

All inputs end on D. Models are fitted on the odd-numbered issue days and
evaluated on the even-numbered ones. Sample: every target fire pixel in the
25 px zone and NEG_FRAC of the others, weighted back. Reported per distance
band (0-3, 3-10, 10-25 px): ROC-AUC and AUC-PR of every model and the
paired differences B - C and SF - S with 95 % intervals from resampling
evaluation days.

Usage
  SMOLDER_DATA=... python -m smolder.evaluation.frp_gate
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import rasterio
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier

from smolder.data.io import daily_cube, open_zarr_root
from smolder.evaluation.evaluate_national import OFFSETS, ap_auc, valid_days

EVAL_YEAR = 2019
FIRMS = os.environ.get("FIRMS", "/home/saturn/gwgi/gwgi107h/wildfire_data/y_fire/firms_viirs_snpp/"
                                "firms_viirs_snpp_c2_2015_2020.npz")
LABEL_TIF = os.environ.get("LABEL_TIF", "/home/saturn/gwgi/gwgi107h/wildfire_data/y_fire/"
                                        "VIIRS_binary_veg_nominal_high/2019/VIIRS_bin_veg_2019_100.tif")
OUT = os.environ.get("OUT", "frp_gate_2019.json")
ZONE_PX = 25
NEG_FRAC = float(os.environ.get("NEG_FRAC", 0.005))
WORKERS = int(os.environ.get("WORKERS", 12))
N_BOOT = int(os.environ.get("N_BOOT", 200))
SEED = 0
BANDS = [("0-3 km", -1, 3), ("3-10 km", 3, 10), ("10-25 km", 10, 25)]
SCORES = os.environ.get("SCORES", "")
BINARY = ["log_dist", "hist_near", "hist_own", "log_cluster_size", "cluster_growth"]
FRP = ["log_frp", "n_det", "night_share", "log_cluster_frp"]
ALL = BINARY + FRP + ["smolder_logit"]
MODELS = {"A": ["log_dist"], "C": BINARY, "B": BINARY + FRP,
          "S": BINARY + ["smolder_logit"], "SF": BINARY + FRP + ["smolder_logit"]}
PAIRS = [("C", "B"), ("S", "SF")]
EIGHT = np.ones((3, 3), bool)

G = {}


def _init():
    g = open_zarr_root(daily_cube(EVAL_YEAR))
    G["y"] = g["y_fire_3d"]
    G["land"] = np.asarray(g["landmask"][...]) > 0
    z = np.load(FIRMS)
    keep = np.isin(z["conf"], ["n", "h"]) & (z["type"] == 0)
    with rasterio.open(LABEL_TIF) as r:
        T = r.transform
    H, W = G["land"].shape
    row = np.floor((z["lat"][keep].astype(np.float64) - T.f) / T.e).astype(np.int64)
    col = np.floor((z["lon"][keep].astype(np.float64) - T.c) / T.a).astype(np.int64)
    ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    day = (z["utc_minutes"][keep].astype(np.int64) // 1440)[ok]
    G["det"] = dict(day=day, pix=(row * W + col)[ok], frp=z["frp"][keep][ok].astype(np.float64),
                    night=(z["daynight"][keep] == "N")[ok])
    G["order"] = np.argsort(G["det"]["day"], kind="stable")
    if SCORES:
        G["scores"] = np.load(SCORES, mmap_mode="r")
        G["row"] = {int(D): i for i, D in enumerate(np.load(SCORES + ".days.npy"))}
    G["day_sorted"] = G["det"]["day"][G["order"]]


def _dets(g_lo, g_hi):
    """Detections with global day in [g_lo, g_hi]."""
    i0 = np.searchsorted(G["day_sorted"], g_lo, side="left")
    i1 = np.searchsorted(G["day_sorted"], g_hi, side="right")
    idx = G["order"][i0:i1]
    d = G["det"]
    return d["pix"][idx], d["frp"][idx], d["night"][idx]


def one_day(D):
    land = G["land"]
    H, W = land.shape
    gD = OFFSETS[EVAL_YEAR] + D
    y = (np.asarray(G["y"][D]) > 0) & land
    now = (np.asarray(G["y"][D - 3]) > 0) & land                     # fire on D-2..D
    prev = (np.asarray(G["y"][D - 6]) > 0) & land                    # fire on D-5..D-3
    hist = sum(((np.asarray(G["y"][D - L]) > 0) & land).astype(np.float32) for L in (3, 4, 5))
    if not now.any():
        return None
    dist, (ir, ic) = ndimage.distance_transform_edt(~now, return_indices=True)
    zone = land & (dist <= ZONE_PX)
    rng = np.random.default_rng([SEED, D])
    samp = zone & (y | (rng.random(land.shape) < NEG_FRAC))
    rr, cc = np.nonzero(samp)
    near = ir[rr, cc] * W + ic[rr, cc]                               # nearest recent-fire pixel

    pix, frp, night = _dets(gD - 2, gD)
    frp_px = np.bincount(pix, weights=frp, minlength=H * W)
    n_px = np.bincount(pix, minlength=H * W).astype(np.float64)
    night_px = np.bincount(pix, weights=night.astype(np.float64), minlength=H * W)
    lab, _ = ndimage.label(now | prev, structure=EIGHT)
    labf = lab.ravel()
    size_now = np.bincount(labf, weights=now.ravel().astype(np.float64))
    size_prev = np.bincount(labf, weights=prev.ravel().astype(np.float64))
    cl_frp = np.bincount(labf, weights=frp_px * now.ravel())
    L = labf[near]
    f = dict(log_dist=np.log1p(dist[rr, cc]),
             log_frp=np.log1p(frp_px[near]), n_det=n_px[near],
             night_share=np.where(n_px[near] > 0, night_px[near] / np.maximum(n_px[near], 1), np.nan),
             log_cluster_size=np.log1p(size_now[L]), log_cluster_frp=np.log1p(cl_frp[L]),
             cluster_growth=np.log1p(size_now[L]) - np.log1p(size_prev[L]),
             hist_near=hist.ravel()[near], hist_own=hist[rr, cc])
    if SCORES:
        full = np.full(H * W, np.nan, np.float32)
        full[np.flatnonzero(land.ravel())] = G["scores"][G["row"][D]]
        sc = np.clip(full[rr * W + cc].astype(np.float64), 1e-6, 1 - 1e-6)
        f["smolder_logit"] = np.log(sc / (1 - sc))
    else:
        f["smolder_logit"] = np.full(rr.size, np.nan)
    yy = y[rr, cc]
    return dict(D=np.full(rr.size, D, np.int32), y=yy, w=np.where(yy, 1.0, 1.0 / NEG_FRAC),
                d=dist[rr, cc].astype(np.float32), X=np.stack([f[k] for k in ALL], 1).astype(np.float32),
                frp_matched=float((n_px[near] > 0).mean()))


def band_mask(d, lo, hi):
    return (d > lo) & (d <= hi)


def scores(p, y, w, d, D, days):
    """Per band ROC-AUC and AUC-PR on the samples of `days`."""
    m = np.isin(D, days)
    out = {}
    for name, lo, hi in BANDS:
        b = m & band_mask(d, lo, hi)
        ap, auc = ap_auc(p[b], y[b], w[b])
        out[name] = (auc, ap)
    return out


def main():
    t0 = time.time()
    days, _ = valid_days(EVAL_YEAR, 1)
    with Pool(WORKERS, initializer=_init) as pool:
        res = [r for r in pool.imap(one_day, days) if r is not None]
    S = {k: np.concatenate([r[k] for r in res]) for k in ("D", "y", "w", "d", "X")}
    print(f"[frp] {S['y'].sum():,} fire and {(~S['y']).sum():,} sampled other pixels within {ZONE_PX} px "
          f"({time.time() - t0:.0f} s); nearest fire pixel has a FIRMS detection in "
          f"{100 * np.mean([r['frp_matched'] for r in res]):.1f} % of samples", flush=True)
    fit = S["D"] % 2 == 1
    ev_days = np.unique(S["D"][~fit])
    preds = {}
    for name, feats in MODELS.items():
        if "smolder_logit" in feats and not SCORES:
            continue
        cols = [ALL.index(f) for f in feats]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=SEED)
        clf.fit(S["X"][fit][:, cols], S["y"][fit], sample_weight=S["w"][fit])
        preds[name] = clf.predict_proba(S["X"][:, cols])[:, 1]
        print(f"  fitted {name} ({time.time() - t0:.0f} s)", flush=True)
    point = {k: scores(v, S["y"], S["w"], S["d"], S["D"], ev_days) for k, v in preds.items()}
    pairs = [(a, b) for a, b in PAIRS if a in preds and b in preds]
    rng = np.random.default_rng(SEED)
    boot = {(p, n): {"d_auc": [], "d_ap": []} for p in pairs for n, _, _ in BANDS}
    by_day = {D: np.flatnonzero(S["D"] == D) for D in ev_days}
    for _ in range(N_BOOT):
        pick = rng.choice(ev_days, ev_days.size)
        idx = np.concatenate([by_day[D] for D in pick])
        for name, lo, hi in BANDS:
            m = idx[band_mask(S["d"][idx], lo, hi)]
            for p in pairs:
                pa, aa = ap_auc(preds[p[0]][m], S["y"][m], S["w"][m])
                pb, ab = ap_auc(preds[p[1]][m], S["y"][m], S["w"][m])
                boot[(p, name)]["d_auc"].append(ab - aa)
                boot[(p, name)]["d_ap"].append(pb - pa)
    out = dict(eval_year=EVAL_YEAR, zone_px=ZONE_PX, neg_frac=NEG_FRAC, n_boot=N_BOOT,
               scores=os.path.basename(SCORES) if SCORES else None,
               fit_days="odd-numbered issue days", eval_days="even-numbered issue days",
               models=MODELS, bands={})
    for name, lo, hi in BANDS:
        m = np.isin(S["D"], ev_days) & band_mask(S["d"], lo, hi)
        e = dict(n_fire=int(S["y"][m].sum()), base_rate=float(S["y"][m].sum() / S["w"][m].sum()),
                 roc_auc={k: v[name][0] for k, v in point.items()},
                 auc_pr={k: v[name][1] for k, v in point.items()}, gains={})
        for p in pairs:
            g = dict(d_auc=e["roc_auc"][p[1]] - e["roc_auc"][p[0]], d_ap=e["auc_pr"][p[1]] - e["auc_pr"][p[0]])
            g["d_ap_relative"] = g["d_ap"] / e["auc_pr"][p[0]]
            for k in ("d_auc", "d_ap"):
                v = np.asarray(boot[(p, name)][k])
                g[k + "_ci95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            e["gains"][f"{p[1]}-{p[0]}"] = g
        out["bands"][name] = e
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"wrote {OUT} ({time.time() - t0:.0f} s)")
    for name, e in out["bands"].items():
        print(f"{name:9} fire {e['n_fire']:7d} base {e['base_rate']:.5f}")
        print("   ROC-AUC " + "  ".join(f"{k} {v:.3f}" for k, v in e["roc_auc"].items()))
        print("   AUC-PR  " + "  ".join(f"{k} {v:.4f}" for k, v in e["auc_pr"].items()))
        for k, g in e["gains"].items():
            print(f"   {k:5} dAUC {g['d_auc']:+.3f} [{g['d_auc_ci95'][0]:+.3f},{g['d_auc_ci95'][1]:+.3f}]  "
                  f"dAP {g['d_ap']:+.4f} ({100*g['d_ap_relative']:+.0f} %) "
                  f"[{g['d_ap_ci95'][0]:+.4f},{g['d_ap_ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
