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

Directional spread features (DIR), relative to the nearest recent-fire
pixel q of a sample pixel p, using only the issue day D:
  wind_speed_q   BARRA-C2 daily mean 10 m wind speed at q on D (from uas, vas)
  wind_align     cosine between that wind and the direction q -> p (0 at q)
  wind_push      wind_speed_q * wind_align
  gust_push      BARRA-C2 daily maximum wind at q on D * wind_align
  upslope        (elevation p - elevation q) / distance, m per px (ETOPO1)
  slope_p        slope at p
  aspect_align   cosine between the aspect at p and the direction q -> p
and, as an upper bound only (future weather), FUT: wind_align and wind_push
from the mean wind of D+1..D+3. Models CW = C + DIR, SW = S + DIR,
SWF = S + DIR + FUT (perfect-forecast wind) and SFW = S + FRP + DIR.
If PM_STORE (barra_c2_pm_wind.zarr) exists, PM adds the afternoon wind of D
(05 UTC, about 15:00 AEST) at q: speed, alignment and push; model SPM uses
the terrain features of DIR with PM instead of the daily wind, and is
compared with S and with SW (daily-mean wind).
EXTRA_VEG=1 (default) adds NDVI (MODIS 500 m) and LAI (HiQ-LAI 5 km) at p on
D from the daily cube: models SN = S + NDVI and SLAI = S + LAI. SMOLDER
already has LAI, so SLAI - S shows the noise level for this comparison.
If AGE_STORE (fire_age_continental.zarr) exists, fuel_age_p = days since the
pixel last burned (read at D-3; no fire since 2015 = elapsed days) is added:
CA = C + age, SA = S + age, SLA = SLAI + age, SNA = SN + age, so the pairs
show what fuel age adds beyond the fire maps, SMOLDER, LAI and NDVI.

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
from scipy.ndimage import map_coordinates
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
DIR = ["wind_speed_q", "wind_align", "wind_push", "gust_push", "upslope", "slope_p", "aspect_align"]
FUT = ["f_wind_align", "f_wind_push"]
ALL = BINARY + FRP + DIR + FUT + ["smolder_logit"]
SL = ["smolder_logit"]
MODELS = {"A": ["log_dist"], "C": BINARY, "B": BINARY + FRP, "CW": BINARY + DIR,
          "S": BINARY + SL, "SF": BINARY + FRP + SL, "SW": BINARY + DIR + SL,
          "SWF": BINARY + DIR + FUT + SL, "SFW": BINARY + FRP + DIR + SL}
PAIRS = [("C", "B"), ("S", "SF"), ("C", "CW"), ("S", "SW"), ("S", "SWF"), ("SF", "SFW")]
PM_STORE = os.environ.get("PM_STORE", "/home/saturn/gwgi/gwgi107h/wildfire_data/firecastnet/barra_c2_pm_wind.zarr")
PM = ["pm_speed_q", "pm_wind_align", "pm_wind_push"]
TERRAIN = ["upslope", "slope_p", "aspect_align"]
EXTRA_VEG = os.environ.get("EXTRA_VEG", "1") == "1"
if EXTRA_VEG:
    ALL = ALL[:-1] + ["ndvi_p", "lai_p"] + ["smolder_logit"]
    MODELS["SN"] = BINARY + ["ndvi_p"] + SL
    MODELS["SLAI"] = BINARY + ["lai_p"] + SL
    PAIRS += [("S", "SN"), ("S", "SLAI"), ("SLAI", "SN")]
AGE_STORE = os.environ.get("AGE_STORE", "/home/saturn/gwgi/gwgi107h/wildfire_data/firecastnet/fire_age_continental.zarr")
if os.path.exists(AGE_STORE):
    ALL = ALL[:-1] + ["fuel_age_p"] + ["smolder_logit"]
    MODELS["CA"] = BINARY + ["fuel_age_p"]
    MODELS["SA"] = BINARY + ["fuel_age_p"] + SL
    if EXTRA_VEG:
        MODELS["SLA"] = BINARY + ["lai_p", "fuel_age_p"] + SL
        MODELS["SNA"] = BINARY + ["ndvi_p", "fuel_age_p"] + SL
        PAIRS += [("SLAI", "SLA"), ("SN", "SNA")]
    PAIRS += [("C", "CA"), ("S", "SA")]
if os.path.exists(PM_STORE):
    ALL = ALL[:-1] + PM + ["smolder_logit"]
    MODELS["SPM"] = BINARY + TERRAIN + PM + SL
    PAIRS += [("S", "SPM"), ("SW", "SPM")]
AUX = os.environ.get("AUX", "/home/saturn/gwgi/gwgi107h/wildfire_data/firecastnet/aux_rasters")
BARRA = os.environ.get("BARRA_STORE", "/home/saturn/gwgi/gwgi107h/wildfire_data/firecastnet/barra_c2_daily.zarr")
EIGHT = np.ones((3, 3), bool)

G = {}


def _init():
    g = open_zarr_root(daily_cube(EVAL_YEAR))
    G["y"] = g["y_fire_3d"]
    G["X"] = g["X"]
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
    for k in ("elevation", "slope", "aspect_sin", "aspect_cos"):
        G[k] = np.load(f"{AUX}/static_{k}.npy", mmap_mode="r")
    G["barra"] = open_zarr_root(BARRA)
    if os.path.exists(PM_STORE):
        G["pm"] = open_zarr_root(PM_STORE)
    if os.path.exists(AGE_STORE):
        G["age"] = open_zarr_root(AGE_STORE)["age_px"]
    blat, blon = np.asarray(G["barra"]["lat"]), np.asarray(G["barra"]["lon"])
    G["barra_ij"] = lambda r, c: ((T.f + (r + 0.5) * T.e - blat[0]) / (blat[1] - blat[0]),
                                  (T.c + (c + 0.5) * T.a - blon[0]) / (blon[1] - blon[0]))
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
    # directional spread features relative to the nearest recent-fire pixel q
    qr, qc = ir[rr, cc], ic[rr, cc]
    east, north = (cc - qc).astype(np.float64), -(rr - qr).astype(np.float64)
    dpx = np.hypot(east, north)
    ue, un = np.where(dpx > 0, east / np.maximum(dpx, 1e-9), 0.0), np.where(dpx > 0, north / np.maximum(dpx, 1e-9), 0.0)
    fi, fj = G["barra_ij"](qr, qc)
    bz = G["barra"]

    def at_q(var, days):
        return np.mean([map_coordinates(np.asarray(bz[var][g], np.float32), [fi, fj], order=1, mode="nearest")
                        for g in days], 0)
    u, v = at_q("uas", [gD]), at_q("vas", [gD])
    ws = np.hypot(u, v)
    align = (u * ue + v * un) / np.maximum(ws, 1e-6)
    f["wind_speed_q"], f["wind_align"], f["wind_push"] = ws, align, ws * align
    f["gust_push"] = at_q("sfcWindmax", [gD]) * align
    elev = G["elevation"]
    f["upslope"] = (elev[rr, cc] - elev[qr, qc]) / np.maximum(dpx, 1.0)
    f["slope_p"] = np.asarray(G["slope"][rr, cc], np.float64)
    f["aspect_align"] = G["aspect_sin"][rr, cc] * ue + G["aspect_cos"][rr, cc] * un
    fu, fv = at_q("uas", [gD + 1, gD + 2, gD + 3]), at_q("vas", [gD + 1, gD + 2, gD + 3])   # future: upper bound only
    fw = np.hypot(fu, fv)
    f["f_wind_align"] = (fu * ue + fv * un) / np.maximum(fw, 1e-6)
    f["f_wind_push"] = fu * ue + fv * un
    if EXTRA_VEG:                                                   # NDVI / LAI at p on the issue day
        xv = np.asarray(G["X"][D, :, :, 5:7], np.float32)
        f["ndvi_p"], f["lai_p"] = xv[rr, cc, 0], xv[rr, cc, 1]
    if "age" in G:                                                  # days since p last burned, up to D
        a_ = np.asarray(G["age"][gD - 3][rr, cc], np.float64)
        f["fuel_age_p"] = np.where(a_ == 65535, gD - 3 + 1, a_)
    if "pm" in G:                                                   # afternoon (05 UTC) wind of D at q
        pu = map_coordinates(np.asarray(G["pm"]["uas_05"][gD], np.float32), [fi, fj], order=1, mode="nearest")
        pv = map_coordinates(np.asarray(G["pm"]["vas_05"][gD], np.float32), [fi, fj], order=1, mode="nearest")
        pw = np.hypot(pu, pv)
        f["pm_speed_q"], f["pm_wind_align"] = pw, (pu * ue + pv * un) / np.maximum(pw, 1e-6)
        f["pm_wind_push"] = pu * ue + pv * un
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
