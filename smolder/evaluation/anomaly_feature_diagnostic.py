"""Do anomaly and trend features carry signal for new fire? (2019, no model)

1. Climatology, 2015-2018 only: per pixel, mean and standard deviation of
   soil moisture, LAI, VPD and LST for every 8th day of year (anchors at day
   1, 9, ..., 361), each over all days within +-15 days of the anchor in the
   four years (circular in day of year). Values for other days are linearly
   interpolated between anchors. Written to climatology_2015_2018.zarr
   (float16, <var>_mean / <var>_std, shape (46, H, W)) for later use.
2. Sample (2019 issue days of evaluate_national.py): every fire pixel of the
   target y_fire_3d[D] (fire on D+1..D+3) and N_NEG random land pixel-days
   without fire. ROC-AUC does not depend on the class ratio, so this
   case-control sample estimates the same ROC-AUC as random sampling with
   far fewer negatives.
3. Features at issue day D, each with its raw variable for comparison:
     z_sm, z_lai, z_vpd, z_lst   standardized anomaly vs the climatology
     sm_slope_144d, lai_slope_144d   OLS slope over days D-143..D
     vpd_change_14d              OLS slope over D-13..D times 13 days
     ppt_pct_144d                sum over D-143..D as a percentile of the
                                 pixel's own 2015-2018 144-day sums ending
                                 within +-15 days of the same day of year
   All inputs end on day D: no value after the issue day is used.
4. ROC-AUC per feature, (a) all pixels and (b) pixels more than 10 km from
   any fire of the last 32 days (fire_dist30_continental.zarr at D-3), both
   overall and per land cover. Reported raw (direction kept) and oriented,
   max(AUC, 1 - AUC). A feature is marked if its oriented ROC-AUC in (b) is
   >= 0.60 and >= 0.03 above that of its raw variable.
5. Median trajectories of the inputs before fire, for pixels > 10 km from
   fire of the last 32 days (burned vs not burned, by land cover), in the
   format of explain_2020_summary.json, for figures/make_explain_figures.py.
   Unlike the model's slow bins, these 8-day bins end on the issue day.

Usage (a whole node: reads the 2015-2019 daily cubes once)
  SMOLDER_DATA=... python -m smolder.evaluation.anomaly_feature_diagnostic
"""
import datetime as dt
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import zarr
from numcodecs import Blosc
from sklearn.metrics import roc_auc_score

from smolder.data.io import daily_cube, open_zarr_root, resolve
from smolder.evaluation.evaluate_national import OFFSETS, valid_days

EVAL_YEAR = 2019
CLIM_YEARS = (2015, 2016, 2017, 2018)
READ_YEARS = (2015, 2016, 2017, 2018, 2019)
N_CLIM_DAYS = OFFSETS[2019]                     # global days 0..1460 = 2015-2018
CH = {"sm": 0, "vpd": 2, "ppt": 3, "lst": 4, "lai": 6}
CLIM_VARS = ("sm", "lai", "vpd", "lst")
SD_FLOOR = {"sm": 1e-3, "lai": 0.05, "vpd": 0.01, "lst": 1.0}
ANCHORS = np.arange(1, 366, 8)                  # 46 anchors
HALF = 15
TILE = 512
N_NEG = int(os.environ.get("N_NEG", 400_000))
SEED = int(os.environ.get("SEED", 0))
WORKERS = int(os.environ.get("WORKERS", 8))
DIST_STORE = os.environ.get("DIST_STORE", "fire_dist30_continental.zarr")
CLIM_STORE = os.environ.get("CLIM_STORE", "climatology_2015_2018.zarr")
OUT = os.environ.get("OUT", "anomaly_feature_diagnostic_2019.json")
FAR_PX = 10
LANDCOVER = {20: "shrubland", 30: "grassland", 40: "cropland", 112: "closed forest",
             114: "closed forest", 115: "closed forest", 116: "closed forest",
             121: "open forest", 122: "open forest", 124: "open forest", 125: "open forest",
             126: "open forest"}
LANDCOVERS = ["grassland", "shrubland", "open forest", "closed forest", "cropland"]
FEATURES = [  # (feature, raw variable it is compared with)
    ("z_sm", "sm"), ("z_lai", "lai"), ("z_vpd", "vpd"), ("z_lst", "lst"),
    ("sm_slope_144d", "sm"), ("lai_slope_144d", "lai"), ("vpd_change_14d", "vpd"),
    ("ppt_pct_144d", "ppt_144d")]
RAWS = ["sm", "lai", "vpd", "lst", "ppt_144d"]


def doy_of_global(g):
    """Day of year (1..366) of global day indices."""
    base = dt.date(2015, 1, 1)
    return np.array([(base + dt.timedelta(int(x))).timetuple().tm_yday for x in np.atleast_1d(g)])


def circ_dist(a, b):
    a = np.minimum(a, 365); b = np.minimum(b, 365)
    d = np.abs(a - b)
    return np.minimum(d, 365 - d)


DOY_CLIM = doy_of_global(np.arange(N_CLIM_DAYS))


def ols_slope(v):
    """Per-row OLS slope of v (n, T) against 0..T-1, ignoring NaN."""
    t = np.arange(v.shape[1], dtype=np.float64)[None, :]
    m = np.isfinite(v)
    n = m.sum(1)
    tm = np.where(m, t, 0).sum(1) / np.maximum(n, 1)
    vm = np.where(m, v, 0).sum(1) / np.maximum(n, 1)
    dt_ = np.where(m, t - tm[:, None], 0)
    num = (dt_ * np.where(m, v - vm[:, None], 0)).sum(1)
    den = (dt_ ** 2).sum(1)
    out = num / np.where(den > 0, den, np.nan)
    out[n < 0.5 * v.shape[1]] = np.nan
    return out


def interp_anchor(arr, doy):
    """arr (46, n) values at ANCHORS for n samples; doy (n,) -> (n,)."""
    a = np.r_[ANCHORS, 366]
    d = np.minimum(doy, 365).astype(np.float64)
    k0 = np.searchsorted(a, d, side="right") - 1
    f = (d - a[k0]) / (a[k0 + 1] - a[k0])
    k1 = (k0 + 1) % len(ANCHORS)
    i = np.arange(arr.shape[1])
    return arr[k0, i] * (1 - f) + arr[k1, i] * f


def work_tile(args):
    (r0, c0), S = args
    t0 = time.time()
    data = {v: [] for v in CH}
    for y in READ_YEARS:
        X = open_zarr_root(daily_cube(y))["X"]
        blk = np.asarray(X[:, r0:r0 + TILE, c0:c0 + TILE, :], np.float32)
        for v, c in CH.items():
            data[v].append(blk[..., c])
        del blk
    data = {v: np.concatenate(a, 0) for v, a in data.items()}          # (1826, h, w)
    h, w = data["sm"].shape[1:]

    # ---- climatology from 2015-2018 only
    clim = {}
    for v in CLIM_VARS:
        mu = np.empty((len(ANCHORS), h, w), np.float32)
        sd = np.empty_like(mu)
        for k, a in enumerate(ANCHORS):
            sel = np.nonzero(circ_dist(DOY_CLIM, a) <= HALF)[0]
            x = data[v][sel]                                           # sel < N_CLIM_DAYS: 2015-2018 only
            with np.errstate(all="ignore"):
                mu[k] = np.nanmean(x, 0)
                sd[k] = np.nanstd(x, 0)
        clim[v] = (mu, sd)
    if CLIM_STORE:
        root = zarr.open_group(str(resolve(CLIM_STORE)), mode="r+")
        for v, (mu, sd) in clim.items():
            root[f"{v}_mean"][:, r0:r0 + h, c0:c0 + w] = mu.astype(np.float16)
            root[f"{v}_std"][:, r0:r0 + h, c0:c0 + w] = sd.astype(np.float16)

    rr, cc, gD = S["r"] - r0, S["c"] - c0, S["gD"]
    doy = doy_of_global(gD) if gD.size else np.zeros(0, int)
    f = {}
    ser = {}
    for v in CH:
        lag = 144 if v in ("sm", "lai", "ppt") else 14
        idx = gD[:, None] + np.arange(-lag + 1, 1)[None, :]
        ser[v] = data[v][idx, rr[:, None], cc[:, None]].astype(np.float64)   # (n, lag), ends on D
    for v in CLIM_VARS:
        x = data[v][gD, rr, cc].astype(np.float64)
        mu, sd = clim[v]
        m = interp_anchor(mu[:, rr, cc], doy)
        s = np.maximum(interp_anchor(sd[:, rr, cc], doy), SD_FLOOR[v])
        f[v] = x
        f[f"z_{v}"] = (x - m) / s
    f["sm_slope_144d"] = ols_slope(ser["sm"]) * 143
    f["lai_slope_144d"] = ols_slope(ser["lai"]) * 143
    f["vpd_change_14d"] = ols_slope(ser["vpd"]) * 13
    f["ppt_144d"] = np.nansum(ser["ppt"], 1)
    # percentile of the 144-day sum among the pixel's own 2015-2018 sums ending within +-15 days of year
    cs = np.cumsum(data["ppt"][:N_CLIM_DAYS].astype(np.float64), 0)          # (1461, h, w)
    ends = np.arange(143, N_CLIM_DAYS)
    sums = cs[ends]                                                         # (1318, h, w): sum of days e-143..e
    sums[1:] -= cs[ends[1:] - 144]
    del cs
    doy_e = DOY_CLIM[ends]
    pct = np.full(gD.size, np.nan)
    for lo in range(0, gD.size, 20000):
        sl = slice(lo, lo + 20000)
        ref = sums[:, rr[sl], cc[sl]].T                                     # (n, 1318)
        ok = circ_dist(doy_e[None, :], doy[sl][:, None]) <= HALF
        v = f["ppt_144d"][sl][:, None]
        n_ok = ok.sum(1)
        below = ((ref < v - 1e-6) & ok).sum(1) + 0.5 * ((np.abs(ref - v) <= 1e-6) & ok).sum(1)
        pct[sl] = np.where(n_ok > 0, 100.0 * below / np.maximum(n_ok, 1), np.nan)
    f["ppt_pct_144d"] = pct
    # trajectories, bins ending on D (oldest first)
    traj = dict(sm=np.nanmean(ser["sm"].reshape(-1, 18, 8), 2), lai=np.nanmean(ser["lai"].reshape(-1, 18, 8), 2),
                ppt=np.nansum(ser["ppt"].reshape(-1, 18, 8), 2), vpd=ser["vpd"])
    return dict(i=S["i"], f={k: v.astype(np.float32) for k, v in f.items()},
                traj={k: v.astype(np.float32) for k, v in traj.items()}, secs=time.time() - t0)


def sample():
    days, g = valid_days(EVAL_YEAR, 1)
    land = np.asarray(g["landmask"][...]) > 0
    lc = np.asarray(g["landcover"][...])
    dist = open_zarr_root(DIST_STORE)["dist_px"]
    y = g["y_fire_3d"]
    rng = np.random.default_rng(SEED)
    land_idx = np.flatnonzero(land.ravel())
    neg_day = rng.choice(np.asarray(days), N_NEG)
    neg_pix = rng.choice(land_idx, N_NEG)
    rows = []
    for D in days:
        yd = (np.asarray(y[D]) > 0) & land
        dd = np.asarray(dist[OFFSETS[EVAL_YEAR] + D - 3])
        pos = np.flatnonzero(yd.ravel())
        nm = neg_pix[neg_day == D]
        nm = nm[~yd.ravel()[nm]]                                          # negatives must not burn
        for pix, lab in ((pos, 1), (nm, 0)):
            r, c = np.divmod(pix, land.shape[1])
            rows.append(np.stack([np.full(pix.size, D), r, c, np.full(pix.size, lab),
                                  dd[r, c].astype(np.int64), lc[r, c].astype(np.int64)], 1))
    a = np.concatenate(rows)
    S = dict(D=a[:, 0], r=a[:, 1], c=a[:, 2], y=a[:, 3].astype(bool), d32=a[:, 4], lc=a[:, 5])
    S["gD"] = S["D"] + OFFSETS[EVAL_YEAR]
    S["i"] = np.arange(a.shape[0])
    return S, land.shape


def auc_table(S, F, mask):
    out = {}
    for name in [f for f, _ in FEATURES] + RAWS:
        v = F[name][mask]; yy = S["y"][mask]
        ok = np.isfinite(v)
        if yy[ok].all() or (~yy[ok]).all() or yy[ok].sum() < 20:
            out[name] = dict(auc=None, auc_oriented=None, n_fire=int(yy[ok].sum()), n_nofire=int((~yy[ok]).sum()))
            continue
        a = float(roc_auc_score(yy[ok], v[ok]))
        out[name] = dict(auc=a, auc_oriented=max(a, 1 - a), n_fire=int(yy[ok].sum()), n_nofire=int((~yy[ok]).sum()))
    return out


def main():
    t0 = time.time()
    S, (H, W) = sample()
    print(f"[anomaly] {S['y'].sum():,} fire and {(~S['y']).sum():,} non-fire pixel-days of {EVAL_YEAR} "
          f"({time.time() - t0:.0f} s)", flush=True)
    if CLIM_STORE:
        root = zarr.open_group(str(resolve(CLIM_STORE)), mode="w")
        comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        for v in CLIM_VARS:
            for s in ("mean", "std"):
                root.create_dataset(f"{v}_{s}", shape=(len(ANCHORS), H, W), chunks=(len(ANCHORS), TILE, TILE),
                                    dtype=np.float16, fill_value=np.nan, compressor=comp)
        root.attrs.update(years=list(CLIM_YEARS), anchors_doy=ANCHORS.tolist(), half_window_days=HALF,
                          variables=list(CLIM_VARS), cube_channel=CH,
                          description="per-pixel climatology from 2015-2018 only; interpolate linearly "
                                      "between anchors in day of year")
    land = np.asarray(open_zarr_root(daily_cube(EVAL_YEAR))["landmask"][...]) > 0
    land_tiles = [(r0, c0) for r0 in range(0, H, TILE) for c0 in range(0, W, TILE)
                  if land[r0:r0 + TILE, c0:c0 + TILE].any()]
    tid = (S["r"] // TILE) * 100 + S["c"] // TILE
    jobs = []
    for (r0, c0) in land_tiles:
        m = tid == (r0 // TILE) * 100 + c0 // TILE
        jobs.append(((r0, c0), {k: v[m] for k, v in S.items()}))
    print(f"[anomaly] {len(jobs)} tiles with land", flush=True)
    n = S["y"].size
    F = {k: np.full(n, np.nan, np.float32) for k in [f for f, _ in FEATURES] + RAWS}
    T = dict(sm=np.full((n, 18), np.nan, np.float32), lai=np.full((n, 18), np.nan, np.float32),
             ppt=np.full((n, 18), np.nan, np.float32), vpd=np.full((n, 14), np.nan, np.float32))
    with Pool(WORKERS) as pool:
        for k, res in enumerate(pool.imap_unordered(work_tile, jobs), 1):
            for name in F:
                F[name][res["i"]] = res["f"][name]
            for name in T:
                T[name][res["i"]] = res["traj"][name]
            print(f"  tile {k}/{len(jobs)} ({res['secs']:.0f} s), elapsed {time.time() - t0:.0f} s", flush=True)

    far = S["d32"] > FAR_PX
    lcn = np.array([LANDCOVER.get(int(v), "other") for v in S["lc"]])
    out = dict(eval_year=EVAL_YEAR, climatology_years=list(CLIM_YEARS), n_fire=int(S["y"].sum()),
               n_nofire=int((~S["y"]).sum()), sampling="every fire pixel of the 2019 issue days plus "
               f"{N_NEG} random land pixel-days without fire (case-control)", far_threshold_px=FAR_PX,
               history="fire detected on days D-31..D", features=dict(FEATURES), tables={})
    for sel_name, sel in (("a_all", np.ones(n, bool)), ("b_far_10km", far)):
        tab = dict(overall=auc_table(S, F, sel))
        tab["by_landcover"] = {lc: auc_table(S, F, sel & (lcn == lc)) for lc in LANDCOVERS}
        out["tables"][sel_name] = tab
    b = out["tables"]["b_far_10km"]["overall"]
    out["marked"] = [f for f, raw in FEATURES
                     if b[f]["auc_oriented"] is not None and b[raw]["auc_oriented"] is not None
                     and b[f]["auc_oriented"] >= 0.60 and b[f]["auc_oriented"] >= b[raw]["auc_oriented"] + 0.03]
    traj = {}
    for key, n_ in (("sm", 18), ("ppt", 18), ("lai", 18), ("vpd", 14)):
        traj[key] = {}
        for lc in LANDCOVERS:
            for lab, name in ((True, "fire"), (False, "no fire")):
                m = far & (lcn == lc) & (S["y"] == lab)
                if m.sum() < 200:
                    continue
                v = T[key][m]
                traj[key][f"{name}|{lc}"] = dict(q25=np.nanpercentile(v, 25, 0).tolist(),
                                                 q50=np.nanmedian(v, 0).tolist(),
                                                 q75=np.nanpercentile(v, 75, 0).tolist(), n=int(m.sum()))
    out["trajectories_far_10km"] = traj
    out["trajectory_bins"] = "8-day bins ending on the issue day (oldest first); VPD daily D-13..D"
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"wrote {OUT} ({time.time() - t0:.0f} s); marked: {out['marked']}", flush=True)


if __name__ == "__main__":
    main()
