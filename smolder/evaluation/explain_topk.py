"""Why does SMOLDER put a pixel in its top 1 %?

Uses the patch protocol of evaluate.py (fire-active 384 px patches of the 2020
hold-out year, fixed seed) and the released model.

1. Permutation importance of each input group for the top-1 % selection.
   An input group is shuffled across the land pixels of the patch (the same
   pixel permutation at every time step, so each pixel keeps a realistic
   time series but loses its location). Reported per group:
     retention   share of the original top-1 % pixels still in the top 1 %
                 (1 = the group does not matter for the selection)
     ap_drop     relative drop of pooled AUC-PR
   both overall and split by the climate and land-cover class of the pixel.
   Stability: patches are split into N_FOLD temporal folds (consecutive
   blocks of issue days) and N_FOLD spatial folds (west-to-east bands of equal
   patch count); every statistic is also computed per fold and reported as
   mean and standard deviation across folds.

2. Pixel sample for exploratory analysis. Per patch up to PER_CLASS pixels of
   each of four classes, with the inputs (in native units) as they stood on
   the issue day and their 144-day history:
     hit          in the top 1 %, burned in the next 3 days
     false_alarm  in the top 1 %, did not burn
     miss         burned, not in the top 1 %
     background   neither (random land pixels)

Outputs (working directory): explain_<year>.json, explain_<year>_pixels.csv.gz
EVAL_YEAR (default 2020) selects the year; USE_VPD_ANOMALY=1 adds the VPD
anomaly channel (last fast channel) as its own input group.

Usage:
    SMOLDER_DATA=/path/to/cubes python -m smolder.evaluation.explain_topk
"""
import json
import os

import numpy as np
import pandas as pd
import torch

from smolder.data.io import CHANNEL_STATS, daily_cube, open_zarr_root
from smolder.data.zarr_dual_datamodule import (check_checkpoint_inputs, CH, FAST_CHANNELS, SLOW_CHANNELS, DualPatchConfig,
                                               DualWindowDataset)
from smolder.evaluation.evaluate_national import ap_auc
from smolder.models.conv_lstm_lit_dual import ConvLSTMLitDual

CKPT = os.environ.get("CKPT", "checkpoints/smolder_swa.ckpt")
N_PATCH = int(os.environ.get("N_PATCH", 1500))
N_FOLD = 5
PATCH = 384
TOP = 0.01
PER_CLASS = int(os.environ.get("PER_CLASS", 60))
NEG_FRAC = 0.02
SEED = 21
EVAL_YEAR = int(os.environ.get("EVAL_YEAR", 2020))
OFFSETS = {2019: 1461, 2020: 1826}

CLIMATE = {**{k: "tropical" for k in (1, 2, 3)}, **{k: "arid" for k in (4, 5, 6, 7)},
           **{k: "temperate" for k in range(8, 30)}}
LANDCOVER = {20: "shrubland", 30: "grassland", 40: "cropland", 112: "closed forest",
             114: "closed forest", 115: "closed forest", 116: "closed forest",
             121: "open forest", 122: "open forest", 124: "open forest", 125: "open forest",
             126: "open forest"}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)
    check_checkpoint_inputs(CKPT)
    model = ConvLSTMLitDual.load_from_checkpoint(CKPT, map_location=device).eval().to(device)
    cube = daily_cube(EVAL_YEAR)
    times = list(open_zarr_root(cube).attrs["time"])
    ds = DualWindowDataset(DualPatchConfig(
        zarr_paths=(cube,), stats_path="channel_stats_2015_2018.json",
        slow_cube_path="cube_slow_8day.zarr", day_offset=OFFSETS[EVAL_YEAR], patch_size=PATCH,
        samples_per_epoch=N_PATCH * 3, seed=SEED, min_pos_pixels=45, pos_frac=1.0,
        deterministic=True, fire_history=True, fire_history_lags=(3, 4, 5),
        fire_history_distance=True))
    fh0 = ds.fire_hist_start_idx
    sl = {c: i for i, c in enumerate(SLOW_CHANNELS)}
    fa = {c: i for i, c in enumerate(FAST_CHANNELS)}
    # input groups: (name, [(tensor, channel indices)], x_cat channel or None)
    GROUPS = [
        ("leaf area index", [("s", [sl["LAI"]])], None),
        ("soil moisture", [("s", [sl["SM"]])], None),
        ("precipitation", [("s", [sl["PPT"]])], None),
        ("vapour pressure deficit", [("f", [fa["VPD"]])], None),
        ("land surface temperature", [("f", [fa["LST"]])], None),
        ("wind speed", [("f", [fa["WIND"]])], None),
        ("fire history", [("f", list(range(fh0, fh0 + ds.n_fire_hist_channels)))], None),
        ("biomass", [("s", [len(SLOW_CHANNELS)]), ("f", [len(FAST_CHANNELS)])], None),
        ("land cover", [], 0),
        ("climate zone", [], 1),
    ]
    if ds.slow_veg == "lai500":                 # 500 m LAI sits in the LAI position
        GROUPS[0] = ("leaf area index (500 m)", GROUPS[0][1], None)
    if ds.slow_veg == "ndvi":                   # NDVI sits in the LAI position
        GROUPS[0] = ("NDVI", GROUPS[0][1], None)
    elif ds.slow_veg == "lai+ndvi":             # appended as the last slow channel
        GROUPS.insert(1, ("NDVI", [("s", [-1])], None))
    if ds.use_fast_ndvi:
        GROUPS.insert(1, ("NDVI (fast branch)", [("f", [ds.fast_ndvi_idx])], None))
    lay = ds.channel_layout()                  # every optional input as its own group
    if ds.use_vpd_anomaly:
        GROUPS.insert(6, ("VPD anomaly", [("f", lay["fast"]["VPD anomaly"])], None))
    for name, label in (("fire radiative power", "fire radiative power"), ("wind u/v", "wind direction (u, v)"),
                        ("fuel age", "fuel age")):
        if name in lay["fast"]:
            GROUPS.append((label, [("f", lay["fast"][name])], None))
    for name in ("elevation", "slope", "aspect", "lightning"):     # statics: in both branches
        if name in lay["fast"]:
            GROUPS.append((name, [("s", lay["slow"][name]), ("f", lay["fast"][name])], None))
    stats = json.loads(CHANNEL_STATS.read_text())
    mu, sd = np.asarray(stats["x_mean"]), np.asarray(stats["x_std"])
    sd = np.where(sd < 1e-6, 1.0, sd)

    def denorm(z, name):
        return z * sd[CH[name]] + mu[CH[name]]

    def denorm_slow(z, name):
        """Slow-branch values use the dataset's per-bin statistics (8-day sums
        of precipitation are scaled by the bin length)."""
        i = SLOW_CHANNELS.index(name)
        return z * ds.slow_std[i] + ds.slow_mean[i]

    def predict(xs, xf, xc):
        with torch.no_grad():
            return torch.sigmoid(model.forward_seq(xs[None].to(device), xf[None].to(device),
                                                   xc[None].to(device))[:, -1])[0].cpu().numpy()

    def top_mask(p, land):
        v = p[land]
        k = max(1, int(round(TOP * v.size)))
        return land & (p >= np.partition(v, -k)[-k])

    names = [g[0] for g in GROUPS]
    patches = []                                        # per patch: t_end, x centre, counts
    pool = {"base": dict(s=[], y=[], w=[], clim=[], lc=[], pid=[])}
    pool.update({n: dict(s=[]) for n in names})
    rows = []
    kept = 0
    for i in range(N_PATCH * 3):
        if kept >= N_PATCH:
            break
        b = ds[i]
        land = b["mask"].numpy() > 0.5
        y = (b["y"][-1].numpy() > 0) & land
        if y.sum() < 45 or land.mean() < 0.5:
            continue
        kept += 1
        xs, xf, xc = b["x_slow"], b["x_fast"], b["x_cat"]
        lcode, kcode = xc[..., 0].numpy(), xc[..., 1].numpy()
        clim = np.vectorize(lambda k: CLIMATE.get(int(k), "other"))(kcode)
        lcov = np.vectorize(lambda k: LANDCOVER.get(int(k), "other"))(lcode)
        p0 = predict(xs, xf, xc)
        t0 = top_mask(p0, land)

        neg = land & ~y & (rng.random(land.shape) < NEG_FRAC)
        keep = y | neg
        pool["base"]["s"].append(p0[keep]); pool["base"]["y"].append(y[keep])
        pool["base"]["w"].append(np.where(y[keep], 1.0, 1.0 / NEG_FRAC))
        pool["base"]["clim"].append(clim[keep]); pool["base"]["lc"].append(lcov[keep])
        pool["base"]["pid"].append(np.full(int(keep.sum()), kept - 1))
        cnt = {n: {} for n in names}
        patches.append(dict(t_end=int(b["t_end"]), x=int(b["x0"]) + PATCH // 2, cnt=cnt))

        li = np.flatnonzero(land.ravel())
        perm = li.copy()
        rng.shuffle(perm)
        for name, parts, cat in GROUPS:
            xs2, xf2, xc2 = xs.clone(), xf.clone(), xc.clone()
            for which, chans in parts:
                x = xs2 if which == "s" else xf2
                flat = x.reshape(x.shape[0], -1, x.shape[-1])
                for c in chans:
                    flat[:, li, c] = flat[:, perm, c]
            if cat is not None:
                flat = xc2.reshape(-1, xc2.shape[-1])
                flat[li, cat] = flat[perm, cat]
            p1 = predict(xs2, xf2, xc2)
            t1 = top_mask(p1, land)
            pool[name]["s"].append(p1[keep])
            cnt[name]["all"] = (int((t0 & t1).sum()), int(t0.sum()))
            for stratum, lab in (("clim", clim), ("lc", lcov)):
                for g in np.unique(lab[t0]):
                    m = t0 & (lab == g)
                    cnt[name][f"{stratum}:{g}"] = (int((m & t1).sum()), int(m.sum()))

        # pixel sample with native-unit inputs
        date = str(times[int(b["t_end"]) - 1])
        classes = {"hit": t0 & y, "false_alarm": t0 & ~y, "miss": y & ~t0,
                   "background": land & ~y & ~t0}
        slow = xs.numpy(); fast = xf.numpy()
        for cname, m in classes.items():
            yy, xx = np.nonzero(m)
            if yy.size == 0:
                continue
            pick = rng.choice(yy.size, min(PER_CLASS, yy.size), replace=False)
            for j in pick:
                a, c = yy[j], xx[j]
                sm_traj = denorm_slow(slow[:, a, c, sl["SM"]], "SM")
                lai_traj = denorm_slow(slow[:, a, c, sl["LAI"]], "LAI")
                ppt_traj = denorm_slow(slow[:, a, c, sl["PPT"]], "PPT")
                if ds.vpd_source == "barra":           # BARRA-C2 VPD is normalized with its own statistics
                    bm, bs = ds.barra_stats["vpd"]
                    vpd = fast[:, a, c, fa["VPD"]] * bs + bm
                else:
                    vpd = denorm(fast[:, a, c, fa["VPD"]], "VPD")
                d = fast[-1, a, c, fh0 + ds.n_fire_hist_channels - 1]
                rows.append(dict(
                    cls=cname, date=date, patch=kept - 1, climate=clim[a, c], landcover=lcov[a, c],
                    risk=float(p0[a, c]),
                    recent_fire=int(fast[-1, a, c, fh0:fh0 + 3].max() > 0.5),
                    dist_recent_fire_km=float(-5.0 * np.log(d)) if d > 1e-6 else np.nan,
                    soil_moisture=float(sm_traj[-1]), lai=float(lai_traj[-1]),
                    precip_32d=float(ppt_traj[-4:].sum()),
                    vpd_7d=float(vpd[-7:].mean()),
                    lst_7d=float(denorm(fast[-7:, a, c, fa["LST"]], "LST").mean()),
                    wind_7d=float(denorm(fast[-7:, a, c, fa["WIND"]], "WIND").mean()),
                    biomass=float(slow[-1, a, c, len(SLOW_CHANNELS)] * ds.agb_std + ds.agb_mean),
                    **{f"sm_b{k:02d}": float(v) for k, v in enumerate(sm_traj)},
                    **{f"lai_b{k:02d}": float(v) for k, v in enumerate(lai_traj)},
                    **{f"ppt_b{k:02d}": float(v) for k, v in enumerate(ppt_traj)},
                    **{f"vpd_d{k:02d}": float(v) for k, v in enumerate(vpd)},
                ))
        if kept % 50 == 0:
            print(f"  {kept}/{N_PATCH} patches", flush=True)

    base = {k: np.concatenate(v) for k, v in pool["base"].items()}
    perm_s = {n: np.concatenate(pool[n]["s"]) for n in names}
    npat = len(patches)
    t_order = np.argsort([p["t_end"] for p in patches], kind="stable")
    x_order = np.argsort([p["x"] for p in patches], kind="stable")
    folds = {"time": np.empty(npat, int), "space": np.empty(npat, int)}
    for kind, order in (("time", t_order), ("space", x_order)):
        for f, idx in enumerate(np.array_split(order, N_FOLD)):
            folds[kind][idx] = f

    def stats_for(pids):
        pids = set(int(p) for p in pids)
        m = np.isin(base["pid"], list(pids))
        ap0, _ = ap_auc(base["s"][m], base["y"][m], base["w"][m])
        res = {}
        for name in names:
            acc = {}
            for p in pids:
                for key, (k, n) in patches[p]["cnt"][name].items():
                    a = acc.setdefault(key, [0, 0]); a[0] += k; a[1] += n
            ap1, _ = ap_auc(perm_s[name][m], base["y"][m], base["w"][m])
            g = dict(retention=acc["all"][0] / max(acc["all"][1], 1), ap_drop=(ap0 - ap1) / ap0,
                     by_climate={k[5:]: v[0] / v[1] for k, v in acc.items() if k.startswith("clim:") and v[1] >= 100},
                     by_landcover={k[3:]: v[0] / v[1] for k, v in acc.items() if k.startswith("lc:") and v[1] >= 100},
                     ap_drop_by_climate={})
            for c in ("tropical", "arid", "temperate"):
                mc = m & (base["clim"] == c)
                if base["y"][mc].sum() >= 50:
                    a0, _ = ap_auc(base["s"][mc], base["y"][mc], base["w"][mc])
                    a1, _ = ap_auc(perm_s[name][mc], base["y"][mc], base["w"][mc])
                    g["ap_drop_by_climate"][c] = (a0 - a1) / a0
            res[name] = g
        return ap0, res

    ap0, allres = stats_for(range(npat))
    out = dict(checkpoint=os.path.basename(CKPT), n_patches=npat, top_fraction=TOP, n_fold=N_FOLD,
               base_auc_pr=ap0, groups=allres, folds={})
    for kind in ("time", "space"):
        per = [stats_for(np.flatnonzero(folds[kind] == f)) for f in range(N_FOLD)]
        dates = [p["t_end"] for p in patches]
        out["folds"][kind] = dict(
            fold_patches=[int((folds[kind] == f).sum()) for f in range(N_FOLD)],
            fold_range=[[int(min(np.array(dates if kind == "time" else [p["x"] for p in patches])[folds[kind] == f])),
                         int(max(np.array(dates if kind == "time" else [p["x"] for p in patches])[folds[kind] == f]))]
                        for f in range(N_FOLD)],
            base_auc_pr=[a for a, _ in per],
            groups={n: dict(retention=[r[n]["retention"] for _, r in per],
                            ap_drop=[r[n]["ap_drop"] for _, r in per],
                            by_climate=[r[n]["by_climate"] for _, r in per],
                            by_landcover=[r[n]["by_landcover"] for _, r in per])
                    for n in names})
    with open(f"explain_{EVAL_YEAR}.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    pd.DataFrame(rows).to_csv(f"explain_{EVAL_YEAR}_pixels.csv.gz", index=False)
    print(f"\nbase pooled AUC-PR {ap0:.4f} on {npat} patches")
    print(f"{'input group':26s} {'retention':>9s} {'time-fold sd':>12s} {'space-fold sd':>13s} {'AP drop':>8s}")
    for name in names:
        g = out["groups"][name]
        ft, fs = out["folds"]["time"]["groups"][name], out["folds"]["space"]["groups"][name]
        print(f"{name:26s} {g['retention']:9.3f} {np.std(ft['retention']):12.3f} {np.std(fs['retention']):13.3f} "
              f"{100*g['ap_drop']:7.1f}%   clim {g['by_climate']}")
    print(f"wrote explain_{EVAL_YEAR}.json and explain_{EVAL_YEAR}_pixels.csv.gz ({len(rows)} pixels)")


if __name__ == "__main__":
    main()
