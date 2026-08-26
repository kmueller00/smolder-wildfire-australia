"""Per-(region x season) isotonic recalibration -- fit on 2019, evaluated on 2020.

WHY THIS CAN MOVE THE METRIC AT ALL (the global version cannot):
`fit_recalibration.py` fits ONE isotonic map over all pixels. Isotonic
regression is monotonic, so it cannot change the ranking and therefore cannot
change AUC-PR by construction -- that script's own log says so ("AP unchanged
is EXPECTED"). It only fixes the probability VALUES for map display.

Fitting a SEPARATE monotone map per group does change global ranking, because
two different monotone maps reorder scores ACROSS groups. That is exactly the
degree of freedom needed here: this model's skill varies ~3x by group on 2020
test (SON new-fire lift 13.9x vs MAM 4.1x; Tropical N 11.4x vs Temperate S
4.7x), so a single global threshold spends its top-k budget suboptimally --
a 0.6 in a weak group and a 0.6 in a strong group do not carry the same
evidence of fire, but pooled ranking treats them identically.

NOT the same as percentile-normalizing per group. Flattening every group to a
uniform distribution would erase genuine base-rate differences (Tropical N
really does burn more), which would destroy signal. Isotonic maps raw score ->
observed fire RATE within the group, so real base-rate differences survive and
only the group-specific miscalibration is removed.

Leakage: maps are fit on 2019 (the validation year) and applied to 2020 (the
held-out test year), so no test labels are used to build the transform.
Groups with too few positives fall back to the global map.

Usage:
    CKPT=... PATCH=384 N_FIT=900 N_TEST=900 OUT=recal_grouped_<tag>.pkl \
      python fit_grouped_recalibration.py
"""
import os
import pickle

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from conv_lstm_lit_dual import ConvLSTMLitDual
from zarr_dual_datamodule import DualPatchConfig, DualWindowDataset

CKPT = os.environ["CKPT"]
PATCH = int(os.environ.get("PATCH", 384))
N_FIT = int(os.environ.get("N_FIT", 900))
N_TEST = int(os.environ.get("N_TEST", 900))
OUT = os.environ.get("OUT", "recal_grouped.pkl")
MIN_POS_PER_GROUP = int(os.environ.get("MIN_POS_PER_GROUP", 200))

LAT0, PX = -9.005000113999998, 0.01
OFFS = {"2015": 0, "2016": 365, "2017": 731, "2018": 1096, "2019": 1461, "2020": 1826}
SEAS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
        6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def region_of(y0):
    """Same definition as operational_stats_dual.py -- keep in sync."""
    lat = LAT0 - (y0 + PATCH / 2) * PX
    return "Tropical N" if lat > -20 else ("Temperate S" if lat < -30 else "Central")


def harvest(model, device, year, n_patches, pos_frac, seed):
    """Return (probs, labels, group_key) flattened over land pixels."""
    import zarr
    g = zarr.open_group(f"cube_daily_smgrid_{year}.zarr", mode="r")
    times = list(g.attrs.get("time", []))

    ds = DualWindowDataset(DualPatchConfig(
        zarr_paths=(f"cube_daily_smgrid_{year}.zarr",),
        stats_path="channel_stats_2015_2018.json",
        slow_cube_path="cube_slow_8day.zarr", day_offset=OFFS[year],
        patch_size=PATCH, samples_per_epoch=n_patches, seed=seed, deterministic=True,
        min_pos_pixels=0, pos_frac=pos_frac, min_valid_frac=0.5,
        fire_history=True, fire_history_lags=(3, 4, 5), fire_history_distance=True,
        use_lightning=os.environ.get("USE_LIGHTNING", "0") == "1",
        use_elevation=os.environ.get("USE_ELEVATION", "0") == "1",
        use_slope_aspect=os.environ.get("USE_SLOPE_ASPECT", "0") == "1",
        use_fuel_age=os.environ.get("USE_FUEL_AGE", "0") == "1",
        use_wind_dir=os.environ.get("USE_WIND_DIR", "0") == "1"))

    P, Y, G = [], [], []
    for i in range(n_patches):
        b = ds[i]
        land = b["mask"].numpy() > 0.5
        if land.mean() < 0.5:
            continue
        y = (b["y"][-1].numpy() > 0) & land
        with torch.no_grad():
            p = torch.sigmoid(model.forward_seq(
                b["x_slow"].unsqueeze(0).to(device),
                b["x_fast"].unsqueeze(0).to(device),
                b["x_cat"].unsqueeze(0).to(device))[:, -1])[0].cpu().numpy()
        tt = int(b["t_end"])
        date = times[tt] if tt < len(times) else ""
        mth = int(date[5:7]) if date else 0
        key = f"{region_of(int(b['y0']))}|{SEAS.get(mth, '?')}"
        n = int(land.sum())
        P.append(p[land].astype(np.float32))
        Y.append(y[land].astype(np.int8))
        G.append(np.full(n, key, dtype=object))
        if (i + 1) % 100 == 0:
            print(f"[info] {year}: harvested {i+1}/{n_patches}", flush=True)
    return np.concatenate(P), np.concatenate(Y), np.concatenate(G)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = ConvLSTMLitDual.load_from_checkpoint(CKPT, map_location=device)
    m.eval()
    m.to(device)
    torch.set_num_threads(8)
    print(f"[info] loaded {os.path.basename(CKPT)} on {device}, patch={PATCH}", flush=True)

    # ---- fit on 2019 (validation year) ----
    print("\n=== [1/2] harvesting 2019 to FIT calibration maps ===", flush=True)
    Pf, Yf, Gf = harvest(m, device, "2019", N_FIT, pos_frac=0.3, seed=999)
    print(f"[info] fit set: {len(Yf):,} px, {100*Yf.mean():.3f}% positive", flush=True)

    glob = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(Pf, Yf)

    groups = {}
    print(f"\n{'group':<26} {'px':>12} {'pos':>8} {'rate':>9}  status")
    for key in sorted(set(Gf.tolist())):
        sel = Gf == key
        npos = int(Yf[sel].sum())
        rate = float(Yf[sel].mean()) if sel.sum() else 0.0
        if npos >= MIN_POS_PER_GROUP:
            groups[key] = IsotonicRegression(
                out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(Pf[sel], Yf[sel])
            status = "fit"
        else:
            status = f"FALLBACK to global (<{MIN_POS_PER_GROUP} pos)"
        print(f"{key:<26} {int(sel.sum()):12,} {npos:8,} {rate:9.5f}  {status}")

    with open(OUT, "wb") as f:
        pickle.dump({"global": glob, "groups": groups, "patch": PATCH,
                     "ckpt": CKPT, "min_pos_per_group": MIN_POS_PER_GROUP}, f)
    print(f"\n[done] wrote {OUT}  ({len(groups)} group maps + 1 global fallback)", flush=True)

    # ---- evaluate on 2020 (held-out test year) ----
    print("\n=== [2/2] harvesting 2020 TEST to evaluate the transform ===", flush=True)
    Pt, Yt, Gt = harvest(m, device, "2020", N_TEST, pos_frac=1.0, seed=21)
    print(f"[info] test set: {len(Yt):,} px, {100*Yt.mean():.3f}% positive", flush=True)

    p_glob = glob.predict(Pt)
    p_grp = np.empty_like(Pt, dtype=np.float64)
    n_fallback = 0
    for key in sorted(set(Gt.tolist())):
        sel = Gt == key
        iso = groups.get(key)
        if iso is None:
            n_fallback += int(sel.sum())
            iso = glob
        p_grp[sel] = iso.predict(Pt[sel])

    print(f"\n[info] {n_fallback:,} test px ({100*n_fallback/len(Yt):.1f}%) used the global fallback")
    print(f"\n{'transform':<22} {'AUC-PR':>9} {'ROC-AUC':>9}")
    rows = [("raw (no calibration)", Pt), ("global isotonic", p_glob), ("GROUPED isotonic", p_grp)]
    aps = {}
    for name, s in rows:
        ap = average_precision_score(Yt, s)
        roc = roc_auc_score(Yt, s)
        aps[name] = ap
        print(f"{name:<22} {ap:9.4f} {roc:9.4f}")

    base = aps["raw (no calibration)"]
    d_glob = 100.0 * (aps["global isotonic"] - base) / base
    d_grp = 100.0 * (aps["GROUPED isotonic"] - base) / base
    # NOTE: global-vs-raw is NOT expected to be exactly 0. Isotonic regression is
    # only WEAKLY monotone -- a piecewise-constant step function -- so while it
    # cannot reorder scores, it does collapse distinct scores into plateaus, and
    # those ties genuinely cost AP (measured -0.72% on the combo checkpoint).
    # Treat global-vs-raw as the tie-induced floor, and judge the grouped map
    # against GLOBAL (the like-for-like comparison), not against raw.
    print(f"\n  global isotonic vs raw : {d_glob:+.3f}%  (tie-induced floor: isotonic is a "
          f"step function, so it creates ties even though it cannot reorder)")
    print(f"  GROUPED isotonic vs raw: {d_grp:+.3f}%")
    d_vs_glob = 100.0 * (aps["GROUPED isotonic"] - aps["global isotonic"]) / aps["global isotonic"]
    print(f"  GROUPED vs GLOBAL      : {d_vs_glob:+.3f}%  <-- the like-for-like result")
    if d_grp > 0.5:
        print("  => grouped calibration IMPROVES pooled ranking over RAW -- a real AP lever")
    elif d_vs_glob > 0.5:
        print("  => grouped beats global, but neither beats raw: worth it for calibrated "
              "probabilities, NOT as a way to improve AP")
    else:
        print("  => no meaningful change")


if __name__ == "__main__":
    main()
