"""Permutation feature importance for the DUAL-branch ConvLSTM (ConvLSTMLitDual).

Adapts convlstm_importance.py (written for the old single-branch model) to the
current dual slow/fast architecture. Motivation: the full-feature+augmentation
run (job 1761503, val_ap=0.4436) essentially tied the pre-feature baseline
(ps384_final, 0.4589) instead of clearly beating it, even though every one of
the 6 new features passed its individual XGBoost gate with a real AP gain
(lightning +7.4%, elevation +5.3%, slope+aspect+fuel_age +22.3%, wind +2.4%).
This checks whether the trained ConvLSTM is actually using those channels, or
whether they're dead weight the network learned to ignore.

Channel order (verified against zarr_dual_datamodule.py's __getitem__, static
block + doy + fire-history/fuel_age/downwind append order):
  x_slow (12): LAI,SM,PPT, agb,lm,lightning,elevation,slope,aspect_sin,aspect_cos, doy_sin,doy_cos
  x_fast (18): VPD,LST,WIND, agb,lm,lightning,elevation,slope,aspect_sin,aspect_cos, doy_sin,doy_cos,
               fire_hist_t-3,fire_hist_t-4,fire_hist_t-5,fire_dist, fuel_age, downwind_align

Statics shared by both branches (agb, lm, lightning, elevation, slope,
aspect_sin, aspect_cos, doy_sin, doy_cos) are permuted in BOTH branches at once
(same permutation index) so the feature's information is fully removed, not
just partially masked in one branch while leaking through the other.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from sklearn.metrics import average_precision_score

from conv_lstm_lit_dual import ConvLSTMLitDual
from zarr_dual_datamodule import DualDataModule

CKPT = os.environ.get("CKPT",
  "/home/saturn/gwgi/gwgi107h/wildfire_data/checkpoints/dual_fh_attn_ps384_fullfeat/job_1761503/"
  "lightning_logs/version_0/checkpoints/best-epoch=15-val_ap=0.4436.ckpt")
PATCH = int(os.environ.get("PATCH", 192))
N_PATCHES = int(os.environ.get("N_PATCHES", 24))
N_REPEAT = int(os.environ.get("N_REPEAT", 2))

# (name, slow_idx or None, fast_idx or None)
FEATURES = [
    ("LAI",            0,  None),
    ("SM",             1,  None),
    ("PPT",            2,  None),
    ("VPD",            None, 0),
    ("LST",            None, 1),
    ("WIND",           None, 2),
    ("agb",            3,  3),
    ("landmask",       4,  4),
    ("lightning",      5,  5),
    ("elevation",      6,  6),
    ("slope",          7,  7),
    ("aspect_sin",     8,  8),
    ("aspect_cos",     9,  9),
    ("doy_sin",       10, 10),
    ("doy_cos",       11, 11),
    ("fire_hist_t-3",  None, 12),
    ("fire_hist_t-4",  None, 13),
    ("fire_hist_t-5",  None, 14),
    ("fire_dist",      None, 15),
    ("fuel_age",       None, 16),
    ("downwind_align", None, 17),
]


def main():
    torch.set_num_threads(8)
    m = ConvLSTMLitDual.load_from_checkpoint(CKPT, map_location="cpu")
    m.eval()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    BASE_PATCH, BASE_MIN_POS = 256, 20
    min_pos_pixels = max(1, round(BASE_MIN_POS * (PATCH / BASE_PATCH) ** 2))

    dm = DualDataModule(
        train_paths=[os.path.join(script_dir, f"cube_daily_smgrid_{y}.zarr") for y in (2015, 2016, 2017, 2018)],
        val_paths=[os.path.join(script_dir, "cube_daily_smgrid_2019.zarr")],
        stats_path=os.path.join(script_dir, "channel_stats_2015_2018.json"),
        slow_cube_path=os.path.join(script_dir, "cube_slow_8day.zarr"),
        y_key="y_fire_3d", valid_key="y_fire_3d_valid",
        slow_days=144, slow_bin=8, fast_days=14, patch_size=PATCH,
        samples_per_epoch=N_PATCHES, batch_size=1, num_workers=0,
        min_pos_pixels=min_pos_pixels, train_seed=123,
        fire_history=True, fire_history_lags=(3, 4, 5), fire_history_distance=True,
        use_lightning=True, use_elevation=True, use_slope_aspect=True,
        use_fuel_age=True, use_wind_dir=True, augment=False,
    )
    dm.setup("fit")

    XS, XF, Y, M, C = [], [], [], [], []
    for i in range(N_PATCHES):
        b = dm.val_ds[i]
        XS.append(b["x_slow"]); XF.append(b["x_fast"])
        Y.append(b["y"][-1]); M.append(b["mask"]); C.append(b["x_cat"])
    XS = torch.stack(XS); XF = torch.stack(XF); Y = torch.stack(Y)
    M = torch.stack(M) > 0.5; C = torch.stack(C)
    print(f"[info] {len(XS)} val patches @ {PATCH}px, x_slow={tuple(XS.shape)} x_fast={tuple(XF.shape)}, "
          f"ckpt={os.path.basename(CKPT)}", flush=True)

    def score(xs, xf):
        with torch.no_grad():
            logits = m.forward(xs, xf, C)[:, 0]   # (N,H,W)
            probs = torch.sigmoid(logits)
        yv = Y[M].numpy()
        pv = probs[M].numpy()
        if yv.sum() == 0 or yv.sum() == len(yv):
            return float("nan")
        return average_precision_score(yv, pv)

    base = score(XS, XF)
    print(f"[base] val_ap = {base:.4f}\n", flush=True)

    rng = np.random.default_rng(0)
    rows = []
    for name, si, fi in FEATURES:
        drops = []
        for r in range(N_REPEAT):
            perm = torch.from_numpy(rng.permutation(len(XS)))
            xs_p, xf_p = XS.clone(), XF.clone()
            if si is not None:
                xs_p[..., si] = XS[perm][..., si]
            if fi is not None:
                xf_p[..., fi] = XF[perm][..., fi]
            drops.append(base - score(xs_p, xf_p))
        d = float(np.mean(drops)); s = float(np.std(drops))
        rows.append((name, d, s))
        print(f"  {name:16} drop={d:+.5f} +/- {s:.5f}", flush=True)

    rows.sort(key=lambda t: -t[1])
    print("\n=== Dual ConvLSTM permutation importance (val_ap drop when shuffled) ===", flush=True)
    tot = sum(max(0, r[1]) for r in rows) or 1
    NEW = {"lightning", "elevation", "slope", "aspect_sin", "aspect_cos", "fuel_age", "downwind_align"}
    for n, d, s in rows:
        star = " <-- NEW (2026-07-25)" if n in NEW else ""
        print(f"  {n:16} {d:+.5f}   {100*max(0,d)/tot:5.1f}%{star}", flush=True)

    json.dump({"ckpt": CKPT, "patch": PATCH, "base_ap": base, "rows": rows},
               open("convlstm_importance_dual.json", "w"), indent=1)
    print("\nwrote convlstm_importance_dual.json", flush=True)


if __name__ == "__main__":
    main()
