"""Fit an isotonic-regression recalibration map for the dual_fh_attn checkpoint.

Why: pos_weight=100 in the training loss buys recall/ranking under extreme class
imbalance at the cost of calibration -- raw sigmoid outputs sit in a compressed
~0.65-0.75 band almost everywhere (measured on 2020 GeoTIFFs: land-wide mean 0.69,
fire-pixel mean 0.70-0.80, i.e. barely distinguishable) even though the RANKING is
real (fire-dense 128px blocks score 0.04-0.21 above local background). Isotonic
regression is monotonic, so it preserves that ranking exactly while remapping the
raw values to true calibrated probabilities fit on held-out 2019 data.

Fits on cube_daily_smgrid_2019.zarr (the val split, never used for model
selection... except that val_ap WAS the selection metric, so this is calibration
only, not a second free look at 2020 test).
Saves recal_isotonic_dual_fh_attn.pkl (or OUT/CKPT-derived name) for
make_australia_geotiff.py to apply. CKPT/PATCH/OUT are env-overridable so this
can be refit for any checkpoint -- a calibration map fit on one checkpoint's
raw output distribution does NOT transfer to another (e.g. constant vs
annealed pos_weight compress the sigmoid output differently).
"""
import os, pickle, numpy as np, torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

from conv_lstm_lit_dual import ConvLSTMLitDual
from zarr_dual_datamodule import DualWindowDataset, DualPatchConfig

CKPT = os.environ.get("CKPT",
    "/home/saturn/gwgi/gwgi107h/wildfire_data/checkpoints/dual_fh_attn/job_1755283/"
    "lightning_logs/version_0/checkpoints/best-epoch=7-val_ap=0.2790.ckpt")
PATCH = int(os.environ.get("PATCH", 128))
N_PATCHES = int(os.environ.get("N_PATCHES", 400))
OUT = os.environ.get("OUT", "recal_isotonic_dual_fh_attn.pkl")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = ConvLSTMLitDual.load_from_checkpoint(CKPT, map_location=device); m.eval(); m.to(device)
    torch.set_num_threads(8)
    print(f"[info] loaded {os.path.basename(CKPT)} on {device}, patch_size={PATCH}", flush=True)

    ds = DualWindowDataset(DualPatchConfig(
        zarr_paths=("cube_daily_smgrid_2019.zarr",), stats_path="channel_stats_2015_2018.json",
        slow_cube_path="cube_slow_8day.zarr", day_offset=1461,   # 2019 offset (2015-2018 = 1461 days)
        patch_size=PATCH, samples_per_epoch=N_PATCHES, seed=999, deterministic=True,
        min_pos_pixels=0, pos_frac=0.3, min_valid_frac=0.5,
        fire_history=True, fire_history_lags=(3, 4, 5), fire_history_distance=True,
        use_lightning=os.environ.get("USE_LIGHTNING", "0") == "1",
        use_elevation=os.environ.get("USE_ELEVATION", "0") == "1",
        use_slope_aspect=os.environ.get("USE_SLOPE_ASPECT", "0") == "1",
        use_fuel_age=os.environ.get("USE_FUEL_AGE", "0") == "1",
        use_wind_dir=os.environ.get("USE_WIND_DIR", "0") == "1"))

    P, Y = [], []
    for i in range(N_PATCHES):
        b = ds[i]
        land = b["mask"].numpy() > 0.5
        y = (b["y"][-1].numpy() > 0) & land
        with torch.no_grad():
            p = torch.sigmoid(m.forward_seq(b["x_slow"].unsqueeze(0).to(device),
                                             b["x_fast"].unsqueeze(0).to(device),
                                             b["x_cat"].unsqueeze(0).to(device))[:, -1])[0].cpu().numpy()
        P.append(p[land]); Y.append(y[land].astype(np.float32))
        if (i + 1) % 50 == 0:
            print(f"[info] harvested {i+1}/{N_PATCHES} patches", flush=True)

    P = np.concatenate(P); Y = np.concatenate(Y)
    print(f"[info] {len(Y)} pixels, {Y.mean()*100:.3f}% positive", flush=True)
    print(f"[info] raw prob: mean {P.mean():.4f}  mean|fire {P[Y>0].mean():.4f}  "
          f"mean|bg {P[Y==0].mean():.4f}  AP(raw)={average_precision_score(Y,P):.4f}", flush=True)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(P, Y)
    P_cal = iso.predict(P)
    print(f"[info] calibrated: mean {P_cal.mean():.4f}  mean|fire {P_cal[Y>0].mean():.4f}  "
          f"mean|bg {P_cal[Y==0].mean():.4f}  AP(cal)={average_precision_score(Y,P_cal):.4f}  "
          f"(AP unchanged is EXPECTED -- isotonic is monotonic, ranking is identical)", flush=True)

    with open(OUT, "wb") as f:
        pickle.dump(iso, f)
    print(f"[done] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
