"""Fit an isotonic map from SMOLDER's raw scores to calibrated probabilities.

The training loss up-weights fire pixels (pos_weight 100 -> 20), which helps
ranking but compresses the raw sigmoid output, so raw scores are not
probabilities. Isotonic regression fitted on the 2019 validation year maps
them to observed fire frequencies. It is monotone, so it cannot change the
ranking (it can only merge ties), and a map fitted for one checkpoint does not
transfer to another.

Needs cube_daily_smgrid_2019.zarr, which is not part of the Zenodo archive
(rebuild it with smolder/data, see README).

Usage:
    SMOLDER_DATA=/path/to/cubes python -m smolder.evaluation.fit_recalibration
"""
import os, pickle, numpy as np, torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

from smolder.models.conv_lstm_lit_dual import ConvLSTMLitDual
from smolder.data.io import daily_cube
from smolder.data.zarr_dual_datamodule import DualWindowDataset, DualPatchConfig

CKPT = os.environ.get("CKPT", "checkpoints/smolder_swa.ckpt")
PATCH = int(os.environ.get("PATCH", 384))
N_PATCHES = int(os.environ.get("N_PATCHES", 400))
OUT = os.environ.get("OUT", "recal_isotonic_smolder.pkl")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = ConvLSTMLitDual.load_from_checkpoint(CKPT, map_location=device); m.eval(); m.to(device)
    torch.set_num_threads(8)
    print(f"[info] loaded {os.path.basename(CKPT)} on {device}, patch_size={PATCH}", flush=True)

    ds = DualWindowDataset(DualPatchConfig(
        zarr_paths=(daily_cube(2019),), stats_path="channel_stats_2015_2018.json",
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
