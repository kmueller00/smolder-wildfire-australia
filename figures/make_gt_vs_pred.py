"""Ground truth vs SMOLDER prediction for four 2020 test-year dates.

One row per date: what actually burned, next to what the model predicted the
day before. Both panels share the same patch, extent and colour conventions so
the comparison is honest and immediate.

Uses the released SWA checkpoint and the 2020 hold-out year -- never seen in
training or model selection.
"""
import os, sys
sys.path.insert(0, "/home/saturn/gwgi/gwgi107h/wildfire_data/firecastnet")
import numpy as np, torch, zarr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap

from conv_lstm_lit_dual import ConvLSTMLitDual
from zarr_dual_datamodule import DualPatchConfig, DualWindowDataset

CKPT = os.environ.get("CKPT", "/home/saturn/gwgi/gwgi107h/wildfire_data/checkpoints/"
    "dual_fh_attn_ps384_final_newfiresample_fhdropout_combo/job_1764897/swa_resume_ep28_29_30.ckpt")
PATCH = 384
N_EX = 4
LON0, LAT0, PX = 112.904998779, -9.005000113999998, 0.01
OUT = "figures/fig_gt_vs_pred_2020.png"

RISK = LinearSegmentedColormap.from_list("risk", ["#f7f7f5", "#ffe9a8", "#ffab3d", "#e8452c", "#8b0000"])


def main():
    torch.set_num_threads(8)
    g = zarr.open_group("cube_daily_smgrid_2020.zarr", mode="r")
    lm_full = g["landmask"][:]
    TIMES = list(g.attrs.get("time", []))

    m = ConvLSTMLitDual.load_from_checkpoint(CKPT, map_location="cpu"); m.eval()
    print("[info] loaded", os.path.basename(CKPT), flush=True)

    ds = DualWindowDataset(DualPatchConfig(
        zarr_paths=("cube_daily_smgrid_2020.zarr",), stats_path="channel_stats_2015_2018.json",
        slow_cube_path="cube_slow_8day.zarr", day_offset=1826,
        patch_size=PATCH, samples_per_epoch=300, seed=21,
        min_pos_pixels=45, pos_frac=1.0, deterministic=True,
        fire_history=True, fire_history_lags=(3, 4, 5), fire_history_distance=True))

    picks, seen = [], set()
    for i in range(300):
        b = ds[i]
        land = b["mask"].numpy() > 0.5
        y = (b["y"][-1].numpy() > 0) & land
        n = int(y.sum())
        tt = int(b["t_end"]); iso = TIMES[tt] if tt < len(TIMES) else str(tt)
        # Dedup on the ISO month BEFORE reformatting, so the four panels stay
        # spread across the year. Display form is "16 June 2020" so that no
        # dash characters appear anywhere in the rendered figure.
        _M = ["January","February","March","April","May","June","July",
              "August","September","October","November","December"]
        try:
            date = f"{int(iso[8:10])} {_M[int(iso[5:7])-1]} {iso[:4]}"
        except Exception:
            date = iso
        if n >= 250 and land.mean() > 0.65 and iso[:7] not in seen:
            seen.add(iso[:7]); picks.append((n, date, b))
        if len(picks) >= N_EX: break
    picks.sort(key=lambda t: -t[0])
    print("[info] dates:", [(p[1], p[0]) for p in picks], flush=True)

    fig, axes = plt.subplots(len(picks), 2, figsize=(9.6, 4.55 * len(picks)))
    if len(picks) == 1: axes = axes[None, :]

    for r, (nfire, date, b) in enumerate(picks):
        land = b["mask"].numpy() > 0.5
        truth = (b["y"][-1].numpy() > 0) & land
        y0, x0 = int(b["y0"]), int(b["x0"])
        ext = [LON0 + x0*PX, LON0 + (x0+PATCH)*PX, LAT0 - (y0+PATCH)*PX, LAT0 - y0*PX]

        with torch.no_grad():
            p = torch.sigmoid(m.forward_seq(b["x_slow"].unsqueeze(0), b["x_fast"].unsqueeze(0),
                                            b["x_cat"].unsqueeze(0))[:, -1])[0].numpy()
        v = p[land]
        thr = np.partition(v, -max(1, int(0.01*v.size)))[-max(1, int(0.01*v.size))]
        top1 = land & (p >= thr)
        caught = int((truth & top1).sum())

        for c in (0, 1):
            ax = axes[r, c]
            base = np.zeros((*land.shape, 3), np.float32)
            base[...] = (0.80, 0.89, 0.95); base[land] = (0.96, 0.96, 0.94)
            ax.imshow(base, extent=ext, origin="upper", interpolation="nearest")

            if c == 0:
                yy, xx = np.where(truth)
                ax.scatter(ext[0] + (xx+0.5)*PX, ext[3] - (yy+0.5)*PX, s=2.0,
                           c="#b00000", marker="s", linewidths=0)
                ax.set_title(f"Observed fire  ·  {date}", fontsize=11, fontweight="bold", pad=7)
                ax.text(0.025, 0.045, f"{nfire} burned pixels", transform=ax.transAxes,
                        fontsize=8.8, fontweight="bold",
                        bbox=dict(fc="white", ec="0.65", lw=0.6, alpha=0.9, pad=2.2))
            else:
                # Shown as within-scene percentile, not raw sigmoid. The raw
                # output is deliberately compressed (pos_weight buys ranking at
                # the cost of calibration -- documented), so a 0-1 colour scale
                # renders as flat orange and hides the structure. Percentile is
                # also exactly how the model is used operationally: rank pixels,
                # take the top k%.
                from scipy.stats import rankdata
                pm = np.full(p.shape, np.nan, np.float32)
                pm[land] = 100.0 * (rankdata(p[land], method="average") - 1) / max(land.sum() - 1, 1)
                im = ax.imshow(pm, extent=ext, origin="upper", cmap=RISK, vmin=0, vmax=100,
                               interpolation="nearest")
                ax.contour(top1.astype(float), levels=[0.5], colors="#111", linewidths=0.7,
                           extent=ext, origin="upper")
                yy, xx = np.where(truth)
                ax.scatter(ext[0] + (xx+0.5)*PX, ext[3] - (yy+0.5)*PX, s=1.4,
                           c="#0b3d91", marker="s", linewidths=0, alpha=0.85)
                ax.set_title("SMOLDER predicted risk  ·  3 days ahead", fontsize=11,
                             fontweight="bold", pad=7)
                ax.text(0.025, 0.045,
                        f"{caught}/{nfire} caught in top 1%  ({100*caught/max(nfire,1):.0f}%)",
                        transform=ax.transAxes, fontsize=8.8, fontweight="bold",
                        bbox=dict(fc="white", ec="0.65", lw=0.6, alpha=0.9, pad=2.2))
                cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
                cb.set_label("risk percentile\n(within scene)", fontsize=7.5); cb.ax.tick_params(labelsize=7)

            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_linewidth(0.8)

    handles = [Patch(facecolor="#b00000", label="Observed fire (left panels)"),
               Patch(facecolor="#0b3d91", label="Observed fire overlaid on prediction"),
               Patch(facecolor="none", edgecolor="#111", label="Model top 1% highest risk area"),
               Patch(facecolor=(0.80, 0.89, 0.95), label="Ocean / masked")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, 0.004))
    fig.suptitle("SMOLDER  |  predicted wildfire risk vs what actually burned\n"
                 "2020 holdout year, never used for training or model selection",
                 fontsize=13.5, fontweight="bold", y=0.997)
    fig.subplots_adjust(bottom=0.055, top=0.945, hspace=0.10, wspace=0.02)
    fig.savefig(OUT, dpi=250, bbox_inches="tight", facecolor="white")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
