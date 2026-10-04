"""Observed fire vs SMOLDER's predicted risk for four 2020 hold-out patches.

Two steps:
  python make_gt_vs_pred.py --compute   needs the model + data (SMOLDER_DATA);
                                        writes data/gt_vs_pred_2020.npz
  python make_gt_vs_pred.py             plots from that cached file only

Patch selection is deterministic (evaluation seed, fire-active patches, one per
calendar month, the four with the most fire). Each row is one forecast: the
right panel is the risk map issued on the stated date, the left panel the fire
observed over the following three days (the model's target).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator

from style_smolder import GRID_COLOR, INK, MUTED, PANEL_BG, SPINE_COLOR

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "gt_vs_pred_2020.npz")
OUT = os.path.join(HERE, "fig_gt_vs_pred_2020.png")
LON0, LAT0, PX = 112.904998779, -9.005000113999998, 0.01
PATCH = 384
N_EX = 4
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

OCEAN = "#C9D6E3"
HATCH = "#3A3A3A"
plt.rcParams["hatch.color"] = HATCH
plt.rcParams["hatch.linewidth"] = 1.1
FIRE_MISS = "#C400FF"    # observed fire outside the top-1% area
FIRE_HIT = "#00E83A"     # observed fire inside the top-1% area
RISK = LinearSegmentedColormap.from_list(
    "risk", [PANEL_BG, "#FFE9A8", "#FFAB3D", "#E8452C", "#8B0000"])


def compute():
    import torch
    from scipy.stats import rankdata
    sys.path.insert(0, os.path.dirname(HERE))
    from smolder.data.io import daily_cube, open_zarr_root
    from smolder.data.zarr_dual_datamodule import DualPatchConfig, DualWindowDataset
    from smolder.models.conv_lstm_lit_dual import ConvLSTMLitDual

    ckpt = os.environ.get("CKPT", os.path.join(os.path.dirname(HERE), "checkpoints", "smolder_swa.ckpt"))
    cube = daily_cube(2020)
    times = list(open_zarr_root(cube).attrs.get("time", []))
    m = ConvLSTMLitDual.load_from_checkpoint(ckpt, map_location="cpu")
    m.eval()
    ds = DualWindowDataset(DualPatchConfig(
        zarr_paths=(cube,), stats_path="channel_stats_2015_2018.json",
        slow_cube_path="cube_slow_8day.zarr", day_offset=1826,
        patch_size=PATCH, samples_per_epoch=300, seed=21,
        min_pos_pixels=45, pos_frac=1.0, deterministic=True,
        fire_history=True, fire_history_lags=(3, 4, 5), fire_history_distance=True))

    picks, seen = [], set()
    for i in range(300):
        b = ds[i]
        land = b["mask"].numpy() > 0.5
        truth = (b["y"][-1].numpy() > 0) & land
        iso = times[int(b["t_end"])]
        if truth.sum() >= 250 and land.mean() > 0.65 and iso[:7] not in seen:
            seen.add(iso[:7])
            picks.append((int(truth.sum()), iso, b, land, truth))
        if len(picks) >= N_EX:
            break
    picks.sort(key=lambda t: -t[0])

    out = {}
    for r, (_, iso, b, land, truth) in enumerate(picks):
        with torch.no_grad():
            p = torch.sigmoid(m.forward_seq(b["x_slow"].unsqueeze(0), b["x_fast"].unsqueeze(0),
                                            b["x_cat"].unsqueeze(0))[:, -1])[0].numpy()
        pct = np.full(p.shape, np.nan, np.float32)
        pct[land] = 100.0 * (rankdata(p[land], method="average") - 1) / max(land.sum() - 1, 1)
        out[f"pct_{r}"] = pct.astype(np.float16)
        out[f"prob_{r}"] = p.astype(np.float32)
        out[f"land_{r}"] = land
        out[f"truth_{r}"] = truth
        out[f"yx_{r}"] = np.array([int(b["y0"]), int(b["x0"])])
        out[f"date_{r}"] = np.array(iso)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, n=len(picks), **out)
    print("wrote", CACHE, [str(out[f"date_{r}"]) for r in range(len(picks))])


def _deg(axis):
    if axis == "lon":
        return FuncFormatter(lambda v, _: f"{v:.0f}°E")
    return FuncFormatter(lambda v, _: f"{abs(v):.0f}°S")


def _style_map(ax):
    """House style on a map panel: grey panel, white dashed gridlines at every
    (1 degree) tick. Gridlines sit above the raster so they stay visible."""
    ax.set_facecolor(PANEL_BG)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_major_formatter(_deg("lon"))
    ax.yaxis.set_major_formatter(_deg("lat"))
    ax.minorticks_off()
    ax.set_axisbelow(False)
    ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.8, alpha=0.9)
    ax.tick_params(colors=INK, labelsize=8, length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE_COLOR)


COAST = "#1F1F22"
LOCATOR = "#E3001B"          # patch outline in the locator inset
INSET_LAND = "#F4F1EA"
INSET_W = 0.30                      # inset width as a fraction of the panel width


def _coastline(ax, land, ext, lw, z):
    if land.all():
        return
    ax.contour(land.astype(float), levels=[0.5], colors=COAST, linewidths=lw,
               extent=ext, origin="upper", zorder=z)


def _locator(ax, nat, y0, x0, yy, xx):
    """Inset map of Australia marking the patch, placed in the free corner
    (upper left, upper right or lower right) that covers the fewest fire pixels."""
    land, down = nat["land"], int(nat["down"])
    H, W = land.shape
    iw = INSET_W
    ih = iw * H / W                         # panels are square, so keep the map's aspect
    corners = {"upper left": (0.02, 0.98 - ih), "upper right": (0.98 - iw, 0.98 - ih),
               "lower right": (0.98 - iw, 0.02)}
    fx, fy = (xx + 0.5) / PATCH, 1.0 - (yy + 0.5) / PATCH          # fire in axes fractions
    cover = {k: int(((fx >= cx) & (fx <= cx + iw) & (fy >= cy) & (fy <= cy + ih)).sum())
             for k, (cx, cy) in corners.items()}
    cx, cy = corners[min(cover, key=cover.get)]
    ins = ax.inset_axes([cx, cy, iw, ih], zorder=6)
    ext = [LON0, LON0 + W * down * PX, LAT0 - H * down * PX, LAT0]
    ins.set_facecolor(OCEAN)
    ins.imshow(np.where(land, 1.0, np.nan), extent=ext,
               cmap=LinearSegmentedColormap.from_list("l", [INSET_LAND, INSET_LAND]),
               interpolation="nearest", zorder=1)
    ins.contour(land.astype(float), levels=[0.5], colors=COAST, linewidths=0.5,
                extent=ext, origin="upper", zorder=2)
    ins.add_patch(Rectangle((LON0 + x0 * PX, LAT0 - (y0 + PATCH) * PX), PATCH * PX, PATCH * PX,
                            facecolor="none", edgecolor=LOCATOR, lw=1.3, zorder=3))
    ins.set_xlim(ext[0], ext[1]); ins.set_ylim(ext[2], ext[3]); ins.set_aspect("equal")
    ins.set_xticks([]); ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_edgecolor(MUTED); sp.set_linewidth(0.6)


def plot():
    d = np.load(CACHE)
    nat = np.load(os.path.join(HERE, "..", "results", "national_2020_maps.npz"))
    n = int(d["n"])
    nrow = (n + 1) // 2                     # two forecasts per row, each as observed | predicted
    fig = plt.figure(figsize=(19.6, 4.75 * nrow + 1.3), facecolor="white")
    gs = fig.add_gridspec(nrow, 5, width_ratios=[1, 1, 0.1, 1, 1])
    axes = np.array([[fig.add_subplot(gs[i // 2, 3 * (i % 2) + c]) for c in (0, 1)] for i in range(n)])
    for r in range(n):
        pct = d[f"pct_{r}"].astype(np.float32)
        prob = d[f"prob_{r}"]
        land, truth = d[f"land_{r}"], d[f"truth_{r}"]
        y0, x0 = d[f"yx_{r}"]
        iso = str(d[f"date_{r}"])
        date = f"{int(iso[8:10])} {MONTHS[int(iso[5:7]) - 1]} {iso[:4]}"
        ext = [LON0 + x0 * PX, LON0 + (x0 + PATCH) * PX, LAT0 - (y0 + PATCH) * PX, LAT0 - y0 * PX]

        v = prob[land]
        k = max(1, int(0.01 * v.size))
        top1 = land & (prob >= np.partition(v, -k)[-k])
        caught = int((truth & top1).sum())
        nfire = int(truth.sum())
        yy, xx = np.where(truth)
        fx, fy = ext[0] + (xx + 0.5) * PX, ext[3] - (yy + 0.5) * PX
        hit = top1[yy, xx]
        ocean = np.ma.masked_where(land, np.ones(land.shape))

        ax = axes[r, 0]
        ax.imshow(ocean, extent=ext, cmap=LinearSegmentedColormap.from_list("o", [OCEAN, OCEAN]),
                  interpolation="nearest", zorder=0)
        ax.scatter(fx[~hit], fy[~hit], s=2.2, c=FIRE_MISS, marker="s", linewidths=0, zorder=3)
        ax.scatter(fx[hit], fy[hit], s=2.2, c=FIRE_HIT, marker="s", linewidths=0, zorder=3)
        _coastline(ax, land, ext, 0.9, 3.5)
        _locator(ax, nat, y0, x0, yy, xx)
        ax.set_title(f"Observed fire, 3 days after {date}", fontsize=10.5,
                     fontweight="bold", color=INK, pad=6)
        ax.text(0.025, 0.04, f"{nfire} fire pixels", transform=ax.transAxes, fontsize=8.5,
                fontweight="bold", color=INK, zorder=5,
                bbox=dict(fc="white", ec=MUTED, lw=0.6, alpha=0.92, pad=2.2))

        ax = axes[r, 1]
        ax.imshow(ocean, extent=ext, cmap=LinearSegmentedColormap.from_list("o", [OCEAN, OCEAN]),
                  interpolation="nearest", zorder=0)
        im = ax.imshow(pct, extent=ext, cmap=RISK, vmin=0, vmax=100, interpolation="nearest", zorder=1)
        ax.contourf(top1.astype(float), levels=[0.5, 1.5], colors=["white"], alpha=0.45,
                    extent=ext, origin="upper", zorder=3)
        ax.contourf(top1.astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"],
                    extent=ext, origin="upper", zorder=3)
        ax.contour(top1.astype(float), levels=[0.5], colors=HATCH, linewidths=0.6,
                   extent=ext, origin="upper", zorder=3)
        _coastline(ax, land, ext, 0.9, 4.5)
        ax.scatter(fx[~hit], fy[~hit], s=1.6, c=FIRE_MISS, marker="s", linewidths=0, zorder=4)
        ax.scatter(fx[hit], fy[hit], s=1.6, c=FIRE_HIT, marker="s", linewidths=0, zorder=4)
        ax.set_title(f"SMOLDER risk, issued {date}", fontsize=10.5, fontweight="bold",
                     color=INK, pad=6)
        ax.text(0.025, 0.04, f"{caught}/{nfire} in top-1% area ({100 * caught / max(nfire, 1):.0f}%)",
                transform=ax.transAxes, fontsize=8.5, fontweight="bold", color=INK, zorder=5,
                bbox=dict(fc="white", ec=MUTED, lw=0.6, alpha=0.92, pad=2.2))

        for c in (0, 1):
            axes[r, c].set_xlim(ext[0], ext[1])
            axes[r, c].set_ylim(ext[2], ext[3])
            axes[r, c].set_aspect("equal")
            _style_map(axes[r, c])

    fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.12, hspace=0.22, wspace=0.2)
    cax = fig.add_axes([0.60, 0.06, 0.36, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Predicted risk, percentile within the patch", fontsize=9, color=INK)
    cb.ax.tick_params(labelsize=8, colors=INK)
    cb.outline.set_edgecolor(SPINE_COLOR)
    handles = [Patch(facecolor=FIRE_HIT, label="Observed fire inside the top-1% area"),
               Patch(facecolor=FIRE_MISS, label="Observed fire outside it"),
               Patch(facecolor="none", edgecolor=HATCH, hatch="////", label="Top-1% risk area"),
               Patch(facecolor=OCEAN, label="Ocean"),
               Patch(facecolor="none", edgecolor=LOCATOR, lw=1.3, label="Location of the patch (inset)")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.05, 0.02), ncol=3,
               fontsize=8.6, frameon=True, facecolor="white", edgecolor=MUTED)
    fig.suptitle("Observed fire vs predicted risk, 2020 hold-out year",
                 fontsize=14, fontweight="bold", color=INK, y=0.995)
    fig.savefig(OUT, dpi=250, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    compute() if "--compute" in sys.argv else plot()
