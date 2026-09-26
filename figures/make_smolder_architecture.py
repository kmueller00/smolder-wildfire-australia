"""SMOLDER overview figure with real input rasters.

Two steps:
  python make_smolder_architecture.py --extract   reads the data cubes (SMOLDER_DATA)
                                                  and, if CKPT exists, the model;
                                                  writes data/architecture_patch.npz
  python make_smolder_architecture.py             draws the figure from that file

The patch is a fire-active 384 x 384 px window on the issue day 2020-11-15.
The output stack shows the model's risk maps for the last three fast time
steps when a checkpoint was available at extraction, otherwise the observed
target for the same steps.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle

from style_smolder import ACCENT, ACCENT2, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "architecture_patch.npz")
OUT = os.path.join(HERE, "fig_smolder_architecture.png")
PATCH, THUMB = 384, 128
ISSUE_DAY = 319                      # 2020-11-15, index in the 2020 cube
N_FRAMES = 5


# ----------------------------------------------------------------- extraction
def _block(a, how):
    k = PATCH // THUMB
    b = a.reshape(THUMB, k, THUMB, k)
    return b.max(axis=(1, 3)) if how == "max" else np.nanmean(b, axis=(1, 3))


def extract():
    from scipy import ndimage
    sys.path.insert(0, os.path.dirname(HERE))
    from smolder.data.io import daily_cube, open_zarr_root

    g = open_zarr_root(daily_cube(2020))
    sc = open_zarr_root("cube_slow_8day.zarr")
    lm = g["landmask"][:] > 0
    D = ISSUE_DAY

    y = (g["y_fire_3d"][D] > 0) & lm
    dens = ndimage.uniform_filter(y.astype(np.float32), PATCH, mode="constant")
    dens[~(ndimage.uniform_filter(lm.astype(np.float32), PATCH) > 0.6)] = 0
    cy, cx = np.unravel_index(np.argmax(dens), dens.shape)
    y0 = int(np.clip(cy - PATCH // 2, 0, lm.shape[0] - PATCH))
    x0 = int(np.clip(cx - PATCH // 2, 0, lm.shape[1] - PATCH))
    sl = (slice(y0, y0 + PATCH), slice(x0, x0 + PATCH))
    land = lm[sl]

    names = list(g.attrs["channels"])
    fast_days = np.linspace(D - 13, D, N_FRAMES).round().astype(int)
    X = np.asarray(g["X"][D - 13:D + 1, sl[0], sl[1], :], np.float32)
    out = dict(y0=y0, x0=x0, date=np.array(g.attrs["time"][D]),
               land=_block(land.astype(np.float32), "mean") > 0.5)

    def masked(a):
        a = np.where(land, a, np.nan)
        return _block(a, "mean").astype(np.float16)

    for key, ch in (("vpd", "vpd"), ("lst", "lst_day"), ("wind", "wind")):
        c = names.index(ch)
        out[key] = np.stack([masked(X[d - (D - 13), :, :, c]) for d in fast_days])

    gday = 1826 + D
    starts = np.asarray(sc["bin_start_day"][...])
    b_end = int(np.searchsorted(starts, gday - 7, side="right"))
    bins = np.linspace(b_end - 18, b_end - 1, N_FRAMES).round().astype(int)
    S = np.asarray(sc["X_slow"][bins[0]:bins[-1] + 1, sl[0], sl[1], :], np.float32)
    for key, c in (("lai", 0), ("sm", 1), ("ppt", 2)):
        out[key] = np.stack([masked(S[b - bins[0], :, :, c]) for b in bins])

    hist = [(g["y_fire_3d"][D - L][sl] > 0) & land for L in (5, 4, 3)]
    dist = np.exp(-ndimage.distance_transform_edt(~hist[-1]) / 5.0) if hist[-1].any() else np.zeros(land.shape)
    out["fire_hist"] = np.stack([_block(h.astype(np.float32), "max") for h in hist]).astype(np.float16)
    out["fire_dist"] = masked(dist)
    out["agb"] = masked(np.log1p(np.asarray(g["agb"][sl], np.float32)))
    out["landcover"] = np.where(_block(land.astype(np.float32), "mean") > 0.5,
                                np.asarray(g["landcover"][sl])[1::3, 1::3], 0).astype(np.uint8)
    out["koppen"] = np.where(_block(land.astype(np.float32), "mean") > 0.5,
                             np.asarray(g["koppen_geiger"][sl])[1::3, 1::3], 0).astype(np.uint8)

    steps = [D - 2, D - 1, D]
    out["target"] = np.stack([_block((((g["y_fire_3d"][s][sl] > 0) & land)).astype(np.float32), "max")
                              for s in steps]).astype(np.float16)
    ckpt = os.environ.get("CKPT", os.path.join(os.path.dirname(HERE), "checkpoints", "smolder_swa.ckpt"))
    out["has_pred"] = False
    if os.path.exists(ckpt):
        import torch
        from scipy.stats import rankdata
        from smolder.data.zarr_dual_datamodule import DualPatchConfig, DualWindowDataset
        from smolder.models.conv_lstm_lit_dual import ConvLSTMLitDual
        m = ConvLSTMLitDual.load_from_checkpoint(ckpt, map_location="cpu").eval()
        ds = DualWindowDataset(DualPatchConfig(
            zarr_paths=(daily_cube(2020),), stats_path="channel_stats_2015_2018.json",
            slow_cube_path="cube_slow_8day.zarr", day_offset=1826, patch_size=PATCH,
            samples_per_epoch=1, seed=0, deterministic=True, fire_history=True,
            fire_history_lags=(3, 4, 5), fire_history_distance=True))
        b = ds.sample_at(D + 1, y0, x0)      # last step forecasts days D+1..D+3
        with torch.no_grad():
            p = torch.sigmoid(m.forward_seq(b["x_slow"][None], b["x_fast"][None], b["x_cat"][None]))[0, -3:].numpy()
        pct = []
        for pi in p:
            r = np.full(pi.shape, np.nan, np.float32)
            r[land] = 100 * (rankdata(pi[land]) - 1) / max(land.sum() - 1, 1)
            pct.append(_block(r, "mean"))
        out["pred"] = np.stack(pct).astype(np.float16)
        out["has_pred"] = True
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **out)
    print("wrote", CACHE, "patch", y0, x0, "prediction" if out["has_pred"] else "target only")


# ----------------------------------------------------------------- drawing
FRAME_EC = "#2B2B2E"
LINE = "#3A3A3E"
FUSE = "#6B46C1"
STATIC = "#4A7A4A"
TIMES = "×"


def rect(ax, x, y, w, h, ec=FRAME_EC, fc="none", lw=1.2, z=1):
    ax.add_patch(Rectangle((x, y), w, h, ec=ec, fc=fc, lw=lw, zorder=z))


def arrow(ax, x1, y1, x2, y2, c=LINE, lw=1.4, z=6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                 lw=lw, color=c, zorder=z, shrinkA=0, shrinkB=0))


def stack(ax, frames, x, y, size, cmap, vmin, vmax, dx=1.3, dy=1.0, z0=3):
    """Time slices drawn back (oldest) to front (newest), offset diagonally."""
    n = len(frames)
    for i, fr in enumerate(frames):
        ox, oy = x + (n - 1 - i) * dx, y + (n - 1 - i) * dy
        ax.imshow(np.asarray(fr, np.float32), extent=[ox, ox + size, oy, oy + size], cmap=cmap,
                  vmin=vmin, vmax=vmax, interpolation="nearest", zorder=z0 + 2 * i, origin="upper")
        rect(ax, ox, oy, size, size, ec="#555558", lw=0.6, z=z0 + 2 * i + 1)
    return x + (size + (n - 1) * dx) / 2          # horizontal centre of the stack


def rng(a, lo=2, hi=98):
    v = np.asarray(a, np.float32)
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi))) if v.size else (0.0, 1.0)


def with_bad(name, bad="#D9D9DC"):
    cm = plt.get_cmap(name).copy()
    cm.set_bad(bad)
    return cm


def label(ax, x, y, text, **kw):
    kw.setdefault("fontsize", 8.4)
    kw.setdefault("color", INK)
    ax.text(x, y, text, ha="center", va="top", linespacing=1.25, **kw)


def draw():
    d = np.load(CACHE, allow_pickle=False)
    fig, ax = plt.subplots(figsize=(17.0, 8.9))
    ax.set_xlim(0, 170)
    ax.set_ylim(-5, 86)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    for tag, (x, w, title) in {"a": (1, 76, "Input"), "b": (79, 36, "ConvLSTM encoders"),
                               "c": (117, 25, "Fusion"), "d": (144, 25, "Output")}.items():
        rect(ax, x, 1, w, 83, lw=1.3, z=0)
        ax.text(x + 1.2, 81.6, f"({tag})  {title}", fontsize=11.5, fontweight="bold",
                color=INK, va="top", ha="left")

    # ---- (a) slow branch
    ax.text(3, 75.5, "Slow branch", fontsize=10.5, fontweight="bold", color=ACCENT, va="top")
    ax.text(3, 72.6, "144 days as 18 bins of 8 days", fontsize=8.8, color=MUTED, va="top")
    for i, (k, lab, cm) in enumerate((("lai", "Leaf area index", "Greens"),
                                      ("sm", "Soil moisture", "YlGnBu"),
                                      ("ppt", "Precipitation", "Blues"))):
        lo, hi = rng(d[k])
        c = stack(ax, d[k], 4 + i * 23.5, 51.5, 12.0, with_bad(cm), lo, hi)
        label(ax, c, 50.0, lab)

    # ---- (a) fast branch
    ax.text(3, 43.5, "Fast branch", fontsize=10.5, fontweight="bold", color=ACCENT2, va="top")
    ax.text(3, 40.6, "14 daily steps", fontsize=8.8, color=MUTED, va="top")
    for i, (k, lab, cm) in enumerate((("vpd", "Vapour pressure\ndeficit", "YlOrRd"),
                                      ("lst", "Land surface\ntemperature", "magma"),
                                      ("wind", "Wind speed", "cividis"))):
        lo, hi = rng(d[k])
        c = stack(ax, d[k], 4 + i * 18.0, 21.5, 10.2, with_bad(cm), lo, hi, dx=1.15, dy=0.9)
        label(ax, c, 20.0, lab)
    fire_cm = ListedColormap(["#F2F2F4", "#B2182B"])
    fire_cm.set_bad("#D9D9DC")
    fh = np.where(d["land"][None], np.asarray(d["fire_hist"], np.float32), np.nan)
    c = stack(ax, fh, 58.5, 21.5, 10.2, fire_cm, 0, 1, dx=1.15, dy=0.9)
    label(ax, c, 20.0, "Fire history\n(VIIRS)")

    # ---- (a) static layers
    ax.text(3, 10.6, "Static layers", fontsize=10.0, fontweight="bold", color=STATIC, va="top")
    ax.text(3, 7.7, "added to both branches", fontsize=8.8, color=MUTED, va="top")
    for i, (k, lab, cm) in enumerate((("agb", "Biomass", "YlGn"), ("landcover", "Land cover", "tab20"),
                                      ("koppen", "Climate zone", "tab20b"))):
        a = np.asarray(d[k], np.float32)
        if k != "agb":
            a = np.where(a > 0, a, np.nan)
        lo, hi = rng(a, 0, 100)
        x0 = 44 + i * 11.0
        ax.imshow(a, extent=[x0, x0 + 7.5, 3.0, 10.5], cmap=with_bad(cm), vmin=lo, vmax=hi,
                  interpolation="nearest", zorder=3)
        rect(ax, x0, 3.0, 7.5, 7.5, ec="#555558", lw=0.6, z=4)
        ax.text(x0 + 3.75, 11.4, lab, ha="center", fontsize=8.0, color=INK)

    # ---- (b) unrolled ConvLSTM chains
    xs = [82.0, 93.8, 105.6]
    for yc, col, steps in ((62.0, ACCENT, "18 steps"), (30.0, ACCENT2, "14 steps")):
        for j, xc in enumerate(xs):
            rect(ax, xc, yc - 4.2, 8.0, 8.4, ec=col, fc="white", lw=1.6, z=3)
            ax.text(xc + 4.0, yc + 0.9, "ConvLSTM", ha="center", va="center", fontsize=6.6,
                    color=col, fontweight="bold", zorder=4)
            ax.text(xc + 4.0, yc - 1.7, "cell", ha="center", va="center", fontsize=6.8, color=col, zorder=4)
            arrow(ax, xc + 4.0, yc - 9.5, xc + 4.0, yc - 4.3, c=col, lw=1.1)
            ax.text(xc + 4.0, yc - 10.4, ["x (t−2)", "x (t−1)", "x (t)"][j], ha="center",
                    va="top", fontsize=7.4, color=MUTED)
            if j < 2:
                arrow(ax, xc + 8.0, yc, xs[j + 1], yc, c=col, lw=1.3)
                ax.text(xc + 9.9, yc + 1.0, "h, c", ha="center", fontsize=6.8, color=MUTED)
        ax.text(97.8, yc + 6.4, f"{steps}, 5 {TIMES} 5 kernels, 64 channels", ha="center",
                fontsize=8.2, color=INK)
    arrow(ax, 77.0, 58.0, 81.8, 62.0, c=ACCENT)
    arrow(ax, 77.0, 28.0, 81.8, 30.0, c=ACCENT2)

    # ---- (c) fusion
    rect(ax, 120.0, 38.0, 19.0, 16.0, ec=FUSE, fc="white", lw=1.8, z=3)
    ax.text(129.5, 49.6, "Cross attention", ha="center", fontsize=9.6, fontweight="bold", color=FUSE)
    ax.text(129.5, 45.6, "4 heads, per pixel", ha="center", fontsize=8.2, color=INK)
    ax.text(129.5, 41.6, "softmax(Q Kᵀ / √d) V", ha="center", fontsize=8.4, color=INK)
    arrow(ax, 113.6, 62.0, 124.0, 54.2, c=ACCENT)
    ax.text(120.2, 61.2, "K, V", fontsize=8.4, color=ACCENT, fontweight="bold")
    ax.text(120.2, 58.4, "final slow state", fontsize=7.2, color=MUTED)
    arrow(ax, 113.6, 30.0, 124.0, 37.8, c=ACCENT2)
    ax.text(117.6, 35.6, "Q", fontsize=8.4, color=ACCENT2, fontweight="bold")
    ax.text(119.6, 31.4, "fast state,\nevery step", fontsize=7.2, color=MUTED, va="top", linespacing=1.2)
    rect(ax, 120.0, 14.0, 19.0, 10.0, ec=FRAME_EC, fc="white", lw=1.4, z=3)
    ax.text(129.5, 20.6, f"1 {TIMES} 1 convolution", ha="center", fontsize=8.8, fontweight="bold", color=INK)
    ax.text(129.5, 16.8, "and sigmoid", ha="center", fontsize=8.4, color=INK)
    arrow(ax, 129.5, 37.8, 129.5, 24.2)

    # ---- (d) output
    arrow(ax, 139.2, 19.0, 147.4, 40.0)
    if bool(d["has_pred"]):
        out = np.asarray(d["pred"], np.float32)
        cm_out, vmin, vmax, lab = with_bad("YlOrRd"), 0, 100, "Predicted fire risk\n(percentile within patch)"
    else:
        out = np.where(d["land"][None], np.asarray(d["target"], np.float32), np.nan)
        cm_out, vmin, vmax, lab = fire_cm, 0, 1, "Observed target\n(VIIRS fire)"
    c = stack(ax, out, 147.0, 40.0, 15.0, cm_out, vmin, vmax, dx=1.6, dy=1.3)
    label(ax, c, 37.6, lab, fontsize=9.0, fontweight="bold")
    label(ax, c, 31.0, "Fire on days D+1 to D+3,\none map per time step,\nlast three steps shown",
          fontsize=8.0, color=MUTED)

    ax.text(85, -1.2,
            f"All rasters are real data for one 384 {TIMES} 384 px patch (about 384 km across) on the "
            f"0.01° grid, issue day D = {str(d['date'])}. Stacks show time slices, oldest at the back. "
            "Grey marks ocean or missing data.",
            ha="center", va="top", fontsize=8.8, color=MUTED)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    extract() if "--extract" in sys.argv else draw()
