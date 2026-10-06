"""SMOLDER architecture figure.

Two steps:
  python make_smolder_architecture.py --extract   reads the data cubes (SMOLDER_DATA)
                                                  and, if CKPT exists, the model;
                                                  writes data/architecture_patch.npz
  python make_smolder_architecture.py             draws the figure from that file

The figure shows the two ConvLSTM encoders, the fusion and the output head
with the tensor shape at every stage; the inputs themselves are shown in the
input-tensor figure. The output stack is real data for a fire-active
384 x 384 px window on the issue day 2020-11-15, one map per fast time step:
the model's risk maps when a checkpoint was available at extraction,
otherwise the observed target for the same steps.
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
CACHE = os.environ.get("ARCH_CACHE", os.path.join(HERE, "data", "architecture_patch.npz"))
OUT = os.path.join(os.environ.get("FIG_OUT", HERE), "fig_smolder_architecture.png")
PATCH, THUMB = 384, 128
ISSUE_DAY = 319                      # 2020-11-15, index in the 2020 cube


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

    names = list(g.attrs.get("channels") or g.attrs["dyn_vars"])   # archived cube | full cube
    fast_days = np.arange(D - 13, D + 1)
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
    bins = np.arange(b_end - 18, b_end)
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

    steps = list(range(D - 13, D + 1))
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
            fire_history_lags=(3, 4, 5), fire_history_distance=True,
        use_elevation=os.environ.get("USE_ELEVATION", "0") == "1",       # the other inputs are read
        use_slope_aspect=os.environ.get("USE_SLOPE_ASPECT", "0") == "1"))  # from the environment
        b = ds.sample_at(D + 1, y0, x0)      # last step forecasts days D+1..D+3
        with torch.no_grad():
            p = torch.sigmoid(m.forward_seq(b["x_slow"][None], b["x_fast"][None], b["x_cat"][None]))[0].numpy()
        pct = []
        for pi in p:
            r = np.full(pi.shape, np.nan, np.float32)
            r[land] = 100 * (rankdata(pi[land]) - 1) / max(land.sum() - 1, 1)
            pct.append(_block(r, "mean"))
        out["pred"] = np.stack(pct).astype(np.float16)
        out["has_pred"] = True
        emb = int(sum(e.embedding_dim for e in (m.lc_emb, m.kg_emb)))          # land cover + climate zone
        out["c_slow"] = int(b["x_slow"].shape[-1]) + emb                         # ConvLSTM input channels
        out["c_fast"] = int(b["x_fast"].shape[-1]) + emb
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


OCEAN = "#CFE0EC"
COAST = "#1F1F22"
LAND = None                     # set in draw(): land mask at thumbnail resolution


def coast(ax, x, y, size, z):
    ax.contour(LAND.astype(float), levels=[0.5], colors=COAST, linewidths=0.7, zorder=z,
               extent=[x, x + size, y + size, y], origin="upper")


def stack(ax, frames, x, y, size, cmap, vmin, vmax, dx=None, dy=None, z0=3):
    """All time slices, oldest at the back, newest at the front. Offsets shrink
    with the number of slices so a stack's footprint stays about the same."""
    n = len(frames)
    dx = (0.38 * size) / max(n - 1, 1) if dx is None else dx
    dy = 0.55 * dx if dy is None else dy
    for i, fr in enumerate(frames):
        ox, oy = x + (n - 1 - i) * dx, y + (n - 1 - i) * dy
        z = z0 + 3 * i
        ax.imshow(np.asarray(fr, np.float32), extent=[ox, ox + size, oy, oy + size], cmap=cmap,
                  vmin=vmin, vmax=vmax, interpolation="nearest", zorder=z, origin="upper")
        rect(ax, ox, oy, size, size, ec="#555558", lw=0.45, z=z + 1)
    coast(ax, x, y, size, z0 + 3 * n)
    ax.text(x + size + (n - 1) * dx + 0.4, y + size + (n - 1) * dy - 0.2, f"{n}", fontsize=7.2,
            color=MUTED, ha="left", va="top")
    return x + (size + (n - 1) * dx) / 2          # horizontal centre of the stack


def rng(a, lo=2, hi=98):
    v = np.asarray(a, np.float32)
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi))) if v.size else (0.0, 1.0)


def with_bad(name, bad=OCEAN):
    cm = plt.get_cmap(name).copy()
    cm.set_bad(bad)
    return cm


def label(ax, x, y, text, **kw):
    kw.setdefault("fontsize", 8.4)
    kw.setdefault("color", INK)
    ax.text(x, y, text, ha="center", va="top", linespacing=1.25, **kw)


def box(ax, x, y, w, h, ec, lines, lw=1.6):
    """Plain rectangle with centred text lines given as (text, fontsize, bold, colour)."""
    rect(ax, x, y, w, h, ec=ec, fc="white", lw=lw, z=3)
    n = len(lines)
    for i, (t, fs, bold, col) in enumerate(lines):
        yy = y + h / 2 + (n - 1) / 2 * 3.0 - i * 3.0
        ax.text(x + w / 2, yy, t, ha="center", va="center", fontsize=fs, color=col,
                fontweight="bold" if bold else "normal", zorder=4)


def path(ax, pts, c, lw=1.4):
    """Polyline with an arrow head on the last segment."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-2],), ys[:-1] + (ys[-2],), color=c, lw=lw, zorder=6, solid_capstyle="butt")
    arrow(ax, xs[-2], ys[-2], xs[-1], ys[-1], c=c, lw=lw)


def draw():
    global LAND
    d = np.load(CACHE, allow_pickle=False)
    LAND = np.asarray(d["land"])
    cs = int(d["c_slow"]) if "c_slow" in d else 17
    cf = int(d["c_fast"]) if "c_fast" in d else 21
    fig, ax = plt.subplots(figsize=(12.6, 7.4))
    ax.set_xlim(0, 126)
    ax.set_ylim(7, 81)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    for tag, (x, w, title) in {"a": (23, 40, "ConvLSTM encoders"), "b": (65, 33, "Fusion"),
                               "c": (100, 25, "Output")}.items():
        rect(ax, x, 8, w, 72, lw=1.3, z=0)
        ax.text(x + 1.2, 78.6, f"({tag})  {title}", fontsize=11.5, fontweight="bold",
                color=INK, va="top", ha="left")
    ax.text(1.0, 78.6, "Inputs", fontsize=11.5, fontweight="bold", color=INK, va="top", ha="left")

    # ---- inputs and (a) unrolled ConvLSTM chains
    xs = [27.0, 38.8, 50.6]
    chains = ((62.0, ACCENT, "Slow-branch input", f"(18, 384, 384, {cs})", "18 steps of 8-day averages",
               "18 steps, 5 × 5 kernels, 64 channels"),
              (30.0, ACCENT2, "Fast-branch input", f"(14, 384, 384, {cf})", "14 daily steps",
               "14 steps, 5 × 5 kernels, 64 channels"))
    for yc, col, name, shape, sub, steps in chains:
        box(ax, 1.0, yc - 17.0, 20.0, 13.0, col, [(name, 8.8, True, col), (shape, 8.6, False, INK),
                                                  (sub, 7.6, False, INK)])
        feed = yc - 10.5
        ax.plot([21.0, xs[-1] + 4.0], [feed, feed], color=col, lw=1.1, zorder=5)
        for j, xc in enumerate(xs):
            rect(ax, xc, yc - 4.2, 8.0, 8.4, ec=col, fc="white", lw=1.6, z=3)
            ax.text(xc + 4.0, yc + 0.9, "ConvLSTM", ha="center", va="center", fontsize=6.6,
                    color=col, fontweight="bold", zorder=4)
            ax.text(xc + 4.0, yc - 1.7, "cell", ha="center", va="center", fontsize=6.8, color=col, zorder=4)
            arrow(ax, xc + 4.0, feed, xc + 4.0, yc - 4.3, c=col, lw=1.1)
            ax.text(xc + 4.6, feed + 1.0, ["x (t−2)", "x (t−1)", "x (t)"][j], ha="left",
                    va="bottom", fontsize=7.0, color=INK)
            if j < 2:
                arrow(ax, xc + 8.0, yc, xs[j + 1], yc, c=col, lw=1.3)
                ax.text(xc + 9.9, yc + 1.0, "h, c", ha="center", fontsize=6.8, color=INK)
        ax.text(42.8, yc + 6.4, steps, ha="center", fontsize=8.2, color=INK)
    ax.text(43.0, 70.6, "final hidden state (384, 384, 64)", ha="center", fontsize=7.8, color=ACCENT)
    ax.text(43.0, 15.2, "hidden state at every step (14, 384, 384, 64)", ha="center", fontsize=7.8,
            color=ACCENT2)

    # ---- (b) fusion: attention, concatenation, output head
    bx, bw = 68.5, 26.0
    box(ax, bx, 50.0, bw, 15.0, FUSE, [("Cross-attention", 9.2, True, FUSE), ("4 heads, per pixel", 8.0, False, INK),
                                        ("+ residual, layer norm", 8.0, False, INK)], lw=1.8)
    box(ax, bx, 34.0, bw, 10.5, FUSE, [("Concatenate", 9.2, True, FUSE), ("(14, 384, 384, 128)", 8.4, False, INK)],
        lw=1.8)
    box(ax, bx, 17.0, bw, 10.5, INK, [(f"5 {TIMES} 5 conv, sigmoid", 9.2, True, INK),
                                       ("(14, 384, 384)", 8.4, False, INK)], lw=1.8)
    arrow(ax, bx + bw / 2, 50.0, bx + bw / 2, 44.6)
    arrow(ax, bx + bw / 2, 34.0, bx + bw / 2, 27.6)
    ax.text(bx + bw / 2, 15.0, "applied at every fast step;\nthe last step is the forecast",
            ha="center", va="top", fontsize=7.8, color=INK, linespacing=1.25)
    # slow state: key and value of the attention, and the second half of the concatenation
    path(ax, [(58.6, 62.0), (bx, 62.0)], ACCENT)
    path(ax, [(66.0, 62.0), (66.0, 68.5), (96.6, 68.5), (96.6, 39.25), (bx + bw, 39.25)], ACCENT, lw=1.2)
    ax.text(60.8, 62.8, "K, V", ha="center", fontsize=8.2, color=ACCENT, fontweight="bold")
    # fast state: query
    path(ax, [(58.6, 30.0), (64.0, 30.0), (64.0, 54.0), (bx, 54.0)], ACCENT2)
    ax.text(60.8, 30.8, "Q", ha="center", fontsize=8.2, color=ACCENT2, fontweight="bold")

    # ---- (c) output
    arrow(ax, bx + bw, 25.0, 103.0, 25.0)
    if bool(d["has_pred"]):
        out = np.asarray(d["pred"], np.float32)
        cm_out, vmin, vmax, lab = with_bad("YlOrRd"), 0, 100, "Predicted fire risk\n(percentile within patch)"
    else:
        out = np.where(d["land"][None], np.asarray(d["target"], np.float32), np.nan)
        cm_out, vmin, vmax, lab = ListedColormap(["#F2F2F4", "#B2182B"]), 0, 1, "Observed target\n(VIIRS fire)"
    c = stack(ax, out, 103.5, 20.0, 13.0, cm_out, vmin, vmax)
    label(ax, c, 18.4, lab, fontsize=9.0, fontweight="bold")
    label(ax, c, 13.0, "Fire on days D+1 to D+3,\none map per fast time step", fontsize=8.0)

    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    extract() if "--extract" in sys.argv else draw()
