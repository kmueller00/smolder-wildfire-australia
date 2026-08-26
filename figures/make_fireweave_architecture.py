"""Clean, self-explanatory architecture overview for FireWeave.

Design goal: a reader who has never seen this project should understand the
whole idea in about ten seconds -- two input streams at different timescales,
each read by a ConvLSTM, fused by attention, producing a risk map. Everything
that is not needed for that story (channel counts, layer widths, tensor shapes)
is deliberately left out; those live in the README table instead.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

SLOW = "#2b6cb0"; FAST = "#dd6b20"; FUSE = "#6b46c1"; OUT = "#c53030"; INK = "#1a202c"

fig, ax = plt.subplots(figsize=(14.5, 8.0))
ax.set_xlim(0, 15.2); ax.set_ylim(-0.75, 8.35); ax.axis("off")


def box(x, y, w, h, fc, ec, lw=1.8, r=0.12, alpha=1.0, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z))


def arrow(x1, y1, x2, y2, c=INK, lw=2.2, style="-|>", z=4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=17, lw=lw, color=c, zorder=z,
                                 shrinkA=2, shrinkB=2))


def cube_stack(x, y, n, w, h, color, dx=0.085, dy=0.055):
    """Little stack of map slices = a time series of rasters."""
    for i in range(n - 1, -1, -1):
        a = 0.30 + 0.70 * (1 - i / max(n - 1, 1))
        ax.add_patch(Rectangle((x + i * dx, y + i * dy), w, h, fc=color, ec="white",
                               lw=1.1, alpha=a, zorder=3 + (n - i)))


# ---------------------------------------------------------------- inputs
ax.text(1.55, 7.95, "INPUT  ·  two timescales", fontsize=11.5, fontweight="bold",
        color=INK, ha="center")

cube_stack(0.45, 5.75, 5, 1.5, 1.05, SLOW)
ax.text(1.55, 5.34, "SLOW branch", fontsize=11.5, fontweight="bold", color=SLOW, ha="center")
ax.text(1.55, 4.97, "144 days  ·  8-day bins", fontsize=9.6, color=INK, ha="center")
ax.text(1.55, 4.62, "leaf area · soil moisture · rainfall", fontsize=8.8, color="#555", ha="center")
ax.text(1.55, 4.02, "“how much fuel is there,\nand how dry is it?”", fontsize=9.4, color=SLOW,
        ha="center", style="italic", linespacing=1.5)

cube_stack(0.45, 2.05, 5, 1.5, 1.05, FAST)
ax.text(1.55, 1.64, "FAST branch", fontsize=11.5, fontweight="bold", color=FAST, ha="center")
ax.text(1.55, 1.27, "14 days  ·  daily", fontsize=9.6, color=INK, ha="center")
ax.text(1.55, 0.92, "vapour deficit · surface temp · wind", fontsize=8.8, color="#555", ha="center")
ax.text(1.55, 0.32, "“will it ignite\nand run today?”", fontsize=9.4, color=FAST,
        ha="center", style="italic", linespacing=1.5)

# ---------------------------------------------------------------- encoders
ax.text(5.55, 7.95, "ENCODE  ·  ConvLSTM", fontsize=11.5, fontweight="bold",
        color=INK, ha="center")
for y, col, lab in ((5.75, SLOW, "slow"), (2.05, FAST, "fast")):
    box(4.35, y, 2.4, 1.05, "#ffffff", col, lw=2.2)
    ax.text(5.55, y + 0.70, "ConvLSTM", fontsize=11.5, fontweight="bold", color=col, ha="center")
    ax.text(5.55, y + 0.36, "convolution = space\nrecurrence = time", fontsize=8.6,
            color="#555", ha="center", linespacing=1.35)
    arrow(3.95, y + 0.52, 4.3, y + 0.52, c=col)

ax.text(5.55, 4.02, "each cell sees its neighbours\nAND remembers the past",
        fontsize=9.2, color="#555", ha="center", style="italic", linespacing=1.5)

# ---------------------------------------------------------------- fusion
ax.text(9.15, 7.95, "FUSE  ·  cross-attention", fontsize=11.5, fontweight="bold",
        color=INK, ha="center")
box(8.0, 3.62, 2.3, 1.5, "#faf5ff", FUSE, lw=2.4)
ax.text(9.15, 4.72, "cross-attention", fontsize=11.5, fontweight="bold", color=FUSE, ha="center")
ax.text(9.15, 3.98, "today's weather asks:\n“which fuel signal\nmatters here?”", fontsize=8.9,
        color="#555", ha="center", linespacing=1.45)

arrow(6.8, 6.27, 8.35, 5.18, c=SLOW)
arrow(6.8, 2.57, 8.35, 3.56, c=FAST)
ax.text(7.32, 5.92, "key / value", fontsize=8.6, color=SLOW, ha="center", rotation=-31)
ax.text(7.32, 3.18, "query", fontsize=8.6, color=FAST, ha="center", rotation=31)

# static context
box(8.0, 1.90, 2.3, 0.95, "#f7fafc", "#718096", lw=1.5)
ax.text(9.15, 2.50, "static context", fontsize=9.6, fontweight="bold", color="#4a5568", ha="center")
ax.text(9.15, 2.10, "biomass · land cover\nclimate zone · fire history", fontsize=8.3,
        color="#666", ha="center", linespacing=1.4)
arrow(9.15, 2.90, 9.15, 3.57, c="#718096", lw=1.7)

# ---------------------------------------------------------------- output
ax.text(13.0, 7.95, "PREDICT", fontsize=11.5, fontweight="bold", color=INK, ha="center")
arrow(10.35, 4.37, 11.35, 4.37, c=FUSE)

rng = np.random.default_rng(3)
risk = rng.random((26, 26)) * 0.28
yy, xx = np.mgrid[0:26, 0:26]
for cy, cx, s, a in ((8, 9, 3.1, 0.95), (17, 18, 2.4, 0.85), (13, 6, 1.9, 0.6)):
    risk += a * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s ** 2))
ax.imshow(np.clip(risk, 0, 1), extent=[11.45, 14.55, 2.82, 5.92], cmap="YlOrRd",
          origin="lower", zorder=3, interpolation="bilinear")
ax.add_patch(Rectangle((11.45, 2.82), 3.1, 3.1, fc="none", ec=OUT, lw=2.4, zorder=5))
ax.text(13.0, 6.15, "fire risk per 1 km pixel", fontsize=10.6, fontweight="bold",
        color=OUT, ha="center")
ax.text(13.0, 2.42, "probability of fire in the NEXT 3 DAYS", fontsize=9.6,
        color=INK, ha="center", fontweight="bold")
ax.text(13.0, 2.06, "continental Australia · 3474 × 4110 grid", fontsize=8.6,
        color="#666", ha="center")

ax.text(7.6, -0.62,
        "FireWeave — cross-attention weaves the slow fuel signal together with fast fire weather.  The two branches are read at "
        "different rates because fuel dries over months while fire weather turns over hours.",
        fontsize=9.3, color="#444", ha="center", style="italic")

fig.savefig("fig_fireweave_architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
print("wrote figures/fig_fireweave_architecture.png")
