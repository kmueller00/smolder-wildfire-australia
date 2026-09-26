"""SMOLDER architecture overview: two input streams at different timescales,
each encoded by a ConvLSTM, fused by per-pixel cross-attention, producing a
risk map. Channel lists are in fig_inputs.png; sizes are in the README.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from style_smolder import ACCENT, ACCENT2, INK

# Same primary/secondary accents as every other figure in the repo (house
# style), plus two roles unique to this diagram (the fusion step, the output).
SLOW = ACCENT; FAST = ACCENT2; FUSE = "#6b46c1"; OUT = "#c53030"

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
ax.text(1.55, 4.97, "144 days  ·  8 day bins", fontsize=9.6, color=INK, ha="center")
ax.text(1.55, 4.62, "leaf area · soil moisture · rainfall", fontsize=8.8, color="#555", ha="center")
ax.text(1.55, 4.25, "fuel amount and dryness", fontsize=9.2, color=SLOW,
        ha="center", style="italic")

cube_stack(0.45, 2.05, 5, 1.5, 1.05, FAST)
ax.text(1.55, 1.64, "FAST branch", fontsize=11.5, fontweight="bold", color=FAST, ha="center")
ax.text(1.55, 1.27, "14 days  ·  daily", fontsize=9.6, color=INK, ha="center")
ax.text(1.55, 0.92, "vapour deficit · surface temp · wind", fontsize=8.8, color="#555", ha="center")
ax.text(1.55, 0.60, "+ fire detected up to the issue day", fontsize=8.8, color="#555", ha="center")
ax.text(1.55, 0.18, "fire weather and recent fire", fontsize=9.2, color=FAST,
        ha="center", style="italic")

# ---------------------------------------------------------------- encoders
ax.text(5.55, 7.95, "ENCODE  ·  ConvLSTM", fontsize=11.5, fontweight="bold",
        color=INK, ha="center")
for y, col, lab in ((5.75, SLOW, "slow"), (2.05, FAST, "fast")):
    box(4.35, y, 2.4, 1.05, "#ffffff", col, lw=2.2)
    ax.text(5.55, y + 0.80, "ConvLSTM", fontsize=11.5, fontweight="bold", color=col, ha="center")
    ax.text(5.55, y + 0.22, "convolution = space\nrecurrence = time", fontsize=8.4,
            color="#555", ha="center", linespacing=1.35)
    arrow(3.95, y + 0.52, 4.3, y + 0.52, c=col)

box(4.0, 3.55, 3.1, 1.0, "#f7fafc", "#718096", lw=1.5)
ax.text(5.55, 4.28, "static context, both branches", fontsize=9.0, fontweight="bold",
        color="#4a5568", ha="center")
ax.text(5.55, 3.68, "biomass · land mask · day of year\nland cover · climate zone",
        fontsize=8.2, color="#666", ha="center", linespacing=1.4)
arrow(5.55, 4.60, 5.55, 5.72, c="#718096", lw=1.6)
arrow(5.55, 3.50, 5.55, 3.13, c="#718096", lw=1.6)

# ---------------------------------------------------------------- fusion
ax.text(9.15, 7.95, "FUSE  ·  cross attention", fontsize=11.5, fontweight="bold",
        color=INK, ha="center")
box(8.0, 3.62, 2.3, 1.5, "#faf5ff", FUSE, lw=2.4)
ax.text(9.15, 4.72, "cross attention", fontsize=11.5, fontweight="bold", color=FUSE, ha="center")
ax.text(9.15, 3.98, "each fast step queries\nthe slow state,\nper pixel", fontsize=8.9,
        color="#555", ha="center", linespacing=1.45)

arrow(6.8, 6.27, 8.35, 5.18, c=SLOW)
arrow(6.8, 2.57, 8.35, 3.56, c=FAST)
ax.text(7.32, 5.92, "key / value", fontsize=8.6, color=SLOW, ha="center", rotation=-31)
ax.text(7.32, 3.18, "query", fontsize=8.6, color=FAST, ha="center", rotation=31)


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
ax.text(13.0, 2.42, "relative risk of fire in the next 3 days", fontsize=9.6,
        color=INK, ha="center", fontweight="bold")
ax.text(13.0, 2.06, "continental Australia · 3474 × 4110 grid", fontsize=8.6,
        color="#666", ha="center")

ax.text(7.6, -0.62,
        "SMOLDER  |  Slow-Memory Operator with Latent Dual-attention for Estimating fire Risk.  "
        "Fuel state is read over 144 days, fire weather over 14 days.",
        fontsize=9.3, color="#444", ha="center", style="italic")

fig.savefig("fig_smolder_architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
print("wrote fig_smolder_architecture.png")
