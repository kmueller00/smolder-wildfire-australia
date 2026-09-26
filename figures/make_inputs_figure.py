"""SMOLDER input factors at 1 km resolution -- the exact channel set of the
released checkpoint, verified against the training run's own startup log
(x_slow=(...,7), x_fast=(...,11)), not just the datamodule source.

No cartesian axes -- this is a labelled panel diagram, matching the visual
language of the architecture figure (same palette, same typography).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from style_smolder import ACCENT, ACCENT2, INK, MUTED

SLOW = ACCENT
FAST = ACCENT2
STATIC = "#4a7a4a"
SAMPLE = "#6b4a9e"

fig, ax = plt.subplots(figsize=(12.5, 7.7))
ax.set_xlim(0, 12.5); ax.set_ylim(0.65, 8.8); ax.axis("off")
fig.patch.set_facecolor("white")

ROW_H = 3.05   # both rows the same height -- the earlier version gave the
               # bottom row less room and its longest card's footnote ran
               # past the border


def card(x, y, w, h, color, title, subtitle, lines, sub=None):
    """Header carries only a short label; the descriptive phrase (e.g. '144
    days, 8-day bins') moves to a second header line, at a smaller size, so
    nothing has to be squeezed to fit one line at 12pt bold."""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                         fc="white", ec=color, lw=2.0, zorder=2)
    ax.add_patch(box)
    head_h = 0.72
    head = Rectangle((x, y + h - head_h), w, head_h, fc=color, ec="none", zorder=3)
    head.set_clip_path(box)
    ax.add_patch(head)
    ax.text(x + 0.28, y + h - 0.28, title, fontsize=12.5, fontweight="bold",
            color="white", va="top", ha="left", zorder=4)
    ax.text(x + 0.28, y + h - 0.58, subtitle, fontsize=9, color="white",
            va="top", ha="left", zorder=4, alpha=0.92)
    ty = y + h - head_h - 0.32
    for ln in lines:
        ax.text(x + 0.3, ty, ln, fontsize=9.5, color=INK, va="top", ha="left")
        ty -= 0.36
    if sub:
        ax.text(x + 0.3, y + 0.22, sub, fontsize=8.3, color=MUTED,
                va="bottom", ha="left", style="italic")


ax.text(6.25, 8.55, "SMOLDER input factors, at 1 km resolution",
        fontsize=16, fontweight="bold", color=INK, ha="center")
ax.text(6.25, 8.14,
        "Every channel actually read by the released checkpoint, confirmed against the training run's own startup log",
        fontsize=9.6, color=MUTED, ha="center", style="italic")

TOP_Y = 4.75
BOT_Y = TOP_Y - ROW_H - 0.35

card(0.3, TOP_Y, 3.85, ROW_H, SLOW, "SLOW branch", "144 days · 8-day bins",
     ["•  Leaf area index (LAI)",
      "•  Soil moisture",
      "•  Precipitation (8-day sum)"],
     sub="Fuel accumulation / drought state.\n18 steps into the slow ConvLSTM.")

card(4.35, TOP_Y, 3.85, ROW_H, FAST, "FAST branch", "14 days · daily",
     ["•  Vapour pressure deficit (VPD)",
      "•  Land-surface temperature",
      "•  Wind speed"],
     sub="Day-to-day fire-weather state.\n14 steps into the fast ConvLSTM.")

card(8.4, TOP_Y, 3.8, ROW_H, FAST, "FIRE HISTORY", "leakage-free · ends t−3",
     ["•  Fire at t−3, t−4, t−5",
      "   (3 binary lag channels)",
      "•  Distance-decay from recent",
      "   fire (exp(−px/5), 5 km scale)"],
     sub="Appended to the fast branch.\nDominant near-field spread signal.")

card(0.3, BOT_Y, 3.85, ROW_H, STATIC, "STATIC", "broadcast every time step",
     ["•  Above-ground biomass",
      "•  Land / ocean mask",
      "•  Day-of-year (sin, cos)"],
     sub="Constant-in-time fields the recurrent\nbranches see at every step.")

card(4.35, BOT_Y, 3.85, ROW_H, STATIC, "CATEGORICAL", "learned embeddings",
     ["•  Land cover class",
      "•  Köppen–Geiger climate zone"],
     sub="Small embedding tables, concatenated\ninto the fused representation.")

card(8.4, BOT_Y, 3.8, ROW_H, SAMPLE, "TRAINING-TIME ONLY", "not a model input",
     ["•  30% of patches forced to contain",
      "   a genuinely new-fire pixel",
      "•  30% of samples: fire-history",
      "   channels blanked to 0"],
     sub="Shapes which examples the model\nis trained on, not what it reads later.")

ax.text(6.25, BOT_Y - 0.35,
        "Grid: 3474 × 4110 px, EPSG:4326, 0.01° pixels (≈ 1 km) · target: fire occurrence in the next 3 days",
        fontsize=9.4, color=MUTED, ha="center", fontweight="bold")

fig.tight_layout()
fig.savefig("fig_inputs.png", dpi=300, bbox_inches="tight", facecolor="white")
print("wrote fig_inputs.png")
