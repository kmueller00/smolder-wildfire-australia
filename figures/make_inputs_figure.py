"""SMOLDER input channels at 1 km resolution (x_slow: 7 channels, x_fast: 11).
A labelled panel diagram (no data axes), in the palette of the other figures.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from style_smolder import ACCENT, ACCENT2, INK, MUTED

SLOW = ACCENT
FAST = ACCENT2
STATIC = "#4a7a4a"
SAMPLE = "#6b4a9e"

fig, ax = plt.subplots(figsize=(12.5, 7.7))
ax.set_xlim(0, 12.5); ax.set_ylim(0.65, 8.8); ax.axis("off")
fig.patch.set_facecolor("white")

ROW_H = 3.05


def card(x, y, w, h, color, title, subtitle, lines, sub=None):
    box = Rectangle((x, y), w, h, fc="white", ec=color, lw=1.6, zorder=2)
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


ax.text(6.25, 8.55, "SMOLDER input channels, 1 km resolution",
        fontsize=16, fontweight="bold", color=INK, ha="center")
ax.text(6.25, 8.14,
        "Every channel read by the released model; all fields on the SMIPS grid (0.01°)",
        fontsize=9.6, color=MUTED, ha="center", style="italic")

TOP_Y = 4.75
BOT_Y = TOP_Y - ROW_H - 0.35

card(0.3, TOP_Y, 3.85, ROW_H, SLOW, "SLOW branch", "144 days in 8-day bins",
     ["Leaf area index (HiQ-LAI)",
      "Soil moisture (SMIPS)",
      "Precipitation, 8-day sum"],
     sub="Fuel amount and dryness.\n18 steps into the slow ConvLSTM.")

card(4.35, TOP_Y, 3.85, ROW_H, FAST, "FAST branch", "14 daily steps",
     ["Vapour pressure deficit (ERA5)",
      "Land-surface temperature",
      "10 m wind speed (BARRA2)"],
     sub="Fire weather.\n14 steps into the fast ConvLSTM.")

card(8.4, TOP_Y, 3.8, ROW_H, FAST, "FIRE HISTORY", "VIIRS, fast branch",
     ["Fire in 3 overlapping 3-day windows",
      "Distance decay, exp(\u2212d / 5 km)"],
     sub="Recomputed for every time step;\nends before that step's target window.")

card(0.3, BOT_Y, 3.85, ROW_H, STATIC, "STATIC", "added to both branches",
     ["Above-ground biomass (ESA CCI)",
      "Land mask",
      "Day of year (sin, cos)"],
     sub="Repeated at every time step of\nboth ConvLSTM inputs.")

card(4.35, BOT_Y, 3.85, ROW_H, STATIC, "CATEGORICAL", "learned embeddings",
     ["Land cover (PROBA-V LC100)",
      "Köppen-Geiger climate zone"],
     sub="Embeddings of 6 and 4 dimensions,\nadded to both branches' inputs.")

card(8.4, BOT_Y, 3.8, ROW_H, SAMPLE, "TRAINING ONLY", "not a model input",
     ["30% of patches contain new fire",
      "30% of samples: fire history set to 0"],
     sub="Shape which examples the model is\ntrained on, not what it reads later.")

ax.text(6.25, BOT_Y - 0.35,
        "Grid 3474 × 4110 px, EPSG:4326. Target: VIIRS fire detected on any of the 3 days after the issue day",
        fontsize=9.4, color=MUTED, ha="center", fontweight="bold")

fig.tight_layout()
fig.savefig("fig_inputs.png", dpi=300, bbox_inches="tight", facecolor="white")
print("wrote fig_inputs.png")
