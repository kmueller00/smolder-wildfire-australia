"""SMOLDER inputs and training setup as a table figure: slow branch, fast
branch, layers shared by both branches, training and validation."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from style_smolder import ACCENT, ACCENT2, INK, MUTED

STATIC = "#4A7A4A"
TRAIN = "#6B46C1"
VALID = "#5A5A60"
HEAD_BG = "#2B2B2E"
ROW_ALT = "#F1F1F3"

COLS = ("Input or setting", "Source", "Native resolution", "Use in the model")
WIDTHS = (0.24, 0.31, 0.16, 0.29)

SECTIONS = [
    ("Slow branch  (144 days as 18 bins of 8 days)", ACCENT, [
        ("Leaf area index", "HiQ-LAI, reprocessed MODIS", "8-day", "8-day mean"),
        ("Soil moisture index", "CSIRO SMIPS", "0.01°, daily", "8-day mean"),
        ("Precipitation", "ANUClimate 2.0", "0.01°, daily", "8-day sum"),
    ]),
    ("Fast branch  (14 daily steps)", ACCENT2, [
        ("Vapour pressure deficit", "ERA5, at daily maximum temperature", "0.25°, daily", "daily value"),
        ("Land surface temperature", "gap-filled MODIS (Zhang et al. 2022)", "1 km, daily", "daily value"),
        ("10 m wind speed", "BARRA2 reanalysis", "about 12 km, daily", "daily value"),
        ("Fire history", "VIIRS 375 m active fire", "375 m, daily", "3 windows of 3 days up to each day"),
        ("Distance to recent fire", "derived from fire history", "0.01°", "exp(−d / 5 km), per time step"),
    ]),
    ("Both branches  (repeated at every time step)", STATIC, [
        ("Above-ground biomass", "ESA CCI Biomass", "100 m", "static"),
        ("Land mask", "SMIPS grid", "0.01°", "static"),
        ("Day of year", "calendar", "daily", "sine and cosine"),
        ("Land cover", "Copernicus LC100 v3.0.1", "100 m", "learned embedding, 6 dimensions"),
        ("Climate zone", "Köppen-Geiger 1991 to 2020", "1 km", "learned embedding, 4 dimensions"),
    ]),
    ("Training", TRAIN, [
        ("Years", "2015 to 2018", "", "1309 valid issue days"),
        ("Target", "VIIRS fire on days D+1 to D+3", "375 m, daily", "per pixel, per fast time step"),
        ("Patches", "384 × 384 px", "", "2000 per epoch, half with ≥ 45 fire px"),
        ("New-fire sampling", "30 % of patches", "", "contain fire absent from the history"),
        ("Fire-history dropout", "30 % of samples", "", "fire-history channels set to 0"),
        ("Loss", "weighted BCE, soft labels", "", "fire weight annealed 100 to 20 over 8 epochs"),
    ]),
    ("Validation", VALID, [
        ("Year", "2019", "", "351 valid issue days"),
        ("Checkpoint selection", "average precision", "", "early stopping, patience 6 epochs"),
        ("Released weights", "mean of the 3 best checkpoints", "", "epochs 13, 14 and 18"),
    ]),
]

ROW_H, SEC_H, HEAD_H = 0.52, 0.62, 0.66
n_rows = sum(len(r) for _, _, r in SECTIONS)
total_h = HEAD_H + len(SECTIONS) * SEC_H + n_rows * ROW_H
fig, ax = plt.subplots(figsize=(14.0, total_h * 0.62 + 1.3))
ax.set_xlim(0, 1)
ax.set_ylim(-total_h, 0.9)
ax.axis("off")
fig.patch.set_facecolor("white")

xs = [0.0]
for w in WIDTHS:
    xs.append(xs[-1] + w)
PAD = 0.008

ax.text(0.0, 0.62, "SMOLDER inputs and training setup", fontsize=15, fontweight="bold",
        color=INK, va="center")
ax.text(0.0, 0.22, "All fields resampled to the SMIPS grid: 3474 × 4110 px, 0.01° (about 1 km), "
        "EPSG:4326. The test year 2020 is used only for evaluation.", fontsize=9.4, color=MUTED, va="center")

y = 0.0
ax.add_patch(Rectangle((0, y - HEAD_H), 1, HEAD_H, fc=HEAD_BG, ec="none"))
for i, c in enumerate(COLS):
    ax.text(xs[i] + PAD, y - HEAD_H / 2, c, fontsize=10.2, fontweight="bold", color="white", va="center")
y -= HEAD_H

for title, color, rows in SECTIONS:
    ax.add_patch(Rectangle((0, y - SEC_H), 1, SEC_H, fc=color, ec="none"))
    ax.text(PAD, y - SEC_H / 2, title, fontsize=10.4, fontweight="bold", color="white", va="center")
    y -= SEC_H
    for j, row in enumerate(rows):
        ax.add_patch(Rectangle((0, y - ROW_H), 1, ROW_H, fc=ROW_ALT if j % 2 else "white", ec="none"))
        for i, cell in enumerate(row):
            ax.text(xs[i] + PAD, y - ROW_H / 2, cell, fontsize=9.3, color=INK, va="center",
                    fontweight="bold" if i == 0 else "normal")
        y -= ROW_H
    ax.plot([0, 1], [y, y], color="#C8C8CC", lw=0.8)

ax.add_patch(Rectangle((0, y), 1, -y, fc="none", ec=HEAD_BG, lw=1.2))
for x in xs[1:-1]:
    ax.plot([x, x], [0, y], color="#D8D8DC", lw=0.7, zorder=0.5)

fig.savefig("fig_inputs.png", dpi=250, bbox_inches="tight", facecolor="white")
print("wrote fig_inputs.png")
