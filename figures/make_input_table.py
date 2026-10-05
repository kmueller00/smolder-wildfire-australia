"""SMOLDER inputs and training setup as a table figure (fig_input_table.png).
Same layout as make_inputs_figure.py, in grey only."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from style_smolder import INK, MUTED

HEAD_BG = "#2B2B2E"
SEC_BG = "#5C5C61"
ROW_ALT = "#F3F3F4"

COLS = ("Input or setting", "Source", "Native resolution", "Use in the model")
WIDTHS = (0.22, 0.33, 0.16, 0.29)

SECTIONS = [
    ("Slow branch: 144 days as 18 bins of 8 days, ending on or before the issue day", [
        ("Leaf area index", "HiQ-LAI (Yan et al. 2024)", "500 m, 8-day", "mean to 1 km, 8-day mean"),
        ("Soil moisture index", "CSIRO SMIPS", "0.01°, daily", "8-day mean"),
        ("Precipitation", "ANUClimate 2.0", "0.01°, daily", "8-day sum"),
    ]),
    ("Fast branch: 14 daily steps", [
        ("Vapour pressure deficit", "BARRA-C2, at Tmax (Su et al. 2024)", "4.4 km, daily", "bilinear to 1 km"),
        ("Wind components u, v", "BARRA-C2, 10 m (Su et al. 2024)", "4.4 km, daily mean", "bilinear to 1 km"),
        ("10 m wind speed", "BARRA-C2 (Su et al. 2024)", "4.4 km, daily mean", "daily value"),
        ("Land surface temperature", "gap-filled MODIS (Zhang et al. 2022)", "1 km, daily", "daily value"),
        ("NDVI", "MODIS MOD09A1", "500 m, 8-day", "newest composite of each day"),
        ("Fire history", "VIIRS 375 m (Schroeder et al. 2014)", "375 m, daily", "3 windows of 3 days up to each day"),
        ("Distance to recent fire", "derived from fire history", "0.01°", "exp(−d / 5 km), per time step"),
        ("Fire radiative power", "NASA FIRMS, VIIRS S-NPP", "375 m, daily", "3-day sums: FRP, detections, night share"),
        ("Fuel age", "derived from fire history since 2015", "0.01°, daily", "days since last fire, capped at 3 years"),
    ]),
    ("Both branches: repeated at every time step", [
        ("Above-ground biomass", "ESA CCI Biomass", "100 m", "static"),
        ("Elevation, slope, aspect", "ETOPO1", "1 arc-minute", "static; aspect as sine and cosine"),
        ("Land mask", "SMIPS grid", "0.01°", "static"),
        ("Day of year", "calendar", "daily", "sine and cosine"),
        ("Land cover", "LC100 v3.0.1 (Buchhorn et al. 2020)", "100 m", "learned embedding, 6 dimensions"),
        ("Climate zone", "Köppen-Geiger (Beck et al. 2023)", "1 km", "learned embedding, 4 dimensions"),
    ]),
    ("Training", [
        ("Years", "2015 to 2018", "", "1309 valid issue days"),
        ("Target", "VIIRS fire, days D+1 to D+3", "375 m, daily", "per pixel, per fast time step"),
        ("Patches", "384 × 384 px", "", "2000 per epoch, half with ≥ 45 fire px"),
        ("New-fire sampling", "30 % of patches", "", "contain fire absent from the history"),
        ("Fire-history dropout", "30 % of samples", "", "fire history and FRP set to 0, fuel age to its cap"),
        ("Loss", "weighted BCE, soft labels", "", "fire weight annealed 100 to 20 over 8 epochs"),
    ]),
    ("Validation", [
        ("Year", "2019", "", "351 valid issue days"),
        ("Checkpoint selection", "average precision", "", "early stopping after epoch 20, patience 6"),
        ("Weights", "mean of the 3 best checkpoints", "", "epochs 23, 25 and 26"),
    ]),
]

ROW_H, SEC_H, HEAD_H = 0.52, 0.6, 0.66
n_rows = sum(len(r) for _, r in SECTIONS)
total_h = HEAD_H + len(SECTIONS) * SEC_H + n_rows * ROW_H
fig, ax = plt.subplots(figsize=(14.0, total_h * 0.62 + 0.6))
ax.set_xlim(0, 1)
ax.set_ylim(-total_h, 0.05)
ax.axis("off")
fig.patch.set_facecolor("white")

xs = [0.0]
for w in WIDTHS:
    xs.append(xs[-1] + w)
PAD = 0.008

y = 0.0
ax.add_patch(Rectangle((0, y - HEAD_H), 1, HEAD_H, fc=HEAD_BG, ec="none"))
for i, c in enumerate(COLS):
    ax.text(xs[i] + PAD, y - HEAD_H / 2, c, fontsize=10.2, color="white", va="center")
y -= HEAD_H

for title, rows in SECTIONS:
    ax.add_patch(Rectangle((0, y - SEC_H), 1, SEC_H, fc=SEC_BG, ec="none"))
    ax.text(PAD, y - SEC_H / 2, title, fontsize=10.0, color="white", va="center")
    y -= SEC_H
    for j, row in enumerate(rows):
        ax.add_patch(Rectangle((0, y - ROW_H), 1, ROW_H, fc=ROW_ALT if j % 2 else "white", ec="none"))
        for i, cell in enumerate(row):
            ax.text(xs[i] + PAD, y - ROW_H / 2, cell, fontsize=9.3, color=INK, va="center")
        y -= ROW_H
    ax.plot([0, 1], [y, y], color="#C8C8CC", lw=0.8)

ax.add_patch(Rectangle((0, y), 1, -y, fc="none", ec=HEAD_BG, lw=1.0))
for x in xs[1:-1]:
    ax.plot([x, x], [0, y], color="#D8D8DC", lw=0.7, zorder=0.5)

fig.savefig("fig_input_table.png", dpi=250, bbox_inches="tight", facecolor="white")
print("wrote fig_input_table.png")
