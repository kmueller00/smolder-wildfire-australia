"""SMOLDER input tensors per sample: slow branch, fast branch, the layers added
to both, and the tensors that enter the two ConvLSTM encoders.

Drawn at print size (full text width of the report, 6.9 in), so the font sizes
below are the sizes on the page.

Written by Cowork for the report; adapted in the repository: the cards are the
map cards of make_input_maps.py (CARD_DIR, card_<name>.png, the map frame of the
supplement maps without title and colour bar), no title above the panels, and the
inputs of the final model (LAI with the 31-day delay, BARRA-C2 VPD and maximum
air temperature, NDVI from the newest complete composite, downwind alignment).
They illustrate the variables; a sample is a patch of 384 x 384 pixels of each.

Usage:  SMOLDER_FONT_DIR=... CARD_DIR=... FIG_OUT=... python make_model_inputs.py
"""
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

from style_smolder import ACCENT, ACCENT2, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
CARD_DIR = os.environ.get("CARD_DIR", os.path.join(HERE, "data", "cards"))
OUT = os.path.join(os.environ.get("FIG_OUT", HERE), "fig_model_inputs.png")

STATIC = "#4A7A4A"
OCEAN = (188 / 255, 220 / 255, 244 / 255)
AX_BOX = (0, 0, 1124, 950)             # the card images (make_input_maps.py)
LON = (112.92, 153.86)
LAT = (-43.69, -9.05)


def load_map(name):
    return np.asarray(Image.open(os.path.join(CARD_DIR, f"card_{name}.png")).convert("RGB").crop(AX_BOX))


# ---- geometry, in inches on the printed page ----
W, H = 6.93, 4.10                       # no title row above the panels
CARD_W = 0.88
CARD_H = CARD_W * (AX_BOX[3] - AX_BOX[1]) / (AX_BOX[2] - AX_BOX[0])
DX, DY = 0.085, 0.27
LABEL_W = 0.84
GAP = 0.05
SLOTS = 5
COL_W = LABEL_W + GAP + CARD_W + (SLOTS - 1) * DX
Y_TOP = 3.12
FS_LABEL, FS_SUB, FS_HEAD, FS_DESC, FS_SHAPE = 6.9, 6.0, 9.2, 6.2, 7.0          # nothing below 6 pt

fig = plt.figure(figsize=(W, H), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")


def label(x, y, name, sub):
    ax.add_patch(FancyBboxPatch((x, y - 0.062), LABEL_W, 0.124, boxstyle="round,pad=0,rounding_size=0.03",
                                fc="#F4F4F6", ec="#6B6B6E", lw=0.5, zorder=6))
    ax.text(x + 0.035, y - 0.004, name, fontsize=FS_LABEL, color=INK, va="center", zorder=7)
    if sub:
        ax.text(x + 0.035, y - 0.125, sub, fontsize=FS_SUB, color=MUTED, va="center", zorder=7)


def stack(col_left, cards, slots=SLOTS):
    """Cards step down and to the left; the first card lies on top, at the upper right."""
    n = len(cards)
    x_stack = col_left + LABEL_W + GAP + (slots - n) * DX
    for i in reversed(range(n)):
        x = x_stack + (n - 1 - i) * DX
        top = Y_TOP - i * DY
        z = 10 + (n - i) * 3
        ax.add_patch(Rectangle((x - 0.012, top - CARD_H - 0.012), CARD_W + 0.024, CARD_H + 0.024,
                               fc="white", ec="#3A3A3E", lw=0.45, zorder=z))
        ax.imshow(cards[i][2], extent=[x, x + CARD_W, top - CARD_H, top], zorder=z + 1, interpolation="lanczos")
    for i, (name, sub, _) in enumerate(cards):
        x_card = x_stack + (n - 1 - i) * DX
        y = Y_TOP - i * DY - 0.10
        ax.plot([col_left + LABEL_W, x_card - 0.012], [y, y], color="#8C8C90", lw=0.5, zorder=5)
        label(col_left, y, name, sub)


def header(col_left, title, color, desc, shape):
    ax.text(col_left, 3.86, title, fontsize=FS_HEAD, fontweight="bold", color=color, va="center")
    for k, t in enumerate(desc):
        ax.text(col_left, 3.685 - 0.115 * k, t, fontsize=FS_DESC, color=MUTED, va="center")
    ax.text(col_left, 3.32, shape, fontsize=FS_SHAPE, color=INK, va="center", fontweight="bold")


MARGIN = 0.06
SEP = (W - 2 * MARGIN - 3 * COL_W) / 2
C_SLOW, C_STAT, C_FAST = MARGIN, MARGIN + COL_W + SEP, MARGIN + 2 * (COL_W + SEP)

slow = [("Soil moisture", "8-day mean", load_map("sm_mean")),
        ("Precipitation", "8-day sum", load_map("precip_annual")),
        ("Leaf area index", "500 m, 31-day delay", load_map("lai_mean"))]
static = [("Land mask", "in both tensors", load_map("landmask")),
          ("Biomass (AGB)", "map of the year before", load_map("agb_mean")),
          ("Climate zone", "embedded to 4", load_map("koppen_geiger")),
          ("Land cover", "embedded to 6", load_map("landcover"))]
fast = [("Wind", "speed, u and v", load_map("wind_mean")),
        ("VPD", "daily, BARRA-C2", load_map("vpd_mean")),
        ("Air temperature", "daily maximum", load_map("tmax_mean")),
        ("NDVI", "last complete composite", load_map("ndvi_mean")),
        ("Fire history", "4 channels", load_map("fire_days"))]

header(C_SLOW, "Slow-branch input", ACCENT,
       ["18 averages of 8 days (144 days)", "3 dynamic + 6 static + 2 day of year"], "(T, H, W, C) = (18, 384, 384, 11)")
header(C_STAT, "Added to both branches", STATIC,
       ["repeated at every time step", "6 static layers and the day of year",
        "categorical input: land cover, climate zone"],
       "(H, W, C) = (384, 384, 2)")
header(C_FAST, "Fast-branch input", ACCENT2,
       ["14 daily steps", "6 dynamic + 4 fire history + 3 fire power + 1 fuel age",
        "+ 1 downwind alignment + 6 static + 2 day of year"], "(T, H, W, C) = (14, 384, 384, 23)")
stack(C_SLOW, slow)
stack(C_STAT, static)
stack(C_FAST, fast)

# layers without a map card
def row(col_left, i, name, sub):
    label(col_left, Y_TOP - i * DY - 0.10, name, sub)


row(C_FAST, 5, "Fire power (FRP)", "3 channels")
row(C_FAST, 6, "Fuel age", "days since last fire")
row(C_FAST, 7, "Wind vs. fire", "downwind alignment")
row(C_STAT, 4, "Terrain", "4 channels")

# day of year: no map, a sine and a cosine over the year
y = Y_TOP - 5 * DY - 0.10
label(C_STAT, y, "Day of year", "sine and cosine")
tt = np.linspace(0, 1, 200)
for fn, ls in ((np.sin, "-"), (np.cos, "--")):
    ax.plot(C_STAT + 0.10 + 0.52 * tt, y - 0.30 + 0.07 * fn(2 * np.pi * tt), ls, color=STATIC, lw=0.9, zorder=6)
ax.text(C_STAT + 0.10, y - 0.43, "Jan", fontsize=6, color=MUTED, ha="center")
ax.text(C_STAT + 0.62, y - 0.43, "Dec", fontsize=6, color=MUTED, ha="center")

BOX_W, BOX_H, BOX_Y = 2.75, 0.62, 0.10


def box(xc, color, title, shape, parts):
    ax.add_patch(Rectangle((xc - BOX_W / 2, BOX_Y), BOX_W, BOX_H, fc="white", ec=color, lw=1.2, zorder=6))
    ax.text(xc, BOX_Y + 0.47, title, fontsize=7.4, color=color, fontweight="bold", ha="center", va="center", zorder=7)
    ax.text(xc, BOX_Y + 0.295, shape, fontsize=9.4, color=INK, fontweight="bold", ha="center", va="center", zorder=7)
    ax.text(xc, BOX_Y + 0.115, parts, fontsize=6, color=MUTED, ha="center", va="center", zorder=7)


X_SLOW_BOX, X_FAST_BOX = W / 2 - BOX_W / 2 - 0.12, W / 2 + BOX_W / 2 + 0.12
box(X_SLOW_BOX, ACCENT, "Input of the slow ConvLSTM encoder", "(18, 384, 384, 21)",
    "11 channels + 10 embedded (land cover 6, climate zone 4)")
box(X_FAST_BOX, ACCENT2, "Input of the fast ConvLSTM encoder", "(14, 384, 384, 33)",
    "23 channels + 10 embedded (land cover 6, climate zone 4)")


def arrow(p0, p1, color):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, lw=1.1, color=color, zorder=8,
                                 shrinkA=0, shrinkB=0))


TOP = BOX_Y + BOX_H + 0.02
x_s = C_SLOW + LABEL_W + GAP + DX + CARD_W / 2
x_f = C_FAST + LABEL_W + GAP + CARD_W / 2 + 1.5 * DX
arrow((x_s, 1.18), (x_s, TOP), ACCENT)
arrow((x_f, 1.02), (x_f, TOP), ACCENT2)
x_m = W / 2
arrow((x_m, 1.0), (x_m - 0.55, TOP), STATIC)
arrow((x_m, 1.0), (x_m + 0.55, TOP), STATIC)


ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal")
fig.savefig(OUT, dpi=450, facecolor="white")
print("wrote", OUT, "column width", round(COL_W, 2), "sep", round(SEP, 2))
