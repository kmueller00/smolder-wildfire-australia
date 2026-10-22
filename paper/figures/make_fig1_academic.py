"""Fig. 1 'SMOLDER in brief': sharp corners, muted palette, Arial like the other figures of the paper.
usage: python fig1_academic.py <architecture_figure.png> <out_stem>   (writes <out_stem>.png/.pdf and <out_stem>_risk.png)"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image
import numpy as np

ARCH, OUT = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "mathtext.fontset": "custom",
                     "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                     "pdf.fonttype": 42})

INK, GREY = "#222222", "#6e6e6e"
SLOW, SLOW_BG = "#3a6b9a", "#e3ebf4"
FAST, FAST_BG = "#a8682a", "#f4e9dc"
RED, RED_BG = "#9b2f2f", "#f4e1e1"
NEUT_BG = "#eeeeee"

W_IN, H_IN, YMAX = 6.97, 2.62, 37.6          # 1 data unit = 5 pt in x and y
fig = plt.figure(figsize=(W_IN, H_IN), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, YMAX); ax.axis("off")


def rect(x, y, w, h, fc="none", ec=INK, lw=0.8, ls="-", z=2):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, ls=ls, joinstyle="miter", zorder=z))


def arrow(x0, y0, x1, y1, c=INK, lw=0.8, ms=7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, lw=lw, color=c,
                                 shrinkA=0, shrinkB=0, joinstyle="miter", capstyle="butt", zorder=3))


def text(x, y, s, size=7, **kw):
    kw.setdefault("ha", "center"); kw.setdefault("va", "center"); kw.setdefault("color", INK)
    ax.text(x, y, s, fontsize=size, zorder=5, **kw)


# ------------------------------------------------------------ (a) input windows and target
text(0.8, 36.6, "(a) What the model reads, and what it forecasts", 7.6, ha="left", fontweight="bold")
text(99.2, 36.6, "not to scale", 6, ha="right", color=GREY, style="italic")
x144, x14, xD, xD3 = 6.0, 52.0, 74.0, 92.0
ty = 25.7
ax.plot([2.5, 97.0], [ty, ty], color=INK, lw=0.7, solid_capstyle="butt", zorder=2)
arrow(96.5, ty, 98.8, ty, lw=0.7)
for x, s in [(x144, r"$D-144$ days"), (x14, r"$D-14$ days"), (xD, r"issue day $D$"), (xD3, r"$D+3$")]:
    ax.plot([x, x], [ty - 0.6, ty + 0.6], color=INK, lw=0.7, zorder=2)
    text(x, ty - 1.9, s, 6.6)
text(98.8, ty - 1.9, "time", 6.6, ha="right", color=GREY)

# slow window with 18 step ticks
rect(x144, 31.4, xD - x144, 3.5, fc=SLOW_BG, ec=SLOW)
for k in range(1, 18):
    xx = x144 + k * (xD - x144) / 18
    ax.plot([xx, xx], [31.4, 32.1], color=SLOW, lw=0.5, zorder=3)
text((x144 + xD) / 2, 33.4, r"Slow branch: 18 eight-day means of leaf area index, soil moisture and rain  $\rightarrow$  fuel amount and dryness", 6.6, color=SLOW)
# fast window with 14 step ticks
rect(x14, 27.5, xD - x14, 3.3, fc=FAST_BG, ec=FAST)
for k in range(1, 14):
    xx = x14 + k * (xD - x14) / 14
    ax.plot([xx, xx], [27.5, 28.2], color=FAST, lw=0.5, zorder=3)
text((x14 + xD) / 2, 29.3, "Fast branch: 14 daily maps", 6.6, color=FAST)
text(x14 - 1.0, 29.15, r"weather, greenness, recent fire  $\rightarrow$  current conditions", 6.6, color=FAST, ha="right")
# target window
rect(xD + 0.5, 27.5, xD3 - xD - 0.5, 7.4, fc=RED_BG, ec=RED)
text((xD + xD3) / 2 + 0.25, 33.1, "Target", 6.8, color=RED, fontweight="bold")
text((xD + xD3) / 2 + 0.25, 30.1, "fire detected on\n" + r"day $D+1$, $D+2$ or $D+3$?", 6.4, color=RED, linespacing=1.15)

# ------------------------------------------------------------ (b) processing pipeline
text(0.8, 21.4, "(b) How the information is processed", 7.6, ha="left", fontweight="bold")
gy0, gy1 = 0.5, 20.0                         # dashed group frames
groups = [(0.8, 23.2, "1", "Inputs"), (25.4, 55.6, "2", "Encoders (ConvLSTM)"), (57.8, 99.2, "3", "Fusion and output")]
for gx0, gx1, num, title in groups:
    rect(gx0, gy0, gx1 - gx0, gy1 - gy0, ec=GREY, lw=0.6, ls=(0, (3, 2)), z=1)
    text(gx0 + 0.9, gy1 - 1.4, num, 7.4, ha="left", fontweight="bold")
    text(gx0 + 2.6, gy1 - 1.4, title, 6.8, ha="left", color=GREY)

yT, yB, h = 10.0, 1.4, 7.0                    # two rows: slow (top), fast (bottom)
cT, cB = yT + h / 2, yB + h / 2
# group 1: inputs
rect(1.8, yT, 20.4, h, fc=SLOW_BG, ec=SLOW)
text(12.0, yT + h - 1.6, "Fuel and moisture", 6.9, color=SLOW, fontweight="bold")
text(12.0, yT + 2.3, "leaf area index,\nsoil moisture, rain", 6.3, linespacing=1.15)
rect(1.8, yB, 20.4, h, fc=FAST_BG, ec=FAST)
text(12.0, yB + h - 1.6, "Weather and fire", 6.9, color=FAST, fontweight="bold")
text(12.0, yB + 2.3, "heat, dry air (VPD), wind, NDVI,\nrecent fire, fire intensity, fuel age", 6.3, linespacing=1.15)


# group 2: encoders drawn as chains of ConvLSTM cells
def chain(y, color, bg, label, steps):
    cw, ch, gap = 3.2, 2.8, 1.9
    xs = [27.4 + i * (cw + gap) for i in range(5)]
    yc = y + 1.1
    for i, x in enumerate(xs):
        if i == 3:
            text(x + cw / 2, yc + ch / 2, r"$\cdots$", 7, color=color)
            continue
        rect(x, yc, cw, ch, fc=bg, ec=color, lw=0.8)
    for i in range(4):
        x0 = xs[i] + cw if i != 3 else xs[i] + cw * 0.85
        x1 = xs[i + 1] if i != 2 else xs[i + 1] + cw * 0.15
        arrow(x0, yc + ch / 2, x1, yc + ch / 2, c=color, lw=0.7, ms=5)
    text(27.4, y + h - 1.4, label, 6.7, ha="left", color=color, fontweight="bold")
    text(xs[-1] + cw, y + h - 1.4, steps, 6.2, ha="right", color=GREY)
    return xs[-1] + cw, yc + ch / 2


sx, sy = chain(yT, SLOW, SLOW_BG, "Slow memory", "18 steps")
fx, fy = chain(yB, FAST, FAST_BG, "Fast memory", "14 steps")
arrow(22.2, sy, 27.4, sy, c=SLOW)
arrow(22.2, fy, 27.4, fy, c=FAST)

# group 3: combine, risk map, flag
cx0, cx1 = 59.0, 67.6
cy0, cy1 = yB + 0.6, yT + h - 0.6
rect(cx0, cy0, cx1 - cx0, cy1 - cy0, fc=NEUT_BG, ec=INK, lw=0.8)
mid = (cy0 + cy1) / 2
text((cx0 + cx1) / 2, mid + 2.2, "Combine", 6.9, fontweight="bold")
text((cx0 + cx1) / 2, mid - 1.6, "slow summary\nadded to each\nfast step", 6.1, linespacing=1.15)
arrow(sx, sy, cx0, sy, c=SLOW)
arrow(fx, fy, cx0, fy, c=FAST)

im = Image.open(ARCH).crop((2540, 485, 3005, 952))
im.save(OUT + "_risk.png")
side = 11.0
tx0, ty0 = 70.6, mid - side / 2 - 0.6
ax.imshow(np.asarray(im), extent=(tx0, tx0 + side, ty0, ty0 + side), aspect="auto", zorder=3, interpolation="lanczos")
rect(tx0, ty0, side, side, ec=INK, lw=0.7, z=4)
text(tx0 + side / 2, ty0 + side + 1.2, "Risk score per 1 km pixel", 6.4, fontweight="bold")
text(tx0 + side / 2, ty0 - 1.0, "darker = higher risk", 5.8, color=GREY)
arrow(cx1, mid, tx0, mid)

fx0, fx1 = 85.0, 98.2
rect(fx0, cy0 + 2.2, fx1 - fx0, cy1 - cy0 - 4.4, fc=RED_BG, ec=RED, lw=0.8)
text((fx0 + fx1) / 2, mid + 2.2, "Flag", 6.9, color=RED, fontweight="bold")
text((fx0 + fx1) / 2, mid - 1.6, "pixels above one\nfixed threshold\n(chosen on 2019)", 6.1, linespacing=1.15)
arrow(tx0 + side, mid, fx0, mid)

fig.savefig(OUT + ".png", dpi=300)
fig.savefig(OUT + ".pdf")
print("saved", OUT)
