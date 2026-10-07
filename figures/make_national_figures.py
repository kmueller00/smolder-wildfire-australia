"""National evaluation figures (results/national_2020*, written by
smolder.evaluation.evaluate_national).

fig_national_maps.png     mean predicted risk and observed fire over 2020
fig_national_skill.png    national lift (with persistence baseline), daily AUC-PR
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FuncFormatter, MultipleLocator
from style_smolder import (ACCENT, ACCENT2, GRID_COLOR, INK, MUTED, PANEL_BG, SPINE_COLOR,
                           new_figure, style_axes)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("NATIONAL_DIR", os.path.join(HERE, "..", "results"))      # a run's national_2020*
FIG_OUT = os.environ.get("FIG_OUT", HERE)
S = json.load(open(os.path.join(RES, "national_2020.json")))
D = pd.read_csv(os.path.join(RES, "national_2020_daily.csv"), parse_dates=["date"])
M = np.load(os.path.join(RES, "national_2020_maps.npz"))
_BP = os.environ.get("PERSIST_JSON", os.path.join(HERE, "..", "results", "national_2020_persistence.json"))
B = json.load(open(_BP)) if os.path.exists(_BP) else None     # evaluate_persistence.py
BASELINE = "#8A8F98"
LON0, LAT0, PX = 112.904998779, -9.005000113999998, 0.01
OCEAN = "#C9D6E3"
FIRE_HIT, FIRE_MISS = "#00E83A", "#C400FF"
HATCH = "#3A3A3A"
RISK = LinearSegmentedColormap.from_list("risk", [PANEL_BG, "#FFE9A8", "#FFAB3D", "#E8452C", "#8B0000"])
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
plt.rcParams["hatch.color"] = HATCH
plt.rcParams["hatch.linewidth"] = 0.9

DOWN = int(M["down"])
LAND = M["land"]
H, W = LAND.shape
EXT = [LON0, LON0 + W * DOWN * PX, LAT0 - H * DOWN * PX, LAT0]


def geo_axes(ax):
    ax.set_facecolor(OCEAN)
    ax.set_xlim(EXT[0], EXT[1]); ax.set_ylim(EXT[2], EXT[3]); ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MultipleLocator(5)); ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}°E"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):.0f}°S"))
    ax.minorticks_off()
    ax.set_axisbelow(False)
    ax.grid(True, color=GRID_COLOR, ls="--", lw=0.7, alpha=0.9)
    ax.tick_params(colors=INK, labelsize=8.5, length=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SPINE_COLOR)


def coast(ax):
    ax.contour(LAND.astype(float), levels=[0.5], colors="#1F1F22", linewidths=0.5,
               extent=EXT, origin="upper", zorder=5)


def maps():
    fig = new_figure((15.0, 6.6))
    a, b = fig.add_subplot(121), fig.add_subplot(122)
    risk = np.where(LAND, M["mean_risk"].astype(np.float32), np.nan)
    im0 = a.imshow(risk, extent=EXT, cmap=RISK, interpolation="nearest",
                   vmin=float(np.nanpercentile(risk, 2)), vmax=float(np.nanpercentile(risk, 99.8)))
    cb0 = fig.colorbar(im0, ax=a, fraction=0.035, pad=0.02, format=FuncFormatter(lambda v, _: f"{v:.3f}"))
    cb0.set_label("mean model score", fontsize=9)
    a.set_title("(a) Mean predicted risk, 2020", fontsize=12, fontweight="bold", color=INK, loc="left")
    fd = M["fire_days"].astype(np.float32)
    b.imshow(np.where(LAND, 1.0, np.nan), extent=EXT, cmap=LinearSegmentedColormap.from_list("l", [PANEL_BG, PANEL_BG]),
             interpolation="nearest")
    im1 = b.imshow(np.where(LAND & (fd > 0), fd, np.nan), extent=EXT, cmap="magma_r",
                   norm=LogNorm(vmin=1, vmax=max(2.0, float(np.nanmax(fd)))), interpolation="nearest")
    cb1 = fig.colorbar(im1, ax=b, fraction=0.035, pad=0.02, format=FuncFormatter(lambda v, _: f"{v:g}"))
    cb1.set_label("issue days with fire in the target (log scale)", fontsize=9)
    b.set_title("(b) Observed fire, 2020", fontsize=12, fontweight="bold", color=INK, loc="left")
    for ax in (a, b):
        coast(ax)
        geo_axes(ax)
    fig.suptitle("Annual picture, 2020 hold-out year", fontsize=14, fontweight="bold", color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_OUT, "fig_national_maps.png"), dpi=220, bbox_inches="tight", facecolor="white")
    print("wrote fig_national_maps.png")


def skill():
    fig = new_figure((11.5, 4.6))
    a, b = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
    t = S["topk_national"]
    k = np.array([x["k"] for x in t]) * 100
    a.plot(k, [x["lift"] for x in t], "-", color=ACCENT, lw=2.2, zorder=4, label="SMOLDER")
    if B is not None:
        tb = B["topk_national"]
        a.plot(np.array([x["k"] for x in tb]) * 100, [x["lift"] for x in tb], ls=(0, (6, 3)),
               color=BASELINE, lw=2.0, zorder=3, label="persistence baseline")
    a.axhline(1, color=MUTED, lw=1.1, ls=(0, (4, 3)), label="random selection")
    a.legend(loc="upper right", fontsize=8.6, frameon=True, facecolor="white", edgecolor=SPINE_COLOR)
    a.set_xscale("log"); a.set_yscale("log")
    xt = [0.1, 0.2, 0.5, 1, 2, 5, 10]
    yt = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    a.set_xticks(xt); a.set_xticklabels([f"{v:g}" for v in xt])
    a.set_yticks(yt); a.set_yticklabels([f"{v:g}" for v in yt])
    a.set_ylim(0.8, 500)
    a.set_xlabel("Share of Australia's land flagged (%)", fontsize=10, fontweight="bold", color=INK)
    a.set_ylabel("Lift over random", fontsize=10, fontweight="bold", color=INK)
    a.set_title("(a) National lift, mean over days", fontsize=11.5, fontweight="bold", color=INK, loc="left")
    style_axes(a)

    ap7 = D.set_index("date")["auc_pr"].rolling(7, center=True, min_periods=3).mean()
    b.plot(ap7.index, ap7.values, "-", color=ACCENT, lw=2.0)
    b.set_ylim(0, max(0.3, float(np.nanmax(ap7)) * 1.15))
    b.set_ylabel("AUC-PR, 7-day mean", fontsize=10, fontweight="bold", color=INK)
    b.set_title("(b) National AUC-PR through 2020", fontsize=11.5, fontweight="bold", color=INK, loc="left")
    b.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
    b.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    style_axes(b)

    fig.suptitle("National skill, 2020 hold-out year", fontsize=14, fontweight="bold", color=INK, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_OUT, "fig_national_skill.png"), dpi=250, bbox_inches="tight", facecolor="white")
    print("wrote fig_national_skill.png")


if __name__ == "__main__":
    maps()
    skill()
