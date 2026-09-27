"""National evaluation figures (results/national_2020*, written by
smolder.evaluation.evaluate_national).

fig_national_example.png  one forecast for the whole continent
fig_national_maps.png     mean predicted risk and observed fire over 2020
fig_national_skill.png    national lift, daily AUC-PR, reliability
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
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.stats import rankdata
from style_smolder import (ACCENT, ACCENT2, GRID_COLOR, INK, MUTED, PANEL_BG, SPINE_COLOR,
                           new_figure, style_axes)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
S = json.load(open(os.path.join(RES, "national_2020.json")))
D = pd.read_csv(os.path.join(RES, "national_2020_daily.csv"), parse_dates=["date"])
M = np.load(os.path.join(RES, "national_2020_maps.npz"))
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


def example():
    """The most fire-active issue day of 2020, whole continent."""
    i = 0
    iso = str(M["example_dates"][i])
    date = f"{int(iso[8:10])} {MONTHS[int(iso[5:7]) - 1]} {iso[:4]}"
    risk = np.where(LAND, M["example_risk"][i].astype(np.float32), np.nan)
    pct = np.full(risk.shape, np.nan, np.float32)
    ok = np.isfinite(risk)
    pct[ok] = 100.0 * (rankdata(risk[ok]) - 1) / max(ok.sum() - 1, 1)
    top1 = np.nan_to_num(pct) >= 99.0
    fire = M["example_fire"][i] & LAND
    yy, xx = np.where(fire)
    fx, fy = EXT[0] + (xx + 0.5) * DOWN * PX, EXT[3] - (yy + 0.5) * DOWN * PX
    hit = top1[yy, xx]
    row = D[D.date == pd.Timestamp(iso)].iloc[0]

    fig = new_figure((12.5, 10.2))
    ax = fig.add_subplot(111)
    im = ax.imshow(pct, extent=EXT, cmap=RISK, vmin=0, vmax=100, interpolation="nearest", zorder=1)
    ax.contourf(top1.astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"],
                extent=EXT, origin="upper", zorder=2)
    coast(ax)
    ax.scatter(fx[~hit], fy[~hit], s=3.0, c=FIRE_MISS, marker="s", linewidths=0, zorder=4)
    ax.scatter(fx[hit], fy[hit], s=3.0, c=FIRE_HIT, marker="s", linewidths=0, zorder=4)
    geo_axes(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Predicted risk, percentile across Australia", fontsize=9.5, color=INK)
    cb.outline.set_edgecolor(SPINE_COLOR)
    ax.legend(handles=[Patch(facecolor="none", edgecolor=HATCH, hatch="////", label="National top-1% risk area"),
                       Patch(facecolor=FIRE_HIT, label="Observed fire inside it"),
                       Patch(facecolor=FIRE_MISS, label="Observed fire outside it"),
                       Patch(facecolor=OCEAN, label="Ocean")],
              loc="lower left", fontsize=9, frameon=True, facecolor="white", edgecolor=MUTED)
    ax.set_title(f"SMOLDER forecast issued {date}: fire risk for the next 3 days",
                 fontsize=13.5, fontweight="bold", color=INK, pad=10)
    fig.text(0.5, 0.035,
             f"The most fire-active issue day of the 2020 hold-out year. Flagging the top 1% of Australia's land "
             f"captures {100*row['tpr_0.01']:.0f}% of the fire observed over the next three days "
             f"({row['lift_0.01']:.0f}× better than random). Displayed at 0.04°.",
             ha="center", fontsize=9.2, color=MUTED, style="italic", wrap=True)
    fig.savefig(os.path.join(HERE, "fig_national_example.png"), dpi=220, bbox_inches="tight", facecolor="white")
    print("wrote fig_national_example.png")


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
    fig.text(0.5, 0.02, "Aggregated to 0.04° for display. Blue is ocean.", ha="center",
             fontsize=9, color=MUTED, style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_national_maps.png"), dpi=220, bbox_inches="tight", facecolor="white")
    print("wrote fig_national_maps.png")


def skill():
    fig = new_figure((16.5, 5.0))
    a, b, c = (fig.add_subplot(1, 3, i + 1) for i in range(3))
    t = S["topk_national"]
    k = np.array([x["k"] for x in t]) * 100
    a.plot(k, [x["lift"] for x in t], "-", color=ACCENT, lw=2.2, label="all fire")
    a.plot(k, [x["lift_new"] for x in t], "--", color=ACCENT2, lw=2.0, label="new fire")
    a.axhline(1, color=MUTED, lw=1.1, ls=(0, (4, 3)))
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xticks([0.1, 1, 10]); a.set_xticklabels(["0.1", "1", "10"])
    a.set_yticks([1, 3, 10, 30, 100, 300]); a.set_yticklabels(["1", "3", "10", "30", "100", "300"])
    a.set_ylim(0.8, 500)
    a.set_xlabel("Share of Australia's land flagged (%)", fontsize=10, fontweight="bold", color=INK)
    a.set_ylabel("Lift over random", fontsize=10, fontweight="bold", color=INK)
    a.set_title("(a) National lift, mean over days", fontsize=11.5, fontweight="bold", color=INK, loc="left")
    style_axes(a)
    a.legend(fontsize=9, loc="upper right", frameon=True, facecolor="white", edgecolor=MUTED)

    ap7 = D.set_index("date")["auc_pr"].rolling(7, center=True, min_periods=3).mean()
    b.plot(ap7.index, ap7.values, "-", color=ACCENT, lw=2.0, label="daily AUC-PR, 7-day mean")
    b.set_ylim(0, max(0.3, float(np.nanmax(ap7)) * 1.15))
    b.set_ylabel("AUC-PR", fontsize=10, fontweight="bold", color=INK)
    b.set_title("(b) National AUC-PR through 2020", fontsize=11.5, fontweight="bold", color=INK, loc="left")
    b.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
    b.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    style_axes(b)
    b.legend(fontsize=9, loc="upper right", frameon=True, facecolor="white", edgecolor=MUTED)

    rel = S["calibration"]["reliability"]
    pr = np.array([r["mean_predicted"] for r in rel]); ob = np.array([r["observed_rate"] for r in rel])
    lo, hi = 1e-5, 0.5
    c.plot([lo, hi], [lo, hi], color=MUTED, lw=1.1, ls=(0, (4, 3)), label="perfect calibration")
    c.plot(pr, ob, "-", color=ACCENT, lw=2.2, label="2020, calibrated on 2019")
    c.set_xscale("log"); c.set_yscale("log"); c.set_xlim(lo, hi); c.set_ylim(lo, hi)
    ticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    lab = ["0.001 %", "0.01 %", "0.1 %", "1 %", "10 %"]
    c.set_xticks(ticks); c.set_xticklabels(lab); c.set_yticks(ticks); c.set_yticklabels(lab)
    c.set_xlabel("Predicted probability", fontsize=10, fontweight="bold", color=INK)
    c.set_ylabel("Observed fire rate", fontsize=10, fontweight="bold", color=INK)
    c.set_title("(c) Reliability", fontsize=11.5, fontweight="bold", color=INK, loc="left")
    style_axes(c)
    c.legend(fontsize=9, loc="upper left", frameon=True, facecolor="white", edgecolor=MUTED)

    fig.suptitle("National skill, 2020 hold-out year", fontsize=14, fontweight="bold", color=INK, y=1.03)
    fig.text(0.5, -0.04,
             f"All {S['n_days']} valid issue days, all {S['n_land_px']/1e6:.1f} M land pixels. New fire: more than "
             "3 px from any fire detected in the three days up to the issue day.",
             ha="center", fontsize=9, color=MUTED, style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_national_skill.png"), dpi=250, bbox_inches="tight", facecolor="white")
    print("wrote fig_national_skill.png")


if __name__ == "__main__":
    example()
    maps()
    skill()
