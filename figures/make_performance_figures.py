"""SMOLDER against persistence on the validation year 2019, from the results
files only:
  fig_budget_curves.png    fire caught and false alarms per hit against the
                              mean daily area flagged, fixed and adaptive budget
                              (results/experiments/smolder/operating_point_2019.json,
                              adaptive_budget_2019.json)
  fig_distance_bands.png   fire caught by distance to the nearest fire of
                              days D-2..D at a mean daily area of 0.13 %
                              (adaptive_budget_2019.json, operating_point_2019.json)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from style_smolder import ACCENT, INK, MUTED, new_figure, style_axes

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("RESULTS_DIR", os.path.join(HERE, "..", "results", "experiments", "smolder"))
OP = json.load(open(os.path.join(RES, "operating_point_2019.json")))
AB = json.load(open(os.path.join(RES, "adaptive_budget_2019.json")))

GREY = "#5F5F64"
LIGHT_BLUE = "#8DB6D9"
LIGHT_GREY = "#B4B4B9"
BUDGET = 0.0013
SERIES = [  # label, colour, linestyle, source
    ("SMOLDER, fixed daily area", ACCENT, "-", ("fixed", "full_s123")),
    ("SMOLDER, adaptive daily area", ACCENT, (0, (5, 3)), ("adaptive", "SMOLDER")),
    ("Persistence, fixed daily area", GREY, "-", ("fixed", "persistence")),
    ("Persistence, adaptive daily area", GREY, (0, (5, 3)), ("adaptive", "persistence")),
]


def curve(src):
    rule, name = src
    if rule == "fixed":
        r = OP["rankings"][name]
        return np.array(r["k"]), np.array(r["recall"]), np.array(r["fp_per_tp"])
    c = AB["adaptive_curve"][name]
    return np.array(c["mean_share"]), np.array(c["recall"]), np.array(c["fp_per_tp"])


def budget_curves():
    fig = new_figure((11.0, 4.6))
    axes = [fig.add_subplot(1, 2, i + 1) for i in range(2)]
    for label, col, ls, src in SERIES:
        k, rec, fpt = curve(src)
        axes[0].plot(100 * k, 100 * rec, color=col, ls=ls, lw=1.8, label=label)
        axes[1].plot(100 * k, fpt, color=col, ls=ls, lw=1.8)
    for ax, ylab in zip(axes, ("Fire caught (%)", "False alarms per fire pixel caught")):
        ax.axvline(100 * BUDGET, color=MUTED, lw=0.9, ls=":", zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(0.02, 20)
        ax.set_xticks([0.03, 0.1, 0.3, 1, 3, 10])
        ax.set_xticklabels(["0.03", "0.1", "0.3", "1", "3", "10"])
        ax.set_xlabel("Mean daily area flagged (% of land)", fontsize=10, color=INK)
        ax.set_ylabel(ylab, fontsize=10, color=INK)
    axes[0].set_ylim(0, 100)
    axes[1].set_yscale("log")
    axes[1].set_ylim(1, 300)
    axes[1].set_yticks([1, 3, 10, 30, 100, 300])
    axes[1].set_yticklabels(["1", "3", "10", "30", "100", "300"])
    for ax, tag in zip(axes, "ab"):
        style_axes(ax)
        ax.text(-0.13, 1.02, f"({tag})", transform=ax.transAxes, fontsize=11, color=INK, va="bottom")
    axes[0].text(100 * BUDGET * 1.12, 3, "0.13 %", fontsize=8.5, color=MUTED)
    axes[0].legend(fontsize=8.6, loc="upper left", frameon=True, facecolor="white", edgecolor="#C8C8CC")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_budget_curves.png"), dpi=300, bbox_inches="tight", facecolor="white")
    print("wrote fig_budget_curves.png")


def distance_bands():
    bands = ["0-3 km", "3-10 km", "> 10 km"]
    share = OP["bands"]["fire_share"]
    bars = [("Persistence, fixed", GREY, AB["fixed"]["persistence"]),
            ("Persistence, adaptive", LIGHT_GREY, AB["adaptive"]["persistence"]),
            ("SMOLDER, fixed", ACCENT, AB["fixed"]["SMOLDER"]),
            ("SMOLDER, adaptive", LIGHT_BLUE, AB["adaptive"]["SMOLDER"])]
    fig = new_figure((8.2, 4.4))
    ax = fig.add_subplot(111)
    x = np.arange(len(bands))
    w = 0.19
    for j, (label, col, src) in enumerate(bars):
        m = src[str(BUDGET)]
        v = [100 * m[f"recall {b}"] for b in bands]
        xx = x + (j - 1.5) * w
        ax.bar(xx, v, w * 0.92, color=col, label=label, zorder=3)
        for xi, vi in zip(xx, v):
            ax.text(xi, vi + 1.2, f"{vi:.1f}" if vi < 10 else f"{vi:.0f}", ha="center", va="bottom",
                    fontsize=7.8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b.replace('-', '–')}\n({100 * s:.0f} % of fire)" for b, s in zip(bands, share)],
                       fontsize=9.5)
    ax.set_xlabel("Distance to the nearest fire of days D−2 to D", fontsize=10, color=INK)
    ax.set_ylabel("Fire caught (%)", fontsize=10, color=INK)
    ax.set_ylim(0, 100)
    style_axes(ax, grid_x=False)
    ax.legend(fontsize=8.6, loc="upper right", frameon=True, facecolor="white", edgecolor="#C8C8CC")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_distance_bands.png"), dpi=300, bbox_inches="tight", facecolor="white")
    print("wrote fig_distance_bands.png")


if __name__ == "__main__":
    budget_curves()
    distance_bands()
