"""Figures of the paper at print size: full text width 6.93 in, 450 dpi, Aptos,
no text below 6 pt, no title above the panels (the caption says it).

  fig_national_skill.png        2.4 in  (a) lift against the share of land flagged, SMOLDER and
                                        persistence; (b) daily AUC-PR of both, 7-day running means
  fig_budget_curves.png         2.6 in  fire caught and false alarms per hit against the area flagged
  fig_distance_bands.png        2.4 in  fire caught by distance to recent fire, 2019 best-F2 thresholds
  fig_explain_importance.png    3.0 in  (a) both importance measures per input, (b, c) displaced
                                        share by land cover and climate zone
  fig_events_well_<Y>.png       3.4 in  four events, 2 x 2 layout, three panels each
  fig_events_poorly_<Y>.png     3.4 in

Inputs (environment): RUN_2020 (a run's test_2020 folder: national_2020.json and _daily.csv,
operating_point_2020.json, adaptive_budget_2020.json, explain_2020.json, event_maps cache via
EVENT_CACHE), PERSIST (results/national_2020_persistence.json, with _daily.csv), OP_RUN (the
run's name in operating_point_2020.json), FIG_OUT. The architecture figure is drawn by
make_smolder_architecture.py with PAPER=1.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style_smolder import ACCENT, ACCENT2, INK, MUTED, PANEL_BG  # noqa: E402
from style_smolder import style_axes as _style_axes  # noqa: E402

W, DPI = 6.93, 450
FS, FS_T = 6.5, 7.5                                    # body text, panel tags; nothing below 6 pt
plt.rcParams.update({"font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
                     "legend.fontsize": 6.2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "lines.linewidth": 1.2})
RUN = os.environ["RUN_2020"]
PERSIST = os.environ.get("PERSIST", os.path.join(HERE, "..", "results", "national_2020_persistence.json"))
FIG_OUT = os.environ.get("FIG_OUT", HERE)
OP_RUN = os.environ.get("OP_RUN", "SMOLDER")
GREY, LIGHT_BLUE, LIGHT_GREY = "#5F5F64", "#8DB6D9", "#B4B4B9"
CRIT = os.environ.get("THRESHOLD_CRIT", "f2")          # adaptive threshold chosen on 2019 by this criterion
EXPLAIN = os.environ.get("EXPLAIN_JSON", "explain_2020_adaptive_f2.json")


def style_axes(ax, **kw):
    """The report style, with tick labels at the paper size."""
    _style_axes(ax, **kw)
    ax.tick_params(labelsize=FS, length=2.5, width=0.6, pad=2)


def save(fig, name, h):
    fig.set_size_inches(W, h)
    fig.savefig(os.path.join(FIG_OUT, name), dpi=DPI, facecolor="white")
    plt.close(fig)
    print("wrote", name, flush=True)


def tag(ax, t, x=-0.12):
    ax.text(x, 1.03, f"({t})", transform=ax.transAxes, fontsize=FS_T, fontweight="bold", color=INK, va="bottom")


def national_skill():
    S = json.load(open(os.path.join(RUN, "national_2020.json"))); P = json.load(open(PERSIST))
    dS = pd.read_csv(os.path.join(RUN, "national_2020_daily.csv"), parse_dates=["date"])
    dP = pd.read_csv(PERSIST[:-len(".json")] + "_daily.csv", parse_dates=["date"])
    fig = plt.figure(figsize=(W, 2.4))
    a = fig.add_axes([0.075, 0.17, 0.39, 0.74]); b = fig.add_axes([0.585, 0.17, 0.40, 0.74])
    for d, lab, col, ls in ((S, "SMOLDER", ACCENT, "-"), (P, "persistence baseline", GREY, (0, (5, 3)))):
        k = np.array([x["k"] for x in d["topk_national"]]); lift = np.array([x["lift"] for x in d["topk_national"]])
        a.plot(100 * k, lift, color=col, ls=ls, lw=1.4, label=lab)
    a.axhline(1, color=MUTED, lw=0.8, ls=":", label="random selection")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10]); a.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10"])
    a.set_yticks([1, 2, 5, 10, 20, 50, 100, 200, 500]); a.set_yticklabels(["1", "2", "5", "10", "20", "50", "100", "200", "500"])
    a.minorticks_off(); a.set_ylim(0.8, 600)
    a.set_xlabel("Share of Australia's land flagged (%)"); a.set_ylabel("Lift over random (mean over days)")
    a.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#C8C8CC")
    for d, lab, col, ls in ((dS, "SMOLDER", ACCENT, "-"), (dP, "persistence baseline", GREY, (0, (5, 3)))):
        s = d.set_index("date")["auc_pr"].rolling(7, center=True, min_periods=4).mean()
        b.plot(s.index, s.values, color=col, ls=ls, lw=1.2, label=lab)
    b.set_ylabel("Daily AUC-PR, 7-day running mean"); b.set_ylim(0, None)
    b.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    b.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#C8C8CC")
    for ax, t in ((a, "a"), (b, "b")):
        style_axes(ax); tag(ax, t)
    save(fig, "fig_national_skill.png", 2.4)


def budget_curves():
    OP = json.load(open(os.path.join(RUN, "operating_point_2020.json")))
    AB = json.load(open(os.path.join(RUN, "adaptive_budget_2020.json")))
    series = [("SMOLDER, same area every day", ACCENT, "-", ("fixed", OP_RUN)),
              ("SMOLDER, one threshold for all days", ACCENT, (0, (5, 3)), ("adaptive", "SMOLDER")),
              ("Persistence, same area every day", GREY, "-", ("fixed", "persistence")),
              ("Persistence, one threshold for all days", GREY, (0, (5, 3)), ("adaptive", "persistence"))]
    fig = plt.figure(figsize=(W, 2.6))
    axes = [fig.add_axes([0.075, 0.16, 0.40, 0.76]), fig.add_axes([0.585, 0.16, 0.40, 0.76])]
    for lab, col, ls, (rule, name) in series:
        if rule == "fixed":
            r = OP["rankings"][name]; k, rec, fpt = np.array(r["k"]), np.array(r["recall"]), np.array(r["fp_per_tp"])
        else:
            c = AB["adaptive_curve"][name]; k, rec, fpt = np.array(c["mean_share"]), np.array(c["recall"]), np.array(c["fp_per_tp"])
        axes[0].plot(100 * k, 100 * rec, color=col, ls=ls, lw=1.3, label=lab)
        axes[1].plot(100 * k, fpt, color=col, ls=ls, lw=1.3)
    for name, col in (("SMOLDER", ACCENT), ("persistence", GREY)):        # best-F2 threshold of 2019
        b = AB["adaptive_best"][name][CRIT]
        axes[0].plot(100 * b["mean_share"], 100 * b["recall"], "o", ms=3.5, mfc="white", mec=col, mew=1.0, zorder=5)
        axes[1].plot(100 * b["mean_share"], b["fp_per_tp"], "o", ms=3.5, mfc="white", mec=col, mew=1.0, zorder=5)
    axes[0].plot([], [], "o", ms=3.5, mfc="white", mec=INK, mew=1.0, ls="none", label="threshold chosen on 2019 (best F2)")
    for ax, ylab in zip(axes, ("Fire caught (%)", "False alarms per fire pixel caught")):
        ax.set_xscale("log"); ax.set_xlim(0.02, 20)
        ax.set_xticks([0.03, 0.1, 0.3, 1, 3, 10]); ax.set_xticklabels(["0.03", "0.1", "0.3", "1", "3", "10"]); ax.minorticks_off()
        ax.set_xlabel("Mean daily area flagged (% of land)"); ax.set_ylabel(ylab)
    axes[0].set_ylim(0, 100)
    axes[1].set_yscale("log"); axes[1].set_ylim(1, 300)
    axes[1].set_yticks([1, 3, 10, 30, 100, 300]); axes[1].set_yticklabels(["1", "3", "10", "30", "100", "300"]); axes[1].minorticks_off()
    axes[0].legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#C8C8CC")
    for ax, t in zip(axes, "ab"):
        style_axes(ax); tag(ax, t)
    save(fig, "fig_budget_curves.png", 2.6)


def distance_bands():
    OP = json.load(open(os.path.join(RUN, "operating_point_2020.json")))
    AB = json.load(open(os.path.join(RUN, "adaptive_budget_2020.json")))["adaptive_best"]
    bands = ["0-3 km", "3-10 km", "> 10 km"]; share = OP["bands"]["fire_share"]
    bars = [("Persistence", GREY, AB["persistence"][CRIT]), ("SMOLDER", ACCENT, AB["SMOLDER"][CRIT])]
    fig = plt.figure(figsize=(W, 2.4)); ax = fig.add_axes([0.075, 0.21, 0.9, 0.75])
    x = np.arange(len(bands)); w = 0.32
    for j, (lab, col, src) in enumerate(bars):
        v = [100 * src[f"recall {b}"] for b in bands]; xx = x + (j - 0.5) * w
        ax.bar(xx, v, w * 0.9, color=col, zorder=3,
               label=f"{lab} (flags {100 * src['mean_share']:.3f} % of the land per day on average)")
        for xi, vi in zip(xx, v):
            ax.text(xi, vi + 1.2, f"{vi:.1f}" if vi < 10 else f"{vi:.0f}", ha="center", va="bottom", fontsize=6, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b.replace('-', ' to ')}  ({100 * s:.0f} % of fire)" for b, s in zip(bands, share)])
    ax.set_xlabel("Distance to the nearest fire detected on the issue day or the two days before")
    ax.set_ylabel("Fire caught (%)"); ax.set_ylim(0, 100)
    style_axes(ax, grid_x=False)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#C8C8CC")
    save(fig, "fig_distance_bands.png", 2.4)


def importance():
    E = json.load(open(os.path.join(RUN, EXPLAIN))); G = E["groups"]
    names = sorted(G, key=lambda n: G[n]["retention"])
    disp = np.array([1 - G[n]["retention"] for n in names]); drop = np.array([G[n]["ap_drop"] for n in names])
    fd = lambda key: np.array([max(np.std([1 - v for v in E["folds"][f]["groups"][n]["retention"]]) if key == "r"
                                   else np.std(E["folds"][f]["groups"][n].get("ap_drop", [0])) for f in ("time", "space"))
                               for n in names])                                                     # noqa: E731
    fig = plt.figure(figsize=(W, 3.0))
    y = np.arange(len(names))[::-1]
    a1 = fig.add_axes([0.235, 0.19, 0.13, 0.74]); a2 = fig.add_axes([0.395, 0.19, 0.11, 0.74], sharey=a1)
    a1.barh(y, disp, height=0.62, color=[ACCENT2 if n == "fire history" else ACCENT for n in names], zorder=3)
    a1.errorbar(disp, y, xerr=fd("r"), fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.5, zorder=4)
    a1.set_xscale("log"); a1.set_xlim(0.008, 1.3); a1.set_xticks([0.01, 0.1, 1]); a1.set_xticklabels(["1", "10", "100"]); a1.minorticks_off()
    a1.set_yticks(y); a1.set_yticklabels(names, fontsize=6); a1.set_xlabel("Flagged pixels no\nlonger flagged (%)")
    a2.barh(y, 100 * drop, height=0.62, color=[ACCENT2 if n == "fire history" else LIGHT_BLUE for n in names], zorder=3)
    a2.set_xlim(min(-2, 100 * drop.min() - 1), 100 * drop.max() * 1.15); a2.set_xlabel("Relative fall of\nAUC-PR (%)")
    a2.tick_params(labelleft=False); a2.axvline(0, color=INK, lw=0.5)
    for ax in (a1, a2):
        style_axes(ax, grid_y=False)
    tag(a1, "a", x=-1.3)
    cmap = LinearSegmentedColormap.from_list("imp", [PANEL_BG, "#FFE9A8", "#FFAB3D", "#E8452C", "#8B0000"])
    cats = {"by_landcover": ["grassland", "shrubland", "open forest", "closed forest", "cropland"],
            "by_climate": ["tropical", "arid", "temperate"]}
    for k, (key, x0, wd, t) in enumerate((("by_landcover", 0.53, 0.26, "b"), ("by_climate", 0.83, 0.155, "c"))):
        ax = fig.add_axes([x0, 0.19, wd, 0.74], sharey=a1)
        M = np.array([[1 - G[n][key].get(c, np.nan) for c in cats[key]] for n in names])
        ax.imshow(np.log10(np.clip(M, 0.005, 1)), cmap=cmap, vmin=np.log10(0.005), vmax=0, aspect="auto",
                  extent=(-0.5, M.shape[1] - 0.5, -0.5, M.shape[0] - 0.5))
        for i, yy in enumerate(y):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, yy, f"{100*M[i, j]:.0f}" if M[i, j] >= 0.1 else f"{100*M[i, j]:.1f}", ha="center",
                            va="center", fontsize=6, color="white" if M[i, j] > 0.3 else INK)
        ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(cats[key], rotation=30, ha="right")
        ax.tick_params(length=0, labelleft=False)
        for s in ax.spines.values():
            s.set_visible(False)
        tag(ax, t, x=-0.02)
    save(fig, "fig_explain_importance.png", 3.0)


def events():
    year = int(os.environ.get("YEAR", 2020)); z = np.load(os.environ["EVENT_CACHE"])
    # muted but distinct: recent fire amber, later fire brick red, flagged blue tones; coastline grey
    OCEAN, LAND, RECENT, HIT, MISS, FALSE = "#FFFFFF", "#E9E9EC", "#D69A3C", "#2F6DA8", "#B8473F", "#A9C8E6"
    COAST = "#7A7A80"
    cmap = ListedColormap([OCEAN, LAND, RECENT, HIT, MISS, FALSE]); S = 192
    LON0, LAT0, PX = 112.904998779, -9.005000113999998, 0.01

    def rgb(land, recent, y, flag):
        c = np.where(land, 1, 0); c = np.where(recent, 2, c)
        if flag is None:
            return np.where(y, 4, c)
        c = np.where(flag & ~y, 5, c); c = np.where(flag & y, 3, c)
        return np.where(~flag & y, 4, c)

    for kind in ("well", "poorly"):
        n = sum(1 for k in z.files if k.startswith(kind) and k.endswith("_meta"))
        fig = plt.figure(figsize=(W, 3.4))
        pw, gap, x_ev = 0.155, 0.008, (0.012, 0.512)
        for i in range(n):
            p = f"{kind}{i}_"; D, y0, x0, _ = z[p + "meta"]
            land, recent, y = z[p + "land"], z[p + "recent"], z[p + "y"]
            lat = LAT0 - (y0 + S / 2) * PX; lon = LON0 + (x0 + S / 2) * PX; nf = int(y.sum())
            col, row = i % 2, i // 2
            yb = 0.565 - row * 0.455
            for j, (c, title) in enumerate((("obs", None), ("persistence", "Persistence"), ("full", "SMOLDER"))):
                ax = fig.add_axes([x_ev[col] + j * (pw + gap), yb, pw, pw * W / 3.4])
                flag = None if c == "obs" else z[p + c]
                ax.imshow(rgb(land, recent, y, flag), cmap=cmap, vmin=-0.5, vmax=5.5, interpolation="nearest")
                ax.contour(land.astype(float), levels=[0.5], colors=COAST, linewidths=0.5)
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_color("#9A9AA0"); s.set_linewidth(0.5)
                if c == "obs":
                    ax.set_title(f"{str(z[p + 'date'])}, {abs(lat):.1f}°S {lon:.1f}°E\n{nf:,} fire px",
                                 fontsize=6, color=INK, loc="left", pad=2)
                    ax.plot([8, 58], [S - 10, S - 10], color=INK, lw=1.0)
                    ax.text(33, S - 14, "50 km", ha="center", va="bottom", fontsize=6, color=INK)
                else:
                    hit = int((flag & y).sum()); fa = int((flag & ~y).sum())
                    ax.set_title(f"{title}\n{100 * hit / max(nf, 1):.0f} % caught, {fa:,} false", fontsize=6,
                                 color=INK, loc="left", pad=2)
        handles = [Patch(color=RECENT, label="fire on days D-2 to D"), Patch(color=HIT, label="flagged, burned"),
                   Patch(color=MISS, label="burned, not flagged (left: all fire D+1 to D+3)"),
                   Patch(color=FALSE, label="flagged, no fire"),
                   matplotlib.lines.Line2D([], [], color=COAST, lw=0.8, label="coastline")]
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=6, bbox_to_anchor=(0.5, 0.0),
                   handlelength=1.2, columnspacing=1.0)
        save(fig, f"fig_events_{kind}_{year}.png", 3.4)


if __name__ == "__main__":
    national_skill(); budget_curves(); distance_bands(); importance(); events()
