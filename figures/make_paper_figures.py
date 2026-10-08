"""Figures of the paper at print size: full text width 6.93 in, 450 dpi, Aptos,
no text below 6 pt, no title above the panels (the caption says it).

  fig_national_skill.png        2.4 in  (a) lift against the share of land flagged, SMOLDER and
                                        persistence; (b) daily AUC-PR of both, 7-day running means
  fig_budget_curves.png         2.8 in  fire caught and false alarms per hit against the area flagged
  fig_distance_bands.png        3.35 x 2.4 in (one column)  fire caught by distance to recent fire in fine bands, 2019 best-F2 thresholds
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
    fig.set_size_inches(fig.get_size_inches()[0], h)          # full width (6.93 in) or one column (3.35 in)
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
    fig = plt.figure(figsize=(W, 2.8))
    axes = [fig.add_axes([0.075, 0.28, 0.40, 0.665]), fig.add_axes([0.585, 0.28, 0.40, 0.665])]
    for lab, col, ls, (rule, name) in series:
        if rule == "fixed":
            r = OP["rankings"][name]; k, rec, fpt = np.array(r["k"]), np.array(r["recall"]), np.array(r["fp_per_tp"])
        else:
            c = AB["adaptive_curve"][name]; k, rec, fpt = np.array(c["mean_share"]), np.array(c["recall"]), np.array(c["fp_per_tp"])
        axes[0].plot(100 * k, 100 * rec, color=col, ls=ls, lw=1.3, label=lab)
        axes[1].plot(100 * k, fpt, color=col, ls=ls, lw=1.3)
    fpath = os.path.join(RUN, "fine_bands_area_2020.json")
    if os.path.exists(fpath):                                  # area needed for 50 % and 75 % (one threshold)
        AR = json.load(open(fpath))["area_for_recall"]
        for name, col in (("SMOLDER", ACCENT), ("persistence", GREY)):
            for tgt in ("50 %", "75 %"):
                a_ = AR[name]["one threshold"][tgt]
                axes[0].plot(100 * a_, float(tgt[:2]), "D", ms=3.0, color=col, zorder=6)
        for tgt in (50, 75):
            axes[0].axhline(tgt, color=MUTED, lw=0.6, ls=":", zorder=1)
        axes[0].plot([], [], "D", ms=3.0, color=INK, ls="none", label="area that catches 50 % and 75 % (one threshold)")
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
    h_, l_ = axes[0].get_legend_handles_labels()              # one legend for both panels, below them
    fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.53, 0.0), columnspacing=1.6)
    for ax, t in zip(axes, "ab"):
        style_axes(ax); tag(ax, t)
    save(fig, "fig_budget_curves.png", 2.8)


def distance_bands():
    """One-column figure (3.35 in): fire caught by distance to the nearest fire of days
    D-2..D in fine bands, at the best-F2 thresholds of 2019 (fine_bands_area_2020.json,
    smolder.evaluation.fine_bands_area). Horizontal bars, bands from near (top) to far."""
    F = json.load(open(os.path.join(RUN, "fine_bands_area_2020.json")))
    fb = F["fine_bands"]; share = fb["fire_share"]
    labels = ["0 km (burned itself)", "up to 1 km", "1 to 2 km", "2 to 3 km", "3 to 5 km", "5 to 10 km",
              "10 to 25 km", "more than 25 km"]
    bars = [("Persistence", GREY, fb["persistence"]), ("SMOLDER", ACCENT, fb["SMOLDER"])]
    WC = 3.35
    fig = plt.figure(figsize=(WC, 2.4)); ax = fig.add_axes([0.355, 0.15, 0.615, 0.7])
    y = np.arange(len(labels))[::-1]; h = 0.38
    for j, (lab, col, src) in enumerate(bars):
        v = np.array([100 * r for r in src["recall"]]); yy = y + (0.5 - j) * h
        ax.barh(yy, v, h * 0.9, color=col, zorder=3,
                label=f"{lab} ({100 * src['mean_share_flagged']:.3f} % of land per day)")
        for yi, vi in zip(yy, v):
            if vi > 0:                                         # persistence catches nothing beyond 2 km
                ax.text(vi + 1.5, yi, f"{vi:.1f}" if vi < 10 else f"{vi:.0f}", ha="left", va="center",
                        fontsize=6, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{l} ({100 * s_:.1f} %)" for l, s_ in zip(labels, share)], fontsize=6)
    ax.set_xlim(0, 112); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Fire caught (%)", labelpad=1)
    style_axes(ax, grid_y=False)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.55, 1.01), ncol=1, frameon=False, fontsize=6,
              handlelength=1.0, borderaxespad=0)
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
    """Event maps: per event two panels, persistence and SMOLDER, each flagging at its
    best-F2 threshold of 2019. Warm colours are fire, the cool colour is the flagged
    area: fire of the next 3 days dark red inside the flagged area (caught) and orange
    outside it (missed); fire of the previous 3 days as a dark outline on top."""
    year = int(os.environ.get("YEAR", 2020)); z = np.load(os.environ["EVENT_CACHE"])
    OCEAN, LAND, FLAG, CAUGHT, MISSED = "#FFFFFF", "#ECECEE", "#BCD4EA", "#9E2A2B", "#EE9B3B"
    PREV, COAST = "#2E2E33", "#8A8A90"
    cmap = ListedColormap([OCEAN, LAND, FLAG, CAUGHT, MISSED]); S = 192
    LON0, LAT0, PX = 112.904998779, -9.005000113999998, 0.01

    def rgb(land, y, flag):
        c = np.where(land, 1, 0)
        c = np.where(flag & ~y, 2, c)
        c = np.where(flag & y, 3, c)
        return np.where(~flag & y, 4, c)

    for kind in ("well", "poorly"):
        n = sum(1 for k in z.files if k.startswith(kind) and k.endswith("_meta"))
        fig = plt.figure(figsize=(W, 3.4))
        pw, ph, gap, x_ev, y_row = 0.170, 0.170 * W / 3.4, 0.01, (0.11, 0.54), (0.555, 0.085)
        for i in range(n):
            p = f"{kind}{i}_"; D, y0, x0, _ = z[p + "meta"]
            land, recent, y = z[p + "land"], z[p + "recent"], z[p + "y"]
            lat = LAT0 - (y0 + S / 2) * PX; lon = LON0 + (x0 + S / 2) * PX; nf = int(y.sum())
            col, row = i % 2, i // 2
            fig.text(x_ev[col], y_row[row] + ph + 0.075, f"{str(z[p + 'date'])}, {abs(lat):.1f}°S {lon:.1f}°E, "
                     f"{nf:,} fire pixels in the next 3 days", fontsize=6, color=INK, fontweight="bold")
            for j, (c, title) in enumerate((("persistence", "Persistence"), ("full", "SMOLDER"))):
                ax = fig.add_axes([x_ev[col] + j * (pw + gap), y_row[row], pw, ph])
                flag = z[p + c]
                ax.imshow(rgb(land, y, flag), cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest")
                ax.contour(land.astype(float), levels=[0.5], colors=COAST, linewidths=0.5)
                if recent.any():
                    ax.contour(recent.astype(float), levels=[0.5], colors=PREV, linewidths=0.45)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color("#9A9AA0"); sp.set_linewidth(0.5)
                hit = int((flag & y).sum()); fa = int((flag & ~y).sum())
                ax.set_title(f"{title}\n{100 * hit / max(nf, 1):.0f} % caught, {fa:,} false alarms", fontsize=6,
                             color=INK, loc="left", pad=2)
                if j == 0:
                    ax.plot([8, 58], [S - 10, S - 10], color=INK, lw=1.0)
                    ax.text(33, S - 14, "50 km", ha="center", va="bottom", fontsize=6, color=INK)
        handles = [Patch(facecolor="none", edgecolor=PREV, lw=0.8, label="fire on the previous 3 days"),
                   Patch(color=CAUGHT, label="fire in the next 3 days, flagged"),
                   Patch(color=MISSED, label="fire in the next 3 days, not flagged"),
                   Patch(color=FLAG, label="flagged, no fire"),
                   matplotlib.lines.Line2D([], [], color=COAST, lw=0.8, label="coastline")]
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=6, bbox_to_anchor=(0.5, 0.0),
                   handlelength=1.2, columnspacing=1.0)
        save(fig, f"fig_events_{kind}_{year}.png", 3.4)


if __name__ == "__main__":
    national_skill(); budget_curves(); distance_bands(); importance(); events()
