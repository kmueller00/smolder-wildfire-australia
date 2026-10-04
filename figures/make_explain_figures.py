"""Figures for 'why is a pixel in the top 1 %' (smolder.evaluation.explain_topk).

  python make_explain_figures.py --summarize PIXELS.csv.gz
        condenses the per-pixel sample into results/explain_2020_summary.json
  python make_explain_figures.py [--pixels PIXELS.csv.gz]
        draws fig_explain_importance.png and fig_explain_prefire.png from
        results/explain_2020*.json, and fig_explain_conditions.png (violins)
        from the per-pixel sample when --pixels is given
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from style_smolder import ACCENT, ACCENT2, INK, MUTED, PANEL_BG, new_figure, style_axes

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
CLASSES = ["hit", "false_alarm", "miss", "background"]
CLASS_LABEL = {"hit": "top 1 %, burned", "false_alarm": "top 1 %, no fire",
               "miss": "burned, not top 1 %", "background": "neither"}
CLASS_COLOR = {"hit": "#00A83A", "false_alarm": "#E07A1A", "miss": "#9B30D9", "background": "#8C8C90"}
CLIMATES = ["tropical", "arid", "temperate"]
LANDCOVERS = ["grassland", "shrubland", "open forest", "closed forest", "cropland"]
SCALARS = [("dist_recent_fire_km", "Distance to recent fire (km)"),
           ("soil_moisture", "Soil moisture index"),
           ("precip_32d", "Precipitation, last 32 days (mm)"),
           ("vpd_7d", "Vapour pressure deficit, last 7 days (kPa)"),
           ("lai", "Leaf area index"),
           ("biomass", "Above-ground biomass (Mg/ha)")]
TRAJ = [("sm", 18, "Soil moisture index", 8), ("ppt", 18, "Precipitation per 8 days (mm)", 8),
        ("lai", 18, "Leaf area index", 8), ("vpd", 14, "Vapour pressure deficit (kPa)", 1)]


def summarize(csv):
    import pandas as pd
    p = pd.read_csv(csv)
    p.loc[p.dist_recent_fire_km.isna(), "dist_recent_fire_km"] = 70.0     # beyond the feature's range
    out = {"n_pixels": int(len(p)), "counts": {}, "scalars": {}, "trajectories": {}}
    out["counts"] = {f"{c}|{k}": int(v) for (c, k), v in p.groupby(["cls", "climate"]).size().items()}
    for var, _ in SCALARS:
        q = p.groupby(["cls", "climate"])[var].quantile([0.25, 0.5, 0.75]).unstack()
        out["scalars"][var] = {f"{c}|{k}": [float(r[0.25]), float(r[0.5]), float(r[0.75])]
                               for (c, k), r in q.iterrows()}
    p["fire"] = np.where(p.cls.isin(["hit", "miss"]), "fire", np.where(p.cls == "background", "no fire", "other"))
    for prefix, n, _, _ in TRAJ:
        cols = [f"{prefix}_{'d' if prefix == 'vpd' else 'b'}{k:02d}" for k in range(n)]
        out["trajectories"][prefix] = {}
        for (f, lc), g in p[p.fire != "other"].groupby(["fire", "landcover"]):
            if len(g) < 200:
                continue
            v = g[cols].values
            out["trajectories"][prefix][f"{f}|{lc}"] = dict(
                q25=np.nanpercentile(v, 25, axis=0).tolist(), q50=np.nanmedian(v, axis=0).tolist(),
                q75=np.nanpercentile(v, 75, axis=0).tolist(), n=int(len(g)))
    with open(os.path.join(RES, "explain_2020_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote results/explain_2020_summary.json")


def importance():
    E = json.load(open(os.path.join(RES, "explain_2020.json")))
    G = E["groups"]
    names = sorted(G, key=lambda n: G[n]["retention"])
    change = np.array([1 - G[n]["retention"] for n in names])
    ft = np.array([np.std(1 - np.array(E["folds"]["time"]["groups"][n]["retention"])) for n in names])
    fs = np.array([np.std(1 - np.array(E["folds"]["space"]["groups"][n]["retention"])) for n in names])

    fig = new_figure((16.0, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.25, 0.95], wspace=0.55)
    a = fig.add_subplot(gs[0])
    ypos = np.arange(len(names))[::-1]
    a.barh(ypos, change, height=0.62, color=[ACCENT2 if n == "fire history" else ACCENT for n in names], zorder=3)
    a.errorbar(change, ypos, xerr=np.maximum(ft, fs), fmt="none", ecolor=INK, elinewidth=1.1, capsize=3, zorder=4)
    for y, v, e in zip(ypos, change, np.maximum(ft, fs)):
        a.text((v + e) * 1.25, y, f"{100*v:.1f} %", va="center", fontsize=8.6, color=INK)
    a.set_xscale("log"); a.set_xlim(0.005, 4)
    a.set_xticks([0.01, 0.03, 0.1, 0.3, 1]); a.set_xticklabels(["1 %", "3 %", "10 %", "30 %", "100 %"])
    a.set_yticks(ypos); a.set_yticklabels(names)
    a.set_xlabel("Top-1 % pixels displaced when the input is shuffled", fontsize=9.8, fontweight="bold", color=INK)
    a.set_title("(a) Importance for the top-1 % selection", fontsize=11.5, fontweight="bold", color=INK, loc="left")
    style_axes(a, grid_y=False)

    cmap = LinearSegmentedColormap.from_list("imp", [PANEL_BG, "#FFE9A8", "#FFAB3D", "#E8452C", "#8B0000"])
    for k, (key, cats, title) in enumerate((("by_landcover", LANDCOVERS, "(b) By land cover"),
                                            ("by_climate", CLIMATES, "(c) By climate zone"))):
        ax = fig.add_subplot(gs[k + 1])
        M = np.array([[1 - G[n][key].get(c, np.nan) for c in cats] for n in names])
        ax.imshow(np.log10(np.clip(M, 0.005, 1)), cmap=cmap, vmin=np.log10(0.005), vmax=0, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{100*M[i, j]:.0f}" if M[i, j] >= 0.1 else f"{100*M[i, j]:.1f}",
                            ha="center", va="center", fontsize=8.4,
                            color="white" if M[i, j] > 0.3 else INK)
        ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names if k == 0 else [], fontsize=9)
        ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle("What decides the top-1 % risk area", fontsize=14, fontweight="bold", color=INK, y=1.02)
    fig.savefig(os.path.join(HERE, "fig_explain_importance.png"), dpi=250, bbox_inches="tight", facecolor="white")
    print("wrote fig_explain_importance.png")


OUTCOMES = [("hit", "TP", "true positive (flagged, burned)", "#0E8C6E"),
            ("false_alarm", "FP", "false positive (flagged, no fire)", "#DB8A0A"),
            ("miss", "FN", "false negative (not flagged, burned)", "#C2185B"),
            ("background", "TN", "true negative (not flagged, no fire)", "#7A869A")]
DIST_CAP = 70.0


def conditions(pixels):
    """Violins of the issue-day conditions per forecast outcome and climate zone,
    from the per-pixel sample written by smolder.evaluation.explain_topk."""
    import pandas as pd
    from scipy.stats import gaussian_kde
    cols = ["cls", "climate"] + [v for v, _ in SCALARS]
    p = pd.read_csv(pixels, usecols=cols)
    p.loc[p.dist_recent_fire_km.isna(), "dist_recent_fire_km"] = DIST_CAP
    rng = np.random.default_rng(0)
    group_w, half = 1.0, 0.105                 # climate group spacing, max violin half-width
    offs = (np.arange(len(OUTCOMES)) - 1.5) * 0.23

    fig = new_figure((16.0, 9.0))
    for k, (var, label) in enumerate(SCALARS):
        ax = fig.add_subplot(2, 3, k + 1)
        pooled = p[var].dropna().values
        lo, hi = np.percentile(pooled, [1, 99])
        pad = 0.04 * (hi - lo)
        for gi, clim in enumerate(CLIMATES):
            for oi, (cls, short, _, col) in enumerate(OUTCOMES):
                v = p.loc[(p.cls == cls) & (p.climate == clim), var].dropna().values
                if v.size < 20:
                    continue
                x0 = gi * group_w + offs[oi]
                p01, p99 = np.percentile(v, [1, 99])     # violin cut at the group's own 1st-99th percentile
                a, b = max(p01, lo - pad), min(p99, hi + pad)
                if b > a and np.ptp(v) > 0:
                    sub = v if v.size <= 20000 else rng.choice(v, 20000, replace=False)
                    ys = np.linspace(a, b, 200)
                    dens = gaussian_kde(sub)(ys)
                    w = half * dens / dens.max()
                    ax.fill_betweenx(ys, x0 - w, x0 + w, color=col, alpha=0.45, lw=0, zorder=3)
                    ax.plot(np.r_[x0 - w, (x0 + w)[::-1], x0 - w[0]], np.r_[ys, ys[::-1], ys[0]],
                            color=col, lw=0.7, zorder=3)
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                ax.plot([x0, x0], [q1, q3], color=INK, lw=1.0, zorder=4)
                ax.plot([x0 - half * 0.7, x0 + half * 0.7], [med, med], color=col, lw=2.6, zorder=5,
                        solid_capstyle="butt")
                ax.text(x0, -0.025, short, transform=ax.get_xaxis_transform(), ha="center", va="top",
                        fontsize=8, color=col, fontweight="bold")
        ax.set_xlim(-0.55, (len(CLIMATES) - 1) * group_w + 0.55)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks([gi * group_w for gi in range(len(CLIMATES))])
        ax.set_xticklabels(CLIMATES, fontsize=9.5)
        ax.tick_params(axis="x", pad=24, length=0)
        ax.set_title(label, fontsize=10.5, fontweight="bold", color=INK, loc="left")
        style_axes(ax, grid_x=False)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.6, edgecolor=c) for *_, c in OUTCOMES]
    fig.legend(handles, [f"{s}: {lbl}" for _, s, lbl, _ in OUTCOMES], loc="lower center", ncol=2,
               fontsize=10, frameon=True, facecolor="white", edgecolor=MUTED, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Conditions on the issue day by forecast outcome", fontsize=14, fontweight="bold",
                 color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=2.6)
    fig.savefig(os.path.join(HERE, "fig_explain_conditions.png"), dpi=250, bbox_inches="tight", facecolor="white")
    print("wrote fig_explain_conditions.png")
    return p


def prefire():
    S = json.load(open(os.path.join(RES, "explain_2020_summary.json")))["trajectories"]
    lcs = [lc for lc in LANDCOVERS if all(f"{f}|{lc}" in S["sm"] for f in ("fire", "no fire"))]
    fig = new_figure((16.0, 2.6 * len(lcs) + 0.8))
    for r, lc in enumerate(lcs):
        for c, (prefix, n, label, step) in enumerate(TRAJ):
            ax = fig.add_subplot(len(lcs), len(TRAJ), r * len(TRAJ) + c + 1)
            x = -(np.arange(n)[::-1]) * step - (step if step > 1 else 0)
            for f, col, ls in (("fire", ACCENT2, "-"), ("no fire", ACCENT, "--")):
                t = S[prefix][f"{f}|{lc}"]
                ax.plot(x, t["q50"], ls, color=col, lw=2.0, zorder=3, label="burned" if f == "fire" else "did not burn")
            ax.set_xlim(x[0], x[-1])
            if r == 0:
                ax.set_title(label, fontsize=10.5, fontweight="bold", color=INK, loc="left")
            if c == 0:
                ax.set_ylabel(lc, fontsize=10.5, fontweight="bold", color=INK)
            if r == len(lcs) - 1:
                ax.set_xlabel("Days before the issue day", fontsize=9.5, color=INK)
            style_axes(ax)
            if r == 0 and c == 0:
                ax.legend(fontsize=8.8, loc="upper left", frameon=True, facecolor="white", edgecolor=MUTED)
    fig.suptitle("What precedes fire: inputs over the model's look-back window, by land cover", fontsize=14,
                 fontweight="bold", color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0.01, 1, 0.99))
    fig.savefig(os.path.join(HERE, "fig_explain_prefire.png"), dpi=220, bbox_inches="tight", facecolor="white")
    print("wrote fig_explain_prefire.png")


if __name__ == "__main__":
    if "--summarize" in sys.argv:
        summarize(sys.argv[sys.argv.index("--summarize") + 1])
    else:
        importance()
        if "--pixels" in sys.argv:
            conditions(sys.argv[sys.argv.index("--pixels") + 1])
        else:
            print("skipped fig_explain_conditions.png (needs --pixels explain_2020_pixels.csv.gz)")
        prefire()
