"""Figures for 'why does SMOLDER flag a pixel' (smolder.evaluation.explain_topk).
Flagged = above the adaptive score threshold chosen on 2019 (best F2) when the
explain run used THRESHOLD_FROM, else the top 1 % of each patch.

  python make_explain_figures.py --summarize PIXELS.csv.gz
        condenses the per-pixel sample into results/explain_2020_summary.json
  python make_explain_figures.py [--pixels PIXELS.csv.gz]
        draws fig_explain_importance.png from results/explain_2020*.json and,
        when --pixels is given, the paper figures (6.93 in, 450 dpi)
          fig_explain_conditions.png  (a) distance to recent fire by outcome,
                                      (b-e) issue-day conditions by outcome and climate
          fig_explain_prefire.png     burned minus not burned in the same patch
                                      over the look-back window, 95 % interval
        plus explain_figures_numbers.json with the plotted numbers.
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
RES = os.environ.get("EXPLAIN_DIR", os.path.join(HERE, "..", "results"))       # a run's explain_2020*
FIG_OUT = os.environ.get("FIG_OUT", HERE)
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
    path = os.path.join(RES, os.environ.get("EXPLAIN_JSON", "explain_2020.json"))
    if not os.path.exists(path) or os.environ.get("SKIP_IMPORTANCE") == "1":
        print("skipped the large importance figure (paper version: make_paper_figures.py)")
        return
    E = json.load(open(path))
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
    fig.savefig(os.path.join(FIG_OUT, "fig_explain_importance.png"), dpi=250, bbox_inches="tight", facecolor="white")
    print("wrote fig_explain_importance.png")


# ---------------------------------------------------------------- paper figures
# Print size as make_paper_figures.py: 6.93 in wide, 450 dpi, no text below 6 pt,
# no title above the panels. Outcomes are named in words.
PW, PDPI, PFS = 6.93, 450, 6.5
OUTCOMES = [("hit", "Caught", "flagged, burned", "#0E8C6E"),
            ("miss", "Missed", "burned, not flagged", "#C2185B"),
            ("false_alarm", "False alarm", "flagged, no fire", "#DB8A0A"),
            ("background", "Other land", "not flagged, no fire", "#7A869A")]
DIST_BANDS = [("within 3 km", 0, 3, "#3B3B40"), ("3 to 10 km", 3, 10, "#8E8E95"),
              ("more than 10 km", 10, np.inf, "#D3D3D8")]
COND_VARS = [("soil_moisture", "Soil moisture index", 1.0, "{:.2f}"),
             ("precip_32d", "Rain, last 32 days (mm)", 1.0, "{:.0f}"),
             ("vpd_7d", "Vapour pressure deficit,\nlast 7 days (kPa)", 1.0, "{:.1f}"),
             ("lai", "Leaf area index", 1.0, "{:.1f}")]


def _paper_rc():
    plt.rcParams.update({"font.size": PFS, "axes.labelsize": PFS, "xtick.labelsize": PFS, "ytick.labelsize": PFS,
                         "legend.fontsize": 6.2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "lines.linewidth": 1.2})


def _paper_axes(ax, **kw):
    style_axes(ax, **kw)
    ax.tick_params(labelsize=PFS, length=2.5, width=0.6, pad=2)


def _tag(ax, t, x=-0.12, y=1.03):
    ax.text(x, y, f"({t})", transform=ax.transAxes, fontsize=7.5, fontweight="bold", color=INK, va="bottom")


def _weights(p):
    """Weight of each sampled pixel so that every class counts with its true
    size in its patch: cls_n / pixels drawn (1 when cls_n is absent)."""
    if "cls_n" not in p:
        return np.ones(len(p))
    drawn = p.groupby(["patch", "cls"]).cls.transform("size").values
    return p.cls_n.values / drawn


def _wquantile(v, w, q):
    o = np.argsort(v); v, w = v[o], w[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(q, c, v)


def conditions(pixels):
    """(a) Distance to recent fire by forecast outcome, as stacked shares.
    (b-e) Issue-day conditions per outcome and climate zone: median (dot) and
    middle half (bar) of the pixels, weighted to the true class sizes."""
    import pandas as pd
    _paper_rc()
    p = pd.read_csv(pixels, usecols=lambda c: c in {"cls", "cls_n", "patch", "climate", "dist_recent_fire_km"}
                    | {v for v, *_ in COND_VARS})
    p["w"] = _weights(p)
    d = p.dist_recent_fire_km.fillna(np.inf).values           # NaN: beyond the feature's range (~70 km)
    fig = plt.figure(figsize=(PW, 3.5))

    # (a) where each outcome lies relative to fire already burning
    a = fig.add_axes([0.16, 0.70, 0.60, 0.26])
    yy = np.arange(len(OUTCOMES))[::-1]
    shares = {}
    for (cls, name, desc, col), y in zip(OUTCOMES, yy):
        m = (p.cls == cls).values
        w = p.w.values[m]; left = 0.0
        shares[cls] = []
        for lab, lo, hi, bc in DIST_BANDS:
            sh = 100 * w[(d[m] >= lo) & ((d[m] < hi) | np.isinf(hi))].sum() / w.sum()
            shares[cls].append(sh)
            a.barh(y, sh, left=left, height=0.66, color=bc, edgecolor="white", lw=0.4, zorder=3)
            if sh >= 7:
                a.text(left + sh / 2, y, f"{sh:.0f} %", ha="center", va="center", fontsize=6,
                       color="white" if bc != "#D3D3D8" else INK, zorder=4)
            left += sh
    a.set_yticks(yy)
    a.set_yticklabels([f"{name}\n({desc})" for _, name, desc, _ in OUTCOMES], fontsize=6, linespacing=0.95)
    for t, (_, _, _, col) in zip(a.get_yticklabels(), OUTCOMES):
        t.set_color(col); t.set_fontweight("bold")
    a.set_xlim(0, 100); a.set_xlabel("Share of pixels (%)", labelpad=1)
    _paper_axes(a, grid_y=False)
    a.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=bc) for *_, bc in DIST_BANDS],
             labels=[lab for lab, *_ in DIST_BANDS], title="Distance to fire of the\nissue day and 2 days before",
             title_fontsize=6.2, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, handlelength=1.2)
    _tag(a, "a", x=-0.25)

    # (b-e) conditions on the issue day
    x0, wd, gap = 0.085, 0.2, 0.03
    ydot = {"tropical": 2, "arid": 1, "temperate": 0}
    off = (np.arange(len(OUTCOMES)) - 1.5) * 0.17
    quartiles = {}                                         # var -> climate -> outcome -> [q25, median, q75, n]
    for k, (var, label, scale, fmt) in enumerate(COND_VARS):
        ax = fig.add_axes([x0 + k * (wd + gap), 0.11, wd, 0.40])
        for clim, yc in ydot.items():
            for o, (cls, name, _, col) in enumerate(OUTCOMES):
                m = ((p.cls == cls) & (p.climate == clim)).values & np.isfinite(p[var].values)
                if m.sum() < 30:
                    continue
                q1, q2, q3 = _wquantile(p[var].values[m] * scale, p.w.values[m], [0.25, 0.5, 0.75])
                quartiles.setdefault(var, {}).setdefault(clim, {})[cls] = [float(q1), float(q2), float(q3), int(m.sum())]
                yv = yc - off[o]
                ax.plot([q1, q3], [yv, yv], color=col, lw=1.6, alpha=0.45, solid_capstyle="butt", zorder=3)
                ax.plot(q2, yv, "o", ms=3.2, color=col, zorder=4)
        ax.set_ylim(-0.5, 2.5)
        ax.set_yticks(list(ydot.values()))
        ax.set_yticklabels([c if k == 0 else "" for c in ydot], fontsize=6.5)
        if k > 0:
            ax.tick_params(axis="y", length=0)
        ax.set_xlabel(label, labelpad=1)
        ax.set_xlim(left=0)
        _paper_axes(ax, grid_y=False)
        for yb in (0.5, 1.5):
            ax.axhline(yb, color="white", lw=1.0, zorder=1)
        _tag(ax, "bcde"[k], x=-0.33 if k == 0 else -0.06)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=col, marker="o", ms=3.2, lw=1.6, alpha=1) for *_, col in OUTCOMES],
               labels=[name for _, name, _, _ in OUTCOMES], loc="lower left", bbox_to_anchor=(0.075, 0.535),
               ncol=4, frameon=False, handlelength=1.6, columnspacing=1.4,
               title="Conditions on the issue day (dot: median; bar: middle half of the pixels)",
               title_fontsize=6.5, alignment="left")
    fig.savefig(os.path.join(FIG_OUT, "fig_explain_conditions.png"), dpi=PDPI, facecolor="white")
    plt.close(fig)
    print("wrote fig_explain_conditions.png; distance shares (%):",
          {c: [round(v, 1) for v in s_] for c, s_ in shares.items()})
    return shares, quartiles


TRAJ_PAPER = [("sm", 18, "Soil moisture index", "slow"), ("ppt", 18, "Rain per 8 days (mm)", "slow"),
              ("lai", 18, "Leaf area index", "lai"), ("vpd", 14, "Vapour pressure deficit (kPa)", "fast")]
LAI_LAG = 31


def prefire(pixels, n_boot=1000):
    """Burned pixels minus not-burned pixels of the same patch (same place and
    issue day), over the model's look-back window. Per patch: weighted mean of
    the burned sample (caught + missed, weighted to their true sizes) minus the
    mean of the not-flagged, not-burned sample; then the mean over patches,
    weighted by the patch's burned pixels, with a 95 % bootstrap interval over
    patches. Removes differences of region and season."""
    import pandas as pd
    _paper_rc()
    p = pd.read_csv(pixels)
    p["w"] = _weights(p)
    burned = p.cls.isin(["hit", "miss"]).values; ctrl = (p.cls == "background").values
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(PW, 1.6))
    x0, wd, gap = 0.075, 0.2, 0.04
    out = {}
    for k, (prefix, n, label, kind) in enumerate(TRAJ_PAPER):
        cols = [f"{prefix}_{'d' if prefix == 'vpd' else 'b'}{j:02d}" for j in range(n)]
        V = p[cols].values
        diffs, wts = [], []
        for pid in np.unique(p.patch.values):
            m = p.patch.values == pid
            b, c = m & burned, m & ctrl
            if b.sum() < 5 or c.sum() < 5:
                continue
            wb = p.w.values[b]
            mb = np.nansum(V[b] * wb[:, None], axis=0) / np.sum(np.isfinite(V[b]) * wb[:, None], axis=0)
            diffs.append(mb - np.nanmean(V[c], axis=0)); wts.append(wb.sum())
        D, Wt = np.array(diffs), np.array(wts)
        ok = np.isfinite(D)
        mean = np.nansum(D * Wt[:, None], axis=0) / np.sum(ok * Wt[:, None], axis=0)
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(D), len(D))
            boots.append(np.nansum(D[i] * Wt[i, None], axis=0) / np.sum(ok[i] * Wt[i, None], axis=0))
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        if kind == "fast":
            x = np.arange(-(n - 1), 1)                       # day 0 = issue day
        else:                                                # centre of each 8-day period; newest ends 0-7 days before
            x = -(np.arange(n)[::-1] * 8 + 7.5) - (LAI_LAG if kind == "lai" else 0)
        ax = fig.add_axes([x0 + k * (wd + gap), 0.2, wd, 0.69])
        ax.axhline(0, color=INK, lw=0.6, zorder=2)
        ax.fill_between(x, lo, hi, color=ACCENT2, alpha=0.25, lw=0, zorder=3)
        ax.plot(x, mean, color=ACCENT2, lw=1.4, zorder=4)
        ax.set_xlim(x[0], x[-1] if kind == "fast" else 0)
        ax.set_title(f"({'abcd'[k]}) {label}", fontsize=PFS, color=INK, loc="left", pad=3, x=-0.02)
        ax.set_xlabel("Days before the issue day", labelpad=1)
        if k == 0:
            ax.set_ylabel("Burned minus not burned", labelpad=1)
        _paper_axes(ax)
        out[prefix] = dict(x=x.tolist(), mean=mean.tolist(), ci95_lo=lo.tolist(), ci95_hi=hi.tolist(),
                           n_patches=int(len(D)))
    fig.savefig(os.path.join(FIG_OUT, "fig_explain_prefire.png"), dpi=PDPI, facecolor="white")
    plt.close(fig)
    print("wrote fig_explain_prefire.png")
    return out


def _prefire_figure(S, title, out, bins_end_on_issue_day=False):
    """Median input trajectories of burned vs not burned pixels by land cover.
    The model's slow bins are drawn at -144..-8 (their position in the
    original figure); bins that end on the issue day at -136..0."""
    lcs = [lc for lc in LANDCOVERS if all(f"{f}|{lc}" in S["sm"] for f in ("fire", "no fire"))]
    fig = new_figure((16.0, 2.6 * len(lcs) + 0.8))
    for r, lc in enumerate(lcs):
        for c, (prefix, n, label, step) in enumerate(TRAJ):
            ax = fig.add_subplot(len(lcs), len(TRAJ), r * len(TRAJ) + c + 1)
            x = -(np.arange(n)[::-1]) * step - (step if step > 1 and not bins_end_on_issue_day else 0)
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
    fig.suptitle(title, fontsize=14, fontweight="bold", color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0.01, 1, 0.99))
    fig.savefig(os.path.join(FIG_OUT, out), dpi=220, bbox_inches="tight", facecolor="white")
    print("wrote", out)


def prefire_newfire():
    """Same figure for pixels more than 10 km from any fire of the last 32 days
    (2019, results/anomaly_feature_diagnostic_2019.json)."""
    path = os.path.join(RES, "anomaly_feature_diagnostic_2019.json")
    if not os.path.exists(path):
        print("skipped fig_explain_prefire_newfire.png (needs", path, ")")
        return
    S = json.load(open(path))["trajectories_far_10km"]
    _prefire_figure(S, "What precedes new fire: pixels more than 10 km from fire of the last 32 days, 2019",
                    "fig_explain_prefire_newfire.png", bins_end_on_issue_day=True)


if __name__ == "__main__":
    if "--summarize" in sys.argv:
        summarize(sys.argv[sys.argv.index("--summarize") + 1])
    else:
        importance()
        if "--pixels" in sys.argv:
            px = sys.argv[sys.argv.index("--pixels") + 1]
            shares, quartiles = conditions(px)
            traj = prefire(px)
            with open(os.path.join(FIG_OUT, "explain_figures_numbers.json"), "w") as fh:
                json.dump(dict(pixels=os.path.abspath(px), distance_shares_pct=shares,
                               distance_bands=[b[0] for b in DIST_BANDS],
                               conditions_weighted_q25_median_q75_n=quartiles, prefire=traj), fh, indent=1)
        else:
            print("skipped fig_explain_conditions.png and fig_explain_prefire.png (need --pixels PIXELS.csv.gz)")
        prefire_newfire()
