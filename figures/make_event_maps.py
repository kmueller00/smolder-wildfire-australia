"""Zoomed-in 2019 fire events: where SMOLDER works exceptionally well, and
where it does not -- SMOLDER and the persistence baseline side by side at the
same national alert budget.

Every ranking flags its national top 0.13 % of land pixels per issue day (SMOLDER's
best-F2 budget, results/experiments/smolder/operating_point_2019.md).
For each 192 x 192 km window and day the hits, false alarms and misses of
every ranking are counted; events are then chosen from fire-active windows
(>= MIN_FIRE target fire pixels):
  well     highest F2 of SMOLDER above persistence, SMOLDER F2 >= 0.5
  poorly   lowest SMOLDER F2 (<= 0.25), most target fire first
with at most one event per 30 days and 300 km, so each row is a different fire.

Two steps:
  python make_event_maps.py --compute   needs data + score files (SMOLDER_DATA);
                                        writes data/event_maps_2019.npz
  python make_event_maps.py             plots from that cached file only
  python make_event_maps.py --table     per-event numbers, fixed and adaptive area
                                        (results/experiments/smolder/event_maps_2019.json)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from style_smolder import INK, PANEL_BG

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "event_maps_2019.npz")
OUTS = {"well": os.path.join(HERE, "fig_events_well_2019.png"),
        "poorly": os.path.join(HERE, "fig_events_poorly_2019.png")}
LON0, LAT0, PX = 112.904998779, -9.005000113999998, 0.01
BUDGET = 0.0013
BLOCK, WIN = 64, 3                        # windows of 3 x 3 blocks = 192 px, stride 64 px
MIN_FIRE = 400
N_EV = 4
SEP_DAYS, SEP_PX = 30, 300
RANKINGS = ["persistence", "full"]
TITLES = {"persistence": "Persistence", "full": "SMOLDER"}
SCORE_DIR = "/home/saturn/gwgi/gwgi107h/wildfire_data/firecastnet/experiments_scores"
SCORES = {"full": f"{SCORE_DIR}/full_model_seed123_2019.npy"}

OCEAN = "#FFFFFF"
LAND = "#ECECEE"
RECENT = "#505055"          # fire on days D-2..D (what persistence knows)
HIT = "#2166AC"             # flagged, and fire followed
MISS = "#B2182B"            # fire followed, not flagged
FALSE = "#9EC3E0"           # flagged, no fire followed
G = {}


# ------------------------------------------------------------------ compute
def _init():
    from smolder.data.io import daily_cube, open_zarr_root
    g = open_zarr_root(daily_cube(2019))
    G["y"] = g["y_fire_3d"]
    G["land"] = np.asarray(g["landmask"][...]) > 0
    G["scores"] = {n: np.load(p, mmap_mode="r") for n, p in SCORES.items()}


def _maps(D, row):
    """Target, recent fire, distance to it, and each ranking's flagged map for day D."""
    from scipy import ndimage
    from smolder.evaluation.evaluate_persistence import persistence_score
    land = G["land"]
    y = (np.asarray(G["y"][D]) > 0) & land
    recent = (np.asarray(G["y"][D - 3]) > 0) & land
    sb, d = persistence_score(recent, np.random.default_rng([0, D]))
    flags = {}
    for n in RANKINGS:
        s = sb[land] if n == "persistence" else np.asarray(G["scores"][n][row], np.float32)
        k = max(1, int(round(BUDGET * s.size)))
        thr = np.partition(s, s.size - k)[s.size - k]
        f = np.zeros(land.shape, bool)
        f[land] = s >= thr
        flags[n] = f
    prob = np.full(land.shape, np.nan, np.float32)
    prob[land] = np.asarray(G["scores"]["full"][row], np.float32)
    return y, recent, d, flags, prob


def _blocks(a):
    H, W = (a.shape[0] // BLOCK) * BLOCK, (a.shape[1] // BLOCK) * BLOCK
    return a[:H, :W].reshape(H // BLOCK, BLOCK, W // BLOCK, BLOCK).sum(axis=(1, 3))


def _day(args):
    D, row = args
    y, recent, d, flags, _ = _maps(D, row)
    out = {"fire": _blocks(y), "far": _blocks(y & (d > 10))}
    for n, f in flags.items():
        out[f"hit_{n}"] = _blocks(f & y)
        out[f"fa_{n}"] = _blocks(f & ~y)
    return D, row, out


def _win(b):
    """Sum over WIN x WIN blocks for every window position."""
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(b, (WIN, WIN)).sum(axis=(-1, -2))


def _f2(h, fa, fire):
    rec = h / np.maximum(fire, 1)
    prec = h / np.maximum(h + fa, 1)
    return np.where(h > 0, 5 * prec * rec / np.maximum(4 * prec + rec, 1e-9), 0.0)


def compute():
    from multiprocessing import Pool
    sys.path.insert(0, os.path.dirname(HERE))
    from smolder.data.io import daily_cube, open_zarr_root
    days = np.load(SCORES["full"] + ".days.npy")
    times = list(open_zarr_root(daily_cube(2019)).attrs["time"])
    with Pool(int(os.environ.get("WORKERS", 24)), initializer=_init) as pool:
        res = list(pool.imap(_day, [(int(D), i) for i, D in enumerate(days)], chunksize=2))
    cand = []                                  # (kind, sort key, D, row, wy, wx, stats)
    for D, row, b in res:
        w = {k: _win(v) for k, v in b.items()}
        fire = w["fire"]
        f2 = {n: _f2(w[f"hit_{n}"], w[f"fa_{n}"], fire) for n in RANKINGS}
        ok = fire >= MIN_FIRE
        for wy, wx in zip(*np.nonzero(ok)):
            st = dict(fire=int(fire[wy, wx]), far=int(w["far"][wy, wx]),
                      **{f"f2_{n}": float(f2[n][wy, wx]) for n in RANKINGS},
                      **{f"hit_{n}": int(w[f"hit_{n}"][wy, wx]) for n in RANKINGS},
                      **{f"fa_{n}": int(w[f"fa_{n}"][wy, wx]) for n in RANKINGS})
            if st["f2_full"] >= 0.5:
                cand.append(("well", st["f2_full"] - st["f2_persistence"], D, row, wy, wx, st))
            if st["f2_full"] <= 0.25:
                cand.append(("poorly", st["fire"], D, row, wy, wx, st))
    chosen = {"well": [], "poorly": []}
    for kind in chosen:
        for c in sorted([c for c in cand if c[0] == kind], key=lambda c: -c[1]):
            y0, x0 = c[4] * BLOCK, c[5] * BLOCK
            if all(abs(c[2] - o[2]) >= SEP_DAYS or max(abs(y0 - o[4] * BLOCK), abs(x0 - o[5] * BLOCK)) >= SEP_PX
                   for o in chosen[kind]):
                chosen[kind].append(c)
            if len(chosen[kind]) == N_EV:
                break
    _init()
    out = {}
    S = WIN * BLOCK
    for kind, evs in chosen.items():
        for i, (_, key, D, row, wy, wx, st) in enumerate(evs):
            y, recent, d, flags, prob = _maps(D, row)
            y0, x0 = wy * BLOCK, wx * BLOCK
            sl = (slice(y0, y0 + S), slice(x0, x0 + S))
            p = f"{kind}{i}_"
            out[p + "y"], out[p + "recent"], out[p + "land"] = y[sl], recent[sl], G["land"][sl]
            out[p + "prob"] = prob[sl]
            for n in RANKINGS:
                out[p + n] = flags[n][sl]
            out[p + "meta"] = np.array([D, y0, x0, key])
            out[p + "date"] = np.array(str(times[D])[:10])
            out[p + "stats"] = np.array(str(st))
            print(kind, i, str(times[D])[:10], y0, x0, st, flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **out)
    print("wrote", CACHE)


# --------------------------------------------------------------------- plot
def _rgb(land, recent, y, flag):
    """Category map: 0 ocean, 1 land, 2 recent fire, 3 hit, 4 miss, 5 false alarm."""
    c = np.where(land, 1, 0)
    c = np.where(recent, 2, c)
    if flag is None:
        c = np.where(y, 4, c)
    else:
        c = np.where(flag & ~y, 5, c)
        c = np.where(flag & y, 3, c)
        c = np.where(~flag & y, 4, c)
    return c


def plot():
    z = np.load(CACHE)
    cmap = ListedColormap([OCEAN, LAND, RECENT, HIT, MISS, FALSE])
    S = WIN * BLOCK
    for kind, out in OUTS.items():
        n = sum(1 for k in z.files if k.startswith(kind) and k.endswith("_meta"))
        fig, axes = plt.subplots(n, 1 + len(RANKINGS), figsize=(10.6, 3.65 * n), facecolor="white")
        axes = np.atleast_2d(axes)
        for i in range(n):
            p = f"{kind}{i}_"
            D, y0, x0, _ = z[p + "meta"]
            land, recent, y = z[p + "land"], z[p + "recent"], z[p + "y"]
            lat = LAT0 - (y0 + S / 2) * PX
            lon = LON0 + (x0 + S / 2) * PX
            n_fire = int(y.sum())
            for j, col in enumerate(["obs"] + RANKINGS):
                ax = axes[i, j]
                flag = None if col == "obs" else z[p + col]
                ax.imshow(_rgb(land, recent, y, flag), cmap=cmap, vmin=-0.5, vmax=5.5, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_color("#9A9AA0")
                if col == "obs":
                    ax.set_title(f"{str(z[p + 'date'])}  ({abs(lat):.1f}°S, {lon:.1f}°E)\n"
                                 f"fire next 3 days: {n_fire:,} px", fontsize=10, color=INK, loc="left")
                    ax.set_ylabel(f"Event {i + 1}", fontsize=11, color=INK)
                else:
                    hit = int((flag & y).sum()); fa = int((flag & ~y).sum())
                    rec = hit / max(n_fire, 1)
                    ax.set_title(f"{TITLES[col]}\ncaught {100 * rec:.0f} %, {fa:,} false alarms",
                                 fontsize=10, color=INK, loc="left")
            axes[i, 0].plot([8, 8 + 50], [S - 10, S - 10], color=INK, lw=2)
            axes[i, 0].text(8 + 25, S - 14, "50 km", ha="center", va="bottom", fontsize=8, color=INK,
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
        handles = [Patch(color=RECENT, label="fire on issue day and 2 days before (D-2..D)"),
                   Patch(color=HIT, label="flagged, fire followed (hit)"),
                   Patch(color=MISS, label="fire followed, not flagged (miss; left: all fire D+1..D+3)"),
                   Patch(color=FALSE, label="flagged, no fire followed (false alarm)")]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("wrote", out)


def event_table():
    """Per event: fire caught and false alarms of each ranking at the fixed
    0.13 % and of SMOLDER with the adaptive threshold of the same mean area
    (results/experiments/smolder/adaptive_budget_2019.json); written next to
    that file as event_maps_2019.json."""
    import json
    res = os.path.join(HERE, "..", "results", "experiments", "smolder")
    thr = json.load(open(os.path.join(res, "adaptive_budget_2019.json")))["adaptive"]["SMOLDER"][str(BUDGET)]["threshold"]
    z = np.load(CACHE)
    out = dict(budget=BUDGET, adaptive_logit_threshold=thr, events={})
    for kind in OUTS:
        n = sum(1 for k in z.files if k.startswith(kind) and k.endswith("_meta"))
        for i in range(n):
            p = f"{kind}{i}_"
            y, land = z[p + "y"], z[p + "land"]
            pr = np.clip(z[p + "prob"].astype(np.float64), 1e-7, 1 - 1e-7)
            flags = {n_: z[p + n_] for n_ in RANKINGS}
            flags["full_adaptive"] = (np.log(pr / (1 - pr)) >= thr) & land
            out["events"][f"{kind} {i + 1}"] = dict(date=str(z[p + "date"]), fire_px=int(y.sum()), **{
                n_: dict(caught=float((f & y).sum() / max(y.sum(), 1)), false_alarms=int((f & ~y).sum()))
                for n_, f in flags.items()})
    path = os.path.join(res, "event_maps_2019.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    if "--compute" in sys.argv:
        compute()
    elif "--table" in sys.argv:
        event_table()
    else:
        plot()
