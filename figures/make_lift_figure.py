"""Lift at each top-k fraction on the 2020 hold-out year (results/eval_2020.json,
written by smolder.evaluation.evaluate)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from style_smolder import ACCENT, ACCENT2, INK, MUTED, new_figure, style_axes

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "..", "results", "eval_2020.json")))
KS = np.array([t["k"] for t in R["topk"]]) * 100
LIFT_ALL = np.array([t["lift_all"] for t in R["topk"]])
LIFT_NEW = np.array([t["lift_new"] for t in R["topk"]])
TPR_ALL = np.array([t["tpr_all"] for t in R["topk"]])

fig = new_figure((10.5, 6.2))
ax = fig.add_subplot(111)
ax.plot(KS, LIFT_ALL, "o-", color=ACCENT, lw=2.2, ms=6, label="All fire", zorder=4)
ax.plot(KS, LIFT_NEW, "o-", color=ACCENT2, lw=2.2, ms=6,
        label="New fire (> 3 px from fire in the history window)", zorder=4)
ax.axhline(1.0, color=MUTED, lw=1.3, ls=(0, (4, 3)), zorder=2)
ax.text(0.0105, 1.12, "random", fontsize=8.6, color=MUTED, style="italic")

i5 = int(np.argmin(np.abs(KS - 0.5)))
for arr, col, dy in ((LIFT_ALL, ACCENT, 2.2), (LIFT_NEW, ACCENT2, 0.42)):
    ax.annotate(f"{arr[i5]:.1f}×", xy=(KS[i5], arr[i5]), xytext=(KS[i5] * 2.2, arr[i5] * dy),
                fontsize=9.5, fontweight="bold", color=col,
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2))

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.008, 13)
lo = max(0.1, min(np.nanmin(LIFT_NEW), np.nanmin(LIFT_ALL)) / 1.8)
hi = max(np.nanmax(LIFT_ALL), np.nanmax(LIFT_NEW)) * 1.8
ax.set_ylim(lo, hi)
ax.set_xticks([0.01, 0.1, 1, 10]); ax.set_xticklabels(["0.01", "0.1", "1", "10"])
yt = [v for v in (0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000) if lo <= v <= hi]
ax.set_yticks(yt); ax.set_yticklabels([f"{v:g}" for v in yt])
ax.set_xlabel("Land pixels flagged as highest risk, per patch (%)", fontsize=11,
              fontweight="bold", color=INK)
ax.set_ylabel("Lift over random (log scale)", fontsize=11, fontweight="bold", color=INK)
style_axes(ax)
ax.legend(fontsize=9.5, loc="upper right", frameon=True, facecolor="white", edgecolor=MUTED)
ax.set_title("Lift by predicted-risk rank, 2020 hold-out year", fontsize=13.5,
             fontweight="bold", color=INK, pad=12)
fig.text(0.5, -0.04,
         f"Mean over {R['n_patches']} fire-active 384 px patches (≥ {R['min_fire_px_per_patch']} fire pixels each). "
         f"Flagging the top 0.5% of a patch captures on average {100*TPR_ALL[i5]:.0f}% of its fire. "
         f"New fire is {100*R['new_fire_share']:.0f}% of all fire pixels.",
         ha="center", fontsize=8.8, color=MUTED, style="italic", wrap=True)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_lift_curve.png"), dpi=300, bbox_inches="tight", facecolor="white")
print("wrote fig_lift_curve.png")
