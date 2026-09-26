"""How the new-fire lift depends on the distance threshold that defines "new"
(results/newfire_sweep_2020.json, written by
smolder.evaluation.newfire_definition_sweep)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from style_smolder import ACCENT, ACCENT2, INK, MUTED, new_figure, style_axes

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "..", "results", "newfire_sweep_2020.json")))
RADII = np.array(S["radii_px"])
lift = next(iter(S["lift"].values()))
SHOW = ((1, "No fire nearby in the last 3 days", ACCENT, "-"), (30, "No fire nearby in the last 32 days", ACCENT2, "--"))

fig = new_figure((12.6, 5.1))
axA = fig.add_subplot(121)
axB = fig.add_subplot(122)
for w, lab, col, sty in SHOW:
    y = np.array([np.nan if v is None else v for v in lift[str(w)]], float)
    axA.plot(RADII, y, sty, color=col, lw=2.0, label=lab)
    axB.plot(RADII, S["share_new_pct"][str(w)], sty, color=col, lw=2.0, label=lab)
axA.axhline(1.0, color=MUTED, lw=1.2, ls=(0, (4, 3)))
axA.text(30.5, 1.12, "random", fontsize=8.4, color=MUTED, style="italic")
axA.set_yscale("log")
vals = [v for w, *_ in SHOW for v in lift[str(w)] if v]
lo, hi = min(min(vals), 1.0) / 2, max(vals) * 2
axA.set_ylim(lo, hi)
yt = [v for v in (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100) if lo <= v <= hi]
axA.set_yticks(yt); axA.set_yticklabels([f"{v:g}" for v in yt])
axA.set_ylabel(f"New-fire lift at top {100*S['report_k']:g}% (log scale)", fontsize=10.3,
               fontweight="bold", color=INK)
axA.set_title("Lift falls with distance from recent fire", fontsize=12, fontweight="bold", color=INK)
axB.set_ylim(0, 100)
axB.set_ylabel("Fire pixels counted as new (%)", fontsize=10.3, fontweight="bold", color=INK)
axB.set_title("How much fire counts as new", fontsize=12, fontweight="bold", color=INK)
for ax in (axA, axB):
    ax.set_xlim(-1.5, 41.5)
    ax.set_xticks([0, 5, 10, 20, 30, 40])
    ax.set_xlabel("Minimum distance from fire in the history (px ≈ km)",
                  fontsize=10.3, fontweight="bold", color=INK)
    style_axes(ax)
    ax.legend(fontsize=8.6, loc="upper right", frameon=True, facecolor="white", edgecolor=MUTED)
fig.suptitle("New-fire lift depends on how “new” is defined, 2020 hold-out year",
             fontsize=14, fontweight="bold", color=INK, y=1.03)
fig.text(0.5, -0.05,
         f"{S['n_patches']} fire-active patches. A fire pixel counts as new if no VIIRS fire was detected within the "
         "given distance during the stated period up to the issue day.",
         ha="center", fontsize=9, color=MUTED, style="italic", wrap=True)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_newfire_distance_decay.png"), dpi=300, bbox_inches="tight",
            facecolor="white")
print("wrote fig_newfire_distance_decay.png")
