"""How SMOLDER's lift depends on distance from the nearest recent fire.

Single model (the incumbent, SWA(resume ep28,29,30)). Two things this answers
about the model's own headline number, not a second model:
  (a) how much of what a fixed radius calls "new fire" is actually within
      that radius of a fire that was already burning, and
  (b) how the model's own enrichment falls off as that radius widens.

Numbers are hardcoded from newfire_definition_sweep.py's evaluation logs
(job 1765273, 600 patches, 2020 hold-out) so the figure regenerates without a
GPU or the data cubes.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from style_smolder import ACCENT, ACCENT2, INK, MUTED, new_figure, style_axes

RADII      = np.array([0, 1, 3, 5, 10, 20, 40])
LIFT_3D    = np.array([53.0, 33.0, 11.1, 3.4, 0.1, 0.1, 0.2])
LIFT_30D   = np.array([53.3, 34.1, 12.1, 3.7, 0.1, 0.2, 0.8])
SHARE_3D   = np.array([59.7, 48.2, 35.7, 29.6, 20.9, 11.5, 4.8])
SHARE_30D  = np.array([57.2, 42.6, 26.0, 18.3, 9.8, 3.9, 1.5])

fig = new_figure((12.6, 5.1))
axA = fig.add_subplot(121)
axB = fig.add_subplot(122)

axA.plot(RADII, LIFT_3D, "o-", color=ACCENT, lw=2.1, ms=6.5, label="3-day history window")
axA.plot(RADII, LIFT_30D, "o--", color=ACCENT2, lw=1.8, ms=5.5, label="30-day history window")
axA.axhline(1.0, color=MUTED, lw=1.2, ls=(0, (4, 3)))
axA.text(0.6, 1.5, "random", fontsize=8.4, color=MUTED, style="italic")
axA.annotate("the r ≥ 3 px point behind\nthe 14.5× headline figure",
             xy=(3, 11.1), xytext=(9, 24), fontsize=8.7, fontweight="bold",
             color=ACCENT, arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.2))
axA.set_yscale("log")
axA.set_xlim(-1.5, 41.5)
axA.set_ylim(0.06, 90)
axA.set_xticks([0, 5, 10, 20, 30, 40])
axA.set_xlabel("Minimum distance from any recent fire (px ≈ km)",
               fontsize=10.3, fontweight="bold", color=INK)
axA.set_ylabel("Lift at top 0.5% (log scale)", fontsize=10.3, fontweight="bold", color=INK)
style_axes(axA)
axA.legend(fontsize=8.6, loc="upper right", frameon=True, facecolor="white", edgecolor=MUTED)
axA.set_title("Enrichment falls off with distance", fontsize=12, fontweight="bold", color=INK)

axB.plot(RADII, SHARE_3D, "o-", color=ACCENT, lw=2.1, ms=6.5, label="3-day history window")
axB.plot(RADII, SHARE_30D, "o--", color=ACCENT2, lw=1.8, ms=5.5, label="30-day history window")
axB.annotate(f"{SHARE_3D[2]:.0f}%", xy=(3, SHARE_3D[2]), xytext=(10, 46),
             fontsize=9.5, fontweight="bold", color=ACCENT,
             arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.2))
axB.annotate(f"{SHARE_30D[5]:.0f}%", xy=(20, SHARE_30D[5]), xytext=(26, 16),
             fontsize=9.5, fontweight="bold", color=ACCENT2,
             arrowprops=dict(arrowstyle="-|>", color=ACCENT2, lw=1.2))
axB.set_xlim(-1.5, 41.5)
axB.set_ylim(0, 65)
axB.set_xticks([0, 5, 10, 20, 30, 40])
axB.set_xlabel("Minimum distance from any recent fire (px ≈ km)",
               fontsize=10.3, fontweight="bold", color=INK)
axB.set_ylabel("Fire pixels meeting the threshold (%)", fontsize=10.3,
               fontweight="bold", color=INK)
style_axes(axB)
axB.legend(fontsize=8.6, loc="upper right", frameon=True, facecolor="white", edgecolor=MUTED)
axB.set_title("How much fire counts as “new”, by radius", fontsize=12,
              fontweight="bold", color=INK)

fig.suptitle("What a distance threshold changes, 2020 hold-out test year",
             fontsize=14, fontweight="bold", color=INK, y=1.03)
fig.text(0.5, -0.05,
         "At the conventional 3 px / 3-day threshold, 35.7% of fire pixels qualify as “new” and the model "
         "still shows real skill (14.5×). Widening the radius to 20 px drops that to 3.9% of pixels and the "
         "model's own enrichment there falls to ≈1× (random) — near-field spread is where this model has "
         "demonstrated skill; genuinely isolated new ignition is not.",
         ha="center", fontsize=9, color=MUTED, style="italic", wrap=True)

fig.tight_layout()
fig.savefig("fig_newfire_distance_decay.png", dpi=300, bbox_inches="tight", facecolor="white")
print("wrote fig_newfire_distance_decay.png")
