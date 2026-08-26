"""New fire distance decay figure.

Shows new fire lift as a function of how far a fire pixel must be from any
recent fire to count as "new". This is the headline diagnostic: skill collapses
with distance and falls below random beyond about 10 km, showing the model
predicts near field spread rather than genuinely new ignition.

Numbers are hardcoded from the evaluation logs (jobs 1765273 / 1766529) so the
figure regenerates without a GPU or the data cubes.

The model vs XGBoost comparison lives in make_comparison_figure.py.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "figures"

# ---------------------------------------------------------------- (b)
radii = np.array([0, 1, 3, 5, 10, 20, 40])
lift_3d = np.array([53.0, 33.0, 11.1, 3.4, 0.1, 0.1, 0.2])
lift_30d = np.array([53.3, 34.1, 12.1, 3.7, 0.1, 0.2, 0.8])
share_3d = np.array([59.7, 48.2, 35.7, 29.6, 20.9, 11.5, 4.8])
share_30d = np.array([57.2, 42.6, 26.0, 18.3, 9.8, 3.9, 1.5])

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 4.9))

spec_3d = np.array([14.63, 8.48, 2.63, 1.26, 0.60, 0.43, 1.55])
axA.plot(radii, lift_3d, "o-", lw=2, ms=7, color="#1a6faa",
         label="main model (with fire history)")
axA.plot(radii, lift_30d, "s--", lw=1.4, ms=5, color="#4d9de0", alpha=0.8,
         label="main model, 30 day window")
axA.plot(radii, spec_3d, "^-", lw=2, ms=7, color="#e07a1a",
         label="specialist (no fire history)")
axA.axhline(1.0, color="#c94f4f", lw=1.5, ls=":")
# Sits in the empty band under the left end of both curves, well clear of the
# dotted line it labels (both curves are above 8x at this x range).
axA.text(0.6, 1.55, "random (lift = 1)", color="#c94f4f", fontsize=8.8,
         ha="left", va="bottom", fontweight="bold")
axA.axvspan(10, 41, color="#c94f4f", alpha=0.07)
# The shaded band plus the "random" line already carry this; an extra label
# here only collided with the axis title.
axA.annotate("specialist is 5x better\nin the far field", xy=(10, 0.60),
             xytext=(11.5, 4.2), fontsize=8.5, fontweight="bold", color="#e07a1a",
             arrowprops=dict(arrowstyle="->", lw=1.3, color="#e07a1a"))
axA.annotate("current metric\n(r = 3 px)", xy=(3, 11.1), xytext=(5.4, 26),
             fontsize=9, fontweight="bold",
             arrowprops=dict(arrowstyle="->", lw=1.3, color="black"))
axA.set_yscale("log")
axA.set_xlabel("Minimum distance from any recent fire (px = km)", fontsize=10)
axA.set_ylabel("New fire lift at top 0.5%", fontsize=10, fontweight="bold")
axA.set_title("Skill collapses with distance, except for the specialist",
              fontsize=11, fontweight="bold")
axA.grid(alpha=0.25, which="both")
axA.legend(fontsize=8, loc="lower left")

axB.plot(radii, share_3d, "o-", lw=2, ms=7, color="#2a9d8f", label="3 day history window")
axB.plot(radii, share_30d, "s--", lw=1.6, ms=6, color="#83c5be", label="30 day history window")
axB.annotate("35.7%", xy=(3, 35.7), xytext=(6.2, 46), fontsize=9, fontweight="bold",
             color="#2a9d8f", arrowprops=dict(arrowstyle="->", lw=1.2, color="#2a9d8f"))
axB.annotate("3.9%", xy=(20, 3.9), xytext=(24, 15), fontsize=9, fontweight="bold",
             color="#2a9d8f", arrowprops=dict(arrowstyle="->", lw=1.2, color="#2a9d8f"))
axB.set_xlabel("Minimum distance from any recent fire (px = km)", fontsize=10)
axB.set_ylabel("Fire pixels still counted as new (%)", fontsize=10, fontweight="bold")
axB.set_title("89% of new fire is within 20 km of recent fire",
              fontsize=11, fontweight="bold")
axB.grid(alpha=0.25)
axB.legend(fontsize=8.5)

fig.suptitle("What the new fire metric actually measures  |  2020 holdout year, "
             "600 patches, final model",
             fontsize=12.5, fontweight="bold", y=1.015)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_newfire_distance_decay.png", dpi=300,
            bbox_inches="tight", facecolor="white")
print(f"wrote {OUT_DIR}/fig_newfire_distance_decay.png")
