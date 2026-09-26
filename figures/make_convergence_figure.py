"""SMOLDER training convergence: 2019 validation AP across the full lineage.

Single model only (the incumbent): job 1764635 (epochs 0-19, ended at a 6h
wall-clock limit, not convergence) continued as job 1764897 (epochs 20-36,
resumed from last.ckpt, early-stopped). Numbers read directly from both runs'
metrics.csv.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from style_smolder import ACCENT, ACCENT2, INK, MUTED, new_figure, style_axes

VAL_AP = {
    0: 0.1316, 1: 0.2169, 2: 0.2098, 3: 0.2758, 4: 0.2868, 5: 0.3510,
    6: 0.3764, 7: 0.3943, 8: 0.4406, 9: 0.4604, 10: 0.4763, 11: 0.4739,
    12: 0.4747, 13: 0.4259, 14: 0.4692, 15: 0.5046, 16: 0.4971, 17: 0.5035,
    18: 0.5042, 19: 0.4981, 20: 0.5091, 21: 0.5064, 22: 0.5092, 23: 0.5111,
    24: 0.5117, 25: 0.5117, 26: 0.5111, 27: 0.5111, 28: 0.5125, 29: 0.5128,
    30: 0.5144, 31: 0.4975, 32: 0.4764, 33: 0.4801, 34: 0.4796, 35: 0.5036,
    36: 0.4426,
}
RESUME_AT = 20
SWA_EPOCHS = (28, 29, 30)

ep = np.array(sorted(VAL_AP))
ap = np.array([VAL_AP[e] for e in ep])

fig = new_figure((10.5, 5.6))
ax = fig.add_subplot(111)

ax.plot(ep, ap, "-", color=ACCENT, lw=2.0, zorder=3)
ax.plot(ep, ap, "o", color=ACCENT, ms=4.2, zorder=4,
        markerfacecolor="white", markeredgewidth=1.4)

ax.axvline(RESUME_AT - 0.5, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2)
ax.text(RESUME_AT - 0.5, 0.135, "  6h wall-clock limit\n  training resumed",
        fontsize=8.7, color=MUTED, va="bottom", ha="left", style="italic")

swa_ap = [VAL_AP[e] for e in SWA_EPOCHS]
ax.scatter(SWA_EPOCHS, swa_ap, s=95, facecolor=ACCENT2, edgecolor="white",
           linewidth=1.3, zorder=5)
ax.annotate("epochs 28-30\nweight-averaged\n(final model)",
            xy=(29, max(swa_ap)), xytext=(24.5, 0.30),
            fontsize=9, fontweight="bold", color=ACCENT2, ha="center",
            arrowprops=dict(arrowstyle="-|>", color=ACCENT2, lw=1.4))

ax.set_xlim(-0.5, 36.5)
ax.set_ylim(0.10, 0.56)
ax.set_xticks(np.arange(0, 37, 4))
ax.set_yticks(np.arange(0.10, 0.56, 0.05))
ax.set_xlabel("Training epoch", fontsize=11, fontweight="bold", color=INK)
ax.set_ylabel("Average precision (2019 validation)", fontsize=11,
              fontweight="bold", color=INK)
style_axes(ax)

ax.set_title("SMOLDER training convergence", fontsize=13.5, fontweight="bold",
             color=INK, pad=12)
fig.text(0.5, -0.02,
         "One continuous training lineage, resumed once after a scheduler wall-clock limit. "
         "The released weights are a simple average (SWA) of the three checkpoints shown in orange.",
         ha="center", fontsize=9, color=MUTED, style="italic")

fig.tight_layout()
fig.savefig("fig_convergence.png", dpi=300, bbox_inches="tight", facecolor="white")
print("wrote fig_convergence.png")
