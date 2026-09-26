"""Validation AP (2019) per epoch of the SMOLDER training run
(results/training_curve.csv; in_swa marks the averaged checkpoints)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from style_smolder import ACCENT, ACCENT2, INK, MUTED, new_figure, style_axes

HERE = os.path.dirname(os.path.abspath(__file__))
C = pd.read_csv(os.path.join(HERE, "..", "results", "training_curve.csv"))
T_MAX = 25

fig = new_figure((10.5, 5.6))
ax = fig.add_subplot(111)
ax.plot(C.epoch, C.val_ap, "-", color=ACCENT, lw=2.0, zorder=3)
ax.plot(C.epoch, C.val_ap, "o", color=ACCENT, ms=4.2, zorder=4, markerfacecolor="white",
        markeredgewidth=1.4)
swa = C[C.in_swa == 1]
ax.scatter(swa.epoch, swa.val_ap, s=95, facecolor=ACCENT2, edgecolor="white", linewidth=1.3,
           zorder=5, label="averaged into the released model")
top = float(C.val_ap.max())
ax.set_xlim(-0.5, C.epoch.max() + 0.5)
ax.set_ylim(0, np.ceil(top * 20) / 20 + 0.05)
ax.set_xticks(np.arange(0, C.epoch.max() + 1, 4))
ax.set_yticks(np.arange(0, ax.get_ylim()[1] + 1e-9, 0.05))
if C.epoch.max() > T_MAX:
    ax.axvline(T_MAX, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.text(T_MAX + 0.3, 0.01, " learning rate reaches 0\n (cosine period 25 epochs)",
            fontsize=8.5, color=MUTED, va="bottom", style="italic")
ax.set_xlabel("Training epoch", fontsize=11, fontweight="bold", color=INK)
ax.set_ylabel("Average precision, 2019 validation", fontsize=11, fontweight="bold", color=INK)
style_axes(ax)
ax.legend(fontsize=9.5, loc="upper left", frameon=True, facecolor="white", edgecolor=MUTED)
ax.set_title("SMOLDER training convergence", fontsize=13.5, fontweight="bold", color=INK, pad=12)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_convergence.png"), dpi=300, bbox_inches="tight", facecolor="white")
print("wrote fig_convergence.png")
