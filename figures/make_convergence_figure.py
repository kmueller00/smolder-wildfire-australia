"""SMOLDER training convergence (results/training_curve.csv):
(a) training and validation loss, (b) validation average precision (2019).
in_swa marks the checkpoints averaged into the released model."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from style_smolder import ACCENT, ACCENT2, INK, MUTED, new_figure, style_axes

HERE = os.path.dirname(os.path.abspath(__file__))
C = pd.read_csv(os.environ.get("CURVE", os.path.join(HERE, "..", "results", "training_curve.csv")))

fig = new_figure((13.0, 5.2))
axA = fig.add_subplot(121)
axB = fig.add_subplot(122)
xmax = C.epoch.max()

axA.plot(C.epoch, C.train_loss, "-", color=ACCENT, lw=2.0, label="training", zorder=4)
axA.plot(C.epoch, C.val_loss, "--", color=ACCENT2, lw=1.8, label="validation (2019)", zorder=4)
lo, hi = min(C.train_loss.min(), C.val_loss.min()), max(C.train_loss.max(), C.val_loss.max())
axA.set_ylim(np.floor(lo * 5) / 5 - 0.2, np.ceil(hi * 5) / 5 + 0.1)
axA.set_yticks(np.arange(axA.get_ylim()[0], axA.get_ylim()[1] + 1e-9, 0.2))
axA.set_ylabel("Loss (weighted BCE)", fontsize=10.5, fontweight="bold", color=INK)
axA.set_title("(a) Loss", fontsize=12, fontweight="bold", color=INK, loc="left")
axA.legend(fontsize=9, loc="upper right", frameon=True, facecolor="white", edgecolor=MUTED)

axB.plot(C.epoch, C.val_ap, "-", color=ACCENT, lw=2.0, zorder=4, label="validation AP (2019)")
swa = C[C.in_swa == 1]
axB.scatter(swa.epoch, swa.val_ap, s=90, facecolor=ACCENT2, edgecolor="white", linewidth=1.3,
            zorder=5, label="averaged into the released model")
axB.set_ylim(0, np.ceil(C.val_ap.max() * 20) / 20 + 0.05)
axB.set_yticks(np.arange(0, axB.get_ylim()[1] + 1e-9, 0.05))
axB.set_ylabel("Average precision", fontsize=10.5, fontweight="bold", color=INK)
axB.set_title("(b) Validation average precision", fontsize=12, fontweight="bold", color=INK, loc="left")
axB.legend(fontsize=9, loc="lower right", frameon=True, facecolor="white", edgecolor=MUTED)

for ax in (axA, axB):
    ax.set_xlim(-0.5, xmax + 0.5)
    ax.set_xticks(np.arange(0, xmax + 1, 2))
    ax.set_xlabel("Training epoch", fontsize=10.5, fontweight="bold", color=INK)
    style_axes(ax)

fig.suptitle("SMOLDER training convergence", fontsize=13.5, fontweight="bold", color=INK, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(os.environ.get("FIG_OUT", HERE), "fig_convergence.png"), dpi=300, bbox_inches="tight", facecolor="white")
print("wrote fig_convergence.png")
