"""SMOLDER lift by predicted-risk percentile, 2020 hold-out test.

Single model: SWA(resume ep28,29,30), the released weights. Numbers are the
pooled real-test topk table (n=1500 patches, 188.6M land px, job 1765197
member2 -- verified as a reproducibility anchor to 4dp against an independent
earlier run of the identical checkpoint).

Two lines because they answer different questions: ALL-fire lift is "how much
better than random is flagging the top-k% of the map", the metric SMOLDER is
selected on. NEW-fire lift restricts the positive set to fire >=3px from any
recent fire (near-field spread, NOT far-field new ignition -- see the
methodological note in the README) and is reported because it is the
harder, more operationally interesting number.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from style_smolder import ACCENT, ACCENT2, INK, MUTED, new_figure, style_axes

# pooled 2020 hold-out test, SWA(resume ep28,29,30), n=1500 patches
KS        = np.array([0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10])
LIFT_ALL  = np.array([1006.0, 841.3, 541.7, 352.7, 214.7, 103.0, 56.7, 30.5, 13.2, 7.0])
LIFT_NEW  = np.array([0.7, 1.3, 2.9, 7.9, 12.5, 14.5, 12.7, 9.5, 5.8, 3.9])
TPR_ALL   = np.array([0.101, 0.167, 0.271, 0.352, 0.430, 0.515, 0.567, 0.610, 0.660, 0.702])

fig = new_figure((10.5, 6.2))
ax = fig.add_subplot(111)

ax.plot(100 * KS, LIFT_ALL, "o-", color=ACCENT, lw=2.2, ms=6,
        label="All fire", zorder=4)
ax.plot(100 * KS, LIFT_NEW, "o-", color=ACCENT2, lw=2.2, ms=6,
        label="Near-field new fire (≥ 3 km from recent fire)", zorder=4)
ax.axhline(1.0, color=MUTED, lw=1.3, ls=(0, (4, 3)), zorder=2)
ax.text(0.011, 1.35, "random", fontsize=8.6, color=MUTED, style="italic")

peak_i = int(np.argmax(LIFT_NEW))
ax.annotate(f"peak {LIFT_NEW[peak_i]:.1f}× @ top {100*KS[peak_i]:g}%",
            xy=(100 * KS[peak_i], LIFT_NEW[peak_i]),
            xytext=(100 * KS[peak_i] * 2.3, LIFT_NEW[peak_i] * 1.55),
            fontsize=9.5, fontweight="bold", color=ACCENT2,
            arrowprops=dict(arrowstyle="-|>", color=ACCENT2, lw=1.3))
ax.annotate(f"{LIFT_ALL[3]:.0f}× @ top {100*KS[3]:g}%",
            xy=(100 * KS[3], LIFT_ALL[3]),
            xytext=(100 * KS[3] * 0.42, LIFT_ALL[3] * 2.6),
            fontsize=9.5, fontweight="bold", color=ACCENT,
            arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.3))

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.008, 13)
ax.set_ylim(0.5, 1600)
ax.set_xticks([0.01, 0.1, 1, 10])
ax.set_xticklabels(["0.01", "0.1", "1", "10"])
ax.set_yticks([1, 3, 10, 30, 100, 300, 1000])
ax.set_yticklabels(["1", "3", "10", "30", "100", "300", "1000"])
ax.set_xlabel("Share of the 1 km grid flagged as highest risk (%)",
              fontsize=11, fontweight="bold", color=INK)
ax.set_ylabel("Enrichment over random (lift, log scale)", fontsize=11,
              fontweight="bold", color=INK)
style_axes(ax)
ax.legend(fontsize=9.5, loc="lower left", frameon=True, facecolor="white",
          edgecolor=MUTED)

ax.set_title("Predicted-risk enrichment, 2020 hold-out test year",
             fontsize=13.5, fontweight="bold", color=INK, pad=12)
fig.text(0.5, -0.03,
         "Flagging the top 0.5% of the map captures 51.5% of all fire (103× enrichment) and, within that "
         "same budget, 14.5× enrichment for near-field new fire. Pooled over 1500 patches, 188.6M land pixels, "
         "base fire rate 0.175%. Never used for training or model selection.",
         ha="center", fontsize=8.8, color=MUTED, style="italic", wrap=True)

fig.tight_layout()
fig.savefig("fig_lift_curve.png", dpi=300, bbox_inches="tight", facecolor="white")
print("wrote fig_lift_curve.png")
