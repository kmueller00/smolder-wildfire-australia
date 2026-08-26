"""SMOLDER vs XGBoost on next 3 day wildfire risk, 2020 hold out year.

Both numbers are pooled over the same 1500 patches / 188.6M land pixels, so the
comparison is like for like. The XGBoost baseline is not a strawman: it carries
21 engineered features including every one that passed an independent feature
gate (lightning climatology, terrain, fuel age, downwind alignment).

No dash characters anywhere in the rendered text, and every label is placed
clear of the plotted lines.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SM = "#1a6faa"; XG = "#c94f4f"; INK = "#1a202c"

# pooled 2020 hold out test, n=1500 patches, 188.6M land px, base rate 0.175%
KS      = np.array([0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
TPR_ALL = np.array([0.101, 0.352, 0.430, 0.515, 0.567, 0.610, 0.660, 0.702])
LIFT    = np.array([1006.0, 352.7, 214.7, 103.0, 56.7, 30.5, 13.2, 7.0])

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.2, 5.3),
                               gridspec_kw={"width_ratios": [0.72, 1]})

# ---------------------------------------------------------------- panel A
names = ["SMOLDER", "XGBoost\n(21 features)"]
vals  = [0.4485, 0.0611]
bars = axA.bar([0, 1], vals, color=[SM, XG], edgecolor="black", lw=0.8, width=0.55)
for x, v in zip([0, 1], vals):
    axA.text(x, v + 0.013, f"{v:.4f}", ha="center", fontsize=13, fontweight="bold")
axA.set_xticks([0, 1]); axA.set_xticklabels(names, fontsize=11)
axA.set_ylabel("Average Precision (AUC PR)", fontsize=11.5, fontweight="bold")
axA.set_ylim(0, 0.62); axA.set_xlim(-0.6, 1.6)
axA.grid(axis="y", alpha=0.25); axA.set_axisbelow(True)
# Comparison bracket drawn ABOVE both bars so it crosses nothing. The base
# rate line is omitted here: at 0.00175 on a 0 to 0.62 axis it is
# indistinguishable from the axis and only invites overlap. It is stated in
# the caption and drawn properly in panel B instead.
axA.annotate("", xy=(0, 0.545), xytext=(1, 0.545),
             arrowprops=dict(arrowstyle="<->", lw=1.7, color="#333"))
axA.text(0.5, 0.562, "7.3x better", ha="center", va="bottom", fontsize=12.5,
         fontweight="bold", color="#333")
axA.set_title("Overall skill", fontsize=12, fontweight="bold", pad=9)

# ---------------------------------------------------------------- panel B
axB.plot(KS, 100 * TPR_ALL, "o-", color=SM, lw=2.4, ms=7, label="SMOLDER", zorder=5)
axB.set_xscale("log")
axB.set_xlabel("Share of the map flagged as high risk (%)", fontsize=11)
axB.set_ylabel("Fire pixels captured (%)", fontsize=11.5, fontweight="bold")
axB.set_ylim(0, 82); axB.set_xlim(0.008, 13)
axB.grid(alpha=0.25, which="both"); axB.set_axisbelow(True)
axB.set_xticks([0.01, 0.1, 1, 10]); axB.set_xticklabels(["0.01", "0.1", "1", "10"])

# lift callouts placed BELOW the curve so they never sit on the line
for k, t, l in ((0.5, 0.515, 103.0), (2.0, 0.610, 30.5), (10.0, 0.702, 7.0)):
    axB.annotate(f"{l:.0f}x", xy=(k, 100 * t), xytext=(k, 100 * t - 13),
                 fontsize=9.5, fontweight="bold", color=SM, ha="center",
                 arrowprops=dict(arrowstyle="-", lw=0.9, color=SM, alpha=0.6))
axB.text(0.055, 68, "labels show enrichment\nover the base rate",
         fontsize=8.6, color=SM, style="italic", linespacing=1.4)

axB.axhline(0.175, color="#555", ls=":", lw=1.4)
axB.text(11.5, 2.6, "random", fontsize=8.6, color="#555", ha="right", style="italic")
axB.set_title("How much fire is caught for a given alert budget",
              fontsize=12, fontweight="bold", pad=9)
axB.legend(fontsize=10, loc="upper left", frameon=True)

fig.suptitle("Next 3 day wildfire risk over Australia   |   2020 hold out year, "
             "never used for training or model selection",
             fontsize=12.8, fontweight="bold", y=1.005)
fig.text(0.5, -0.045,
         "Pooled over 1500 patches and 188.6 million land pixels. Flagging just 0.5% of the map "
         "captures 52% of all fire, a 103 fold enrichment over chance.",
         ha="center", fontsize=9.3, color="#444", style="italic")
fig.tight_layout()
fig.savefig("fig_model_vs_xgboost.png", dpi=300, bbox_inches="tight", facecolor="white")
print("wrote figures/fig_model_vs_xgboost.png")
