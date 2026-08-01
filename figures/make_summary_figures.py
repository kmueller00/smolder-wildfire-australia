"""Two publication figures summarising the final evaluation.

(a) fig_model_progression.png -- 2020 test AUC-PR and near-field new-fire lift
    for every validated checkpoint, in the order they were developed. Only
    numbers confirmed on the 2020 HOLD-OUT year are plotted; validation-only
    (val_ap) numbers are deliberately excluded, because this project repeatedly
    found them non-predictive of test performance.

(b) fig_newfire_distance_decay.png -- new-fire lift as a function of how far a
    fire pixel must be from any recent fire to count as "new". This is the
    headline diagnostic: skill collapses with distance and falls BELOW random
    (lift < 1) beyond ~10 km, showing the model predicts near-field spread
    rather than genuinely new ignition.

Numbers are hardcoded from the evaluation logs (job 1764905 / 1765197 / 1765273)
so the figures can be regenerated without a GPU or the data cubes.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "figures"

# ---------------------------------------------------------------- (a)
# label, 2020 test AUC-PR, near-field new-fire lift @top-0.5%, is_final
MODELS = [
    ("ConvLSTM\n256px\n(early)",              0.3664,  4.4, False),
    ("XGBoost\n(21 feat.)",                   0.0611,  np.nan, False),
    ("ps128_final",                           0.4020,  5.6, False),
    ("ps384_final\n(baseline)",               0.4230, 11.9, False),
    ("+ new-fire\nsampler +\nfh-dropout",     0.4368, 12.6, False),
    ("+ SWA\n(combo)",                        0.4429, 12.6, False),
    ("+ longer\ntraining",                    0.4459, 14.1, False),
    ("+ SWA\n(final)",                        0.4485, 14.5, True),
]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True,
                               gridspec_kw={"height_ratios": [1, 0.85], "hspace": 0.12})
x = np.arange(len(MODELS))
ap = [m[1] for m in MODELS]
lift = [m[2] for m in MODELS]
final = [m[3] for m in MODELS]
cols = ["#c94f4f" if m[0].startswith("XGBoost") else ("#1a6faa" if f else "#7fb3d5")
        for m, f in zip(MODELS, final)]

b1 = ax1.bar(x, ap, color=cols, edgecolor="black", linewidth=0.6, width=0.68)
for xi, v in zip(x, ap):
    ax1.text(xi, v + 0.008, f"{v:.4f}", ha="center", fontsize=8.5, fontweight="bold")
ax1.set_ylabel("AUC-PR  (2020 hold-out test)", fontsize=10.5, fontweight="bold")
ax1.set_ylim(0, 0.52)
ax1.grid(axis="y", alpha=0.25, zorder=0)
ax1.set_axisbelow(True)
ax1.axhline(0.4230, ls="--", lw=1.0, color="#444",
            label="ps384_final baseline (0.4230)")
ax1.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
ax1.set_title("FireCastNet — next-3-day fire risk over Australia\n"
              "Validated on the 2020 hold-out year (n=1500 patches, 188.6M land pixels, "
              "base rate 0.175%)",
              fontsize=12, fontweight="bold", pad=12)

b2 = ax2.bar(x, lift, color=cols, edgecolor="black", linewidth=0.6, width=0.68)
for xi, v in zip(x, lift):
    if np.isfinite(v):
        ax2.text(xi, v + 0.25, f"{v:.1f}×", ha="center", fontsize=8.5, fontweight="bold")
    else:
        ax2.text(xi, 0.4, "n/a", ha="center", fontsize=8, color="#777")
ax2.set_ylabel("Near-field new-fire lift\n@ top-0.5%", fontsize=10.5, fontweight="bold")
ax2.set_ylim(0, 17)
ax2.grid(axis="y", alpha=0.25)
ax2.set_axisbelow(True)
ax2.set_xticks(x)
ax2.set_xticklabels([m[0] for m in MODELS], fontsize=8.5)
ax2.annotate("+6.0% AUC-PR\n+21.8% lift",
             xy=(7, 14.5), xytext=(5.55, 8.4), fontsize=9, fontweight="bold",
             color="#1a6faa",
             arrowprops=dict(arrowstyle="->", color="#1a6faa", lw=1.4))
fig.text(0.5, 0.012,
         '"Near-field" lift uses the r=3 px / t−3 definition; see the companion figure — '
         "at this radius the metric largely measures fire-front spread, not new ignition.",
         ha="center", fontsize=8.2, style="italic", color="#444")
fig.savefig(f"{OUT_DIR}/fig_model_progression.png", dpi=300,
            bbox_inches="tight", facecolor="white")
print(f"wrote {OUT_DIR}/fig_model_progression.png")
plt.close(fig)

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
         label="main model, 30-day window")
axA.plot(radii, spec_3d, "^-", lw=2, ms=7, color="#e07a1a",
         label="specialist (NO fire history)")
axA.axhline(1.0, color="#c94f4f", lw=1.5, ls=":")
axA.text(41, 1.25, "random (lift = 1)", color="#c94f4f", fontsize=8.5,
         ha="right", fontweight="bold")
axA.axvspan(10, 41, color="#c94f4f", alpha=0.07)
axA.text(24, 0.030, "BELOW RANDOM", ha="center",
         fontsize=9, color="#c94f4f", fontweight="bold")
axA.annotate("specialist is ~5x better\nin the far field", xy=(10, 0.60),
             xytext=(11.5, 4.2), fontsize=8.5, fontweight="bold", color="#e07a1a",
             arrowprops=dict(arrowstyle="->", lw=1.3, color="#e07a1a"))
axA.annotate("current metric\n(r = 3 px)", xy=(3, 11.1), xytext=(5.4, 26),
             fontsize=9, fontweight="bold",
             arrowprops=dict(arrowstyle="->", lw=1.3, color="black"))
axA.set_yscale("log")
axA.set_xlabel("Minimum distance from any recent fire  (px = km)", fontsize=10)
axA.set_ylabel("New-fire lift @ top-0.5%", fontsize=10, fontweight="bold")
axA.set_title("Skill collapses with distance -- except for the specialist",
              fontsize=11, fontweight="bold")
axA.grid(alpha=0.25, which="both")
axA.legend(fontsize=8, loc="lower left")

axB.plot(radii, share_3d, "o-", lw=2, ms=7, color="#2a9d8f", label="3-day history window")
axB.plot(radii, share_30d, "s--", lw=1.6, ms=6, color="#83c5be", label="30-day history window")
axB.annotate("35.7%", xy=(3, 35.7), xytext=(6.2, 46), fontsize=9, fontweight="bold",
             color="#2a9d8f", arrowprops=dict(arrowstyle="->", lw=1.2, color="#2a9d8f"))
axB.annotate("3.9%", xy=(20, 3.9), xytext=(24, 15), fontsize=9, fontweight="bold",
             color="#2a9d8f", arrowprops=dict(arrowstyle="->", lw=1.2, color="#2a9d8f"))
axB.set_xlabel("Minimum distance from any recent fire  (px = km)", fontsize=10)
axB.set_ylabel("Fire pixels still counted as “new” (%)", fontsize=10, fontweight="bold")
axB.set_title("89% of “new fire” is within 20 km of recent fire",
              fontsize=11, fontweight="bold")
axB.grid(alpha=0.25)
axB.legend(fontsize=8.5)

fig.suptitle("What the “new-fire” metric actually measures — 2020 hold-out year, "
             "600 patches, SWA(resume) model",
             fontsize=12.5, fontweight="bold", y=1.015)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_newfire_distance_decay.png", dpi=300,
            bbox_inches="tight", facecolor="white")
print(f"wrote {OUT_DIR}/fig_newfire_distance_decay.png")
