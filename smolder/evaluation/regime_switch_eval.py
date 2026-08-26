"""Regime-switched combination of the main model and the no-fire-history specialist.

MOTIVATION (job 1766529): the two models are strong in DISJOINT regimes.
  near field (<10 km from recent fire): main 11.1x lift, specialist 2.6x
  far  field (>=10 km):                 main  0.11x,     specialist 0.60x
Naive prob-mean scored 0.07x -- WORSE than either -- because the main model's
confident near-field predictions swamp the specialist's weak far-field signal.
Averaging is simply the wrong operator for disjoint competence.

THE FIX: switch, don't blend. Each pixel is assigned to exactly one regime by
its distance to recent fire, and scored by the model that owns that regime.

WHY PERCENTILE-RANK AND NOT RAW PROBABILITY: the two models have completely
different output distributions (the specialist never saw fire-history inputs
and is far less confident). Splicing raw sigmoids would create a discontinuity
at the regime boundary and wreck the global ranking that AUC-PR measures.
Converting each regime's scores to WITHIN-REGIME percentiles makes them
commensurable, and then `beta` controls how much of the top-k budget the far
field is allowed to claim:
    beta = 0  -> pure main model (baseline, far field never selected)
    beta = 1  -> regimes compete purely on percentile, i.e. each gets a share
                 of the budget proportional to its pixel count
    beta > 1  -> deliberately over-weight the far field
This is explicitly a BUDGET-ALLOCATION trade: top-k is a fixed budget, so any
gain in far-field recall must come out of near-field recall. The point of the
sweep is to find whether a favourable trade exists at all.

Usage:
  CKPT_MAIN=... CKPT_SPEC=... PATCH=384 N_PATCH=600 python regime_switch_eval.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, zarr
from scipy import ndimage
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from conv_lstm_lit_dual import ConvLSTMLitDual
from zarr_dual_datamodule import DualPatchConfig, DualWindowDataset

CKPT_MAIN = os.environ["CKPT_MAIN"]
CKPT_SPEC = os.environ["CKPT_SPEC"]
PATCH   = int(os.environ.get("PATCH", 384))
N_PATCH = int(os.environ.get("N_PATCH", 600))
EVAL_YEAR = os.environ.get("EVAL_YEAR", "2020")
R_SWITCH = [int(x) for x in os.environ.get("R_SWITCH", "5:10:20").split(":")]
BETAS    = [float(x) for x in os.environ.get("BETAS", "0:0.25:0.5:0.75:1.0:1.5").split(":")]
_BP, _BM = 256, 20
MIN_POS = max(1, round(_BM * (PATCH / _BP) ** 2))
OFFS = {"2015":0,"2016":365,"2017":731,"2018":1096,"2019":1461,"2020":1826}
K = 0.005                      # headline operating point
NEAR_R, FAR_R = 3, 10          # radii defining the two reported lift numbers


def rank01(a):
    if a.size == 0: return a
    return ((rankdata(a, method="average") - 1.0) / max(a.size - 1, 1)).astype(np.float32)


def lift(prob, truth, land, f=K):
    v = prob[land]
    if v.size == 0 or truth.sum() == 0: return np.nan
    k = max(1, int(round(f * v.size)))
    thr = np.partition(v, -k)[-k]
    sel = land & (prob >= thr)
    tp = float((truth & sel).sum())
    prec = tp / max(float(sel.sum()), 1)
    base = float(truth.sum()) / float(land.sum())
    return prec / base if base > 0 else np.nan


def main():
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = zarr.open_group(f"cube_daily_smgrid_{EVAL_YEAR}.zarr", mode="r")

    m_main = ConvLSTMLitDual.load_from_checkpoint(CKPT_MAIN, map_location=dev).eval().to(dev)
    m_spec = ConvLSTMLitDual.load_from_checkpoint(CKPT_SPEC, map_location=dev).eval().to(dev)
    print(f"[info] main={os.path.basename(CKPT_MAIN)}\n[info] spec={os.path.basename(CKPT_SPEC)}", flush=True)

    def mk(fh):
        return DualWindowDataset(DualPatchConfig(
            zarr_paths=(f"cube_daily_smgrid_{EVAL_YEAR}.zarr",),
            stats_path="channel_stats_2015_2018.json",
            slow_cube_path="cube_slow_8day.zarr", day_offset=OFFS[EVAL_YEAR],
            patch_size=PATCH, samples_per_epoch=N_PATCH*3, seed=21,
            min_pos_pixels=MIN_POS, pos_frac=1.0, deterministic=True,
            fire_history=fh, fire_history_lags=(3,4,5), fire_history_distance=fh,
            use_lightning=not fh, use_slope_aspect=not fh, use_wind_dir=not fh))
    ds_m, ds_s = mk(True), mk(False)

    res = {(r, b): {"near": [], "far": [], "all": []} for r in R_SWITCH for b in BETAS}
    base_near, base_far, base_all = [], [], []
    pooled_y, pooled_base, pooled_cmb = [], [], {(r,b): [] for r in R_SWITCH for b in BETAS}
    kept = 0

    for i in range(N_PATCH*3):
        if kept >= N_PATCH: break
        bm = ds_m[i]
        land = bm["mask"].numpy() > 0.5
        truth = (bm["y"][-1].numpy() > 0) & land
        if truth.sum() < MIN_POS or land.mean() < 0.5: continue
        bs = ds_s[i]

        with torch.no_grad():
            pm = torch.sigmoid(m_main.forward_seq(bm["x_slow"].unsqueeze(0).to(dev),
                 bm["x_fast"].unsqueeze(0).to(dev), bm["x_cat"].unsqueeze(0).to(dev))[:,-1])[0].cpu().numpy()
            ps = torch.sigmoid(m_spec.forward_seq(bs["x_slow"].unsqueeze(0).to(dev),
                 bs["x_fast"].unsqueeze(0).to(dev), bs["x_cat"].unsqueeze(0).to(dev))[:,-1])[0].cpu().numpy()

        recent = (bm["x_fast"][-1,:,:,7].numpy() > 0.5) & land
        dist = ndimage.distance_transform_edt(~recent) if recent.any() else np.full(land.shape, np.inf, np.float32)
        newfire_near = truth & ~(dist <= NEAR_R)
        newfire_far  = truth & ~(dist <= FAR_R)

        base_all.append(lift(pm, truth, land))
        if newfire_near.sum() >= 3: base_near.append(lift(pm, newfire_near, land))
        if newfire_far.sum()  >= 3: base_far.append(lift(pm, newfire_far, land))
        pooled_y.append(truth[land].astype(np.int8)); pooled_base.append(pm[land].astype(np.float32))

        for r in R_SWITCH:
            far = (dist >= r) & land
            near = (~far) & land
            comb_base = np.zeros_like(pm, np.float32)
            if near.any(): comb_base[near] = rank01(pm[near])
            rf = rank01(ps[far]) if far.any() else None
            for b in BETAS:
                comb = comb_base.copy()
                # beta scales the far field's percentile band; b=0 pins the far
                # field below every near-field pixel (pure main-model behaviour)
                if far.any(): comb[far] = rf * b
                res[(r,b)]["all"].append(lift(comb, truth, land))
                if newfire_near.sum() >= 3: res[(r,b)]["near"].append(lift(comb, newfire_near, land))
                if newfire_far.sum()  >= 3: res[(r,b)]["far"].append(lift(comb, newfire_far, land))
                pooled_cmb[(r,b)].append(comb[land].astype(np.float32))
        kept += 1
        if kept % 50 == 0: print(f"  {kept}...", flush=True)

    Y = np.concatenate(pooled_y)
    ap_base = average_precision_score(Y, np.concatenate(pooled_base))
    print(f"\n*** {kept} patches, {len(Y):,} land px, base rate {Y.mean():.5f} ***")
    print(f"\nBASELINE (main model alone)")
    print(f"  AUC-PR {ap_base:.4f} | all-fire lift {np.nanmean(base_all):6.1f}x | "
          f"near-field(r={NEAR_R}) {np.nanmean(base_near):5.2f}x | FAR-field(r={FAR_R}) {np.nanmean(base_far):5.2f}x")

    print(f"\nREGIME-SWITCHED  (beta = far-field budget weight)")
    print(f"  {'switch':>7} {'beta':>5} {'AUC-PR':>8} {'all-fire':>9} {'near r=3':>9} {'FAR r=10':>9}")
    best = None
    for r in R_SWITCH:
        for b in BETAS:
            ap = average_precision_score(Y, np.concatenate(pooled_cmb[(r,b)]))
            fa = np.nanmean(res[(r,b)]["far"]); ne = np.nanmean(res[(r,b)]["near"]); al = np.nanmean(res[(r,b)]["all"])
            flag = ""
            if fa > np.nanmean(base_far) and al > 0.8*np.nanmean(base_all):
                flag = "  <-- far-field GAIN"
                if best is None or fa > best[0]: best = (fa, r, b, ap, al, ne)
            print(f"  {r:>7} {b:>5} {ap:>8.4f} {al:>9.1f} {ne:>9.2f} {fa:>9.2f}{flag}")
    print(f"\n  baseline far-field = {np.nanmean(base_far):.2f}x  (main model alone)")
    if best:
        print(f"  BEST far-field: {best[0]:.2f}x at switch={best[1]}px beta={best[2]} "
              f"(AUC-PR {best[3]:.4f}, all-fire {best[4]:.1f}x)")
        print(f"  => far-field improves {best[0]/max(np.nanmean(base_far),1e-9):.1f}x over baseline")
    else:
        print("  => NO setting improved far-field without destroying all-fire lift.")
        print("     Conclusion: the budget trade is unfavourable; switching cannot")
        print("     rescue far-field skill at a tolerable near-field cost.")

if __name__ == "__main__":
    main()
