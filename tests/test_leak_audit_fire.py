"""Leak audit of every fire-derived input and the target, against the raw
daily VIIRS detections (firms_daily.zarr: FIRMS archive, nominal/high
confidence, vegetation fires, UTC days; independent of the label rasters that
y_fire_3d was built from, >= 99.9 % identical per day).

For random patches and every fast step j (issue day s of that step, global
day g) the sample built by DualWindowDataset must match:
  target        any detection on g+1 .. g+3
  history lag L any detection on g-L+1 .. g-L+3 (L = 3, 4, 5; newest ends on g)
  distance      exp(-EDT(no fire in the lag-3 window) / 5)
  FRP features  log1p(sum frp g-2..g)/5, log1p(sum n_det)/3, sum n_night / sum n_det
  fuel age      from age_px[g-3]; age_px itself must only use y_fire_3d up to its own index
  downwind alignment  zero wherever the lag-3 window holds no fire in the patch
Agreement is reported as the share of land pixels that differ; the label
rasters and FIRMS differ on < 0.1 % of fire pixels, so the threshold is 1e-3 of
pixels for the binary channels. A window shifted by one day would differ on
a large share of fire pixels, so the test separates the two clearly.

Run: SMOLDER_DATA=... python tests/test_leak_audit_fire.py  (writes results/leak_audit_fire.json)
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smolder.data.io import open_zarr_root                                        # noqa: E402
from smolder.data.zarr_dual_datamodule import DualPatchConfig, DualWindowDataset  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = dict(fire_history=True, fire_history_lags=(3, 4, 5), fire_history_distance=True,
         use_elevation=True, use_slope_aspect=True, use_fuel_age=True, fuel_age_lookback=1095,
         vpd_source="barra", use_frp=True, use_barra_uv=True, slow_veg="lai500", use_fast_ndvi=True,
         perfect_forecast=False, use_vpd_anomaly=False, use_wind_align=True)
OFF = {2019: 1461, 2020: 1826}
N_PATCH = int(os.environ.get("N_PATCH", 30))


def main():
    firms = open_zarr_root("firms_daily.zarr")
    age_store = open_zarr_root("fire_inputs_continental.zarr")["age_px"]
    yc = open_zarr_root("fire_inputs_continental.zarr")["y_fire_3d"]
    res = {}
    rng = np.random.default_rng(1)
    for year, off in OFF.items():
        ds = DualWindowDataset(DualPatchConfig(
            zarr_paths=(f"cube_daily_smgrid_{year}.zarr",), stats_path="channel_stats_2015_2018.json",
            slow_cube_path="cube_slow_8day.zarr", day_offset=off, patch_size=384, samples_per_epoch=1,
            seed=0, deterministic=True, fast_stores=True, **B))
        lay = ds.channel_layout()["fast"]
        P = ds.ph
        bad = {k: [0, 0] for k in ("target", "hist lag3", "hist lag4", "hist lag5", "hist lag3 shifted +1 (control)",
                                   "distance", "frp sum", "frp n_det", "frp night share", "fuel age",
                                   "age_px causal", "downwind zero without fire")}
        n_done = 0
        while n_done < N_PATCH:
            t_end = int(rng.choice(ds.targets))
            y0 = int(rng.integers(0, 3474 - P)); x0 = int(rng.integers(0, 4110 - P))
            g_last = t_end - 1 + off
            nd_win = np.asarray(firms["n_det"][g_last - 20:g_last + 4, y0:y0 + P, x0:x0 + P]) > 0
            if nd_win.sum() < 200:                         # patches with fire only
                continue
            b = ds.sample_at(t_end, y0, x0)
            xf = b["x_fast"].numpy(); y = b["y"].numpy()
            lm = np.asarray(ds.landmask[y0:y0 + P, x0:x0 + P]) > 0
            T = xf.shape[0]
            lo = t_end - T + off - 10                      # raw days lo .. g_last+3
            nd = np.asarray(firms["n_det"][lo:g_last + 4, y0:y0 + P, x0:x0 + P]).astype(np.float64)
            frp = np.asarray(firms["frp_sum"][lo:g_last + 4, y0:y0 + P, x0:x0 + P]).astype(np.float64)
            nn = np.asarray(firms["n_night"][lo:g_last + 4, y0:y0 + P, x0:x0 + P]).astype(np.float64)
            fire = (nd > 0) & lm[None]
            anyf = lambda a, c: fire[a - lo:c - lo + 1].any(0)            # raw days a..c inclusive
            for j in [0, T // 2, T - 1]:
                g = t_end - T + j + off
                def cmp(name, got, ref):
                    d = (got.astype(np.float64) != ref.astype(np.float64)) & lm
                    bad[name][0] += int(d.sum()); bad[name][1] += int(lm.sum())
                cmp("target", y[j] > 0.5, anyf(g + 1, g + 3))
                for L, ch in zip((3, 4, 5), lay["fire history"][:3]):
                    cmp(f"hist lag{L}", xf[j, :, :, ch] > 0.5, anyf(g - L + 1, g - L + 3))
                cmp("hist lag3 shifted +1 (control)", xf[j, :, :, lay["fire history"][0]] > 0.5, anyf(g - 1, g + 1))
                h0 = xf[j, :, :, lay["fire history"][0]] > 0.5
                ref_d = np.exp(-ndimage.distance_transform_edt(~h0) / 5.0) if h0.any() else np.zeros_like(h0, float)
                dd = np.abs(xf[j, :, :, lay["fire history"][3]] - ref_d) > 1e-4
                bad["distance"][0] += int((dd & lm).sum()); bad["distance"][1] += int(lm.sum())
                s3 = slice(g - 2 - lo, g - lo + 1)
                fs, ns, ws = frp[s3].sum(0), nd[s3].sum(0), nn[s3].sum(0)
                ref = [np.log1p(fs) / 5, np.log1p(ns) / 3, np.where(ns > 0, ws / np.maximum(ns, 1), 0)]
                for name, ch, r in zip(("frp sum", "frp n_det", "frp night share"), lay["fire radiative power"], ref):
                    dd = (np.abs(xf[j, :, :, ch] - r * lm) > 1e-3) & lm
                    bad[name][0] += int(dd.sum()); bad[name][1] += int(lm.sum())
                LB = 1095
                a = np.asarray(age_store[g - 3, y0:y0 + P, x0:x0 + P]).astype(np.float64)
                a = np.minimum(np.where(a == 65535, LB, a), LB)
                dd = (np.abs(xf[j, :, :, lay["fuel age"][0]] - (a - LB / 2) / (LB / 2)) > 1e-5) & lm
                bad["fuel age"][0] += int(dd.sum()); bad["fuel age"][1] += int(lm.sum())
                # age_px[i] = k means y_fire_3d[i-k] > 0 and y_fire_3d[i-k+1 .. i] == 0 (only the past)
                i = g - 3
                ar = np.asarray(age_store[i, y0:y0 + P, x0:x0 + P]).astype(np.int64)
                sel = lm & (ar < 60)
                if sel.any():
                    yy = np.asarray(yc[i - 60:i + 1, y0:y0 + P, x0:x0 + P]) > 0       # index i-60 .. i
                    rr, cc = np.nonzero(sel)
                    k = ar[rr, cc]
                    ok = yy[60 - k, rr, cc]
                    for q in range(len(k)):
                        if k[q] > 0 and yy[60 - k[q] + 1:61, rr[q], cc[q]].any():
                            ok[q] = False
                    bad["age_px causal"][0] += int((~ok).sum()); bad["age_px causal"][1] += int(sel.sum())
                wa = xf[j, :, :, lay["downwind alignment"][0]]
                if not h0.any():
                    bad["downwind zero without fire"][0] += int((np.abs(wa) > 0).sum())
                bad["downwind zero without fire"][1] += 1
            n_done += 1
            print(year, n_done, flush=True)
        res[year] = {k: dict(differ=v[0], checked=v[1], share=(v[0] / v[1] if v[1] else None)) for k, v in bad.items()}
        for k, v in res[year].items():
            print(year, f"{k:34s} {v['differ']:>9} / {v['checked']:>11}  share {v['share']}")
    json.dump(res, open(os.path.join(REPO, "results", "leak_audit_fire.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
