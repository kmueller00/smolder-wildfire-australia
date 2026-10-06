"""Leak audit on the inputs as the datamodule builds them (model B's configuration).

The event study of leak_audit_event_study.py, but on the values in x_fast
(last step) and x_slow (newest 8-day average) of a sample whose issue day is
s = F + k, for new fires that are first detected on day F (2019). For k <= -1
the fire lies after the issue day (it is in the target window for k = -3..-1),
so a causal input can show no burn signal there; for k >= 0 it may.

  d(k) = mean over events of [x_event(k) - x_ring(k)] - [x_event(K0) - x_ring(K0)]

with the ring 15 to 40 px around the event pixel (pixels without fire within
3 px from F-30 to F+10). Run twice: causal_inputs=True (what the retrained
model uses) and causal_inputs=False (the earlier inputs, positive control).
The verdict per input: z = d(k) / se(k) for k = -3, -2, -1; |z| < 3 at all three
for the causal inputs.

Writes results/leak_audit_model_inputs_2019.json.
Run: SMOLDER_DATA=... WORKERS=16 python tests/leak_audit_model_inputs.py
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smolder.data.io import open_zarr_root                                        # noqa: E402
from smolder.data.zarr_dual_datamodule import DualPatchConfig, DualWindowDataset  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFF = 1461
P = 384
KS = [-16, -12, -8, -5, -3, -2, -1, 0, 3, 8, 16, 40]
K0 = -16
TS = 256
LAT0, LON0, PX = -9.005, 112.905, 0.01
CENTRES = {"Kimberley": (-17.0, 126.0), "Top End": (-13.5, 132.5), "Gulf": (-16.5, 137.0),
           "Cape York": (-14.0, 143.0), "central Queensland": (-24.0, 150.0),
           "NSW north coast": (-29.5, 152.5), "NSW south coast": (-35.5, 150.0), "Victoria east": (-37.3, 148.5)}
N_EV = int(os.environ.get("N_EV", 120))          # per region
B = dict(fire_history=True, fire_history_lags=(3, 4, 5), fire_history_distance=True,
         use_elevation=True, use_slope_aspect=True, use_fuel_age=True, fuel_age_lookback=1095,
         vpd_source="barra", use_frp=True, use_barra_uv=True, slow_veg="lai500", use_fast_ndvi=True,
         perfect_forecast=False, use_vpd_anomaly=False, use_wind_align=True)
FAST = {"NDVI (fast)": "NDVI (fast)", "temperature (LST or TMAX)": None, "VPD": "VPD", "wind speed": "WIND",
        "wind u": ("wind u/v", 0), "wind v": ("wind u/v", 1)}
SLOW = {"LAI 500 m (newest 8-day average)": "LAI", "soil moisture": "SM", "precipitation": "PPT",
        "biomass": "biomass"}
_DS = {}


def events():
    firms = open_zarr_root("firms_daily.zarr")["n_det"]
    land = np.asarray(open_zarr_root("cube_daily_smgrid_2019.zarr")["landmask"][...]) > 0
    rng = np.random.default_rng(0)
    out = []
    for region, (lat, lon) in CENTRES.items():
        r0 = int((LAT0 - lat) / PX) - TS // 2
        c0 = int((lon - LON0) / PX) - TS // 2
        nd = np.asarray(firms[OFF - 70:OFF + 365, r0:r0 + TS, c0:c0 + TS]) > 0
        dil = np.stack([ndimage.binary_dilation(a, iterations=3) for a in nd])
        ev = []
        for F in range(70, 300):
            i = F + 70
            first = nd[i] & land[r0:r0 + TS, c0:c0 + TS] & ~dil[i - 60:i].any(0)
            for r, c in zip(*np.nonzero(first)):
                if 40 <= r < TS - 40 and 40 <= c < TS - 40:
                    # ring pixels clean from F-30 to F+10
                    ev.append((F, r0 + r, c0 + c))
        sel = rng.choice(len(ev), min(N_EV, len(ev)), replace=False)
        out += [ev[j] for j in sel]
        print(region, len(sel), "events", flush=True)
    return out


def ds_for(causal):
    if causal not in _DS:
        _DS[causal] = DualWindowDataset(DualPatchConfig(
            zarr_paths=("cube_daily_smgrid_2019.zarr",), stats_path="channel_stats_2015_2018.json",
            slow_cube_path="cube_slow_8day.zarr", day_offset=OFF, patch_size=P, samples_per_epoch=1,
            seed=0, deterministic=True, fast_stores=True, causal_inputs=causal, **B))
    return _DS[causal]


def one(args):
    causal, (F, R, C) = args
    ds = ds_for(causal)
    lay = ds.channel_layout()
    y0 = int(np.clip(R - P // 2, 0, 3474 - P)); x0 = int(np.clip(C - P // 2, 0, 4110 - P))
    r, c = R - y0, C - x0
    firms = open_zarr_root("firms_daily.zarr")["n_det"]
    fd = np.asarray(firms[OFF + F - 30:OFF + F + 11, r - 40 + y0:r + 41 + y0, c - 40 + x0:c + 41 + x0]) > 0
    dil = np.stack([ndimage.binary_dilation(a, iterations=3) for a in fd]).any(0)
    yy, xx = np.mgrid[-40:41, -40:41]
    ring = (np.hypot(yy, xx) >= 15) & (np.hypot(yy, xx) <= 40) & ~dil
    land = np.asarray(ds.landmask[y0 + r - 40:y0 + r + 41, x0 + c - 40:x0 + c + 41]) > 0
    ring &= land
    if ring.sum() < 50:
        return None
    vals = {}
    for k in KS:
        t_end = F + k + 1
        if t_end not in ds._target_set:
            return None
        b = ds.sample_at(t_end, y0, x0)
        xf, xs = b["x_fast"].numpy()[-1], b["x_slow"].numpy()[-1]
        win = (slice(r - 40, r + 41), slice(c - 40, c + 41))
        def diff(img):
            return float(img[r, c] - img[win][ring].mean())
        for name, key in FAST.items():
            if key is None:
                key = "TMAX" if "TMAX" in lay["fast"] else "LST"
            if isinstance(key, tuple):
                ch = lay["fast"][key[0]][key[1]]
            else:
                ch = lay["fast"][key][0]
            vals.setdefault(name, {})[k] = diff(xf[..., ch])
        for name, key in SLOW.items():
            vals.setdefault(name, {})[k] = diff(xs[..., lay["slow"][key][0]])
    return vals


def main():
    ev = events()
    res = {}
    for causal in (True, False):
        ds = ds_for(causal)
        ds._target_set = set(int(t) for t in ds.targets)
        with Pool(int(os.environ.get("WORKERS", 16))) as pool:          # forked: the dataset is inherited
            outs = [o for o in pool.map(one, [(causal, e) for e in ev], chunksize=4) if o is not None]
        tab = {}
        for name in outs[0]:
            m = np.array([[o[name][k] - o[name][K0] for k in KS] for o in outs])
            mean, se = m.mean(0), m.std(0) / np.sqrt(len(m))
            z = {k: float(mean[i] / se[i]) if se[i] > 0 else 0.0 for i, k in enumerate(KS)}
            tab[name] = {"n_events": len(m), "k": KS, "mean": mean.tolist(), "se": se.tolist(),
                         "max_abs_z_before_fire": max(abs(z[k]) for k in (-3, -2, -1)),
                         "clean": all(abs(z[k]) < 3 for k in (-3, -2, -1))}
        res["causal_inputs" if causal else "earlier inputs (control)"] = tab
        print("causal_inputs =", causal, "events", len(outs))
        for name, t in tab.items():
            print(f"   {name:36s} " + "  ".join(f"{k}:{v:+.3f}({s:.3f})" for k, v, s in zip(KS, t["mean"], t["se"]))
                  + f"   max|z| k=-3..-1 {t['max_abs_z_before_fire']:.1f} {'clean' if t['clean'] else 'LEAK'}")
    json.dump(res, open(os.path.join(REPO, "results", "leak_audit_model_inputs_2019.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
