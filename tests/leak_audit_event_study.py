"""Leak audit of the continuous inputs: event study around the start of new fires.

Events: pixels whose first VIIRS detection (firms_daily) is on day F of 2019,
with no detection within 3 px during the 60 days before F, in 8 fire-active
256 px tiles. Control: mean over the ring 15 to 40 px around the event of
pixels with no detection within 3 px from F-30 to F+10.

For an input x labelled by day, d(k) = mean over events of
  [x_event(F+k) - x_ring(F+k)] - [x_event(F-16) - x_ring(F-16)]
for k = -21 .. +7. The model sees x up to its issue day s, and a target window
that contains F has s in F-3 .. F-1. An input built only from data up to its
own day can show the burn at k >= 0 at the earliest (the fire is detected on
F; it may have started late on F-1). A burn signal that is already present at
k <= -2, or that steps up at k = -1 like at k = 0, means the input on day s
holds information from after s. Weather that is spatially smooth over 15 to
40 km cancels in the difference.

Inputs: daily cube channels sm, wind, precip, lst_day, ndvi, lai (5 km),
BARRA-C2 vpd, uas, vas (nearest 0.04 deg cell; the ring uses the cells of
its pixels), and the 500 m LAI slow store (8-day averages, indexed by the end
day of each average relative to F).

Writes results/leak_audit_event_study_2019.json.
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smolder.data.io import open_zarr_root   # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFF = 1461                                   # 2019-01-01, global day
TS = 256
LAT0, LON0, PX = -9.005, 112.905, 0.01
CENTRES = {"Kimberley": (-17.0, 126.0), "Top End": (-13.5, 132.5), "Gulf": (-16.5, 137.0),
           "Cape York": (-14.0, 143.0), "central Queensland": (-24.0, 150.0),
           "NSW north coast": (-29.5, 152.5), "NSW south coast": (-35.5, 150.0), "Victoria east": (-37.3, 148.5)}
KS = np.arange(-21, 8)
K0 = -16
MAX_EV = 400


def tile(lat, lon):
    r = int((LAT0 - lat) / PX) - TS // 2
    c = int((lon - LON0) / PX) - TS // 2
    return r, c


def main():
    rng = np.random.default_rng(0)
    cube = open_zarr_root("cube_daily_smgrid_2019.zarr")
    names = list(cube.attrs["dyn_vars"])
    firms = open_zarr_root("firms_daily.zarr")["n_det"]
    barra = open_zarr_root("barra_c2_daily.zarr")
    blat, blon = np.asarray(barra["lat"][...]), np.asarray(barra["lon"][...])
    lai500 = open_zarr_root(os.environ.get("LAI_STORE", "cube_slow_8day_lai500.zarr"))
    bstart = np.asarray(lai500["bin_start_day"][...])
    acc = {}
    def add(name, k, v):
        acc.setdefault(name, {}).setdefault(int(k), []).append(float(v))
    n_events = 0
    for region, (lat, lon) in CENTRES.items():
        r0, c0 = tile(lat, lon)
        sl = (slice(r0, r0 + TS), slice(c0, c0 + TS))
        land = np.asarray(cube["landmask"][sl]) > 0
        nd = np.asarray(firms[OFF - 70:OFF + 365, sl[0], sl[1]]) > 0          # index i -> 2019 day i-70
        dil = np.stack([ndimage.binary_dilation(a, iterations=3) for a in nd])
        ev = []
        for F in range(70, 355):
            i = F + 70
            first = nd[i] & land & ~dil[i - 60:i].any(0)
            rr, cc = np.nonzero(first)
            ev += [(F, r, c) for r, c in zip(rr, cc) if 40 <= r < TS - 40 and 40 <= c < TS - 40]
        if not ev:
            continue
        ev = [ev[j] for j in rng.choice(len(ev), min(MAX_EV, len(ev)), replace=False)]
        print(region, len(ev), "events", flush=True)
        X = {nm: np.asarray(cube["X"][:, sl[0], sl[1], ch], np.float32) for ch, nm in enumerate(names)
             if nm in ("sm", "wind", "precip", "lst_day", "ndvi", "lai")}
        bi = np.abs(blat[None, :] - (LAT0 - (r0 + np.arange(TS) + 0.5) * PX)[:, None]).argmin(1)
        bj = np.abs(blon[None, :] - (LON0 + (c0 + np.arange(TS) + 0.5) * PX)[:, None]).argmin(1)
        i0, i1, j0, j1 = bi.min(), bi.max() + 1, bj.min(), bj.max() + 1
        Bv = {v: np.asarray(barra[v][OFF:OFF + 365, i0:i1, j0:j1], np.float32)[:, bi - i0][:, :, bj - j0]
              for v in ("vpd", "uas", "vas")}
        L5 = np.asarray(lai500["X_slow"][:, sl[0], sl[1], 0], np.float32)
        yy, xx = np.mgrid[-40:41, -40:41]
        ring = (np.hypot(yy, xx) >= 15) & (np.hypot(yy, xx) <= 40)
        for F, r, c in ev:
            i = F + 70
            win = (slice(r - 40, r + 41), slice(c - 40, c + 41))
            clean = ring & ~dil[i - 30:i + 11, win[0], win[1]].any(0) & land[win]
            if clean.sum() < 50:
                continue
            n_events += 1
            for nm, A in (list(X.items()) + list(Bv.items())) if os.environ.get("ONLY_LAI500") != "1" else []:
                e = A[F + KS, r, c]; ctl = np.nanmean(A[F + KS][:, win[0], win[1]][:, clean], 1)
                e0 = A[F + K0, r, c]; c0_ = np.nanmean(A[F + K0][win[0], win[1]][clean])
                for k, de, dc in zip(KS, e, ctl):
                    v = (de - dc) - (e0 - c0_)
                    if np.isfinite(v):
                        add(nm, k, v)
            gF = F + OFF
            ends = bstart + 7
            for b in np.nonzero((ends - gF >= -48) & (ends - gF <= 16))[0]:
                m = int(ends[b] - gF)
                b0 = int(np.argmin(np.abs(ends - (gF - 40))))
                de = L5[b, r, c] - np.nanmean(L5[b, win[0], win[1]][clean])
                d0 = L5[b0, r, c] - np.nanmean(L5[b0, win[0], win[1]][clean])
                if np.isfinite(de - d0):
                    add("lai500 (8-day average, k = end day - F)", m, de - d0)
    out = {"n_events": n_events, "k0": K0, "series": {}}
    for nm, d in acc.items():
        ks = sorted(d)
        out["series"][nm] = {str(k): dict(mean=float(np.mean(d[k])), se=float(np.std(d[k]) / np.sqrt(len(d[k]))),
                                          n=len(d[k])) for k in ks}
    json.dump(out, open(os.path.join(REPO, "results", os.environ.get("OUT_NAME", "leak_audit_event_study_2019.json")), "w"), indent=1)
    print("events", n_events)
    for nm, s in out["series"].items():
        print(nm)
        print("   " + "  ".join(f"{k}:{v['mean']:+.4f}({v['se']:.4f})" for k, v in s.items()))


if __name__ == "__main__":
    main()
