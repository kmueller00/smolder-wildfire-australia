"""Stage 0 of the anomaly-feature diagnostic, for BARRA-C2 VPD.

Same 2019 case-control sample as anomaly_feature_diagnostic.py (every fire
pixel of the issue days plus N_NEG random non-fire pixel-days, same seed),
same distance threshold and land covers. Features at issue day D, bilinear
from the native 0.04 deg grid at each pixel centre:
  vpd_barra    BARRA-C2 VPD at tasmax
  z_vpd_barra  its standardized anomaly against the BARRA 2015-2018
               climatology (vpd_clim_*, build_barra_climatology.py; floor 0.01 kPa)
Marking rule as before: oriented ROC-AUC far from fire >= 0.60 and >= 0.03
above the raw variable.

Output: barra_anomaly_check_2019.json (OUT).
"""
import datetime as dt
import json
import os

import numpy as np
from scipy.ndimage import map_coordinates
from sklearn.metrics import roc_auc_score

from smolder.data.io import open_zarr_root
from smolder.evaluation.anomaly_feature_diagnostic import FAR_PX, LANDCOVER, LANDCOVERS, OFFSETS, sample

OUT = os.environ.get("OUT", "barra_anomaly_check_2019.json")
GT = (112.904998779, 0.009997566018978103, -9.005000113999998, -0.009997121616580312)


def auc(y, v):
    ok = np.isfinite(v)
    if y[ok].sum() < 20 or (~y[ok]).sum() < 20:
        return None
    a = float(roc_auc_score(y[ok], v[ok]))
    return dict(auc=a, auc_oriented=max(a, 1 - a), n_fire=int(y[ok].sum()))


def main():
    S, _ = sample()
    z = open_zarr_root("barra_c2_daily.zarr")
    lat, lon = np.asarray(z["lat"]), np.asarray(z["lon"])
    c0, a, f0, e = GT
    fi = (f0 + (S["r"] + 0.5) * e - lat[0]) / (lat[1] - lat[0])
    fj = (c0 + (S["c"] + 0.5) * a - lon[0]) / (lon[1] - lon[0])
    info = dict(z.attrs["vpd_climatology"])
    anchors = np.r_[np.asarray(info["anchors_doy"], float), 366.0]
    mu_all = np.asarray(z["vpd_clim_mean"], np.float32)
    sd_all = np.asarray(z["vpd_clim_std"], np.float32)
    raw = np.full(S["y"].size, np.nan)
    zz = np.full(S["y"].size, np.nan)
    for D in np.unique(S["D"]):
        m = S["D"] == D
        g = int(D) + OFFSETS[2019]
        v = map_coordinates(np.asarray(z["vpd"][g], np.float32), [fi[m], fj[m]], order=1, mode="nearest")
        doy = min((dt.date(2019, 1, 1) + dt.timedelta(days=int(D))).timetuple().tm_yday, 365)
        k0 = int(np.searchsorted(anchors, doy, side="right") - 1)
        f = (doy - anchors[k0]) / (anchors[k0 + 1] - anchors[k0])
        k1 = (k0 + 1) % (len(anchors) - 1)
        mu = map_coordinates(mu_all[k0] * (1 - f) + mu_all[k1] * f, [fi[m], fj[m]], order=1, mode="nearest")
        sd = map_coordinates(sd_all[k0] * (1 - f) + sd_all[k1] * f, [fi[m], fj[m]], order=1, mode="nearest")
        raw[m] = v
        zz[m] = (v - mu) / np.maximum(sd, 0.01)
    far = S["d32"] > FAR_PX
    lcn = np.array([LANDCOVER.get(int(v), "other") for v in S["lc"]])
    out = dict(eval_year=2019, n_fire=int(S["y"].sum()), n_nofire=int((~S["y"]).sum()), tables={})
    for sel_name, sel in (("a_all", np.ones(S["y"].size, bool)), ("b_far_10km", far)):
        t = dict(overall={"vpd_barra": auc(S["y"][sel], raw[sel]), "z_vpd_barra": auc(S["y"][sel], zz[sel])})
        t["by_landcover"] = {lc: {"vpd_barra": auc(S["y"][sel & (lcn == lc)], raw[sel & (lcn == lc)]),
                                  "z_vpd_barra": auc(S["y"][sel & (lcn == lc)], zz[sel & (lcn == lc)])}
                             for lc in LANDCOVERS}
        out["tables"][sel_name] = t
    b = out["tables"]["b_far_10km"]["overall"]
    out["marked"] = bool(b["z_vpd_barra"]["auc_oriented"] >= 0.60
                         and b["z_vpd_barra"]["auc_oriented"] >= b["vpd_barra"]["auc_oriented"] + 0.03)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    for k in ("a_all", "b_far_10km"):
        o = out["tables"][k]["overall"]
        print(k, {n: round(v["auc_oriented"], 3) for n, v in o.items()})
    print("by land cover (far):", {lc: {n: round(v["auc_oriented"], 3) for n, v in d.items()}
                                   for lc, d in out["tables"]["b_far_10km"]["by_landcover"].items()})
    print("marked:", out["marked"])


if __name__ == "__main__":
    main()
