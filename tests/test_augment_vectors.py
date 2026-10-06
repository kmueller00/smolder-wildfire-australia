"""Flips and rotations (augment=True) keep the direction inputs consistent.

For real 2019 samples of model B's configuration (causal inputs, compact
statics as in training) and every one of the 8 orientations:
  1. scalar channels are only moved: x_aug == rot_flip(x)
  2. aspect is the downhill direction of the elevation channel: the mean
     cosine between (aspect east, aspect north) and the downhill direction of
     the elevation channel exceeds 0.9, before and after the transform
  3. wind u/v still agree with the fire geometry: the downwind alignment
     recomputed from the augmented u, v and the augmented newest fire map
     equals the (rotation-invariant) alignment channel of the augmented sample
Run: SMOLDER_DATA=... python tests/test_augment_vectors.py
"""
import os
import sys

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smolder.data.zarr_dual_datamodule import DualPatchConfig, DualWindowDataset, expand_compact  # noqa: E402

B = dict(fire_history=True, fire_history_lags=(3, 4, 5), fire_history_distance=True,
         use_elevation=True, use_slope_aspect=True, use_fuel_age=True, fuel_age_lookback=1095,
         vpd_source="barra", use_frp=True, use_barra_uv=True, slow_veg="lai500", use_fast_ndvi=True,
         perfect_forecast=False, use_vpd_anomaly=False, use_wind_align=True)


def gradient_cos(elev, ae, an):
    """mean cosine between the aspect vector and the downhill direction of elev (pixels with slope)"""
    gy, gx = np.gradient(elev)
    de, dn = -gx, gy                      # downhill: east = -dZ/dcol, north = -dZ/dnorth = +dZ/drow
    m = np.hypot(de, dn) > np.percentile(np.hypot(de, dn), 50)
    return float(((de * ae + dn * an) / (np.hypot(de, dn) * np.hypot(ae, an) + 1e-9))[m].mean())


def align_from(u, v, fire):
    rr, cc = np.meshgrid(np.arange(fire.shape[0]), np.arange(fire.shape[1]), indexing="ij")
    dist, (nr, nc) = ndimage.distance_transform_edt(~fire, return_indices=True)
    de = (cc - nc).astype(np.float32); dn = (nr - rr).astype(np.float32)
    norm = np.hypot(de, dn) * np.hypot(u, v)
    return np.where((dist > 0) & (norm > 1e-6), (u * de + v * dn) / np.maximum(norm, 1e-6), 0.0)


def main():
    kw = dict(zarr_paths=("cube_daily_smgrid_2019.zarr",), stats_path="channel_stats_2015_2018.json",
              slow_cube_path="cube_slow_8day.zarr", day_offset=1461, patch_size=256, samples_per_epoch=1,
              seed=0, fast_stores=True, causal_inputs=True, compact_statics=True, **B)
    ds = DualWindowDataset(DualPatchConfig(deterministic=True, augment=False, **kw))
    lay = ds.channel_layout()["fast"]
    iel = lay["elevation"][0]; ias = lay["aspect"][0]; iu = lay["wind u/v"][0]
    ifh = lay["fire history"][0]; iwa = lay["downwind alignment"][0]
    bst = ds.barra_stats
    rng = np.random.default_rng(5)
    n_ok = 0
    for _ in range(6):
        t_end = int(rng.choice(ds.targets)); y0 = int(rng.integers(1500, 2900)); x0 = int(rng.integers(2900, 3800))
        base = ds.sample_at(t_end, y0, x0)
        if (base["x_fast"][-1, :, :, iwa - ds.static_doy_width] != 0).sum() < 100:
            continue                                     # needs fire in the patch for check 3
        shares = []
        for k in range(4):
            for flip in (0, 1):
                ds.rng = type("R", (), {"_v": iter([k, flip]), "integers": lambda self, a, b: next(self._v)})()
                out = {key: val.clone() for key, val in base.items()}
                ds._augment(out)
                def rf(t, dims):
                    t = torch.rot90(t, k, dims=dims)
                    return torch.flip(t, dims=(dims[1],)) if flip else t
                full0 = expand_compact({kk: v[None] for kk, v in base.items()})
                full1 = expand_compact({kk: v[None] for kk, v in out.items()})
                x0f, x1f = full0["x_fast"][0], full1["x_fast"][0]
                # 1. scalar channels moved only
                vec = {ias, ias + 1, iu, iu + 1}
                for c in range(x0f.shape[-1]):
                    if c not in vec:
                        assert torch.allclose(rf(x0f[..., c], (1, 2)), x1f[..., c], atol=1e-6), ("scalar", c, k, flip)
                # 2. aspect vs elevation
                g0 = gradient_cos(x0f[-1, :, :, iel].numpy(), x0f[-1, :, :, ias].numpy(), x0f[-1, :, :, ias + 1].numpy())
                g1 = gradient_cos(x1f[-1, :, :, iel].numpy(), x1f[-1, :, :, ias].numpy(), x1f[-1, :, :, ias + 1].numpy())
                assert g0 > 0.9 and g1 > 0.9, ("aspect", k, flip, g0, g1)    # aspect = downhill direction
                # 3. wind vs fire geometry
                u = x1f[-1, :, :, iu].numpy() * bst["uas"][1] + bst["uas"][0]
                v = x1f[-1, :, :, iu + 1].numpy() * bst["vas"][1] + bst["vas"][0]
                fire = (x1f[-1, :, :, ifh].numpy() > 0.5) & (full1["mask"][0].numpy() > 0)
                lm = full1["mask"][0].numpy() > 0
                got = x1f[-1, :, :, iwa].numpy()
                ref = align_from(u, v, fire) * lm
                # equidistant nearest fires are resolved differently once the patch is turned,
                # so compare the share of pixels that disagree, with a wrong turn as control
                bad = float((np.abs(ref - got) > 0.05)[lm].mean())
                u0 = x0f[-1, :, :, iu].numpy() * bst["uas"][1] + bst["uas"][0]
                v0 = x0f[-1, :, :, iu + 1].numpy() * bst["vas"][1] + bst["vas"][0]
                ctl = align_from(rf(torch.tensor(u0), (0, 1)).numpy(), rf(torch.tensor(v0), (0, 1)).numpy(), fire) * lm
                bad_ctl = float((np.abs(ctl - got) > 0.05)[lm].mean())
                shares.append((k, flip, bad, bad_ctl))
                assert bad < 0.02, ("wind", k, flip, bad, bad_ctl)
        n_ok += 1
        print(f"sample t_end={t_end} ({y0},{x0}): 8 orientations ok, aspect-elevation cosine {g0:+.3f}; "
              "wind: share of pixels disagreeing (turned u/v | u/v only moved, control) "
              + " ".join(f"k{k}f{f}:{b:.4f}|{c:.3f}" for k, f, b, c in shares), flush=True)
    assert n_ok >= 3, n_ok
    print("augmentation keeps aspect and wind consistent in all 8 orientations")


if __name__ == "__main__":
    main()
