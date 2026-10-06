"""Appendix maps of the input data sources, 2015-2020 means, in the layout of
the report's input maps (north arrow, 1000 km scale bar, ocean in light blue).

  map_lai_mean.png    HiQ-LAI 500 m averaged to the 1 km grid (cube_slow_8day_lai500_lag31.zarr),
                      mean of all 8-day averages 2015-2020
  map_vpd_mean.png    BARRA-C2 VPD at the daily maximum temperature (barra_c2_daily.zarr, 4.4 km)
  map_tmax_mean.png   BARRA-C2 daily maximum 2 m air temperature (barra_c2_daily.zarr, 4.4 km)

The BARRA-C2 means are computed on their native 0.04 deg grid and shown there,
with ocean masked by the nearest model-grid land mask.

  SMOLDER_DATA=... [FIG_OUT=...] python make_input_maps.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from style_smolder import INK                              # noqa: E402  (sets the font)
from smolder.data.io import open_zarr_root                 # noqa: E402

FIG_OUT = os.environ.get("FIG_OUT", HERE)
LON0, LAT0, PX = 112.904998779, -9.005000113999998, 0.01
H, W = 3474, 4110
OCEAN = "#BEDDF2"
N_DAYS = 2192                                              # 2015-01-01 .. 2020-12-31


def draw(field, lon, lat, title, label, cmap, out, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(OCEAN)
    extent = (lon[0], lon[-1], lat[-1], lat[0]) if lat[0] > lat[-1] else (lon[0], lon[-1], lat[0], lat[-1])
    origin = "upper" if lat[0] > lat[-1] else "lower"
    im = ax.imshow(np.ma.masked_invalid(field), extent=extent, origin=origin, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xlim(113, 154); ax.set_ylim(-44, -9.2)
    ax.set_xticks(range(115, 155, 5)); ax.set_xticklabels([f"{v}°E" for v in range(115, 155, 5)])
    ax.set_yticks(range(-40, -9, 5)); ax.set_yticklabels([f"{-v}°S" for v in range(-40, -9, 5)])
    ax.grid(color="#C8C8CC", lw=0.6)
    ax.set_xlabel("Longitude", color=INK); ax.set_ylabel("Latitude", color=INK)
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK)
    ax.annotate("", xy=(116.2, -13.0), xytext=(116.2, -16.6),
                arrowprops=dict(facecolor="black", edgecolor="black", width=4, headwidth=12, headlength=10))
    ax.text(116.2, -12.4, "N", ha="center", va="bottom", fontsize=11, fontweight="bold", color="black")
    km = 1000.0 / (111.32 * np.cos(np.deg2rad(41.0)))            # 1000 km in degrees of longitude at 41 S
    ax.plot([117.0, 117.0 + km], [-41.0, -41.0], color="black", lw=1.8)
    for x in (117.0, 117.0 + km):
        ax.plot([x, x], [-41.3, -40.7], color="black", lw=1.2)
    ax.text(117.0 + km / 2, -40.5, "1000 km", ha="center", va="bottom", fontsize=9, color="black")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cb.set_label(label, color=INK)
    fig.savefig(os.path.join(FIG_OUT, out), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out, flush=True)


def main():
    lm = np.asarray(open_zarr_root("cube_daily_smgrid_2019.zarr")["landmask"][...]) > 0
    lon = LON0 + (np.arange(W) + 0.5) * PX
    lat = LAT0 - (np.arange(H) + 0.5) * PX

    # LAI 500 m: mean of the 8-day averages, every second pixel for drawing
    lg = open_zarr_root("cube_slow_8day_lai500_lag31.zarr")
    first = int(lg.attrs.get("first_full_bin", 0))
    z = lg["X_slow"]
    acc = np.zeros((H // 2 + H % 2, W // 2 + W % 2), np.float64); n = 0
    for b in range(first, z.shape[0]):
        acc += np.nan_to_num(np.asarray(z[b, ::2, ::2, 0], np.float64)); n += 1
    lai = np.where(lm[::2, ::2], acc / n, np.nan)
    draw(lai, lon[::2], lat[::2], "Mean leaf area index for 2015–2020 (HiQ-LAI, 500 m; Yan et al., 2024)",
         "LAI (m² m⁻²)", "YlGn", "map_lai_mean.png", vmin=0, vmax=np.nanpercentile(lai, 99))

    # BARRA-C2 on its native grid
    bg = open_zarr_root("barra_c2_daily.zarr")
    blat, blon = np.asarray(bg["lat"][...]), np.asarray(bg["lon"][...])
    ri = np.clip(np.round((LAT0 - blat) / PX - 0.5).astype(int), 0, H - 1)
    ci = np.clip(np.round((blon - LON0) / PX - 0.5).astype(int), 0, W - 1)
    inside = ((LAT0 - blat) / PX >= 0) & ((LAT0 - blat) / PX < H)
    inside_c = ((blon - LON0) / PX >= 0) & ((blon - LON0) / PX < W)
    blm = lm[ri][:, ci] & inside[:, None] & inside_c[None, :]
    for var, title, label, cmap, out in (
            ("vpd", "Mean vapour pressure deficit at the daily maximum temperature for 2015–2020 (BARRA-C2, 4.4 km)",
             "VPD (kPa)", "magma", "map_vpd_mean.png"),
            ("tasmax", "Mean daily maximum air temperature for 2015–2020 (BARRA-C2, 4.4 km)",
             "Temperature (°C)", "inferno", "map_tmax_mean.png")):
        acc = np.zeros(blm.shape, np.float64); cnt = np.zeros(blm.shape, np.int64)
        for g0 in range(0, N_DAYS, 32):
            a = np.asarray(bg[var][g0:min(g0 + 32, N_DAYS)], np.float32)
            ok = np.isfinite(a)
            acc += np.where(ok, a, 0).sum(0); cnt += ok.sum(0)
        m = np.where(blm & (cnt > 0), acc / np.maximum(cnt, 1), np.nan)
        draw(m, blon, blat, title, label, cmap, out)


if __name__ == "__main__":
    main()
