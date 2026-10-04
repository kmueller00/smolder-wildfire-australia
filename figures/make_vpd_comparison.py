"""BARRA-C2 VPD (4.4 km) against the ERA5-based VPD at Tmax (Montes et al.
2021, about 31 km) on four dates of 2019, both on the 1 km model grid.

Rows: dates. Columns: ERA5-based VPD, BARRA-C2 VPD (bilinear from 0.04 deg),
BARRA-C2 minus ERA5-based, and a 3 x 3 degree zoom (green box, south-east
New South Wales coast) of both at full 1 km resolution.
Needs SMOLDER_DATA and barra_c2_daily.zarr. Writes fig_vpd_comparison.png.
"""
import datetime as dt
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import map_coordinates

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from smolder.data.io import daily_cube, open_zarr_root   # noqa: E402
from style_smolder import INK, MUTED, PANEL_BG, SPINE_COLOR, new_figure  # noqa: E402

DATES = [dt.date(2019, 1, 15), dt.date(2019, 4, 15), dt.date(2019, 7, 15), dt.date(2019, 10, 15)]
GT = (112.904998779, 0.009997566018978103, -9.005000113999998, -0.009997121616580312)
DOWN = 4
ZOOM = (-34.5, 149.0, 3.0)          # south-east NSW: top lat, left lon, size in degrees
OCEAN = "#C9D6E3"


def load(day):
    cube = open_zarr_root(daily_cube(day.year))
    land = np.asarray(cube["landmask"][...]) > 0
    names = cube.attrs.get("channels")
    era = np.asarray(cube["X"][day.timetuple().tm_yday - 1, :, :, list(names).index("vpd") if names else 2],
                     np.float32)
    z = open_zarr_root("barra_c2_daily.zarr")
    lat, lon = np.asarray(z["lat"]), np.asarray(z["lon"])
    H, W = land.shape
    c0, a, f0, e = GT
    fi = (f0 + (np.arange(H) + 0.5) * e - lat[0]) / (lat[1] - lat[0])
    fj = (c0 + (np.arange(W) + 0.5) * a - lon[0]) / (lon[1] - lon[0])
    FI, FJ = np.meshgrid(fi, fj, indexing="ij")
    g = (day - dt.date(2015, 1, 1)).days
    bar = map_coordinates(np.asarray(z["vpd"][g], np.float32), [FI, FJ], order=1, mode="nearest")
    era[~land], bar[~land] = np.nan, np.nan
    return era, bar, land


def show(ax, img, ext, **kw):
    ax.set_facecolor(OCEAN)
    im = ax.imshow(img, extent=ext, interpolation="nearest", **kw)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(SPINE_COLOR); s.set_linewidth(0.6)
    return im


def main():
    c0, a, f0, e = GT
    fig = new_figure((19.5, 3.45 * len(DATES) + 0.6))
    gs = fig.add_gridspec(len(DATES), 9, width_ratios=[1, 1, 0.035, 0.16, 1, 0.035, 0.2, 0.62, 0.62],
                          wspace=0.08, hspace=0.1, left=0.04, right=0.99, top=0.93, bottom=0.02)
    for r, day in enumerate(DATES):
        era, bar, land = load(day)
        H, W = land.shape
        ext = [c0, c0 + W * a, f0 + H * e, f0]
        ok = land & np.isfinite(era) & np.isfinite(bar)
        vmax = float(np.percentile(np.r_[era[ok], bar[ok]], 99))
        diff = bar - era
        dm = float(np.percentile(np.abs(diff[ok]), 99))
        r_ = np.corrcoef(era[ok], bar[ok])[0, 1]
        ax_e, ax_b, cax1, ax_d, cax2, ax_ze, ax_zb = [fig.add_subplot(gs[r, k]) for k in (0, 1, 2, 4, 5, 7, 8)]
        im1 = show(ax_e, era[::DOWN, ::DOWN], ext, cmap="magma", vmin=0, vmax=vmax)
        show(ax_b, bar[::DOWN, ::DOWN], ext, cmap="magma", vmin=0, vmax=vmax)
        im3 = show(ax_d, diff[::DOWN, ::DOWN], ext, cmap="RdBu_r", norm=TwoSlopeNorm(0, -dm, dm))
        top, left, size = ZOOM
        r0, r1 = int((top - f0) / e), int((top - size - f0) / e)
        k0, k1 = int((left - c0) / a), int((left + size - c0) / a)
        zext = [left, left + size, top - size, top]
        show(ax_ze, era[r0:r1, k0:k1], zext, cmap="magma", vmin=0, vmax=vmax)
        show(ax_zb, bar[r0:r1, k0:k1], zext, cmap="magma", vmin=0, vmax=vmax)
        for ax in (ax_e, ax_b, ax_d):
            ax.add_patch(plt.Rectangle((left, top - size), size, size, fill=False, ec="#00E83A", lw=1.4))
        ax_e.set_ylabel(day.strftime("%-d %B %Y"), fontsize=11.5, fontweight="bold", color=INK)
        ax_d.text(0.02, 0.04, f"r = {r_:.2f}   mean difference {np.mean(diff[ok]):+.2f} kPa",
                  transform=ax_d.transAxes, fontsize=9, color=INK,
                  bbox=dict(fc="white", ec=MUTED, lw=0.6, alpha=0.9, pad=2))
        for cax, im, lab in ((cax1, im1, "VPD (kPa)"), (cax2, im3, "difference (kPa)")):
            cb = fig.colorbar(im, cax=cax)
            cb.set_label(lab, fontsize=9, color=INK); cb.ax.tick_params(labelsize=8.5)
            cb.outline.set_edgecolor(SPINE_COLOR)
        if r == 0:
            for ax, t in ((ax_e, "ERA5-based, about 31 km"), (ax_b, "BARRA-C2, 4.4 km"),
                          (ax_d, "BARRA-C2 minus ERA5-based"), (ax_ze, "Zoom, ERA5-based"),
                          (ax_zb, "Zoom, BARRA-C2")):
                ax.set_title(t, fontsize=11, fontweight="bold", color=INK, loc="left")
    fig.suptitle("Vapour pressure deficit at the daily maximum temperature: ERA5-based against BARRA-C2",
                 fontsize=14.5, fontweight="bold", color=INK, y=0.985)
    fig.savefig(os.path.join(HERE, "fig_vpd_comparison.png"), dpi=170, bbox_inches="tight", facecolor="white")
    print("wrote fig_vpd_comparison.png")


if __name__ == "__main__":
    main()
