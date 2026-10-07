"""Maps of the input data, 2015-2020 (supplement), and map cards for the input
figure (fig_model_inputs).

  map_landcover.png     Copernicus LC100 2018 (the map the final model uses for 2020), grouped classes
  map_koppen_geiger.png Koppen-Geiger zones 1991-2020 (Beck et al. 2023)
  map_agb_mean.png      ESA CCI above-ground biomass, mean of the maps 2015-2020
  map_fire_days.png     days with a VIIRS fire detection, 2015-2020 (firms_daily.zarr)
  map_sm_mean.png       SMIPS soil moisture index, mean of the 8-day averages
  map_precip_annual.png ANUClimate precipitation, mean annual sum
  map_lai_mean.png      HiQ-LAI 500 m averaged to 1 km (31-day delay store), mean of the 8-day averages
  map_ndvi_mean.png     MODIS MOD09A1 NDVI, mean of the 8-day composites
  map_wind_mean.png     10 m wind speed of the daily cubes (BARRA-C2 daily mean), mean
  map_vpd_mean.png      BARRA-C2 VPD at the daily maximum temperature, mean (4.4 km)
  map_tmax_mean.png     BARRA-C2 daily maximum 2 m air temperature, mean (4.4 km)

Cards (CARD_DIR, default FIG_OUT/cards): the same fields without title, axis
labels or colour bar, 1124 x 950 px over lon 112.92-153.86, lat -43.69 to -9.05,
with grid, north arrow and scale bar: card_<name>.png, plus card_landmask.png.

The fields are computed once and cached in data/input_maps.npz (MAPS_CACHE).
  SMOLDER_DATA=... [FIG_OUT=...] [CARD_DIR=...] python make_input_maps.py
"""
import glob
import io
import os
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from style_smolder import INK                              # noqa: E402  (sets the font)
from smolder.data.io import open_zarr_root                 # noqa: E402

FIG_OUT = os.environ.get("FIG_OUT", HERE)
CARD_DIR = os.environ.get("CARD_DIR", os.path.join(FIG_OUT, "cards"))
CACHE = os.environ.get("MAPS_CACHE", os.path.join(HERE, "data", "input_maps.npz"))
LON0, LAT0, PX = 112.904998779, -9.005000113999998, 0.01
H, W = 3474, 4110
OCEAN = "#BEDDF2"
N_DAYS = 2192                                              # 2015-01-01 .. 2020-12-31
D = 2                                                      # display every 2nd pixel
LC_GROUPS = [("closed forest", (111, 112, 113, 114, 115, 116), "#1B5E20"),
             ("open forest", (121, 122, 123, 124, 125, 126), "#66A84F"),
             ("shrubland", (20,), "#C8B560"),
             ("herbaceous vegetation", (30,), "#F2E3A0"),
             ("cropland", (40,), "#E58A3C"),
             ("herbaceous wetland", (90,), "#5FB3B3"),
             ("urban", (50,), "#C0392B"),
             ("bare or sparse vegetation", (60,), "#D9D2C5"),
             ("water", (80,), "#3A7CC2")]
KG = {1: ("Af", "#0000FF"), 2: ("Am", "#0078FF"), 3: ("Aw", "#46AAFA"), 4: ("BWh", "#FF0000"),
      5: ("BWk", "#FF9696"), 6: ("BSh", "#F5A500"), 7: ("BSk", "#FFDC64"), 8: ("Csa", "#FFFF00"),
      9: ("Csb", "#C8C800"), 11: ("Cwa", "#96FF96"), 12: ("Cwb", "#64C864"), 14: ("Cfa", "#C8FF50"),
      15: ("Cfb", "#64FF50"), 16: ("Cfc", "#32C800"), 26: ("Dfb", "#00AAFF"), 27: ("Dfc", "#007D7D")}


# ------------------------------------------------------------------ fields
def f_lai():
    lg = open_zarr_root("cube_slow_8day_lai500_lag31.zarr"); z = lg["X_slow"]
    first = int(lg.attrs.get("first_full_bin", 0)); acc = 0; n = 0
    for b in range(first, z.shape[0]):
        acc = acc + np.nan_to_num(np.asarray(z[b, ::D, ::D, 0], np.float64)); n += 1
    return "lai", acc / n


def f_slow():
    s = open_zarr_root("cube_slow_8day.zarr"); z = s["X_slow"]
    ch = list(s.attrs["channels"]); acc = {"SM": 0, "PPT": 0}
    for b in range(z.shape[0]):
        a = np.asarray(z[b, ::D, ::D, :], np.float64)
        for c in acc:
            acc[c] = acc[c] + np.nan_to_num(a[..., ch.index(c)])
    return "slow", np.stack([acc["SM"] / z.shape[0], acc["PPT"] / 6.0])   # mean SM; annual precipitation sum


def f_cube():
    acc = {"ndvi": 0, "wind": 0}; n = 0
    for y in range(2015, 2021):
        g = open_zarr_root(f"cube_daily_smgrid_{y}.zarr"); names = list(g.attrs["dyn_vars"])
        for t in range(0, g["X"].shape[0], 8):                  # one day per 8-day composite
            a = np.asarray(g["X"][t, ::D, ::D, :], np.float64)
            for c in acc:
                acc[c] = acc[c] + np.nan_to_num(a[..., names.index(c)])
            n += 1
    return "cube", np.stack([acc["ndvi"] / n, acc["wind"] / n])


def f_fire():
    f = open_zarr_root("firms_daily.zarr")["n_det"]; cnt = np.zeros((H, W), np.int32)
    for g in range(0, N_DAYS, 16):
        cnt += (np.asarray(f[g:min(g + 16, N_DAYS)]) > 0).sum(0)
    h2, w2 = H // D * D, W // D * D
    return "fire", cnt[:h2, :w2].reshape(h2 // D, D, w2 // D, D).max(axis=(1, 3))


def f_barra(var):
    bg = open_zarr_root("barra_c2_daily.zarr"); acc = 0; cnt = 0
    for g0 in range(0, N_DAYS, 32):
        a = np.asarray(bg[var][g0:min(g0 + 32, N_DAYS)], np.float32); ok = np.isfinite(a)
        acc = acc + np.where(ok, a, 0).sum(0); cnt = cnt + ok.sum(0)
    return var, acc / np.maximum(cnt, 1)


def compute():
    with Pool(6) as pool:
        res = [pool.apply_async(f) for f in (f_lai, f_slow, f_cube, f_fire)]
        res += [pool.apply_async(f_barra, (v,)) for v in ("vpd", "tasmax")]
        out = dict(r.get() for r in res)
    g = open_zarr_root("cube_daily_smgrid_2019.zarr")
    out["land"] = np.asarray(g["landmask"][::D, ::D]) > 0
    out["kg"] = np.asarray(g["koppen_geiger"][::D, ::D])
    lc = open_zarr_root("landcover_yearly.zarr"); yrs = [int(v) for v in lc["years"][...]]
    out["lc"] = np.asarray(lc["landcover"][yrs.index(2018), ::D, ::D])
    ag = open_zarr_root("agb_yearly.zarr"); ay = [int(v) for v in ag["years"][...]]
    out["agb"] = np.nanmean(np.stack([np.asarray(ag["agb"][ay.index(y), ::D, ::D], np.float32) for y in range(2015, 2021)]), 0)
    bg = open_zarr_root("barra_c2_daily.zarr")
    out["blat"], out["blon"] = np.asarray(bg["lat"][...]), np.asarray(bg["lon"][...])
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **out)


# ------------------------------------------------------------------ drawing
def _decor(ax, big):
    ax.set_facecolor(OCEAN)
    ax.set_xlim(112.92, 153.86); ax.set_ylim(-43.69, -9.05)
    ax.set_xticks(range(115, 155, 5)); ax.set_yticks(range(-40, -9, 5))
    ax.grid(color="#C8C8CC", lw=1.2 if not big else 0.6)
    s = 1.0 if big else 2.2
    ax.annotate("", xy=(116.2, -13.0), xytext=(116.2, -16.6),
                arrowprops=dict(facecolor="black", edgecolor="black", width=4 * s, headwidth=12 * s, headlength=10 * s))
    if big:                                       # cards: no text (it would print below 6 pt)
        ax.text(116.2, -12.4, "N", ha="center", va="bottom", fontsize=11, fontweight="bold", color="black")
    km = 1000.0 / (111.32 * np.cos(np.deg2rad(41.0)))
    ax.plot([117.0, 117.0 + km], [-41.0, -41.0], color="black", lw=1.8 * s)
    for x in (117.0, 117.0 + km):
        ax.plot([x, x], [-41.3, -40.7], color="black", lw=1.2 * s)
    if big:
        ax.text(117.0 + km / 2, -40.5, "1000 km", ha="center", va="bottom", fontsize=9, color="black")


def _extent(lat, lon):
    return (lon[0], lon[-1], lat[-1], lat[0]) if lat[0] > lat[-1] else (lon[0], lon[-1], lat[0], lat[-1])


def draw(name, field, lat, lon, title, label, cmap, vmin=None, vmax=None, norm=None, legend=None):
    origin = "upper" if lat[0] > lat[-1] else "lower"
    kw = dict(extent=_extent(lat, lon), origin=origin, cmap=cmap, interpolation="nearest")
    kw.update(dict(norm=norm) if norm is not None else dict(vmin=vmin, vmax=vmax))
    # full map
    fig, ax = plt.subplots(figsize=(10.5, 6.2)); fig.patch.set_facecolor("white")
    im = ax.imshow(np.ma.masked_invalid(field), **kw)
    _decor(ax, True)
    ax.set_xticklabels([f"{v}°E" for v in range(115, 155, 5)])
    ax.set_yticklabels([f"{-v}°S" for v in range(-40, -9, 5)])
    ax.set_xlabel("Longitude", color=INK); ax.set_ylabel("Latitude", color=INK)
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK)
    if legend:
        ax.legend(handles=[Patch(color=c, label=l) for l, c in legend], loc="lower left", bbox_to_anchor=(1.02, 0.0),
                  fontsize=8.5, frameon=False, ncol=1 if len(legend) <= 10 else 2)
    else:
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04); cb.set_label(label, color=INK)
    fig.savefig(os.path.join(FIG_OUT, f"map_{name}.png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # card
    fig = plt.figure(figsize=(11.24, 9.50), dpi=100); ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.ma.masked_invalid(field), **kw); _decor(ax, False); ax.set_aspect("auto")
    ax.tick_params(length=0, labelbottom=False, labelleft=False)
    buf = io.BytesIO(); fig.savefig(buf, dpi=100); plt.close(fig); buf.seek(0)
    Image.open(buf).convert("RGB").save(os.path.join(CARD_DIR, f"card_{name}.png"))
    print("wrote", name, flush=True)


def main():
    if not os.path.exists(CACHE) or os.environ.get("RECOMPUTE") == "1":
        compute()
    z = np.load(CACHE)
    os.makedirs(CARD_DIR, exist_ok=True)
    land = z["land"]
    lat = LAT0 - (np.arange(H)[::D] + 0.5) * PX; lon = LON0 + (np.arange(W)[::D] + 0.5) * PX
    m = lambda a: np.where(land[:a.shape[0], :a.shape[1]], a, np.nan)            # noqa: E731
    # categorical
    lc = z["lc"]; codes = np.full(lc.shape, np.nan)
    for i, (_, cs, _) in enumerate(LC_GROUPS):
        codes[np.isin(lc, cs)] = i
    codes[~land] = np.nan
    cm = ListedColormap([c for _, _, c in LC_GROUPS]); nm = BoundaryNorm(np.arange(-0.5, len(LC_GROUPS)), cm.N)
    draw("landcover", codes, lat, lon, "Land cover 2018 (Copernicus LC100, 100 m)", "", cm, norm=nm,
         legend=[(l, c) for l, _, c in LC_GROUPS])
    kg = z["kg"].astype(float); kidx = np.full(kg.shape, np.nan); keys = sorted(KG)
    for i, k in enumerate(keys):
        kidx[kg == k] = i
    kidx[~land] = np.nan
    cm = ListedColormap([KG[k][1] for k in keys]); nm = BoundaryNorm(np.arange(-0.5, len(keys)), cm.N)
    draw("koppen_geiger", kidx, lat, lon, "Köppen-Geiger climate zones 1991–2020 (Beck et al., 2023)", "", cm,
         norm=nm, legend=[(KG[k][0], KG[k][1]) for k in keys])
    # continuous
    agb = m(z["agb"]); draw("agb_mean", agb, lat, lon, "Mean above-ground biomass 2015–2020 (ESA CCI Biomass)",
                            "AGB (Mg ha⁻¹)", "YlGn", vmin=0, vmax=np.nanpercentile(agb, 99))
    fd = m(z["fire"].astype(float)); fd[fd == 0] = np.nan
    draw("fire_days", fd, lat[:fd.shape[0]], lon[:fd.shape[1]], "Days with a VIIRS fire detection 2015–2020",
         "days (max over 2 x 2 km)", "inferno_r", vmin=1, vmax=np.nanpercentile(fd, 99))
    sm, ppt = z["slow"]
    draw("sm_mean", m(sm), lat, lon, "Mean soil moisture index 2015–2020 (SMIPS)", "soil moisture index (0 to 1)",
         "YlGnBu", vmin=0, vmax=np.nanpercentile(m(sm), 99))
    draw("precip_annual", m(ppt), lat, lon, "Mean annual precipitation 2015–2020 (ANUClimate 2.0)", "mm per year",
         "Blues", vmin=0, vmax=np.nanpercentile(m(ppt), 99))
    lai = m(z["lai"]); draw("lai_mean", lai, lat, lon, "Mean leaf area index 2015–2020 (HiQ-LAI, 500 m)",
                            "LAI (m² m⁻²)", "YlGn", vmin=0, vmax=np.nanpercentile(lai, 99))
    ndvi, wind = z["cube"]
    draw("ndvi_mean", m(ndvi), lat, lon, "Mean NDVI 2015–2020 (MODIS MOD09A1, 500 m)", "NDVI", "YlGn",
         vmin=0, vmax=np.nanpercentile(m(ndvi), 99))
    draw("wind_mean", m(wind), lat, lon, "Mean 10 m wind speed 2015–2020 (BARRA-C2, daily mean)", "wind speed (m s⁻¹)",
         "PuBu", vmin=np.nanpercentile(m(wind), 1), vmax=np.nanpercentile(m(wind), 99))
    blat, blon = z["blat"], z["blon"]
    ri = np.clip(np.round((LAT0 - blat) / PX - 0.5).astype(int) // D, 0, land.shape[0] - 1)
    ci = np.clip(np.round((blon - LON0) / PX - 0.5).astype(int) // D, 0, land.shape[1] - 1)
    inside = (((LAT0 - blat) / PX >= 0) & ((LAT0 - blat) / PX < H))[:, None] & \
             (((blon - LON0) / PX >= 0) & ((blon - LON0) / PX < W))[None, :]
    bland = land[ri][:, ci] & inside
    for var, nm_, title, lab, cmap in (
            ("vpd", "vpd_mean", "Mean VPD at the daily maximum temperature 2015–2020 (BARRA-C2, 4.4 km)", "VPD (kPa)", "magma"),
            ("tasmax", "tmax_mean", "Mean daily maximum air temperature 2015–2020 (BARRA-C2, 4.4 km)", "temperature (°C)", "inferno")):
        f = np.where(bland, z[var], np.nan)
        draw(nm_, f, blat, blon, title, lab, cmap, vmin=np.nanpercentile(f, 1), vmax=np.nanpercentile(f, 99))
    # land mask card
    lmk = np.where(land, 1.0, np.nan)
    fig = plt.figure(figsize=(11.24, 9.50), dpi=100); ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(lmk, extent=_extent(lat, lon), origin="upper", cmap=ListedColormap(["black"]), interpolation="nearest")
    _decor(ax, False); ax.set_aspect("auto"); ax.tick_params(length=0, labelbottom=False, labelleft=False)
    buf = io.BytesIO(); fig.savefig(buf, dpi=100); plt.close(fig); buf.seek(0)
    Image.open(buf).convert("RGB").save(os.path.join(CARD_DIR, "card_landmask.png"))
    print("wrote landmask card")


if __name__ == "__main__":
    main()
