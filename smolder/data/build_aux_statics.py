"""Build extra static predictor rasters on the smips grid (3474x4110, EPSG:4326,
0.01 deg, origin lon 112.904998779 / lat -9.005).

Outputs float32 .npy on the exact model grid:
  aux_rasters/static_pop.npy         WorldPop 1km 2020 population count per pixel
  aux_rasters/static_dist_road.npy   distance (km) to nearest GRIP4 road
  aux_rasters/static_lightning.npy   LIS/OTD flash density  [PENDING source]

These are appended to each cube as new static arrays by add_aux_to_cubes.py and
fed to the models exactly like agb (broadcast over time). See CLAUDE.md.

Provenance:
  population   WorldPop Global 1km 2020, aus_ppp_2020_1km_Aggregated.tif
  roads        GRIP4 Region 6 (Oceania) vector, PBL dataportaal, 2018 (DOI 10.1038/s41893-018-0059-3)
  lightning    LIS/OTD HRFC V2.3.2015 -- to be added (open mirror; swap for the
               Earthdata GHRC copy DOI 10.5067/LIS/LIS-OTD/DATA302 for final cite)
"""
import os, numpy as np, zipfile, glob
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
import geopandas as gpd
from scipy.ndimage import distance_transform_edt

AUX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aux_rasters")

# authoritative grid (from cube landmask)
H, W = 3474, 4110
LON0, LAT0, PX = 112.904998779, -9.005000113999998, 0.01
# pixel-centre coordinate vectors
LONS = LON0 + (np.arange(W) + 0.5) * PX
LATS = LAT0 - (np.arange(H) + 0.5) * PX
# rasterio affine for the target grid (north-up)
from rasterio.transform import from_origin
DST_TF = from_origin(LON0, LAT0, PX, PX)


def build_population():
    src_path = os.path.join(AUX, "aus_pop_1km_2020.tif")
    out = os.path.join(AUX, "static_pop.npy")
    with rasterio.open(src_path) as src:
        dst = np.zeros((H, W), np.float32)
        from rasterio.warp import reproject
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=DST_TF, dst_crs="EPSG:4326",
            resampling=Resampling.average,   # 1km->1km, average preserves density
        )
    dst = np.nan_to_num(dst, nan=0.0)
    dst[dst < 0] = 0.0                       # WorldPop nodata is a large negative
    np.save(out, dst)
    print(f"[pop] wrote {out}  range [{dst.min():.2f}, {dst.max():.1f}] "
          f"nonzero {100*(dst>0).mean():.1f}%", flush=True)


def build_dist_road():
    out = os.path.join(AUX, "static_dist_road.npy")
    # GRIP4 Region 7 = Oceania/Australia (verified: bounds lat -49.4 to +15.3).
    # Region 6 is Asia and Region 5 is the northern hemisphere -- neither contains
    # mainland Australia, despite the "Region6=Oceania" guess. Verified by latitude.
    shp = os.path.join(AUX, "grip7", "GRIP4_region7.shp")
    # only the geometry inside our bbox -- GRIP region 6 covers all of Oceania/SE-Asia.
    # NOTE: geopandas.read_file(bbox=...) over /vsizip/ silently under-matched here
    # (returned 11875 of 5.24M features). Push the bbox filter to GDAL via fiona and
    # rasterize while streaming, so we never hold the whole network in memory.
    import fiona
    from shapely.geometry import shape
    bbox = (LON0, LAT0 - H * PX, LON0 + W * PX, LAT0)   # (minx, miny, maxx, maxy)
    print("[road] streaming GRIP4 region6 within AUS bbox via fiona ...", flush=True)
    shapes = []
    n = 0
    with fiona.open(shp) as src:
        for feat in src.filter(bbox=bbox):
            g = feat["geometry"]
            if g is None:
                continue
            shapes.append((shape(g), 1))
            n += 1
            if n % 200000 == 0:
                print(f"[road]   ...{n} segments read", flush=True)
    print(f"[road] {n} road segments in bbox", flush=True)
    # burn roads onto the grid: 1 where a road passes, 0 elsewhere
    road_mask = rasterize(shapes, out_shape=(H, W), transform=DST_TF,
                          fill=0, default_value=1, dtype="uint8")
    print(f"[road] rasterized, road pixels {int(road_mask.sum())}", flush=True)
    # euclidean distance transform (in pixels) from nearest road, then -> km
    # sampling accounts for lat/lon->km: dy ~ 1.113 km/px, dx ~ cos(lat) scaled.
    # use a mid-latitude constant (-26 deg) so distances are km, not degrees.
    km_per_px_y = 111.32 * PX
    km_per_px_x = 111.32 * PX * np.cos(np.deg2rad(26.0))
    dist_px = distance_transform_edt(road_mask == 0,
                                     sampling=[km_per_px_y, km_per_px_x])
    dist_km = dist_px.astype(np.float32)
    np.save(out, dist_km)
    print(f"[road] wrote {out}  range [{dist_km.min():.2f}, {dist_km.max():.1f}] km", flush=True)


def build_lightning():
    """LIS/OTD 0.5deg High Resolution Full Climatology (HRFC), combined flash rate
    (hrfc_com_fr, flashes/km^2/yr), fetched live via APDRC's open OPeNDAP mirror
    (no Earthdata login needed): apdrc.soest.hawaii.edu ... LIS-OTD-lightning-data
    /High-Resolution-Full-Climatology. Official product: LIS/OTD Science Team,
    DOI 10.5067/LIS/LIS-OTD/DATA302 -- swap in the Earthdata copy for final citation
    if you want the canonical file rather than this (identical-content) mirror.
    """
    import xarray as xr
    out = os.path.join(AUX, "static_lightning.npy")
    url = ("http://apdrc.soest.hawaii.edu:80/dods/public_data/satellite_product/"
           "LIS-OTD-lightning-data/High-Resolution-Full-Climatology")
    print("[lightning] opening APDRC OPeNDAP ...", flush=True)
    ds = xr.open_dataset(url)
    fr = ds["hrfc_com_fr"].isel(time=0, lev=0).values.astype(np.float32)  # (360,720), 0.5deg
    fr = np.nan_to_num(fr, nan=0.0)
    lat05 = ds["lat"].values; lon05 = ds["lon"].values   # lat ascending, lon -180..180
    # HRFC lon is -180..180; our grid is 0..360-style positive-east (112.9..154.0), same thing
    lon05_pos = np.where(lon05 < 0, lon05 + 360, lon05)
    order = np.argsort(lon05_pos); lon05_pos = lon05_pos[order]; fr = fr[:, order]
    # nearest-neighbour resample 0.5deg -> 0.01deg smips grid (fine enough for a
    # climatology this coarse; bilinear would imply false precision)
    lat_idx = np.clip(np.searchsorted(lat05, LATS) - 0, 0, len(lat05) - 1)
    # searchsorted needs ascending; LATS is descending (north->south) so handle directly
    lat_idx = np.array([np.abs(lat05 - la).argmin() for la in LATS])
    lon_idx = np.array([np.abs(lon05_pos - lo).argmin() for lo in LONS])
    grid = fr[np.ix_(lat_idx, lon_idx)]
    np.save(out, grid.astype(np.float32))
    print(f"[lightning] wrote {out}  range [{grid.min():.2f}, {grid.max():.1f}] flashes/km2/yr", flush=True)


def build_elevation():
    """Elevation (m), ETOPO1 1 arc-minute (~1.85km, coarser than the 0.01deg/~1.1km
    target grid -- bilinear-interpolated, not nearest-neighbour, to avoid blocky
    upsampling artifacts). Fetched live from NOAA's ERDDAP griddap endpoint
    (coastwatch.pfeg.noaa.gov, public, no auth), bbox-subset to Australia.
    Includes bathymetry (negative) over ocean -- masked to 0 there at use time
    via the landmask, same convention as agb.
    """
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator
    out = os.path.join(AUX, "static_elevation.npy")
    nc_path = os.path.join(AUX, "etopo1_aus.nc")
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"{nc_path} missing -- fetch via ERDDAP griddap first")
    ds = xr.open_dataset(nc_path)
    alt = ds["altitude"].values.astype(np.float64)   # (lat, lon), lat ascending
    lat_src = ds["latitude"].values.astype(np.float64)
    lon_src = ds["longitude"].values.astype(np.float64)
    interp = RegularGridInterpolator((lat_src, lon_src), alt, method="linear",
                                      bounds_error=False, fill_value=None)
    # query at target grid pixel centres; LATS is descending, interpolator doesn't care
    grid_lat, grid_lon = np.meshgrid(LATS, LONS, indexing="ij")
    pts = np.stack([grid_lat.ravel(), grid_lon.ravel()], axis=-1)
    elev = interp(pts).reshape(H, W).astype(np.float32)
    np.save(out, elev)
    print(f"[elevation] wrote {out}  range [{elev.min():.0f}, {elev.max():.0f}] m", flush=True)


def build_slope_aspect():
    """Slope (degrees) and aspect (circular, stored as sin/cos of compass
    bearing) derived from static_elevation.npy -- no new external source, just
    a gradient on the DEM we already fetched for elevation. Both are
    established fire-behaviour drivers: fire spreads faster upslope, and in the
    southern hemisphere north-facing slopes get more sun -> drier fuel.
    Gradients are converted from per-pixel-index to per-metre using the same
    lat-dependent km/px scaling used for the road distance transform (dx
    shrinks with cos(lat) toward the poles; dy is constant).
    """
    elev_path = os.path.join(AUX, "static_elevation.npy")
    if not os.path.exists(elev_path):
        raise FileNotFoundError(f"{elev_path} missing -- run build_elevation first")
    Z = np.load(elev_path).astype(np.float64)

    dy_m = 111320.0 * PX                                  # metres per row-step (constant)
    dx_m = 111320.0 * PX * np.cos(np.deg2rad(LATS))[:, None]  # metres per col-step, varies by row (lat)

    gy = np.gradient(Z, axis=0)   # dZ per row-step (row increases southward)
    gx = np.gradient(Z, axis=1)   # dZ per col-step (col increases eastward)
    dz_dsouth = gy / dy_m
    dz_deast = gx / dx_m

    slope_deg = np.degrees(np.arctan(np.sqrt(dz_dsouth ** 2 + dz_deast ** 2))).astype(np.float32)
    # downslope bearing (compass, 0=N/90=E/180=S/270=W): north_comp=-dz_dsouth, east_comp=-dz_deast
    bearing = (np.degrees(np.arctan2(-dz_deast, -dz_dsouth)) + 360.0) % 360.0
    aspect_sin = np.sin(np.deg2rad(bearing)).astype(np.float32)
    aspect_cos = np.cos(np.deg2rad(bearing)).astype(np.float32)

    np.save(os.path.join(AUX, "static_slope.npy"), slope_deg)
    np.save(os.path.join(AUX, "static_aspect_sin.npy"), aspect_sin)
    np.save(os.path.join(AUX, "static_aspect_cos.npy"), aspect_cos)
    print(f"[slope]  wrote static_slope.npy  range [{slope_deg.min():.1f}, {slope_deg.max():.1f}] deg, "
          f"land mean ~{np.nanmean(slope_deg):.1f} deg", flush=True)
    print(f"[aspect] wrote static_aspect_{{sin,cos}}.npy", flush=True)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["pop", "road"]
    if "pop" in which:
        build_population()
    if "road" in which:
        build_dist_road()
    if "lightning" in which:
        build_lightning()
    if "elevation" in which:
        build_elevation()
    if "slope" in which:
        build_slope_aspect()
    print("[done]", flush=True)
