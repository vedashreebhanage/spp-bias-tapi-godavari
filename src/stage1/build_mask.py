"""
STAGE 1 - Step 1: Basin mask + terrain attributes on the IMD 0.25 deg grid.

What this produces (in data/processed/):
    tapi_mask.npz    - boolean mask (lat, lon) of in-basin cells, plus the grid
                       coordinates, cell elevations, ruggedness, and zone labels
    mask_map.png     - visual check: basin polygon over the masked cells
    elevation_map.png- visual check: cell mean elevation + zone boundaries

Why this file exists:
    The original pipeline computed every metric over a rectangular bounding box
    (~165,000 km2) instead of the Tapi basin (65,145 km2), because the shapefile
    was downloaded but never used. Roughly 60% of every reported statistic came
    from outside the basin. This file is the fix, and it validates itself:
    it prints the implied basin area and refuses to proceed if that area is
    wildly off the known value.

Method choices (stated so they can be reviewed):
    * A cell is "in basin" if its CENTER lies inside the basin polygon.
      Simple, unambiguous, standard for 0.25 deg cells over a 65,000 km2 basin.
      (Fractional-coverage masking matters for small basins; not needed here.)
    * Elevation per cell = mean of all SRTM 90 m pixels inside the cell.
      Ruggedness per cell = standard deviation of those pixels.
    * Zones = terciles of in-basin cell elevations (equal thirds by cell count),
      labelled lower / middle / upper. Terciles avoid hand-picked thresholds;
      the exact cut values are printed and saved for the methods section.

Run from the project root:
    python src/stage1/build_mask.py
"""

import sys
from pathlib import Path

import numpy as np

try:
    import geopandas as gpd
    import rasterio
    from shapely.geometry import Point
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    sys.exit(f"Missing dependency: {e}. Run: pip install geopandas rasterio matplotlib shapely")

# ---------------------------------------------------------------- constants --
SHAPEFILE = Path("data/raw/shapefiles/tapi_hydrobasins.shp")
DEM_FILE = Path("data/raw/dem/tapi_dem_srtm.tif")
OUT_DIR = Path("data/processed")

# The shapefile contains 14 HydroBASINS polygons covering much of western/central
# India (combined bbox ~10-28 N). Only ONE of them is the Tapi. Selecting by ID
# rather than dissolving everything is essential -- the union of all 14 would
# produce a "basin" of ~800,000 km2. Identified in docs/DATASETS.md:
#   HYBAS_ID 4050031520, PFAF_ID 45404, SUB_AREA 64,337.5 km2
TAPI_HYBAS_ID = 4050031520

KNOWN_AREA_KM2 = 65_145          # CWC figure for the Tapi basin
AREA_TOLERANCE = 0.25            # accept +/- 25% (cell-center masking is coarse)

# IMD grid definition (must match the IMD reader exactly)
IMD_LAT = 6.5 + 0.25 * np.arange(129)     # 6.5 .. 38.5
IMD_LON = 66.5 + 0.25 * np.arange(135)    # 66.5 .. 100.0


# ------------------------------------------------------------------ helpers --
def cell_area_km2(lat_deg: float, dres: float = 0.25) -> float:
    """Area of a dres x dres cell centred at lat_deg, in km2."""
    r = 6371.0
    dlat = np.deg2rad(dres)
    dlon = np.deg2rad(dres)
    return (r ** 2) * dlat * dlon * np.cos(np.deg2rad(lat_deg))


def load_basin_polygon():
    gdf = gpd.read_file(SHAPEFILE)
    print(f"Shapefile: {len(gdf)} feature(s), CRS = {gdf.crs}")
    if gdf.crs is None:
        print("  WARNING: no CRS in shapefile -- assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        print(f"  Reprojecting {gdf.crs} -> EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")

    # select the Tapi polygon by HydroBASINS ID -- never dissolve all records
    if "HYBAS_ID" not in gdf.columns:
        sys.exit(f"HYBAS_ID column missing; found {list(gdf.columns)}. "
                 "Wrong shapefile?")
    sel = gdf[gdf["HYBAS_ID"].astype("int64") == TAPI_HYBAS_ID]
    if len(sel) != 1:
        sys.exit(f"Expected exactly 1 feature with HYBAS_ID={TAPI_HYBAS_ID}, "
                 f"found {len(sel)}. IDs present: "
                 f"{sorted(gdf['HYBAS_ID'].astype('int64').tolist())}")
    row = sel.iloc[0]
    polygon = row.geometry
    print(f"Selected HYBAS_ID {TAPI_HYBAS_ID} | PFAF_ID {row.get('PFAF_ID', '?')} "
          f"| SUB_AREA attribute: {float(row.get('SUB_AREA', float('nan'))):,.1f} km2")
    print(f"Polygon bbox: lon {polygon.bounds[0]:.3f}..{polygon.bounds[2]:.3f}, "
          f"lat {polygon.bounds[1]:.3f}..{polygon.bounds[3]:.3f}")

    # cross-check the geometry's actual area in an equal-area projection
    poly_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    shp_area = float(poly_gdf.to_crs("ESRI:54034").area.iloc[0]) / 1e6
    print(f"Geometry area (equal-area proj): {shp_area:,.0f} km2 "
          f"(CWC reference: {KNOWN_AREA_KM2:,} km2)")
    if abs(shp_area - KNOWN_AREA_KM2) / KNOWN_AREA_KM2 > 0.10:
        sys.exit("FATAL: selected polygon's area differs >10% from the CWC "
                 "figure. Wrong polygon selected -- check TAPI_HYBAS_ID.")
    return polygon


def build_mask(polygon):
    mask = np.zeros((len(IMD_LAT), len(IMD_LON)), dtype=bool)
    for i, la in enumerate(IMD_LAT):
        # cheap pre-filter: skip latitudes far outside the polygon bounds
        if la < polygon.bounds[1] - 0.25 or la > polygon.bounds[3] + 0.25:
            continue
        for j, lo in enumerate(IMD_LON):
            if lo < polygon.bounds[0] - 0.25 or lo > polygon.bounds[2] + 0.25:
                continue
            if polygon.contains(Point(lo, la)):
                mask[i, j] = True
    return mask


def terrain_per_cell(mask):
    """Mean elevation and ruggedness (std) of DEM pixels within each masked cell."""
    elev = np.full(mask.shape, np.nan)
    rugg = np.full(mask.shape, np.nan)
    with rasterio.open(DEM_FILE) as src:
        for i, la in enumerate(IMD_LAT):
            for j, lo in enumerate(IMD_LON):
                if not mask[i, j]:
                    continue
                win = rasterio.windows.from_bounds(
                    lo - 0.125, la - 0.125, lo + 0.125, la + 0.125,
                    transform=src.transform)
                block = src.read(1, window=win, masked=True).astype("float64")
                block = block.filled(np.nan)
                block[block < -100] = np.nan          # SRTM voids
                if np.isfinite(block).sum() < 100:
                    print(f"  WARNING: cell ({la:.2f},{lo:.2f}) has "
                          f"<100 valid DEM pixels")
                    continue
                elev[i, j] = np.nanmean(block)
                rugg[i, j] = np.nanstd(block)
    return elev, rugg


def make_zones(mask, elev):
    """Terciles of in-basin elevation -> 1=lower, 2=middle, 3=upper, 0=outside."""
    vals = elev[mask]
    t1, t2 = np.nanpercentile(vals, [33.333, 66.667])
    zones = np.zeros(mask.shape, dtype=np.int8)
    zones[mask & (elev <= t1)] = 1
    zones[mask & (elev > t1) & (elev <= t2)] = 2
    zones[mask & (elev > t2)] = 3
    print(f"\nZone cut points (elevation terciles): {t1:.0f} m, {t2:.0f} m")
    for z, name in [(1, "lower"), (2, "middle"), (3, "upper")]:
        n = int((zones == z).sum())
        zvals = elev[zones == z]
        print(f"  {name:<7} {n:3d} cells   elev {np.nanmin(zvals):5.0f}"
              f"-{np.nanmax(zvals):5.0f} m")
    return zones, (t1, t2)


def plot_maps(polygon, mask, elev, zones):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ii, jj = np.where(mask)
    la, lo = IMD_LAT[ii], IMD_LON[jj]

    fig, ax = plt.subplots(figsize=(10, 6))
    gpd.GeoSeries([polygon]).plot(ax=ax, facecolor="none", edgecolor="crimson", lw=2)
    ax.scatter(lo, la, s=120, marker="s", c="steelblue", alpha=0.6,
               label=f"in-basin cells (n={mask.sum()})")
    ax.set_title("Tapi basin polygon vs masked 0.25\N{DEGREE SIGN} cells")
    ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.legend()
    fig.savefig(OUT_DIR / "mask_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(lo, la, s=120, marker="s", c=elev[mask], cmap="terrain")
    gpd.GeoSeries([polygon]).plot(ax=ax, facecolor="none", edgecolor="crimson", lw=1)
    plt.colorbar(sc, ax=ax, label="mean elevation (m)")
    ax.set_title("Cell mean elevation (SRTM 90 m aggregated)")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.savefig(OUT_DIR / "elevation_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nMaps written: {OUT_DIR/'mask_map.png'}, {OUT_DIR/'elevation_map.png'}")


# --------------------------------------------------------------------- main --
def main():
    for f in (SHAPEFILE, DEM_FILE):
        if not f.exists():
            sys.exit(f"{f} not found. Run from the project root.")

    polygon = load_basin_polygon()

    print("\nBuilding cell-center mask on the IMD grid...")
    mask = build_mask(polygon)
    n = int(mask.sum())
    ii = np.where(mask)[0]
    implied = float(sum(cell_area_km2(IMD_LAT[i]) for i in ii))
    print(f"  in-basin cells : {n}")
    print(f"  implied area   : {implied:,.0f} km2")
    print(f"  CWC reference  : {KNOWN_AREA_KM2:,} km2  "
          f"(diff {100*(implied-KNOWN_AREA_KM2)/KNOWN_AREA_KM2:+.1f}%)")

    if abs(implied - KNOWN_AREA_KM2) / KNOWN_AREA_KM2 > AREA_TOLERANCE:
        sys.exit("\nFATAL: implied area is outside tolerance. The mask is wrong "
                 "(bad shapefile, CRS problem, or grid definition error). "
                 "Do NOT proceed to metrics with this mask.")

    print("\nAggregating DEM to cell means (a minute or two)...")
    elev, rugg = terrain_per_cell(mask)
    print(f"  elevation range : {np.nanmin(elev[mask]):.0f} to "
          f"{np.nanmax(elev[mask]):.0f} m  (mean {np.nanmean(elev[mask]):.0f} m)")

    zones, cuts = make_zones(mask, elev)
    plot_maps(polygon, mask, elev, zones)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "tapi_mask.npz",
        mask=mask, lat=IMD_LAT, lon=IMD_LON,
        elevation=elev, ruggedness=rugg, zones=zones,
        zone_cuts=np.array(cuts),
        implied_area_km2=implied, n_cells=n)
    print(f"\nSaved {OUT_DIR/'tapi_mask.npz'}")
    print("\nNOW LOOK AT THE TWO PNG MAPS before going further.")
    print("The blue squares must fill the red outline -- no more, no less.")


if __name__ == "__main__":
    main()
