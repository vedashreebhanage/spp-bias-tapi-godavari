"""
Shared study-area definitions. Every download and analysis script imports from here
so the bounding boxes can never drift apart between products.

Boxes are deliberately a little larger than each basin polygon: edge grid cells need
full coverage before they can be averaged or clipped to the HydroBASINS outline.
"""

BASINS = {
    "tapi": dict(
        south=19.75, north=23.00, west=72.25, east=79.00,
        area_km2=65_145,
        shapefile="data/raw/shapefiles/tapi_hydrobasins.shp",
    ),
    "godavari": dict(
        south=16.00, north=23.00, west=72.50, east=83.50,
        area_km2=312_812,
        shapefile="data/raw/shapefiles/godavari_hydrobasins.shp",
    ),
}

YEARS = list(range(2015, 2024))

# Flood windows. PLACEHOLDER DATES -- replace with CWC/IMD-verified periods and record
# the source for each before computing anything that depends on them.
FLOOD_EVENTS = {
    "tapi": [
        ("2021-08-01", "2021-08-20", "UNVERIFIED - needs CWC HFL citation"),
        ("2023-09-14", "2023-09-20", "UNVERIFIED - needs CWC HFL citation"),
    ],
    "godavari": [
        ("2022-07-10", "2022-07-20", "UNVERIFIED - needs CWC HFL citation"),
    ],
}


def bbox(basin: str) -> dict:
    if basin not in BASINS:
        raise KeyError(f"Unknown basin '{basin}'. Options: {sorted(BASINS)}")
    return BASINS[basin]


def cds_area(basin: str) -> list:
    """Copernicus wants [North, West, South, East]."""
    b = bbox(basin)
    return [b["north"], b["west"], b["south"], b["east"]]


def gesdisc_region(basin: str) -> str:
    """GES DISC subsetter wants 'W, S, E, N'."""
    b = bbox(basin)
    return f"{b['west']}, {b['south']}, {b['east']}, {b['north']}"
