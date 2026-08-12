"""
Shared study-area definitions. Every download and analysis script imports from here
so the bounding boxes can never drift apart between products.

Boxes are deliberately a little larger than each basin polygon: edge grid cells need
full coverage before they can be averaged or clipped to the HydroBASINS outline.
"""

import json
from pathlib import Path

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

# ---------------------------------------------------------------------------
# Extreme rainfall episodes
#
# There is deliberately NO hardcoded event list here any more. The previous
# FLOOD_EVENTS dict held placeholder windows marked UNVERIFIED; checking them
# against IMD showed the 2021 window (Aug 1-20) was almost entirely dry, so an
# "event" metric computed over it would mostly have measured dry-day behaviour.
#
# Episodes are now derived from the reference dataset by a rule fixed in
# src/stage1/define_episodes.py and cached to data/processed/rainfall_episodes.json.
# They are extreme *rainfall* episodes, not stage-defined floods -- see the
# module docstring there for why that distinction is kept.
# ---------------------------------------------------------------------------

EPISODES_FILE = "data/processed/rainfall_episodes.json"


def load_episodes(basin: str = "tapi") -> list:
    """Return the derived episode list. Fails loudly rather than defaulting.

    Raises FileNotFoundError if the episodes have not been generated, so that no
    analysis can silently run on an assumed or stale event set.
    """
    if basin != "tapi":
        raise NotImplementedError(
            f"Episodes are only defined for the Tapi basin so far, not '{basin}'. "
            "Godavari is a replication basin: derive its episodes separately and "
            "never pool the two."
        )
    path = Path(EPISODES_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"{EPISODES_FILE} not found. Run: python src/stage1/define_episodes.py"
        )
    return json.loads(path.read_text())["episodes"]


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
