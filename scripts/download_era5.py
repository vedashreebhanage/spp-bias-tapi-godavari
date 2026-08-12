"""
ERA5-Land daily total precipitation, 2015-2023.

Why this script does NOT use the "daily statistics" dataset
------------------------------------------------------------
CDS rejects it:

    Daily statistics of accumulated variables are not supported for this dataset,
    skipping: total_precipitation

Accumulated variables (precipitation, radiation, fluxes) are excluded from the derived
daily-statistics product. So we go to the source dataset instead.

Why requesting only 00:00 gives daily totals
--------------------------------------------
ERA5-Land accumulates precipitation from the beginning of each day. The value stamped
00:00 UTC on day D+1 is the complete accumulation for day D. Therefore:

    daily total for day D  ==  tp at 00:00 UTC on day D+1

This means:
  * we request ONE timestep per day, not 24  (24x smaller download)
  * NO summing is required -- and summing hourly values would be badly wrong, since
    they are cumulative, not per-hour. That mistake inflates totals several-fold and
    is the most common ERA5 precipitation error.

Consequence for file contents
-----------------------------
The file for year Y contains timestamps 00:00 on each day of Y, which represent
rainfall on the PRECEDING day. So year Y's file covers 31 Dec (Y-1) through 30 Dec (Y).
At analysis time you must shift the time axis back by one day:

    da = da.assign_coords(time=da.time - np.timedelta64(1, "D"))

The script also fetches 2024-01-01 so that 31 Dec 2023 is covered.

Time-zone caveat
----------------
These are 00-24 UTC days. IMD's rain day runs 0830 IST to 0830 IST (UTC+05:30), so the
two windows are offset by 3 hours. Decide how you handle that and state it in your
methods -- do not leave it implicit.

Units
-----
METRES. Multiply by 1000 for mm at analysis time; keep raw files raw.

Setup
-----
  1. https://cds.climate.copernicus.eu/  -> register
  2. Open the "ERA5-Land hourly data from 1950 to present" dataset page and
     ACCEPT THE LICENCE (separate from registering).
  3. C:/Users/<you>/.cdsapirc:
         url: https://cds.climate.copernicus.eu/api
         key: <personal access token>
  4. pip install cdsapi

Usage
-----
  python scripts/download_era5.py --clean
  python scripts/download_era5.py --basin godavari
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from basins import YEARS, cds_area  # noqa: E402

DATASET = "reanalysis-era5-land"
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]


def clean_partials(out_dir: Path) -> None:
    junk = [p for p in out_dir.glob("*.nc") if "_h_" in p.name]
    for p in junk:
        p.unlink()
        print(f"  removed stale hourly file: {p.name}")
    print("  no stale hourly files found" if not junk else "")


def request_for(year: int, months, days, area) -> dict:
    return {
        "variable": ["total_precipitation"],
        "year": str(year),
        "month": months,
        "day": days,
        "time": ["00:00"],          # <- the whole trick; see module docstring
        "area": area,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", default="tapi")
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    try:
        import cdsapi
    except ImportError:
        sys.exit("Run:  pip install cdsapi")

    out_dir = Path("data/raw/era5")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        clean_partials(out_dir)

    area = cds_area(args.basin)
    print(f"{args.basin.upper()}  area [N,W,S,E] = {area}")
    print(f"Dataset: {DATASET}   one timestep per day (00:00 UTC)")
    print("Reminder: each 00:00 value is the total for the PREVIOUS day.\n")

    client = cdsapi.Client()
    failed = []

    # main years
    for year in YEARS:
        target = out_dir / f"era5_{args.basin}_{year}.nc"
        if target.exists() and target.stat().st_size > 0:
            print(f"{year}: present, skipping")
            continue

        print(f"{year}: requesting (CDS queues jobs; this can take minutes)...")
        try:
            client.retrieve(DATASET, request_for(year, MONTHS, DAYS, area), str(target))
            print(f"{year}: saved  ({target.stat().st_size / 1e6:.2f} MB)")
        except Exception as exc:
            failed.append(year)
            target.unlink(missing_ok=True)
            print(f"{year}: FAILED -- {exc}")
            print("  If 403: accept the licence on the ERA5-Land hourly dataset page.")
            if len(failed) >= 2:
                sys.exit("Two failures -- fix the cause before retrying.")

    # one extra day so that 31 Dec of the final year is covered
    tail_year = max(YEARS) + 1
    tail = out_dir / f"era5_{args.basin}_{tail_year}0101.nc"
    if not tail.exists():
        print(f"\n{tail_year}-01-01: fetching (covers 31 Dec {max(YEARS)})...")
        try:
            client.retrieve(DATASET, request_for(tail_year, ["01"], ["01"], area),
                            str(tail))
            print("  saved")
        except Exception as exc:
            print(f"  FAILED -- {exc}")
            print(f"  Without it, 31 Dec {max(YEARS)} will be missing.")

    print(f"\nDone. failed years: {failed if failed else 'none'}")
    print("Re-run to retry failures (existing files are skipped).")
    print("\nAt analysis time, shift the time axis back one day:")
    print("    da = da.assign_coords(time=da.time - np.timedelta64(1, 'D'))")


if __name__ == "__main__":
    main()
