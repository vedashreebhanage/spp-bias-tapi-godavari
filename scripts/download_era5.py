r"""
Download ERA5-Land daily total precipitation, 2015-2023, Tapi bbox.

Prerequisites
-------------
1. Register at https://cds.climate.copernicus.eu/
2. Open the dataset page for "ERA5-Land daily statistics" and ACCEPT THE LICENCE.
   Without this the API returns 403 no matter what your key is.
3. Create the file  C:/Users/<you>/.cdsapirc  containing:

       url: https://cds.climate.copernicus.eu/api
       key: <your-token>

4. pip install cdsapi

Then:  python download_era5.py

IMPORTANT
---------
This requests the *derived daily statistics* product so that Copernicus performs
the hourly -> daily accumulation server-side. Doing it yourself is the single most
common source of ERA5 precipitation errors.

Units are METRES. Convert to mm (x1000) during processing, not here -- keep raw
files raw.

Accumulation is in UTC. IMD's rain day runs 0830 IST to 0830 IST (UTC+05:30).
Decide how you handle this offset and document it.

If the request is rejected, open the dataset page, configure the same selection in
the web form, click "Show API request", and reconcile the keys below with what CDS
prints. The web form is always authoritative.
"""

import sys
from pathlib import Path

OUT_DIR = Path("data/raw/era5")
DATASET = "derived-era5-land-daily-statistics"

# [North, West, South, East]
AREA = [23.00, 72.25, 19.75, 79.00]

YEARS = [str(y) for y in range(2015, 2024)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]


def main() -> None:
    try:
        import cdsapi
    except ImportError:
        sys.exit("cdsapi not installed.  Run:  pip install cdsapi")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    for year in YEARS:
        target = OUT_DIR / f"era5_tapi_{year}.nc"
        if target.exists():
            print(f"{year}: already present, skipping")
            continue

        print(f"{year}: requesting...")
        request = {
            "variable": ["total_precipitation"],
            "year": year,
            "month": MONTHS,
            "day": DAYS,
            "daily_statistic": "daily_sum",
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "area": AREA,
            "data_format": "netcdf",
        }
        try:
            client.retrieve(DATASET, request, str(target))
            print(f"{year}: saved -> {target}  "
                  f"({target.stat().st_size / 1e6:.1f} MB)")
        except Exception as exc:
            print(f"{year}: FAILED -- {exc}")
            print("  Check: licence accepted? .cdsapirc correct? request keys "
                  "still current (use 'Show API request' on the CDS web form)?")
            break


if __name__ == "__main__":
    main()
