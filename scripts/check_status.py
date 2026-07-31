"""
One-command inventory: what data is present, what is missing, what is suspect.

  python scripts/check_status.py
  python scripts/check_status.py --basin godavari

Run from the project root. Downloads nothing.
"""

import argparse
import calendar
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from basins import BASINS, FLOOD_EVENTS, YEARS  # noqa: E402

GREEN, YELLOW, RED, RESET = "\033[92m", "\033[93m", "\033[91m", "\033[0m"


def mark(state: str) -> str:
    return {"ok": f"{GREEN}OK{RESET}",
            "warn": f"{YELLOW}CHECK{RESET}",
            "bad": f"{RED}MISSING{RESET}"}[state]


def line(label: str, state: str, detail: str = "") -> None:
    print(f"  {label:<28} {mark(state):<20} {detail}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", default="tapi")
    args = ap.parse_args()
    basin = args.basin

    print(f"\n=== DATA STATUS: {basin.upper()} ===\n")

    # IMD
    imd = Path("data/raw/imd")
    present = [y for y in YEARS if (imd / f"RF{y}.GRD").exists()]
    bad_size = []
    for y in present:
        expect = (366 if calendar.isleap(y) else 365) * 129 * 135 * 4
        if (imd / f"RF{y}.GRD").stat().st_size != expect:
            bad_size.append(y)
    if len(present) == len(YEARS) and not bad_size:
        line("IMD (reference)", "ok", f"{len(present)}/{len(YEARS)} years, sizes verified")
    elif bad_size:
        line("IMD (reference)", "warn", f"wrong size: {bad_size}")
    else:
        line("IMD (reference)", "bad", f"{len(present)}/{len(YEARS)} years")

    # CHIRPS
    chirps = sorted(Path("data/raw/chirps").glob("chirps_*.nc")) \
        if Path("data/raw/chirps").exists() else []
    if len(chirps) >= len(YEARS):
        line("CHIRPS", "warn", f"{len(chirps)} files -- run verify_chirps.py")
    elif chirps:
        line("CHIRPS", "warn", f"only {len(chirps)}/{len(YEARS)} years")
    else:
        line("CHIRPS", "bad", "none")

    # ERA5
    era5_dir = Path("data/raw/era5")
    daily = sorted(era5_dir.glob(f"era5_{basin}_*.nc")) if era5_dir.exists() else []
    hourly = sorted(era5_dir.glob("*_h_*.nc")) if era5_dir.exists() else []
    if len(daily) >= len(YEARS):
        line("ERA5-Land (daily)", "ok", f"{len(daily)}/{len(YEARS)} years")
    elif daily:
        line("ERA5-Land (daily)", "warn", f"{len(daily)}/{len(YEARS)} years")
    else:
        line("ERA5-Land (daily)", "bad",
             f"none ({len(hourly)} stale hourly files)" if hourly else "none")
    if hourly:
        line("  -> stale hourly files", "warn",
             "run download_era5.py --clean to remove")

    # IMERG
    imerg_dir = Path("data/raw/imerg")
    granules = sorted(imerg_dir.glob("*.nc4")) if imerg_dir.exists() else []
    urls = imerg_dir / "imerg_urls.txt"
    if len(granules) > 300:
        line("IMERG V07", "ok", f"{len(granules)} granules")
    elif granules:
        line("IMERG V07", "warn", f"only {len(granules)} granules")
    else:
        line("IMERG V07", "bad",
             "url list ready" if urls.exists() else "run --show-settings first")

    # DEM
    dem = Path(f"data/raw/dem/{basin}_dem_srtm.tif")
    if dem.exists():
        try:
            import rasterio
            with rasterio.open(dem) as src:
                src.read(1)
            line("DEM", "ok", f"{dem.stat().st_size / 1e6:.1f} MB, reads cleanly")
        except ImportError:
            line("DEM", "warn", "present, rasterio not installed to verify")
        except Exception:
            line("DEM", "bad", "TRUNCATED -- re-run download_dem.py --tiles 4")
    else:
        line("DEM", "bad", "not downloaded")

    # Shapefile
    shp = Path(BASINS[basin]["shapefile"])
    if shp.exists():
        missing_side = [e for e in (".shx", ".dbf", ".prj")
                        if not shp.with_suffix(e).exists()]
        line("Basin shapefile", "warn" if missing_side else "ok",
             f"missing sidecars: {missing_side}" if missing_side else "complete")
    else:
        line("Basin shapefile", "bad", str(shp))

    # Flood events
    events = FLOOD_EVENTS.get(basin, [])
    unverified = [e for e in events if "UNVERIFIED" in e[2]]
    if not events:
        line("Flood event windows", "bad", "none defined in basins.py")
    elif unverified:
        line("Flood event windows", "warn",
             f"{len(unverified)}/{len(events)} lack a citation")
    else:
        line("Flood event windows", "ok", f"{len(events)} events, all cited")

    print("\nNext actions in order:")
    print("  1. python scripts/verify_chirps.py")
    print("  2. python scripts/download_dem.py            (--tiles 4 if it fails)")
    print("  3. python scripts/download_era5.py --clean")
    print("  4. python scripts/download_imerg.py --show-settings --events-only")
    print("  5. cite the flood windows in basins.py before computing DPBA\n")


if __name__ == "__main__":
    main()
