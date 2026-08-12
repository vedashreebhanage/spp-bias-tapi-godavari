"""
Verify CHIRPS files before trusting them. Downloads nothing.

    python scripts/verify_chirps.py            (run from the project root)
    python scripts/verify_chirps.py --dir some/other/folder

Filename-agnostic: any .nc / .nc4 in the folder is matched to a year by looking for a
4-digit year in its name, so it works with chirps_tapi_2015.nc,
chirps-v2.0.2015.days_p05.nc, or anything else sensible.

Handles global files as well as pre-clipped ones. Global CHIRPS years are several GB,
so data is opened lazily and subset to the study window before anything is read into
memory -- loading a global year in full would exhaust RAM.

Checks: provenance, units, 0.05 deg spacing, coverage of the study window, complete
time axis, plausible values, plausible annual total over the Tapi box.
"""

import argparse
import calendar
import re
import sys
from pathlib import Path

try:
    import numpy as np
    import xarray as xr
except ImportError:
    sys.exit("Run:  pip install xarray netCDF4 numpy")

# Coverage requirement: the Tapi BASIN plus a half-cell margin -- NOT the generous
# download bbox in basins.py. Pre-clipped files (e.g. 72.5-78.5 E) fully cover the
# basin (~72.55-78.28 E, ~20.1-22.1 N) and must not fail verification for lacking
# margin they never needed. The download bbox stays generous for NEW downloads only.
LAT_MIN, LAT_MAX = 19.95, 22.25
LON_MIN, LON_MAX = 72.50, 78.40

YEARS = list(range(2015, 2024))


def find_files(folder: Path) -> dict:
    """Map year -> file path by scanning for a 4-digit year in each filename."""
    found = {}
    unmatched = []
    for p in sorted(list(folder.glob("*.nc")) + list(folder.glob("*.nc4"))):
        years = [int(y) for y in re.findall(r"(?<!\d)(19|20)\d{2}(?!\d)",
                                            p.name.replace("v2.0", ""))]
        years = [int(m.group()) for m in
                 re.finditer(r"(?<!\d)(?:19|20)\d{2}(?!\d)", p.name.replace("v2.0", ""))]
        cands = [y for y in years if y in YEARS]
        if len(cands) == 1:
            found.setdefault(cands[0], p)
        else:
            unmatched.append(p)
    return found, unmatched


def coord_names(ds) -> tuple:
    lat = next((c for c in ("latitude", "lat", "y") if c in ds.coords), None)
    lon = next((c for c in ("longitude", "lon", "x") if c in ds.coords), None)
    return lat, lon


def check(path: Path, year: int) -> bool:
    ok = True
    size_mb = path.stat().st_size / 1e6
    print(f"\n{year}  {path.name}  ({size_mb:.1f} MB)")

    ds = xr.open_dataset(path, chunks={})     # lazy: nothing read yet

    blob = " ".join(str(v) for v in ds.attrs.values()).lower() + " " + path.name.lower()
    if "chirps" in blob:
        print("  provenance           : CHIRPS identified")
    else:
        print("  provenance           : not identified  <-- verify the source")
        ok = False

    vars3d = [v for v in ds.data_vars if ds[v].ndim == 3]
    if not vars3d:
        print("  no 3-D variable  <-- FAIL")
        ds.close()
        return False
    da = ds[vars3d[0]]
    units = da.attrs.get("units", "(none)")
    print(f"  variable / units     : {vars3d[0]} / {units}")
    if "mm" not in str(units).lower():
        print("    ^ expected mm/day")
        ok = False

    latn, lonn = coord_names(ds)
    if latn is None or lonn is None:
        print(f"  cannot identify lat/lon among {list(ds.coords)}  <-- FAIL")
        ds.close()
        return False

    lat, lon = ds[latn].values, ds[lonn].values
    dlat, dlon = float(abs(np.diff(lat).mean())), float(abs(np.diff(lon).mean()))
    descending = lat[0] > lat[-1]
    span = (lat.max() - lat.min()) * (lon.max() - lon.min())
    scope = "GLOBAL" if span > 10000 else "regional subset"
    print(f"  coverage             : {scope}")
    print(f"  lat  {lat.min():.3f}..{lat.max():.3f}  step {dlat:.4f}"
          f"  ({'descending' if descending else 'ascending'})")
    print(f"  lon  {lon.min():.3f}..{lon.max():.3f}  step {dlon:.4f}")

    if not (abs(dlat - 0.05) < 0.005 and abs(dlon - 0.05) < 0.005):
        print("    ^ expected 0.05 deg CHIRPS spacing  <-- FAIL")
        ok = False

    # Coordinates are CELL CENTERS: a cell centered at 72.525 with 0.05 deg spacing
    # extends to 72.50. Compare against centers +/- half a cell, or files that cover
    # the window exactly fail by one half-cell.
    half = dlat / 2
    covers = (lat.min() - half <= LAT_MIN and lat.max() + half >= LAT_MAX
              and lon.min() - half <= LON_MIN and lon.max() + half >= LON_MAX)
    if not covers:
        print(f"    ^ does not cover the study window "
              f"({LAT_MIN}-{LAT_MAX} N, {LON_MIN}-{LON_MAX} E)  <-- FAIL")
        ok = False

    n_expected = 366 if calendar.isleap(year) else 365
    n_actual = da.shape[0]
    print(f"  time steps           : {n_actual} (expected {n_expected})"
          f"{'' if n_actual == n_expected else '  <-- FAIL'}")
    if n_actual != n_expected:
        ok = False

    # subset BEFORE loading -- a global year is several GB
    if covers:
        lat_slice = (slice(LAT_MAX, LAT_MIN) if descending
                     else slice(LAT_MIN, LAT_MAX))
        sub = da.sel({latn: lat_slice, lonn: slice(LON_MIN, LON_MAX)})
    else:
        sub = da
    print(f"  study-window grid    : {sub.shape[1]} x {sub.shape[2]} cells")

    vals = np.asarray(sub.values, dtype="float64")
    vals[vals < -100] = np.nan          # unmasked fill values
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        print("  all values missing  <-- FAIL")
        ds.close()
        return False

    vmin, vmax = float(finite.min()), float(finite.max())
    nan_pct = 100 * (1 - np.isfinite(vals).mean())
    print(f"  values               : {vmin:.2f}..{vmax:.2f} mm/day   NaN {nan_pct:.1f}%")
    if vmin < -0.01:
        print("    ^ negative rainfall -- fill value not masked")
        ok = False
    if vmax > 1500:
        print("    ^ implausibly large -- unit or fill problem")
        ok = False

    annual = float(np.nansum(np.nanmean(vals.reshape(vals.shape[0], -1), axis=1)))
    good = 400 <= annual <= 1600
    print(f"  window-mean annual   : {annual:.0f} mm  "
          f"<- {'plausible' if good else 'SUSPICIOUS'}")
    if not good:
        ok = False

    ds.close()
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/raw/chirps")
    args = ap.parse_args()

    folder = Path(args.dir)
    if not folder.exists():
        sys.exit(f"{folder.resolve()} does not exist. Run from the project root.")

    print(f"Scanning {folder.resolve()}")
    found, unmatched = find_files(folder)

    if not found:
        listing = list(folder.iterdir())
        print(f"\nNo year-matchable .nc files found. Folder contains {len(listing)} item(s):")
        for p in listing[:20]:
            print(f"  {p.name}")
        sys.exit("\nRename files to include the year, or point --dir elsewhere.")

    if unmatched:
        print(f"\nSkipped {len(unmatched)} file(s) with no clear single year:")
        for p in unmatched[:10]:
            print(f"  {p.name}")

    print(f"\nMatched {len(found)} year(s): {sorted(found)}")

    results = {}
    for year in YEARS:
        if year in found:
            results[year] = check(found[year], year)
        else:
            print(f"\n{year}: no file found")
            results[year] = False

    bad = [y for y, good in results.items() if not good]
    print("\n" + "=" * 56)
    if bad:
        print(f"FAILED or missing: {bad}")
        print("Re-download those years rather than patching them.")
    else:
        print("All years passed structural checks.")
    print("This tests structure and plausibility, not scientific accuracy.")
    print("Still plot one monsoon day on a map before running Stage 1.")


if __name__ == "__main__":
    main()
