"""
Verify the 9 restored CHIRPS files before trusting them.

Their provenance is unknown (recovered from the Recycle Bin), so check that they
are genuinely CHIRPS, correctly georeferenced, complete, and in mm/day.

    python verify_chirps.py

Any year that fails should be re-downloaded rather than patched.
"""

import calendar
import sys
from pathlib import Path

try:
    import numpy as np
    import xarray as xr
except ImportError:
    sys.exit("Run:  pip install xarray netCDF4 numpy")

CHIRPS_DIR = Path("data/raw/chirps")
YEARS = range(2015, 2024)

# Expected domain (files were clipped to a Tapi bbox by whoever made them)
LAT_RANGE = (19.0, 24.0)
LON_RANGE = (72.0, 79.0)


def check_year(year: int) -> bool:
    path = CHIRPS_DIR / f"chirps_tapi_{year}.nc"
    if not path.exists():
        print(f"{year}: MISSING")
        return False

    ok = True
    print(f"\n{year}  ({path.stat().st_size / 1e6:.1f} MB)")
    ds = xr.open_dataset(path)

    # --- provenance ---
    blob = " ".join(str(v) for v in ds.attrs.values()).lower()
    print(f"  attrs mention CHIRPS : {'yes' if 'chirps' in blob else 'NO  <-- suspicious'}")
    if "chirps" not in blob:
        ok = False

    # --- variable ---
    candidates = [v for v in ds.data_vars if ds[v].ndim == 3]
    if not candidates:
        print("  no 3-D data variable found  <-- FAIL")
        ds.close()
        return False
    var = candidates[0]
    da = ds[var]
    units = da.attrs.get("units", "(none)")
    print(f"  variable / units     : {var} / {units}")
    if "mm" not in str(units).lower():
        print("    ^ expected mm/day -- check before use")
        ok = False

    # --- coordinates ---
    latname = "latitude" if "latitude" in ds.coords else "lat"
    lonname = "longitude" if "longitude" in ds.coords else "lon"
    lat, lon = ds[latname].values, ds[lonname].values
    dlat = float(abs(np.diff(lat).mean()))
    dlon = float(abs(np.diff(lon).mean()))
    print(f"  lat  {lat.min():.3f} to {lat.max():.3f}   step {dlat:.4f}  "
          f"({'ascending' if lat[0] < lat[-1] else 'DESCENDING'})")
    print(f"  lon  {lon.min():.3f} to {lon.max():.3f}   step {dlon:.4f}")
    if not (abs(dlat - 0.05) < 0.005 and abs(dlon - 0.05) < 0.005):
        print("    ^ expected 0.05 deg CHIRPS spacing  <-- FAIL")
        ok = False
    if not (lat.min() >= LAT_RANGE[0] - 1 and lat.max() <= LAT_RANGE[1] + 1):
        print("    ^ latitude outside expected Tapi window  <-- FAIL")
        ok = False

    # --- time completeness ---
    n_expected = 366 if calendar.isleap(year) else 365
    n_actual = da.shape[0]
    print(f"  time steps           : {n_actual} (expected {n_expected})"
          f"{'' if n_actual == n_expected else '  <-- FAIL'}")
    if n_actual != n_expected:
        ok = False

    # --- values ---
    vals = da.values
    finite = vals[np.isfinite(vals)]
    finite = finite[finite > -100]          # drop any unmasked fill
    if finite.size == 0:
        print("  all values missing  <-- FAIL")
        ds.close()
        return False

    vmin, vmax = float(finite.min()), float(finite.max())
    nan_pct = 100 * (1 - np.isfinite(vals).mean())
    print(f"  value range          : {vmin:.2f} to {vmax:.2f} mm/day   "
          f"NaN {nan_pct:.1f}%")
    if vmin < -0.01:
        print("    ^ negative rainfall  <-- fill value not masked")
        ok = False
    if vmax > 1500:
        print("    ^ implausibly large  <-- unit or fill problem")
        ok = False

    # --- domain-mean annual total (the headline sanity check) ---
    annual = float(np.nansum(np.nanmean(vals.reshape(n_actual, -1), axis=1)))
    verdict = "plausible" if 400 <= annual <= 1600 else "SUSPICIOUS"
    print(f"  domain-mean annual   : {annual:.0f} mm   <- {verdict}")
    print("    (bbox includes drier areas outside the basin; the strict "
          "basin-masked value comes later)")
    if not 400 <= annual <= 1600:
        ok = False

    ds.close()
    return ok


if __name__ == "__main__":
    print(f"Checking CHIRPS files in {CHIRPS_DIR.resolve()}")
    results = {y: check_year(y) for y in YEARS}

    print("\n" + "=" * 50)
    bad = [y for y, good in results.items() if not good]
    if bad:
        print(f"FAILED: {bad}")
        print("Re-download these years rather than patching them.")
    else:
        print("All 9 years passed. Safe to use.")
    print("Note: this checks structure and plausibility, not scientific accuracy.")
    print("Still plot one monsoon day on a map before running Stage 1.")
