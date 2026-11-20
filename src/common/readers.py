"""
Readers for the four precipitation datasets.

Each reader returns data in ONE canonical form, so that everything downstream can
stop caring which product it came from:

    values : (time, lat, lon) float64, mm/day, latitude ASCENDING, NaN for missing
    lat    : 1-D ascending cell centers
    lon    : 1-D ascending cell centers
    dates  : 1-D datetime64[D], the day each field belongs to

Every product-specific quirk is handled here and nowhere else. The quirks are
documented in docs/DATASETS.md; the ones that silently corrupt results if missed:

    IMD     no metadata at all -- shape is asserted, -999 is the fill value, and
            there is NO upper cap. v1 set values >500 mm/day to NaN, deleting real
            extremes from five of nine years. IMD also labels each value with the
            day its 24 h window ENDS (0830 IST), so it is relabelled one day back
            onto the day the rain actually fell -- see IMD_LABEL_SHIFT_DAYS.
    CHIRPS  already mm/day and ascending; nothing to do but read it.
    ERA5    values are METRES and the 00:00 UTC stamp on day D is the accumulated
            total for day D-1. Both must be fixed, and latitude flipped.
    IMERG   arrays are stored (time, lon, lat) -- transposed relative to everyone
            else. One granule (2022-11-23) was downloaded at global extent instead
            of subsetted; it is sliced to the common window rather than skipped,
            because skipping it would leave a hole in the record.
"""

import re
import sys
from pathlib import Path

import numpy as np

# xarray is imported lazily. The IMD reader is pure numpy on a headerless binary
# file, and it is the reference dataset -- it must stay usable (and testable) in an
# environment that has no NetCDF stack installed.
try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None


def _need_xarray(product):
    if xr is None:
        raise ImportError(
            f"Reading {product} needs xarray and a NetCDF backend. "
            "Run: pip install xarray netCDF4"
        )
    return xr

IMD_DIR = Path("data/raw/imd")
CHIRPS_DIR = Path("data/raw/chirps")
ERA5_DIR = Path("data/raw/era5")
IMERG_DIR = Path("data/raw/imerg")

IMD_SHAPE = (129, 135)
IMD_FILL = -999.0
IMD_LAT = np.round(np.arange(129) * 0.25 + 6.5, 3)
IMD_LON = np.round(np.arange(135) * 0.25 + 66.5, 3)

# IMD's value stamped day D is the 24 h accumulation ending 0830 IST on D, i.e.
# 0830 IST D-1 .. 0830 IST D (= 0300 UTC D-1 .. 0300 UTC D). 21 of its 24 hours
# fall on calendar/UTC day D-1, which is the day every product here calls D-1.
# Left uncorrected, IMD sits one day ahead of all three products: in the aligned
# 2015-2023 data the basin-mean correlation with IMD is 0.57 (CHIRPS), 0.69
# (ERA5) and 0.57 (IMERG) at the file's own dates, against 0.82 / 0.85 / 0.93 once
# IMD is moved back a day. The shift is applied here, once, so that nothing
# downstream can compare IMD's label with a product's label by accident.
# Consequence: year file Y covers 31 Dec (Y-1) .. 30 Dec (Y).
IMD_LABEL_SHIFT_DAYS = 1

# Spatial window kept from every product: the basin plus a margin, so that the
# 0.25 deg target cells at the basin edge are fully covered by source pixels.
WINDOW_LAT = (19.6, 23.1)
WINDOW_LON = (72.1, 79.1)


def _leap(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _pick(ds, *names):
    for n in names:
        if n in ds.coords or n in ds.variables:
            return n
    raise KeyError(f"none of {names} found in {list(ds.coords)}")


def _ascending(values, lat):
    """Flip the latitude axis if it is stored descending (ERA5)."""
    if lat[0] > lat[-1]:
        return values[:, ::-1, :], lat[::-1]
    return values, lat


def _window(values, lat, lon):
    """Slice to WINDOW_LAT / WINDOW_LON so every product covers the same area."""
    j = (lat >= WINDOW_LAT[0]) & (lat <= WINDOW_LAT[1])
    k = (lon >= WINDOW_LON[0]) & (lon <= WINDOW_LON[1])
    if j.sum() < 2 or k.sum() < 2:
        raise ValueError(
            f"product does not cover the basin window: lat {lat[0]}..{lat[-1]}, "
            f"lon {lon[0]}..{lon[-1]}"
        )
    return values[:, j, :][:, :, k], lat[j], lon[k]


def _conform(values, lat, lon, lat_ref, lon_ref, tol=1e-6):
    """Cut a wider-extent file down to a reference grid.

    One IMERG granule (2022-11-23) was downloaded globally rather than spatially
    subsetted. Its 0.1 deg grid is the same global grid the others were cut from,
    but a fixed lat/lon window lands on a slightly different set of cell centers,
    so a naive comparison sees "a different grid" and aborts. Skipping the granule
    is not acceptable either -- that would leave a hole in the record. Instead,
    pick out exactly the cells that match the reference grid.

    Returns the conformed array, or None if the reference cells are not all present.
    """
    def indices(src, ref):
        out = []
        for r in ref:
            hit = np.nonzero(np.abs(src - r) <= tol)[0]
            if hit.size != 1:
                return None
            out.append(hit[0])
        return np.array(out)

    j = indices(lat, lat_ref)
    k = indices(lon, lon_ref)
    if j is None or k is None:
        return None
    return values[:, j, :][:, :, k]


# --------------------------------------------------------------------- IMD --
def read_imd_year(year, root=IMD_DIR):
    """IMD is the reference grid, so it is read but never regridded.

    Returned dates are the day the rain FELL, not IMD's file label: entry k of
    RF{year}.GRD is labelled ``year-01-01 + k`` by IMD and returned here as one
    day earlier (see IMD_LABEL_SHIFT_DAYS).
    """
    path = Path(root) / f"RF{year}.GRD"
    raw = np.fromfile(path, dtype="<f4")
    ndays, rem = divmod(raw.size, IMD_SHAPE[0] * IMD_SHAPE[1])
    if rem:
        raise ValueError(f"{path}: size {raw.size} is not a whole number of fields")
    expected = 366 if _leap(year) else 365
    if ndays != expected:
        raise ValueError(f"{path}: {ndays} days, expected {expected}")

    values = raw.reshape(ndays, *IMD_SHAPE).astype("float64")
    values[values == IMD_FILL] = np.nan
    # deliberately NO upper cap -- see module docstring
    labels = np.arange(f"{year}-01-01", f"{year + 1}-01-01", dtype="datetime64[D]")
    dates = labels - np.timedelta64(IMD_LABEL_SHIFT_DAYS, "D")
    return values, IMD_LAT.copy(), IMD_LON.copy(), dates


# ------------------------------------------------------------------ CHIRPS --
def read_chirps_year(year, root=CHIRPS_DIR):
    _need_xarray("CHIRPS")
    path = Path(root) / f"chirps_tapi_{year}.nc"
    with xr.open_dataset(path) as ds:
        var = "precip" if "precip" in ds else list(ds.data_vars)[0]
        units = ds[var].attrs.get("units", "")
        if units and "mm" not in units:
            raise ValueError(f"{path}: unexpected units {units!r}, expected mm/day")
        latn, lonn = _pick(ds, "lat", "latitude"), _pick(ds, "lon", "longitude")
        values = np.asarray(ds[var].transpose("time", latn, lonn).values, dtype="float64")
        lat = np.asarray(ds[latn].values, dtype="float64")
        lon = np.asarray(ds[lonn].values, dtype="float64")
        dates = ds["time"].values.astype("datetime64[D]")

    values, lat = _ascending(values, lat)
    values, lat, lon = _window(values, lat, lon)
    expected = 366 if _leap(year) else 365
    if len(dates) != expected:
        raise ValueError(f"{path}: {len(dates)} timesteps, expected {expected}")
    return values, lat, lon, dates


# -------------------------------------------------------------------- ERA5 --
def read_era5_all(root=ERA5_DIR):
    """All ERA5 files at once: metres -> mm, and the time axis shifted back one day.

    The 00:00 UTC value stamped day D is the completed accumulation for day D-1,
    so year file Y actually covers 31 Dec (Y-1) .. 30 Dec (Y). The 20240101 tail
    file is what supplies 31 Dec 2023. Reading everything and keying by date makes
    that bookkeeping impossible to get subtly wrong.
    """
    _need_xarray("ERA5-Land")
    paths = sorted(Path(root).glob("era5_tapi_*.nc"))
    if not paths:
        raise FileNotFoundError(f"no ERA5 files in {root}")

    by_date, lat_ref, lon_ref = {}, None, None
    for path in paths:
        with xr.open_dataset(path) as ds:
            var = "tp" if "tp" in ds else list(ds.data_vars)[0]
            units = ds[var].attrs.get("units", "")
            if units and units not in ("m", "metres", "meter", "m of water equivalent"):
                raise ValueError(f"{path}: unexpected units {units!r}, expected metres")
            timen = _pick(ds, "valid_time", "time")
            latn = _pick(ds, "latitude", "lat")
            lonn = _pick(ds, "longitude", "lon")
            da = ds[var]
            for extra in ("number", "expver", "step", "surface"):
                if extra in da.dims:
                    da = da.isel({extra: 0}, drop=True)
                elif extra in da.coords:
                    da = da.reset_coords(extra, drop=True)
            values = np.asarray(da.transpose(timen, latn, lonn).values, dtype="float64")
            lat = np.asarray(ds[latn].values, dtype="float64")
            lon = np.asarray(ds[lonn].values, dtype="float64")
            stamps = ds[timen].values.astype("datetime64[D]")

        values = values * 1000.0                       # metres -> mm
        values, lat = _ascending(values, lat)
        values, lat, lon = _window(values, lat, lon)
        dates = stamps - np.timedelta64(1, "D")        # 00:00 on D totals day D-1

        if lat_ref is None:
            lat_ref, lon_ref = lat, lon
        elif not (np.allclose(lat, lat_ref) and np.allclose(lon, lon_ref)):
            raise ValueError(f"{path}: grid differs from the other ERA5 files")

        for i, d in enumerate(dates):
            by_date[d] = values[i]

    ordered = sorted(by_date)
    stacked = np.stack([by_date[d] for d in ordered])
    return stacked, lat_ref, lon_ref, np.array(ordered, dtype="datetime64[D]")


# ------------------------------------------------------------------- IMERG --
_IMERG_DATE = re.compile(r"3IMERG\.(\d{8})")


def read_imerg_all(root=IMERG_DIR, progress_every=500):
    """One file per day. Arrays are (time, lon, lat) and must be transposed."""
    _need_xarray("IMERG")
    paths = sorted(Path(root).glob("*.nc4"))
    if not paths:
        raise FileNotFoundError(f"no IMERG granules in {root}")

    fields, dates, lat_ref, lon_ref = [], [], None, None
    resliced, conformed_files = [], []
    for n, path in enumerate(paths, 1):
        m = _IMERG_DATE.search(path.name)
        if not m:
            raise ValueError(f"cannot parse a date from {path.name}")
        day = np.datetime64(f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}", "D")

        with xr.open_dataset(path) as ds:
            var = "precipitation" if "precipitation" in ds else list(ds.data_vars)[0]
            latn, lonn = _pick(ds, "lat", "latitude"), _pick(ds, "lon", "longitude")
            timen = _pick(ds, "time")
            values = np.asarray(
                ds[var].transpose(timen, latn, lonn).values, dtype="float64")
            lat = np.asarray(ds[latn].values, dtype="float64")
            lon = np.asarray(ds[lonn].values, dtype="float64")

        values, lat = _ascending(values, lat)
        values, lat, lon = _window(values, lat, lon)

        if lat_ref is None:
            lat_ref, lon_ref = lat, lon
        elif not (lat.shape == lat_ref.shape and lon.shape == lon_ref.shape
                  and np.allclose(lat, lat_ref) and np.allclose(lon, lon_ref)):
            conformed = _conform(values, lat, lon, lat_ref, lon_ref)
            if conformed is None:
                raise ValueError(
                    f"{path.name}: grid differs from the first granule and could "
                    f"not be conformed to it "
                    f"(lat {lat[0]:.2f}..{lat[-1]:.2f} n={lat.size} vs "
                    f"n={lat_ref.size})"
                )
            values, lat, lon = conformed, lat_ref, lon_ref
            conformed_files.append(path.name)
        fields.append(values[0])
        dates.append(day)
        if progress_every and n % progress_every == 0:
            print(f"    IMERG {n}/{len(paths)} granules", flush=True)

    if conformed_files:
        print(f"    note: {len(conformed_files)} granule(s) were stored at a wider "
              f"extent and were cut down to the common grid rather than skipped: "
              f"{', '.join(conformed_files[:3])}")

    order = np.argsort(dates)
    stacked = np.stack([fields[i] for i in order])
    days = np.array([dates[i] for i in order], dtype="datetime64[D]")
    if len(np.unique(days)) != len(days):
        raise ValueError("duplicate IMERG dates -- two granules for the same day")
    return stacked, lat_ref, lon_ref, days
