"""
STAGE 2 — PART 1: Data Loader & Shared Utilities
=================================================
What this file does:
    Loads IMD and CHIRPS data the same way Stage 1 did, but now
    organises it into a clean pandas DataFrame where every row is
    one (date, lat, lon) observation. This flat table format is what
    machine learning models expect as input. Every metric computation,
    feature engineering step, and model training step in Parts 2-4
    will use this DataFrame.

Why a DataFrame instead of 3D arrays?
    3D arrays (time × lat × lon) are efficient for spatial operations
    but awkward for ML. A flat table lets us:
    - Filter by zone, season, or event window in one line
    - Add feature columns (elevation zone, day-of-year) trivially
    - Feed directly into sklearn and XGBoost without reshaping
"""

import numpy as np
import pandas as pd
import struct
import zlib
import calendar
import datetime
import os

# ── Constants ─────────────────────────────────────────────────────────────────

# Sub-basin longitude boundaries (based on Hathnur and Ukai dams)
UPPER_LON_MIN  = 75.97   # Upper Tapi: east of Hathnur Dam
LOWER_LON_MAX  = 73.57   # Lower Tapi: west of Ukai Dam

# Tapi bounding box in IMD grid coordinates
TAPI_LAT_MIN, TAPI_LAT_MAX = 20.0, 22.5
TAPI_LON_MIN, TAPI_LON_MAX = 72.5, 78.5

# Flood event windows — (year, start_date, end_date)
# These are the documented CWC HFL breach periods
FLOOD_EVENTS = [
    (2021, '2021-08-01', '2021-08-20'),
    (2023, '2023-09-14', '2023-09-20'),
]

# Season month mapping
SEASON_MAP = {6:'JJAS', 7:'JJAS', 8:'JJAS', 9:'JJAS',
              10:'ON',  11:'ON',
              12:'DJF', 1:'DJF',  2:'DJF',
              3:'MAM',  4:'MAM',  5:'MAM'}


# ── IMD Reader ────────────────────────────────────────────────────────────────

def read_imd_grd(filepath, year):
    """
    Reads an IMD .GRD binary file and returns a 3D numpy array.

    IMD stores data as:
        - Shape: (n_days, 129 lat, 135 lon)
        - Format: little-endian float32 (confirmed in Stage 1)
        - Fill value: -999 (replaced with NaN)
        - Units: mm/day
        - Grid: 6.5°N to 38.5°N, 66.5°E to 100.0°E at 0.25° spacing
    """
    n_lon, n_lat = 135, 129
    lat = np.linspace(6.5, 38.5, n_lat)
    lon = np.linspace(66.5, 100.0, n_lon)
    n_days = 366 if calendar.isleap(year) else 365
    with open(filepath, 'rb') as f:
        raw = f.read()
    arr = np.frombuffer(raw[:n_days * n_lat * n_lon * 4], dtype='<f4').copy()
    arr = arr.reshape(n_days, n_lat, n_lon)
    arr[arr <= -999] = np.nan
    arr[arr < 0]    = np.nan
    arr[arr > 500]  = np.nan
    return arr, lat, lon, n_days


# ── CHIRPS Reader ─────────────────────────────────────────────────────────────

_clat, _clon = None, None   # coordinate cache so we don't re-scan every file

def read_chirps_nc(filepath):
    """
    Reads a CHIRPS .nc file using raw HDF5/zlib parsing (no netCDF4 needed).

    CHIRPS stores data as:
        - Shape: (n_days, ~80 lat, ~120 lon) — already clipped to Tapi bbox
        - Format: little-endian float32, zlib compressed inside HDF5
        - Fill value: -9999 (replaced with NaN)
        - Units: mm/day
        - Resolution: 0.05° (~5.5km)

    The coordinate arrays are found by scanning for float32 sequences
    in the known lat/lon ranges (19-24°N, 72-79°E).
    """
    global _clat, _clon
    with open(filepath, 'rb') as f:
        raw = f.read()

    def scan_f(data, vmin, vmax, min_run):
        i = 0
        while i < len(data) - 4:
            v = struct.unpack_from('<f', data, i)[0]
            if vmin <= v <= vmax:
                run, j = [], i
                while j + 4 <= len(data):
                    vv = struct.unpack_from('<f', data, j)[0]
                    if vmin - 0.5 <= vv <= vmax + 0.5:
                        run.append(round(vv, 4)); j += 4
                    else:
                        break
                if len(run) >= min_run:
                    return np.array(run)
            i += 4
        return None

    lat = scan_f(raw, 19.0, 24.0, 50)
    lon = scan_f(raw, 72.0, 79.0, 80)
    if lat is not None: _clat = lat
    else: lat = _clat
    if lon is not None: _clon = lon
    else: lon = _clon

    for i in range(len(raw) - 4):
        if raw[i:i+2] in (b'\x78\x9c', b'\x78\x01', b'\x78\xda', b'\x78\x5e'):
            try:
                dec = zlib.decompress(raw[i:])
                arr = np.frombuffer(dec, dtype='<f4').copy()
                n_t = len(arr) // (len(lat) * len(lon))
                if n_t in [365, 366]:
                    arr = arr.reshape(n_t, len(lat), len(lon))
                    arr[arr <= -9000] = np.nan
                    return arr, lat, lon, n_t
            except:
                pass
    return None, lat, lon, 0


# ── Regridder ─────────────────────────────────────────────────────────────────

def regrid_chirps_to_imd(chirps_data, lat_c, lon_c, lat_t, lon_t):
    """
    Regrids CHIRPS from 0.05° to IMD's 0.25° grid.

    Method: simple averaging. For each IMD cell centred at (lt, ln),
    find all CHIRPS cells within ±0.125° and average them.
    This is called 'block averaging' or 'upscaling'.

    Why not interpolation?
        Interpolation would create artificial precision. We are going
        from fine (0.05°) to coarse (0.25°), so averaging is correct
        — it conserves total rainfall volume across the domain.
    """
    n_t = chirps_data.shape[0]
    out = np.full((n_t, len(lat_t), len(lon_t)), np.nan, dtype=np.float32)
    for li, lt in enumerate(lat_t):
        lmask = (lat_c >= lt - 0.125) & (lat_c < lt + 0.125)
        for loi, ln in enumerate(lon_t):
            omask = (lon_c >= ln - 0.125) & (lon_c < ln + 0.125)
            if lmask.any() and omask.any():
                sub = chirps_data[:, np.where(lmask)[0], :]
                sub = sub[:, :, np.where(omask)[0]]
                out[:, li, loi] = np.nanmean(sub, axis=(1, 2))
    return out


# ── Zone labeller ─────────────────────────────────────────────────────────────

def get_zone(lon_val):
    """
    Returns the sub-basin zone label for a given longitude.

    Upper Tapi: east of Hathnur Dam (75.97°E) — hilly, orographic rain
    Middle Tapi: between the two dams — semi-arid plateau
    Lower Tapi: west of Ukai Dam (73.57°E) — flat coastal Gujarat
    """
    if lon_val > UPPER_LON_MIN:  return 'upper'
    if lon_val > LOWER_LON_MAX:  return 'middle'
    return 'lower'


# ── Main builder: create flat DataFrame ──────────────────────────────────────

def build_dataframe(years, imd_dir, chirps_dir):
    """
    Loads all years of IMD and CHIRPS data and returns a single
    flat pandas DataFrame with one row per (date, grid_cell).

    Columns in the output DataFrame:
        date        — calendar date
        year        — year integer
        doy         — day of year (1-366), used as a seasonal feature
        month       — month integer
        season      — JJAS / ON / DJF / MAM
        lat         — latitude of the IMD grid cell
        lon         — longitude of the IMD grid cell
        zone        — upper / middle / lower (Tapi sub-basin)
        imd         — IMD observed rainfall (mm/day) — ground truth
        chirps      — CHIRPS rainfall regridded to IMD grid (mm/day)
        is_flood    — 1 if this date falls in a documented flood event, 0 otherwise
        is_wet_imd  — 1 if IMD >= 1mm/day (actual rain day)
        is_wet_chp  — 1 if CHIRPS >= 1mm/day (satellite detected rain)
    """
    print("Building flat DataFrame from IMD + CHIRPS...")
    all_rows = []

    for yr in years:
        print(f"  Loading {yr}...")

        # Load IMD
        imd_path = os.path.join(imd_dir, f'RF{yr}.GRD')
        imd_arr, lat_imd, lon_imd, n_days = read_imd_grd(imd_path, yr)

        # Clip IMD to Tapi bbox
        lat_ti = np.where((lat_imd >= TAPI_LAT_MIN) & (lat_imd <= TAPI_LAT_MAX))[0]
        lon_ti = np.where((lon_imd >= TAPI_LON_MIN) & (lon_imd <= TAPI_LON_MAX))[0]
        imd_tapi = imd_arr[:, lat_ti, :][:, :, lon_ti]
        lat_t = lat_imd[lat_ti]
        lon_t = lon_imd[lon_ti]

        # Load CHIRPS and regrid
        chp_path = os.path.join(chirps_dir, f'chirps_tapi_{yr}.nc')
        chp_arr, lat_c, lon_c, _ = read_chirps_nc(chp_path)
        chp_rg = regrid_chirps_to_imd(chp_arr, lat_c, lon_c, lat_t, lon_t)

        # Build date list for this year
        start_date = datetime.date(yr, 1, 1)
        dates = [start_date + datetime.timedelta(days=d) for d in range(n_days)]

        # Flood event lookup for this year
        flood_dates = set()
        for fy, fd0, fd1 in FLOOD_EVENTS:
            if fy == yr:
                d0 = datetime.date.fromisoformat(fd0)
                d1 = datetime.date.fromisoformat(fd1)
                current = d0
                while current <= d1:
                    flood_dates.add(current)
                    current += datetime.timedelta(days=1)

        # Flatten: iterate over days and cells
        for di, date in enumerate(dates):
            season = SEASON_MAP[date.month]
            is_flood = 1 if date in flood_dates else 0

            for li, lat_val in enumerate(lat_t):
                for loi, lon_val in enumerate(lon_t):
                    imd_val = float(imd_tapi[di, li, loi])
                    chp_val = float(chp_rg[di, li, loi])

                    # Skip cells where both are NaN — no information
                    if np.isnan(imd_val) and np.isnan(chp_val):
                        continue

                    all_rows.append({
                        'date'       : date,
                        'year'       : yr,
                        'doy'        : date.timetuple().tm_yday,
                        'month'      : date.month,
                        'season'     : season,
                        'lat'        : round(float(lat_val), 2),
                        'lon'        : round(float(lon_val), 2),
                        'zone'       : get_zone(lon_val),
                        'imd'        : imd_val,
                        'chirps'     : chp_val,
                        'is_flood'   : is_flood,
                        'is_wet_imd' : 1 if (not np.isnan(imd_val) and imd_val >= 1.0) else 0,
                        'is_wet_chp' : 1 if (not np.isnan(chp_val) and chp_val >= 1.0) else 0,
                    })

    df = pd.DataFrame(all_rows)
    print(f"  Done. Shape: {df.shape}")
    print(f"  Date range: {df.date.min()} to {df.date.max()}")
    print(f"  Zones: {df.zone.value_counts().to_dict()}")
    print(f"  Flood rows: {df.is_flood.sum()} / {len(df)}")
    return df


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == '__main__':
    IMD_DIR    = '/mnt/user-data/uploads'
    CHIRPS_DIR = '/mnt/user-data/uploads'

    # Rename IMD files to match expected pattern
    import glob
    imd_files = glob.glob(f'{IMD_DIR}/RF*.GRD')
    print(f"Found {len(imd_files)} IMD files")
    chirps_files = glob.glob(f'{CHIRPS_DIR}/chirps_tapi_*.nc')
    print(f"Found {len(chirps_files)} CHIRPS files")

    years = list(range(2015, 2024))
    df = build_dataframe(years, IMD_DIR, CHIRPS_DIR)

    # Save for use by Parts 2, 3, 4
    df.to_parquet('/home/claude/stage2/tapi_data.parquet', index=False)
    print("\nSaved to tapi_data.parquet")
    print(df.head(10).to_string())
