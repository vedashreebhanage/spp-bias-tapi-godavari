# Dataset Reference — SPP Bias Audit, Tapi Basin

Exact contents of every dataset in `data/raw/`, as read from the files themselves
(binary headers parsed directly; NetCDF contents from `scripts/inspect_data.py`,
report dated 2026-08-06). Facts only; handling requirements listed where the file's
own conventions demand them.

---

## 1. IMD Gridded Rainfall — `data/raw/imd/RF{2015..2023}.GRD`

**Role: reference ("ground truth") dataset.**

| Property | Value |
|---|---|
| Source | India Meteorological Department, Pune |
| Format | Raw binary, little-endian float32, no header, no metadata |
| Grid | 129 lat × 135 lon; lat 6.5→38.5 °N, lon 66.5→100.0 °E; 0.25° spacing |
| Array layout | `(n_days, 129, 135)`, latitude ascending |
| Units | mm/day |
| Fill value | −999.0 (ocean / outside India); first values in every file are −999 (Indian Ocean corner) |
| Rain-day window | 0830 IST to 0830 IST, **labelled by the day it ends** (value stamped D = rain from 0830 IST D−1 to 0830 IST D) |
| Land cells per day | 4,964 (constant across all years) |
| Completeness | 9/9 years, byte counts exact for (days×129×135×4) |

Per-year values (all-India, finite cells only):

| Year | Days | Max (mm/day) | Daily mean (mm) |
|---|---|---|---|
| 2015 | 365 | 679.2 | 2.95 |
| 2016 | 366 | 673.7 | 3.00 |
| 2017 | 365 | 540.3 | 3.10 |
| 2018 | 365 | 534.2 | 2.76 |
| 2019 | 365 | 487.1 | 3.48 |
| 2020 | 366 | 539.9 | 3.49 |
| 2021 | 365 | 526.7 | 3.36 |
| 2022 | 365 | 979.1 | 3.44 |
| 2023 | 365 | 689.0 | 2.96 |

Sample (1 Aug 2021): Surat cell (21.25N, 72.75E) = 3.48 mm; Bhusawal (21.00N,
75.75E) = 0.58 mm; Burhanpur (21.25N, 76.25E) = 0.00 mm; Multai (21.75N, 78.25E)
= 0.60 mm.

**Handling requirements**
- Shape and coordinates must be asserted by the reader (no metadata in file).
- Mask value −999 → NaN. No other value transformation.
- **Relabel dates one day back** (`readers.IMD_LABEL_SHIFT_DAYS = 1`): the value
  stamped D covers 0300 UTC D−1 → 0300 UTC D, i.e. 21 of its 24 hours fall on
  day D−1, which is what every product calls D−1. Year file Y therefore covers
  31 Dec (Y−1) .. 30 Dec (Y), and the aligned record ends 2023-12-30 (31 Dec 2023
  would need `RF2024.GRD`). Found 2026-11-20: without the shift, basin-mean
  correlation with IMD is 0.57 / 0.69 / 0.57 (CHIRPS / ERA5 / IMERG); with it,
  0.82 / 0.85 / 0.93. `build_dataset.py` now fails if any product's best lag
  against IMD is not 0.
- Note: five of nine years contain daily values above 500 mm (max 979.1 mm in
  2022). Any cap at 500 mm deletes real data.

---

## 2. CHIRPS v2.0 — `data/raw/chirps/chirps_tapi_{2015..2023}.nc`

**Role: satellite product under test #1.**

| Property | Value |
|---|---|
| Source | Climate Hazards Group, UC Santa Barbara (attrs verbatim: title "CHIRPS Version 2.0", creator Pete Peterson, CF-1.6) |
| Format | NetCDF-4 (HDF5 container) |
| Dimensions | time × 80 lat × 120 lon |
| Grid | lat 19.525→23.475 °N **ascending**, lon 72.525→78.475 °E; 0.05° spacing (values are cell centers; cell edges span 19.5–23.5, 72.5–78.5) |
| Variable | `precip`, float32 |
| Units | mm/day |
| Time | one step per day, complete years (365/366) |
| Finite fraction | 98.1% (2015 file; remainder NaN) |

Per-year values (finite cells):

| Year | Max (mm/day) | Mean (mm/day) |
|---|---|---|
| 2015 | 206.69 | 2.483 |
| 2016 | 316.58 | 3.190 |
| 2017 | 273.97 | 2.669 |
| 2018 | 330.26 | 2.505 |
| 2019 | 338.73 | 3.755 |
| 2020 | 276.53 | 3.559 |
| 2021 | 245.12 | 2.986 |
| 2022 | 320.51 | 3.867 |
| 2023 | 334.58 | 2.862 |

**Attribute quirks (present in the files, verbatim)**
- `standard_name = "convective precipitation rate"` — a CHIRPS labelling quirk;
  the variable is total precipitation, not convective-only.
- `geostatial_lat_min/max = ±50`, `lon ±180` — describe the parent global
  product, not this clipped file.
- `comments = "time variable denotes the first day of the given day"`.

**Handling requirements**
- None beyond NaN awareness. Units and orientation are already analysis-ready.
- Regrid 0.05° → 0.25° by **area-weighted averaging, not a 5×5 block mean.**
  An earlier note in this file claimed exact 5×5 nesting; that is wrong and was
  corrected on 2026-08-13 after checking the grids against each other. CHIRPS cell
  edges lie on 19.5 + 0.05k (lat) and 72.5 + 0.05k (lon); IMD cell edges lie on
  6.375 + 0.25k and 66.375 + 0.25k. An IMD edge such as 20.125 sits at 12.5 CHIRPS
  steps — **exactly half a cell out**. Each 0.25° cell therefore covers six partial
  CHIRPS cells per axis, weighted 0.5, 1, 1, 1, 1, 0.5. A 5×5 block mean would
  average a footprint displaced 0.025° (~2.8 km) from the IMD cell it claims to
  represent: a small, uniform, and completely invisible misalignment of one product
  against the reference. `src/common/regrid.py` handles this correctly; the case is
  pinned by `tests/test_regrid.py::test_real_chirps_to_imd_is_offset_by_half_a_cell`.
- Coordinates are stored as **float32**, so the spacing reads as 0.049995–0.050003
  rather than 0.05. The axis is fitted to its regular grid from the full span
  before edges are computed; inferring the step from individual differences
  accumulates ~3.6e-4° of drift across the axis.

---

## 3. ERA5-Land — `data/raw/era5/era5_tapi_{2015..2023}.nc` + `era5_tapi_20240101.nc`

**Role: reanalysis product under test #2.**

| Property | Value |
|---|---|
| Source | ECMWF via Copernicus CDS (GRIB converted by cfgrib; dataset `reanalysis-era5-land`) |
| Format | NetCDF (from GRIB), CF-1.7 |
| Dimensions | valid_time × 33 lat × 68 lon |
| Grid | lat 23.0→19.8 °N **descending**, lon 72.3→79.0 °E; 0.1° spacing |
| Variable | `tp` ("Total precipitation"), float32 |
| Units | **metres** (attr `units = m`) |
| Step type | `accum` (accumulated variable) |
| Timesteps | 00:00 UTC only, one per day |
| Finite fraction | 95.1% (ERA5-Land is land-only; sea cells are NaN) |
| Extra coords | `number`, `expver` (=0001 throughout) — ensemble/version bookkeeping, ignorable |
| Tail file | `era5_tapi_20240101.nc`, 1 timestep (covers 31 Dec 2023 after the shift) |

Per-year values (finite cells, in native metres):

| Year | Max (m) | = mm | Mean (m) |
|---|---|---|---|
| 2015 | 0.3351 | 335.1 | 0.0025 |
| 2016 | 0.2438 | 243.8 | 0.0032 |
| 2017 | 0.3993 | 399.3 | 0.0025 |
| 2018 | 0.2804 | 280.4 | 0.0025 |
| 2019 | 0.4525 | 452.5 | 0.0041 |
| 2020 | 0.2725 | 272.5 | 0.0031 |
| 2021 | 0.2874 | 287.4 | 0.0028 |
| 2022 | 0.2345 | 234.5 | 0.0037 |
| 2023 | 0.2909 | 290.9 | 0.0031 |

**Handling requirements (all mandatory)**
1. Multiply by 1000 (metres → mm).
2. Shift the time axis **back one day**: the 00:00 UTC value stamped day D is the
   accumulated total for day D−1. Year files therefore cover 31 Dec (Y−1) through
   30 Dec (Y); the 20240101 tail file supplies 31 Dec 2023.
3. Flip latitude to ascending (or index accordingly).
4. Aggregate 0.1° → 0.25° by area-weighted averaging (non-integer nesting).
5. Day boundary is UTC; IMD's is 0830 IST (UTC+05:30) — a ~3 h offset to be
   stated in methods.

---

## 4. IMERG V07 Final Daily — `data/raw/imerg/3B-DAY.*.V07B.nc4`

**Role: satellite product under test #3.**

| Property | Value |
|---|---|
| Source | NASA GES DISC, GPM_3IMERGDF v07 (DOI 10.5067/GPM/IMERGDF/DAY/07), spatially subsetted via OPeNDAP |
| Format | NetCDF-4, one file per day |
| Dimensions | time(1) × **lon(69) × lat(33)** — longitude-major |
| Grid | lat 19.75→22.95 °N ascending, lon 72.25→79.05 °E; 0.1° spacing |
| Variables | `precipitation` (float32, mm/day, "Daily mean precipitation rate (combined microwave-IR)"), `time_bnds` |
| Finite fraction | 100% (no sea masking in this product) |
| File size | 35–43 KB per granule, no zero-byte files |

**Download status as of the report: INCOMPLETE.**
Present: 2,693 subsetted granules, first = 2015-01-01, last = 2022-05-17.
Expected: 3,287. Missing ≈ 594 granules (roughly mid-2022 through 2023) — the
tail of the interrupted download run. Re-running `scripts/download_imerg.py`
resumes automatically. One global-size granule from an earlier manual download
(`...20221123...`, 30.5 MB) also present; excluded from analysis by file size.

**Handling requirements**
1. Transpose arrays: files store `(time, lon, lat)`; analysis convention is
   `(time, lat, lon)`.
2. IMERG's day is 00:00–23:59 UTC (per `time_bnds`) — same 3 h offset vs IMD as
   ERA5; state once in methods.
3. Aggregate 0.1° → 0.25° by area-weighted averaging.
4. Concatenate 3,287 single-day files along time (`open_mfdataset`).

---

## 5. HydroBASINS Shapefile — `data/raw/shapefiles/tapi_hydrobasins.*`

**Role: basin mask source.**

| Property | Value |
|---|---|
| Format | ESRI Shapefile (file code 9994, version 1000), shape type 5 = Polygon |
| CRS | GCS_WGS_1984 (geographic, degrees) |
| Records | **14 polygons** — multiple basins of western/central India, not Tapi alone |
| Combined bbox | lon 71.149→81.888, lat 10.179→28.058 |
| Attributes | 13 HydroBASINS fields (HYBAS_ID, NEXT_DOWN, SUB_AREA, UP_AREA, PFAF_ID, …) |

`SUB_AREA` (km²) of the 14 records: 130,776 / 142,339 / 108,513 / 30,041 /
55,542 / 92,287 / **64,337** / 2,499 / 96,047 / 6,081 / 34,076 / 28 / 1,892 /
29,977.

Record 7 — `HYBAS_ID 4050031520`, `PFAF_ID 45404`, `SUB_AREA 64,337.5 km²` — is
the polygon whose area matches the Tapi basin (CWC figure 65,145 km²).

**Handling requirements**
- Select the Tapi polygon by `HYBAS_ID = 4050031520`; do **not** dissolve all 14
  records (their union spans ~10–28 °N).
- Validate the selected polygon's area against both its own SUB_AREA attribute
  and the CWC figure before building any mask.

---

## 6. SRTM DEM — `data/raw/dem/tapi_dem_srtm.tif`

**Role: terrain attributes (elevation, ruggedness) per grid cell.**

| Property | Value |
|---|---|
| Source | OpenTopography, SRTMGL3 (90 m) |
| Format | GeoTIFF, little-endian, LZW-compressed, 256×256 tiles |
| Size | 8100 × 3900 pixels, 16-bit signed integer |
| Pixel size | 0.000833° (~90 m) |
| Top-left corner | 72.2496 °E, 23.0004 °N (matches requested bbox 72.25–79.0 E, 19.75–23.0 N) |
| Elevation range | −32 m to 1,556 m, mean 359 m (full bbox, pre-masking) |
| Integrity | full-read verified (not truncated) |

**Handling requirements**
- Negative values near the coast are SRTM noise/tidal flats; treat <−100 as void.
- Values above ~1,350 m lie outside the basin (Pachmarhi corner of the bbox) and
  disappear after basin masking.

---

## Cross-dataset summary

| | IMD | CHIRPS | ERA5-Land | IMERG |
|---|---|---|---|---|
| Native resolution | 0.25° | 0.05° | 0.1° | 0.1° |
| Nests into IMD? | — | **no** (half-cell offset) | no | no |
| Native units | mm/day | mm/day | **m/day** | mm/day |
| Lat orientation | ascending | ascending | **descending** | ascending |
| Array order | (t, lat, lon) | (t, lat, lon) | (t, lat, lon) | **(t, lon, lat)** |
| Day definition | 0830–0830 IST, stamped by end day (shift −1 day) | calendar day | 00–24 UTC (shift −1 day) | 00–24 UTC |
| Fill/missing | −999 | NaN (~2%) | NaN over sea (~5%) | none |
| Files | 9 | 9 | 9 + 1 tail | 3,287 expected (2,693 present) |
| Status | complete, verified | complete, verified | complete | **incomplete — resume download** |

Every product converges, after preprocessing, to: `(time, lat-ascending,
lon)` at 0.25°, mm/day, NaN outside data, on the IMD grid.
