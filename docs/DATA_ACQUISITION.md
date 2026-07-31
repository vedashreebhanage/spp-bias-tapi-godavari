# Data Acquisition Plan — SPP Bias Audit, Tapi Basin

Target period: **2015-01-01 to 2023-12-31** (9 years)
Study domain (generous bbox, clipped to basin polygon later): **19.75–23.00 °N, 72.25–79.00 °E**

Everything lands in this structure:

```
spp-bias-tapi/
└── data/raw/
    ├── imd/          RF2015.GRD ... RF2023.GRD
    ├── chirps/       chirps_tapi_2015.nc ... 2023.nc      [HAVE — verify]
    ├── era5/         era5_tapi_2015.nc ... 2023.nc
    ├── imerg/        3B-DAY...V07B.nc4 (subsetted)
    ├── dem/          tapi_dem_srtm.tif
    └── shapefiles/   tapi_hydrobasins.shp/.shx/.dbf/.prj/.cpg   [HAVE]
```

---

## Status summary

| # | Dataset | Role | Status | Est. size | Effort |
|---|---------|------|--------|-----------|--------|
| 1 | IMD 0.25° gridded rainfall | **Reference / ground truth** | MISSING | ~90 MB | Low |
| 2 | CHIRPS v2.0 daily | SPP under test | HAVE (verify) | ~20 MB | — |
| 3 | ERA5-Land daily precip | SPP under test | MISSING | ~50 MB | Medium |
| 4 | IMERG V07 daily | SPP under test | 1 day only | ~350 MB subsetted | High |
| 5 | SRTM DEM | Terrain zones | MISSING | ~30 MB | Low |
| 6 | HydroBASINS Tapi | Basin mask | HAVE | <1 MB | — |
| 7 | Flood event evidence | DPBA definition | MISSING | — | Manual |

**Do them in this order: 1 → 5 → 3 → 4.** IMD is the reference — nothing works without it. DEM + shapefile let you build and validate the basin mask before any rainfall data arrives.

---

## 1. IMD 0.25° Gridded Rainfall — the reference dataset

The only true ground-truth-ish product here. Note it is itself gauge-interpolated, not truth
(state this as a limitation in the writeup).

**Preferred: `imdlib`** — a maintained Python package that pulls directly from IMD's server.

```powershell
pip install imdlib
python download_imd.py
```

**Fallback if imdlib's endpoint breaks:** manual download from
<https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html>
— select "Rainfall 0.25 x 0.25", one year at a time, save as `RF{year}.GRD`.

**Format facts to verify after download (do not assume):**
- Binary, little-endian float32, no header
- Shape `(n_days, 129, 135)`; lat 6.5→38.5 °N, lon 66.5→100.0 °E, 0.25° spacing
- Fill value `-999.0`
- Units mm/day; IMD rain-day = 0830 IST → 0830 IST
- File size check: `365 × 129 × 135 × 4 bytes = 25,414,200 bytes` (leap: 25,483,860)

If the byte count matches exactly, your shape assumption is right.

---

## 2. CHIRPS — verify, don't re-download

You have 9 restored files of unknown provenance. Before trusting them, run `verify_chirps.py`.
It checks: global attributes name CHIRPS, units are mm/day, lat/lon cover the Tapi bbox at
0.05°, time axis is complete for the year, no absurd values, and basin-mean annual total is
plausible (~600–1100 mm).

If any file fails, re-download that year. CHIRPS global daily NetCDFs are ~5–10 GB/year, so
for a small bbox use Google Earth Engine export or the CHC ClimateSERV API rather than the
full global file.

---

## 3. ERA5-Land — daily precipitation

Use the **derived daily statistics** dataset so Copernicus does the hourly→daily summing
server-side. This avoids the single most common ERA5 error (mis-summing hourly accumulations).

**Setup:**
1. Register at <https://cds.climate.copernicus.eu/> and accept the ERA5-Land licence
   (you must click "Accept terms" on the dataset page or the API returns 403).
2. Get your API token from your profile page.
3. Create `C:\Users\shita\.cdsapirc` containing exactly:

```
url: https://cds.climate.copernicus.eu/api
key: <your-token>
```

4. `pip install cdsapi`
5. `python download_era5.py`

**Critical unit note:** ERA5 total precipitation is in **metres**. Multiply by 1000 for mm.
ERA5 also accumulates in **UTC**, while IMD's rain day is 0830 IST (UTC+05:30). You must
either shift ERA5 by the 3-hour offset or explicitly document the mismatch as a limitation.
Do not silently ignore it.

**Verify the API request:** open the dataset page on CDS, configure your selection in the web
form, click **"Show API request"**, and compare against `download_era5.py`. CDS occasionally
renames request keys; the web form is always authoritative.

---

## 4. IMERG V07 — subset, do not bulk download

Your existing file is a **global** daily granule (29 MB). Nine years globally = **~95 GB**.
Spatially subsetted to the Tapi bbox it's roughly **350 MB**. Always subset.

**Path A — GES DISC subsetter (recommended, reliable):**
1. Free Earthdata account: <https://urs.earthdata.nasa.gov/>
2. In Earthdata profile → Applications → Authorized Apps → approve **NASA GESDISC DATA ARCHIVE**
3. Go to <https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGDF_07/summary>
4. Click **Subset / Get Data** → Download Method: *OPeNDAP*
   → Date range 2015-01-01 to 2023-12-31
   → Region: `72.25, 19.75, 79.00, 23.00` (W, S, E, N)
   → Variables: `precipitation`, `precipitationQualityIndex` (and `randomError` if offered)
5. Click **Get Data** → **Download links list** → save as `imerg_urls.txt`
6. `python download_imerg.py` (reads the URL list, handles Earthdata auth, resumes)

**Path B — scope reduction (pragmatic).** If 3,287 files is too slow, download IMERG **only
for the flood-event and control windows**. You then report the full 9-year bias matrix for
CHIRPS + ERA5, and IMERG only in the DPBA analysis. This is a legitimate, statable scope
decision — just be explicit about it in the methods.

**Unit note:** IMERG V07 daily `precipitation` is **mm/day** (V06 used mm/hr — do not copy
V06 code). Confirm against the variable's `units` attribute.

---

## 5. DEM — for real terrain zones

Needed to replace the longitude-slice pseudo-stratification. At 0.25° (~28 km) cells, a 90 m
DEM is far more than enough.

1. Free API key: <https://portal.opentopography.org/> → My Account → Request API key
2. `python download_dem.py` (single HTTP request, returns one GeoTIFF for the bbox)

Alternative without a key: Copernicus GLO-90 tiles from AWS Open Data (`s3://copernicus-dem-90m`,
no credentials), but you must mosaic several tiles yourself.

**Use:** aggregate to mean elevation per 0.25° IMD cell, then define zones. Either fixed
thresholds (e.g. <150 m / 150–400 m / >400 m) or terciles of the in-basin cell distribution.
Justify whichever you pick; do not tune it to make results look good.

---

## 6. Flood event evidence — the weakest link in the current design

DPBA is meaningless if the event windows aren't defensible. Before Stage 1 you need, for each
event, a **citable source** for the dates:

- CWC Flood Forecasting: <https://ffs.india-water.gov.in/> (HFL / danger-level crossings)
- IMD Ukai/Hathnur district rainfall bulletins and monsoon reports
- Dartmouth Flood Observatory global archive
- EM-DAT (<https://www.emdat.be/>, free academic registration)
- Contemporary news reports for Surat/Bhusawal flooding as corroboration only

Record for each event: start/end date, evidence type, source URL, and — importantly — the
rule used to define the **control window** (e.g. same calendar days ± 15 in all non-event
years). Fix that rule *before* looking at results.

---

## After downloading

Do not start Stage 1 analysis until you've run a per-dataset verification pass:
open each file, print dimensions, units, coordinate ranges, time completeness, and plot one
monsoon day on a map. A flipped latitude axis or a metres/mm error is invisible in a CSV and
obvious on a map.
