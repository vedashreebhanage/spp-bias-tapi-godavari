# scripts/ — data acquisition & verification

Copy this whole folder into your project as `SPP BIAS\scripts\`, replacing what's
there. **Run everything from the project root**, never from inside `scripts\`, or the
relative `data/raw/...` paths won't resolve:

```powershell
cd "C:\Users\shita\Desktop\SPP BIAS"
python scripts\check_status.py
```

## Start here

`check_status.py` inventories every dataset and tells you exactly what to do next.
It downloads nothing. Run it whenever you're unsure where things stand.

## Files

| File | What it does | Downloads? |
|---|---|---|
| `basins.py` | Single source of truth for bounding boxes, years, flood windows | no |
| `check_status.py` | Inventory + next-actions list | no |
| `verify_chirps.py` | Structural check on the 9 recovered CHIRPS files | no |
| `download_imd.py` | IMD reference data (already complete — use `--verify-only`) | only if missing |
| `download_dem.py` | SRTM DEM, truncation-proof, with tiled fallback | yes |
| `download_era5.py` | ERA5-Land **daily** precipitation | yes |
| `download_imerg.py` | IMERG V07 daily, subsetted via GES DISC | yes |

Every script takes `--basin tapi` (default) or `--basin godavari`.

## Run order

```powershell
python scripts\check_status.py                              # where am I?
python scripts\verify_chirps.py                             # free, do it first
python scripts\download_imd.py --verify-only                # confirms sizes only
python scripts\download_dem.py                              # --tiles 4 if it fails
python scripts\download_era5.py --clean                     # removes stale hourly files
python scripts\download_imerg.py --show-settings --events-only
#   ...then generate imerg_urls.txt in the browser, and:
python scripts\download_imerg.py
```

## Why `basins.py` exists

Every product previously carried its own hardcoded bounding box. When those drift
apart by even a fraction of a degree, the products silently cover different areas and
every comparison between them is wrong — an error that produces plausible-looking
numbers and no warning. One definition, imported everywhere, removes that failure mode.

It deliberately holds **no** event list. An earlier `FLOOD_EVENTS` dict carried
placeholder windows marked UNVERIFIED; checking them against IMD showed the 2021
window (Aug 1–20) was almost entirely dry, so any "event" metric over it would have
measured dry-day behaviour. Episodes are now derived by
`src/stage1/define_episodes.py` and read back through `basins.load_episodes()`,
which raises if they have not been generated — no analysis can run on an assumed
event set.

## Accounts you'll need

| Service | For | Gotcha |
|---|---|---|
| [OpenTopography](https://portal.opentopography.org/) | DEM | key in `OPENTOPO_API_KEY`; restart VS Code after `setx` |
| [Copernicus CDS](https://cds.climate.copernicus.eu/) | ERA5-Land | must click **Accept licence** on the dataset page or every request 403s |
| [NASA Earthdata](https://urs.earthdata.nasa.gov/) | IMERG | must authorize **NASA GESDISC DATA ARCHIVE** in your profile or every request 401s |

Keys stay out of tracked files: `OPENTOPO_API_KEY` lives in the environment,
`.cdsapirc` and `.env` are gitignored.

## Units — the errors that don't announce themselves

| Product | Native unit | Action |
|---|---|---|
| IMD | mm/day | none |
| CHIRPS | mm/day | none |
| ERA5-Land | **metres** | ×1000 at analysis time |
| IMERG V07 | mm/day | V06 was mm/hr — don't reuse V06 code |

ERA5 accumulates in UTC; IMD's rain day runs 0830–0830 IST. Decide how you handle that
offset and state it. A silent mismatch here shifts every daily comparison.

## Known state

- IMD — complete, sizes verified
- CHIRPS — recovered, needs `verify_chirps.py`
- Shapefile — present (Tapi only; Godavari would need its own)
- DEM — first attempt truncated; re-run with this version
- ERA5 — 3 partial *hourly* files, wrong product; `--clean` then re-download
- IMERG — 1 global sample day; needs the subsetter workflow
