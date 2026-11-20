# Terrain-Stratified Satellite Precipitation Bias in the Tapi Basin

Evaluation of satellite and reanalysis precipitation products against IMD gridded
rainfall over the Tapi basin, with attention to how bias behaves during documented
flood events.

> **Status: Stage 1 is being rebuilt from scratch.**
> An audit of the first implementation found methodological errors serious enough that
> its results should not be used. That code and its outputs are preserved in
> [`archive/ai_v1/`](archive/ai_v1/) for comparison, not as findings.
> See [Why the rebuild](#why-the-rebuild).

---

## What is in this repository right now

```
.
├── scripts/              data acquisition + verification (current, working)
├── docs/                 acquisition plan, drafts, reference papers
├── archive/ai_v1/        superseded first implementation — do not build on this
├── data/                 raw + processed data (gitignored, not distributed)
├── src/common/           readers, area-weighted regridding, aligned-dataset builder
├── src/stage1/           basin mask, episodes, metrics, block bootstrap, audit steps
└── tests/                unit tests (pytest) for readers, regrid, metrics, bootstrap
```

Nothing in `archive/ai_v1/` should be cited or reused. It is kept because comparing
old and corrected numbers is itself informative, and because the failure modes are
worth documenting.

---

## Why the rebuild

An audit of the original Stage 1 implementation found the following. Each is a real
defect, not a stylistic preference:

| # | Problem | Consequence |
|---|---|---|
| 1 | **No basin mask.** Metrics were computed over a rectangular bounding box (20–22.5 °N, 72.5–78.5 °E ≈ 165,000 km²) rather than the Tapi polygon (65,145 km²). The HydroBASINS shapefile was downloaded but never used. | ~60% of reported "Tapi" statistics came from outside the basin. |
| 2 | **Terrain stratification was longitude slicing.** "Upper/middle/lower" were defined by two longitude thresholds, with no elevation data involved. | The central claim — that bias varies with terrain — was not actually tested. |
| 3 | **Extreme rainfall was discarded.** The IMD reader set any value > 500 mm/day to NaN. | Self-defeating in a study about bias during floods; IMD records do exceed this. |
| 4 | **Sample size overstated.** N was reported as day-cells (~880,000) with no account of spatial or temporal autocorrelation. | Implied precision far beyond what ~90 grid cells over 9 years supports. |
| 5 | **DPBA was ill-defined.** Computed as signed `PBIAS_event − PBIAS_control`, so a bias that *improved* during a flood could register as "amplification." | The headline metric did not measure what its name claims. |
| 6 | **Stage 1 code did not exist.** Results were produced in an ephemeral session and never saved; only Stage 2 scripts were committed. | The published numbers were unreproducible. |
| 7 | **Only one product, one basin.** Despite claims of three products across two basins, only CHIRPS over Tapi was ever analysed. | README described work that had not been done. |

The arithmetic in the original pipeline was internally consistent — figures matched
tables, zone counts summed correctly. It was consistently computing the wrong thing,
which is exactly the failure mode that survives casual review.

---

## Corrected approach

**Domain.** All metrics computed on grid cells whose centres fall inside the
HydroBASINS Tapi polygon. Cell count and implied basin area are reported and checked
against the known 65,145 km² before any analysis runs.

**Terrain.** Mean elevation per 0.25° cell derived from SRTM 90 m. Bias is examined as
a *continuous* function of elevation (with bootstrap confidence intervals) rather than
binned into three zones — this uses every cell, avoids arbitrary thresholds, and
yields a testable slope. Discrete zones are retained only as a descriptive summary.

**Uncertainty.** Confidence intervals from block bootstrap over contiguous time blocks
and spatial clusters, not from raw day-cell counts. Grid-cell counts reported
alongside every N.

**Event windows.** No hand-picked flood dates. Episodes are *derived* from the
reference dataset by a rule fixed in `src/stage1/define_episodes.py`: the ten
wettest non-overlapping 3-day windows by basin-mean IMD rainfall, at least 7 days
apart. They are called extreme **rainfall** episodes, not floods — a flood is a
river-stage phenomenon and Ukai's regulation partly decouples basin rainfall from
stage at Surat, whereas rainfall is what the satellite products actually observe.
Where an episode coincides with an independently documented flood, that is recorded
as an annotation and never as part of the selection. The control-day rule (JJAS,
basin mean ≥ 1 mm, excluding episodes ±3 days) is fixed in the same module *before*
results are inspected.

**Amplification metric.** Redefined in terms of bias *magnitude* so that a bias moving
toward zero can never register as amplification, with significance tested rather than
assumed.

**Reference data caveat.** IMD gridded rainfall is interpolated from a sparse gauge
network. Over the Satpura terrain it is not ground truth, and part of any measured
"satellite bias" is IMD's own interpolation error. This is stated as a limitation
rather than glossed over.

---

## Data

Raw data is not distributed here. Sizes and licensing prohibit it; `data/` is
gitignored.

| Dataset | Role | Resolution | Source | Status |
|---|---|---|---|---|
| IMD gridded rainfall | Reference | 0.25° daily | [IMD Pune](https://www.imdpune.gov.in/) | Complete, sizes verified |
| CHIRPS v2.0 | Product under test | 0.05° daily | [UCSB CHC](https://data.chc.ucsb.edu/products/CHIRPS-2.0/) | Complete, verified |
| ERA5-Land | Product under test | ~0.1° daily | [Copernicus CDS](https://cds.climate.copernicus.eu/) | Complete (+ 2024-01-01 tail file) |
| IMERG V07 | Product under test | 0.1° daily | [NASA GES DISC](https://disc.gsfc.nasa.gov/) | Complete, 3,287 granules |
| SRTM (SRTMGL3) | Terrain | 90 m | [OpenTopography](https://portal.opentopography.org/) | Complete, verified |
| HydroBASINS | Basin mask | vector | [HydroSHEDS](https://www.hydrosheds.org/) | Present |

Study period: 2015-01-01 to 2023-12-30 (3,286 days; see the IMD note below for why
the last day is 30 Dec).

### Units — mismatches that fail silently

| Product | Native unit | Handling |
|---|---|---|
| IMD | mm/day | none |
| CHIRPS | mm/day | none |
| ERA5-Land | **metres** | ×1000 at analysis time |
| IMERG V07 | mm/day | V06 was mm/hr — V06 code is not reusable |

ERA5 accumulates in UTC; the IMD rain day runs 0830–0830 IST and is **labelled by
the day it ends**, so IMD's value for date D is rain that fell mostly on D−1. The
reader relabels IMD one day back, and `build_dataset.py` fails unless every product
lines up with IMD at lag 0. The remaining ~3 h offset (0830 IST = 0300 UTC) is
stated as a limitation.

---

## Setup

```bash
pip install -r scripts/requirements.txt
```

Three free accounts are needed, each with one non-obvious step:

| Service | For | Required step people miss |
|---|---|---|
| [OpenTopography](https://portal.opentopography.org/) | DEM | key goes in `OPENTOPO_API_KEY`; restart your editor after `setx` |
| [Copernicus CDS](https://cds.climate.copernicus.eu/) | ERA5-Land | accept the licence on the dataset page, or every request returns 403 |
| [NASA Earthdata](https://urs.earthdata.nasa.gov/) | IMERG | authorize "NASA GESDISC DATA ARCHIVE" in your profile, or every request returns 401 |

Credentials stay out of version control: `OPENTOPO_API_KEY` in the environment,
`.cdsapirc` and `.env` gitignored.

---

## Usage

Run everything **from the repository root**.

```bash
python scripts/check_status.py                  # inventory: what exists, what is missing
python scripts/verify_chirps.py                 # structural check, downloads nothing
python scripts/download_imd.py --verify-only    # size check against expected byte count
python scripts/download_dem.py                  # add --tiles 4 if a single request fails
python scripts/download_era5.py --clean
python scripts/download_imerg.py --show-settings --events-only
python scripts/download_imerg.py                # after generating imerg_urls.txt

python src/stage1/build_mask.py                 # basin mask + terrain per cell
python src/stage1/define_episodes.py            # extreme rainfall episodes + control days
python src/common/build_dataset.py              # align all products on the IMD grid (slow: IMERG)
python src/stage1/audit_step1_overall.py        # overall bias with bootstrap CIs
python src/stage1/audit_step2_season_zone.py    # bias by season x elevation zone
python -m pytest tests                          # unit tests
```

Every script accepts `--basin tapi` (default) or `--basin godavari`.

`scripts/basins.py` is the single source of truth for bounding boxes, study years, and
flood windows. Every other script imports from it. Previously each script carried its
own hardcoded box; when those drift apart the products silently cover different areas
and every comparison between them is wrong, with no error raised.

---

## Roadmap

- [x] Audit the original implementation
- [x] Reproducible acquisition + verification tooling
- [x] IMD reference data acquired and verified
- [x] Remaining products acquired (CHIRPS, ERA5, IMERG, DEM)
- [x] Basin mask built and validated against known basin area (92 cells, +1.8%)
- [x] Elevation zones derived from DEM (terciles)
- [x] Extreme rainfall episodes derived by a fixed rule (replaces hand-picked flood windows)
- [x] All products aligned on the IMD grid, with automated calendar-alignment check
- [x] Metrics and block-bootstrap modules with unit tests
- [x] Audit step 1: overall basin-wide bias, all products, with CIs
- [x] Audit step 2: bias by season × elevation zone (descriptive; shared-draw CIs)
- [ ] Audit step 3: bias as a continuous function of elevation
- [ ] Audit step 4: episode amplification (|PBIAS| episode vs control), significance-tested
- [ ] Godavari added as a replication basin
- [ ] Stage 2: correction methods benchmarked against quantile mapping

Stage 2 will not begin until Stage 1 is reproducible and verified. The original Stage 2
scaffolding (BMA, XGBoost ensembles) inherited Stage 1's domain error and is archived
accordingly.

---

## Acknowledgments

IntelliPair Team, Vishwakarma Institute of Technology, Pune — FY-DESH CSSE Program.

## License

Code: MIT. Data subject to the terms of its original providers (IMD, CHC/UCSB,
Copernicus, NASA, HydroSHEDS).
