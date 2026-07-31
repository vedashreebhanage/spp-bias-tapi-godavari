# Terrain-Stratified SPP Bias During Flood Events and Ensemble Correction

**A Tapi and Godavari Case Study**

Evaluation of satellite precipitation products (IMERG V07, CHIRPS v2.0, ERA5-Land) against IMD gridded rainfall over two peninsular Indian river basins, with a novel metric — **DPBA (Disaster-Period Bias Amplification)** — quantifying how satellite bias changes during documented flood events. Stage 2 uses DPBA-informed ensemble learning (BMA + XGBoost) to correct for this amplification.

Target journal: *Journal of Hydrology: Regional Studies* (Q1, IF 4.7)

---

## Project Structure

```
spp-bias-tapi-godavari/
│
├── README.md                          <- this file
├── requirements.txt                   <- Python dependencies
├── .gitignore                         <- excludes raw data and large files
│
├── data/
│   ├── raw/                           <- NOT tracked by git (see .gitignore)
│   │   ├── imd/                       <- RF{year}.GRD files (2015-2023)
│   │   ├── chirps/                    <- chirps_tapi_{year}.nc
│   │   ├── era5/                      <- era5_tapi_{year}.nc
│   │   ├── imerg/                     <- IMERG daily granules
│   │   └── shapefiles/                <- HydroSHEDS basin polygons
│   └── processed/
│       ├── tapi_data.parquet          <- flat merged dataframe (Stage 2 input)
│       └── basin_masks/               <- .npy grid masks per basin
│
├── src/
│   ├── stage1/
│   │   ├── readers.py                 <- IMD, CHIRPS, ERA5, IMERG file readers
│   │   ├── regrid.py                  <- regridding to common 0.25° grid
│   │   ├── metrics.py                 <- CC, RMSE, PBIAS, KGE, POD, FAR, CSI, DPBA
│   │   └── run_stage1.py              <- full bias matrix + DPBA computation
│   │
│   └── stage2/
│       ├── part1_loader.py            <- builds flat DataFrame from raw sources
│       ├── part2_baseline.py          <- inverse-RMSE weighted ensemble
│       ├── part3_bma.py               <- Bayesian Model Averaging
│       ├── part4_xgboost.py           <- DPBA-adaptive XGBoost ensemble
│       └── significance.py            <- bootstrap significance testing for DPBA
│
├── notebooks/
│   └── exploratory/                   <- scratch analysis, not for production
│
├── results/
│   ├── figures/                       <- all generated PNG figures
│   ├── tables/                        <- CSV output (bias matrix, DPBA results)
│   └── stage2_outputs/                <- ensemble comparison tables
│
├── paper/
│   ├── manuscript.md                  <- working draft (or .tex if using LaTeX)
│   ├── literature_review.md           <- annotated bibliography (25 papers)
│   └── figures_for_paper/             <- final polished versions for submission
│
└── presentation/
    └── SPP_Bias_Presentation.pptx     <- project overview slides
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/spp-bias-tapi-godavari.git
cd spp-bias-tapi-godavari
pip install -r requirements.txt
```

### Data Acquisition

Raw data is **not included in this repository** due to size and licensing. Download instructions:

| Dataset | Source | Notes |
|---|---|---|
| IMD 0.25° Gridded Rainfall | [IMD Pune](https://www.imdpune.gov.in/) | Registration required |
| CHIRPS v2.0 | [UCSB Climate Hazards Center](https://data.chc.ucsb.edu/products/CHIRPS-2.0/) | Public FTP |
| ERA5-Land | [Copernicus CDS](https://cds.climate.copernicus.eu/) | Free API key required |
| IMERG V07 | [NASA GES DISC](https://disc.gsfc.nasa.gov/) | Free Earthdata account required |
| HydroBASINS shapefiles | [HydroSHEDS](https://www.hydrosheds.org/products/hydrobasins) | Public download |

Place downloaded files into `data/raw/` following the structure above.

---

## Usage

```bash
# Step 1: Run Stage 1 — bias evaluation
python src/stage1/run_stage1.py

# Step 2: Build the flat dataframe for Stage 2
python src/stage2/part1_loader.py

# Step 3: Run baseline ensemble
python src/stage2/part2_baseline.py

# Step 4: Run BMA
python src/stage2/part3_bma.py

# Step 5: Run DPBA-adaptive XGBoost ensemble
python src/stage2/part4_xgboost.py
```

Each script reads from `data/processed/` and writes results to `results/`.

---

## Key Outputs

- **`results/tables/stage1_bias_matrix.csv`** — CC, RMSE, PBIAS, KGE, POD, FAR, CSI by product × zone × season
- **`results/tables/dpba_results.csv`** — Disaster-Period Bias Amplification for all flood events
- **`results/figures/fig1_pbias_spatial.png`** — spatial bias maps
- **`results/figures/fig2_metrics_terrain.png`** — terrain-stratified metric curves
- **`results/figures/fig3_dpba.png`** — DPBA comparison across flood events

---

## Methodology Summary

**Stage 1 — Diagnostic Evaluation**
Classical statistical comparison of three SPPs against IMD ground truth, decomposed by sub-basin terrain (upper/middle/lower) and season (JJAS/ON/DJF/MAM). Introduces DPBA to quantify bias change during flood events relative to climatological control windows.

**Stage 2 — AI-Driven Correction**
Uses Stage 1 diagnostic findings to inform an ensemble merging framework:
- Inverse-RMSE weighting (baseline)
- Bayesian Model Averaging (probabilistic ensemble)
- DPBA-adaptive XGBoost (learns flood-critical periods, adjusts weights dynamically)

Validation: does the AI ensemble reduce DPBA during flood events compared to any individual product?

---

## Study Basins

| Basin | Area | Flood Events Analyzed |
|---|---|---|
| Tapi | 65,145 km² | Aug 2021, Sep 2023 (CWC-verified HFL breaches) |
| Godavari | 312,812 km² | Aug 2022 |

---

## Citation

If you use this work, please cite:

```
[Author names]. "Terrain-Stratified SPP Bias During Flood Events and Ensemble
Correction: A Tapi and Godavari Case Study." [Journal, Year, In Review]
```

---

## License

Code: MIT License. Data: subject to original source licensing (IMD, CHIRPS, ERA5, IMERG terms of use apply).

---

## Acknowledgments

IntelliPair Team, Vishwakarma Institute of Technology (VIT), Pune — FY-DESH CSSE Program
