"""
STAGE 2 — PART 2: Metrics + Inverse-RMSE Baseline Ensemble
===========================================================
What this file does:
    1. Defines all bias metrics as clean functions
    2. Computes the Stage 1 baseline metrics for CHIRPS vs IMD
       (to confirm Stage 1 numbers reproduce correctly)
    3. Builds the SIMPLEST possible ensemble — inverse-RMSE weighting.
       This is our non-AI baseline. Every more complex method (BMA,
       XGBoost) must beat this to justify the added complexity.

What is inverse-RMSE weighting?
    The idea is simple: if product A has lower RMSE than product B,
    trust product A more. The weight for each product is:

        weight_A = (1 / RMSE_A) / (1/RMSE_A + 1/RMSE_B + ...)

    Right now we only have CHIRPS, so the "ensemble" is still just
    CHIRPS. When ERA5 and IMERG are added, each gets its own weight
    and the ensemble output is the weighted average of all three.

Why build this now if it is just CHIRPS?
    Because we are building the FRAMEWORK. The code structure is
    identical whether we have 1, 2, or 3 products. Adding ERA5
    and IMERG means changing one list — not rewriting the code.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ── Load the DataFrame from Part 1 ───────────────────────────────────────────
df = pd.read_parquet('/home/claude/stage2/tapi_data.parquet')
print(f"Loaded DataFrame: {df.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: METRIC FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
# Each function takes two numpy arrays (obs = IMD, sim = satellite)
# and returns a single float. NaN pairs are automatically dropped.

def cc(obs, sim):
    """
    Pearson Correlation Coefficient.
    Measures linear agreement between satellite and ground truth.
    Range: -1 to +1. We want > 0.7 for JJAS.
    """
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask], sim[mask]
    if len(o) < 10: return np.nan
    return float(np.corrcoef(o, s)[0, 1])

def rmse(obs, sim):
    """
    Root Mean Square Error in mm/day.
    Penalises large errors more than small ones (squared).
    Lower is better. No maximum — depends on rainfall magnitude.
    """
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask], sim[mask]
    if len(o) < 10: return np.nan
    return float(np.sqrt(np.mean((s - o) ** 2)))

def pbias(obs, sim):
    """
    Percent Bias — the signed volumetric error.
    +ve = satellite overestimates total rainfall
    -ve = satellite underestimates total rainfall
    Formula: 100 × (sum_sim - sum_obs) / sum_obs
    """
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask], sim[mask]
    if len(o) < 10 or o.sum() == 0: return np.nan
    return float(100 * (s.sum() - o.sum()) / o.sum())

def kge(obs, sim):
    """
    Kling-Gupta Efficiency — combines three components:
        r     = correlation (pattern match)
        alpha = std_sim / std_obs (variability ratio)
        beta  = mean_sim / mean_obs (volume ratio)

    KGE = 1 - sqrt((r-1)² + (alpha-1)² + (beta-1)²)
    Optimal = 1. Negative KGE = worse than using mean as prediction.
    """
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask].astype(np.float64), sim[mask].astype(np.float64)
    if len(o) < 10: return np.nan
    r = np.corrcoef(o, s)[0, 1]
    alpha = s.std() / o.std() if o.std() > 0 else np.nan
    beta  = s.mean() / o.mean() if o.mean() > 0 else np.nan
    return float(1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2))

def dpba(obs_ev, sim_ev, obs_ctrl, sim_ctrl):
    """
    Disaster-Period Bias Amplification — OUR NOVEL METRIC.
    Measures how much the satellite bias changes during a flood event
    compared to normal conditions (same calendar window, other years).

    DPBA = PBIAS(event) - PBIAS(control)

    Positive DPBA: bias gets worse (more positive) during the flood
    Negative DPBA: bias flips or reverses during the flood
    Zero DPBA: satellite behaviour is unaffected by the extreme event
    """
    pb_event   = pbias(obs_ev,   sim_ev)
    pb_control = pbias(obs_ctrl, sim_ctrl)
    if np.isnan(pb_event) or np.isnan(pb_control):
        return np.nan
    return round(pb_event - pb_control, 2)

def all_metrics(obs, sim, label=''):
    """
    Computes all metrics in one call. Returns a dict.
    Used for printing summary tables throughout Stage 2.
    """
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask], sim[mask]
    thr = 1.0
    ow, sw = o >= thr, s >= thr
    hits   = np.sum(ow & sw)
    misses = np.sum(ow & ~sw)
    fa     = np.sum(~ow & sw)
    pod = hits / (hits + misses) if (hits + misses) > 0 else np.nan
    far = fa   / (hits + fa)     if (hits + fa)     > 0 else np.nan
    csi = hits / (hits + misses + fa) if (hits + misses + fa) > 0 else np.nan
    return {
        'label': label,
        'N':     int(mask.sum()),
        'CC':    round(cc(obs, sim), 3),
        'RMSE':  round(rmse(obs, sim), 3),
        'PBIAS': round(pbias(obs, sim), 2),
        'KGE':   round(kge(obs, sim), 3),
        'POD':   round(float(pod), 3),
        'FAR':   round(float(far), 3),
        'CSI':   round(float(csi), 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: REPRODUCE STAGE 1 METRICS (SANITY CHECK)
# ══════════════════════════════════════════════════════════════════════════════
# Before building any ensemble, verify the DataFrame gives the same
# numbers as Stage 1 direct computation. If these match, the loader
# is correct.

print("\n── Stage 1 Reproduction Check (CHIRPS vs IMD, JJAS) ──")
jjas = df[df.season == 'JJAS']

rows = []
for zone in ['upper', 'middle', 'lower', 'basin']:
    subset = jjas if zone == 'basin' else jjas[jjas.zone == zone]
    m = all_metrics(subset.imd.values, subset.chirps.values, label=zone)
    rows.append(m)
    print(f"  {zone:8s}: CC={m['CC']:.3f}  RMSE={m['RMSE']:.2f}  "
          f"PBIAS={m['PBIAS']:+.1f}%  KGE={m['KGE']:.3f}")

# Store zone-level RMSE for CHIRPS — needed for inverse-RMSE weights
chirps_rmse_by_zone = {
    r['label']: r['RMSE']
    for r in rows if r['label'] != 'basin'
}
print(f"\nCHIRPS RMSE by zone: {chirps_rmse_by_zone}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: INVERSE-RMSE BASELINE ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════

def compute_inverse_rmse_weights(rmse_dict):
    """
    Given a dict of {product_name: rmse_value}, computes normalised
    inverse-RMSE weights that sum to 1.0.

    Example:
        CHIRPS RMSE = 16.65, ERA5 RMSE = 14.0, IMERG RMSE = 18.0
        inv_weights = [1/16.65, 1/14.0, 1/18.0] = [0.060, 0.071, 0.056]
        normalised  = [0.318, 0.376, 0.296]  (sum = 1.0)

    Right now with only CHIRPS the weight is trivially 1.0,
    but the function is written to handle any number of products.
    """
    products = list(rmse_dict.keys())
    inv = {p: 1.0 / rmse_dict[p] for p in products}
    total = sum(inv.values())
    weights = {p: inv[p] / total for p in products}
    return weights

def apply_ensemble(df_subset, product_cols, weights):
    """
    Applies weighted ensemble to a DataFrame subset.

    product_cols: list of column names e.g. ['chirps', 'era5', 'imerg']
    weights: dict of {col_name: weight}

    Returns a numpy array of ensemble estimates, one per row.
    For rows where a product is NaN, the weight is redistributed
    to the other products (handles missing data gracefully).
    """
    result = np.zeros(len(df_subset))
    weight_sum = np.zeros(len(df_subset))

    for col in product_cols:
        w = weights.get(col, 0)
        vals = df_subset[col].values.astype(np.float64)
        valid = np.isfinite(vals)
        result[valid]     += w * vals[valid]
        weight_sum[valid] += w

    # Normalise by actual weight sum (handles NaN products)
    with np.errstate(invalid='ignore', divide='ignore'):
        result = np.where(weight_sum > 0, result / weight_sum, np.nan)

    return result


# ── Build JJAS weights per zone (seasonal approach) ───────────────────────────
# We compute RMSE separately for each zone in JJAS and use those
# zone-specific weights. This is "terrain-stratified" weighting —
# more weight to the product that performs better in each terrain type.

print("\n── Inverse-RMSE Weights (JJAS, per zone) ──")
zone_weights = {}
for zone in ['upper', 'middle', 'lower']:
    # For now, only CHIRPS — weight is 1.0 trivially
    # When ERA5 and IMERG are added: compute their RMSE here and add to dict
    rmse_dict = {'chirps': chirps_rmse_by_zone[zone]}
    w = compute_inverse_rmse_weights(rmse_dict)
    zone_weights[zone] = w
    print(f"  {zone:8s}: {w}")


# ── Apply ensemble to entire DataFrame ───────────────────────────────────────
# For each row, look up its zone's weights and apply the ensemble.
# This adds a new column 'ensemble' to df.

product_cols = ['chirps']  # Will become ['chirps', 'era5', 'imerg'] later

ensemble_vals = np.full(len(df), np.nan)
for zone in ['upper', 'middle', 'lower']:
    mask = df.zone == zone
    w = zone_weights[zone]
    ensemble_vals[mask] = apply_ensemble(df[mask], product_cols, w)

df['ensemble'] = ensemble_vals

print(f"\nEnsemble column added. Non-NaN: {np.isfinite(df.ensemble).sum()}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: BASELINE PERFORMANCE — CHIRPS vs ENSEMBLE vs IMD
# ══════════════════════════════════════════════════════════════════════════════
# With one product, CHIRPS and ensemble are identical.
# This section is the TEMPLATE for the final comparison table
# — it will show real differences once ERA5 and IMERG are added.

print("\n── Baseline Performance: CHIRPS vs Ensemble (JJAS) ──")
print(f"{'Zone':<10} {'Metric':<8} {'CHIRPS':>10} {'Ensemble':>10} {'Change':>10}")
print("-" * 50)

jjas = df[df.season == 'JJAS'].copy()
for zone in ['upper', 'middle', 'lower', 'basin']:
    sub = jjas if zone == 'basin' else jjas[jjas.zone == zone]
    for metric_name, fn in [('PBIAS', pbias), ('RMSE', rmse), ('CC', cc)]:
        chp_val = fn(sub.imd.values, sub.chirps.values)
        ens_val = fn(sub.imd.values, sub.ensemble.values)
        change  = ens_val - chp_val if not (np.isnan(chp_val) or np.isnan(ens_val)) else np.nan
        print(f"  {zone:<10} {metric_name:<8} {chp_val:>10.3f} {ens_val:>10.3f} {change:>+10.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: DPBA FOR BASELINE ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
# Check whether the inverse-RMSE ensemble changes DPBA during flood events.
# For now (CHIRPS only) DPBA will be identical — the test becomes
# meaningful once multiple products are merged.

print("\n── DPBA: CHIRPS vs Ensemble ──")
import sys; sys.path.insert(0, '/home/claude')
from stage2.part1_loader import FLOOD_EVENTS
import datetime

dpba_rows = []
for event_yr, d0_str, d1_str in FLOOD_EVENTS:
    d0 = datetime.date.fromisoformat(d0_str)
    d1 = datetime.date.fromisoformat(d1_str)

    for zone in ['upper', 'middle', 'lower', 'basin']:
        zone_mask = (df.zone == zone) if zone != 'basin' else pd.Series(True, index=df.index)

        # Event window
        ev_mask   = zone_mask & (df.date >= d0) & (df.date <= d1)
        # Control: same calendar window, all other years
        ctrl_mask = zone_mask & (df.year != event_yr) & \
                    (df.month == d0.month) & \
                    (df.doy >= d0.timetuple().tm_yday) & \
                    (df.doy <= d1.timetuple().tm_yday)

        ev_sub   = df[ev_mask]
        ctrl_sub = df[ctrl_mask]

        if len(ev_sub) < 5 or len(ctrl_sub) < 5:
            continue

        dpba_chp = dpba(ev_sub.imd.values, ev_sub.chirps.values,
                        ctrl_sub.imd.values, ctrl_sub.chirps.values)
        dpba_ens = dpba(ev_sub.imd.values, ev_sub.ensemble.values,
                        ctrl_sub.imd.values, ctrl_sub.ensemble.values)

        dpba_rows.append({
            'event': event_yr, 'zone': zone,
            'DPBA_chirps': dpba_chp, 'DPBA_ensemble': dpba_ens,
            'improvement': round(abs(dpba_chp) - abs(dpba_ens), 2)
                           if not (np.isnan(dpba_chp) or np.isnan(dpba_ens)) else np.nan
        })
        print(f"  {event_yr} {zone:8s}: CHIRPS DPBA={dpba_chp:+.1f}%  "
              f"Ensemble DPBA={dpba_ens:+.1f}%  "
              f"Improvement={abs(dpba_chp)-abs(dpba_ens):+.1f}%")

dpba_df = pd.DataFrame(dpba_rows)


# ── Save outputs ──────────────────────────────────────────────────────────────
os.makedirs('/home/claude/stage2', exist_ok=True)
df.to_parquet('/home/claude/stage2/tapi_data_with_ensemble.parquet', index=False)
dpba_df.to_csv('/home/claude/stage2/dpba_baseline.csv', index=False)
print(f"\nSaved: tapi_data_with_ensemble.parquet, dpba_baseline.csv")
print("Part 2 complete. Ready for Part 3 (BMA).")
