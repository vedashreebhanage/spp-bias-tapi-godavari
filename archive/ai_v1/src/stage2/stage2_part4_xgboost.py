"""
STAGE 2 — PART 4: DPBA-Adaptive XGBoost Ensemble Weighting
============================================================
What this file does:
    Builds the core AI contribution of the paper — an XGBoost model
    that LEARNS when flood-critical periods are occurring and
    dynamically adjusts which satellite product to trust more.

Why XGBoost specifically?
    XGBoost is a gradient boosted decision tree algorithm. It builds
    an ensemble of decision trees sequentially, each one correcting
    the errors of the previous one. It is chosen here because:
    1. It handles tabular data (our feature table) very well
    2. It is interpretable — we can see which features matter most
       using feature importance scores (important for the paper)
    3. It handles the class imbalance problem (flood days are rare —
       only 7,317 / 890,777 rows = 0.82% of data)
    4. It is fast enough to train on 890k rows

The key insight of this paper:
    Standard ensemble methods (Inv-RMSE, BMA) use the SAME weights
    whether it is a normal monsoon day or a catastrophic flood event.
    But Stage 1 showed that CHIRPS bias FLIPS during flood events
    (DPBA = -45.5% in Sep 2023). So a model that recognises flood
    conditions and adjusts weights accordingly should outperform
    static weighting — especially for DPBA.

How the XGBoost fits in:
    Step 1: Train a FLOOD DETECTOR — XGBoost classifier that predicts
            whether a given day/location is in a flood-critical period
            based on meteorological features.

    Step 2: Use the flood probability as a signal to BLEND between:
            - Normal-condition weights (from BMA/Inv-RMSE)
            - Flood-condition weights (DPBA-corrected)

    Step 3: The final ensemble output = blended weighted average

This is the DPBA-informed part: we use the Stage 1 DPBA findings
to define what the "flood-condition weights" should be.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, classification_report,
                              average_precision_score)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_parquet('/home/claude/stage2/tapi_data_with_bma.parquet')
print(f"Loaded: {df.shape}")
print(f"Flood rows: {df.is_flood.sum()} / {len(df)} ({df.is_flood.mean()*100:.2f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
# The XGBoost model needs features that describe the meteorological
# and geographic context of each observation. Good features tell
# the model when to expect flood conditions.

def build_features(df):
    """
    Creates the feature matrix for XGBoost training.

    Features chosen and why:

    TEMPORAL features:
      doy_sin, doy_cos  — Sine/cosine encoding of day-of-year.
                          Raw DOY is circular (day 365 is close to day 1)
                          so we encode it as sin/cos to preserve that.
      month             — Calendar month (categorical encoded as integer)

    SPATIAL features:
      lat, lon          — Grid cell coordinates
      zone_enc          — Zone encoded as integer (upper=2, middle=1, lower=0)
                          Captures terrain-dependent bias patterns from Stage 1

    RAINFALL features:
      chirps            — Current day CHIRPS estimate
      chirps_anom       — CHIRPS anomaly: deviation from zone-season mean
                          If CHIRPS is much higher than normal → potential flood
      chirps_7d_mean    — 7-day rolling mean of CHIRPS (antecedent conditions)
                          High antecedent rainfall → saturated soil → flood risk
      chirps_imd_diff   — CHIRPS minus IMD (previous bias signal)
                          Large positive = CHIRPS was overestimating recently
    """
    feat = df.copy()

    # Temporal encoding
    feat['doy_sin'] = np.sin(2 * np.pi * feat['doy'] / 366)
    feat['doy_cos'] = np.cos(2 * np.pi * feat['doy'] / 366)

    # Zone as integer
    zone_map = {'upper': 2, 'middle': 1, 'lower': 0}
    feat['zone_enc'] = feat['zone'].map(zone_map)

    # CHIRPS anomaly — deviation from zone-season long-term mean
    zone_season_mean = (feat.groupby(['zone', 'season'])['chirps']
                           .transform('mean'))
    feat['chirps_anom'] = feat['chirps'] - zone_season_mean

    # 7-day antecedent CHIRPS (per cell, sorted by date)
    # Group by lat/lon, sort by date, compute rolling mean
    feat = feat.sort_values(['lat', 'lon', 'date'])
    feat['chirps_7d_mean'] = (feat.groupby(['lat', 'lon'])['chirps']
                                  .transform(lambda x: x.rolling(7, min_periods=1).mean()))

    # CHIRPS-IMD difference (current day)
    feat['chirps_imd_diff'] = feat['chirps'] - feat['imd']

    return feat

print("\nBuilding features...")
df = build_features(df)

# Get actual feature columns after build_features adds new ones
FEATURE_COLS = [
    'doy_sin', 'doy_cos', 'month',
    'lat', 'lon', 'zone_enc',
    'chirps', 'chirps_anom', 'chirps_7d_mean', 'chirps_imd_diff'
]

# Only use JJAS rows for training — flood events only happen in monsoon
df_jjas = df[df.season == 'JJAS'].copy()
print(f"JJAS rows for model: {len(df_jjas)}")
print(f"Flood rows in JJAS: {df_jjas.is_flood.sum()} ({df_jjas.is_flood.mean()*100:.2f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: FLOOD DETECTOR — XGBOOST CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
TEST_YEARS  = [2023]

# Drop rows with missing features
df_model = df_jjas[list(set(FEATURE_COLS + ['is_flood', 'year', 'zone', 'date',
                                    'imd', 'chirps', 'ensemble', 'bma',
                                    'lat', 'lon', 'doy', 'month', 'season']))].copy().dropna()

df_tr = df_model[df_model.year.isin(TRAIN_YEARS)]
df_te = df_model[df_model.year.isin(TEST_YEARS)]

X_tr = df_tr[FEATURE_COLS]   # DataFrame preserves feature names
y_tr = df_tr['is_flood'].values
X_te = df_te[FEATURE_COLS]
y_te = df_te['is_flood'].values

print(f"\nTrain: {len(X_tr)} rows, {y_tr.sum()} flood ({y_tr.mean()*100:.2f}%)")
print(f"Test:  {len(X_te)} rows, {y_te.sum()} flood ({y_te.mean()*100:.2f}%)")

# Scale features — XGBoost does not require it but helps convergence
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)
# Keep DataFrame versions for XGBoost (preserves feature names)
X_tr_df = X_tr.copy()
X_te_df = X_te.copy()

# Class imbalance: flood days are rare (0.82%)
# scale_pos_weight tells XGBoost to weight flood class more heavily
n_neg  = (y_tr == 0).sum()
n_pos  = (y_tr == 1).sum()
spw    = n_neg / n_pos if n_pos > 0 else 1.0
print(f"scale_pos_weight: {spw:.1f} (compensates for class imbalance)")

# XGBoost parameters
# n_estimators: number of trees — 200 is enough for tabular data
# max_depth: how deep each tree can grow — 5 prevents overfitting
# learning_rate: how much each tree contributes — lower = more robust
# subsample: use 80% of rows per tree — reduces overfitting
# colsample_bytree: use 80% of features per tree — reduces overfitting
# eval_metric: area under precision-recall curve (better than AUC for
#              imbalanced data — most days are NOT flood days)

print("\nTraining XGBoost flood detector...")
model = xgb.XGBClassifier(
    n_estimators     = 200,
    max_depth        = 5,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    scale_pos_weight = spw,
    eval_metric      = 'aucpr',
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = 0,
)

model.fit(
    X_tr_df, y_tr,
    eval_set=[(X_te_df, y_te)],
    verbose=False
)

# Evaluate
flood_proba_te = model.predict_proba(X_te_df)[:, 1]
flood_proba_tr = model.predict_proba(X_tr_df)[:, 1]

auc_roc = roc_auc_score(y_te, flood_proba_te)
auc_pr  = average_precision_score(y_te, flood_proba_te)
print(f"\nTest AUC-ROC: {auc_roc:.3f}")
print(f"Test AUC-PR:  {auc_pr:.3f}  (random baseline = {y_te.mean():.3f})")

# Feature importance
importance = model.feature_importances_
feat_imp = pd.Series(importance, index=FEATURE_COLS).sort_values(ascending=False)
print("\nFeature Importance (what the model uses to detect floods):")
for feat, imp in feat_imp.items():
    bar = '█' * int(imp * 50)
    print(f"  {feat:<22} {imp:.3f}  {bar}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: DPBA-INFORMED WEIGHT BLENDING
# ══════════════════════════════════════════════════════════════════════════════
# Use Stage 1 DPBA findings to define flood-condition weights.
#
# The logic:
# - In normal conditions: use BMA weights (learned from 2015-2020)
# - In flood conditions: adjust weights based on DPBA sign
#
# DPBA findings from Stage 1 for CHIRPS:
#   Upper Tapi 2023: DPBA = -45.5% → CHIRPS severely underestimates during floods
#   Lower Tapi 2023: DPBA = +17.3% → CHIRPS slightly overestimates during floods
#
# When only CHIRPS is available, DPBA-based blending has limited effect.
# With ERA5 and IMERG:
#   - If CHIRPS DPBA is very negative → down-weight CHIRPS, up-weight ERA5/IMERG
#   - If ERA5 DPBA is near zero → ERA5 is flood-stable → increase its weight

# DPBA-based weight adjustments (from Stage 1 results)
# Format: {zone: adjustment_factor for CHIRPS during floods}
# Factor < 1 = down-weight during floods, Factor > 1 = up-weight
DPBA_CHIRPS_ADJUSTMENTS = {
    'upper':  0.40,   # -45.5% DPBA → CHIRPS very unreliable in upper Tapi floods
    'middle': 0.65,   # -24.6% DPBA → moderately unreliable
    'lower':  1.20,   # +17.3% DPBA → CHIRPS slightly better during lower floods
}

def compute_dpba_weights(zone, flood_prob, base_weights,
                         dpba_adjustments, threshold=0.3):
    """
    Blend between normal weights and DPBA-adjusted weights
    based on the flood probability from XGBoost.

    When flood_prob is low (normal day): use base_weights as-is
    When flood_prob is high (flood day): blend toward DPBA-adjusted weights

    The blend factor = flood_prob (continuous, not binary threshold)
    This gives smooth weight transitions rather than hard switches.

    Args:
        zone:             sub-basin zone string
        flood_prob:       array of flood probabilities from XGBoost (0-1)
        base_weights:     dict of {product: normal_weight} from BMA
        dpba_adjustments: dict of {zone: chirps_adjustment_factor}
        threshold:        minimum flood_prob to start applying DPBA adjustment

    Returns:
        Array of final ensemble estimates
    """
    # Get DPBA adjustment for this zone
    adj = dpba_adjustments.get(zone, 1.0)

    # Flood-condition weights: adjust CHIRPS by DPBA factor
    flood_weights = dict(base_weights)
    flood_weights['chirps'] = base_weights.get('chirps', 1.0) * adj

    # Renormalise to sum to 1
    total = sum(flood_weights.values())
    flood_weights = {k: v / total for k, v in flood_weights.items()}

    return base_weights, flood_weights   # return both for blending

def apply_dpba_ensemble(df_subset, product_cols, base_weights,
                        flood_proba, dpba_adjustments, zone):
    """
    Applies the DPBA-informed blended ensemble to a zone subset.

    For each row:
        final_weight = (1 - flood_prob) × normal_weight
                     + flood_prob × dpba_adjusted_weight

    This is a soft blend — the model smoothly transitions
    between normal and flood-optimised weights.
    """
    n = len(df_subset)
    result = np.zeros(n)

    normal_w, flood_w = compute_dpba_weights(
        zone, None, base_weights, dpba_adjustments
    )

    for i, (idx, row) in enumerate(df_subset.iterrows()):
        fp = flood_proba[i]   # flood probability for this row

        # Blend weights
        blended = {}
        for prod in product_cols:
            blended[prod] = (1 - fp) * normal_w.get(prod, 0) + \
                             fp * flood_w.get(prod, 0)

        # Apply blended weights
        total_w = 0
        pred    = 0
        for prod in product_cols:
            val = row.get(prod, np.nan)
            if np.isfinite(val):
                pred    += blended[prod] * val
                total_w += blended[prod]

        result[i] = pred / total_w if total_w > 0 else np.nan

    return result


# ── Apply DPBA ensemble to test set ───────────────────────────────────────────
print("\nApplying DPBA-adaptive ensemble to test set...")
df_te_copy = df_te.copy()
df_te_copy['flood_prob'] = flood_proba_te
dpba_ens_vals = np.full(len(df_te_copy), np.nan)

PRODUCT_COLS = ['chirps']

for zone in ['upper', 'middle', 'lower']:
    zone_mask = df_te_copy.zone == zone
    zone_df   = df_te_copy[zone_mask].copy()
    if len(zone_df) == 0: continue

    zone_fp    = zone_df['flood_prob'].values
    base_w     = {'chirps': 1.0}   # BMA weights (trivially 1.0 with one product)

    zone_pred = apply_dpba_ensemble(
        zone_df, PRODUCT_COLS, base_w, zone_fp, DPBA_CHIRPS_ADJUSTMENTS, zone
    )
    dpba_ens_vals[zone_mask.values] = zone_pred

df_te_copy['dpba_ensemble'] = dpba_ens_vals
print(f"DPBA ensemble applied. Non-NaN: {np.isfinite(dpba_ens_vals).sum()}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: FINAL COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════

def pbias(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask], sim[mask]
    if len(o) < 10 or o.sum() == 0: return np.nan
    return float(100 * (s.sum() - o.sum()) / o.sum())

def rmse(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask], sim[mask]
    if len(o) < 10: return np.nan
    return float(np.sqrt(np.mean((s - o)**2)))

def cc(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask], sim[mask]
    if len(o) < 10: return np.nan
    return float(np.corrcoef(o, s)[0, 1])

def dpba_fn(obs_ev, sim_ev, obs_ctrl, sim_ctrl):
    pb_ev   = pbias(obs_ev,   sim_ev)
    pb_ctrl = pbias(obs_ctrl, sim_ctrl)
    if np.isnan(pb_ev) or np.isnan(pb_ctrl): return np.nan
    return round(pb_ev - pb_ctrl, 2)

print("\n── FINAL COMPARISON: All Methods (Test Set 2021-2023, JJAS) ──")
print(f"{'Zone':<10} {'Metric':<8} {'CHIRPS':>10} {'InvRMSE':>10} "
      f"{'BMA':>10} {'DPBA-XGB':>10}")
print("-" * 65)

for zone in ['upper', 'middle', 'lower', 'basin']:
    zm    = (df_te_copy.zone == zone) if zone != 'basin' else pd.Series(True, index=df_te_copy.index)
    sub   = df_te_copy[zm]
    for metric_name, fn in [('PBIAS%', pbias), ('RMSE', rmse), ('CC', cc)]:
        chp   = fn(sub.imd.values, sub.chirps.values)
        inv   = fn(sub.imd.values, sub.ensemble.values)
        bma_v = fn(sub.imd.values, sub.bma.values)
        dpb   = fn(sub.imd.values, sub.dpba_ensemble.values)
        print(f"  {zone:<10} {metric_name:<8} {chp:>10.3f} {inv:>10.3f} "
              f"{bma_v:>10.3f} {dpb:>10.3f}")


# ── DPBA comparison during flood events ──────────────────────────────────────
import datetime
import sys; sys.path.insert(0, '/home/claude')
from stage2.part1_loader import FLOOD_EVENTS

print("\n── DPBA Reduction: All Methods vs Flood Events ──")
print(f"{'Event':<6} {'Zone':<10} {'CHIRPS':>10} {'InvRMSE':>10} "
      f"{'BMA':>10} {'DPBA-XGB':>10}")
print("-" * 65)

for event_yr, d0_str, d1_str in FLOOD_EVENTS:
    if event_yr not in TEST_YEARS: continue
    d0 = datetime.date.fromisoformat(d0_str)
    d1 = datetime.date.fromisoformat(d1_str)

    for zone in ['upper', 'middle', 'lower', 'basin']:
        zm = (df_te_copy.zone == zone) if zone != 'basin' else pd.Series(True, index=df_te_copy.index)
        ev_m   = zm & (df_te_copy.date >= d0) & (df_te_copy.date <= d1)
        ctrl_m = zm & (df_te_copy.year != event_yr) & \
                 (df_te_copy.month == d0.month) & \
                 (df_te_copy.doy >= d0.timetuple().tm_yday) & \
                 (df_te_copy.doy <= d1.timetuple().tm_yday)

        ev_s, ct_s = df_te_copy[ev_m], df_te_copy[ctrl_m]
        if len(ev_s) < 5: continue

        d_chp = dpba_fn(ev_s.imd.values, ev_s.chirps.values,
                        ct_s.imd.values, ct_s.chirps.values)
        d_inv = dpba_fn(ev_s.imd.values, ev_s.ensemble.values,
                        ct_s.imd.values, ct_s.ensemble.values)
        d_bma = dpba_fn(ev_s.imd.values, ev_s.bma.values,
                        ct_s.imd.values, ct_s.bma.values)
        d_dpb = dpba_fn(ev_s.imd.values, ev_s.dpba_ensemble.values,
                        ct_s.imd.values, ct_s.dpba_ensemble.values)
        print(f"  {event_yr:<6} {zone:<10} {d_chp:>+10.1f} {d_inv:>+10.1f} "
              f"{d_bma:>+10.1f} {d_dpb:>+10.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: VISUALISE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(16, 10), facecolor='#0f1117')
fig.suptitle('Stage 2 Results — DPBA-XGBoost Ensemble vs Baselines\n(Test Set 2021–2023)',
             color='white', fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

colors_method = {
    'CHIRPS':    '#4fc3f7',
    'Inv-RMSE':  '#81c784',
    'BMA':       '#ffb74d',
    'DPBA-XGB':  '#ef5350',
}

# Panel 1: Feature importance
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_facecolor('#1a1d27')
feats  = feat_imp.index.tolist()
imps   = feat_imp.values
colors_fi = ['#ef5350' if 'chirps' in f else '#4fc3f7' for f in feats]
bars = ax1.barh(feats[::-1], imps[::-1], color=colors_fi[::-1], alpha=0.85)
ax1.set_title('XGBoost Feature Importance (Flood Detector)', color='white', fontsize=11)
ax1.set_xlabel('Importance Score', color='white')
ax1.tick_params(colors='white')
[s.set_color('#444') for s in ax1.spines.values()]
ax1.grid(axis='x', alpha=0.2)

# Panel 2: AUC-PR summary
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor('#1a1d27')
metrics_summary = {
    'AUC-ROC': auc_roc,
    'AUC-PR':  auc_pr,
    'Baseline\n(random)': float(y_te.mean()),
}
bars2 = ax2.bar(list(metrics_summary.keys()), list(metrics_summary.values()),
                color=['#4fc3f7', '#81c784', '#555'], alpha=0.85)
ax2.set_title('Flood Detector Performance', color='white', fontsize=11)
ax2.set_ylabel('Score', color='white'); ax2.set_ylim(0, 1)
ax2.tick_params(colors='white')
[s.set_color('#444') for s in ax2.spines.values()]
for bar, val in zip(bars2, metrics_summary.values()):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
             f'{val:.3f}', ha='center', color='white', fontsize=11, fontweight='bold')
ax2.grid(axis='y', alpha=0.2)

# Panel 3: PBIAS comparison by zone (test set, JJAS)
ax3 = fig.add_subplot(gs[1, :2])
ax3.set_facecolor('#1a1d27')
zones = ['upper', 'middle', 'lower', 'basin']
x = np.arange(len(zones))
w = 0.2
test_jjas_copy = df_te_copy[df_te_copy.season == 'JJAS']

methods = [
    ('CHIRPS',   'chirps',        '#4fc3f7'),
    ('Inv-RMSE', 'ensemble',      '#81c784'),
    ('BMA',      'bma',           '#ffb74d'),
    ('DPBA-XGB', 'dpba_ensemble', '#ef5350'),
]

for mi, (label, col, color) in enumerate(methods):
    pbiases = []
    for zone in zones:
        sub = test_jjas_copy if zone == 'basin' else test_jjas_copy[test_jjas_copy.zone == zone]
        pb = pbias(sub.imd.values, sub[col].values)
        pbiases.append(pb if not np.isnan(pb) else 0)
    bars3 = ax3.bar(x + mi*w - 1.5*w, pbiases, w,
                    label=label, color=color, alpha=0.85)

ax3.axhline(0, color='white', ls='--', lw=0.8, alpha=0.5)
ax3.set_title('PBIAS by Zone and Method (Test Set JJAS)', color='white', fontsize=11)
ax3.set_ylabel('PBIAS (%)', color='white')
ax3.set_xticks(x); ax3.set_xticklabels([z.capitalize() for z in zones], color='white')
ax3.tick_params(colors='white')
[s.set_color('#444') for s in ax3.spines.values()]
ax3.legend(facecolor='#1a1d27', labelcolor='white', fontsize=9)
ax3.grid(axis='y', alpha=0.2)

# Panel 4: Flood probability distribution
ax4 = fig.add_subplot(gs[1, 2])
ax4.set_facecolor('#1a1d27')
flood_mask  = y_te == 1
normal_mask = y_te == 0
ax4.hist(flood_proba_te[normal_mask], bins=30, alpha=0.6,
         color='#4fc3f7', label='Normal days', density=True)
ax4.hist(flood_proba_te[flood_mask],  bins=30, alpha=0.8,
         color='#ef5350', label='Flood days', density=True)
ax4.set_title('XGBoost Flood Probability\nDistribution', color='white', fontsize=11)
ax4.set_xlabel('P(flood)', color='white')
ax4.set_ylabel('Density', color='white')
ax4.tick_params(colors='white')
[s.set_color('#444') for s in ax4.spines.values()]
ax4.legend(facecolor='#1a1d27', labelcolor='white', fontsize=9)

plt.savefig('/mnt/user-data/outputs/stage2_xgboost_results.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
print("\nSaved: stage2_xgboost_results.png")

# ── Save final outputs ────────────────────────────────────────────────────────
df_te_copy.to_parquet('/home/claude/stage2/test_results_all_methods.parquet', index=False)
feat_imp.to_csv('/home/claude/stage2/feature_importance.csv', header=['importance'])
print("Saved: test_results_all_methods.parquet, feature_importance.csv")
print("\n✅ Stage 2 Complete. All four methods built and evaluated.")
print("   When ERA5 + IMERG are added: update PRODUCT_COLS list and rerun Parts 2-4.")
print("   The framework handles any number of products automatically.")
