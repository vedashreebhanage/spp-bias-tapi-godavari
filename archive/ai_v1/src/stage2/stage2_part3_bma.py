"""
STAGE 2 — PART 3: Bayesian Model Averaging (BMA)
=================================================
What this file does:
    Implements BMA — a probabilistic ML method that treats each
    satellite product as a model and learns the probability that
    each model is "correct" given the observed IMD data.

What is BMA in plain English?
    Imagine three friends each give you a different weather forecast.
    Instead of averaging them equally, you ask: historically, which
    friend has been most accurate? You weight their forecasts by their
    historical track record. BMA does exactly this — but mathematically,
    using probability theory.

    The output of BMA is not just a weighted average — it gives you
    a PROBABILITY DISTRIBUTION over possible rainfall values. This
    means instead of "it will rain 8mm tomorrow", BMA says "there
    is a 70% chance of 6-10mm and a 30% chance of 10-20mm". This
    is more useful for flood risk assessment.

How BMA works mathematically:
    1. Assume each SPP predicts rainfall from a Normal distribution
       centred at its estimate, with some variance σ²
    2. The BMA forecast is a weighted mixture of these Normal distributions:
           p(y) = Σ w_k × N(y | f_k, σ_k²)
       where w_k is the weight for product k and f_k is its forecast
    3. Weights and variances are learned using the EM (Expectation-
       Maximisation) algorithm on training data (non-event years)
    4. The posterior mean (best estimate) is just: Σ w_k × f_k

Why EM algorithm?
    We can't solve for w_k directly because we don't know which
    product is "right" for each observation. EM handles this:
    - E step: estimate how likely each product generated each observation
    - M step: update weights based on those likelihoods
    - Repeat until weights converge

Training vs test split:
    We train BMA on non-flood years (2015-2020) and test on
    flood years (2021, 2023). This is honest evaluation — the
    model never sees flood conditions during training.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from scipy import stats

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_parquet('/home/claude/stage2/tapi_data_with_ensemble.parquet')
print(f"Loaded: {df.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: BMA IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════

class BMAEnsemble:
    """
    Bayesian Model Averaging for satellite precipitation products.

    Parameters:
        product_cols: list of column names to include in the ensemble
                      e.g. ['chirps'] now, ['chirps','era5','imerg'] later
        n_iter:       number of EM iterations (50 is usually enough)
        tol:          convergence threshold — stop if weights change < tol

    The model learns separate weights for each zone because the terrain
    affects which product performs best (Upper Tapi vs Lower Tapi have
    very different error characteristics as Stage 1 showed).
    """

    def __init__(self, product_cols, n_iter=100, tol=1e-4):
        self.product_cols = product_cols
        self.n_iter       = n_iter
        self.tol          = tol
        self.weights_     = {}   # {zone: array of weights, one per product}
        self.sigmas_      = {}   # {zone: array of std devs, one per product}

    def _em_step(self, obs, forecasts, weights, sigmas):
        """
        One full EM iteration.

        E step: compute responsibilities (how much credit does each
                product get for each observation?)
        M step: update weights and sigmas based on responsibilities.

        obs:       (N,)   IMD observed values
        forecasts: (N,K)  K product forecasts for N observations
        weights:   (K,)   current weights
        sigmas:    (K,)   current standard deviations
        """
        N, K = forecasts.shape

        # E step: responsibilities
        # r[i,k] = P(product k generated observation i)
        # = w_k × Normal(obs_i | forecast_k_i, sigma_k²) / sum over k
        r = np.zeros((N, K))
        for k in range(K):
            r[:, k] = weights[k] * stats.norm.pdf(obs, forecasts[:, k], sigmas[k])

        # Normalise so each row sums to 1
        row_sum = r.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum > 0, row_sum, 1.0)   # avoid div by zero
        r = r / row_sum

        # M step: update weights and sigmas
        r_sum = r.sum(axis=0)   # total responsibility per product
        new_weights = r_sum / N
        new_sigmas  = np.zeros(K)
        for k in range(K):
            residuals    = obs - forecasts[:, k]
            new_sigmas[k] = np.sqrt(np.sum(r[:, k] * residuals**2) / r_sum[k])
            new_sigmas[k] = max(new_sigmas[k], 0.1)   # minimum sigma = 0.1mm

        return new_weights, new_sigmas

    def fit(self, df_train):
        """
        Train BMA on training data (non-event years).
        Learns separate weights and sigmas per zone.
        """
        print("\nTraining BMA...")
        K = len(self.product_cols)

        for zone in ['upper', 'middle', 'lower']:
            subset = df_train[
                (df_train.zone == zone) &
                (df_train.season == 'JJAS') &
                df_train.imd.notna()
            ]

            # Only use rows where all products have valid data
            valid_mask = subset[self.product_cols].notna().all(axis=1)
            subset = subset[valid_mask]

            if len(subset) < 100:
                print(f"  {zone}: insufficient data ({len(subset)} rows), skipping")
                continue

            obs       = subset.imd.values.astype(np.float64)
            forecasts = subset[self.product_cols].values.astype(np.float64)

            # Initialise: equal weights, sigma = RMSE of each product
            weights = np.ones(K) / K
            sigmas  = np.array([
                np.sqrt(np.mean((forecasts[:, k] - obs)**2))
                for k in range(K)
            ])

            # EM iterations
            for iteration in range(self.n_iter):
                new_weights, new_sigmas = self._em_step(obs, forecasts, weights, sigmas)
                weight_change = np.max(np.abs(new_weights - weights))
                weights, sigmas = new_weights, new_sigmas
                if weight_change < self.tol:
                    print(f"  {zone}: converged at iteration {iteration+1}")
                    break

            self.weights_[zone] = weights
            self.sigmas_[zone]  = sigmas
            print(f"  {zone}: weights={dict(zip(self.product_cols, weights.round(3)))}  "
                  f"sigmas={sigmas.round(2)}")

        return self

    def predict(self, df_test):
        """
        Apply BMA weights to test data.
        Returns the BMA posterior mean (weighted average of product forecasts).
        The posterior mean minimises MSE — it is the best point estimate.
        """
        result = np.full(len(df_test), np.nan)

        for zone in ['upper', 'middle', 'lower']:
            if zone not in self.weights_:
                continue
            mask      = df_test.zone == zone
            subset    = df_test[mask]
            forecasts = subset[self.product_cols].values.astype(np.float64)
            weights   = self.weights_[zone]

            # Handle NaN products: redistribute weight to non-NaN products
            bma_pred = np.zeros(len(subset))
            for i in range(len(subset)):
                row     = forecasts[i]
                valid   = np.isfinite(row)
                if not valid.any():
                    bma_pred[i] = np.nan
                else:
                    w_valid  = weights[valid]
                    w_valid  = w_valid / w_valid.sum()   # renormalise
                    bma_pred[i] = np.sum(w_valid * row[valid])

            result[mask.values] = bma_pred

        return result

    def predict_uncertainty(self, df_test, zone):
        """
        Returns the BMA predictive variance — a measure of uncertainty.
        High variance = products disagree = more uncertain forecast.
        This is useful for flagging high-risk flood scenarios.

        BMA variance = Σ w_k × (σ_k² + (f_k - μ_BMA)²)
        where μ_BMA is the BMA posterior mean.
        """
        if zone not in self.weights_:
            return np.full(len(df_test), np.nan)

        subset    = df_test[df_test.zone == zone]
        forecasts = subset[self.product_cols].values.astype(np.float64)
        weights   = self.weights_[zone]
        sigmas    = self.sigmas_[zone]
        K         = len(self.product_cols)

        bma_mean = np.sum(weights * forecasts, axis=1)
        variance = np.zeros(len(subset))
        for k in range(K):
            variance += weights[k] * (sigmas[k]**2 + (forecasts[:, k] - bma_mean)**2)

        return np.sqrt(variance)   # return std dev, not variance


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: TRAIN AND EVALUATE BMA
# ══════════════════════════════════════════════════════════════════════════════

# Training: non-flood years (2015-2020)
# Testing: flood years (2021, 2022, 2023)
# This is a legitimate train/test split — the model never sees
# flood conditions during training

PRODUCT_COLS = ['chirps']   # Will be ['chirps','era5','imerg'] when ready
TRAIN_YEARS  = [2015, 2016, 2017, 2018, 2019, 2020]
TEST_YEARS   = [2021, 2022, 2023]

df_train = df[df.year.isin(TRAIN_YEARS)].copy()
df_test  = df[df.year.isin(TEST_YEARS)].copy()

print(f"Train set: {len(df_train)} rows ({TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]})")
print(f"Test set:  {len(df_test)} rows ({TEST_YEARS[0]}-{TEST_YEARS[-1]})")

# Train BMA
bma = BMAEnsemble(product_cols=PRODUCT_COLS, n_iter=100, tol=1e-5)
bma.fit(df_train)

# Apply to full dataset
df['bma'] = bma.predict(df)
# Refresh train/test splits to include new bma column
df_train = df[df.year.isin(TRAIN_YEARS)].copy()
df_test  = df[df.year.isin(TEST_YEARS)].copy()

print(f"\nBMA column added. Non-NaN: {np.isfinite(df['bma']).sum()}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: COMPARE CHIRPS vs INVERSE-RMSE vs BMA (TEST SET)
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

def kge(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask].astype(np.float64), sim[mask].astype(np.float64)
    if len(o) < 10: return np.nan
    r = np.corrcoef(o, s)[0, 1]
    alpha = s.std() / o.std() if o.std() > 0 else np.nan
    beta  = s.mean() / o.mean() if o.mean() > 0 else np.nan
    return float(1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2))

print("\n── Test Set Performance (2021-2023, JJAS) ──")
print(f"{'Zone':<10} {'Metric':<8} {'CHIRPS':>10} {'Inv-RMSE':>10} {'BMA':>10}")
print("-" * 55)

test_jjas = df_test[df_test.season == 'JJAS']
results = []
for zone in ['upper', 'middle', 'lower', 'basin']:
    sub = test_jjas if zone == 'basin' else test_jjas[test_jjas.zone == zone]
    for metric_name, fn in [('PBIAS%', pbias), ('RMSE', rmse), ('CC', cc), ('KGE', kge)]:
        chp = fn(sub.imd.values, sub.chirps.values)
        ens = fn(sub.imd.values, sub.ensemble.values)
        bma_v = fn(sub.imd.values, sub.bma.values)
        print(f"  {zone:<10} {metric_name:<8} {chp:>10.3f} {ens:>10.3f} {bma_v:>10.3f}")
        results.append({'zone': zone, 'metric': metric_name,
                        'chirps': chp, 'inv_rmse': ens, 'bma': bma_v})

results_df = pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: DPBA COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

import datetime
import sys; sys.path.insert(0, '/home/claude')
from stage2.part1_loader import FLOOD_EVENTS

def dpba_fn(obs_ev, sim_ev, obs_ctrl, sim_ctrl):
    pb_ev   = pbias(obs_ev,   sim_ev)
    pb_ctrl = pbias(obs_ctrl, sim_ctrl)
    if np.isnan(pb_ev) or np.isnan(pb_ctrl): return np.nan
    return round(pb_ev - pb_ctrl, 2)

print("\n── DPBA Comparison: CHIRPS vs Inv-RMSE vs BMA ──")
dpba_rows = []
for event_yr, d0_str, d1_str in FLOOD_EVENTS:
    d0 = datetime.date.fromisoformat(d0_str)
    d1 = datetime.date.fromisoformat(d1_str)

    for zone in ['upper', 'middle', 'lower', 'basin']:
        zm = (df.zone == zone) if zone != 'basin' else pd.Series(True, index=df.index)
        ev_m   = zm & (df.date >= d0) & (df.date <= d1)
        ctrl_m = zm & (df.year != event_yr) & \
                 (df.month == d0.month) & \
                 (df.doy >= d0.timetuple().tm_yday) & \
                 (df.doy <= d1.timetuple().tm_yday)

        ev_s, ctrl_s = df[ev_m], df[ctrl_m]
        if len(ev_s) < 5 or len(ctrl_s) < 5: continue

        d_chp = dpba_fn(ev_s.imd.values, ev_s.chirps.values,
                        ctrl_s.imd.values, ctrl_s.chirps.values)
        d_ens = dpba_fn(ev_s.imd.values, ev_s.ensemble.values,
                        ctrl_s.imd.values, ctrl_s.ensemble.values)
        d_bma = dpba_fn(ev_s.imd.values, ev_s.bma.values,
                        ctrl_s.imd.values, ctrl_s.bma.values)

        dpba_rows.append({
            'event': event_yr, 'zone': zone,
            'DPBA_chirps': d_chp, 'DPBA_invrmse': d_ens, 'DPBA_bma': d_bma,
        })
        print(f"  {event_yr} {zone:8s}: CHIRPS={d_chp:+.1f}%  "
              f"InvRMSE={d_ens:+.1f}%  BMA={d_bma:+.1f}%")

dpba_results = pd.DataFrame(dpba_rows)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: VISUALISE BMA WEIGHTS AND PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#0f1117')
fig.suptitle('BMA Training Results — CHIRPS (2015-2020 JJAS)',
             color='white', fontsize=13, fontweight='bold')

zones = ['upper', 'middle', 'lower']
colors = ['#81c784', '#ffb74d', '#e57373']

for ax, zone, col in zip(axes, zones, colors):
    ax.set_facecolor('#1a1d27')
    if zone not in bma.weights_:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                color='white', transform=ax.transAxes)
        continue

    # Weight pie (just CHIRPS for now — shows 100%)
    weights = bma.weights_[zone]
    ax.bar(PRODUCT_COLS, weights, color=[col], alpha=0.85)
    ax.set_title(f'{zone.capitalize()} Tapi', color='white', fontsize=11)
    ax.set_ylabel('BMA Weight', color='white')
    ax.tick_params(colors='white')
    [s.set_color('#444') for s in ax.spines.values()]
    ax.set_ylim(0, 1.2)
    for i, (prod, w) in enumerate(zip(PRODUCT_COLS, weights)):
        sigma = bma.sigmas_[zone][i]
        ax.text(i, w + 0.03, f'w={w:.3f}\nσ={sigma:.2f}mm',
                ha='center', color='white', fontsize=10)
    ax.grid(axis='y', alpha=0.2)
    ax.text(0.5, 0.08, f'Note: With ERA5+IMERG added,\nweights will redistribute',
            ha='center', color='#aaa', fontsize=8, transform=ax.transAxes, style='italic')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/stage2_bma_weights.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
print("\nSaved: stage2_bma_weights.png")

# ── Save outputs ──────────────────────────────────────────────────────────────
df.to_parquet('/home/claude/stage2/tapi_data_with_bma.parquet', index=False)
results_df.to_csv('/home/claude/stage2/bma_performance.csv', index=False)
dpba_results.to_csv('/home/claude/stage2/dpba_bma.csv', index=False)
print("Saved: tapi_data_with_bma.parquet, bma_performance.csv, dpba_bma.csv")
print("\nPart 3 complete. Ready for Part 4 (XGBoost adaptive weighting).")
