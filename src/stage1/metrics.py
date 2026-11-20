"""
STAGE 1 - Step 4: the metrics, and nothing else.

Every function here takes paired observed/simulated arrays and returns one number.
No file reading, no plotting, no stratification -- so that each formula can be
tested in isolation against a value worked out by hand (tests/test_metrics.py).

That isolation is the point. The first implementation of this project produced
arithmetic that was internally consistent and consistently wrong; nothing about
its output would have looked odd. A formula that is only ever exercised inside a
600-line analysis script is a formula nobody has actually checked.

Conventions
-----------
* ``obs`` is IMD, ``sim`` is the satellite/reanalysis product.
* Pairs where either value is NaN are dropped, pairwise, in every function.
* Dry days are KEPT. They are signal: FAR is uncomputable without them, and
  removing them would turn a wet-bias question into a different question.
* Positive PBIAS means the product is WETTER than IMD.

The reference caveat that must travel with every number these produce: IMD is
itself interpolated from a sparse gauge network, so part of what is reported here
as "product bias" is IMD's own interpolation error. audit_step1_overall.py reports
every result with and without reference-suspect cells for this reason.
"""

import numpy as np

__all__ = [
    "RAIN_THRESHOLD_MM", "pair", "pbias", "corr", "rmse", "mae", "kge",
    "contingency", "pod", "far", "csi", "quantile_bias", "all_metrics",
]

# A "rain day". 1 mm/day is the conventional threshold in Indian
# hydrometeorology and matches what IMD uses for rain-day counts.
RAIN_THRESHOLD_MM = 1.0


def pair(obs, sim):
    """Flatten to matching 1-D arrays, dropping any pair with a NaN in it."""
    obs = np.asarray(obs, dtype=float).ravel()
    sim = np.asarray(sim, dtype=float).ravel()
    if obs.shape != sim.shape:
        raise ValueError(f"shape mismatch: obs {obs.shape} vs sim {sim.shape}")
    keep = np.isfinite(obs) & np.isfinite(sim)
    return obs[keep], sim[keep]


def pbias(obs, sim):
    """Percent bias: 100 * sum(sim - obs) / sum(obs). Positive = product too wet."""
    o, s = pair(obs, sim)
    total = o.sum()
    if total == 0:
        return np.nan
    return 100.0 * (s - o).sum() / total


def corr(obs, sim):
    """Pearson correlation. NaN if either series is constant."""
    o, s = pair(obs, sim)
    if o.size < 2 or o.std() == 0 or s.std() == 0:
        return np.nan
    return float(np.corrcoef(o, s)[0, 1])


def rmse(obs, sim):
    o, s = pair(obs, sim)
    if o.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((s - o) ** 2)))


def mae(obs, sim):
    o, s = pair(obs, sim)
    if o.size == 0:
        return np.nan
    return float(np.mean(np.abs(s - o)))


def kge(obs, sim):
    """Kling-Gupta efficiency (Gupta et al. 2009).

        KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)

    with alpha = sd(sim)/sd(obs) the variability ratio and beta = mean(sim)/mean(obs)
    the bias ratio. KGE = 1 is perfect. Unlike NSE it does not reward a product for
    simply being smooth, which matters when comparing a reanalysis against
    satellite retrievals.
    """
    o, s = pair(obs, sim)
    if o.size < 2 or o.std() == 0 or o.mean() == 0:
        return np.nan
    r = corr(o, s)
    if not np.isfinite(r):
        return np.nan
    alpha = s.std() / o.std()
    beta = s.mean() / o.mean()
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def contingency(obs, sim, threshold=RAIN_THRESHOLD_MM):
    """Rain/no-rain contingency counts (hits, misses, false_alarms, correct_negatives)."""
    o, s = pair(obs, sim)
    o_wet, s_wet = o >= threshold, s >= threshold
    return (
        int(np.sum(o_wet & s_wet)),
        int(np.sum(o_wet & ~s_wet)),
        int(np.sum(~o_wet & s_wet)),
        int(np.sum(~o_wet & ~s_wet)),
    )


def pod(obs, sim, threshold=RAIN_THRESHOLD_MM):
    """Probability of detection: of the days it really rained, what fraction was caught."""
    h, m, _, _ = contingency(obs, sim, threshold)
    return np.nan if (h + m) == 0 else h / (h + m)


def far(obs, sim, threshold=RAIN_THRESHOLD_MM):
    """False alarm ratio: of the days the product called rain, what fraction was wrong."""
    h, _, f, _ = contingency(obs, sim, threshold)
    return np.nan if (h + f) == 0 else f / (h + f)


def csi(obs, sim, threshold=RAIN_THRESHOLD_MM):
    """Critical success index: hits / (hits + misses + false alarms)."""
    h, m, f, _ = contingency(obs, sim, threshold)
    return np.nan if (h + m + f) == 0 else h / (h + m + f)


def quantile_bias(obs, sim, q):
    """Percent bias of the q-th percentile (q in 0-100).

    For a study about rainfall extremes this is arguably the most relevant number
    on the page: a product can have near-zero mean bias while systematically
    flattening its heaviest days.
    """
    o, s = pair(obs, sim)
    if o.size == 0:
        return np.nan
    qo = np.percentile(o, q)
    if qo == 0:
        return np.nan
    return 100.0 * (np.percentile(s, q) - qo) / qo


def all_metrics(obs, sim, threshold=RAIN_THRESHOLD_MM):
    """Every metric at once, as a dict. n is the number of valid PAIRS, not cells."""
    o, s = pair(obs, sim)
    return {
        "n_pairs": int(o.size),
        "obs_mean": float(o.mean()) if o.size else np.nan,
        "sim_mean": float(s.mean()) if o.size else np.nan,
        "pbias": pbias(o, s),
        "cc": corr(o, s),
        "rmse": rmse(o, s),
        "mae": mae(o, s),
        "kge": kge(o, s),
        "pod": pod(o, s, threshold),
        "far": far(o, s, threshold),
        "csi": csi(o, s, threshold),
        "p95_bias": quantile_bias(o, s, 95),
        "p99_bias": quantile_bias(o, s, 99),
    }
