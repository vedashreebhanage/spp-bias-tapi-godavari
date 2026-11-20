"""
STAGE 1 - Step 5: block bootstrap over time AND space.

Why not an ordinary bootstrap
-----------------------------
The aligned dataset has 3,287 days x 92 cells = 302,404 cell-days. Treating those
as 302,404 independent samples is what let the first implementation claim N was in
the hundreds of thousands, which made every confidence interval far too narrow.
They are nothing like independent:

  * in time -- monsoon rainfall arrives in spells; consecutive days are strongly
    correlated, and a single storm can dominate a fortnight;
  * in space -- the basin is ~250 km across and a rain system covers much of it,
    so neighbouring cells see the same event.

Resampling individual cell-days destroys both structures and produces intervals
that are confidently wrong. This module resamples contiguous TIME BLOCKS and
whole SPATIAL CLUSTERS instead, so each draw keeps the correlation that actually
exists in the data.

The honest N to quote alongside any result is not the cell-day count but the
number of independent-ish units resampled: ~110 time blocks and ~12 spatial
clusters.
"""

import numpy as np

__all__ = ["make_time_blocks", "make_spatial_clusters", "block_bootstrap",
           "percentile_ci", "BLOCK_DAYS"]

# Long enough to contain a whole monsoon spell plus its decay. Rainfall
# autocorrelation over the basin dies out within a week or two; 30 days is
# comfortably beyond it without leaving too few blocks to resample.
BLOCK_DAYS = 30


def make_time_blocks(n_days, block_days=BLOCK_DAYS):
    """Split a record into contiguous blocks of consecutive day indices."""
    if n_days <= 0:
        raise ValueError("n_days must be positive")
    edges = np.arange(0, n_days, block_days)
    return [np.arange(a, min(a + block_days, n_days)) for a in edges]


def make_spatial_clusters(lat, lon, n_lon=6, n_lat=2):
    """Geographic clusters of cells, as a cluster id per cell.

    Deliberately geographic, NOT by elevation zone. The zones are terciles of
    elevation and are not spatially contiguous -- high cells sit on the northern
    and southern rims with the valley between them -- so zones would not capture
    the spatial correlation that the bootstrap is meant to preserve.

    The basin is long and thin (about 5.3 deg of longitude by 1.8 of latitude),
    so it is cut into more bands along longitude than latitude.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat.shape != lon.shape:
        raise ValueError("lat and lon must have the same shape")

    def bands(values, k):
        cuts = np.quantile(values, np.linspace(0, 1, k + 1)[1:-1])
        return np.searchsorted(cuts, values, side="right")

    return bands(lon, n_lon) * n_lat + bands(lat, n_lat)


def block_bootstrap(obs, sim, stat, blocks, clusters, n_iter=500, seed=0):
    """Resample time blocks and spatial clusters jointly; return the statistic each time.

    Parameters
    ----------
    obs, sim : (n_days, n_cells) arrays, already restricted to the stratum of
        interest in the CELL dimension (blocks index rows, clusters index columns).
    stat : callable(obs_sample, sim_sample) -> float
    blocks : list of day-index arrays, from make_time_blocks
    clusters : (n_cells,) cluster id per cell, from make_spatial_clusters
    """
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    if obs.shape != sim.shape:
        raise ValueError(f"shape mismatch: {obs.shape} vs {sim.shape}")
    if obs.ndim != 2:
        raise ValueError("expected (days, cells)")
    if len(clusters) != obs.shape[1]:
        raise ValueError("clusters must have one entry per cell")

    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    members = {c: np.nonzero(clusters == c)[0] for c in unique}

    out = np.empty(n_iter)
    for it in range(n_iter):
        day_idx = np.concatenate(
            [blocks[b] for b in rng.integers(0, len(blocks), len(blocks))])
        cell_idx = np.concatenate(
            [members[unique[c]] for c in rng.integers(0, unique.size, unique.size)])
        out[it] = stat(obs[np.ix_(day_idx, cell_idx)], sim[np.ix_(day_idx, cell_idx)])
    return out


def percentile_ci(samples, level=95):
    """Two-sided percentile interval, ignoring any NaN draws."""
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return (np.nan, np.nan)
    tail = (100 - level) / 2
    return (float(np.percentile(s, tail)), float(np.percentile(s, 100 - tail)))
