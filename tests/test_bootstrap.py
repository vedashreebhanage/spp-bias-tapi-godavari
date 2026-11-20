"""
Tests for src/stage1/bootstrap.py.

The bootstrap exists to stop confidence intervals being too narrow, so the
important test is not "does it run" but "does it actually widen intervals on
autocorrelated data compared with resampling individual observations". That
property is checked directly.

    python tests/test_bootstrap.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stage1.bootstrap import (block_bootstrap, make_spatial_clusters,  # noqa: E402
                              make_time_blocks, percentile_ci)


def test_time_blocks_partition_the_record_exactly_once():
    blocks = make_time_blocks(3287, 30)
    allidx = np.concatenate(blocks)
    assert allidx.size == 3287
    assert np.array_equal(np.sort(allidx), np.arange(3287))
    assert len(blocks) == 110          # 109 full blocks + a 17-day remainder
    assert blocks[-1].size == 3287 - 109 * 30


def test_time_blocks_are_contiguous():
    for b in make_time_blocks(100, 7):
        assert np.array_equal(b, np.arange(b[0], b[-1] + 1))


def test_spatial_clusters_cover_every_cell():
    rng = np.random.default_rng(0)
    lat = rng.uniform(20.25, 22.0, 92)
    lon = rng.uniform(73.0, 78.25, 92)
    c = make_spatial_clusters(lat, lon)
    assert c.size == 92
    assert np.unique(c).size <= 12
    assert c.min() >= 0


def test_spatial_clusters_are_geographic_not_arbitrary():
    """Cells close together must usually land in the same cluster."""
    lat = np.repeat(np.linspace(20.25, 22.0, 8), 8)
    lon = np.tile(np.linspace(73.0, 78.25, 8), 8)
    c = make_spatial_clusters(lat, lon)
    west = c[lon < 74.0]
    east = c[lon > 77.0]
    assert not set(west) & set(east), "west and east must not share a cluster"


def test_bootstrap_recovers_a_known_mean_difference():
    """sim = obs + 5 everywhere: every resample must give a mean difference of 5."""
    rng = np.random.default_rng(1)
    obs = rng.gamma(0.5, 8.0, (300, 20))
    sim = obs + 5.0
    blocks = make_time_blocks(300, 30)
    clusters = make_spatial_clusters(rng.uniform(20, 22, 20), rng.uniform(73, 78, 20))
    draws = block_bootstrap(obs, sim, lambda o, s: float((s - o).mean()),
                            blocks, clusters, n_iter=50)
    assert np.allclose(draws, 5.0)


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(2)
    obs = rng.gamma(0.5, 8.0, (300, 20))
    sim = obs * 1.2
    blocks = make_time_blocks(300, 30)
    clusters = make_spatial_clusters(rng.uniform(20, 22, 20), rng.uniform(73, 78, 20))
    stat = lambda o, s: 100.0 * (s - o).sum() / o.sum()   # noqa: E731  (pbias)
    point = stat(obs, sim)
    lo, hi = percentile_ci(block_bootstrap(obs, sim, stat, blocks, clusters, n_iter=200))
    assert lo <= point <= hi, (lo, point, hi)


def test_block_bootstrap_is_wider_than_naive_resampling_on_correlated_data():
    """The whole justification for this module, checked directly.

    Build a strongly autocorrelated series, then compare the spread of the mean
    under (a) 30-day block resampling and (b) resampling individual days. If the
    blocks are doing their job, (a) must be clearly wider.
    """
    rng = np.random.default_rng(3)
    n = 1200
    noise = rng.normal(0, 1, n)
    series = np.empty(n)
    series[0] = noise[0]
    for t in range(1, n):                      # AR(1), phi = 0.95
        series[t] = 0.95 * series[t - 1] + noise[t]
    obs = np.tile(series[:, None], (1, 10))
    sim = obs + 1.0
    clusters = np.zeros(10, dtype=int)
    stat = lambda o, s: float(o.mean())        # noqa: E731

    wide = block_bootstrap(obs, sim, stat, make_time_blocks(n, 30), clusters, n_iter=300)
    naive = block_bootstrap(obs, sim, stat, make_time_blocks(n, 1), clusters, n_iter=300)
    assert wide.std() > 2 * naive.std(), (wide.std(), naive.std())


def test_percentile_ci_ignores_nan_draws():
    lo, hi = percentile_ci([1.0, 2.0, np.nan, 3.0, 4.0], level=50)
    assert np.isfinite(lo) and np.isfinite(hi) and lo < hi


def test_percentile_ci_all_nan_returns_nan():
    lo, hi = percentile_ci([np.nan, np.nan])
    assert np.isnan(lo) and np.isnan(hi)


def test_bootstrap_is_reproducible_from_the_seed():
    rng = np.random.default_rng(4)
    obs = rng.gamma(0.5, 8.0, (120, 8))
    sim = obs * 0.9
    blocks = make_time_blocks(120, 30)
    clusters = make_spatial_clusters(rng.uniform(20, 22, 8), rng.uniform(73, 78, 8))
    stat = lambda o, s: float(s.mean() - o.mean())   # noqa: E731
    a = block_bootstrap(obs, sim, stat, blocks, clusters, n_iter=30, seed=7)
    b = block_bootstrap(obs, sim, stat, blocks, clusters, n_iter=30, seed=7)
    assert np.array_equal(a, b)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
