"""
Tests for src/stage1/metrics.py.

Every expected value below is either worked out by hand in the docstring, or
derived from a property that must hold no matter how the function is implemented.
Nothing here checks a metric against the module's own output.

    python tests/test_metrics.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stage1 import metrics as M  # noqa: E402


# ------------------------------------------------------------------ pairing --
def test_pair_drops_nan_pairs_from_both_sides():
    o, s = M.pair([1, 2, np.nan, 4], [1, np.nan, 3, 4])
    assert list(o) == [1, 4] and list(s) == [1, 4]


def test_pair_rejects_mismatched_shapes():
    try:
        M.pair([1, 2, 3], [1, 2])
    except ValueError:
        return
    raise AssertionError("mismatched shapes should raise")


def test_pair_flattens_2d_grids():
    o, s = M.pair(np.ones((3, 4)), np.zeros((3, 4)))
    assert o.size == 12 and s.size == 12


# -------------------------------------------------------------------- pbias --
def test_pbias_hand_computed():
    """obs=[1,2,3,4] sum 10; sim=[2,2,3,5]; diffs 1,0,0,1 sum 2 -> 100*2/10 = 20%."""
    assert np.isclose(M.pbias([1, 2, 3, 4], [2, 2, 3, 5]), 20.0)


def test_pbias_sign_convention_positive_means_wetter():
    assert M.pbias([1, 1, 1], [2, 2, 2]) > 0
    assert M.pbias([2, 2, 2], [1, 1, 1]) < 0


def test_pbias_doubling_gives_exactly_100_percent():
    rng = np.random.default_rng(1)
    o = rng.gamma(0.5, 8.0, 500)
    assert np.isclose(M.pbias(o, 2 * o), 100.0)


def test_pbias_is_nan_when_observed_total_is_zero():
    assert np.isnan(M.pbias([0, 0, 0], [1, 2, 3]))


# --------------------------------------------------------------- rmse / mae --
def test_rmse_hand_computed():
    """errors 1,0,0,1 -> mean square 2/4 = 0.5 -> sqrt = 0.70710678."""
    assert np.isclose(M.rmse([1, 2, 3, 4], [2, 2, 3, 5]), np.sqrt(0.5))


def test_mae_hand_computed():
    """|errors| 1,0,0,1 -> mean 0.5."""
    assert np.isclose(M.mae([1, 2, 3, 4], [2, 2, 3, 5]), 0.5)


def test_rmse_is_zero_for_a_perfect_product():
    assert M.rmse([3, 1, 4, 1, 5], [3, 1, 4, 1, 5]) == 0.0


def test_rmse_never_below_mae():
    rng = np.random.default_rng(2)
    o, s = rng.gamma(0.5, 8, 300), rng.gamma(0.5, 8, 300)
    assert M.rmse(o, s) >= M.mae(o, s) - 1e-12


# --------------------------------------------------------------------- corr --
def test_corr_perfect_and_inverse():
    assert np.isclose(M.corr([1, 2, 3, 4], [2, 4, 6, 8]), 1.0)
    assert np.isclose(M.corr([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)


def test_corr_is_nan_for_a_constant_series():
    assert np.isnan(M.corr([1, 1, 1, 1], [1, 2, 3, 4]))


# ---------------------------------------------------------------------- kge --
def test_kge_is_one_for_a_perfect_product():
    assert np.isclose(M.kge([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]), 1.0)


def test_kge_hand_computed_for_a_doubled_series():
    """sim = 2*obs: r=1, alpha=2, beta=2 -> KGE = 1 - sqrt(0 + 1 + 1) = 1 - sqrt(2)."""
    o = np.array([1.0, 2, 3, 4, 5])
    assert np.isclose(M.kge(o, 2 * o), 1 - np.sqrt(2))


def test_kge_penalises_bias_even_when_correlation_is_perfect():
    o = np.array([1.0, 2, 3, 4, 5])
    assert M.kge(o, o + 10) < M.kge(o, o)


# ------------------------------------------------------- contingency scores --
def test_contingency_hand_counted():
    """threshold 1 mm.
    obs 0.0 0.5 2.0 5.0 0.2 3.0   wet? N N Y Y N Y
    sim 0.0 2.0 3.0 0.1 0.0 4.0   wet? N Y Y N N Y
    -> hits 2 (idx 2,5); misses 1 (idx 3); false alarms 1 (idx 1); correct neg 2.
    """
    obs = [0.0, 0.5, 2.0, 5.0, 0.2, 3.0]
    sim = [0.0, 2.0, 3.0, 0.1, 0.0, 4.0]
    assert M.contingency(obs, sim) == (2, 1, 1, 2)
    assert np.isclose(M.pod(obs, sim), 2 / 3)      # 2 / (2 + 1)
    assert np.isclose(M.far(obs, sim), 1 / 3)      # 1 / (2 + 1)
    assert np.isclose(M.csi(obs, sim), 2 / 4)      # 2 / (2 + 1 + 1)


def test_threshold_is_inclusive_at_exactly_one_mm():
    """A day of exactly 1.0 mm counts as rain, on both sides."""
    assert M.contingency([1.0], [1.0]) == (1, 0, 0, 0)
    assert M.contingency([0.999], [0.999]) == (0, 0, 0, 1)


def test_perfect_detection_scores():
    obs = [0.0, 5.0, 0.0, 9.0]
    assert M.pod(obs, obs) == 1.0
    assert M.far(obs, obs) == 0.0
    assert M.csi(obs, obs) == 1.0


def test_a_product_that_always_says_rain_has_pod_one_and_high_far():
    obs = [0.0, 0.0, 0.0, 5.0]
    sim = [9.0, 9.0, 9.0, 9.0]
    assert M.pod(obs, sim) == 1.0
    assert np.isclose(M.far(obs, sim), 3 / 4)
    assert np.isclose(M.csi(obs, sim), 1 / 4)


def test_dry_days_matter_for_far_but_not_pod():
    """Adding correct dry days must not change POD, and must not change FAR either
    -- but dropping true dry days makes false alarms impossible to count at all."""
    obs = [0.0, 5.0]
    sim = [9.0, 9.0]
    assert np.isclose(M.far(obs, sim), 0.5)
    obs2 = obs + [0.0] * 10
    sim2 = sim + [0.0] * 10
    assert np.isclose(M.pod(obs2, sim2), M.pod(obs, sim))
    assert np.isclose(M.far(obs2, sim2), 0.5)


# ----------------------------------------------------------- quantile bias --
def test_quantile_bias_hand_computed():
    """A uniformly doubled series shifts every quantile by exactly +100%."""
    rng = np.random.default_rng(3)
    o = rng.gamma(0.4, 10.0, 2000)
    assert np.isclose(M.quantile_bias(o, 2 * o, 95), 100.0)
    assert np.isclose(M.quantile_bias(o, 2 * o, 99), 100.0)


def test_quantile_bias_catches_flattened_extremes_that_pbias_misses():
    """The case the metric exists for: right mean, squashed tail.

    obs has a heavy tail; sim replaces the top 5% with the median and compensates
    by adding the same total back across all days. Mean bias ~0, P99 badly low.
    """
    rng = np.random.default_rng(4)
    o = rng.gamma(0.4, 10.0, 5000)
    s = o.copy()
    cut = np.percentile(o, 95)
    lost = s[s > cut].sum() - np.median(o) * (s > cut).sum()
    s[s > cut] = np.median(o)
    s = s + lost / s.size
    pb = M.pbias(o, s)
    q99 = M.quantile_bias(o, s, 99)
    assert abs(pb) < 1.0, f"mean bias should be near zero, got {pb}"
    assert q99 < -25, f"P99 bias should expose the flattened tail, got {q99}"
    # the point of the metric: it must see something PBIAS cannot
    assert abs(q99) > 25 * abs(pb), (q99, pb)


# ------------------------------------------------------------------ bundle --
def test_all_metrics_reports_pairs_not_cells():
    o = np.ones((10, 5))
    s = np.ones((10, 5))
    s[0, 0] = np.nan
    out = M.all_metrics(o, s)
    assert out["n_pairs"] == 49, out["n_pairs"]


def test_all_metrics_perfect_product():
    rng = np.random.default_rng(5)
    o = rng.gamma(0.5, 8.0, (100, 6))
    out = M.all_metrics(o, o)
    assert np.isclose(out["pbias"], 0.0)
    assert np.isclose(out["cc"], 1.0)
    assert np.isclose(out["kge"], 1.0)
    assert out["rmse"] == 0.0
    assert np.isclose(out["p99_bias"], 0.0)


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
