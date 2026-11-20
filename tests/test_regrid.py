"""
Unit tests for src/common/regrid.py.

These exist because the first implementation of this project was internally
consistent and wrong. Regridding is the step where that is easiest to repeat:
misaligned weights produce arrays of exactly the right shape full of plausible
rainfall values. Every expected value below is derived independently of the
module -- by hand, or from a property that must hold regardless of implementation.

    python -m pytest tests/test_regrid.py -v
    python tests/test_regrid.py          # runs without pytest too
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.regrid import (apply_weights, build_weights, cell_edges,  # noqa: E402
                           regular_axis)


# ----------------------------------------------------------------- edges --
def test_cell_edges_simple():
    e = cell_edges([0.05, 0.15, 0.25])
    assert np.allclose(e, [0.0, 0.10, 0.20, 0.30])


def test_cell_edges_rejects_irregular():
    try:
        cell_edges([0.0, 1.0, 3.0])
    except ValueError:
        return
    raise AssertionError("irregular spacing should raise")


def test_cell_edges_rejects_descending():
    try:
        cell_edges([3.0, 2.0, 1.0])          # ERA5 latitude, unflipped
    except ValueError:
        return
    raise AssertionError("descending centers should raise -- flip the axis first")


# ------------------------------------------------------------- constants --
def test_constant_field_is_preserved_exactly():
    """A uniform field must regrid to the same constant, whatever the weights."""
    src_lat = np.arange(20.0, 22.0, 0.1) + 0.05
    src_lon = np.arange(74.0, 76.0, 0.1) + 0.05
    dst_lat = np.array([20.625, 21.375])
    dst_lon = np.array([74.625, 75.125])
    w = build_weights(src_lat, src_lon, dst_lat, dst_lon)
    data = np.full((3, src_lat.size, src_lon.size), 7.25)
    vals, frac = apply_weights(data, w)
    assert np.allclose(vals, 7.25), vals
    assert np.allclose(frac, 1.0), frac


def test_aligned_grids_do_nest_exactly():
    """When target edges land on source edges, 0.05 into 0.25 gives a 5x5 block."""
    src_lat = np.arange(20.0, 22.0, 0.05) + 0.025      # edges on 20.0 + 0.05k
    src_lon = np.arange(74.0, 76.0, 0.05) + 0.025
    w = build_weights(src_lat, src_lon, [20.625], [74.375])   # edges on 0.05k
    assert w[0]["w"].shape == (5, 5), w[0]["w"].shape
    assert np.all(w[0]["w"] > 0)


def test_real_chirps_to_imd_is_offset_by_half_a_cell():
    """The actual grids do NOT nest -- 6 partial cells per axis, ends at half weight.

    CHIRPS edges lie on 19.5 + 0.05k; IMD edges lie on 6.375 + 0.25k. The IMD edge
    20.125 is 12.5 CHIRPS steps from 19.5. A 5x5 block mean would therefore sample
    a footprint shifted 0.025 deg (~2.8 km) from the IMD cell it claims to be.
    """
    chirps_lat = np.arange(80) * 0.05 + 19.525
    chirps_lon = np.arange(120) * 0.05 + 72.525
    w = build_weights(chirps_lat, chirps_lon, [20.25], [73.00])[0]["w"]
    assert w.shape == (6, 6), w.shape

    rel_row = w[2] / w[2].max()
    assert np.isclose(rel_row[0], 0.5, atol=0.002), rel_row
    assert np.isclose(rel_row[-1], 0.5, atol=0.002), rel_row
    assert np.allclose(rel_row[1:-1], 1.0, atol=0.002), rel_row

    rel_col = w[:, 2] / w[:, 2].max()
    assert np.isclose(rel_col[0], 0.5, atol=0.002), rel_col
    assert np.isclose(rel_col[-1], 0.5, atol=0.002), rel_col


def test_float32_coordinate_noise_is_tolerated():
    """CHIRPS stores coordinates as float32; the axis must still be recognised."""
    lat32 = (np.arange(80) * 0.05 + 19.525).astype("float32").astype("float64")
    lon32 = (np.arange(120) * 0.05 + 72.525).astype("float32").astype("float64")
    d = np.diff(lon32)
    assert d.min() < 0.05 < d.max(), "test premise: spacing wobbles either side"
    fitted, step = regular_axis(lon32)
    assert step == 0.05, step
    assert np.abs(fitted - lon32).max() < 1e-5
    # and the fitted edges must land on the true grid
    e = cell_edges(lat32)
    assert np.isclose(e[0], 19.5) and np.isclose(e[-1], 23.5), (e[0], e[-1])


def test_area_weighting_favours_the_equatorward_row():
    """Southern rows of a northern-hemisphere block cover more area."""
    src_lat = np.arange(20.0, 21.0, 0.05) + 0.025
    src_lon = np.arange(74.0, 75.0, 0.05) + 0.025
    w = build_weights(src_lat, src_lon, [20.625], [74.375])[0]["w"]
    col = w[:, 0]
    assert col[0] > col[-1], "row weights should decrease northward"


# ------------------------------------------------------------------ NaN --
def test_nan_pixels_are_dropped_and_weights_renormalised():
    """Half the cell missing -> value is the mean of what remains, frac = 0.5."""
    src_lat = np.array([20.5125, 20.5375, 20.5625, 20.5875])   # 0.025 spacing
    src_lon = np.array([74.5125, 74.5375, 74.5625, 74.5875])
    w = build_weights(src_lat, src_lon, [20.55], [74.55], dst_size=0.1)
    data = np.zeros((1, 4, 4))
    data[0, :, :2] = 4.0
    data[0, :, 2:] = np.nan
    vals, frac = apply_weights(data, w)
    assert np.isclose(vals[0, 0], 4.0), vals
    assert np.isclose(frac[0, 0], 0.5), frac


def test_all_nan_cell_is_nan_not_zero():
    src_lat = np.array([20.5125, 20.5375, 20.5625, 20.5875])
    src_lon = np.array([74.5125, 74.5375, 74.5625, 74.5875])
    w = build_weights(src_lat, src_lon, [20.55], [74.55], dst_size=0.1)
    data = np.full((1, 4, 4), np.nan)
    vals, frac = apply_weights(data, w)
    assert np.isnan(vals[0, 0]), "an empty cell must be NaN, never 0.0"
    assert frac[0, 0] == 0.0


# --------------------------------------------------------- hand-computed --
def test_hand_computed_two_by_two():
    """Worked out independently of the module.

    Target cell: 0.2 deg wide, centered (20.0, 74.0) -> lat 19.9..20.1,
    lon 73.9..74.1. Source: 0.1 deg cells centered at lat 19.95, 20.05 and
    lon 73.95, 74.05 -- exactly four cells, each filling one quadrant.

    Latitude weights are sin(top) - sin(bot) in radians:
        south row: sin(20.0 deg) - sin(19.9 deg)
        north row: sin(20.1 deg) - sin(20.0 deg)
    Longitude weights are equal (0.1 each), so they cancel in the mean.
    Values 1, 2 (south row) and 3, 4 (north row).
    """
    s = np.sin(np.deg2rad([19.9, 20.0, 20.1]))
    w_south, w_north = s[1] - s[0], s[2] - s[1]
    expected = ((1 + 2) / 2 * w_south + (3 + 4) / 2 * w_north) / (w_south + w_north)

    w = build_weights([19.95, 20.05], [73.95, 74.05], [20.0], [74.0], dst_size=0.2)
    data = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    vals, frac = apply_weights(data, w)
    assert np.isclose(vals[0, 0], expected), (vals[0, 0], expected)
    assert np.isclose(frac[0, 0], 1.0)
    # and the plain unweighted mean is 2.5 -- confirm we are NOT computing that
    assert not np.isclose(vals[0, 0], 2.5)


# --------------------------------------------------------------- extent --
def test_source_not_covering_target_raises():
    """A source file that stops short of the basin must fail loudly."""
    src_lat = np.arange(20.0, 20.5, 0.1) + 0.05
    src_lon = np.arange(74.0, 74.5, 0.1) + 0.05
    try:
        build_weights(src_lat, src_lon, [20.25], [74.75])   # lon outside source
    except ValueError as exc:
        assert "does not span" in str(exc)
        return
    raise AssertionError("a target cell outside the source grid should raise")


def test_mean_is_bounded_by_the_data():
    """No weighting scheme may produce a value outside the source range."""
    rng = np.random.default_rng(0)
    src_lat = np.arange(20.0, 22.0, 0.1) + 0.05
    src_lon = np.arange(74.0, 76.0, 0.1) + 0.05
    dst_lat = np.array([20.375, 21.625])
    dst_lon = np.array([74.875, 75.125])
    w = build_weights(src_lat, src_lon, dst_lat, dst_lon)
    data = rng.gamma(0.4, 12.0, size=(50, src_lat.size, src_lon.size))
    vals, _ = apply_weights(data, w)
    assert vals.min() >= data.min() - 1e-9
    assert vals.max() <= data.max() + 1e-9
    assert np.all(vals >= 0), "rainfall must never regrid negative"


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
