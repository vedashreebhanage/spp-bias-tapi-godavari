"""
Tests for src/common/readers.py that need no NetCDF stack.

The IMD reader is pure numpy over a headerless binary file with no metadata
whatsoever -- nothing in the file itself would object if the shape were wrong, so
the assertions in the reader ARE the file format. These tests check those
assertions still hold against the real data, and that the cap which destroyed
v1's extremes has not crept back.

    python tests/test_readers.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common.readers import (IMD_LABEL_SHIFT_DAYS, IMD_LAT, IMD_LON,  # noqa: E402
                            _conform, read_imd_year)

HAVE_IMD = (ROOT / "data/raw/imd/RF2022.GRD").exists()


def test_conform_picks_the_matching_cells():
    """A globally-stored granule must be cut down to the subsetted grid exactly."""
    lat_ref = np.round(np.arange(19.75, 23.0, 0.1), 2)
    lon_ref = np.round(np.arange(72.25, 79.1, 0.1), 2)
    lat_wide = np.round(np.arange(19.65, 23.06, 0.1), 2)
    lon_wide = np.round(np.arange(72.15, 79.06, 0.1), 2)
    vals = np.arange(lat_wide.size * lon_wide.size, dtype=float).reshape(
        1, lat_wide.size, lon_wide.size)

    out = _conform(vals, lat_wide, lon_wide, lat_ref, lon_ref)
    assert out is not None
    assert out.shape == (1, lat_ref.size, lon_ref.size)
    j = int(np.nonzero(np.abs(lat_wide - lat_ref[0]) < 1e-6)[0][0])
    k = int(np.nonzero(np.abs(lon_wide - lon_ref[0]) < 1e-6)[0][0])
    assert out[0, 0, 0] == vals[0, j, k]


def test_conform_refuses_when_a_reference_cell_is_absent():
    lat = np.arange(20.0, 21.0, 0.1)
    lon = np.arange(74.0, 75.0, 0.1)
    vals = np.zeros((1, lat.size, lon.size))
    assert _conform(vals, lat, lon, np.array([99.0]), lon) is None


def test_imd_grid_constants_match_the_documented_grid():
    assert IMD_LAT.size == 129 and IMD_LON.size == 135
    assert np.isclose(IMD_LAT[0], 6.5) and np.isclose(IMD_LAT[-1], 38.5)
    assert np.isclose(IMD_LON[0], 66.5) and np.isclose(IMD_LON[-1], 100.0)


def test_imd_reader_shape_and_calendar():
    if not HAVE_IMD:
        print("      (skipped: raw IMD not present)")
        return
    v, lat, lon, d = read_imd_year(2020)          # leap year
    assert v.shape == (366, 129, 135), v.shape
    # IMD stamps each value with the day its window ENDS; the reader relabels to
    # the day the rain fell, so RF2020 runs 31 Dec 2019 .. 30 Dec 2020.
    assert str(d[0]) == "2019-12-31" and str(d[-1]) == "2020-12-30"
    assert np.all(np.diff(d) == np.timedelta64(1, "D"))
    assert np.allclose(lat, IMD_LAT) and np.allclose(lon, IMD_LON)


def test_imd_label_shift_is_one_day_back():
    """Pins the convention: IMD's value stamped D is rain that fell on D-1."""
    assert IMD_LABEL_SHIFT_DAYS == 1


def test_lag_check_detects_a_one_day_offset():
    """The build-time alignment check must catch exactly the error it exists for."""
    from common.build_dataset import lag_correlations
    rng = np.random.default_rng(0)
    truth = rng.gamma(0.4, 10.0, 400)
    # aligned: best at lag 0
    lc = lag_correlations(truth, truth + rng.normal(0, 1, 400))
    assert max(lc, key=lc.get) == 0
    # reference stamped one day late (IMD's convention): ref[t] = truth[t-1],
    # so the product lines up at lag -1 and the check must not report 0
    ref = np.concatenate([[0.0], truth[:-1]])
    lc = lag_correlations(ref, truth)
    assert max(lc, key=lc.get) == -1


def test_imd_fill_becomes_nan_not_a_negative_number():
    if not HAVE_IMD:
        print("      (skipped: raw IMD not present)")
        return
    v, *_ = read_imd_year(2022)
    assert np.isnan(v[0, 0, 0]), "the Indian Ocean corner should be NaN"
    assert np.nanmin(v) >= 0, "no -999 should survive as a value"


def test_imd_extremes_are_not_capped():
    """v1 set values above 500 mm/day to NaN. Five of nine years contain such days."""
    if not HAVE_IMD:
        print("      (skipped: raw IMD not present)")
        return
    v, *_ = read_imd_year(2022)
    assert np.nanmax(v) > 900, f"expected ~979 mm/day in 2022, got {np.nanmax(v):.1f}"


def test_imd_rejects_a_truncated_file(tmp=None):
    """A short file must raise rather than silently reshape into fewer days."""
    import tempfile
    if not HAVE_IMD:
        print("      (skipped: raw IMD not present)")
        return
    raw = np.fromfile(ROOT / "data/raw/imd/RF2021.GRD", dtype="<f4")
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "RF2021.GRD"
        raw[: 129 * 135 * 300].tofile(bad)        # 300 days instead of 365
        try:
            read_imd_year(2021, root=td)
        except ValueError as exc:
            assert "expected 365" in str(exc)
            return
    raise AssertionError("a truncated file should raise")


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
