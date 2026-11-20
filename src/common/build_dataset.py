"""
STAGE 1 / STAGE 2 shared - Step 3: build the aligned analysis dataset.

Brings all four products onto the IMD 0.25 deg grid, masked to the 92 in-basin
cells, over 2015-01-01 .. 2023-12-30, and saves them as one array per product
plus a combined file. Everything downstream -- bias metrics, and the Stage 2
correction models -- reads these arrays and nothing else.

IMD is never regridded. It defines the grid, so its cells are extracted directly
by index; that is exact, not an approximation. The other three are area-weighted
onto the same cells by src/common/regrid.py (unit-tested in tests/test_regrid.py).

The integrity checks at the end are deliberately gross: date coverage, sign,
magnitude, coverage fraction. They are there to catch a units error, a shifted
calendar, or a transposed array -- failures that produce plausible-looking
rainfall and poison everything downstream. They are NOT results, and no bias
statistic is computed here; that belongs in the metrics module, after this output
has been inspected.

Run from the project root:
    python src/common/build_dataset.py
    python src/common/build_dataset.py --products imd,chirps      # subset
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import readers                      # noqa: E402
from common.regrid import apply_weights, build_weights   # noqa: E402

MASK_FILE = Path("data/processed/tapi_mask.npz")
OUT_DIR = Path("data/processed/aligned")
COMBINED = Path("data/processed/analysis_dataset.npz")
FIG_DIR = Path("data/processed")

YEARS = list(range(2015, 2024))
# The record ends on 30 Dec, not 31 Dec: IMD's value for 31 Dec 2023 is stamped
# 1 Jan 2024 (see readers.IMD_LABEL_SHIFT_DAYS) and lives in RF2024.GRD, which is
# outside the downloaded set. One dry-season day is not worth a tenth IMD file.
START, END = np.datetime64("2015-01-01"), np.datetime64("2023-12-30")

# Calendar-alignment check: the basin-mean series of every product must correlate
# best with IMD at lag 0. A one-day IMD labelling offset survived an eyeball check
# of the JJAS 2019 plot once already; it is now asserted instead.
MAX_LAG_DAYS = 2
ALL_PRODUCTS = ["imd", "chirps", "era5", "imerg"]

# A target cell whose source coverage falls below this on a given day is set to
# NaN rather than reported as a value derived from a sliver of the cell.
MIN_VALID_FRACTION = 0.5


def expected_dates():
    return np.arange(START, END + np.timedelta64(1, "D"), dtype="datetime64[D]")


def load_cells():
    if not MASK_FILE.exists():
        sys.exit("FATAL: no mask -- run src/stage1/build_mask.py first.")
    d = np.load(MASK_FILE)
    mask = d["mask"]
    ii, jj = np.nonzero(mask)
    return {
        "mask": mask,
        "i": ii, "j": jj,
        "lat": d["lat"][ii], "lon": d["lon"][jj],
        "elevation": d["elevation"][ii, jj],
        "ruggedness": d["ruggedness"][ii, jj],
        "zones": d["zones"][ii, jj],
        "n": len(ii),
    }


def align(values, valid, dates, want):
    """Reindex onto the required calendar, failing loudly on any missing day."""
    index = {d: k for k, d in enumerate(dates)}
    missing = [str(d) for d in want if d not in index]
    if missing:
        raise ValueError(
            f"{len(missing)} day(s) missing from the record, first few: "
            f"{missing[:5]}"
        )
    take = np.array([index[d] for d in want])
    return values[take], valid[take]


# ------------------------------------------------------------------ builders --
def build_imd(cells, want):
    print("  IMD  (reference grid -- extracted by index, not regridded)")
    vals, dates = [], []
    for year in YEARS:
        v, lat, lon, d = readers.read_imd_year(year)
        if not (np.allclose(lat, readers.IMD_LAT) and np.allclose(lon, readers.IMD_LON)):
            raise ValueError(f"IMD {year}: grid does not match the reference grid")
        vals.append(v[:, cells["i"], cells["j"]])
        dates.append(d)
    values = np.concatenate(vals)
    dates = np.concatenate(dates)
    valid = np.isfinite(values).astype(float)
    return align(values, valid, dates, want)


def _regrid_stream(name, chunks, cells):
    """Regrid successive (values, lat, lon, dates) chunks with shared weights."""
    weights, grid = None, None
    out_v, out_f, out_d = [], [], []
    for values, lat, lon, dates in chunks:
        if weights is None:
            weights = build_weights(lat, lon, cells["lat"], cells["lon"])
            grid = (lat, lon)
            per_cell = [w["w"].size for w in weights]
            print(f"    source grid {len(lat)}x{len(lon)}, "
                  f"{min(per_cell)}-{max(per_cell)} source pixels per target cell")
        elif not (np.allclose(lat, grid[0]) and np.allclose(lon, grid[1])):
            raise ValueError(f"{name}: grid changed between files")
        v, f = apply_weights(values, weights)
        out_v.append(v)
        out_f.append(f)
        out_d.append(dates)
    return np.concatenate(out_v), np.concatenate(out_f), np.concatenate(out_d)


def build_chirps(cells, want):
    print("  CHIRPS")
    chunks = (readers.read_chirps_year(y) for y in YEARS)
    v, f, d = _regrid_stream("CHIRPS", chunks, cells)
    return align(v, f, d, want)


def build_era5(cells, want):
    print("  ERA5-Land  (x1000 m->mm, time shifted back one day, latitude flipped)")
    v, f, d = _regrid_stream("ERA5", [readers.read_era5_all()], cells)
    return align(v, f, d, want)


def build_imerg(cells, want):
    print("  IMERG  (transposed from (time, lon, lat); 3,287 granules)")
    v, f, d = _regrid_stream("IMERG", [readers.read_imerg_all()], cells)
    return align(v, f, d, want)


BUILDERS = {"imd": build_imd, "chirps": build_chirps,
            "era5": build_era5, "imerg": build_imerg}


# -------------------------------------------------------------------- checks --
def lag_correlations(ref, other, max_lag=MAX_LAG_DAYS):
    """Correlation of two daily series with ``other`` shifted by -max_lag..+max_lag.

    Lag L pairs ref[t] with other[t + L]. A product whose calendar runs one day
    late relative to the reference peaks at L = +1; one day early, at L = -1.
    """
    ref = np.asarray(ref, dtype=float)
    other = np.asarray(other, dtype=float)
    n = len(ref)
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        a = ref[max_lag:n - max_lag]
        b = other[max_lag + lag:n - max_lag + lag]
        ok = np.isfinite(a) & np.isfinite(b)
        out[lag] = float(np.corrcoef(a[ok], b[ok])[0, 1])
    return out


def check_calendar(data):
    """Print lag correlations against IMD; return False if any product is off."""
    ok = True
    print("\n  Calendar alignment vs IMD (basin-mean daily series; best lag must be 0):")
    ref = np.nanmean(data["imd"][0], axis=1)
    for name, (v, _) in data.items():
        if name == "imd":
            continue
        lc = lag_correlations(ref, np.nanmean(v, axis=1))
        best = max(lc, key=lc.get)
        row = "  ".join(f"{lag:+d}:{r:.3f}" for lag, r in lc.items())
        print(f"    {name.upper():7s} {row}   best lag {best:+d}")
        if best != 0:
            print(f"    FAIL: {name.upper()} lines up with IMD at lag {best:+d} "
                  "-- a calendar/day-labelling error")
            ok = False
    return ok


def integrity_checks(data, cells, want):
    print("\n" + "=" * 74)
    print("INTEGRITY CHECKS  (gross sanity only -- these are not results)")
    print("=" * 74)
    ok = True

    for name, (v, f) in data.items():
        n_neg = int(np.nansum(v < -1e-9))
        n_nan = int(np.isnan(v).sum())
        worst_cell = np.nanmin(f.mean(axis=0)) if f.size else float("nan")
        print(f"\n  {name.upper()}")
        print(f"    shape                : {v.shape}  (days x cells)")
        print(f"    negative values      : {n_neg}")
        print(f"    NaN cell-days        : {n_nan}  ({100 * n_nan / v.size:.3f}%)")
        print(f"    max cell-day         : {np.nanmax(v):.1f} mm")
        print(f"    least-covered cell   : {worst_cell:.1%} of its area, on average")
        if n_neg:
            print("    FAIL: rainfall cannot be negative")
            ok = False
        if np.nanmax(v) > 2000:
            print("    FAIL: implausible daily total -- check units")
            ok = False

    if "imd" in data:
        imd_max = np.nanmax(data["imd"][0])
        print(f"\n  IMD extremes retained  : max {imd_max:.1f} mm/day in the basin")
        if imd_max < 100:
            print("    FAIL: suspiciously low -- has a cap been applied?")
            ok = False
        if len(data) > 1 and not check_calendar(data):
            ok = False

    print("\n  Annual basin-mean totals (mm) -- order-of-magnitude check for unit")
    print("  and calendar errors, NOT a bias result:\n")
    years = want.astype("datetime64[Y]").astype(int) + 1970
    header = "    year  " + "".join(f"{n.upper():>10s}" for n in data)
    print(header)
    for y in YEARS:
        sel = years == y
        row = "".join(f"{np.nansum(np.nanmean(v[sel], axis=1)):10.1f}"
                      for v, _ in data.values())
        print(f"    {y}  {row}")

    print("\n" + ("  ALL CHECKS PASSED" if ok else "  *** CHECKS FAILED ***"))
    return ok


def make_figures(data, cells, want):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(data)
    # 1. mean annual total per cell, one panel per product, shared colour scale
    totals = {k: np.nansum(v, axis=0) / len(YEARS) for k, (v, _) in data.items()}
    vmax = max(np.nanmax(t) for t in totals.values())
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (name, t) in zip(axes, totals.items()):
        sc = ax.scatter(cells["lon"], cells["lat"], c=t, s=180, marker="s",
                        cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"{name.upper()}\nmean annual total")
        ax.set_xlabel("lon")
        ax.set_aspect("equal")
    axes[0].set_ylabel("lat")
    fig.colorbar(sc, ax=axes, label="mm/year", shrink=0.85)
    fig.savefig(FIG_DIR / "aligned_annual_totals.png", dpi=110)
    plt.close(fig)

    # 2. one monsoon of daily basin means, to expose calendar shifts by eye
    sel = (want >= np.datetime64("2019-06-01")) & (want <= np.datetime64("2019-09-30"))
    fig, ax = plt.subplots(figsize=(13, 4.5), constrained_layout=True)
    for name, (v, _) in data.items():
        ax.plot(want[sel], np.nanmean(v[sel], axis=1), lw=1.4, label=name.upper(),
                alpha=0.85)
    ax.set_title("Basin-mean daily rainfall, JJAS 2019 — peaks must line up in time")
    ax.set_ylabel("mm/day")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(FIG_DIR / "aligned_timeseries_2019.png", dpi=110)
    plt.close(fig)
    print(f"\n  Wrote {FIG_DIR / 'aligned_annual_totals.png'}")
    print(f"  Wrote {FIG_DIR / 'aligned_timeseries_2019.png'}")


# ---------------------------------------------------------------------- main --
def main():
    products = ALL_PRODUCTS
    for arg in sys.argv[1:]:
        if arg.startswith("--products"):
            spec = arg.split("=", 1)[1] if "=" in arg else sys.argv[sys.argv.index(arg) + 1]
            products = [p.strip() for p in spec.split(",")]
    unknown = set(products) - set(ALL_PRODUCTS)
    if unknown:
        sys.exit(f"Unknown product(s): {sorted(unknown)}")

    cells = load_cells()
    want = expected_dates()
    print(f"Target: {cells['n']} basin cells x {len(want)} days "
          f"({want[0]} .. {want[-1]})\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    for name in products:
        values, valid = BUILDERS[name](cells, want)
        thin = valid < MIN_VALID_FRACTION
        if thin.any():
            print(f"    {int(thin.sum())} cell-days below "
                  f"{MIN_VALID_FRACTION:.0%} coverage set to NaN")
            values = np.where(thin, np.nan, values)
        data[name] = (values, valid)
        np.savez_compressed(
            OUT_DIR / f"{name}.npz",
            values=values.astype("float32"),
            valid_fraction=valid.astype("float32"),
            dates=want.astype("datetime64[D]"),
            cell_lat=cells["lat"], cell_lon=cells["lon"],
            cell_i=cells["i"], cell_j=cells["j"],
        )
        print(f"    saved {OUT_DIR / f'{name}.npz'}")

    ok = integrity_checks(data, cells, want)
    make_figures(data, cells, want)

    if set(products) == set(ALL_PRODUCTS):
        np.savez_compressed(
            COMBINED,
            dates=want.astype("datetime64[D]"),
            cell_lat=cells["lat"], cell_lon=cells["lon"],
            cell_i=cells["i"], cell_j=cells["j"],
            elevation=cells["elevation"], ruggedness=cells["ruggedness"],
            zones=cells["zones"],
            generated_utc=np.array(
                datetime.now(timezone.utc).isoformat(timespec="seconds")),
            **{p: data[p][0].astype("float32") for p in ALL_PRODUCTS},
            **{f"{p}_valid": data[p][1].astype("float32") for p in ALL_PRODUCTS},
        )
        print(f"\n  Wrote {COMBINED}")

    print("\nInspect aligned_timeseries_2019.png before using these arrays: if a "
          "product's peaks sit a day off the others, its calendar handling is wrong.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
