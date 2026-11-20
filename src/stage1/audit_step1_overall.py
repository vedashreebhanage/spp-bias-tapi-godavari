"""
AUDIT STEP 1 - Overall bias, basin-wide, all days.

The simplest question in the study: across the whole basin and the whole record,
how does each product compare with IMD? No seasons, no elevation zones, no
episodes -- those are later steps, and mixing them in here makes it impossible to
tell which stratification is driving what.

Every number carries a block-bootstrap confidence interval and an honest N. The
cell-day count is printed for completeness but it is NOT the sample size: 302,404
cell-days come from 110 time blocks and about 12 spatial clusters, and quoting the
former is what made v1's intervals meaningless.

A sensitivity column re-runs everything with reference-suspect cells removed. Those
are cells where IMD is far wetter than its neighbours while all three products
independently disagree -- a pattern that points at the reference, not the products.
The rule is stated in SUSPECT_* below and applied mechanically; nothing is removed
by hand, and both columns are reported.

Run from the project root:
    python src/stage1/audit_step1_overall.py
    python src/stage1/audit_step1_overall.py --iters 200      # quicker draft
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1 import metrics as M                                    # noqa: E402
from stage1.bootstrap import (BLOCK_DAYS, block_bootstrap,         # noqa: E402
                              make_spatial_clusters, make_time_blocks,
                              percentile_ci)

DATASET = Path("data/processed/analysis_dataset.npz")
OUT = Path("data/processed/audit/step1_overall.json")
PRODUCTS = ["chirps", "era5", "imerg"]

# Reference-suspect rule (applied to 9-year mean annual totals per cell):
# IMD stands far above its neighbours AND no product agrees that it should.
SUSPECT_IMD_RATIO = 1.40      # IMD cell total vs mean of its 4 nearest cells
SUSPECT_PRODUCT_RATIO = 1.20  # every product must be below this to flag the cell
N_NEIGHBOURS = 4

# Metrics that get a bootstrap interval. The rest are reported as point estimates;
# these are the ones any conclusion would rest on.
CI_METRICS = {
    "pbias": M.pbias,
    "cc": M.corr,
    "kge": M.kge,
    "p95_bias": lambda o, s: M.quantile_bias(o, s, 95),
    "p99_bias": lambda o, s: M.quantile_bias(o, s, 99),
}


def find_suspect_cells(data, lat, lon, n_years):
    """Cells where IMD disagrees with every product at once. Returns (indices, table)."""
    totals = {p: np.nansum(data[p], axis=0) / n_years
              for p in ["imd"] + PRODUCTS}
    n = len(lat)
    ratios = {}
    for p, t in totals.items():
        r = np.empty(n)
        for k in range(n):
            d = np.hypot(lat - lat[k], lon - lon[k])
            d[k] = np.inf
            r[k] = t[k] / t[np.argsort(d)[:N_NEIGHBOURS]].mean()
        ratios[p] = r

    flag = ratios["imd"] > SUSPECT_IMD_RATIO
    for p in PRODUCTS:
        flag &= ratios[p] < SUSPECT_PRODUCT_RATIO
    idx = np.nonzero(flag)[0]

    table = [{
        "cell": int(k), "lat": float(lat[k]), "lon": float(lon[k]),
        "imd_mm_yr": float(totals["imd"][k]),
        "imd_vs_neighbours": float(ratios["imd"][k]),
        **{f"{p}_mm_yr": float(totals[p][k]) for p in PRODUCTS},
        **{f"{p}_vs_neighbours": float(ratios[p][k]) for p in PRODUCTS},
    } for k in idx]
    return idx, table


def evaluate(obs, sim, lat, lon, n_iter, seed=0):
    """Point estimates for everything, bootstrap intervals for the headline metrics."""
    point = M.all_metrics(obs, sim)
    blocks = make_time_blocks(obs.shape[0], BLOCK_DAYS)
    clusters = make_spatial_clusters(lat, lon)

    ci = {}
    for name, fn in CI_METRICS.items():
        # Same seed for every metric, so all intervals come from the SAME resamples
        # and can be read against each other.
        draws = block_bootstrap(obs, sim, fn, blocks, clusters,
                                n_iter=n_iter, seed=seed)
        lo, hi = percentile_ci(draws)
        ci[name] = {"lo": lo, "hi": hi}

    return point, ci, len(blocks), int(np.unique(clusters).size)


def fmt(point, ci, key, unit="", places=1):
    v = point[key]
    c = ci.get(key)
    if c is None or not np.isfinite(c["lo"]):
        return f"{v:.{places}f}{unit}"
    return f"{v:.{places}f}{unit} [{c['lo']:.{places}f}, {c['hi']:.{places}f}]"


def main():
    n_iter = 500
    if "--iters" in sys.argv:
        n_iter = int(sys.argv[sys.argv.index("--iters") + 1])

    if not DATASET.exists():
        sys.exit(f"FATAL: {DATASET} not found -- run src/common/build_dataset.py first.")
    d = np.load(DATASET, allow_pickle=True)
    lat, lon, dates = d["cell_lat"], d["cell_lon"], d["dates"]
    data = {p: d[p] for p in ["imd"] + PRODUCTS}

    print("=" * 78)
    print("AUDIT STEP 1 - overall bias, basin-wide, all days")
    print("=" * 78)
    print(f"  record      : {dates[0]} .. {dates[-1]}  ({len(dates)} days)")
    print(f"  cells       : {len(lat)}")
    print(f"  bootstrap   : {n_iter} iterations, {BLOCK_DAYS}-day time blocks")
    print("  reference   : IMD gridded rainfall (itself interpolated -- see caveat)")

    n_years = len(dates) / 365.25
    suspect, suspect_table = find_suspect_cells(data, lat, lon, n_years)
    print(f"\n  Reference-suspect cells (IMD > {SUSPECT_IMD_RATIO}x its neighbours "
          f"while every product is < {SUSPECT_PRODUCT_RATIO}x): {len(suspect)}")
    for row in suspect_table:
        print(f"    cell {row['cell']:2d}  lat {row['lat']:.2f} lon {row['lon']:.2f}  "
              f"IMD {row['imd_mm_yr']:.0f} mm/yr ({row['imd_vs_neighbours']:.2f}x)  "
              + "  ".join(f"{p.upper()} {row[f'{p}_mm_yr']:.0f} "
                          f"({row[f'{p}_vs_neighbours']:.2f}x)" for p in PRODUCTS))

    keep = np.setdiff1d(np.arange(len(lat)), suspect)
    variants = {"all cells": np.arange(len(lat))}
    if len(suspect):
        variants[f"minus {len(suspect)} suspect"] = keep

    results = {}
    for label, cells in variants.items():
        print("\n" + "-" * 78)
        print(f"  {label.upper()}  ({len(cells)} cells)")
        print("-" * 78)
        header = (f"  {'product':8s} {'PBIAS %':>22s} {'CC':>20s} {'KGE':>20s}"
                  f" {'RMSE':>7s} {'POD':>6s} {'FAR':>6s} {'CSI':>6s}"
                  f" {'P95 %':>20s} {'P99 %':>20s}")
        print(header)
        for p in PRODUCTS:
            obs = data["imd"][:, cells]
            sim = data[p][:, cells]
            point, ci, nblocks, nclust = evaluate(obs, sim, lat[cells], lon[cells], n_iter)
            results.setdefault(label, {})[p] = {
                "point": point, "ci": ci,
                "n_cells": int(len(cells)), "n_time_blocks": nblocks,
                "n_spatial_clusters": nclust,
            }
            print(f"  {p.upper():8s}"
                  f" {fmt(point, ci, 'pbias'):>22s}"
                  f" {fmt(point, ci, 'cc', places=3):>20s}"
                  f" {fmt(point, ci, 'kge', places=3):>20s}"
                  f" {point['rmse']:7.2f}"
                  f" {point['pod']:6.3f} {point['far']:6.3f} {point['csi']:6.3f}"
                  f" {fmt(point, ci, 'p95_bias'):>20s}"
                  f" {fmt(point, ci, 'p99_bias'):>20s}")

        any_p = results[label][PRODUCTS[0]]
        print(f"\n  N: {any_p['point']['n_pairs']:,} cell-days -- but the sample size "
              f"for inference is\n     {any_p['n_time_blocks']} time blocks x "
              f"{any_p['n_spatial_clusters']} spatial clusters. Quote the latter.")
        print(f"  IMD basin mean: {any_p['point']['obs_mean']:.3f} mm/day")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "step": "1 - overall bias, basin-wide, all days",
        "bootstrap_iterations": n_iter,
        "block_days": BLOCK_DAYS,
        "suspect_rule": {
            "imd_vs_neighbours_above": SUSPECT_IMD_RATIO,
            "every_product_vs_neighbours_below": SUSPECT_PRODUCT_RATIO,
            "n_neighbours": N_NEIGHBOURS,
        },
        "suspect_cells": suspect_table,
        "results": results,
    }, indent=2, default=float))
    print(f"\n  Wrote {OUT}")
    print("\n  Reminder: positive PBIAS = product wetter than IMD. IMD is not truth;")
    print("  it is interpolated from a sparse gauge network, and over hill terrain")
    print("  part of every number above is IMD's own interpolation error.")


if __name__ == "__main__":
    main()
