"""
AUDIT STEP 2 - Bias by season x elevation zone.

Step 1 asked how each product compares with IMD over the whole basin and record.
This step splits that along the two axes the study is about:

  * SEASON -- JJAS (monsoon), ON (post-monsoon), DJF (winter), MAM (pre-monsoon).
    ~88% of the basin's rain falls in JJAS, so the all-days numbers of step 1 are
    essentially monsoon numbers. The other seasons are reported so that is visible,
    not so they can carry conclusions: their totals are small, and a percent bias
    of a small total is large and unstable. Each season's IMD mean and wet-day
    count are printed beside it for that reason.

  * ELEVATION ZONE -- terciles of in-basin cell elevation (lower / middle / upper,
    31 / 30 / 31 cells), fixed in build_mask.py. This is a DESCRIPTIVE summary.
    The test of whether bias changes with elevation is step 3, which treats
    elevation as continuous; three bins throw information away and the cut points
    are arbitrary.

Uncertainty
-----------
For each season, one set of bootstrap draws is made (time blocks within that
season, split at year boundaries, x geographic clusters of cells) and EVERY
product and zone is evaluated on the same draws. That is what allows the
"upper - lower" contrast to carry an interval: both zones see the same storms in
the same resample. A zone can be absent from a draw (its cells all fall in
clusters that were not picked); those draws give NaN for that zone and are
dropped from its interval, and the count is recorded.

Run from the project root:
    python src/stage1/audit_step2_season_zone.py
    python src/stage1/audit_step2_season_zone.py --iters 100     # quicker draft
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1 import metrics as M                                     # noqa: E402
from stage1.bootstrap import (BLOCK_DAYS, make_spatial_clusters,    # noqa: E402
                              make_subset_blocks, percentile_ci,
                              resample_indices)

DATASET = Path("data/processed/analysis_dataset.npz")
OUT = Path("data/processed/audit/step2_season_zone.json")
FIG = Path("data/processed/audit/step2_season_zone.png")
PRODUCTS = ["chirps", "era5", "imerg"]

SEASONS = {
    "JJAS": (6, 7, 8, 9),
    "ON": (10, 11),
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
}
ZONES = {1: "lower", 2: "middle", 3: "upper"}
GROUPS = ["all", "lower", "middle", "upper"]

# Metrics given bootstrap intervals; everything else is a point estimate.
CI_METRICS = {
    "pbias": M.pbias,
    "cc": M.corr,
    "kge": M.kge,
    "p95_bias": lambda o, s: M.quantile_bias(o, s, 95),
    "p99_bias": lambda o, s: M.quantile_bias(o, s, 99),
}
QUANTILE_METRICS = {"p95_bias": 95, "p99_bias": 99}
# Contrasts between zones, evaluated on the same draw.
CONTRASTS = {"upper_minus_lower": ("upper", "lower")}
CONTRAST_METRICS = ["pbias", "cc"]


def season_days(dates, months):
    m = dates.astype("datetime64[M]").astype(int) % 12 + 1
    return np.nonzero(np.isin(m, months))[0]


def group_cells(zones, group, cell_idx=None):
    """Cells of a group, optionally within one bootstrap draw (keeping repeats)."""
    idx = np.arange(len(zones)) if cell_idx is None else cell_idx
    if group == "all":
        return idx
    z = {v: k for k, v in ZONES.items()}[group]
    return idx[zones[idx] == z]


def evaluate_season(data, zones, clusters, days, n_iter, seed):
    """Point estimates and shared-draw intervals for one season."""
    blocks = make_subset_blocks(days, BLOCK_DAYS)
    obs_all = data["imd"]

    point = {p: {g: M.all_metrics(obs_all[np.ix_(days, group_cells(zones, g))],
                                  data[p][np.ix_(days, group_cells(zones, g))])
                 for g in GROUPS} for p in PRODUCTS}

    draws = {p: {g: {m: np.full(n_iter, np.nan) for m in CI_METRICS}
                 for g in GROUPS} for p in PRODUCTS}
    missing = {g: 0 for g in GROUPS}

    for it, (d_idx, c_idx) in enumerate(
            resample_indices(blocks, clusters, n_iter, seed)):
        cells_by_group = {g: group_cells(zones, g, c_idx) for g in GROUPS}
        for g, cells in cells_by_group.items():
            if cells.size == 0:
                missing[g] += 1
        o_by_group = {g: obs_all[np.ix_(d_idx, c)] for g, c in cells_by_group.items()
                      if c.size}
        for p in PRODUCTS:
            sim = data[p]
            for g, cells in cells_by_group.items():
                if not cells.size:
                    continue
                o = o_by_group[g]
                s = sim[np.ix_(d_idx, cells)]
                for m, fn in CI_METRICS.items():
                    draws[p][g][m][it] = fn(o, s)

    # A quantile bias is only meaningful if IMD's quantile is itself a rain day.
    # In the dry seasons IMD's 95th-percentile cell-day can be 0.2 mm, and a
    # "bias" relative to that runs into thousands of percent while saying nothing
    # about extremes. Such strata are reported as n/a (point AND interval), by a
    # rule fixed here rather than by inspecting which numbers look silly.
    suppressed = []
    for g in GROUPS:
        o = obs_all[np.ix_(days, group_cells(zones, g))]
        for m, q in QUANTILE_METRICS.items():
            if np.percentile(o, q) < M.RAIN_THRESHOLD_MM:
                suppressed.append(f"{g}:{m}")
                for p in PRODUCTS:
                    point[p][g][m] = np.nan
                    draws[p][g][m][:] = np.nan

    ci = {p: {g: {m: dict(zip(("lo", "hi"), percentile_ci(draws[p][g][m])))
                  for m in CI_METRICS} for g in GROUPS} for p in PRODUCTS}

    contrasts = {}
    for name, (a, b) in CONTRASTS.items():
        for p in PRODUCTS:
            for m in CONTRAST_METRICS:
                diff = draws[p][a][m] - draws[p][b][m]
                lo, hi = percentile_ci(diff)
                contrasts.setdefault(p, {})[f"{name}_{m}"] = {
                    "point": point[p][a][m] - point[p][b][m],
                    "lo": lo, "hi": hi,
                    "excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
                }

    return point, ci, contrasts, missing, len(blocks), suppressed


def fmt(v, c, places=1):
    if not np.isfinite(v):
        return "n/a"
    if c is None or not np.isfinite(c["lo"]):
        return f"{v:.{places}f}"
    return f"{v:.{places}f} [{c['lo']:.{places}f}, {c['hi']:.{places}f}]"


def make_figure(results, season_names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"lower": "#4c9f70", "middle": "#d9a441", "upper": "#8e5a9e"}
    offsets = {"lower": -0.2, "middle": 0.0, "upper": 0.2}
    fig, axes = plt.subplots(2, len(PRODUCTS), figsize=(14, 7.5),
                             constrained_layout=True, sharey="row")
    for col, p in enumerate(PRODUCTS):
        for row, (metric, label) in enumerate([("pbias", "PBIAS (%)"),
                                               ("cc", "correlation")]):
            ax = axes[row, col]
            for z in ["lower", "middle", "upper"]:
                xs, ys, lo, hi = [], [], [], []
                for k, s in enumerate(season_names):
                    r = results[s]
                    v = r["point"][p][z][metric]
                    c = r["ci"][p][z][metric]
                    xs.append(k + offsets[z])
                    ys.append(v)
                    lo.append(v - c["lo"] if np.isfinite(c["lo"]) else 0)
                    hi.append(c["hi"] - v if np.isfinite(c["hi"]) else 0)
                ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", capsize=3,
                            color=colours[z], label=z)
            if metric == "pbias":
                ax.axhline(0, color="grey", lw=0.8)
            ax.set_xticks(range(len(season_names)))
            ax.set_xticklabels(season_names)
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel(label)
            if row == 0:
                ax.set_title(p.upper())
    axes[0, 0].legend(title="elevation zone")
    fig.suptitle("Bias by season and elevation zone vs IMD (95% block-bootstrap CIs)")
    fig.savefig(FIG, dpi=110)
    plt.close(fig)


def main():
    n_iter = 500
    if "--iters" in sys.argv:
        n_iter = int(sys.argv[sys.argv.index("--iters") + 1])
    seed = 0

    if not DATASET.exists():
        sys.exit(f"FATAL: {DATASET} not found -- run src/common/build_dataset.py first.")
    d = np.load(DATASET, allow_pickle=True)
    lat, lon, dates = d["cell_lat"], d["cell_lon"], d["dates"]
    zones = d["zones"].astype(int)
    elev = d["elevation"]
    data = {p: d[p].astype("float64") for p in ["imd"] + PRODUCTS}
    clusters = make_spatial_clusters(lat, lon)

    print("=" * 78)
    print("AUDIT STEP 2 - bias by season x elevation zone")
    print("=" * 78)
    print(f"  record    : {dates[0]} .. {dates[-1]}  ({len(dates)} days), "
          f"{len(lat)} cells")
    print(f"  bootstrap : {n_iter} draws per season, {BLOCK_DAYS}-day blocks within "
          f"season, {np.unique(clusters).size} spatial clusters; same draws for "
          "every product and zone")
    zone_info = {}
    for z, name in ZONES.items():
        e = elev[zones == z]
        zone_info[name] = {"n_cells": int(e.size), "elev_min_m": float(e.min()),
                           "elev_max_m": float(e.max()), "elev_mean_m": float(e.mean())}
        print(f"  zone {name:6s}: {e.size:2d} cells, {e.min():4.0f}-{e.max():4.0f} m "
              f"(mean {e.mean():.0f} m)")

    results = {}
    for season, months in SEASONS.items():
        days = season_days(dates, months)
        imd = data["imd"][days]
        basin = imd.mean(axis=1)
        print("\n" + "-" * 78)
        print(f"  {season}  -- {days.size} days | IMD basin mean {imd.mean():.2f} mm/day"
              f" | basin wet days (>= 1 mm) {int((basin >= 1).sum())}"
              f" | share of annual rain {imd.sum() / data['imd'].sum():.1%}")
        print("-" * 78)

        point, ci, contrasts, missing, nblocks, suppressed = evaluate_season(
            data, zones, clusters, days, n_iter, seed)
        results[season] = {
            "n_days": int(days.size), "n_time_blocks": nblocks,
            "quantile_bias_suppressed": suppressed,
            "imd_mean_mm_day": float(imd.mean()),
            "imd_share_of_annual": float(imd.sum() / data["imd"].sum()),
            "basin_wet_days": int((basin >= 1).sum()),
            "draws_missing_zone": missing,
            "point": point, "ci": ci, "contrasts": contrasts,
        }

        print(f"  {'':8s} {'zone':7s} {'PBIAS %':>22s} {'CC':>20s} {'KGE':>20s}"
              f" {'POD':>6s} {'FAR':>6s} {'P95 %':>20s}")
        for p in PRODUCTS:
            for g in GROUPS:
                pt, c = point[p][g], ci[p][g]
                print(f"  {p.upper() if g == 'all' else '':8s} {g:7s}"
                      f" {fmt(pt['pbias'], c['pbias']):>22s}"
                      f" {fmt(pt['cc'], c['cc'], 3):>20s}"
                      f" {fmt(pt['kge'], c['kge'], 3):>20s}"
                      f" {pt['pod']:6.3f} {pt['far']:6.3f}"
                      f" {fmt(pt['p95_bias'], c['p95_bias']):>20s}")
            con = contrasts[p]
            for m in CONTRAST_METRICS:
                k = con[f"upper_minus_lower_{m}"]
                flag = "  <- CI excludes 0" if k["excludes_zero"] else ""
                print(f"  {'':8s} {'up-low':7s} {m:>5s} "
                      f"{fmt(k['point'], k, 1 if m == 'pbias' else 3)}{flag}")
        if any(missing.values()):
            print(f"  draws with a zone absent: {missing}")
        if suppressed:
            print(f"  quantile bias n/a (IMD quantile < {M.RAIN_THRESHOLD_MM} mm): "
                  f"{', '.join(suppressed)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "step": "2 - bias by season x elevation zone (descriptive)",
        "bootstrap_iterations": n_iter,
        "block_days": BLOCK_DAYS,
        "seed": seed,
        "seasons": {k: list(v) for k, v in SEASONS.items()},
        "zones": zone_info,
        "note": ("Zones are elevation terciles and descriptive only; the elevation "
                 "test is step 3. Percent biases in DJF/MAM/ON rest on small totals."),
        "results": results,
    }, indent=2, default=float))
    make_figure(results, list(SEASONS))
    print(f"\n  Wrote {OUT}")
    print(f"  Wrote {FIG}")
    print("\n  Positive PBIAS = product wetter than IMD. IMD is interpolated from a")
    print("  sparse gauge network; over the upper zone especially, part of every")
    print("  number above is IMD's own error.")


if __name__ == "__main__":
    main()
