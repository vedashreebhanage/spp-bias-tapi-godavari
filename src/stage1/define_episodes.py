"""
STAGE 1 - Step 2: Define the extreme-rainfall episode set (and its control days).

Why this file exists
--------------------
`scripts/basins.py` previously carried hardcoded "flood windows" marked
UNVERIFIED. Checking them against IMD showed the 2021 window (Aug 1-20) was
almost entirely dry -- only its last three days were wet -- so an "event" PBIAS
computed over it would mostly have measured dry-day behaviour. Hand-picked
windows are also unfalsifiable: a reviewer cannot tell whether they were chosen
before or after seeing the bias results.

So episodes are DERIVED, by a rule fixed in advance and recorded here:

    * score every 3-day window in 2015-2023 by basin-mean IMD rainfall
      (mean over the 92 masked cells, summed over the 3 days);
    * take the top N greedily, requiring >= MIN_SEPARATION_DAYS between the
      start of any two selected windows, so one storm cannot occupy several
      slots via its own shoulders;
    * that's it. No manual additions, no manual removals.

Naming: these are "extreme rainfall episodes", NOT flood events. A flood is a
river-stage phenomenon, and Ukai's regulation partly decouples basin rainfall
from stage at Surat. The satellite products only ever observe rainfall, so
rainfall is what this study can honestly claim to be measuring bias in.
Documented flood confirmations are recorded as an ANNOTATION on an episode
(see CONFIRMATIONS) and never as part of the selection rule.

Control days -- FIXED BEFORE ANY BIAS RESULT IS INSPECTED
--------------------------------------------------------
For the episode-vs-normal comparison, control days are:

    * in the same year as the episode,
    * inside the monsoon season (1 Jun - 30 Sep),
    * wet, defined as basin-mean >= CONTROL_WET_THRESHOLD_MM,
    * not inside any episode, and not within CONTROL_BUFFER_DAYS of one.

The buffer keeps the storm's own build-up and decay out of the "normal"
comparison group. This rule is recorded here so it cannot be quietly retuned
once the numbers are visible.

Run from the project root:
    python src/stage1/define_episodes.py            # select, report, write JSON
    python src/stage1/define_episodes.py --report   # per-year diagnostic only
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.readers import read_imd_year  # noqa: E402

IMD_DIR = Path("data/raw/imd")
MASK_FILE = Path("data/processed/tapi_mask.npz")
OUT_FILE = Path("data/processed/rainfall_episodes.json")

YEARS = list(range(2015, 2024))

# ------------------------------------------------------- the rule (fixed) --
WINDOW_DAYS = 3
N_EPISODES = 10
MIN_SEPARATION_DAYS = 7
CONTROL_SEASON = (6, 9)              # inclusive month range, JJAS
CONTROL_WET_THRESHOLD_MM = 1.0       # basin-mean; matches the POD/FAR rain-day threshold
CONTROL_BUFFER_DAYS = 3

# Independent corroboration for episodes that coincide with a documented flood.
# Annotation only -- these never influence selection. Keyed by the documented
# OCCURRENCE date, and attached to whichever episode window contains that date
# (not matched on the window's start, which depends on how IMD days are labelled).
CONFIRMATIONS = {
    "2022-07-12": (
        "Sentinel Asia emergency observation request (requester: ISRO), "
        "occurrence date 12 July 2022, GLIDE FL-2022-000213-IND; worst-affected "
        "districts Navsari, Valsad, Dang, Narmada, Chhota Udepur, Panchmahal. "
        "https://sentinel-asia.org/EO/2022/article20220713IN.html"
    ),
}

# CWC gauge records (Tapi at Burhanpur / Sarangkheda / Ghala) would allow a
# stage-based definition and are the stronger source. As of 2026-08-12 the CWC
# Water Year Book index returns HTTP 401 and only the 2012-13 and 2017-18
# editions are reachable, neither covering the study period. If those records
# are obtained later, add a stage-based episode set alongside this one rather
# than replacing it -- the comparison is itself informative.


def load_year(year, mask):
    """Return (basin_mean_per_day, basin_max_per_day, dates) for one IMD year file.

    Goes through the shared reader so that IMD's date relabelling (its value
    stamped D is rain that fell on D-1; see readers.IMD_LABEL_SHIFT_DAYS) happens
    in exactly one place. A private copy of the reader here is how the episode
    dates and the analysis dataset could silently disagree by a day.
    """
    path = IMD_DIR / f"RF{year}.GRD"
    if not path.exists():
        sys.exit(f"FATAL: missing {path}")
    try:
        arr, _, _, dates = read_imd_year(year, root=IMD_DIR)
    except ValueError as exc:
        sys.exit(f"FATAL: {exc}")
    cells = arr[:, mask]
    if np.isnan(cells).any():
        sys.exit(f"FATAL: {year} has NaN inside the basin mask -- investigate before proceeding")
    return cells.mean(axis=1), cells.max(axis=1), dates


def load_all(mask):
    means, maxes, dates = [], [], []
    for year in YEARS:
        m, x, d = load_year(year, mask)
        means.append(m)
        maxes.append(x)
        dates.append(d)
    mean, cmax, dates = (np.concatenate(means), np.concatenate(maxes),
                         np.concatenate(dates))
    # After IMD's relabelling, RF2015 starts on 2014-12-31; keep the study period.
    keep = dates >= np.datetime64(f"{YEARS[0]}-01-01")
    return mean[keep], cmax[keep], dates[keep]


def select_episodes(mean, dates):
    """Greedy top-N non-overlapping WINDOW_DAYS windows by basin-mean total."""
    roll = np.convolve(mean, np.ones(WINDOW_DAYS), mode="valid")
    # a window must not straddle a year boundary (files are contiguous but years are not events)
    years = dates.astype("datetime64[Y]").astype(int)
    valid = years[: len(roll)] == years[WINDOW_DAYS - 1 :]
    roll = np.where(valid, roll, -np.inf)

    chosen = []
    for i in np.argsort(roll)[::-1]:
        if not np.isfinite(roll[i]):
            break
        if all(abs(int(i) - j) >= MIN_SEPARATION_DAYS for j in chosen):
            chosen.append(int(i))
        if len(chosen) == N_EPISODES:
            break
    if len(chosen) < N_EPISODES:
        sys.exit(f"FATAL: only {len(chosen)} episodes met the separation rule")
    return sorted(chosen, key=lambda i: -roll[i]), roll


def build_control(mean, dates, episode_starts):
    """Boolean arrays: which days are episode days, which are control days."""
    n = len(mean)
    is_episode = np.zeros(n, bool)
    is_buffered = np.zeros(n, bool)
    for i in episode_starts:
        is_episode[i : i + WINDOW_DAYS] = True
        lo = max(0, i - CONTROL_BUFFER_DAYS)
        hi = min(n, i + WINDOW_DAYS + CONTROL_BUFFER_DAYS)
        is_buffered[lo:hi] = True

    months = dates.astype("datetime64[M]").astype(int) % 12 + 1
    in_season = (months >= CONTROL_SEASON[0]) & (months <= CONTROL_SEASON[1])
    is_control = in_season & (mean >= CONTROL_WET_THRESHOLD_MM) & ~is_buffered
    return is_episode, is_control


def report_per_year(mean, dates):
    print("Per-year diagnostic (basin mean over masked cells)\n")
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    for year in YEARS:
        sel = years == year
        m = mean[sel]
        d = dates[sel]
        print(f"  {year}: annual total {m.sum():7.1f} mm | "
              f"wettest day {m.max():5.1f} mm on {d[m.argmax()]} | "
              f"days >= 1 mm: {(m >= 1).sum():3d}")
    print()


def main():
    if not MASK_FILE.exists():
        sys.exit("FATAL: no mask found -- run src/stage1/build_mask.py first.")
    mask = np.load(MASK_FILE)["mask"]
    ncells = int(mask.sum())

    mean, cmax, dates = load_all(mask)
    print(f"Basin mask: {ncells} cells | {len(dates)} days, "
          f"{dates[0]} .. {dates[-1]}\n")
    report_per_year(mean, dates)

    if "--report" in sys.argv:
        return

    order, roll = select_episodes(mean, dates)
    is_episode, is_control = build_control(mean, dates, order)

    print(f"Top {N_EPISODES} {WINDOW_DAYS}-day extreme rainfall episodes "
          f"(>= {MIN_SEPARATION_DAYS} days apart):\n")
    episodes = []
    for rank, i in enumerate(order, 1):
        sl = slice(i, i + WINDOW_DAYS)
        peak = int(np.argmax(mean[sl])) + i
        rec = {
            "rank": rank,
            "start": str(dates[i]),
            "end": str(dates[i + WINDOW_DAYS - 1]),
            "basin_mean_total_mm": round(float(roll[i]), 2),
            "peak_day": str(dates[peak]),
            "peak_day_basin_mean_mm": round(float(mean[peak]), 2),
            "max_cell_day_mm": round(float(cmax[sl].max()), 2),
        }
        note = next((text for day, text in CONFIRMATIONS.items()
                     if rec["start"] <= day <= rec["end"]), None)
        if note:
            rec["documented_flood"] = note
        episodes.append(rec)
        flag = "  [documented flood]" if note else ""
        print(f"  {rank:2d}. {rec['start']} .. {rec['end']}  "
              f"total {rec['basin_mean_total_mm']:6.1f} mm | "
              f"peak day {rec['peak_day_basin_mean_mm']:5.1f} mm | "
              f"wettest cell {rec['max_cell_day_mm']:6.1f} mm{flag}")

    years = dates.astype("datetime64[Y]").astype(int) + 1970
    print(f"\nEpisode days: {int(is_episode.sum())}  "
          f"({int(is_episode.sum())} = {N_EPISODES} x {WINDOW_DAYS})")
    print(f"Control days: {int(is_control.sum())} "
          f"(JJAS, basin mean >= {CONTROL_WET_THRESHOLD_MM} mm, "
          f"excluding episodes +/- {CONTROL_BUFFER_DAYS} d)")
    print("  per year: " + ", ".join(
        f"{y}:{int(is_control[years == y].sum())}" for y in YEARS))

    per_year = {str(y): sum(1 for e in episodes if e["start"].startswith(str(y)))
                for y in YEARS}
    print("Episodes per year: " + ", ".join(f"{y}:{n}" for y, n in per_year.items()))

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "IMD 0.25 deg gridded rainfall, masked to Tapi basin",
        "mask_cells": ncells,
        "definition": "extreme rainfall episodes (NOT stage-defined floods)",
        "rule": {
            "window_days": WINDOW_DAYS,
            "n_episodes": N_EPISODES,
            "min_separation_days": MIN_SEPARATION_DAYS,
            "score": "basin-mean daily rainfall summed over the window",
            "windows_may_cross_year_boundary": False,
        },
        "control_rule": {
            "season_months": list(CONTROL_SEASON),
            "wet_threshold_mm": CONTROL_WET_THRESHOLD_MM,
            "buffer_days": CONTROL_BUFFER_DAYS,
            "same_year_as_episode": True,
            "n_control_days": int(is_control.sum()),
            "n_control_days_per_year": {str(y): int(is_control[years == y].sum())
                                        for y in YEARS},
            "fixed_before_results_inspected": True,
        },
        "episodes": episodes,
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT_FILE}")


if __name__ == "__main__":
    main()
