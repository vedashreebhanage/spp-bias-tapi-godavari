"""
Find the wettest multi-day episodes in the basin from IMD gridded rainfall.

Purpose: the flood windows in scripts/basins.py were placeholders. Event dates
must be defensible. This script provides the *rainfall* evidence -- the basin-mean
and basin-max daily totals actually recorded by the reference dataset -- so that
candidate windows can be checked against IMD rather than assumed.

It does NOT define flood events by itself: a flood is a river-stage phenomenon,
and the final windows must still be tied to a CWC/IMD/press citation. This
narrows the search and catches any placeholder that sits on a dry period.

Run from the project root:
    python scripts/find_wet_episodes.py
"""

import sys
from pathlib import Path

import numpy as np

IMD_DIR = Path("data/raw/imd")
MASK_FILE = Path("data/processed/tapi_mask.npz")
YEARS = range(2015, 2024)
NLAT, NLON = 129, 135
FILL = -999.0
TOP_N = 6          # episodes to report per year
WINDOW = 3         # days, for the rolling-sum ranking


def load_year(year, mask):
    path = IMD_DIR / f"RF{year}.GRD"
    raw = np.fromfile(path, dtype="<f4")
    ndays, rem = divmod(raw.size, NLAT * NLON)
    if rem:
        sys.exit(f"FATAL: {path} size {raw.size} is not a multiple of {NLAT*NLON}")
    expected = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    if ndays != expected:
        sys.exit(f"FATAL: {path} has {ndays} days, expected {expected}")
    arr = raw.reshape(ndays, NLAT, NLON).astype("float64")
    arr[arr == FILL] = np.nan
    return arr[:, mask], ndays          # (ndays, ncells)


def main():
    if not MASK_FILE.exists():
        sys.exit("FATAL: run src/stage1/build_mask.py first -- no mask found.")
    mask = np.load(MASK_FILE)["mask"]
    ncells = int(mask.sum())
    print(f"Basin mask: {ncells} cells\n")

    all_rows = []
    for year in YEARS:
        cells, ndays = load_year(year, mask)
        if np.isnan(cells).any():
            n = int(np.isnan(cells).sum())
            print(f"  note: {year} has {n} NaN cell-days inside the basin")
        mean = np.nanmean(cells, axis=1)
        cmax = np.nanmax(cells, axis=1)
        dates = np.arange(f"{year}-01-01", f"{year+1}-01-01", dtype="datetime64[D]")

        # rolling WINDOW-day sum of the basin mean, labelled by window start
        roll = np.convolve(mean, np.ones(WINDOW), mode="valid")

        print(f"{year}: annual basin mean {mean.sum():7.1f} mm | "
              f"wettest day {mean.max():6.1f} mm (basin-mean) on {dates[mean.argmax()]}")

        order = np.argsort(roll)[::-1]
        chosen = []
        for i in order:                      # greedy, non-overlapping episodes
            if all(abs(int(i) - int(j)) >= WINDOW for j in chosen):
                chosen.append(int(i))
            if len(chosen) == TOP_N:
                break
        for i in sorted(chosen):
            sl = slice(i, i + WINDOW)
            print(f"    {dates[i]} .. {dates[i+WINDOW-1]}  "
                  f"{WINDOW}-day basin mean {roll[i]:6.1f} mm | "
                  f"peak day {mean[sl].max():5.1f} mm | "
                  f"wettest cell {cmax[sl].max():6.1f} mm")
            all_rows.append((float(roll[i]), str(dates[i]), str(dates[i+WINDOW-1])))
        print()

    print("=" * 64)
    print(f"Top 12 {WINDOW}-day episodes across all years (basin-mean mm):")
    for tot, a, b in sorted(all_rows, reverse=True)[:12]:
        print(f"  {a} .. {b}   {tot:6.1f}")


if __name__ == "__main__":
    main()
