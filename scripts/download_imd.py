"""
Download IMD 0.25 deg gridded daily rainfall, 2015-2023.

    pip install imdlib
    python download_imd.py

Writes RF{year}.GRD into data/raw/imd/ and verifies file sizes against the
expected byte count for a (n_days, 129, 135) float32 array.

If imdlib fails (IMD changes their endpoint from time to time), download
manually from:
    https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html
and drop the .GRD files into the same folder, then re-run this script with
--verify-only to check them.
"""

import argparse
import calendar
import sys
from pathlib import Path

OUT_DIR = Path("data/raw/imd")
START_YEAR, END_YEAR = 2015, 2023

N_LAT, N_LON = 129, 135          # IMD 0.25 deg grid: 6.5-38.5 N, 66.5-100.0 E
BYTES_PER_VALUE = 4              # float32


def expected_bytes(year: int) -> int:
    n_days = 366 if calendar.isleap(year) else 365
    return n_days * N_LAT * N_LON * BYTES_PER_VALUE


def verify(out_dir: Path) -> bool:
    print("\n=== Verifying IMD files ===")
    all_ok = True
    for year in range(START_YEAR, END_YEAR + 1):
        path = out_dir / f"RF{year}.GRD"
        if not path.exists():
            print(f"  {year}  MISSING")
            all_ok = False
            continue
        actual, expect = path.stat().st_size, expected_bytes(year)
        if actual == expect:
            print(f"  {year}  OK        {actual:,} bytes")
        else:
            diff = actual - expect
            print(f"  {year}  SIZE MISMATCH  got {actual:,}, expected {expect:,} "
                  f"({diff:+,}) -- your assumed grid shape or dtype is wrong, "
                  f"or the file is truncated")
            all_ok = False
    return all_ok


def download(out_dir: Path) -> None:
    try:
        import imdlib as imd
    except ImportError:
        sys.exit("imdlib not installed.  Run:  pip install imdlib")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading IMD rainfall {START_YEAR}-{END_YEAR} into {out_dir.resolve()}")
    print("(IMD's server is slow; expect several minutes.)\n")

    # variable 'rain' = 0.25 deg gridded rainfall; 'yearwise' gives one .GRD per year
    imd.get_data("rain", START_YEAR, END_YEAR,
                 fn_format="yearwise", file_dir=str(out_dir.parent.parent.parent))

    # imdlib nests output under <file_dir>/rain/ -- move files up if needed
    nested = out_dir.parent.parent.parent / "rain"
    if nested.exists():
        for f in nested.glob("*.GRD"):
            target = out_dir / f.name
            if not target.exists():
                f.replace(target)
        print(f"Moved files from {nested} to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true",
                    help="skip download, just check files already present")
    args = ap.parse_args()

    if not args.verify_only:
        download(OUT_DIR)

    ok = verify(OUT_DIR)
    print("\nAll files present and correctly sized."
          if ok else
          "\nSome files are missing or malformed -- see above before proceeding.")
