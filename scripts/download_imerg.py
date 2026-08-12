"""
IMERG V07 daily -- spatially subsetted download from NASA GES DISC.

Why a URL list rather than a direct API call
--------------------------------------------
IMERG daily granules are GLOBAL (~29 MB each). Nine years = ~95 GB.
Subsetted to a basin bbox they are ~0.1 MB each = ~350 MB. Always subset.
GES DISC does the subsetting server-side and gives you a list of URLs.

Step 1 -- print the exact settings to paste into the subsetter
--------------------------------------------------------------
  python scripts/download_imerg.py --show-settings
  python scripts/download_imerg.py --show-settings --events-only

Step 2 -- generate the URL list (one-time, in a browser)
--------------------------------------------------------
1. Free account: https://urs.earthdata.nasa.gov/
2. Profile -> Applications -> Authorized Apps -> approve "NASA GESDISC DATA ARCHIVE".
   Skipping this causes nearly every 401 error people hit here.
3. https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGDF_07/summary
   -> "Subset / Get Data", using the settings printed in step 1
4. Get Data -> "Download links list" -> save as  data/raw/imerg/imerg_urls.txt

Step 3 -- download
------------------
  python scripts/download_imerg.py

Scope note
----------
--events-only restricts to flood windows plus matched control windows, cutting
~3,287 granules to a few hundred. Perfectly defensible: report the full 9-year bias
matrix for CHIRPS + ERA5 and use IMERG only in the event analysis -- but say so
explicitly in your methods.

Unit note
---------
V07 daily 'precipitation' is mm/day. V06 used mm/hr. Do not reuse V06 handling code;
check the variable's units attribute after download.
"""

import argparse
import datetime as dt
import getpass
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from basins import gesdisc_region, load_episodes  # noqa: E402

try:
    import requests
except ImportError:
    sys.exit("Run:  pip install requests")

CONTROL_PAD_DAYS = 15   # control window = same calendar days +/- this, in other years


class EarthdataSession(requests.Session):
    """Keeps auth across the URS OAuth redirect chain (NASA's documented pattern)."""

    AUTH_HOST = "urs.earthdata.nasa.gov"

    def __init__(self, username: str, password: str):
        super().__init__()
        self.auth = (username, password)

    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        if "Authorization" in headers:
            original = requests.utils.urlparse(response.request.url)
            redirect = requests.utils.urlparse(prepared_request.url)
            if (original.hostname != redirect.hostname
                    and redirect.hostname != self.AUTH_HOST
                    and original.hostname != self.AUTH_HOST):
                del headers["Authorization"]


def event_ranges(basin: str) -> list:
    """Episode windows plus padded surrounds, as (start, end) date strings.

    Episodes come from src/stage1/define_episodes.py via basins.load_episodes().
    Raises if they have not been derived yet -- there is no default event set.
    """
    out = []
    for ep in load_episodes(basin):
        d0 = dt.date.fromisoformat(ep["start"])
        d1 = dt.date.fromisoformat(ep["end"])
        out.append((str(d0), str(d1)))
        out.append((str(d0 - dt.timedelta(days=CONTROL_PAD_DAYS)),
                    str(d1 + dt.timedelta(days=CONTROL_PAD_DAYS))))
    return out


def show_settings(basin: str, events_only: bool) -> None:
    print("=" * 66)
    print("GES DISC subsetter settings")
    print("=" * 66)
    print("  Dataset          : GPM_3IMERGDF v07  (GPM IMERG Final Daily)")
    print("  Download Method  : OPeNDAP")
    print(f"  Region (W,S,E,N) : {gesdisc_region(basin)}")
    print("  Variables        : precipitation, precipitationQualityIndex")
    if events_only:
        print("  Date ranges      : (submit one request per range, append the lists)")
        for a, b in event_ranges(basin):
            print(f"                     {a}  to  {b}")
        print("\n  NOTE: this pads +/- "
              f"{CONTROL_PAD_DAYS} days around each episode purely so the DOWNLOAD"
              " covers the surrounding days.")
        print("  It is not the analysis control rule -- that lives in "
              "src/stage1/define_episodes.py.")
    else:
        print("  Date range       : 2015-01-01  to  2023-12-31   (~3,287 granules)")
    print("\n  Save the link list to:  data/raw/imerg/imerg_urls.txt")
    print("=" * 66)
    print("\nNote: episode windows are derived from IMD by "
          "src/stage1/define_episodes.py, not hand-picked.")


def download(out_dir: Path) -> None:
    url_file = out_dir / "imerg_urls.txt"
    if not url_file.exists():
        sys.exit(f"{url_file} not found.\n"
                 f"Run with --show-settings first, then generate the list "
                 f"via the GES DISC subsetter.")

    urls = [ln.strip() for ln in url_file.read_text().splitlines()
            if ln.strip().startswith("http")]
    if not urls:
        sys.exit(f"{url_file} contains no http URLs.")

    print(f"{len(urls)} granules listed.")
    username = input("Earthdata username: ")
    password = getpass.getpass("Earthdata password: ")
    session = EarthdataSession(username, password)

    done = skipped = failed = 0
    for i, url in enumerate(urls, 1):
        # Derive a safe, unique filename.
        # OPeNDAP URLs embed slashes and colons inside the dap4.ce query string, so
        # naive "last path segment" logic is wrong: strip the query FIRST, then take
        # the last segment, then sanitise characters Windows cannot store (":" would
        # silently create an NTFS alternate data stream -- every granule collapsing
        # onto one file, which is exactly the bug this replaces).
        m = re.search(r"(3B-DAY\.MS\.MRG\.3IMERG\.\d{8}-S\d{6}-E\d{6}\.V\d+[A-Z]?)", url)
        if m:
            name = m.group(1) + ".nc4"
        else:
            name = url.split("?")[0].split("/")[-1]
            name = re.sub(r"%[0-9A-Fa-f]{2}", "_", name).replace(":", "_")
            if not name.endswith((".nc4", ".nc", ".HDF5", ".pdf")):
                name += ".nc4"
        target = out_dir / name

        if target.exists() and target.stat().st_size > 0:
            skipped += 1
            continue

        try:
            with session.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                with open(target, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        fh.write(chunk)
            done += 1
        except Exception as exc:
            failed += 1
            target.unlink(missing_ok=True)
            print(f"  [{i}/{len(urls)}] FAILED {name}: {exc}")
            if failed == 1 and "401" in str(exc):
                print("  401 -> authorize 'NASA GESDISC DATA ARCHIVE' in your "
                      "Earthdata profile, then re-run.")

        if i % 100 == 0:
            print(f"  [{i}/{len(urls)}] downloaded={done} skipped={skipped} failed={failed}")

    print(f"\nFinished. downloaded={done} skipped={skipped} failed={failed}")
    if failed:
        print("Re-run to retry failures (existing files are skipped).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", default="tapi")
    ap.add_argument("--show-settings", action="store_true")
    ap.add_argument("--events-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path("data/raw/imerg")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.show_settings:
        show_settings(args.basin, args.events_only)
    else:
        download(out_dir)


if __name__ == "__main__":
    main()
