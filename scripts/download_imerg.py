"""
Download spatially-subsetted IMERG V07 daily granules from NASA GES DISC.

Why a URL list instead of a direct API call
-------------------------------------------
IMERG daily granules are GLOBAL (~29 MB each). Nine years = ~95 GB.
Subsetted to the Tapi bbox they are ~0.1 MB each = ~350 MB total.
GES DISC performs the subsetting server-side and hands you a list of URLs.

Steps
-----
1. Create a free Earthdata account: https://urs.earthdata.nasa.gov/
2. Profile -> Applications -> Authorized Apps -> approve "NASA GESDISC DATA ARCHIVE".
   (Skipping this is the #1 cause of 401 errors.)
3. https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGDF_07/summary
   -> "Subset / Get Data"
      Download Method : OPeNDAP
      Date range      : 2015-01-01 to 2023-12-31
      Region          : 72.25, 19.75, 79.00, 23.00   (W, S, E, N)
      Variables       : precipitation, precipitationQualityIndex
   -> Get Data -> "Download links list" -> save next to this script as imerg_urls.txt
4. python download_imerg.py

Scope reduction option
----------------------
If 3,287 files is impractical, restrict the date range in step 3 to your flood-event
and control windows only. Then report the full 9-year bias matrix for CHIRPS + ERA5
and use IMERG only in the DPBA analysis -- a legitimate scope decision, but state it
explicitly in your methods.

Unit note
---------
V07 daily 'precipitation' is mm/day. V06 used mm/hr. Do not copy V06 handling code;
check the variable's units attribute after download.
"""

import getpass
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("requests not installed.  Run:  pip install requests")

URL_FILE = Path("imerg_urls.txt")
OUT_DIR = Path("data/raw/imerg")


class EarthdataSession(requests.Session):
    """Preserves auth across the URS OAuth redirect chain (NASA's documented pattern)."""

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


def main() -> None:
    if not URL_FILE.exists():
        sys.exit(f"{URL_FILE} not found. Generate it via the GES DISC subsetter "
                 f"(see the docstring at the top of this file).")

    urls = [ln.strip() for ln in URL_FILE.read_text().splitlines()
            if ln.strip() and ln.strip().startswith("http")]
    if not urls:
        sys.exit(f"{URL_FILE} contains no http URLs.")

    print(f"{len(urls)} granules listed.")
    username = input("Earthdata username: ")
    password = getpass.getpass("Earthdata password: ")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = EarthdataSession(username, password)

    done = skipped = failed = 0
    for i, url in enumerate(urls, 1):
        name = url.split("/")[-1].split("?")[0]
        if not name.endswith((".nc4", ".nc", ".HDF5")):
            name += ".nc4"
        target = OUT_DIR / name

        if target.exists() and target.stat().st_size > 0:
            skipped += 1
            continue

        try:
            with session.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(target, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        fh.write(chunk)
            done += 1
        except Exception as exc:
            failed += 1
            print(f"  [{i}/{len(urls)}] FAILED {name}: {exc}")
            target.unlink(missing_ok=True)
            if failed == 1 and "401" in str(exc):
                print("  401 => check that 'NASA GESDISC DATA ARCHIVE' is an "
                      "authorized app in your Earthdata profile.")

        if i % 100 == 0:
            print(f"  [{i}/{len(urls)}]  downloaded={done} skipped={skipped} failed={failed}")

    print(f"\nFinished. downloaded={done} skipped={skipped} failed={failed}")
    print(f"Output: {OUT_DIR.resolve()}")
    if failed:
        print("Re-run the script to retry failures (existing files are skipped).")


if __name__ == "__main__":
    main()
