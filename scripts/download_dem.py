"""
Download an SRTM DEM for the study bbox from OpenTopography -- robust version.

Fixes the truncated-download problem: OpenTopography frequently cuts off large
single requests, producing a GeoTIFF with a valid header but incomplete pixel
data. That file opens fine and only fails when you actually read it
("TIFFFillTile:Read error ... got N bytes, expected M").

This version:
  1. verifies the byte count against the server's Content-Length
  2. fully reads the raster after download to prove it is not truncated
  3. retries, and if the single request keeps failing, splits the bbox into
     tiles, downloads each, and mosaics them

Setup
-----
  setx OPENTOPO_API_KEY "your-key-here"      (then open a NEW terminal;
                                              restart VS Code entirely)
  pip install requests rioxarray rasterio

Usage
-----
  python download_dem.py                # Tapi
  python download_dem.py --basin godavari
  python download_dem.py --tiles 3      # force a 3x3 tiled download
"""

import argparse
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Run:  pip install requests")

API = "https://portal.opentopography.org/API/globaldem"
DEM_TYPE = "SRTMGL3"          # 90 m; "SRTMGL1" = 30 m (4x the data, 4x the pain)

# Bounding boxes come from basins.py so every product uses the identical domain
sys.path.insert(0, str(Path(__file__).parent))
from basins import BASINS  # noqa: E402


def fetch(bbox: dict, key: str, target: Path, attempt: int) -> bool:
    """One download attempt. Returns True only if the file is complete and readable."""
    params = {
        "demtype": DEM_TYPE,
        "south": bbox["south"], "north": bbox["north"],
        "west": bbox["west"], "east": bbox["east"],
        "outputFormat": "GTiff",
        "API_Key": key,
    }
    try:
        with requests.get(API, params=params, timeout=1800, stream=True) as r:
            if r.status_code != 200:
                print(f"    HTTP {r.status_code}: {r.text[:300]}")
                return False

            ctype = r.headers.get("Content-Type", "")
            if "tif" not in ctype and "octet-stream" not in ctype:
                print(f"    unexpected Content-Type '{ctype}': {r.text[:300]}")
                return False

            expected = r.headers.get("Content-Length")
            expected = int(expected) if expected else None

            written = 0
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    written += len(chunk)

        if expected is not None and written != expected:
            print(f"    TRUNCATED: got {written:,} bytes, server said {expected:,}")
            target.unlink(missing_ok=True)
            return False

        print(f"    transferred {written / 1e6:.1f} MB"
              + (" (matches Content-Length)" if expected else " (no Content-Length header)"))
        return verify(target)

    except Exception as exc:
        print(f"    attempt {attempt} failed: {exc}")
        target.unlink(missing_ok=True)
        return False


def verify(path: Path) -> bool:
    """Force a full read. A truncated GeoTIFF has a valid header but unreadable tiles."""
    try:
        import rioxarray  # noqa: F401
        import rasterio
        with rasterio.open(path) as src:
            _ = src.read(1)          # this is what actually catches truncation
        return True
    except ImportError:
        print("    (rasterio not installed -- cannot verify; install it before trusting this file)")
        return True
    except Exception as exc:
        print(f"    UNREADABLE (truncated or corrupt): {exc}")
        path.unlink(missing_ok=True)
        return False


def tiled(bbox: dict, key: str, target: Path, n: int) -> bool:
    """Split the bbox into n x n tiles, download each, mosaic."""
    try:
        import rioxarray
        from rioxarray.merge import merge_arrays
    except ImportError:
        sys.exit("Tiled mode needs rioxarray:  pip install rioxarray")

    tmpdir = target.parent / "_tiles"
    tmpdir.mkdir(parents=True, exist_ok=True)

    dlat = (bbox["north"] - bbox["south"]) / n
    dlon = (bbox["east"] - bbox["west"]) / n
    paths = []

    for i in range(n):
        for j in range(n):
            sub = dict(
                south=bbox["south"] + i * dlat,
                north=bbox["south"] + (i + 1) * dlat,
                west=bbox["west"] + j * dlon,
                east=bbox["west"] + (j + 1) * dlon,
            )
            tile_path = tmpdir / f"tile_{i}_{j}.tif"
            if tile_path.exists() and verify(tile_path):
                print(f"  tile {i},{j}: already present")
                paths.append(tile_path)
                continue

            print(f"  tile {i},{j}:  lat {sub['south']:.2f}-{sub['north']:.2f}  "
                  f"lon {sub['west']:.2f}-{sub['east']:.2f}")
            for attempt in range(1, 4):
                if fetch(sub, key, tile_path, attempt):
                    paths.append(tile_path)
                    break
                time.sleep(5 * attempt)
            else:
                print(f"  tile {i},{j} FAILED after 3 attempts -- stopping")
                return False

    print(f"\nMosaicking {len(paths)} tiles...")
    arrays = [rioxarray.open_rasterio(p, masked=True) for p in paths]
    merged = merge_arrays(arrays)
    merged.rio.to_raster(target)
    for a in arrays:
        a.close()
    print(f"Mosaic written -> {target}")
    return verify(target)


def report(path: Path) -> None:
    try:
        import rioxarray
    except ImportError:
        return
    dem = rioxarray.open_rasterio(path, masked=True)
    print(f"\n  file   : {path.resolve()}  ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"  shape  : {dem.shape}")
    print(f"  CRS    : {dem.rio.crs}")
    print(f"  bounds : {tuple(round(b, 4) for b in dem.rio.bounds())}")
    print(f"  elev   : min={float(dem.min()):.0f} m  max={float(dem.max()):.0f} m  "
          f"mean={float(dem.mean()):.0f} m")
    print("  (Tapi: expect min near 0 m at the coast, max ~1000-1300 m in the Satpuras.)")
    dem.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", default="tapi", choices=sorted(BASINS))
    ap.add_argument("--tiles", type=int, default=0,
                    help="force an NxN tiled download instead of a single request")
    args = ap.parse_args()

    key = os.environ.get("OPENTOPO_API_KEY")
    if not key:
        sys.exit('OPENTOPO_API_KEY not set.  setx OPENTOPO_API_KEY "your-key"  '
                 "then restart VS Code entirely (a new terminal tab is not enough).")

    bbox = BASINS[args.basin]
    target = Path(f"data/raw/dem/{args.basin}_dem_srtm.tif")
    target.unlink(missing_ok=True)          # never keep a half-written file around

    print(f"{args.basin.upper()}  {DEM_TYPE}  "
          f"lat {bbox['south']}-{bbox['north']}  lon {bbox['west']}-{bbox['east']}")

    ok = False
    if not args.tiles:
        for attempt in range(1, 4):
            print(f"\nSingle request, attempt {attempt}/3...")
            if fetch(bbox, key, target, attempt):
                ok = True
                break
            time.sleep(10 * attempt)

    if not ok:
        n = args.tiles or 3
        print(f"\nFalling back to {n}x{n} tiled download...")
        ok = tiled(bbox, key, target, n)

    if ok:
        print("\nDEM downloaded and verified readable.")
        report(target)
    else:
        sys.exit("\nFailed. Try:  python download_dem.py --tiles 4")


if __name__ == "__main__":
    main()
