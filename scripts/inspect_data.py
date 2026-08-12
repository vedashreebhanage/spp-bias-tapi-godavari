"""
Dump the exact contents of every NetCDF dataset (CHIRPS, ERA5, IMERG) in readable
form. No analysis, no judgements -- dimensions, coordinates, attributes, and raw
values only.

    python scripts/inspect_data.py

Writes data_report.txt in the project root and prints the same to the console.
"""

import sys
from pathlib import Path

try:
    import numpy as np
    import xarray as xr
except ImportError:
    sys.exit("Run:  pip install xarray netCDF4")

OUT = Path("data_report.txt")
lines = []


def w(s=""):
    print(s)
    lines.append(s)


def dump_dataset(path: Path, sample_spec: str):
    w("-" * 68)
    w(f"FILE: {path}  ({path.stat().st_size/1e6:.2f} MB)")
    ds = xr.open_dataset(path)

    w(f"dimensions : {dict(ds.sizes)}")
    for c in ds.coords:
        v = ds[c].values
        if v.ndim == 0 or v.size == 1:
            w(f"coord {c:<12}: {v}")
        else:
            w(f"coord {c:<12}: {v.size} values, first={v[0]}, last={v[-1]}")
    w("global attributes:")
    if not ds.attrs:
        w("  (none)")
    for k, v in ds.attrs.items():
        w(f"  {k} = {str(v)[:120]}")
    for name, da in ds.data_vars.items():
        w(f"variable '{name}': shape {da.shape}, dtype {da.dtype}")
        for k, v in da.attrs.items():
            w(f"  attr {k} = {str(v)[:100]}")
        vals = da.values.astype("float64")
        finite = vals[np.isfinite(vals)]
        w(f"  finite values: {finite.size:,} of {vals.size:,} "
          f"({100*finite.size/vals.size:.1f}%)")
        if finite.size:
            w(f"  min={finite.min():.4f}  max={finite.max():.4f}  "
            f"mean={finite.mean():.4f}")
        w(f"  sample ({sample_spec}):")
        w(f"    {np.array2string(sample(da), precision=3, max_line_width=76)}")
    ds.close()


def sample(da):
    """A small fixed block of raw values from the middle of the array."""
    idx = []
    for dim, size in zip(da.dims, da.shape):
        if dim in ("time", "valid_time"):
            idx.append(slice(size // 2, size // 2 + 1))
        else:
            idx.append(slice(size // 2, size // 2 + 4))
    return np.squeeze(da[tuple(idx)].values)


def main():
    w("DATA REPORT -- exact file contents, no interpretation")
    w("")

    # CHIRPS: dump one file fully, then one line per remaining year
    w("=" * 68)
    w("CHIRPS  data/raw/chirps/")
    w("=" * 68)
    chirps = sorted(Path("data/raw/chirps").glob("chirps_tapi_*.nc"))
    if chirps:
        dump_dataset(chirps[0], "middle day, 4x4 cells")
        w("")
        w("remaining years, one line each (var / days / min / max / mean of finite):")
        for p in chirps[1:]:
            ds = xr.open_dataset(p)
            name = [v for v in ds.data_vars if ds[v].ndim == 3][0]
            v = ds[name].values.astype("float64")
            f = v[np.isfinite(v)]
            w(f"  {p.name}: {name} {ds[name].shape} "
              f"min={f.min():.2f} max={f.max():.2f} mean={f.mean():.3f}")
            ds.close()
    else:
        w("  no files found")

    # ERA5
    w("")
    w("=" * 68)
    w("ERA5-LAND  data/raw/era5/")
    w("=" * 68)
    era5 = sorted(Path("data/raw/era5").glob("era5_tapi_*.nc"))
    if era5:
        dump_dataset(era5[0], "middle day, 4x4 cells")
        w("")
        w("remaining files, one line each:")
        for p in era5[1:]:
            ds = xr.open_dataset(p)
            name = [v for v in ds.data_vars if ds[v].ndim >= 3][0]
            v = ds[name].values.astype("float64")
            f = v[np.isfinite(v)]
            w(f"  {p.name}: {name} {ds[name].shape} "
              f"min={f.min():.6f} max={f.max():.6f} mean={f.mean():.6f}")
            ds.close()
    else:
        w("  no files found")

    # IMERG: dump one subsetted granule fully + count the collection
    w("")
    w("=" * 68)
    w("IMERG  data/raw/imerg/")
    w("=" * 68)
    granules = sorted(Path("data/raw/imerg").glob("3B-DAY*.nc4"))
    small = [p for p in granules if p.stat().st_size < 5_000_000]
    big = [p for p in granules if p.stat().st_size >= 5_000_000]
    w(f"subsetted granules found : {len(small)}")
    w(f"global-size granules     : {len(big)}  "
      f"({[p.name for p in big]})" if big else "global-size granules     : 0")
    if small:
        dump_dataset(small[len(small) // 2], "middle of grid")
        first, last = small[0].name, small[-1].name
        w("")
        w(f"first granule file : {first}")
        w(f"last granule file  : {last}")
        sizes = np.array([p.stat().st_size for p in small])
        w(f"sizes: min={sizes.min():,} B  max={sizes.max():,} B  "
          f"mean={sizes.mean():,.0f} B  zero-byte files: {(sizes==0).sum()}")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    w("")
    w(f"Report saved to {OUT.resolve()}")


if __name__ == "__main__":
    main()
