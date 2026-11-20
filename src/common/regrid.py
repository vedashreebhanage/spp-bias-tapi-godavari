"""
Area-weighted regridding onto the IMD 0.25 deg grid.

Why this module exists
----------------------
Three products must be brought onto IMD's grid before anything can be compared:
CHIRPS at 0.05 deg, ERA5-Land and IMERG at 0.1 deg. Getting this wrong is silent:
the arrays still have the right shape, the numbers still look like rainfall, and
every downstream statistic is quietly built on misaligned cells. So the weights
are computed exactly and the module is unit-tested (tests/test_regrid.py).

Method
------
A target cell and a source cell are both lat-lon rectangles. Their intersection
area on the sphere is exact and separable:

    area = (sin(lat_top) - sin(lat_bot)) * (lon_right - lon_left)

so the weight matrix for one target cell is the outer product of a latitude
overlap vector and a longitude overlap vector. No interpolation, no assumption
that the grids nest, no cos(lat) approximation.

CHIRPS does NOT nest into the IMD grid, despite 0.05 dividing 0.25 exactly.
CHIRPS cell edges fall on 19.5 + 0.05k; IMD cell edges fall on 6.375 + 0.25k, so
an IMD edge such as 20.125 sits at 12.5 CHIRPS steps -- exactly half a cell out.
Each 0.25 deg target therefore covers SIX partial CHIRPS cells per axis, weighted
0.5, 1, 1, 1, 1, 0.5, not a 5x5 block. A 5x5 block mean would average a footprint
displaced by 0.025 deg (~2.8 km) from the IMD cell it claims to represent -- a
small, uniform, entirely invisible spatial shift of one product against the
reference. The area-weighted calculation gets this right without being told.

NaN handling
------------
Source pixels that are NaN (CHIRPS ~2%, ERA5-Land over sea ~5%) are dropped and
the remaining weights renormalised, so a target cell is the mean of the data that
actually exists inside it. The fraction of the cell's area that was valid is
returned alongside the value, per timestep -- a cell that is only ever 30% observed
is a different thing from a fully observed one, and the caller should be able to
see that rather than have it hidden inside a mean.
"""

import numpy as np

__all__ = ["cell_edges", "build_weights", "apply_weights"]


# CHIRPS stores its coordinates as float32. At longitude ~78 that leaves about
# 5e-6 deg of rounding noise per value, so the differences between consecutive
# centers come out as 0.049995..0.050003 rather than exactly 0.05. The grid is
# regular by construction -- the wobble is a storage artefact, not data. Comparing
# raw differences with a tight tolerance rejects a perfectly good file (and the
# leftover noise would shift cell edges enough to leave slivers of weight in
# neighbouring cells). So the axis is fitted to the regular
# grid it is supposed to be, and the fit is checked before it is trusted.
SPACING_RTOL = 1e-3          # 0.05 deg grid: rejects anything off by > 5e-5 deg
SPACING_DECIMALS = 6         # snap the inferred step to a clean value


def regular_axis(centers):
    """Return (fitted centers, step) for an axis that should be regularly spaced.

    Raises ValueError if the spacing is genuinely irregular, or descending.
    """
    centers = np.asarray(centers, dtype=float)
    if centers.size < 2:
        raise ValueError("need at least two cell centers to infer spacing")

    # Infer the step from the FULL SPAN, not from individual differences. Each
    # difference carries the rounding noise of two float32 values, and the median
    # of those noisy differences rounds to 0.050003 rather than 0.05 -- an error of
    # 3e-6 per cell, which accumulates to 3.6e-4 deg across a 120-cell axis. The
    # span divides that same noise by the number of intervals instead.
    span = float(centers[-1] - centers[0])
    step = round(span / (centers.size - 1), SPACING_DECIMALS)
    if step <= 0:
        raise ValueError("centers must be strictly ascending -- flip the axis first")
    d = np.diff(centers)
    if not np.allclose(d, step, rtol=SPACING_RTOL, atol=0):
        raise ValueError(
            f"grid spacing is not uniform: min {d.min()}, max {d.max()}, "
            f"median {step}. This is more than float32 rounding -- the axis is "
            "genuinely irregular."
        )

    idx = np.arange(centers.size)
    origin = float(np.mean(centers - step * idx))     # least-squares intercept
    fitted = origin + step * idx
    drift = np.abs(fitted - centers).max()
    if drift > step * SPACING_RTOL:
        raise ValueError(
            f"axis does not fit a regular grid: max deviation {drift:g} deg"
        )
    return fitted, step


def cell_edges(centers):
    """Edges of a regular 1-D grid of cell centers."""
    fitted, step = regular_axis(centers)
    return np.concatenate([fitted - step / 2, [fitted[-1] + step / 2]])


def _overlap_lat(src_edges, lo, hi):
    """Spherical latitude overlap, proportional to sin(top) - sin(bottom)."""
    bot = np.maximum(src_edges[:-1], lo)
    top = np.minimum(src_edges[1:], hi)
    ok = top > bot
    out = np.zeros(src_edges.size - 1)
    out[ok] = np.sin(np.deg2rad(top[ok])) - np.sin(np.deg2rad(bot[ok]))
    return out


def _overlap_lon(src_edges, lo, hi):
    left = np.maximum(src_edges[:-1], lo)
    right = np.minimum(src_edges[1:], hi)
    return np.maximum(right - left, 0.0)


# Wherever a target edge coincides with a source edge (grids that genuinely nest,
# as in the synthetic cases in tests/test_regrid.py), the floating-point comparison
# lands either side of the truth and the neighbouring cell picks up a meaningless
# overlap. Left alone these slivers widen the weight block by a row and a column of
# near-zeros -- harmless arithmetically, but they make the block shape depend on
# rounding. Discard them. (None of the real products nest into IMD: CHIRPS is
# offset half a cell, see the module docstring.)
#
# The tolerance sits between the two scales involved: leftover float32 coordinate
# noise contributes ~1e-4 of a cell, while the smallest overlap any real product
# here produces is ~0.25 of a cell (0.1 deg sources cut by 0.25 deg targets). 1e-3
# is clear of both by a wide margin.
SLIVER_REL_TOL = 1e-3


def _drop_slivers(w):
    peak = w.max()
    if peak <= 0:
        return w
    return np.where(w < SLIVER_REL_TOL * peak, 0.0, w)


def build_weights(src_lat, src_lon, dst_lat, dst_lon, dst_size=0.25,
                  min_coverage=0.99):
    """Weights mapping a regular source grid onto a list of target cell centers.

    Parameters
    ----------
    src_lat, src_lon : 1-D ascending arrays of source cell centers.
    dst_lat, dst_lon : 1-D arrays of target cell centers (one entry per target
        cell; they need not form a regular grid -- here they are the 92 in-basin
        cells).
    dst_size : target cell size in degrees.
    min_coverage : fail if a target cell is not at least this fraction covered by
        the source grid's extent. Catches a source file that does not actually
        span the basin -- an error that would otherwise show up as a suspiciously
        dry basin edge rather than as an exception.

    Returns
    -------
    list of dicts, one per target cell: ``{"jslice", "kslice", "w"}`` where ``w``
    is the 2-D weight block for that cell.
    """
    src_lat = np.asarray(src_lat, dtype=float)
    src_lon = np.asarray(src_lon, dtype=float)
    lat_edges = cell_edges(src_lat)
    lon_edges = cell_edges(src_lon)
    half = dst_size / 2

    out = []
    for la, lo in zip(np.asarray(dst_lat, float), np.asarray(dst_lon, float)):
        wlat = _drop_slivers(_overlap_lat(lat_edges, la - half, la + half))
        wlon = _drop_slivers(_overlap_lon(lon_edges, lo - half, lo + half))

        full = ((np.sin(np.deg2rad(la + half)) - np.sin(np.deg2rad(la - half)))
                * dst_size)
        covered = wlat.sum() * wlon.sum()
        if covered < min_coverage * full:
            raise ValueError(
                f"target cell ({la:.3f}, {lo:.3f}) is only "
                f"{covered / full:.1%} covered by the source grid "
                f"(lat {src_lat[0]:.3f}..{src_lat[-1]:.3f}, "
                f"lon {src_lon[0]:.3f}..{src_lon[-1]:.3f}). "
                "The source file does not span the basin."
            )

        (j,) = np.nonzero(wlat)
        (k,) = np.nonzero(wlon)
        jsl = slice(j[0], j[-1] + 1)
        ksl = slice(k[0], k[-1] + 1)
        out.append({
            "jslice": jsl,
            "kslice": ksl,
            "w": np.outer(wlat[jsl], wlon[ksl]),
        })
    return out


def apply_weights(data, weights):
    """Regrid ``data`` (time, lat, lon) onto the target cells.

    Returns
    -------
    values : (time, n_cells) float64, NaN where nothing valid fell in the cell.
    valid_fraction : (time, n_cells) float64, the fraction of each target cell's
        area that had finite source data.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 3:
        raise ValueError(f"expected (time, lat, lon), got shape {data.shape}")

    nt = data.shape[0]
    values = np.full((nt, len(weights)), np.nan)
    valid = np.zeros((nt, len(weights)))

    for c, spec in enumerate(weights):
        sub = data[:, spec["jslice"], spec["kslice"]]
        w = spec["w"]
        finite = np.isfinite(sub)
        wsum = np.einsum("tjk,jk->t", finite.astype(float), w)
        total = w.sum()
        with np.errstate(invalid="ignore"):
            num = np.einsum("tjk,jk->t", np.where(finite, sub, 0.0), w)
        nonzero = wsum > 0
        values[nonzero, c] = num[nonzero] / wsum[nonzero]
        valid[:, c] = wsum / total
    return values, valid
