"""Standalone Refined Lee filtering of detected SAR intensity (linear power).

Adapted from the MIT-licensed gee_s1_ard RefinedLee implementation by
Mullissa et al., itself credited to Guido Lemoine. No Earth Engine or SNAP.
Source: https://github.com/adugnag/gee_s1_ard/blob/main/python-api/speckle_filter.py
Citation: Mullissa et al. (2021), Remote Sensing 13, 1954,
https://doi.org/10.3390/rs13101954

The array API requires only NumPy; seasonal means are filtered once after compositing.
The optional GeoTIFF API also needs rasterio.
The fixed 7x7 directional filter estimates normalized noise variance locally
from the five lowest variance/mean**2 values in nine sampled 3x3 windows.

Differences from upstream: deterministic gradient ties, weights clipped to
[0, 1], safe constant/zero handling, and conservative nodata handling. Where
a complete valid 7x7 neighbourhood is unavailable, retain the valid input
pixel; do not invent values across holes or raster boundaries. This is an
adaptation, not a claim of pixel-identical Earth Engine or SNAP output.

MIT License
Copyright (c) 2021 Adugna Mullissa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from pathlib import Path
import tempfile

import numpy as np

__all__ = ["refined_lee", "filter_geotiff", "NODATA"]
__version__ = "2.0.0"
HALO = 3
NODATA = -9999.0


def _shift(a, dy, dx, fill=0):
    """At (row,col), return a[row+dy,col+dx]; outside the array use fill."""
    h, w = a.shape
    out = np.full(a.shape, fill, dtype=a.dtype)
    y0, y1 = max(0, -dy), min(h, h - dy)
    x0, x1 = max(0, -dx), min(w, w - dx)
    if y0 < y1 and x0 < x1:
        out[y0:y1, x0:x1] = a[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return out


def _sum_kernel(a, kernel):
    out = np.zeros_like(a, dtype="float64")
    radius = kernel.shape[0] // 2
    h, w = a.shape
    for y, x in np.argwhere(kernel):
        dy, dx = int(y) - radius, int(x) - radius
        y0, y1 = max(0, -dy), min(h, h - dy)
        x0, x1 = max(0, -dx), min(w, w - dx)
        if y0 < y1 and x0 < x1:
            # Accumulate views directly: no whole-image allocation per offset.
            # Row-major accumulation order is unchanged from version 1.0.0.
            out[y0:y1, x0:x1] += a[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return out


def _stats(a, squared, kernel):
    n = int(kernel.sum())
    mean = _sum_kernel(a, kernel) / n
    # Population variance, as in ee.Reducer.variance (not sampleVariance).
    variance = np.maximum(_sum_kernel(squared, kernel) / n - mean**2, 0)
    return mean, variance


def _complete_neighborhood(valid):
    # Separable 7x7 validity erosion: 14 shifts instead of 49.
    rows_complete = valid.copy()
    for dy in range(-HALO, HALO + 1):
        rows_complete &= _shift(valid, dy, 0, fill=False)
    complete = rows_complete.copy()
    for dx in range(-HALO, HALO + 1):
        complete &= _shift(rows_complete, 0, dx, fill=False)
    return complete


def _filter_block(image, valid):
    # Convert only this processing block, not the entire source burst.
    image = np.asarray(image, dtype="float64")
    out = np.where(valid, image, np.nan).astype("float64")
    complete = _complete_neighborhood(valid)
    if not complete.any():
        return out.astype("float32")

    # Scale for stable moments; the filter is invariant to positive scaling.
    scale = float(np.max(image[valid]))
    if scale == 0:
        return out.astype("float32")
    a = np.where(valid, image / scale, 0)
    squared = a * a
    mean3, var3 = _stats(a, squared, np.ones((3, 3), dtype=bool))
    offsets = [(dy, dx) for dy in (-2, 0, 2) for dx in (-2, 0, 2)]
    means = np.stack([_shift(mean3, dy, dx) for dy, dx in offsets])
    # Contiguous per-pixel samples make the nine-element sort much cheaper.
    ratios = np.empty((*a.shape, 9), dtype="float64")
    for i, (dy, dx) in enumerate(offsets):
        ratio = _shift(var3, dy, dx)
        np.divide(ratio, means[i]**2, out=ratio, where=means[i] > 0)
        ratio[means[i] <= 0] = np.inf
        ratios[..., i] = ratio
    ratios.sort(axis=-1)
    noise = ratios[..., :5].mean(axis=-1)

    pairs = [(1, 7), (6, 2), (3, 5), (0, 8)]
    # Track the maximum without allocating a four-band gradient stack.
    strongest = np.full(a.shape, -1.0, dtype="float64")
    direction = np.empty(a.shape, dtype="uint8")
    for k, (i, j) in enumerate(pairs):
        gradient = np.abs(means[i] - means[j])
        chosen = gradient > strongest  # Strict > preserves first-gradient ties.
        np.maximum(strongest, gradient, out=strongest)
        first = means[i] - means[4] > means[4] - means[j]
        direction[chosen] = np.where(first, k + 1, k + 5)[chosen]

    del means, ratios, strongest
    noise_finite = np.isfinite(noise)
    safe_noise = np.where(noise_finite, noise, 0)

    rect = np.zeros((7, 7), dtype=bool)
    rect[3:, :] = True
    diagonal = np.tri(7, dtype=bool)
    for rotation in range(4):
        # Earth Engine's positive Kernel.rotate is clockwise.
        for number, base in ((2 * rotation + 1, rect),
                             (2 * rotation + 2, diagonal)):
            kernel = np.rot90(base, -rotation)
            chosen = complete & (direction == number) & noise_finite
            if not chosen.any():
                continue
            # Only these pixels use this direction. Preserve the original
            # row-major sum order while avoiding eight full-image moments.
            rows, cols = np.nonzero(chosen)
            total = np.zeros(rows.size, dtype="float64")
            total_squared = np.zeros_like(total)
            for y, x in np.argwhere(kernel):
                yy, xx = rows + int(y) - HALO, cols + int(x) - HALO
                values = a[yy, xx]
                total += values
                total_squared += values * values
            n = int(kernel.sum())
            mean = total / n
            variance = np.maximum(total_squared / n - mean**2, 0)
            local_noise = safe_noise[rows, cols]
            signal_variance = (variance - mean**2 * local_noise) / (1 + local_noise)
            weight = np.zeros(rows.size, dtype="float64")
            np.divide(signal_variance, variance, out=weight, where=variance > 0)
            np.clip(weight, 0, 1, out=weight)
            out[rows, cols] = (mean + weight * (a[rows, cols] - mean)) * scale
    return out.astype("float32")


def _tiles(height, width, size):
    for row in range(0, height, size):
        for col in range(0, width, size):
            bottom, right = min(row + size, height), min(col + size, width)
            top, left = max(0, row - HALO), max(0, col - HALO)
            stop, end = min(height, bottom + HALO), min(width, right + HALO)
            yield (slice(top, stop), slice(left, end)), (
                slice(row - top, bottom - top), slice(col - left, right - left)), (
                slice(row, bottom), slice(col, right))


def refined_lee(image, valid_mask=None, *, tile_size=512):
    """Filter a 2-D intensity image or (bands, rows, cols) stack independently.

    Input must be nonnegative linear power, NOT dB, amplitude, complex SLC,
    mask codes, or observation counts. -9999, NaN, infinity, and masked-array pixels
    are excluded. valid_mask can match the image or be a 2-D spatial mask;
    True means valid. Valid zeros are supported. Return Float32, with -9999
    at invalid input pixels. No input mutation, gap filling or repeated passes.
    Temporary arrays are tiled with a three-pixel halo; output remains in RAM.
    """
    if isinstance(tile_size, bool) or not isinstance(tile_size, (int, np.integer)) or tile_size < 1:
        raise ValueError("tile_size must be a positive integer")
    if np.iscomplexobj(image):
        raise ValueError("Supply detected linear power, not complex SAR samples")
    a = np.ma.asarray(image)
    if a.ndim not in (2, 3) or any(n == 0 for n in a.shape):
        raise ValueError("image must have shape (rows,cols) or (bands,rows,cols)")
    if not np.issubdtype(a.dtype, np.number):
        raise ValueError("image must contain real numeric backscatter values")
    values = np.asarray(a.data)
    valid = np.isfinite(values) & (values != NODATA) & ~np.ma.getmaskarray(a)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape not in (values.shape, values.shape[-2:]):
            raise ValueError("valid_mask must match the image or its spatial dimensions")
        valid &= mask
    if np.any(valid & (values < 0)):
        raise ValueError("Valid backscatter must be nonnegative linear power, not dB")
    if np.any(valid & (values > np.finfo("float32").max)):
        raise ValueError("Input exceeds the Float32 output range")
    squeezed = values.ndim == 2
    if squeezed:
        values, valid = values[None], valid[None]
    out = np.full(values.shape, np.nan, dtype="float32")
    for b in range(values.shape[0]):
        for expanded, core, target in _tiles(*values.shape[-2:], tile_size):
            filtered = _filter_block(values[b][expanded], valid[b][expanded])
            out[b][target] = filtered[core]
    out[~valid] = NODATA
    return out[0] if squeezed else out


def filter_geotiff(source, destination, *, mask_path=None, tile_size=512,
                   overwrite=False):
    """Write a filtered COG without loading an entire raster into memory.

    Filter SAR backscatter in linear power. Metadata-identified count bands
    pass through without filtering. Rasterio is required.
    Optional mask_path is a co-registered OPERA acquisition mask: keep code 0.
    Source nodata/internal masks are always respected. Preserve CRS, transform,
    band descriptions and source dataset tags. Intensity output is Float32 with
    -9999 nodata. Metadata-identified counts are copied unchanged; a packed
    Int64 metadata-only raster is copied without any floating-point conversion.
    The destination must differ from the source and is not overwritten by default.
    """
    import rasterio
    from rasterio.shutil import copy as rio_copy
    from rasterio.windows import Window
    from contextlib import ExitStack

    if isinstance(tile_size, bool) or not isinstance(tile_size, (int, np.integer)) or tile_size < 1:
        raise ValueError("tile_size must be a positive integer")
    destination = Path(destination)
    if not str(source).startswith(("s3://", "https://")) and Path(source).resolve() == destination.resolve():
        raise ValueError("Source and destination must differ")
    if mask_path is not None and not str(mask_path).startswith(("s3://", "https://")) and Path(mask_path).resolve() == destination.resolve():
        raise ValueError("Destination must not overwrite the source mask")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(source))
        mask_src = stack.enter_context(rasterio.open(mask_path)) if mask_path else None
        if mask_src and (mask_src.crs != src.crs or mask_src.transform != src.transform
                         or mask_src.shape != src.shape):
            raise ValueError("Mask must have the same CRS, transform and shape as the source")
        if any(v != 1 for v in src.scales) or any(v != 0 for v in src.offsets):
            raise ValueError("Apply source scale/offset before filtering")
        temporary = stack.enter_context(tempfile.TemporaryDirectory(
            prefix="refined_lee_", dir=destination.parent))
        gtiff, cog = Path(temporary) / "filtered.tif", Path(temporary) / "filtered_cog.tif"
        # Never send packed Int64 metadata through floating-point filter I/O.
        if all(src.tags(b).get("units") == "packed_metadata" for b in range(1, src.count + 1)):
            rio_copy(source, cog, driver="COG", compress="DEFLATE", blocksize=512,
                     overview_resampling="nearest")
            cog.replace(destination)
            return str(destination)
        profile = dict(driver="GTiff", height=src.height, width=src.width,
                       count=src.count, dtype="float32", crs=src.crs,
                       transform=src.transform, nodata=NODATA, tiled=True,
                       blockxsize=256, blockysize=256, compress="deflate")
        with rasterio.open(gtiff, "w", **profile) as dst:
            source_tags = src.tags()
            for key in ("filter_backend", "speckle_filter_backend"):
                source_tags.pop(key, None)
            dst.update_tags(**source_tags)
            dst.update_tags(speckle_filter="Refined_Lee_GEE_ARD_adaptation",
                            speckle_filter_version=__version__, filter_window="7x7",
                            filter_stage="after_seasonal_compositing",
                            noise_estimation="mean_of_five_lowest_3x3_normalized_variances",
                            filter_units="linear_power", filter_invalid_edges="retain_input")
            for b in range(1, src.count + 1):
                is_metadata = (src.tags(b).get("role") in ("source_provenance", "packed_qa") or
                            src.tags(b).get("units") in ("valid_pass_count", "calendar_year", "nominal_day_of_year", "packed_qa", "packed_metadata") or
                            "count_" in (src.descriptions[b - 1] or "").lower())
                dst.update_tags(b, **src.tags(b))
                dst.update_tags(b, speckle_filter_applied="false" if is_metadata else "true")
                if src.descriptions[b - 1]:
                    dst.set_band_description(b, src.descriptions[b - 1])
                for expanded, core, target in _tiles(src.height, src.width, tile_size):
                    win = Window.from_slices(*expanded)
                    image = src.read(b, window=win, masked=True).astype("float64")
                    if is_metadata:
                        dst.write(image.filled(NODATA).astype("float32")[core], b,
                                  window=Window.from_slices(*target))
                        continue
                    valid = None
                    if mask_src:
                        m = mask_src.read(1, window=win, masked=True)
                        valid = ~np.ma.getmaskarray(m) & (m.data == 0)
                    filtered = refined_lee(image, valid, tile_size=tile_size + 2 * HALO)
                    dst.write(filtered[core], b, window=Window.from_slices(*target))
        rio_copy(gtiff, cog, driver="COG", compress="DEFLATE", blocksize=512,
                 overview_resampling="nearest")
        cog.replace(destination)
    return str(destination)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Refined Lee filter of linear-power SAR GeoTIFF")
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--mask", help="Co-registered OPERA acquisition mask; keep code 0")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(filter_geotiff(args.source, args.destination, mask_path=args.mask,
                        tile_size=args.tile_size, overwrite=args.overwrite))
