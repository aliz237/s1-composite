"""Three-season OPERA VV/VH composites on MAAP (AWS us-west-2).

Native-grid burst masking -> user-selected reprojection -> overlap mean within
pass -> equal-pass seasonal mean -> optional NumPy Refined Lee on the mean.
Write six Float32 means and an optional single-band Int64 metadata COG.
Requires maap-py, numpy, rasterio, boto3 and the companion refined_lee.py;
SciPy supplies compiled component/neighborhood operations for gap eligibility.
Season intervals are calendar dates [start, end), within the requested year.
"""

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import random
import tempfile
import time
from contextlib import ExitStack

import numpy as np
from refined_lee import HALO, NODATA, __version__ as FILTER_VERSION, refined_lee

SEASONS = ("frozen", "summer", "shoulder")
DEFAULT_WINDOWS = {
    "frozen": ("01-01", "04-01"),
    "summer": ("06-15", "09-01"),
    "shoulder": ("09-15", "11-01"),
}
ARCHIVE_START = datetime(2016, 4, 14, tzinfo=timezone.utc)


def parse_bbox(value, bbox_crs="EPSG:4326"):
    """Accept a DPS comma-separated string or a four-coordinate sequence."""
    try:
        coords = value.split(",") if isinstance(value, str) else value
        west, south, east, north = (float(x) for x in coords)
    except (ValueError, TypeError):
        raise ValueError("bbox must be xmin,ymin,xmax,ymax (four coordinates)") from None
    if not all(math.isfinite(x) for x in (west, south, east, north)) or west >= east or south >= north:
        raise ValueError("Use finite ordered bbox coordinates: xmin < xmax and ymin < ymax")
    if bbox_crs is not None and resolve_bbox_crs(bbox_crs).is_geographic and not (
            -180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Use finite WGS84 bounds without dateline crossing")
    return west, south, east, north


def resolve_bbox_crs(value):
    """Read an explicit input CRS, including @WKT files."""
    from rasterio.crs import CRS
    if value is None or isinstance(value, str) and not value.strip():
        raise ValueError("Provide an explicit CRS")
    if isinstance(value, Path) or isinstance(value, str) and value.startswith("@"):
        path = value if isinstance(value, Path) else Path(value[1:])
        value = path.read_text().strip()
    return CRS.from_user_input(value)


def resolve_crs(value):
    """Explicit projected metre CRS for the output; no fallback."""
    if value is None or isinstance(value, str) and not value.strip():
        raise ValueError("Provide an explicit crs or a reference template raster")
    crs = resolve_bbox_crs(value)
    if not crs.is_projected or crs.linear_units != "metre":
        raise ValueError("Use an explicit projected CRS with metre units")
    return crs


def _bbox_geometry(bbox):
    # Densify in the INPUT CRS so reprojection preserves curved boundaries.
    west, south, east, north = bbox
    corners = [(west, south), (east, south), (east, north), (west, north)]
    ring = []
    for a, b in zip(corners, corners[1:] + corners[:1]):
        ring.extend([(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
                     for t in np.linspace(0, 1, 101, endpoint=False)])
    ring.append(ring[0])
    return {"type": "Polygon", "coordinates": [ring]}


def _search_bbox(geometry, bbox_crs):
    from rasterio.warp import transform_geom
    geographic = transform_geom(bbox_crs, "EPSG:4326", geometry)
    xs, ys = zip(*geographic["coordinates"][0])
    if any(abs(b - a) > 180 for a, b in zip(xs, xs[1:])):
        raise ValueError("Bounding box crosses the dateline after projection; split the tile")
    return parse_bbox((min(xs), min(ys), max(xs), max(ys)))


def _grid_from_bbox(bbox, bbox_crs, crs, spacing):
    """Preserve native bounds/origin; enclose reprojected bounds otherwise."""
    from rasterio.transform import from_origin
    from rasterio.warp import transform_geom
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive")
    if bbox_crs == crs:
        left, bottom, right, top = bbox
        dimensions = ((right - left) / spacing, (top - bottom) / spacing)
        if any(not math.isclose(v, round(v), rel_tol=0, abs_tol=1e-7) or round(v) < 1
               for v in dimensions):
            raise ValueError("Bounds in the output CRS must span a whole number of pixels at spacing")
        width, height = map(round, dimensions)
    else:
        geom = transform_geom(bbox_crs, crs, _bbox_geometry(bbox))
        xs, ys = zip(*geom["coordinates"][0])
        left = math.floor(min(xs) / spacing) * spacing
        top = math.ceil(max(ys) / spacing) * spacing
        width = math.ceil((max(xs) - left) / spacing)
        height = math.ceil((top - min(ys)) / spacing)
    return from_origin(left, top, spacing, spacing), width, height


def _write_stac_outputs(cog, manifest):
    # Metadata dependencies are optional and loaded only when requested.
    from stac_output import write_stac
    return write_stac(cog, manifest)


def _season_dates(year, windows=None):
    if isinstance(year, bool) or not isinstance(year, (int, np.integer)) or not 1900 <= year <= 9998:
        raise ValueError("year must be an integer between 1900 and 9998")
    windows = DEFAULT_WINDOWS if windows is None else windows
    if set(windows) != set(SEASONS):
        raise ValueError("season_windows must contain frozen, summer and shoulder")
    dates = {}
    for season in SEASONS:
        first, last = windows[season]
        dates[season] = _calendar_window(year, first, last)
    return dates


def _calendar_window(year, first, last):
    if not all(isinstance(s, str) and re.fullmatch(r"\d{2}-\d{2}", s) for s in (first, last)):
        raise ValueError("Season endpoints must be month-day strings, e.g. '06-15'")
    begin, finish = _utc(f"{year}-{first}"), _utc(f"{year}-{last}")
    if begin >= finish:
        raise ValueError("Each season must have start < exclusive end within the same year")
    return begin, finish


def _resolve_cutoff(as_of=None):
    now = datetime.now(timezone.utc)
    cutoff = now if as_of is None else _utc(as_of.isoformat() if isinstance(as_of, datetime) else as_of)
    if cutoff > now or cutoff <= ARCHIVE_START:
        raise ValueError("as_of must be after the OPERA archive start and no later than now")
    return cutoff


def _season_availability(year, window, cutoff, *, donor=False):
    """Only complete, archive-supported seasons are eligible as donors."""
    info = dict(year=int(year), search_results=0, retained_bursts=0, passes=0, records=[])
    try:
        begin, finish = _calendar_window(year, *window)
    except ValueError:
        if not donor:
            raise
        return dict(info, status="invalid_calendar_window")
    query_start, query_end = max(begin, ARCHIVE_START), min(finish, cutoff)
    info.update(start=begin.isoformat(), end_exclusive=finish.isoformat(),
                nominal_doy=(begin + (finish - begin) / 2).timetuple().tm_yday,
                complete=begin >= ARCHIVE_START and finish <= cutoff,
                query_start=query_start.isoformat(), query_end_exclusive=query_end.isoformat())
    if finish <= ARCHIVE_START:
        return dict(info, status="before_archive_start")
    if begin >= cutoff:
        return dict(info, status="not_started_as_of_cutoff")
    if donor and not info["complete"]:
        return dict(info, status="incomplete_donor_season")
    return dict(info, status="ready")


SOURCE_OFFSETS = (0, -1, 1, -2, 2, -3, 3)
META_MAX_COUNT = 127


def _pack_quality(source_year, counts, year):
    """Pack one season into two 10-bit fields, using integer operations only."""
    if np.any((counts < 0) | (counts > META_MAX_COUNT)):
        raise ValueError("Metadata count exceeds 127; exact counts cannot be packed without loss")
    codes = np.full(source_year.shape, 7, dtype="int64")
    for code, offset in enumerate(SOURCE_OFFSETS):
        codes[(source_year == year + offset) & (counts > 0)] = code
    if np.any((counts > 0) & (codes == 7)):
        raise ValueError("Metadata source year exceeds the supported +/-3-year range")
    packed = np.zeros(counts.shape[1:], dtype="int64")
    for p in range(2):
        packed |= (codes[p] | (counts[p].astype("int64") << 3)) << (10 * p)
    return packed


def _packed_values(values):
    values = np.asanyarray(values)
    # Never accept float arrays: Int64 QA may already have lost low bits above
    # 2**53, and casting back cannot recover or reliably detect that loss.
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Packed metadata must be read as Int64, never Float32/Float64")
    present = ~np.ma.getmaskarray(values) & (np.ma.getdata(values) != int(NODATA))
    values = np.ma.getdata(values)
    if np.any(present & ((values < 0) | (values > 2**60 - 1))):
        raise ValueError("Invalid packed metadata value")
    return np.where(present, values, 0).astype("int64"), present


def _decode_field(codes, present, year, window):
    valid = present & (codes != 7)
    years = np.full(codes.shape, int(NODATA), dtype="int16")
    doys = np.full_like(years, int(NODATA))
    for code, offset in enumerate(SOURCE_OFFSETS):
        use = valid & (codes == code)
        if np.any(use):
            begin, end = _calendar_window(year + offset, *window)
            years[use] = year + offset
            doys[use] = (begin + (end - begin) / 2).timetuple().tm_yday
    return dict(source_year=years, source_doy=doys, valid=valid,
                gap_filled=valid & (years != year))


def decode_meta(values, year, season_windows=None):
    """Decode the single Int64 companion band into six named diagnostic fields.

    Fields (vv_frozen, vh_frozen, vv_summer, vh_summer, vv_shoulder, vh_shoulder)
    contain source_year, source_doy, source_count, target_count, valid and
    gap_filled arrays. Supply the product's target year and seasonal windows.
    Original target counts are zero where filled or unresolved; otherwise they
    equal selected-source counts. Outside-bbox integer fields are -9999.
    Read with Rasterio's native Int64 dtype; float conversion loses packed bits.
    """
    packed, present = _packed_values(values)
    windows = DEFAULT_WINDOWS if season_windows is None else season_windows
    result = {}
    for s, season in enumerate(SEASONS):
        for p, pol in enumerate(("vv", "vh")):
            field = (packed >> (10 * (2 * s + p))) & 1023
            codes, counts = field & 7, field >> 3
            if np.any(present & ((codes == 7) != (counts == 0))):
                raise ValueError("Invalid metadata: source year/count validity differs")
            decoded = _decode_field(codes, present, year, windows[season])
            decoded["source_count"] = np.where(present, counts, int(NODATA)).astype("int16")
            decoded["target_count"] = np.where(present,
                np.where(decoded["valid"] & ~decoded["gap_filled"], counts, 0), int(NODATA)).astype("int16")
            result[f"{pol}_{season}"] = decoded
    return result


def _eligible_gap_mask(missing, min_pixels, core_size):
    """Retain substantial 8-connected gaps containing a fully missing core.

    Eligibility is computed on the original target-year mask, once per season
    and polarization. The entire qualifying component (including its edge) is
    eligible; erosion only tests its width, it does not trim the filled patch.
    """
    missing = np.asarray(missing, dtype=bool)
    total = int(np.count_nonzero(missing))
    eligible = np.zeros_like(missing)
    stats = dict(missing_pixels=total, eligible_pixels=0, skipped_pixels=total,
                 eligible_patches=0, largest_patch_pixels=0 if total == 0 else None)
    # No label scan or donor query for a season with only a handful of holes.
    if total < min_pixels:
        return eligible, stats
    from scipy import ndimage
    labels, count = ndimage.label(missing, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    has_core = np.zeros(count + 1, dtype=bool)
    if core_size == 1:
        has_core[1:] = True
    else:
        core = ndimage.minimum_filter(missing, size=core_size, mode="constant", cval=0)
        has_core[labels[core]] = True
    keep = (sizes >= min_pixels) & has_core
    keep[0] = False
    eligible = keep[labels]
    stats.update(eligible_pixels=int(np.count_nonzero(eligible)),
                 eligible_patches=int(np.count_nonzero(keep)), largest_patch_pixels=int(sizes.max()))
    stats["skipped_pixels"] = total - stats["eligible_pixels"]
    return eligible, stats


def _fill_season(mean, nobs, year, nominal_doy, inside, max_gap_years, load, eligible_gaps=None,
                 gap_min_pixels=1, gap_core_size=1):
    """Select the nearest valid donor per polarization; do not mutate nobs.

    Search -1, +1, -2, +2, ... and keep the first valid donor per pixel and
    polarization. Following years see only still-unresolved eligible gaps;
    stop without loading another donor once these are filled. Never overwrite
    target-year values or previously selected donors, regardless of counts.
    Reapply the patch policy to residual gaps before subsequent donor attempts.
    """
    valid = (nobs > 0) & inside[None]
    source_year = np.where(valid, year, 0).astype("int16")
    source_doy = np.where(valid, nominal_doy, 0).astype("int16")
    source_count = nobs.copy()
    original_gap = ~valid & inside[None]
    if eligible_gaps is not None:
        if eligible_gaps.shape != original_gap.shape:
            raise ValueError("Gap eligibility mask must match both polarization grids")
        original_gap &= eligible_gaps
    attempts = []
    for distance in range(1, max_gap_years + 1):
        if not np.any(original_gap & (source_year == 0)):
            break
        for candidate in (year - distance, year + distance):
            eligible = original_gap & (source_year == 0)
            if not np.any(eligible):
                break
            if attempts and (gap_min_pixels > 1 or gap_core_size > 1):
                for k in range(2):
                    if k == 1 and np.array_equal(original_gap[0] & (source_year[0] == 0), eligible[k]):
                        eligible[k] = eligible[0]
                    else:
                        eligible[k], _ = _eligible_gap_mask(eligible[k], gap_min_pixels, gap_core_size)
                # Once residuals become specks, exclude them from every farther candidate.
                original_gap &= (source_year != 0) | eligible
                if not np.any(eligible):
                    print("No qualifying residual missing patches; skipping remaining donor years", flush=True)
                    return mean, source_year, source_doy, source_count, attempts
            if not ARCHIVE_START.year <= candidate <= 9998:
                continue
            donor_mean, donor_count, info = load(candidate, donor=True, needed=np.any(eligible, axis=0))
            if donor_count is not None:
                use = eligible & (donor_count > 0) & np.isfinite(donor_mean) & (donor_mean != NODATA)
                mean[use] = donor_mean[use]
                source_count[use] = donor_count[use]
                source_year[use] = candidate
                source_doy[use] = info["nominal_doy"]
            attempts.append(info)
            del donor_mean, donor_count
    return mean, source_year, source_doy, source_count, attempts


def _gap_windows(needed, halo=HALO):
    """Partition sparse gaps; keep dense gaps together. Return padded slices.

    Split the longest dimension until each tight bounding rectangle is small
    (<=256x256 pixels in area) or at least 25% of its pixels need filling.
    Dense/fully missing tiles use one large window rather than many requests.
    The halo preserves the full 7x7 Refined Lee neighborhood at requested pixels.
    """
    height, width = needed.shape
    pending = [(0, height, 0, width)]
    windows = []
    while pending:
        top, bottom, left, right = pending.pop()
        mask = needed[top:bottom, left:right]
        rows = np.flatnonzero(np.any(mask, axis=1))
        if not rows.size:
            continue
        cols = np.flatnonzero(np.any(mask, axis=0))
        top, bottom = top + int(rows[0]), top + int(rows[-1]) + 1
        left, right = left + int(cols[0]), left + int(cols[-1]) + 1
        h, w = bottom - top, right - left
        area = h * w
        if area <= 256**2 or np.count_nonzero(mask) >= area / 4:
            windows.append((slice(max(0, top - halo), min(height, bottom + halo)),
                            slice(max(0, left - halo), min(width, right + halo))))
        elif h >= w:
            mid = (top + bottom) // 2
            pending.extend([(top, mid, left, right), (mid, bottom, left, right)])
        else:
            mid = (left + right) // 2
            pending.extend([(top, bottom, left, mid), (top, bottom, mid, right)])
    return sorted(windows, key=lambda window: (window[0].start, window[1].start))


def _footprint_bounds(granule):
    """Optional conservative CMR bounds; absent/ambiguous footprints are kept."""
    geometry = granule.get("Spatial", {}).get("HorizontalSpatialDomain", {}).get("Geometry", {})
    points = [point for polygon in _items(geometry.get("GPolygon", []))
              for point in _items(polygon.get("Boundary", {}).get("Point", []))]
    if not points:
        return None
    xs = [float(p["PointLongitude"]) for p in points]
    ys = [float(p["PointLatitude"]) for p in points]
    if max(xs) - min(xs) > 180 or max(map(abs, ys)) > 85:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _window_groups(groups, bbox):
    # A generous 1 km margin avoids excluding interpolation/support pixels near
    # the catalog footprint edge. Keep all records when footprint is unavailable.
    west, south, east, north = bbox
    pad_y = 1000 / 110000
    pad_x = pad_y / max(0.01, math.cos(math.radians(max(abs(south), abs(north)))))
    def intersects(record):
        bounds = record.get("footprint_bounds")
        return bounds is None or not (bounds[2] < west - pad_x or bounds[0] > east + pad_x
                                     or bounds[3] < south - pad_y or bounds[1] > north + pad_y)
    return [retained for group in groups if (retained := [r for r in group if intersects(r)])]


def _support_mask(needed, inside, halo):
    """All pixels needed for requested outputs, including filter neighbors."""
    if halo:
        from scipy.ndimage import maximum_filter
        return maximum_filter(needed, size=2 * halo + 1, mode="constant", cval=0) & inside
    return needed & inside


def _bounds_window(bounds, affine, shape, margin=0):
    """Conservative pixel rectangle, including rotated grids and edge pixels."""
    height, width = shape
    if not all(math.isfinite(v) for v in bounds):
        return slice(0, height), slice(0, width)
    left, bottom, right, top = bounds
    pixels = [(~affine) * xy for xy in ((left-margin, bottom-margin),
              (right+margin, bottom-margin), (right+margin, top+margin), (left-margin, top+margin))]
    cols, rows = zip(*pixels)
    # One extra output pixel protects boundary rounding in transformed bounds.
    row0, row1 = max(0, math.floor(min(rows))-1), min(height, math.ceil(max(rows))+1)
    col0, col1 = max(0, math.floor(min(cols))-1), min(width, math.ceil(max(cols))+1)
    return slice(min(height, row0), max(0, row1)), slice(min(width, col0), max(0, col1))


def _support_groups(groups, support, crs, affine):
    """Skip catalog envelopes that miss the actual gap plus filter halo.

    Use padded bounding envelopes, not approximate footprint polygons. Missing,
    dateline/polar or unprojectable bounds remain eligible for a raster check.
    """
    from rasterio.errors import RasterioError
    from rasterio.warp import transform_bounds
    cache = {}
    def intersects(record):
        bounds = record.get("footprint_bounds")
        if bounds is None:
            return True
        bounds = tuple(bounds)
        if bounds not in cache:
            if bounds[2] - bounds[0] > 180 or max(abs(bounds[1]), abs(bounds[3])) > 85:
                cache[bounds] = True
            else:
                try:
                    projected = transform_bounds("EPSG:4326", crs, *bounds, densify_pts=101)
                    rows, cols = _bounds_window(projected, affine, support.shape, margin=1000)
                    cache[bounds] = bool(np.any(support[rows, cols]))
                except (RasterioError, ValueError, OverflowError):
                    cache[bounds] = True
        return cache[bounds]
    return [retained for group in groups if (retained := [r for r in group if intersects(r)])]


def _output_name(year, identifier):
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", identifier):
        raise ValueError("identifier must use letters, numbers, underscores or hyphens")
    return f"SAR_S1_{year}_{identifier}.tif"


def _band_names(season):
    return (f"vvmean_{season}", f"vhmean_{season}")


def _utc(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _items(value):
    return value if isinstance(value, list) else [value]


def _records(results, start, end):
    """Parse the supplied ECHO10 schema, retaining the latest burst processing."""
    selected = {}
    for result in results:
        g = result["Granule"]
        acquired = _utc(g["Temporal"]["RangeDateTime"]["BeginningDateTime"])
        if not start <= acquired < end:
            continue
        urls = [x["URL"] for x in _items(g["OnlineAccessURLs"]["OnlineAccessURL"])]
        assets = {band: next((u for u in urls if u.startswith("s3://")
                             and u.endswith(f"_{band}.tif")), None)
                  for band in ("VV", "VH", "mask")}
        if not assets["VV"] or not assets["VH"]:
            continue  # HH/HV acquisitions are outside this VV/VH composite.
        if not assets["mask"]:
            raise ValueError(f"Missing per-acquisition mask: {g['GranuleUR']}")
        attrs = {x["Name"]: x["Values"]["Value"] for x in
                 _items(g["AdditionalAttributes"]["AdditionalAttribute"])}
        platform = g["Platforms"]["Platform"]["ShortName"]
        orbit_info = _items(g["OrbitCalculatedSpatialDomains"]["OrbitCalculatedSpatialDomain"])[0]
        orbit = str(orbit_info["OrbitNumber"])
        burst = attrs["OPERA_BURST_ID"]
        # Semantic product version, then actual production time (not insertion time).
        version = tuple(int(x) for x in str(attrs["PRODUCT_VERSION"]).split("."))
        produced = _utc(g["DataGranule"]["ProductionDateTime"])
        key = (platform, burst, acquired)
        record = dict(id=g["GranuleUR"], acquired=acquired.isoformat(),
                      platform=platform, orbit=orbit, burst=burst, assets=assets,
                      direction=attrs.get("ASCENDING_DESCENDING"),
                      relative_orbit=attrs.get("PATH_NUMBER"), footprint_bounds=_footprint_bounds(g))
        if key not in selected or (version, produced) > selected[key][0]:
            selected[key] = ((version, produced), record)
    groups = defaultdict(list)
    for _, record in selected.values():
        groups[(record["platform"], record["orbit"])].append(record)
    return [sorted(group, key=lambda r: r["id"]) for _, group in sorted(groups.items())]


def _composite(groups, shape, reader):
    """Accumulate equal-pass means from full or compact burst windows.

    A compact reader returns (bands, clear, (rows, cols)) relative to this grid.
    Bursts still merge within their original pass before temporal averaging.
    """
    total = np.zeros((2, *shape), dtype="float64")
    nobs = np.zeros((2, *shape), dtype="uint32")
    for i, group in enumerate(groups, 1):
        pass_sum = np.zeros_like(total)
        pass_count = np.zeros_like(nobs)
        for record in group:
            result = reader(record)
            bands, clear = result[:2]
            rows, cols = result[2] if len(result) == 3 else (slice(None), slice(None))
            valid = clear[None, :, :] & np.isfinite(bands)
            pass_sum[:, rows, cols] += np.where(valid, bands, 0)
            pass_count[:, rows, cols] += valid
        covered = pass_count > 0
        np.divide(pass_sum, pass_count, out=pass_sum, where=covered)
        total += pass_sum
        nobs += covered
        print(f"Pass {i}/{len(groups)}: {len(group)} bursts", flush=True)
    mean = np.full(total.shape, np.nan, dtype="float32")
    np.divide(total, nobs, out=mean, where=nobs > 0)
    return mean, nobs


def _read_burst(record, *, crs, affine, width, height, inside, resampling, support=None, metrics=None):
    """Mask native data before warping; optionally read only donor gap support.

    Compact results include their destination slices. Cross-CRS warps retain
    the original source/destination geometry: padding unread native pixels with
    nodata preserves GDAL's approximate transform and nearest sample selection.
    Same-CRS reads can crop both geometries. Bursts are never speckle-filtered.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject, transform_bounds
    from rasterio.windows import Window, from_bounds

    if support is not None and (support.shape != inside.shape or resampling != Resampling.nearest):
        raise ValueError("Compact reads require nearest resampling and matching support/inside grids")
    if metrics is not None:
        metrics.update(source_read_pixels=0, warp_pixels=0)
    compact_window = None
    def empty():
        if support is not None:
            return np.empty((2, 0, 0), "float32"), np.zeros((0, 0), bool), (slice(0,0), slice(0,0))
        return np.full((2, height, width), np.nan, "float32"), np.zeros((height, width), bool)
    def source_window(grid_affine, grid_width, grid_height, src):
        corners = [grid_affine * xy for xy in ((0,0), (grid_width,0),
                   (grid_width,grid_height), (0,grid_height))]
        xs, ys = zip(*corners)
        bounds = transform_bounds(crs, src.crs, min(xs), min(ys), max(xs), max(ys), densify_pts=41)
        proposed = from_bounds(*bounds, transform=src.transform)
        col0 = max(0, math.floor(proposed.col_off) - 3)
        row0 = max(0, math.floor(proposed.row_off) - 3)
        col1 = min(src.width, math.ceil(proposed.col_off + proposed.width) + 3)
        row1 = min(src.height, math.ceil(proposed.row_off + proposed.height) + 3)
        return Window(col0, row0, max(0, col1-col0), max(0, row1-row0))
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(record["assets"][p])) for p in ("VV", "VH", "mask")]
        vv, vh, mask = sources
        if any(s.crs != vv.crs or s.transform != vv.transform or s.shape != vv.shape for s in (vh, mask)):
            raise ValueError(f"Burst bands/mask grids differ: {record['id']}")
        if vv.crs is None or vv.transform.b != 0 or vv.transform.d != 0 or vv.transform.a <= 0 or vv.transform.e >= 0:
            raise ValueError("Expected a north-up georeferenced OPERA source grid")
        if any(v != 1 for s in (vv, vh) for v in s.scales) or any(v != 0 for s in (vv, vh) for v in s.offsets):
            raise ValueError("Expected OPERA linear power without scale/offset encoding")
        window = source_window(affine, width, height, vv)
        if not window.width or not window.height:
            return empty()
        read_window = window
        if support is not None:
            bounds = transform_bounds(vv.crs, crs, *vv.bounds, densify_pts=101)
            rows, cols = _bounds_window(bounds, affine, inside.shape,
                                       margin=3 * max(abs(vv.transform.a), abs(vv.transform.e)))
            overlap = support[rows, cols]
            occupied_rows = np.flatnonzero(np.any(overlap, axis=1))
            if not occupied_rows.size:
                return empty()
            occupied_cols = np.flatnonzero(np.any(overlap, axis=0))
            compact_window = (slice(rows.start + int(occupied_rows[0]), rows.start + int(occupied_rows[-1]) + 1),
                              slice(cols.start + int(occupied_cols[0]), cols.start + int(occupied_cols[-1]) + 1))
            rows, cols = compact_window
            compact_affine = affine * rasterio.Affine.translation(cols.start, rows.start)
            compact_shape = (rows.stop-rows.start, cols.stop-cols.start)
            proposed = source_window(compact_affine, compact_shape[1], compact_shape[0], vv)
            if not proposed.width or not proposed.height:
                return empty()
            read_window = window.intersection(proposed)
            if vv.crs == crs:
                # Linear transform: the compact geometry has identical sample locations.
                affine = compact_affine
                inside = inside[rows, cols] & support[rows, cols]
                height, width = compact_shape
                window = read_window
            else:
                # Preserve warp/chunk geometry across CRSs. Only the data read is cropped.
                inside = inside & support
        flags = mask.read(1, window=read_window, masked=True)
        read_clear = ~np.ma.getmaskarray(flags) & (flags.data == 0)
        read_bands = np.stack([s.read(1, window=read_window, masked=True).astype("float32").filled(np.nan)
                               for s in (vv, vh)])
        valid = read_clear[None] & np.isfinite(read_bands)
        read_bands = np.where(valid, read_bands, np.nan)
        if window == read_window:
            bands, clear = read_bands, read_clear
        else:
            # This nodata canvas has exactly the rc.5 source shape and transform.
            native_shape = int(window.height), int(window.width)
            bands = np.full((2, *native_shape), np.nan, "float32")
            clear = np.zeros(native_shape, bool)
            row0, col0 = int(read_window.row_off-window.row_off), int(read_window.col_off-window.col_off)
            region = (slice(row0,row0+int(read_window.height)), slice(col0,col0+int(read_window.width)))
            bands[:,region[0],region[1]] = read_bands
            clear[region] = read_clear
        destination = np.full((2, height, width), np.nan, "float32")
        warp = dict(src_transform=vv.window_transform(window), src_crs=vv.crs, dst_transform=affine, dst_crs=crs)
        for b in range(2):
            reproject(bands[b], destination[b], **warp, src_nodata=np.nan, dst_nodata=np.nan, resampling=resampling)
        warped_mask = np.full((height, width), 255, "uint8")
        reproject(np.where(clear, 0, 255).astype("uint8"), warped_mask, **warp,
                  src_nodata=255, dst_nodata=255, resampling=Resampling.nearest)
        clear_target = (warped_mask == 0) & inside
        destination[:, ~clear_target] = np.nan
        if metrics is not None:
            metrics.update(source_read_pixels=3 * int(read_window.height * read_window.width), warp_pixels=height*width)
        if compact_window is None:
            return destination, clear_target
        if vv.crs != crs:
            rows, cols = compact_window
            destination, clear_target = destination[:,rows,cols].copy(), clear_target[rows,cols].copy()
        return destination, clear_target, compact_window


def _remote_error_kind(error):
    messages, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        messages.append(str(error))
        error = error.__cause__ or error.__context__
    text = " ".join(messages).lower()
    if "nosuchkey" in text or re.search(r"http(?: response)?(?: error)?(?: code)?\s*[:=]?\s*404\b", text):
        return "not_found"
    if any(token in text for token in ("expiredtoken", "invalidtoken", "signaturedoesnotmatch", "accessdenied")) or re.search(
            r"http(?: response)?(?: error)?(?: code)?\s*[:=]?\s*(401|403)\b", text):
        return "authentication"
    if "slowdown" in text or re.search(r"\b(429|500|502|503|504)\b", text):
        return "transient_http"
    return "remote_io"


def _retry_remote_read(operation, record, refresh, max_attempts=3):
    """Retry a whole remote burst with fresh handles, never skip observations."""
    from rasterio.errors import RasterioIOError
    remote = any(str(url).startswith(("s3://", "https://")) for url in record["assets"].values())
    refreshed = False
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except RasterioIOError as error:
            kind = _remote_error_kind(error)
            if not remote or kind == "not_found":
                raise
            if attempt == max_attempts or kind == "authentication" and refreshed:
                raise RasterioIOError(
                    f"Remote burst read failed after {attempt} attempts: {record['id']} ({kind}); no observations skipped"
                ) from error
            if not refreshed and kind in ("authentication", "remote_io"):
                # GDAL sometimes hides the HTTP status behind a TIFF read error.
                refresh()
                refreshed = True
            delay = (5 if attempt == 1 else 15) + random.uniform(0, 2)
            print(f"Remote burst {record['id']}: {kind}; retry {attempt + 1}/{max_attempts} "
                  f"in {delay:.1f}s with reopened datasets", flush=True)
            time.sleep(delay)


def _credential_lifetime(credentials):
    """Refresh before actual expiration, with a 45-minute fallback ceiling."""
    expiration = credentials.get("expiration") or credentials.get("Expiration")
    if expiration is None:
        return 45 * 60
    remaining = (_utc(str(expiration)) - datetime.now(timezone.utc)).total_seconds()
    if remaining <= 60:
        raise RuntimeError("Earthdata returned expired credentials or less than one minute of validity")
    return max(1, min(45 * 60, remaining - 120))


def seasonal_composites(maap, bbox, year, identifier, *, outdir="rtc_seasonal",
                        resampling="nearest", speckle_filter=True,
                        season_windows=None, crs=None, spacing=30,
                        template=None, max_results=20000, max_pixels=25000000,
                        filter_tile_size=512, overwrite=False, write_stac=False,
                        bbox_crs="EPSG:4326", gap_fill=False, max_gap_years=1,
                        as_of=None, write_meta=False, gap_min_pixels=1024, gap_core_size=7):
    """Write six Float32 seasonal means and optional packed Int64 metadata COG.

    Default windows: Jan 1-Mar 31, Jun 15-Aug 31, Sep 15-Oct 31. Custom
    season_windows use month-day pairs with EXCLUSIVE ends in the same year.
    Means are linear gamma-zero power; all output nodata is -9999.
    write_meta=True writes one Int64 band with six 10-bit fields: three source
    bits (target, -1, +1, -2, +2, -3, +3, missing), seven exact count bits
    (0-127). Use decode_meta with the recorded target year and seasonal windows
    to obtain per-polarization source year, nominal DOY, counts and fill flags.
    Counts above 127 fail explicitly only when write_meta=True.
    bbox accepts xmin,ymin,xmax,ymax as a DPS string or coordinate sequence,
    in bbox_crs (default EPSG:4326). Explicit output crs or template required.
    Matching input/output CRS preserves bounds/origin at whole-pixel extents.
    resampling: nearest (default), bilinear, average. A template uses its full
    grid with the bbox mask. Refined Lee is applied once per source-year mean.
    gap_fill=True searches complete same-season donors within max_gap_years
    (0-3), previous year first at each distance, keeping the first valid donor.
    Preserve observed values. Only 8-connected missing patches
    with at least gap_min_pixels (1024) and a missing gap_core_size square core
    (7x7) are eligible; recheck residual patches before subsequent donors.
    Skipped holes remain nodata without donor searches.
    Set gap_min_pixels=1 and gap_core_size=1 for rc.2 all-gap behavior.
    as_of is an exclusive UTC cutoff,
    default now; incomplete seasons are never donors. write_stac=True adds
    an optional item/catalog linking the composite, manifest and metadata COG.
    """
    import boto3
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import geometry_mask
    from rasterio.session import AWSSession
    from rasterio.shutil import copy as rio_copy
    from rasterio.warp import transform_geom

    dates = _season_dates(year, season_windows)
    windows = DEFAULT_WINDOWS if season_windows is None else season_windows
    cutoff = _resolve_cutoff(as_of)
    if not ARCHIVE_START.year <= year <= cutoff.year:
        raise ValueError("Target year must be within the OPERA archive through as_of")
    if not isinstance(gap_fill, bool):
        raise ValueError("gap_fill must be True or False")
    if isinstance(max_gap_years, bool) or not isinstance(max_gap_years, (int, np.integer)) or not 0 <= max_gap_years <= 3:
        raise ValueError("max_gap_years must be an integer between 0 and 3 (packed QA range)")
    for name, value in (("gap_min_pixels", gap_min_pixels), ("gap_core_size", gap_core_size)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if gap_core_size % 2 != 1:
        raise ValueError("gap_core_size must be odd")
    filename = _output_name(year, identifier)
    bbox_crs = resolve_bbox_crs(bbox_crs)
    bbox = parse_bbox(bbox, bbox_crs)
    if not template:
        crs = resolve_crs(crs)
    allowed = {"nearest": Resampling.nearest, "bilinear": Resampling.bilinear,
               "average": Resampling.average}
    if isinstance(resampling, Resampling):
        resampling = resampling.name
    if resampling not in allowed:
        raise ValueError("resampling must be 'nearest', 'bilinear' or 'average'")
    if not isinstance(speckle_filter, bool):
        raise ValueError("speckle_filter must be True or False")
    if not isinstance(write_meta, bool):
        raise ValueError("write_meta must be True or False")
    if not isinstance(write_stac, bool):
        raise ValueError("write_stac must be True or False")
    if isinstance(filter_tile_size, bool) or not isinstance(filter_tile_size, (int, np.integer)) or filter_tile_size < 1:
        raise ValueError("filter_tile_size must be a positive integer")
    if speckle_filter:
        print("Refined Lee: NumPy, once per seasonal mean after compositing", flush=True)
    folder = Path(outdir)
    final = folder / filename
    manifest = folder / (final.stem + "_manifest.json")
    meta_final = folder / (final.stem + "_meta.tif")
    if meta_final.exists() and not write_meta:
        raise FileExistsError(f"Stale metadata output exists: {meta_final}; use a fresh output directory")
    if not overwrite and any(path.exists() for path in (final, manifest, meta_final)):
        raise FileExistsError(f"Output exists for {final}; use overwrite=True to replace")
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive.")
    geometry = _bbox_geometry(bbox)
    search_bbox = _search_bbox(geometry, bbox_crs)
    if template:
        with rasterio.open(template) as ref:
            crs, affine, width, height = ref.crs, ref.transform, ref.width, ref.height
        if crs is None:
            raise ValueError("Template must have a CRS")
    else:
        affine, width, height = _grid_from_bbox(bbox, bbox_crs, crs, spacing)
    if width * height > max_pixels:
        raise ValueError("Grid exceeds max_pixels; process smaller tiles.")
    geom = transform_geom(bbox_crs, crs, geometry)
    inside = geometry_mask([geom], (height, width), affine, invert=True)
    if not np.any(inside):
        raise ValueError("Bounding box does not intersect the template grid.")

    endpoint = "https://cumulus.asf.earthdatacloud.nasa.gov/s3credentials"
    aws, refresh_at = None, 0

    def invalidate_credentials():
        nonlocal aws, refresh_at
        aws, refresh_at = None, 0

    def read_once(record, grid, support=None, metrics=None):
        nonlocal aws, refresh_at
        if aws is None or time.monotonic() >= refresh_at:
            credentials = maap.aws.earthdata_s3_credentials(endpoint)
            lifetime = _credential_lifetime(credentials)
            aws = AWSSession(boto3.Session(
                aws_access_key_id=credentials["accessKeyId"],
                aws_secret_access_key=credentials["secretAccessKey"],
                aws_session_token=credentials["sessionToken"], region_name="us-west-2"))
            refresh_at = time.monotonic() + lifetime
        # Invalidate these assets' HTTP caches when their handles close, including
        # failed reads, so retries reopen rather than reuse a poisoned response.
        non_cached = ":".join("/vsis3/" + url[5:] if url.startswith("s3://") else url
                              for url in record["assets"].values())
        with rasterio.Env(aws, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                          GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2",
                          GDAL_HTTP_RETRY_CODES="429,500,502,503,504",
                          GDAL_HTTP_CONNECTTIMEOUT="15", GDAL_HTTP_TIMEOUT="120",
                          CPL_VSIL_CURL_NON_CACHED=non_cached):
            grid_affine, grid_inside = (affine, inside) if grid is None else grid
            grid_height, grid_width = grid_inside.shape
            extra = {"metrics": metrics}
            if support is not None:
                extra["support"] = support
            return _read_burst(record, crs=crs, affine=grid_affine, width=grid_width,
                               height=grid_height, inside=grid_inside, resampling=allowed[resampling], **extra)

    def read(record, grid=None, support=None, metrics=None):
        return _retry_remote_read(lambda: read_once(record, grid, support, metrics), record, invalidate_credentials)

    folder.mkdir(parents=True, exist_ok=True)
    metadata = dict(year=int(year), identifier=identifier, bbox=list(bbox),
                    bbox_crs=bbox_crs.to_wkt(), search_bbox_wgs84=list(search_bbox),
                    crs=crs.to_wkt(), transform=list(affine)[:6], width=width, height=height,
                    template=str(template) if template else None, mask_values_kept=[0],
                    resampling=resampling, mask_resampling="nearest",
                    temporal_weighting="equal_valid_satellite_orbit",
                    speckle_filter="Refined_Lee" if speckle_filter else "none",
                    speckle_filter_version=FILTER_VERSION if speckle_filter else None,
                    nodata=NODATA, counts_zero="no_valid_passes_inside_bbox",
                    filter_stage="after_seasonal_compositing" if speckle_filter else "none",
                    as_of=cutoff.isoformat(), archive_start=ARCHIVE_START.isoformat(),
                    gap_fill=dict(enabled=gap_fill, max_gap_years=int(max_gap_years),
                                  selection="nearest_year_previous_first_first_valid_per_polarization",
                                  candidate_order="-1,+1,-2,+2,-3,+3 (bounded by max_gap_years)",
                                  compare_donor_counts=False, recheck_residual_patches=True, donors_require_complete_season=True,
                                  eligibility=dict(min_pixels=int(gap_min_pixels), core_size=int(gap_core_size),
                                      connectivity=8, scope="original_and_residual_masks_per_polarization", outside_tile="excluded"),
                                  spatial_processing="adaptive_gap_windows_with_burst_support_nearest",
                                  burst_pruning="padded_catalog_envelope_intersects_gap_and_filter_halo",
                                  burst_cropping="native_raster_bounds_and_gap_support",
                                  cross_crs_warp="preserve_original_geometry_with_nodata_padding",
                                  other_resampling="full_tile_to_preserve_warp_results",
                                  donor_filter_halo_pixels=HALO if speckle_filter else 0,
                                  original_counts="target_year_only",
                                  nominal_doy="UTC calendar day containing midpoint of source season window"),
                    remote_read_recovery=dict(burst_attempts=3, http_retries=3, backoff_seconds=[5, 15],
                        jitter_max_seconds=2, expiration_margin_seconds=120, skip_failed_bursts=False),
                    metadata_raster=(dict(filename=meta_final.name, dtype="int64", nodata=int(NODATA),
                        encoding_version=1, bits=60, bits_per_field=10,
                        source_bits=[0, 2], count_bits=[3, 9], count_max=META_MAX_COUNT,
                        order=[f"{pol}_{season}" for season in SEASONS for pol in ("vv", "vh")],
                        source_offsets=list(SOURCE_OFFSETS), missing_source_code=7,
                        bands=[dict(index=1, name="qa", units="packed_metadata", role="source_provenance")]) if write_meta else None),
                    bands=[], seasons={})
    any_usable = False
    source_intervals = []
    with tempfile.TemporaryDirectory(prefix=final.stem + "_", dir=folder) as temp:
        temporary, cog = Path(temp) / "stack.tif", Path(temp) / filename
        meta_temporary, meta_cog = Path(temp) / "diagnostics.tif", Path(temp) / meta_final.name
        packed_meta = np.zeros((height, width), dtype="int64") if write_meta else None
        with ExitStack() as stack:
            profile = dict(driver="GTiff", width=width, height=height, crs=crs,
                           transform=affine, nodata=NODATA, tiled=True, compress="deflate")
            dst = stack.enter_context(rasterio.open(temporary, "w", count=6, dtype="float32", **profile))
            diagnostics = (stack.enter_context(rasterio.open(meta_temporary, "w", count=1,
                           dtype="int64", **profile)) if write_meta else None)
            if diagnostics:
                diagnostics.update_tags(year=year, identifier=identifier,
                    encoding=json.dumps(metadata["metadata_raster"]), composite=filename)
            dst.update_tags(year=year, identifier=identifier, resampling=resampling,
                            mask_resampling="nearest", mask="OPERA_mask_equals_0",
                            temporal_weighting="equal_valid_pass", counts_zero="no_valid_passes_inside_bbox",
                            speckle_filter=metadata["speckle_filter"],
                            filter_stage=metadata["filter_stage"], gap_fill=str(gap_fill).lower(),
                            original_counts="target_year_only", as_of=cutoff.isoformat())
            for s, season in enumerate(SEASONS):
                begin, finish = dates[season]
                def load_season(source_year, *, donor=False, needed=None):
                    started = time.monotonic()
                    info = _season_availability(source_year, windows[season], cutoff, donor=donor)
                    if info["status"] != "ready":
                        print(f"{season} {source_year}: {info['status']}", flush=True)
                        return None, None, info
                    first, last = _utc(info["query_start"]), _utc(info["query_end_exclusive"])
                    search_started = time.monotonic()
                    results = maap.searchGranule(
                        cmr_host="cmr.earthdata.nasa.gov", short_name="OPERA_L2_RTC-S1_V1",
                        temporal=f"{first.strftime('%Y-%m-%dT%H:%M:%SZ')},{last.strftime('%Y-%m-%dT%H:%M:%SZ')}",
                        bounding_box=",".join(map(str, search_bbox)), limit=max_results)
                    if len(results) >= max_results:
                        raise RuntimeError(f"{season} {source_year} search reached max_results; increase it or use smaller tiles")
                    groups = _records(results, first, last)
                    info["search_seconds"] = time.monotonic() - search_started
                    info.update(search_results=len(results), retained_bursts=sum(map(len, groups)),
                                passes=len(groups), records=[r for group in groups for r in group])
                    print(f"{season} {source_year}: {len(results)} search results; {info['retained_bursts']} retained bursts; "
                          f"{len(groups)} satellite passes.", flush=True)
                    # GDAL's non-nearest kernels can depend on warp-window
                    # geometry (XSCALE/YSCALE). Preserve their full-tile results.
                    if needed is None or resampling != "nearest":
                        processing = [(slice(0, height), slice(0, width))]
                    else:
                        processing = _gap_windows(needed, HALO if speckle_filter else 0)
                    info["processing_strategy"] = ("full_target_tile" if needed is None else
                        "adaptive_gap_windows" if resampling == "nearest" else "full_tile_resampling_compatibility")
                    info["processing_windows"] = [[rows.start, rows.stop, cols.start, cols.stop]
                                                   for rows, cols in processing]
                    info["processed_grid_pixels"] = sum((r.stop-r.start)*(c.stop-c.start) for r,c in processing)
                    info["requested_gap_pixels"] = int(np.count_nonzero(needed)) if needed is not None else None
                    if donor:
                        print(f"{season} {source_year}: {info['requested_gap_pixels']} gap pixels; "
                              f"{len(processing)} windows; {info['processed_grid_pixels']} pixels including filter halo "
                              f"(tile has {height * width})", flush=True)
                    values = np.full((2, height, width), np.nan, dtype="float32") if donor else None
                    counts = np.zeros((2, height, width), dtype="uint32") if donor else None
                    accessed = {}
                    info.update(read_burst_windows=0, skipped_burst_windows=0, empty_burst_windows=0,
                                burst_returned_pixels=0, source_read_pixels=0, burst_warp_pixels=0, burst_read_seconds=0.0,
                                composite_seconds=0.0, filter_seconds=0.0)
                    for j, (rows, cols) in enumerate(processing, 1):
                        patch_inside = inside[rows, cols]
                        patch_affine = affine * rasterio.Affine.translation(cols.start, rows.start)
                        patch_shape = patch_inside.shape
                        if needed is None:
                            patch_groups = groups
                        else:
                            # Catalog bounds only prune clearly unrelated bursts; actual
                            # OPERA raster masks still decide valid observations.
                            corners = [patch_affine * xy for xy in ((0, 0), (patch_shape[1], 0),
                                       (patch_shape[1], patch_shape[0]), (0, patch_shape[0]))]
                            xs, ys = zip(*corners)
                            patch_bbox = _search_bbox(_bbox_geometry((min(xs), min(ys), max(xs), max(ys))), crs)
                            patch_groups = _window_groups(groups, patch_bbox)
                            print(f"{season} {source_year}: window {j}/{len(processing)} "
                                  f"{patch_shape[1]}x{patch_shape[0]}; {len(patch_groups)} passes", flush=True)
                        patch_support = None
                        if needed is not None and resampling == "nearest":
                            patch_support = _support_mask(needed[rows, cols], patch_inside,
                                                          HALO if speckle_filter else 0)
                            before = sum(map(len, patch_groups))
                            patch_groups = _support_groups(patch_groups, patch_support, crs, patch_affine)
                            skipped = before - sum(map(len, patch_groups))
                            info["skipped_burst_windows"] += skipped
                            print(f"{season} {source_year}: window {j}/{len(processing)} gap support "
                                  f"retains {sum(map(len, patch_groups))} burst reads; skips {skipped}", flush=True)
                        def reader(record):
                            read_started = time.monotonic()
                            metrics = {}
                            result = read(record, grid=(patch_affine, patch_inside), support=patch_support, metrics=metrics)
                            info["burst_read_seconds"] += time.monotonic() - read_started
                            info["read_burst_windows"] += 1
                            info["burst_returned_pixels"] += result[0].shape[-2] * result[0].shape[-1]
                            info["source_read_pixels"] += metrics.get("source_read_pixels", 0)
                            info["burst_warp_pixels"] += metrics.get("warp_pixels", result[0].shape[-2] * result[0].shape[-1])
                            info["empty_burst_windows"] += int(result[0].size == 0)
                            accessed[record["id"]] = record
                            return result
                        composite_started = time.monotonic()
                        previous_read_time = info["burst_read_seconds"]
                        patch_values, patch_counts = _composite(patch_groups, patch_shape, reader)
                        info["composite_seconds"] += max(0.0, time.monotonic() - composite_started
                            - (info["burst_read_seconds"] - previous_read_time))
                        if speckle_filter and np.any(patch_counts):
                            print(f"{season} {source_year}: filtering completed VV/VH means "
                                  f"({patch_shape[1]}x{patch_shape[0]})", flush=True)
                            filter_started = time.monotonic()
                            patch_values = refined_lee(patch_values,
                                valid_mask=(patch_counts > 0) & patch_inside[None], tile_size=filter_tile_size)
                            info["filter_seconds"] += time.monotonic() - filter_started
                        # Copy only requested donor pixels, never the padded neighborhood
                        # or target-year observations. Overlapping halos are harmless.
                        if needed is None:
                            values, counts = patch_values, patch_counts
                        else:
                            use = needed[rows, cols]
                            values[:, rows, cols][:, use] = patch_values[:, use]
                            counts[:, rows, cols][:, use] = patch_counts[:, use]
                    info["status"] = "processed" if np.any(counts) else "no_usable_pixels"
                    info["read_bursts"] = len(accessed)
                    info["records"] = sorted(accessed.values(), key=lambda rec: rec["id"])
                    info["elapsed_seconds"] = round(time.monotonic() - started, 3)
                    stages = ("search_seconds", "burst_read_seconds", "composite_seconds", "filter_seconds")
                    info["other_seconds"] = max(0.0, info["elapsed_seconds"] - sum(info[k] for k in stages))
                    for key in (*stages, "other_seconds"):
                        info[key] = round(info[key], 3)
                    print(f"{season} {source_year}: timings search={info['search_seconds']:.1f}s, "
                          f"read/warp/retry={info['burst_read_seconds']:.1f}s, "
                          f"accumulate={info['composite_seconds']:.1f}s, filter={info['filter_seconds']:.1f}s, "
                          f"other={info['other_seconds']:.1f}s; "
                          f"native values read={info['source_read_pixels']}, "
                          f"burst warp pixels={info['burst_warp_pixels']}, "
                          f"burst returned pixels={info['burst_returned_pixels']}; "
                          f"empty burst windows={info['empty_burst_windows']}", flush=True)
                    print(f"{season} {source_year}: completed in {info['elapsed_seconds']:.1f}s; "
                          f"{info['read_burst_windows']} burst windows read", flush=True)
                    return values, counts, info

                mean, nobs, info = load_season(year)
                if nobs is None:
                    mean = np.full((2, height, width), np.nan, dtype="float32")
                    nobs = np.zeros((2, height, width), dtype="uint32")
                selected_count = nobs
                donors = []
                if gap_fill:
                    eligible_gaps = np.zeros_like(nobs, dtype=bool)
                    eligibility = {}
                    for k, pol in enumerate(("VV", "VH")):
                        missing = (nobs[k] == 0) & inside
                        if k == 1 and np.array_equal(nobs[0] == 0, nobs[1] == 0):
                            eligible_gaps[k] = eligible_gaps[0]
                            eligibility[pol] = eligibility["VV"].copy()
                        else:
                            eligible_gaps[k], eligibility[pol] = _eligible_gap_mask(missing, gap_min_pixels, gap_core_size)
                    info["gap_eligibility"] = eligibility
                    print(f"{season}: eligible gap pixels "
                          f"{ {p: v['eligible_pixels'] for p,v in eligibility.items()} }; skipped "
                          f"{ {p: v['skipped_pixels'] for p,v in eligibility.items()} }", flush=True)
                    if not np.any(eligible_gaps):
                        print(f"{season}: no qualifying missing patches; skipping all donor searches", flush=True)
                    mean, source_year, source_doy, selected_count, donors = _fill_season(
                        mean, nobs, year, info["nominal_doy"], inside, max_gap_years, load_season, eligible_gaps,
                        gap_min_pixels=gap_min_pixels, gap_core_size=gap_core_size)
                    del eligible_gaps
                    year_pixels = {pol: {str(int(y)): int(np.count_nonzero(source_year[k] == y))
                                        for y in np.unique(source_year[k]) if y}
                                   for k, pol in enumerate(("VV", "VH"))}
                    used_years = {int(y) for values in year_pixels.values() for y in values}
                    info["source_year_pixels"] = year_pixels
                    info["filled_pixels"] = {pol: int(np.count_nonzero((source_year[k] != 0) & (source_year[k] != year)))
                                             for k, pol in enumerate(("VV", "VH"))}
                    for donor_info in donors:
                        donor_info["selected_pixels"] = {pol: year_pixels[pol].get(str(donor_info["year"]), 0)
                                                         for pol in ("VV", "VH")}
                    info["donors"] = donors
                    print(f"{season}: gap-filled pixels {info['filled_pixels']}", flush=True)
                else:
                    source_year = np.where((nobs > 0) & inside[None], year, 0).astype("int16")
                    used_years = {int(year)} if np.any(nobs) else set()
                for source_info in [info, *donors]:
                    if source_info["year"] in used_years:
                        source_intervals.append((source_info["query_start"], source_info["query_end_exclusive"]))
                any_usable |= bool(np.any(selected_count))
                info["remaining_gap_pixels"] = {pol: int(np.count_nonzero(inside & (selected_count[k] == 0)))
                                                 for k, pol in enumerate(("VV", "VH"))}
                metadata["seasons"][season] = info
                if write_meta:
                    packed_meta |= _pack_quality(source_year, selected_count, year) << (20 * s)
                for k, pol in enumerate(("vv", "vh")):
                    b, label = 2 * s + k + 1, f"{pol}mean_{season}"
                    values = np.where((selected_count[k] > 0) & inside, mean[k], NODATA).astype("float32")
                    dst.write(values, b)
                    dst.set_band_description(b, label)
                    dst.update_tags(b, season=season, start=begin.isoformat(),
                                    end_exclusive=finish.isoformat(), units="linear_gamma0_power",
                                    role="seasonal_mean", speckle_filter_applied=str(speckle_filter).lower())
                    metadata["bands"].append(dict(index=b, name=label, units="linear_gamma0_power", role="seasonal_mean"))
                del mean, nobs, selected_count, source_year, values
            dst.update_tags(season_windows=json.dumps(windows))
            if diagnostics:
                diagnostics.write(np.where(inside, packed_meta, int(NODATA)), 1)
                diagnostics.set_band_description(1, "qa")
                diagnostics.update_tags(1, role="source_provenance", units="packed_metadata",
                    speckle_filter_applied="false", encoding=json.dumps(metadata["metadata_raster"]))
                diagnostics.update_tags(season_windows=json.dumps(windows),
                    encoding=json.dumps(metadata["metadata_raster"]),
                    nominal_doy="UTC calendar day containing midpoint of selected source-year season")
        if not any_usable:
            raise RuntimeError("No usable pixels in any season; no final COG written")
        metadata["bands"].sort(key=lambda b: b["index"])
        metadata["source_temporal_extent"] = dict(
            start=min(a for a, b in source_intervals), end_exclusive=max(b for a, b in source_intervals))
        # Nearest overviews preserve packed bit fields; base pixels unchanged.
        rio_copy(temporary, cog, driver="COG", compress="DEFLATE", blocksize=512,
                 overview_resampling="nearest")
        if write_meta:
            rio_copy(meta_temporary, meta_cog, driver="COG", compress="DEFLATE", blocksize=512,
                     overview_resampling="nearest")
        draft_manifest = Path(temp) / manifest.name
        draft_manifest.write_text(json.dumps(metadata, indent=2))
        if write_meta:
            meta_cog.replace(meta_final)
        cog.replace(final)
        draft_manifest.replace(manifest)
    outputs = dict(cog=str(final), manifest=str(manifest),
                   bands=[b["name"] for b in metadata["bands"]])
    if write_meta:
        outputs["meta_cog"] = str(meta_final)
    if write_stac:
        outputs.update(_write_stac_outputs(final, manifest))
    return outputs
