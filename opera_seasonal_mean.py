"""Three-season OPERA VV/VH composites on MAAP (AWS us-west-2).

Native-grid burst masking -> user-selected reprojection -> overlap mean within
pass -> equal-pass seasonal mean -> optional NumPy Refined Lee on the mean.
Write all means and counts in SAR_S1_{year}_{identifier}.tif, a 12-band COG.
Requires maap-py, numpy, rasterio, boto3 and the companion refined_lee.py.
Season intervals are calendar dates [start, end), within the requested year.
"""

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import tempfile
import time
from contextlib import ExitStack

import numpy as np
from refined_lee import NODATA, __version__ as FILTER_VERSION, refined_lee

SEASONS = ("frozen", "summer", "shoulder")
DEFAULT_WINDOWS = {
    "frozen": ("01-01", "04-01"),
    "summer": ("06-15", "09-01"),
    "shoulder": ("09-15", "11-01"),
}


def parse_bbox(value):
    """Accept a DPS comma-separated string or a four-coordinate sequence."""
    try:
        coords = value.split(",") if isinstance(value, str) else value
        west, south, east, north = (float(x) for x in coords)
    except (ValueError, TypeError):
        raise ValueError("bbox must be west,south,east,north (four WGS84 coordinates)") from None
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Use finite WGS84 bounds without dateline crossing")
    return west, south, east, north


def resolve_crs(value):
    """Explicit EPSG/PROJ/WKT, a pathlib.Path, or @path/to/crs.wkt; no fallback."""
    from rasterio.crs import CRS
    if value is None or isinstance(value, str) and not value.strip():
        raise ValueError("Provide an explicit crs or a reference template raster")
    if isinstance(value, Path) or isinstance(value, str) and value.startswith("@"):
        path = value if isinstance(value, Path) else Path(value[1:])
        value = path.read_text().strip()
    crs = CRS.from_user_input(value)
    if not crs.is_projected or crs.linear_units != "metre":
        raise ValueError("Use an explicit projected CRS with metre units")
    return crs


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
        if not all(isinstance(s, str) and re.fullmatch(r"\d{2}-\d{2}", s) for s in (first, last)):
            raise ValueError("Season endpoints must be month-day strings, e.g. '06-15'")
        begin, finish = _utc(f"{year}-{first}"), _utc(f"{year}-{last}")
        if begin >= finish:
            raise ValueError("Each season must have start < exclusive end within the same year")
        dates[season] = (begin, finish)
    return dates


def _output_name(year, identifier):
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", identifier):
        raise ValueError("identifier must use letters, numbers, underscores or hyphens")
    return f"SAR_S1_{year}_{identifier}.tif"


def _band_names(season):
    return (f"vvmean_{season}", f"vhmean_{season}",
            f"vvcount_{season}", f"vhcount_{season}")


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
                      relative_orbit=attrs.get("PATH_NUMBER"))
        if key not in selected or (version, produced) > selected[key][0]:
            selected[key] = ((version, produced), record)
    groups = defaultdict(list)
    for _, record in selected.values():
        groups[(record["platform"], record["orbit"])].append(record)
    return [sorted(group, key=lambda r: r["id"]) for _, group in sorted(groups.items())]


def _composite(groups, shape, reader):
    """Accumulate one pass at a time; reader returns two bands and a valid mask."""
    total = np.zeros((2, *shape), dtype="float64")
    nobs = np.zeros((2, *shape), dtype="uint32")
    for i, group in enumerate(groups, 1):
        pass_sum = np.zeros_like(total)
        pass_count = np.zeros_like(nobs)
        for record in group:
            bands, clear = reader(record)
            valid = clear[None, :, :] & np.isfinite(bands)
            pass_sum += np.where(valid, bands, 0)
            pass_count += valid
        covered = pass_count > 0
        np.divide(pass_sum, pass_count, out=pass_sum, where=covered)
        total += pass_sum
        nobs += covered
        print(f"Pass {i}/{len(groups)}: {len(group)} bursts", flush=True)
    mean = np.full(total.shape, np.nan, dtype="float32")
    np.divide(total, nobs, out=mean, where=nobs > 0)
    return mean, nobs


def _read_burst(record, *, crs, affine, width, height, inside, resampling):
    """Read/mask a native source window, then warp to the target grid.

    The caller supplies any required Rasterio/AWS environment. Extra source
    pixels support the interpolation kernels. Bursts are never speckle-filtered.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject, transform_bounds
    from rasterio.windows import Window, from_bounds

    destination = np.full((2, height, width), np.nan, dtype="float32")
    clear_target = np.zeros((height, width), dtype=bool)
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(record["assets"][p]))
                   for p in ("VV", "VH", "mask")]
        vv, vh, mask = sources
        if any(s.crs != vv.crs or s.transform != vv.transform or s.shape != vv.shape
               for s in (vh, mask)):
            raise ValueError(f"Burst bands/mask grids differ: {record['id']}")
        if vv.crs is None or vv.transform.b != 0 or vv.transform.d != 0 or vv.transform.a <= 0 or vv.transform.e >= 0:
            raise ValueError("Expected a north-up georeferenced OPERA source grid")
        if any(v != 1 for s in (vv, vh) for v in s.scales) or any(v != 0 for s in (vv, vh) for v in s.offsets):
            raise ValueError("Expected OPERA linear power without scale/offset encoding")
        corners = [affine * p for p in ((0, 0), (width, 0), (width, height), (0, height))]
        xs, ys = zip(*corners)
        source_bounds = transform_bounds(crs, vv.crs, min(xs), min(ys), max(xs), max(ys), densify_pts=41)
        proposed = from_bounds(*source_bounds, transform=vv.transform)
        margin = 3  # Interpolation support only.
        col0 = max(0, math.floor(proposed.col_off) - margin)
        row0 = max(0, math.floor(proposed.row_off) - margin)
        col1 = min(vv.width, math.ceil(proposed.col_off + proposed.width) + margin)
        row1 = min(vv.height, math.ceil(proposed.row_off + proposed.height) + margin)
        if col0 >= col1 or row0 >= row1:
            return destination, clear_target
        window = Window(col0, row0, col1 - col0, row1 - row0)
        native_transform = vv.window_transform(window)
        flags = mask.read(1, window=window, masked=True)
        clear = ~np.ma.getmaskarray(flags) & (flags.data == 0)
        bands = np.stack([s.read(1, window=window, masked=True).astype("float32").filled(np.nan)
                          for s in (vv, vh)])
        valid = clear[None] & np.isfinite(bands)
        bands = np.where(valid, bands, np.nan)
        warp = dict(src_transform=native_transform, src_crs=vv.crs,
                    dst_transform=affine, dst_crs=crs)
        # NaN excludes source layover/shadow/nodata BEFORE interpolation.
        for b in range(2):
            reproject(bands[b], destination[b], **warp, src_nodata=np.nan,
                      dst_nodata=np.nan, resampling=resampling)
        # Keep the target's nearest source mask category as an additional guard.
        warped_mask = np.full((height, width), 255, dtype="uint8")
        reproject(np.where(clear, 0, 255).astype("uint8"), warped_mask,
                  **warp, src_nodata=255, dst_nodata=255, resampling=Resampling.nearest)
        clear_target = (warped_mask == 0) & inside
        destination[:, ~clear_target] = np.nan
    return destination, clear_target


def seasonal_composites(maap, bbox, year, identifier, *, outdir="rtc_seasonal",
                        resampling="nearest", speckle_filter=True,
                        season_windows=None, crs=None, spacing=30,
                        template=None, max_results=20000, max_pixels=25000000,
                        filter_tile_size=512, overwrite=False, write_stac=False):
    """Write three seasonal means/counts to one SAR_S1_year_identifier COG.

    Default windows include Jan 1–Mar 31, Jun 15–Aug 31, Sep 15–Oct 31 of
    year, including leap-year days. Set season_windows using month-day pairs
    with EXCLUSIVE ends. resampling: 'nearest', 'bilinear', or 'average'.
    All 12 bands are Float32; counts are integer-valued, with 0 meaning no
    valid passes. Means are linear gamma-zero power; missing output means are
    -9999. Counts are 0 for no observations inside bbox; outside bbox is -9999.
    bbox accepts a comma-separated string (DPS) or a coordinate sequence.
    Optional template uses its full grid with the bbox mask. Run per tile.
    write_stac=True adds an item and catalog; requires the optional STAC helper
    and dependencies. Default False writes only the COG and source manifest.
    """
    import boto3
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import geometry_mask
    from rasterio.session import AWSSession
    from rasterio.shutil import copy as rio_copy
    from rasterio.transform import from_origin
    from rasterio.warp import transform_geom

    dates = _season_dates(year, season_windows)
    filename = _output_name(year, identifier)
    bbox = parse_bbox(bbox)
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
    if not isinstance(write_stac, bool):
        raise ValueError("write_stac must be True or False")
    if isinstance(filter_tile_size, bool) or not isinstance(filter_tile_size, (int, np.integer)) or filter_tile_size < 1:
        raise ValueError("filter_tile_size must be a positive integer")
    if speckle_filter:
        print("Refined Lee: NumPy, once per seasonal mean after compositing", flush=True)
    folder = Path(outdir)
    final = folder / filename
    manifest = folder / (final.stem + "_manifest.json")
    if not overwrite and (final.exists() or manifest.exists()):
        raise FileExistsError(f"Output exists: {final} or {manifest}; use overwrite=True to replace")
    west, south, east, north = map(float, bbox)
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Use a non-dateline-crossing west,south,east,north box.")
    if spacing <= 0:
        raise ValueError("spacing must be positive.")

    # Densify the geographic box before projecting its curved boundaries.
    ring = []
    corners = [(west, south), (east, south), (east, north), (west, north)]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        ring.extend([(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
                     for t in np.linspace(0, 1, 101, endpoint=False)])
    ring.append(ring[0])
    if template:
        with rasterio.open(template) as ref:
            crs, affine, width, height = ref.crs, ref.transform, ref.width, ref.height
        if crs is None:
            raise ValueError("Template must have a CRS")
    else:
        geom = transform_geom("EPSG:4326", crs, {"type": "Polygon", "coordinates": [ring]})
        xs, ys = zip(*geom["coordinates"][0])
        left = math.floor(min(xs) / spacing) * spacing
        top = math.ceil(max(ys) / spacing) * spacing
        width = math.ceil((max(xs) - left) / spacing)
        height = math.ceil((top - min(ys)) / spacing)
        affine = from_origin(left, top, spacing, spacing)
    if width * height > max_pixels:
        raise ValueError("Grid exceeds max_pixels; process smaller tiles.")
    geom = transform_geom("EPSG:4326", crs, {"type": "Polygon", "coordinates": [ring]})
    inside = geometry_mask([geom], (height, width), affine, invert=True)
    if not np.any(inside):
        raise ValueError("Bounding box does not intersect the template grid.")

    endpoint = "https://cumulus.asf.earthdatacloud.nasa.gov/s3credentials"
    aws, refreshed_at = None, 0

    def read(record):
        nonlocal aws, refreshed_at
        if aws is None or time.monotonic() - refreshed_at > 45 * 60:
            credentials = maap.aws.earthdata_s3_credentials(endpoint)
            aws = AWSSession(boto3.Session(
                aws_access_key_id=credentials["accessKeyId"],
                aws_secret_access_key=credentials["secretAccessKey"],
                aws_session_token=credentials["sessionToken"], region_name="us-west-2"))
            refreshed_at = time.monotonic()
        with rasterio.Env(aws, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                          GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2"):
            return _read_burst(record, crs=crs, affine=affine, width=width,
                               height=height, inside=inside, resampling=allowed[resampling])

    folder.mkdir(parents=True, exist_ok=True)
    metadata = dict(year=int(year), identifier=identifier, bbox=list(bbox),
                    crs=crs.to_wkt(), transform=list(affine)[:6], width=width, height=height,
                    template=str(template) if template else None, mask_values_kept=[0],
                    resampling=resampling, mask_resampling="nearest",
                    temporal_weighting="equal_valid_satellite_orbit",
                    speckle_filter="Refined_Lee" if speckle_filter else "none",
                    speckle_filter_version=FILTER_VERSION if speckle_filter else None,
                    nodata=NODATA, counts_zero="no_valid_passes_inside_bbox",
                    filter_stage="after_seasonal_compositing" if speckle_filter else "none",
                    bands=[], seasons={})
    any_usable = False
    with tempfile.TemporaryDirectory(prefix=final.stem + "_", dir=folder) as temp:
        temporary, cog = Path(temp) / "stack.tif", Path(temp) / filename
        with rasterio.open(temporary, "w", driver="GTiff", width=width, height=height,
                           count=12, dtype="float32", crs=crs, transform=affine,
                           nodata=NODATA, tiled=True, compress="deflate") as dst:
            dst.update_tags(year=year, identifier=identifier, resampling=resampling,
                            mask_resampling="nearest", mask="OPERA_mask_equals_0",
                            temporal_weighting="equal_valid_pass", counts_zero="no_valid_passes_inside_bbox",
                            speckle_filter=metadata["speckle_filter"],
                            filter_stage=metadata["filter_stage"])
            for s, season in enumerate(SEASONS):
                begin, finish = dates[season]
                results = maap.searchGranule(
                    cmr_host="cmr.earthdata.nasa.gov", short_name="OPERA_L2_RTC-S1_V1",
                    temporal=f"{begin.strftime('%Y-%m-%dT%H:%M:%SZ')},{finish.strftime('%Y-%m-%dT%H:%M:%SZ')}",
                    bounding_box=",".join(map(str, bbox)), limit=max_results)
                if len(results) >= max_results:
                    raise RuntimeError(f"{season} search reached max_results; increase it or use smaller tiles")
                groups = _records(results, begin, finish)
                print(f"{season}: {len(results)} search results; {sum(map(len, groups))} retained bursts; "
                      f"{len(groups)} satellite passes.", flush=True)
                mean, nobs = _composite(groups, (height, width), read)
                any_usable |= bool(np.any(nobs))
                if np.max(nobs) > 2**24:
                    raise ValueError("Count exceeds the exact integer range of Float32")
                if not np.any(nobs):
                    print(f"{season}: no valid observations; writing -9999 means and zero counts", flush=True)
                elif speckle_filter:
                    print(f"{season}: filtering completed VV/VH means", flush=True)
                    mean = refined_lee(mean, valid_mask=(nobs > 0) & inside[None],
                                       tile_size=filter_tile_size)
                labels = _band_names(season)
                for k, label in enumerate(labels):
                    b = 4 * s + k + 1
                    values = (np.where((nobs[k] > 0) & inside, mean[k], NODATA) if k < 2
                              else np.where(inside, nobs[k - 2], NODATA).astype("float32"))
                    unit = "linear_gamma0_power" if k < 2 else "valid_pass_count"
                    dst.write(values, b)
                    dst.set_band_description(b, label)
                    dst.update_tags(b, season=season, start=begin.isoformat(),
                                    end_exclusive=finish.isoformat(), units=unit,
                                    speckle_filter_applied="true" if speckle_filter and k < 2 else "false")
                    metadata["bands"].append(dict(index=b, name=label, units=unit))
                metadata["seasons"][season] = dict(
                    start=begin.isoformat(), end_exclusive=finish.isoformat(),
                    search_results=len(results), retained_bursts=sum(map(len, groups)),
                    passes=len(groups), records=[r for group in groups for r in group])
                del mean, nobs, groups, results, values
        if not any_usable:
            raise RuntimeError("No usable pixels in any season; no final COG written")
        # Nearest overviews keep count bands integer-valued; base pixels unchanged.
        rio_copy(temporary, cog, driver="COG", compress="DEFLATE", blocksize=512,
                 overview_resampling="nearest")
        draft_manifest = Path(temp) / manifest.name
        draft_manifest.write_text(json.dumps(metadata, indent=2))
        cog.replace(final)
        draft_manifest.replace(manifest)
    outputs = dict(cog=str(final), manifest=str(manifest),
                   bands=[b["name"] for b in metadata["bands"]])
    if write_stac:
        outputs.update(_write_stac_outputs(final, manifest))
    return outputs
