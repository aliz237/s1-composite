"""Noninteractive entry point for one DPS tile/year job."""
import argparse
import json
import logging
from pathlib import Path
import sys

from opera_seasonal_mean import (_output_name, _season_dates, _resolve_cutoff, ARCHIVE_START, parse_bbox as bbox_coords,
                                 resolve_bbox_crs, resolve_crs, seasonal_composites)


def parse_bbox(value):
    try:
        bbox_coords(value, bbox_crs=None)  # CRS-dependent validation follows argument parsing.
        return value  # Preserve the single DPS string; the compositor parses it.
    except ValueError:
        raise argparse.ArgumentTypeError("bbox must be finite xmin,ymin,xmax,ymax in increasing order") from None


def parse_bool(value):
    choices = {"true": True, "false": False, "1": True, "0": False}
    try:
        return choices[value.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError("Use true or false") from None


def parse_windows(value):
    try:
        windows = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("season-windows must be valid JSON") from exc
    if not isinstance(windows, dict):
        raise argparse.ArgumentTypeError("season-windows must be a JSON object")
    return windows or None


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bbox", required=True, type=parse_bbox)
    p.add_argument("--bbox-crs", default="EPSG:4326",
                   help="CRS of bbox coordinates; EPSG/PROJ/WKT or @path/to/crs.wkt")
    p.add_argument("--year", required=True, type=int)
    p.add_argument("--identifier", required=True)
    p.add_argument("--crs", default=None,
                   help="Explicit EPSG/PROJ/WKT or @path/to/crs.wkt; required without a template")
    p.add_argument("--spacing", type=float, default=30)
    p.add_argument("--resampling", choices=("nearest", "bilinear", "average"), default="nearest")
    p.add_argument("--speckle-filter", type=parse_bool, default=True)
    p.add_argument("--write-stac", type=parse_bool, default=True,
                   help="true/false; disable to skip all STAC generation and validation")
    p.add_argument("--season-windows", type=parse_windows, default=None)
    p.add_argument("--gap-fill", type=parse_bool, default=False,
                   help="Fill missing means from nearest complete same-season donor years; record provenance in the optional metadata companion")
    p.add_argument("--max-gap-years", type=int, default=1)
    p.add_argument("--gap-min-pixels", type=int, default=1024,
                   help="Minimum size of an 8-connected missing patch, rechecked before further donors")
    p.add_argument("--gap-core-size", type=int, default=7,
                   help="Odd side length of a fully missing square core required within an eligible patch")
    p.add_argument("--write-meta", type=parse_bool, default=False,
                   help="Write a single-band Int64 companion with exact counts 0-127 and source provenance")
    p.add_argument("--as-of", default=None, help="Exclusive UTC data cutoff; ISO date/time, default now")
    p.add_argument("--input-dir", type=Path, default=Path("input"))
    p.add_argument("--template", type=Path, help="Explicit local reference raster; otherwise use the one staged DPS file")
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--max-results", type=int, default=20000)
    p.add_argument("--max-pixels", type=int, default=25000000)
    p.add_argument("--filter-tile-size", type=int, default=512)
    return p


def staged_template(args):
    if args.template is not None:
        if not args.template.is_file():
            raise FileNotFoundError(args.template)
        return args.template
    files = sorted(p for p in args.input_dir.rglob("*") if p.is_file()) if args.input_dir.exists() else []
    if len(files) > 1:
        raise ValueError("Only one optional template input is supported; multiple files were staged")
    return files[0] if files else None


def _job_crs_value(value):
    if value and value.startswith("@"):
        path = Path(value[1:])
        if not path.is_absolute() and not path.exists():
            path = Path(__file__).resolve().parent / path
        return path
    return value


def run_job(args, maap=None):
    # Validate identities/windows before authenticating or starting expensive work.
    _output_name(args.year, args.identifier)
    _season_dates(args.year, args.season_windows)
    cutoff = _resolve_cutoff(args.as_of or None)
    if not ARCHIVE_START.year <= args.year <= cutoff.year:
        raise ValueError("Target year must be within the OPERA archive through as_of")
    if not 0 <= args.max_gap_years <= 3:
        raise ValueError("max-gap-years must be between 0 and 3 (packed QA range)")
    if args.gap_min_pixels < 1 or args.gap_core_size < 1 or args.gap_core_size % 2 != 1:
        raise ValueError("gap-min-pixels must be positive; gap-core-size must be positive and odd")
    bbox_crs = resolve_bbox_crs(_job_crs_value(args.bbox_crs))
    bbox_coords(args.bbox, bbox_crs)
    template = staged_template(args)
    crs = None
    if template is None:
        crs = resolve_crs(_job_crs_value(args.crs))
    if args.max_results < 1 or args.max_pixels < 1 or args.spacing <= 0:
        raise ValueError("Search/grid limits and spacing must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("sar_s1_dps")
    log.info("Starting tile %s, year %s, bbox %s", args.identifier, args.year, args.bbox)
    if maap is None:
        from maap.maap import MAAP
        maap = MAAP()  # DPS supplies MAAP authentication; no user credentials in args.
    outputs = seasonal_composites(
        maap, bbox=args.bbox, bbox_crs=bbox_crs, year=args.year, identifier=args.identifier,
        outdir=args.output_dir, crs=crs, spacing=args.spacing,
        template=template, resampling=args.resampling,
        speckle_filter=args.speckle_filter,
        season_windows=args.season_windows, max_results=args.max_results,
        max_pixels=args.max_pixels, filter_tile_size=args.filter_tile_size,
        overwrite=False, write_stac=args.write_stac, gap_fill=args.gap_fill,
        max_gap_years=args.max_gap_years, as_of=cutoff, write_meta=args.write_meta,
        gap_min_pixels=args.gap_min_pixels, gap_core_size=args.gap_core_size)
    log.info("Finished: %s", outputs["cog"])
    return outputs


def main(argv=None):
    args = parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.output_dir / "run.log")])
    try:
        outputs = run_job(args)
    except Exception:
        logging.getLogger("sar_s1_dps").exception("Job failed")
        return 1
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
