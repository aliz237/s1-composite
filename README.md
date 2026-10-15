# Sentinel-1 seasonal composites on MAAP DPS

One job processes one tile and year. Native bursts are masked and reprojected,
merged within passes, and averaged with equal-valid-pass weighting. **NumPy
Refined Lee 2.0.0 runs once on each completed seasonal VV/VH mean.** There is no
compiled backend or backend argument. Resampling defaults to nearest neighbour;
output nodata is **-9999**. **CRS is explicit: no default projection.**

## Package files

| File | Purpose |
| --- | --- |
| `environment.yml` | Conda/Python dependencies |
| `build-env.sh` | Install environment and verify imports |
| `algorithm_config.yml` | DPS registration for aliz237/s1-composite, branch main |
| `run.sh` | Ordered DPS arguments to Python CLI |
| `main.py` | Noninteractive execution, staged template discovery and logging |
| `opera_seasonal_mean.py` | Search, masking, reprojection, compositing and final filtering |
| `refined_lee.py` | NumPy-only intensity filter; count bands remain unchanged |
| `stac_output.py` | Optional item/catalog generation and validation |
| `custom_albers.wkt` | Exact nonstandard CRS extracted from the supplied `VV_VH_mean.tif` |

The runner follows the supplied
[MAAP DPS tutorial](https://docs.maap-project.org/en/latest/technical_tutorials/dps_tutorial/dps_tutorial_demo.html).
It preserves DPS's working directory, reads an optional template from `input/`,
and writes deliverables into `output/` for automatic upload.

## Build and test in MAAP

```bash
bash /absolute/path/to/sar_s1_dps/build-env.sh
mkdir -p /absolute/path/to/s1_test/input
cd /absolute/path/to/s1_test
bash /absolute/path/to/sar_s1_dps/run.sh \
  '-147.81927653,64.48911791,-145.1944614,65.56965877' \
  2020 alaska001 '@custom_albers.wkt' 30 nearest true '{}' false
```

This explicitly selects your custom Albers projection and processes **all three
seasons**, with STAC disabled. `@custom_albers.wkt` is resolved in the job
working directory if present, otherwise in the package directory. An absolute
`@/path/to/file.wkt` is also accepted. Literal WKT, PROJ or EPSG can be passed as
one quoted argument. With no CRS, a staged template is required; there is no
fallback to UTM or any other projection.

MAAP supplies authentication. The job refreshes temporary ASF Earthdata S3
credentials while reading raster windows directly from OPERA COGs in us-west-2.
The environment pins MAAP SDK v4.2.0, which supports this compositor's
`searchGranule` and the supplied registration/submission interface. The newer
5.x SDK changes those interfaces.

All available runtime dependencies, including Rasterio/GDAL, NumPy, STAC
libraries and the MAAP SDK's dependencies, are installed from **conda-forge**.
Only `maap-py` is installed by pip, from its tagged v4.2.0 source archive.
The archive installation does not require Git in the environment. The build
checks SDK version/interface and GDAL COG support.

When migrating from the earlier pip-based setup, start with a fresh
`sar_s1_dps` environment instead of updating an environment containing the old
pip-installed scientific packages.

## Register on MAAP

The registration configuration targets
[aliz237/s1-composite](https://github.com/aliz237/s1-composite), branch `main`:

```yaml
repository_url: https://github.com/aliz237/s1-composite.git
algorithm_version: main
build_command: s1-composite/build-env.sh
run_command: s1-composite/run.sh
```

DPS reads the remote repository during registration. Register after the reviewed
files are committed and available on remote `main`. This preparation does not
submit a registration or job.

The container/queue settings are the tutorial's `maap_base:v4.2.0`,
`maap-dps-worker-16gb`, and `20GB` disk. Select resources available to your
account with enough time for a complete tile. Live worker execution has not
been verified here.

Using the SDK installed by the package:

```python
from maap.maap import MAAP

maap = MAAP()
response = maap.register_algorithm_from_yaml_file(
    "/absolute/path/to/s1-composite/algorithm_config.yml"
)
response.raise_for_status()
```

Wait for registration/build completion before submitting jobs. The Register
Algorithm interface can also be populated from the same configuration.

## DPS arguments

The ordered positional fields are:

| Position | Field | Default / meaning |
| --- | --- | --- |
| 1 | `bbox` | Required **string**: xmin,ymin,xmax,ymax in `bbox_crs` |
| 2 | `year` | Required acquisition year; suggested 2020 |
| 3 | `tile_identifier` | Required filename identifier: letters/digits/underscore/hyphen |
| 4 | `crs` | Empty; **required without template**, e.g. `@custom_albers.wkt` |
| 5 | `spacing` | 30 metres; ignored with template |
| 6 | `resampling` | `nearest`; also `bilinear` or `average` |
| 7 | `speckle_filter` | `true`; final seasonal mean filtering |
| 8 | `season_windows` | `{}` for the standard three seasons |
| 9 | `write_stac` | `true`; set `false` to skip STAC generation/validation |
| 10 | `bbox_crs` | `EPSG:4326`; set `@custom_albers.wkt` for Albers bounds |
| 11 | `gap_fill` | `false`; set `true` for nearest-year seasonal filling and provenance |
| 12 | `max_gap_years` | `1`; maximum distance in either direction, 0–3 supported |
| 13 | `as_of` | Empty = UTC job start; optional exclusive ISO data cutoff |
| 14 | `write_meta` | `false`; optional compact COG with exact counts and source provenance |

The Python CLI also accepts a local `--template` or one staged file in `input/`.
A template copies its full projection, transform, dimensions and extent; bbox
masks that grid. The current registration does not declare a DPS File input.
To stage a template through `submitJob`, first add an optional `template` File
input to the registration. Projected bounds matching the output CRS need no
template: their exact upper-left origin is preserved, without snapping to a
zero-based grid. Their width and height must span whole pixels at `spacing`.

`bbox` is passed as a single string in the DPS request and shell runner. It is
parsed into numbers inside the compositor for geometry calculations, then
formatted as a CMR string. Do not submit a Python tuple as the DPS parameter.

```python
job = maap.submitJob(
    identifier="s1-2020-3364",        # DPS job tag
    algo_id="s1-seasonal-rtc",
    version="main",
    queue="maap-dps-worker-16gb",
    bbox="1358522,3123304,1448522,3213304",
    bbox_crs="@custom_albers.wkt",   # Bounds are in Albers metres.
    year="2020",
    tile_identifier="3364",          # COG filename identifier
    crs="@custom_albers.wkt",        # Explicit custom projection
    spacing="30",
    resampling="nearest",
    speckle_filter="true",
    season_windows="{}",
    write_stac="false",             # Use "true" to emit item + catalog
    gap_fill="true",                # Fill missing seasonal means; provenance is packed in QA.
    write_meta="false",             # Set true to retain exact per-pixel counts.
    max_gap_years="1",              # Check both adjacent years.
    as_of="2026-10-14",             # Optional: no data on/after this UTC midnight.
)
```

This produces a 3000 × 3000 grid at 30 m for the supplied tile bounds. For exact
alignment with another raster, obtain coordinates from `src.bounds`, CRS from
`src.crs`, and spacing from `src.res`; retain coordinate precision. This assumes
a north-up square-pixel raster; use a template for a rotated/sheared grid.
Only the search envelope is converted to EPSG:4326. The output and its mask
come from the original bbox in `bbox_crs`, not from that geographic envelope.
Longitude/latitude calls continue to work with `bbox_crs="EPSG:4326"` (default).
If input/output CRSs differ, a bounding rectangle is reprojected and enclosed;
it does not recover an original raster grid from its geographic envelope.

Publish and re-register the changed code/configuration before using the new
`bbox_crs` DPS argument; an already registered build does not update itself.

The `tile_identifier` field avoids a naming conflict with the SDK's
`submitJob(identifier=...)` job-tag parameter. Identifiers produce
`SAR_S1_YEAR_IDENTIFIER.tif`. Keep optional positions in order if calling the
shell runner manually; to omit CRS with a template, pass `''` at position 4.

## Output and season definitions

Defaults stay within the requested year:

| Season | Inclusive dates | Internal [start, end) |
| --- | --- | --- |
| frozen | January 1–March 31 | 01-01, 04-01 |
| summer | June 15–August 31 | 06-15, 09-01 |
| shoulder | September 15–October 31 | 09-15, 11-01 |

Override all three using JSON month-day pairs and exclusive ends:

```json
{"frozen":["01-01","04-01"],"summer":["06-20","09-01"],"shoulder":["09-15","11-01"]}
```

Every successful job writes the COG, `_manifest.json`, and `run.log`:

```text
output/
├── SAR_S1_2020_alaska001.tif
├── SAR_S1_2020_alaska001_manifest.json
└── run.log
```

The main raster always has **six Float32 bands**, whether gap filling is enabled or disabled:

```text
vvmean_frozen, vhmean_frozen
vvmean_summer, vhmean_summer
vvmean_shoulder, vhmean_shoulder
```

Means are linear gamma-zero power. Missing means and pixels outside bbox use
**-9999**. An empty season retains its two mean bands. Refined Lee filters
only means, independently for each source year before selecting filled pixels.

Set **`write_meta="true"`** in DPS (`write_meta=True` in Python) to also write
`SAR_S1_YEAR_IDENTIFIER_meta.tif`. Default is false. This companion shares the
main grid and has **one Int64 band, `qa`**, with **-9999** nodata. It packs all
six source years and exact selected-source counts into **60 bits**. Fields
start at bit 0 in this order: VV frozen, VH frozen, VV summer, VH summer,
VV shoulder, VH shoulder. Each field occupies ten bits:

| Relative bits | Meaning |
|---|---|
| 0–2 | Source code: 0=target, 1=previous year, 2=following year, 3=two years before, 4=two years after, 5=three years before, 6=three years after, 7=missing |
| 3–9 | Exact selected-source valid-pass count, **0–127** |

A missing pixel has count 0 and source code 7. Outside-bbox pixels have the
whole-band nodata value -9999. Counts exceeding 127 fail explicitly when
metadata output is enabled; counts are never clipped. Without the companion,
counts/source years are not retained per pixel; the manifest still records
source acquisitions and summaries. Nearest overviews preserve packed fields.

**Read the QA band as Int64.** Float64 preserves only 53 bits of integer
precision and Float32 only 24, so converting this 60-bit band to either float
can lose information. Do not feed the packed QA number directly into a model.
Use the decoder with the product's target year and recorded seasonal windows:

```python
import json
import rasterio
from opera_seasonal_mean import decode_meta

with rasterio.open("SAR_S1_2020_2710_meta.tif") as src:
    windows = json.loads(src.tags()["season_windows"])
    fields = decode_meta(src.read(1), int(src.tags()["year"]), windows)

source_year = fields["vv_frozen"]["source_year"]
exact_count = fields["vv_frozen"]["source_count"]
original_count = fields["vv_frozen"]["target_count"]
nominal_doy = fields["vv_frozen"]["source_doy"]
is_gap_filled = fields["vv_frozen"]["gap_filled"]
```

Original target-year counts equal selected counts at observed target pixels,
and zero at gap-filled or unresolved pixels. They are reconstructed exactly by
the decoder, avoiding redundant count fields. For model validity use each
mean's nodata mask or the decoded `valid` field. Source years/DOYs decode to
-9999 when missing. Counts decode to zero inside unresolved gaps and -9999
outside bbox. Seasonal windows and packing are recorded in raster tags and the
manifest; nominal DOY is calculated from the selected source year's window.
The optional STAC item links the companion when present.

Both ascending and descending acquisitions contribute. A pass is satellite plus
absolute orbit number. Decoded counts measure valid passes per pixel/polarization, not
bursts or equivalent speckle looks. Empty seasons keep their two mean bands and QA fields. A job
without usable observations in either the target or allowed donor years fails. Existing outputs are protected: use
fresh job directories. Progress goes to stdout, which DPS also preserves.

## Optional nearest-year gap filling

Enable with `gap_fill="true"`; both modes use the same six-band schema. Each
season is processed separately using the same month/day window in donor years.
Only missing means are replaced; valid target-year pixels are preserved.
Candidates are searched at ±1 year, then ±2, up to `max_gap_years`. The nearest
valid year wins per pixel/polarization. Both directions at an equal distance
are evaluated; more valid passes wins, and an exact count tie uses the preceding
year deterministically. Values from different years are not averaged together.
One polarization may select a different year, so provenance is per polarization.
Do not interpret a mixed-year VV/VH ratio as a contemporaneous observation.

Each target/donor season is composited and optionally Refined-Lee filtered
independently before selecting filled pixels. The patchwork is not filtered
again. Donors are searched only while gaps remain, and farther years are skipped
after coverage is complete. Source reads still process the donor season over
the tile; runtime increases on tiles requiring fallback. No extra dependency
is required.

Nominal DOY is the UTC calendar day containing the midpoint of the source
season's [start, exclusive end) interval, calculated in the source year (leap
years included). It is a season label, not a measured or mean acquisition date.
The manifest retains exact source acquisitions, queried intervals, donor
attempts, per-year selected-pixel counts and unresolved gaps.

OPERA's archive starts **April 14, 2016**. Target queries are clipped to that
start and `as_of` (default UTC job start); thus 2016 frozen has no target data
and a current-year season can be partial. Donor seasons must fit entirely
within these bounds. Incomplete recent donors, pre-archive seasons and invalid
non-leap-year February 29 windows are skipped. There is no hardcoded last year:
2026 is the current calendar year, not the mission end. A future target year
or a future cutoff is rejected. Empty donor results leave gaps; HTTP, asset or
authentication errors fail the job instead of being interpreted as no coverage.

Sentinel-1 A/B/C/D OPERA VV/VH products use the same workflow. There is no A/B
filter; pass keys include platform and absolute orbit so satellites are never
merged into one pass accidentally. Original acquisition metadata records the
platform. Real C/D CMR metadata parsing has been checked; this does not replace
radiometric comparison across platforms for a scientific time series.

Temporal borrowing assumes the land surface remains comparable. Disturbances
and year-to-year winter variability can make donors unsuitable; validate using
withheld observed winter patches and the downstream model. The package does
not automatically screen disturbances or harmonize different viewing angles.

Run the offline regression checks in the installed environment with
`python -m unittest discover -s tests -v`. STAC-enabled checks fetch public
schemas. The DPS launcher notebook exposes these optional settings in
`ALGO_PARAMS`; its existing planning/submission workflow is preserved.

## Optional STAC

With `write_stac="true"`, output additionally contains:

```text
SAR_S1_2020_alaska001.json
catalog.json
```

The portable catalog links directly to the item beside the COG. The item records
WGS84 geometry, full projection/grid metadata (including custom WKT), all
band names, Float32/numeric -9999 nodata, exact seasonal windows and processing
settings. The source manifest remains a linked metadata asset.

STAC items link and describe the optional packed Int64 companion and include
the donor-selection policy
and used donor years. Their time extent includes the queried seasons of selected
donors, rather than implying that every pixel is a target-year observation.

With `write_stac="false"`, no STAC writer is imported or run and no catalog/item
is generated or validated. COG and manifest generation are unaffected. The
standalone function has `write_stac=False` by default; DPS registration
keeps STAC enabled by default for compatibility with the previous package.

The writer follows the
[DPS STAC metadata guide](https://docs.maap-project.org/en/develop/technical_tutorials/dps_tutorial/dps_stac_metadata.html).
Relative links survive upload. A source Collection is optional; this package
uses the service-generated collection and its defaults. DPS supplies job
provenance and resolves assets during asynchronous ingestion.

Item IDs hash inputs, grid, processing settings, nodata and source IDs. Different
settings produce distinct IDs; identical inputs in the same algorithm version
can replace the same published record. Use a distinct tile identifier if you
need to retain each repeat run. Schema validation may fetch public schemas over
HTTPS; it occurs only when STAC is enabled and reads raster headers, not pixel
statistics.

## Validation and limits

Local checks pass for exact post-composite filter placement, unfiltered counts,
-9999 nodata, empty seasons, masks before interpolation, custom WKT without an
EPSG code, missing-CRS rejection, bbox string/argument quoting, and STAC on/off.
Enabled catalogs pass schema validation and retain valid links after relocation.
The package contains the same core modules as the standalone outputs.

The default search cap is 20,000 results and grid cap 25 million pixels. Reaching
these guards raises an error. Process mapping tiles; avoid a single circumpolar
array. Advanced `--max-results`, `--max-pixels` and `--filter-tile-size` options
are exposed by the Python CLI, not registered DPS fields.

The conda-first environment was resolved successfully for Linux x86-64 with
strict conda-forge priority (a cross-platform dry run using glibc 2.17 and Linux
5.10 virtual-package values). All MAAP SDK runtime requirements are satisfied
by the selected conda packages. The v4.2.0 source archive also builds the same
SDK wheel as the previous Git-based installation. These checks do not replace
an installation and run on the actual DPS worker.

A live Linux conda build, authenticated OPERA/DPS run and ingestion still need
verification on MAAP. No registration or job was submitted while preparing this package. Refined Lee's MIT attribution is retained in its source.

The NumPy-only filter was also run on the supplied full summer tile (two
3,000 × 3,000 bands). All valid pixels exactly matched the supplied
post-composite Refined Lee product, while invalid pixels became -9999.
Reading, filtering and writing a new COG took 13.4 seconds locally; this is not
a MAAP end-to-end timing or a comparison with the earlier compute-only timing.
