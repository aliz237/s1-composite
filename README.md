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
| 1 | `bbox` | Required **string**: WGS84 west,south,east,north |
| 2 | `year` | Required acquisition year; suggested 2020 |
| 3 | `tile_identifier` | Required filename identifier: letters/digits/underscore/hyphen |
| 4 | `crs` | Empty; **required without template**, e.g. `@custom_albers.wkt` |
| 5 | `spacing` | 30 metres; ignored with template |
| 6 | `resampling` | `nearest`; also `bilinear` or `average` |
| 7 | `speckle_filter` | `true`; final seasonal mean filtering |
| 8 | `season_windows` | `{}` for the standard three seasons |
| 9 | `write_stac` | `true`; set `false` to skip STAC generation/validation |

`template` is an optional **File** input, separate from positional arguments.
Supply a downloadable URL, not a local ADE path. DPS stages it in `input/`.
Exactly one staged file is supported. A template copies its full projection,
transform, dimensions and extent; bbox masks that grid. Leave CRS blank when
using a template, or pass CRS explicitly for a grid derived from bbox/spacing.

`bbox` is passed as a single string in the DPS request and shell runner. It is
parsed into numbers inside the compositor for geometry calculations, then
formatted as a CMR string. Do not submit a Python tuple as the DPS parameter.

```python
job = maap.submitJob(
    identifier="s1-2020-alaska001",   # DPS job tag
    algo_id="sar-s1-seasonal-rtc",
    version="main",
    queue="maap-dps-worker-16gb",
    bbox="-147.81927653,64.48911791,-145.1944614,65.56965877",
    year="2020",
    tile_identifier="alaska001",     # COG filename identifier
    crs="@custom_albers.wkt",        # Explicit custom projection
    spacing="30",
    resampling="nearest",
    speckle_filter="true",
    season_windows="{}",
    write_stac="false",             # Use "true" to emit item + catalog
    # template="https://.../grid.tif",  # If supplied, its grid takes precedence
)
```

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

The COG contains twelve Float32 bands in this order:

```text
vvmean_frozen,   vhmean_frozen,   vvcount_frozen,   vhcount_frozen
vvmean_summer,   vhmean_summer,   vvcount_summer,   vhcount_summer
vvmean_shoulder, vhmean_shoulder, vvcount_shoulder, vhcount_shoulder
```

Means are linear gamma-zero power. Missing means are **-9999**. Counts are
integer-valued Float32: **0 for no observations inside bbox**, **-9999 outside
bbox**. Count bands are never filtered; nearest overviews keep them integer.
The raster nodata value is -9999 for every band. Internal NaN calculations
exclude invalid values safely; no NaN pixels are written in the COG.

Both ascending and descending acquisitions contribute. A pass is satellite plus
absolute orbit number. Counts measure valid passes per pixel/polarization, not
bursts or equivalent speckle looks. Empty seasons keep their four bands. A job
without observations in any season fails. Existing outputs are protected: use
fresh job directories. Progress goes to stdout, which DPS also preserves.

## Optional STAC

With `write_stac="true"`, output additionally contains:

```text
SAR_S1_2020_alaska001.json
catalog.json
```

The portable catalog links directly to the item beside the COG. The item records
WGS84 geometry, full projection/grid metadata (including custom WKT), twelve
band names, Float32/numeric -9999 nodata, exact seasonal windows and processing
settings. The source manifest remains a linked metadata asset.

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
