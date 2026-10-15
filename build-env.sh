#!/usr/bin/env -S bash --login
set -euo pipefail

_s1_package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
conda env update --name sar_s1_dps --file "${_s1_package_dir}/environment.yml"

# Fail at build time if the SDK interface or processing dependencies are unavailable.
conda run --no-capture-output --name sar_s1_dps python -c '
from maap.maap import MAAP
from importlib.metadata import version
import rasterio, rio_stac, pystac, numpy, scipy, jsonschema
assert version("maap-py") == "4.2.0", "Unexpected MAAP SDK version"
assert hasattr(MAAP, "searchGranule"), "Incompatible MAAP SDK"
assert hasattr(MAAP, "register_algorithm_from_yaml_file"), "Incompatible MAAP SDK"
with rasterio.Env() as env:
    assert "COG" in env.drivers(), "GDAL COG driver is unavailable"
print("SAR S1 DPS environment ready")
'
