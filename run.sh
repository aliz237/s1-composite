#!/usr/bin/env -S bash --login
set -euo pipefail

_s1_package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if (( $# < 3 || $# > 9 )); then
  echo 'Usage: run.sh BBOX YEAR IDENTIFIER [CRS_OR_EMPTY_WITH_TEMPLATE] [SPACING] [RESAMPLING] [SPECKLE_FILTER] [SEASON_WINDOWS_JSON] [WRITE_STAC]' >&2
  exit 2
fi

# Preserve DPS's working directory. Do not cd into the source repository.
mkdir -p output
_s1_seasons="${8:-}"
if [[ -z "${_s1_seasons}" ]]; then
  _s1_seasons='{}'
fi
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-west-2}"
export PYTHONUNBUFFERED=1

exec conda run --no-capture-output --name sar_s1_dps \
  python "${_s1_package_dir}/main.py" \
  --bbox="$1" \
  --year="$2" \
  --identifier="$3" \
  --crs="${4:-}" \
  --spacing="${5:-30}" \
  --resampling="${6:-nearest}" \
  --speckle-filter="${7:-true}" \
  --write-stac="${9:-true}" \
  --season-windows="${_s1_seasons}" \
  --input-dir="${PWD}/input" \
  --output-dir="${PWD}/output"
