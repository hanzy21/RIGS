#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../model/head/localagg_prob"
rm -rf build
find local_aggregate_prob -maxdepth 1 -name '_C*.so' -delete
python setup.py build_ext --inplace
python -c "import local_aggregate_prob; print(local_aggregate_prob.__file__)"
