#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${PYTHON:-python}"
nvcc_bin="$(command -v nvcc)"
export CUDA_HOME="$(cd "$(dirname "$nvcc_bin")/.." && pwd)"

echo "Python: $($python_bin -c 'import sys; print(sys.executable)')"
echo "PyTorch/CUDA: $($python_bin -c 'import torch; print(torch.__version__, torch.version.cuda)')"
echo "CUDA_HOME: $CUDA_HOME"
"$nvcc_bin" --version | tail -n 1

cd "$project_root/model/head/localagg_prob"
rm -rf build
find local_aggregate_prob -maxdepth 1 -name '_C*.so' -delete
"$python_bin" setup.py build_ext --inplace
PYTHONPATH="$project_root/model/head/localagg_prob" "$python_bin" -c \
  "import local_aggregate_prob; print(local_aggregate_prob.__file__)"

cd "$project_root/model/encoder/gaussian_encoder/ops"
rm -rf build
find . -maxdepth 1 -name 'deformable_aggregation_ext*.so' -delete
"$python_bin" setup.py build_ext --inplace
PYTHONPATH="$project_root:$project_root/model/head/localagg_prob" "$python_bin" -c \
  "from model.encoder.gaussian_encoder.ops import deformable_aggregation; print(deformable_aggregation.__file__)"
