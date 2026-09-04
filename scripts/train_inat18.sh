#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/inaturalist2018"
    exit 2
fi

DATA_ROOT="$1"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" main.py \
    --dataset inat \
    --data "${DATA_ROOT}" \
    --arch resnet50 \
    --num_experts 3 \
    --epochs 100 \
    --batch-size 512 \
    --lr 0.2 \
    --wd 0.0002 \
    --eta 0.8 \
    --beta 1.0 \
    --cos true \
    --use_norm true \
    --reduce_dimension true

