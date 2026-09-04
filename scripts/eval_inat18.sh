#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 /path/to/inaturalist2018 /path/to/checkpoint.pth.tar [100|200]"
    exit 2
fi

DATA_ROOT="$1"
CHECKPOINT="$2"
EPOCHS="${3:-100}"

case "${EPOCHS}" in
    100)
        BATCH_SIZE=512
        LEARNING_RATE=0.2
        ;;
    200)
        BATCH_SIZE=256
        LEARNING_RATE=0.1
        ;;
    *)
        echo "Epoch setting must be 100 or 200."
        exit 2
        ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" main.py \
    --dataset inat \
    --data "${DATA_ROOT}" \
    --arch resnet50 \
    --num_experts 3 \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LEARNING_RATE}" \
    --wd 0.0002 \
    --eta 0.8 \
    --beta 1.0 \
    --cos true \
    --use_norm true \
    --reduce_dimension true \
    --resume "${CHECKPOINT}" \
    --evaluate
