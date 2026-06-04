#!/usr/bin/env bash
set -euo pipefail

# Example W2A2 ImageNet reproduction command for MobileNetV2.
# Override these variables from the shell if needed.
DATA_DIR="${DATA_DIR:-/path/to/imagenet}"
WEIGHT_PATH="${WEIGHT_PATH:-/path/to/pretrained_weights/mobilenetv2.pth}"
OUT_DIR="${OUT_DIR:-outputs/mobilenetv2_w2a2}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ "${DATA_DIR}" == /path/to/* || "${WEIGHT_PATH}" == /path/to/* ]]; then
  echo "Please set DATA_DIR and WEIGHT_PATH before running this script."
  exit 1
fi

python main_hmadcc.py \
  --arch mobilenetv2 \
  --data-dir "${DATA_DIR}" \
  --weight-path "${WEIGHT_PATH}" \
  --output-dir "${OUT_DIR}" \
  --n-bits-w 2 \
  --n-bits-a 2 \
  --batch-size 64 \
  --num-calibration-samples 1024 \
  --seed 1005 \
  --hma-threshold 0.4 \
  --iters-sensitive 20000 \
  --iters-robust 5000 \
  --infonce-lambda 0.1 \
  --workers 4 \
  --device cuda \
  ${EXTRA_ARGS}
