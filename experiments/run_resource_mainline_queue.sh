#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/hma-dcc-ptq
DATA=/workspace/imagenet
PYTHON=/usr/local/miniconda3/envs/py310/bin/python3.10
OUT=$ROOT/outputs/resource_mainline
EVAL=$ROOT/experiments/ispa_gpu_eval
mkdir -p "$OUT" "$EVAL"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

run_arch() {
  local arch="$1" weight="$2"
  local run_dir="$OUT/${arch}_w2a2_seed1005"
  local checkpoint="$run_dir/${arch}_w2a2_seed1005.pt"
  mkdir -p "$run_dir"
  if [[ ! -f "$checkpoint" ]]; then
    "$PYTHON" "$ROOT/main_hmadcc.py" \
      --arch "$arch" --data-dir "$DATA" --weight-path "$weight" \
      --output-dir "$run_dir" --n-bits-w 2 --n-bits-a 2 --batch-size 64 \
      --num-calibration-samples 1024 --seed 1005 --hma-threshold 0.4 \
      --iters-sensitive 20000 --iters-robust 5000 --infonce-lambda 0.1 \
      --infonce-tau 0.1 --hessian-samples 20 --recon-weight 0.01 \
      --recon-lr 4e-5 --b-start 20 --b-end 2 --warmup 0.2 --opt-mode mse \
      --input-prob 0.5 --asym --act-quant --protect-head --workers 4 \
      --device cuda --save-quantized-checkpoint "$checkpoint" \
      > "$run_dir/run.log" 2>&1
  fi
  "$PYTHON" "$ROOT/experiments/run_ispa_gpu_eval.py" \
    --arch "$arch" --weight-path "$weight" --quantized-checkpoint "$checkpoint" \
    --output-dir "$EVAL/${arch}_w2a2" --label w2a2 \
    > "$EVAL/${arch}_w2a2.log" 2>&1
}

run_arch resnet18 "$ROOT/weights/original_hmaquant/resnet18_imagenet.pth.tar"
run_arch mobilenetv2 "$ROOT/weights/original_hmaquant/mobilenetv2.pth.tar"
date -Is > "$EVAL/resource_mainline.done"
