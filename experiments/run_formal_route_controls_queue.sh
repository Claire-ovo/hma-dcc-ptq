#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/hma-dcc-ptq
PYTHON=/usr/local/miniconda3/envs/py310/bin/python3.10
DATA=/workspace/imagenet
WEIGHT=$ROOT/weights/original_hmaquant/resnet18_imagenet.pth.tar
BASE=$ROOT/outputs/formal_route_controls/resnet18
mkdir -p "$BASE"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
for policy in hma uniform random inverted; do
  OUT="$BASE/${policy}_seed1005"
  mkdir -p "$OUT"
  if compgen -G "$OUT/*.json" >/dev/null 2>&1; then continue; fi
  "$PYTHON" "$ROOT/main_hmadcc.py" \
    --arch resnet18 --data-dir "$DATA" --weight-path "$WEIGHT" \
    --output-dir "$OUT" --n-bits-w 2 --n-bits-a 2 --batch-size 64 \
    --num-calibration-samples 1024 --seed 1005 --hma-threshold 0.4 \
    --iters-sensitive 20000 --iters-robust 5000 --infonce-lambda 0.1 \
    --infonce-tau 0.1 --hessian-samples 20 --recon-weight 0.01 --recon-lr 4e-5 \
    --b-start 20 --b-end 2 --warmup 0.2 --opt-mode mse --input-prob 0.5 \
    --asym --act-quant --protect-head --workers 4 --device cuda \
    --routing-policy "$policy" > "$OUT/run.log" 2>&1
done
date -Is > "$BASE/queue.done"
