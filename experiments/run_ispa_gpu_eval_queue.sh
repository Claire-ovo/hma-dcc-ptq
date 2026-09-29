#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/hma-dcc-ptq
OUT="$ROOT/experiments/ispa_gpu_eval"
PYTHON=/usr/local/miniconda3/envs/py310/bin/python3.10
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"
while pgrep -af 'run_original_protocol_queue.sh|main_hmadcc.py' | grep -qv 'run_ispa_gpu_eval_queue.sh'; do sleep 60; done
$PYTHON "$ROOT/experiments/run_ispa_gpu_eval.py" --arch resnet18 --weight-path "$ROOT/weights/original_hmaquant/resnet18_imagenet.pth.tar" --output-dir "$OUT/resnet18" --label fp32 > "$OUT/resnet18.log" 2>&1
$PYTHON "$ROOT/experiments/run_ispa_gpu_eval.py" --arch mobilenetv2 --weight-path "$ROOT/weights/original_hmaquant/mobilenetv2.pth.tar" --output-dir "$OUT/mobilenetv2" --label fp32 > "$OUT/mobilenetv2.log" 2>&1
echo "GPU_FORWARD_QUEUE_COMPLETE $(date -Is)" > "$OUT/queue.done"
