#!/usr/bin/env bash
set -euo pipefail

# Accuracy-only P0 queue. FLOPs matching is intentionally excluded until the
# unified profiler returns non-zero validated values.
ROOT=/workspace/hma-dcc-ptq
METRICS=$ROOT/outputs/rebuttal/imagenet_resnet18_metrics_seed1005_dev10k
DATA=/workspace/imagenet
WEIGHTS=$ROOT/weights/original_hmaquant/resnet18_imagenet.pth.tar
LOGDIR=$ROOT/outputs/rebuttal/p0_queue
mkdir -p "$LOGDIR"

run_one() {
  local key="$1"
  local slug="$2"
  local out="$ROOT/outputs/rebuttal/imagenet_resnet18_${slug}_recon_seed1005"
  if compgen -G "$out/*.json" >/dev/null 2>&1; then
    echo "[$(date -Is)] SKIP $key (summary exists)"
    return
  fi
  echo "[$(date -Is)] START $key -> $out"
  mkdir -p "$out"
  python "$ROOT/main_hmadcc.py" \
    --arch resnet18 \
    --data-dir "$DATA" --train-dir "$DATA/train" --val-dir "$DATA/val" \
    --calibration-manifest "$METRICS/calibration.files" \
    --validation-manifest "$METRICS/validation.files" \
    --weight-path "$WEIGHTS" \
    --routing-json "$METRICS/summary.json" --routing-key "$key" \
    --iters-sensitive 20000 --iters-robust 5000 \
    --disable-infonce --device cuda --output-dir "$out" \
    > "$out/run.log" 2>&1
  echo "[$(date -Is)] DONE $key"
}

echo "[$(date -Is)] P0 queue waiting for any current block-trace rerun to finish"
while pgrep -af 'imagenet_resnet18_blocktrace_recon_seed1005_v2' | grep -q 'main_hmadcc.py'; do
  sleep 30
done

run_one perturbation_mse_same_k perturbation
run_one hessian_avg_same_k avg_hessian
run_one fusion_h25_p75_same_k fusion_h25_p75
run_one fusion_rank_sum_same_k fusion_rank_sum
run_one random_1005 random1005
run_one random_1029 random1029
run_one random_2023 random2023

echo "[$(date -Is)] P0 queue complete"
