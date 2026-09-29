#!/usr/bin/env bash
# DEPRECATED: dev10k/archived-route control queue. Retained for audit only;
# do not launch for current paper experiments.
set -u

ROOT=/workspace/hma-dcc-ptq
PYTHON=/usr/local/miniconda3/envs/py310/bin/python3.10
DATA=/workspace/imagenet
R18_METRICS=$ROOT/outputs/rebuttal/imagenet_resnet18_metrics_seed1005_dev10k
MNV2_METRICS=$ROOT/outputs/rebuttal/imagenet_mobilenetv2_metrics_seed1005_dev10k
R18_WEIGHTS=$ROOT/weights/original_hmaquant/resnet18_imagenet.pth.tar
# Use the BRECQ release checkpoint used by the original manuscript.  The
# converted torchvision checkpoint remains archived but is not paper-comparable.
MNV2_WEIGHTS=$ROOT/weights/original_hmaquant/mobilenetv2.pth.tar
LOGDIR=$ROOT/outputs/rebuttal/extended_p0_queue
mkdir -p "$LOGDIR"

run_route() {
  local arch="$1" key="$2" slug="$3" weights="$4" metrics="$5"
  local out="$ROOT/outputs/rebuttal/${arch}_brecq_${slug}_recon_seed1005"
  if compgen -G "$out/*.json" >/dev/null 2>&1; then
    echo "[$(date -Is)] SKIP $arch $key (summary exists)" >> "$LOGDIR/queue.log"
    return 0
  fi
  mkdir -p "$out"
  echo "[$(date -Is)] START $arch $key" | tee -a "$LOGDIR/queue.log"
  "$PYTHON" "$ROOT/main_hmadcc.py" \
    --arch "$arch" --data-dir "$DATA" --train-dir "$DATA/train" --val-dir "$DATA/val" \
    --calibration-manifest "$metrics/calibration.files" \
    --validation-manifest "$metrics/validation.files" \
    --weight-path "$weights" --routing-json "$metrics/summary.json" --routing-key "$key" \
    --iters-sensitive 20000 --iters-robust 5000 --disable-infonce \
    --device cuda --output-dir "$out" > "$out/run.log" 2>&1
  rc=$?
  echo "[$(date -Is)] END $arch $key rc=$rc" | tee -a "$LOGDIR/queue.log"
  return 0
}

run_lambda() {
  local lambda="$1" slug="infonce_lambda_${lambda//./p}"
  local out="$ROOT/outputs/rebuttal/imagenet_resnet18_${slug}_seed1005"
  if compgen -G "$out/*.json" >/dev/null 2>&1; then
    echo "[$(date -Is)] SKIP lambda=$lambda (summary exists)" >> "$LOGDIR/queue.log"
    return 0
  fi
  mkdir -p "$out"
  echo "[$(date -Is)] START lambda=$lambda" | tee -a "$LOGDIR/queue.log"
  if [[ "$lambda" == "0" ]]; then
    disable=(--disable-infonce)
  else
    disable=()
  fi
  "$PYTHON" "$ROOT/main_hmadcc.py" \
    --arch resnet18 --data-dir "$DATA" --train-dir "$DATA/train" --val-dir "$DATA/val" \
    --calibration-manifest "$R18_METRICS/calibration.files" \
    --validation-manifest "$R18_METRICS/validation.files" \
    --weight-path "$R18_WEIGHTS" --routing-json "$R18_METRICS/summary.json" \
    --routing-key hma_same_k --iters-sensitive 20000 --iters-robust 5000 \
    --infonce-lambda "$lambda" "${disable[@]}" --device cuda --output-dir "$out" \
    > "$out/run.log" 2>&1
  rc=$?
  echo "[$(date -Is)] END lambda=$lambda rc=$rc" | tee -a "$LOGDIR/queue.log"
  return 0
}

echo "[$(date -Is)] waiting for first P0 queue" | tee -a "$LOGDIR/queue.log"
while ! grep -q 'P0 queue complete' "$ROOT/outputs/rebuttal/p0_queue.log" 2>/dev/null; do sleep 60; done
echo "[$(date -Is)] first P0 queue ended" | tee -a "$LOGDIR/queue.log"

run_route mobilenetv2 hma_same_k hma "$MNV2_WEIGHTS" "$MNV2_METRICS"
run_route mobilenetv2 inverted_same_k inverted "$MNV2_WEIGHTS" "$MNV2_METRICS"
run_route mobilenetv2 perturbation_mse_same_k perturbation "$MNV2_WEIGHTS" "$MNV2_METRICS"
run_route mobilenetv2 hessian_avg_same_k avg_hessian "$MNV2_WEIGHTS" "$MNV2_METRICS"
run_route mobilenetv2 fusion_mean_same_k fusion_mean "$MNV2_WEIGHTS" "$MNV2_METRICS"

run_lambda 0
run_lambda 0.01
run_lambda 0.05
run_lambda 0.1
run_lambda 0.2

echo "[$(date -Is)] extended P0 queue complete" | tee -a "$LOGDIR/queue.log"
