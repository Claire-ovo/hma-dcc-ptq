#!/usr/bin/env bash
set -euo pipefail

# Exact original-paper HMA-DCC reproduction queue.
# This queue is deliberately separate from rebuttal controls (dev10k,
# archived routes, and InfoNCE-off runs).  It must not be used to populate
# controlled ablation tables.

ROOT="${ROOT:-/workspace/hma-dcc-ptq}"
DATA="${DATA:-/workspace/imagenet}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$ROOT/weights/original_hmaquant}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/original_protocol}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

cd "$ROOT"
if [[ -f /root/hma_env.sh ]]; then
  # shellcheck disable=SC1091
  source /root/hma_env.sh
fi

mkdir -p "$OUT_ROOT"

required_files=(
  "$ROOT/main_hmadcc.py"
  "$WEIGHTS_DIR/resnet18_imagenet.pth.tar"
  "$WEIGHTS_DIR/resnet50_imagenet.pth.tar"
  "$WEIGHTS_DIR/mobilenetv2.pth.tar"
  "$WEIGHTS_DIR/regnet_600m.pth.tar"
  "$WEIGHTS_DIR/regnet_3200m.pth.tar"
  "$WEIGHTS_DIR/mnasnet.pth.tar"
)
for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
[[ -d "$DATA/train" && -d "$DATA/val" ]] || {
  echo "ImageNet root must contain train/ and val/: $DATA" >&2
  exit 1
}

read -r train_classes train_images val_classes val_images < <(
  printf '%s %s %s %s\n' \
    "$(find "$DATA/train" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')" \
    "$(find "$DATA/train" -type f | wc -l | tr -d ' ')" \
    "$(find "$DATA/val" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')" \
    "$(find "$DATA/val" -type f | wc -l | tr -d ' ')"
)
[[ "$train_classes" == "1000" && "$train_images" == "1281167" ]] || {
  echo "Unexpected ImageNet train inventory: classes=$train_classes images=$train_images" >&2
  exit 1
}
[[ "$val_classes" == "1000" && "$val_images" == "50000" ]] || {
  echo "Unexpected ImageNet val inventory: classes=$val_classes images=$val_images" >&2
  exit 1
}

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing to run the paper-comparable queue")
print("CUDA:", torch.cuda.get_device_name(0))
PY

write_provenance() {
  local arch="$1" weight="$2" out="$3" json_path
  json_path="$(find "$out" -maxdepth 1 -type f -name '*.json' -print -quit)"
  [[ -n "$json_path" ]] || { echo "No JSON summary produced for $arch" >&2; return 1; }
  python - "$json_path" "$weight" "$out/provenance.json" "$arch" "$DATA" "$train_classes" "$train_images" "$val_classes" "$val_images" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

summary_path, weight_path, provenance_path, arch, data_root = sys.argv[1:6]
inventory = {
    "train_classes": int(sys.argv[6]),
    "train_images": int(sys.argv[7]),
    "val_classes": int(sys.argv[8]),
    "val_images": int(sys.argv[9]),
}

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def command(*args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None

with open(summary_path, encoding="utf-8") as stream:
    summary = json.load(stream)
expected = {
    "protocol": "original_paper_v1",
    "paper_comparable": True,
    "arch": arch,
    "seed": 1005,
    "num_calibration_samples": 1024,
    "calibration_count": 1024,
    "calibration_sampling": "loader_native_seeded",
    "batch_size": 64,
    "device_type": "cuda",
    "calibration_manifest": None,
    "validation_manifest": None,
    "validation_count": 50000,
    "infonce_enabled": True,
    "infonce_lambda": 0.1,
    "infonce_tau": 0.1,
    "protect_head": True,
    "weight_bits": 2,
    "activation_bits": 2,
    "asymmetric_quantization": True,
    "activation_quantization": True,
    "routing_json": None,
    "routing_key": None,
    "hma_threshold": 0.4,
    "iters_sensitive": 20000,
    "iters_robust": 5000,
    "hessian_samples": 20,
    "recon_weight": 0.01,
    "recon_lr": 4e-5,
    "b_range": [20.0, 2.0],
    "warmup": 0.2,
    "opt_mode": "mse",
    "input_prob": 0.5,
}
for key, value in expected.items():
    if summary.get(key) != value:
        raise SystemExit(f"{summary_path}: protocol mismatch for {key}: {summary.get(key)!r} != {value!r}")
if summary.get("calibration_paths") is not None:
    raise SystemExit(
        f"{summary_path}: loader-native calibration must record calibration_paths=null, "
        "not a manifest-derived path list"
    )
validation_paths = summary.get("validation_paths") or []
if len(validation_paths) != 50000:
    raise SystemExit(f"{summary_path}: expected 50000 validation paths, got {len(validation_paths)}")
provenance = {
    "protocol": "original_paper_v1",
    "architecture": arch,
    "checkpoint": os.path.abspath(weight_path),
    "checkpoint_sha256": sha256(weight_path),
    "data_root": os.path.abspath(data_root),
    "inventory": inventory,
    "evaluation": "full_imagenet_val_50000",
    "calibration": "1024 loader-native train images; phase-2 objectives use no class labels; no calibration manifest",
    "required_settings": {
        "weight_bits": 2,
        "activation_bits": 2,
        "batch_size": 64,
        "seed": 1005,
        "calibration_manifest": None,
        "calibration_sampling": "loader_native_seeded",
        "calibration_paths": None,
        "calibration_count": 1024,
        "hma_threshold": 0.4,
        "iters_sensitive": 20000,
        "iters_robust": 5000,
        "infonce_enabled": True,
        "infonce_lambda": 0.1,
        "infonce_tau": 0.1,
        "hessian_samples": 20,
        "recon_weight": 0.01,
        "recon_lr": 4e-5,
        "b_range": [20, 2],
        "warmup": 0.2,
        "opt_mode": "mse",
        "input_prob": 0.5,
        "asymmetric_quantization": True,
        "activation_quantization": True,
        "protect_head": True,
        "routing_source": "computed_in_driver",
        "routing_json": None,
        "routing_key": None,
        "validation_manifest": None,
        "validation_paths": 50000,
        "validation_count": 50000,
    },
    "summary_path": os.path.abspath(summary_path),
    "summary_sha256": sha256(summary_path),
    "audited_metrics": {
        key: summary.get(key)
        for key in (
            "fp32_top1", "quantized_top1", "accuracy_drop",
            "calibration_flops", "calibration_flops_scope",
            "peak_memory_mb", "wall_time_seconds", "stage_metrics",
        )
    },
    "summary_protocol_fields": {
        key: summary.get(key)
        for key in (
            "protocol", "paper_comparable", "arch", "seed", "num_calibration_samples", "batch_size", "device_type",
            "calibration_manifest", "calibration_sampling", "calibration_paths", "calibration_count",
            "validation_manifest", "validation_count",
            "infonce_enabled", "infonce_lambda", "infonce_tau",
            "protect_head", "weight_bits", "activation_bits",
            "asymmetric_quantization", "activation_quantization",
            "routing_json", "routing_key", "hma_threshold",
            "iters_sensitive", "iters_robust", "hessian_samples",
            "recon_weight", "recon_lr", "b_range", "warmup", "opt_mode", "input_prob",
        )
    },
    "git_commit": command("git", "rev-parse", "HEAD"),
    "torch_version": command(sys.executable, "-c", "import torch; print(torch.__version__)"),
    "cuda_version": command(sys.executable, "-c", "import torch; print(torch.version.cuda)"),
}
with open(provenance_path, "w", encoding="utf-8") as stream:
    json.dump(provenance, stream, indent=2)
    stream.write("\n")
PY
}

run_one() {
  local arch="$1" weight_file="$2"
  local weight="$WEIGHTS_DIR/$weight_file"
  local out="$OUT_ROOT/${arch}_w2a2_seed1005"
  local log="$out/run.log"
  mkdir -p "$out"
  if [[ -f "$out/provenance.json" ]] && compgen -G "$out/*.json" >/dev/null 2>&1; then
    echo "[$(date -Is)] SKIP $arch (audited output exists)"
    return 0
  fi
  echo "[$(date -Is)] START $arch"
  # No calibration/validation manifest, routing JSON, or --disable-infonce:
  # the original paper uses loader-native train calibration, in-driver HMA,
  # InfoNCE on, and the complete 50k validation set.
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONUNBUFFERED=1 python "$ROOT/main_hmadcc.py" \
    --arch "$arch" \
    --data-dir "$DATA" \
    --weight-path "$weight" \
    --output-dir "$out" \
    --n-bits-w 2 --n-bits-a 2 \
    --batch-size 64 --num-calibration-samples 1024 --seed 1005 \
    --hma-threshold 0.4 --iters-sensitive 20000 --iters-robust 5000 \
    --infonce-lambda 0.1 --infonce-tau 0.1 \
    --hessian-samples 20 --recon-weight 0.01 --recon-lr 4e-5 \
    --b-start 20 --b-end 2 --warmup 0.2 --opt-mode mse \
    --input-prob 0.5 --asym --act-quant --protect-head \
    --workers 4 --device cuda \
    2>&1 | tee "$log"
  write_provenance "$arch" "$weight" "$out"
  echo "[$(date -Is)] DONE $arch"
}

run_one resnet18 resnet18_imagenet.pth.tar
run_one resnet50 resnet50_imagenet.pth.tar
run_one mobilenetv2 mobilenetv2.pth.tar
run_one regnetx_600mf regnet_600m.pth.tar
run_one regnetx_3200m regnet_3200m.pth.tar
run_one mnasnet mnasnet.pth.tar

echo "[$(date -Is)] ORIGINAL PROTOCOL QUEUE COMPLETE"
