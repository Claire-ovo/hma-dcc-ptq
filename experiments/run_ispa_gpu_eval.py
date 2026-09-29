#!/usr/bin/env python3
"""Reproducible GPU forward benchmark for the ISPA resource-evidence protocol.

This intentionally benchmarks only supplied model checkpoints.  It never labels
an FP32 checkpoint as W2A2; a missing quantized checkpoint is recorded as
``not_available`` so the resulting artifact cannot be mistaken for a paper claim.
"""
import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main_hmadcc import build_model, load_pretrained_weights, seed_all
from quant.quant_model import QuantModel
from quant.set_weight_quantize_params import set_weight_quantize_params


def move_quantizer_tensors(model, device):
    """Move quantizer attributes that predate registered-buffer checkpointing."""
    for module in model.modules():
        for name in ("delta", "zero_point", "running_min", "running_max"):
            value = getattr(module, name, None)
            if (
                isinstance(value, torch.Tensor)
                and value.device != device
                and name not in module._parameters
                and name not in module._buffers
            ):
                setattr(module, name, value.to(device))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def measure(model, device, batch, warmup, iters, repeats):
    x = torch.randn(batch, 3, 224, 224, device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
    torch.cuda.synchronize(device)
    all_ms = []
    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            for _ in range(iters):
                model(x)
        end.record()
        torch.cuda.synchronize(device)
        all_ms.append(start.elapsed_time(end) / iters)
    return {
        "latency_ms_mean": statistics.mean(all_ms),
        "latency_ms_std": statistics.stdev(all_ms) if len(all_ms) > 1 else 0.0,
        "latency_ms_p50": percentile(all_ms, 0.50),
        "latency_ms_p95": percentile(all_ms, 0.95),
        "images_per_second": batch * 1000.0 / statistics.mean(all_ms),
        "peak_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_memory_reserved_mb": torch.cuda.max_memory_reserved(device) / 2**20,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True)
    p.add_argument("--weight-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--label", default="fp32")
    p.add_argument("--quantized-checkpoint")
    p.add_argument("--seed", type=int, default=1005)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P0 GPU benchmark requires CUDA")
    seed_all(args.seed)
    device = torch.device("cuda")
    weight = Path(args.weight_path).resolve()
    if not weight.is_file():
        raise FileNotFoundError(weight)
    model = build_model(args.arch)
    load_pretrained_weights(model, str(weight))
    if args.quantized_checkpoint:
        payload = torch.load(args.quantized_checkpoint, map_location="cpu", weights_only=False)
        if payload.get("format") != "hma_dcc_quantized_v1" or payload.get("arch") != args.arch:
            raise ValueError("Quantized checkpoint format or architecture does not match")
        if "model" in payload:
            model = payload["model"].to(device).eval()
            move_quantizer_tensors(model, device)
        else:
            qnn = QuantModel(
                model=model,
                weight_quant_params={"n_bits": payload["weight_bits"], "channel_wise": True, "scale_method": "mse"},
                act_quant_params={"n_bits": payload["activation_bits"], "channel_wise": False, "scale_method": "mse", "leaf_param": True, "prob": 0.5},
            )
            qnn.set_first_last_layer_to_8bit()
            qnn.disable_network_output_quantization()
            set_weight_quantize_params(qnn)
            qnn.load_state_dict(payload["state_dict"], strict=True)
            model = qnn.to(device).eval()
    else:
        model = model.to(device).eval()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "nvidia_smi": subprocess.run(["nvidia-smi"], text=True, capture_output=True, check=False).stdout,
    }
    rows = []
    for batch in (1, 32):
        result = measure(model, device, batch, args.warmup, args.iters, args.repeats)
        rows.append({"arch": args.arch, "label": args.label, "batch_size": batch, **result})
    payload = {
        "protocol": "ispa_p0a_forward_v1",
        "paper_comparable": False,
        "label": args.label,
        "arch": args.arch,
        "seed": args.seed,
        "checkpoint": str(weight),
        "checkpoint_sha256": sha256(weight),
        "quantized_checkpoint": str(Path(args.quantized_checkpoint).resolve()) if args.quantized_checkpoint else None,
        "input_shape": [3, 224, 224],
        "warmup_iterations": args.warmup,
        "measurement_iterations": args.iters,
        "repeats": args.repeats,
        "timing": "CUDA events around model forward only; no I/O",
        "rows": rows,
        "environment": env,
    }
    (out / f"{args.arch}_{args.label}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out / "inference_latency.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
