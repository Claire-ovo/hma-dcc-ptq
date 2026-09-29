#!/usr/bin/env python3
"""Rebuttal diagnostics for HMA routing.

This entry intentionally separates diagnostic evidence from long reconstruction
runs.  It computes raw and dimension-normalized Hessian scores, physical output
perturbation, one-block-at-a-time task degradation, correlation statistics, and
same-K routing selections.  Every sampled image and every per-block value is
archived so a paper-scale CUDA run can reuse exactly the same protocol.
"""

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main_hmadcc import build_model, load_pretrained_weights, seed_all, set_quant_state_robust
from quant.device import empty_device_cache, peak_memory_mb, reset_peak_memory_stats, resolve_device
from quant.quant_block import BaseQuantBlock
from quant.quant_layer import QuantModule
from quant.quant_model import QuantModel
from quant.set_weight_quantize_params import set_weight_quantize_params


IMAGENETTE_TO_IMAGENET = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701,
}


class SynsetImageDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[str, int]], input_size: int):
        self.samples = list(samples)
        self.transform = transforms.Compose(
            [
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, target, path


def discover_samples(root: str, per_class: int, seed: int, dataset_format: str) -> List[Tuple[str, int]]:
    rng = random.Random(seed)
    root_path = Path(root)
    result = []
    if dataset_format == "imagenette":
        class_mapping = sorted(IMAGENETTE_TO_IMAGENET.items())
    else:
        class_names = sorted(
            path.name for path in root_path.iterdir() if path.is_dir() and not path.name.startswith(".")
        )
        if len(class_names) != 1000:
            raise ValueError(f"Expected 1000 ImageNet class directories, found {len(class_names)}")
        class_mapping = list(zip(class_names, range(len(class_names))))
    for synset, target in class_mapping:
        files = sorted(
            str(path)
            for path in (root_path / synset).iterdir()
            if path.suffix.lower() in {".jpeg", ".jpg", ".png"}
        )
        rng.shuffle(files)
        if per_class > 0:
            files = files[:per_class]
        result.extend((path, target) for path in files)
    return result


def discover_balanced_samples(root: str, count: int, seed: int, dataset_format: str):
    grouped = {}
    for path, target in discover_samples(root, 0, seed, dataset_format):
        grouped.setdefault(target, []).append((path, target))
    targets = sorted(grouped)
    random.Random(seed).shuffle(targets)
    result = []
    offset = 0
    while len(result) < count:
        added = False
        for target in targets:
            if offset < len(grouped[target]):
                result.append(grouped[target][offset])
                added = True
                if len(result) == count:
                    return result
        if not added:
            raise ValueError(f"Requested {count} samples, but only {len(result)} are available.")
        offset += 1
    return result


def load_tensor_subset(samples, input_size, batch_size):
    loader = DataLoader(
        SynsetImageDataset(samples, input_size), batch_size=batch_size, shuffle=False, num_workers=0
    )
    images, targets, paths = [], [], []
    for batch_images, batch_targets, batch_paths in loader:
        images.append(batch_images)
        targets.append(batch_targets)
        paths.extend(batch_paths)
    return torch.cat(images), torch.cat(targets), paths


def atomic_units(model: nn.Module):
    units = []

    def visit(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, (BaseQuantBlock, QuantModule)):
                path_parts = full_name.split(".")
                if not any(part in {"fc", "classifier"} for part in path_parts):
                    units.append((full_name, child))
            else:
                visit(child, full_name)

    visit(model)
    return units


def normalize(values: Dict[str, float]) -> Dict[str, float]:
    low, high = min(values.values()), max(values.values())
    scale = high - low
    if scale <= 1e-12:
        return {name: 0.0 for name in values}
    return {name: (value - low) / scale for name, value in values.items()}


def rankdata(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return (concordant - discordant) / denom if denom else float("nan")


def evaluate(model, images, targets, device, batch_size):
    model.eval()
    loss_sum = correct = count = 0
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size].to(device)
            batch_targets = targets[start : start + batch_size].to(device)
            logits = model(batch_images)
            loss_sum += F.cross_entropy(logits, batch_targets, reduction="sum").item()
            correct += logits.argmax(1).eq(batch_targets).sum().item()
            count += len(batch_images)
    return {"loss": loss_sum / count, "top1": 100.0 * correct / count, "count": count}


def get_module_by_name(model, name):
    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(f"Module {name!r} not found")
    return modules[name]


def capture_input(model, module, images, device):
    captured = []

    def hook(_module, inputs, _output):
        captured.append(inputs[0].detach())

    handle = module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(images.to(device))
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("Failed to capture block input")
    return captured[0]


def init_unit_activation(unit, unit_input, search_steps):
    for submodule in unit.modules():
        if isinstance(submodule, (QuantModule, BaseQuantBlock)):
            submodule.act_quantizer.num = search_steps
            submodule.act_quantizer.set_inited(False)
    set_quant_state_robust(unit, True, True)
    with torch.no_grad():
        unit(unit_input)
    for submodule in unit.modules():
        if isinstance(submodule, (QuantModule, BaseQuantBlock)):
            submodule.act_quantizer.set_inited(True)


class SecondOrderMaxPool2d(nn.MaxPool2d):
    """MaxPool2d variant that preserves indices required by second derivatives."""

    def forward(self, value):
        output, _indices = F.max_pool2d(
            value,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            return_indices=True,
        )
        return output


def enable_second_order_maxpool(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MaxPool2d) and not isinstance(child, SecondOrderMaxPool2d):
            replacement = SecondOrderMaxPool2d(
                child.kernel_size,
                child.stride,
                child.padding,
                dilation=child.dilation,
                return_indices=False,
                ceil_mode=child.ceil_mode,
            )
            setattr(module, name, replacement)
        else:
            enable_second_order_maxpool(child)


def estimate_hessian(clean_model, images, targets, device, samples):
    enable_second_order_maxpool(clean_model)
    clean_model.eval().to(device)
    for parameter in clean_model.parameters():
        parameter.requires_grad_(True)
    logits = clean_model(images.to(device))
    loss = F.cross_entropy(logits, targets.to(device))
    conv_params = {
        name: module.weight
        for name, module in clean_model.named_modules()
        if isinstance(module, nn.Conv2d)
    }
    names = list(conv_params)
    params = [conv_params[name] for name in names]
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    used = [(name, param, grad) for name, param, grad in zip(names, params, grads) if grad is not None]
    estimates = {name: 0.0 for name, _, _ in used}
    for _ in range(samples):
        vectors = [torch.randint(0, 2, param.shape, device=device).float().mul_(2).sub_(1) for _, param, _ in used]
        product = sum((grad * vector).sum() for (_, _, grad), vector in zip(used, vectors))
        hvps = torch.autograd.grad(product, [param for _, param, _ in used], retain_graph=True)
        for (name, _, _), vector, hvp in zip(used, vectors, hvps):
            estimates[name] += (hvp * vector).sum().item() / samples
    dimensions = {name: param.numel() for name, param, _ in used}
    return estimates, dimensions


def block_hessian_scores(unit_names, conv_trace, conv_dims):
    scores = {}
    for unit_name in unit_names:
        clean_name = unit_name.removeprefix("model.")
        members = [
            name for name in conv_trace if name == clean_name or name.startswith(clean_name + ".")
        ]
        signed_trace = sum(conv_trace[name] for name in members)
        abs_traces = [abs(conv_trace[name]) for name in members]
        dimension = sum(conv_dims[name] for name in members)
        scores[unit_name] = {
            "hessian_legacy_max": max(abs_traces) if abs_traces else 0.0,
            "hessian_block_trace": abs(signed_trace),
            "hessian_abs_sum": sum(abs_traces),
            "hessian_avg": abs(signed_trace) / dimension if dimension else 0.0,
            "hessian_dimension": dimension,
            "hessian_conv_count": len(members),
        }
    return scores


def top_k(score_map, k, reverse=True):
    return [name for name, _ in sorted(score_map.items(), key=lambda item: (item[1], item[0]), reverse=reverse)[:k]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="resnet18")
    parser.add_argument("--weight-path", required=True)
    parser.add_argument("--calibration-root", required=True)
    parser.add_argument("--validation-root", required=True)
    parser.add_argument(
        "--dataset-format",
        choices=["imagenette", "imagefolder"],
        default="imagenette",
        help="Imagenette uses its ten ImageNet target IDs; imagefolder assigns sorted class directories 0..N-1.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1005)
    parser.add_argument("--input-size", type=int, default=160)
    parser.add_argument("--calibration-per-class", type=int, default=2)
    parser.add_argument("--calibration-count", type=int, default=0, help="Exact balanced count; overrides --calibration-per-class when positive.")
    parser.add_argument("--validation-per-class", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hessian-batch-size", type=int, default=2)
    parser.add_argument("--hessian-samples", type=int, default=2)
    parser.add_argument("--quant-search-steps", type=int, default=16)
    parser.add_argument("--hma-threshold", type=float, default=0.4)
    parser.add_argument("--random-seeds", type=int, nargs="+", default=[1005, 1029, 2023, 3048, 4096])
    parser.add_argument("--paper-comparable", action="store_true")
    args = parser.parse_args()

    seed_all(args.seed)
    device = resolve_device(args.device)
    if args.paper_comparable:
        protocol = {
            "dataset_format": args.dataset_format == "imagefolder",
            "input_size": args.input_size == 224,
            "separate_calibration_and_validation": Path(args.calibration_root).resolve()
            != Path(args.validation_root).resolve(),
            "calibration_count": args.calibration_count == 1024,
            "hessian_samples": args.hessian_samples == 20,
            "hessian_batch_size": args.hessian_batch_size == 64,
            "quant_search_steps": args.quant_search_steps == 100,
            "validation_per_class": args.validation_per_class in (0, 10),
            "device": device.type == "cuda",
        }
        invalid = [name for name, valid in protocol.items() if not valid]
        if invalid:
            raise ValueError("Paper-comparable protocol mismatch: " + ", ".join(invalid))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f"[protocol] device={device} seed={args.seed} output={output_dir}", flush=True)

    if args.calibration_count > 0:
        cali_samples = discover_balanced_samples(
            args.calibration_root, args.calibration_count, args.seed, args.dataset_format
        )
    else:
        cali_samples = discover_samples(
            args.calibration_root, args.calibration_per_class, args.seed, args.dataset_format
        )
    val_samples = discover_samples(
        args.validation_root, args.validation_per_class, args.seed + 1, args.dataset_format
    )
    cali_images, cali_targets, cali_paths = load_tensor_subset(
        cali_samples, args.input_size, args.batch_size
    )
    val_images, val_targets, val_paths = load_tensor_subset(val_samples, args.input_size, args.batch_size)
    print(
        f"[data] calibration={len(cali_images)} validation={len(val_images)} input={args.input_size}",
        flush=True,
    )

    clean_model = build_model(args.arch)
    load_pretrained_weights(clean_model, args.weight_path)
    clean_model.eval()
    q_base = copy.deepcopy(clean_model).to(device).eval()
    fp_base = copy.deepcopy(clean_model).to(device).eval()
    wq = {"n_bits": 2, "channel_wise": True, "scale_method": "mse"}
    aq = {"n_bits": 2, "channel_wise": False, "scale_method": "mse", "leaf_param": True, "prob": 1.0}
    q_model = QuantModel(q_base, wq, aq).to(device).eval()
    fp_model = QuantModel(fp_base, wq, aq, is_fusing=False).to(device).eval()
    q_model.set_first_last_layer_to_8bit()
    q_model.disable_network_output_quantization()
    set_quant_state_robust(q_model, False, False)
    set_quant_state_robust(fp_model, False, False)
    for module in q_model.modules():
        if isinstance(module, QuantModule):
            module.weight_quantizer.num = args.quant_search_steps
    set_weight_quantize_params(q_model)

    units = atomic_units(q_model)
    unit_names = [name for name, _ in units]
    stage_metrics = {}
    print(f"[model] architecture={args.arch} routing_units={len(unit_names)}", flush=True)
    print("[phase 1/3] estimating Hessian trace", flush=True)
    reset_peak_memory_stats(device)
    stage_started = time.time()
    trace, dimensions, hessian_batches = {}, {}, 0
    for start in range(0, len(cali_images), args.hessian_batch_size):
        batch_trace, batch_dimensions = estimate_hessian(
            clean_model,
            cali_images[start : start + args.hessian_batch_size],
            cali_targets[start : start + args.hessian_batch_size],
            device,
            args.hessian_samples,
        )
        for name, value in batch_trace.items():
            trace[name] = trace.get(name, 0.0) + value
        dimensions = batch_dimensions
        hessian_batches += 1
    trace = {name: value / hessian_batches for name, value in trace.items()}
    rows = block_hessian_scores(unit_names, trace, dimensions)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    stage_metrics["hessian"] = {
        "wall_time_seconds": time.time() - stage_started,
        "peak_memory_mb": peak_memory_mb(device),
    }
    empty_device_cache(device)
    print("[phase 1/3] Hessian complete", flush=True)

    # Initialize each unit independently and measure its physical output perturbation.
    print("[phase 2/3] measuring physical perturbation", flush=True)
    reset_peak_memory_stats(device)
    stage_started = time.time()
    for index, (name, q_unit) in enumerate(units, start=1):
        fp_unit = get_module_by_name(fp_model, name)
        set_quant_state_robust(q_model, False, False)
        unit_input = capture_input(fp_model, fp_unit, cali_images[: args.batch_size], device)
        init_unit_activation(q_unit, unit_input, args.quant_search_steps)
        mse_sum = element_count = 0
        with torch.no_grad():
            for start in range(0, len(cali_images), args.batch_size):
                batch_input = capture_input(
                    fp_model, fp_unit, cali_images[start : start + args.batch_size], device
                )
                fp_output = fp_unit(batch_input)
                quant_output = q_unit(batch_input)
                mse_sum += F.mse_loss(quant_output, fp_output, reduction="sum").item()
                element_count += quant_output.numel()
        rows[name]["perturbation_mse"] = mse_sum / element_count
        set_quant_state_robust(q_unit, False, False)
        print(f"[phase 2/3] {index}/{len(units)} {name}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    stage_metrics["perturbation"] = {
        "wall_time_seconds": time.time() - stage_started,
        "peak_memory_mb": peak_memory_mb(device),
    }

    baseline = evaluate(q_model, val_images, val_targets, device, args.batch_size)
    print(f"[baseline] loss={baseline['loss']:.6f} top1={baseline['top1']:.3f}", flush=True)
    print("[phase 3/3] measuring one-block task degradation", flush=True)
    reset_peak_memory_stats(device)
    stage_started = time.time()
    for index, (name, q_unit) in enumerate(units, start=1):
        set_quant_state_robust(q_model, False, False)
        set_quant_state_robust(q_unit, True, True)
        degraded = evaluate(q_model, val_images, val_targets, device, args.batch_size)
        rows[name]["degraded_loss"] = degraded["loss"]
        rows[name]["degraded_top1"] = degraded["top1"]
        rows[name]["loss_increase"] = degraded["loss"] - baseline["loss"]
        rows[name]["accuracy_drop"] = baseline["top1"] - degraded["top1"]
        set_quant_state_robust(q_unit, False, False)
        print(
            f"[phase 3/3] {index}/{len(units)} {name} "
            f"loss_increase={rows[name]['loss_increase']:.6f} "
            f"acc_drop={rows[name]['accuracy_drop']:.3f}",
            flush=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    stage_metrics["block_degradation"] = {
        "wall_time_seconds": time.time() - stage_started,
        "peak_memory_mb": peak_memory_mb(device),
    }

    metric_names = [
        "hessian_legacy_max",
        "hessian_block_trace",
        "hessian_abs_sum",
        "hessian_avg",
        "perturbation_mse",
    ]
    normalized = {
        metric: normalize({name: rows[name][metric] for name in unit_names}) for metric in metric_names
    }
    for name in unit_names:
        for metric in metric_names:
            rows[name]["norm_" + metric] = normalized[metric][name]
        rows[name]["hma_original"] = max(
            rows[name]["norm_hessian_legacy_max"], rows[name]["norm_perturbation_mse"]
        )
        rows[name]["hma_avg"] = max(
            rows[name]["norm_hessian_avg"], rows[name]["norm_perturbation_mse"]
        )
        rows[name]["fusion_mean"] = 0.5 * (
            rows[name]["norm_hessian_legacy_max"] + rows[name]["norm_perturbation_mse"]
        )
        rows[name]["fusion_h75_p25"] = 0.75 * rows[name]["norm_hessian_legacy_max"] + 0.25 * rows[name]["norm_perturbation_mse"]
        rows[name]["fusion_h25_p75"] = 0.25 * rows[name]["norm_hessian_legacy_max"] + 0.75 * rows[name]["norm_perturbation_mse"]
    h_ranks = rankdata([rows[name]["hessian_legacy_max"] for name in unit_names])
    p_ranks = rankdata([rows[name]["perturbation_mse"] for name in unit_names])
    for index, name in enumerate(unit_names):
        rows[name]["fusion_rank_sum"] = float(h_ranks[index] + p_ranks[index])

    score_names = [
        "hessian_legacy_max",
        "hessian_block_trace",
        "hessian_abs_sum",
        "hessian_avg",
        "perturbation_mse",
        "hma_original",
        "hma_avg",
        "fusion_mean",
        "fusion_h75_p25",
        "fusion_h25_p75",
        "fusion_rank_sum",
    ]
    degradation = [rows[name]["loss_increase"] for name in unit_names]
    correlations = {}
    for score_name in score_names:
        values = [rows[name][score_name] for name in unit_names]
        correlations[score_name] = {
            "spearman_loss_increase": spearman(values, degradation),
            "kendall_loss_increase": kendall_tau_b(values, degradation),
        }

    original_selected = [
        name for name in unit_names if rows[name]["hma_original"] >= args.hma_threshold
    ]
    k = len(original_selected)
    if k == 0:
        k = 1
        original_selected = top_k({name: rows[name]["hma_original"] for name in unit_names}, k)
    route_selections = {
        "hma_threshold": sorted(original_selected),
        "hma_same_k": sorted(top_k({name: rows[name]["hma_original"] for name in unit_names}, k)),
        "inverted_same_k": sorted(top_k({name: rows[name]["hma_original"] for name in unit_names}, k, reverse=False)),
    }
    for score_name in score_names:
        route_selections[score_name + "_same_k"] = sorted(
            top_k({name: rows[name][score_name] for name in unit_names}, k)
        )
    for random_seed in args.random_seeds:
        rng = random.Random(random_seed)
        route_selections[f"random_{random_seed}"] = sorted(rng.sample(unit_names, k))

    positive_degradation = {name: max(rows[name]["loss_increase"], 0.0) for name in unit_names}
    total_positive = sum(positive_degradation.values())
    selection_summary = {}
    for route_name, selected in route_selections.items():
        captured = sum(positive_degradation[name] for name in selected)
        selection_summary[route_name] = {
            "k": len(selected),
            "selected": selected,
            "captured_positive_loss_increase": captured,
            "coverage": captured / total_positive if total_positive else 0.0,
        }

    columns = ["block"] + sorted(next(iter(rows.values())).keys())
    with (output_dir / "block_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name in unit_names:
            writer.writerow({"block": name, **rows[name]})

    manifest = {
        "protocol": {
            **vars(args),
            "device_resolved": str(device),
            "calibration_samples": len(cali_images),
            "validation_samples": len(val_images),
            "hessian_batches": hessian_batches,
            "perturbation_batches": math.ceil(len(cali_images) / args.batch_size),
            "calibration_paths": cali_paths,
            "validation_paths": val_paths,
            "first_last_8bit": True,
            "network_output_quantization_disabled": True,
            "second_order_maxpool_indices": True,
            "paper_comparable": args.paper_comparable,
            "limitation": None if args.paper_comparable else "Exploratory diagnostic; not valid for rebuttal claims.",
        },
        "baseline": baseline,
        "unit_count": len(unit_names),
        "sensitive_k": k,
        "correlations": correlations,
        "route_selections": selection_summary,
        "stage_metrics": stage_metrics,
        "elapsed_seconds": time.time() - started,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    (output_dir / "calibration.files").write_text("\n".join(cali_paths) + "\n", encoding="utf-8")
    (output_dir / "validation.files").write_text("\n".join(val_paths) + "\n", encoding="utf-8")
    print(json.dumps({
        "baseline": baseline,
        "unit_count": len(unit_names),
        "sensitive_k": k,
        "correlations": correlations,
        "stage_metrics": stage_metrics,
        "elapsed_seconds": manifest["elapsed_seconds"],
        "output_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
