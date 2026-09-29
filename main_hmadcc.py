import argparse
import builtins
import copy
import inspect
import json
import os
import random
import time
from typing import Callable, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.imagenet import build_imagenet_data
from models.mnasnet import mnasnet
from models.mobilenetv2 import mobilenetv2
from models.regnet import regnetx_600m, regnetx_3200m
from models.resnet import resnet18, resnet50
from quant.block_recon import block_reconstruction
from quant.layer_recon import layer_reconstruction
from quant.quant_block import BaseQuantBlock
from quant.quant_layer import QuantModule
from quant.quant_model import QuantModel
from quant.device import empty_device_cache, peak_memory_mb, reset_peak_memory_stats, resolve_device
from quant.set_weight_quantize_params import get_init, set_weight_quantize_params


MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "mobilenetv2": mobilenetv2,
    "mnasnet": mnasnet,
    "regnetx_600m": regnetx_600m,
    "regnetx_600mf": regnetx_600m,
    "regnetx_3200m": regnetx_3200m,
}


def seed_all(seed: int = 1029) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def validate_model(val_loader, model, device, print_freq: int = 100) -> float:
    model.eval()
    top1 = AverageMeter()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(images)
            acc1, = accuracy(output, target, topk=(1,))
            top1.update(acc1.item(), images.size(0))
            if i % print_freq == 0:
                print(f"Validation [{i}/{len(val_loader)}] Acc@1 {top1.val:.3f} ({top1.avg:.3f})")
    print(f"Final Acc@1 {top1.avg:.3f}")
    return float(top1.avg)


def get_train_samples(train_loader, num_samples: int, ordered_manifest: bool = False):
    train_data, targets = [], []
    for batch in train_loader:
        train_data.append(batch[0])
        targets.append(batch[1])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return (
        torch.cat(train_data, dim=0)[:num_samples],
        torch.cat(targets, dim=0)[:num_samples],
        [path for path, _ in train_loader.dataset.samples[:num_samples]] if ordered_manifest else None,
    )


def build_model(arch: str) -> nn.Module:
    if arch not in MODEL_REGISTRY:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported architecture '{arch}'. Available choices: {valid}")
    return MODEL_REGISTRY[arch]()


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def strip_module_prefix(state_dict):
    clean_state_dict = {}
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith("module.") else key
        clean_state_dict[clean_key] = value
    return clean_state_dict


def load_pretrained_weights(model: nn.Module, weight_path: str) -> None:
    if not weight_path:
        raise ValueError("--weight-path is required for reproducible ImageNet evaluation.")
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
    state_dict = strip_module_prefix(extract_state_dict(checkpoint))
    if not isinstance(state_dict, dict):
        raise TypeError("The checkpoint must be a state dict or contain a state_dict/model entry.")

    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    matched_keys = model_keys.intersection(checkpoint_keys)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Loaded weights with non-strict key matching. Missing={len(missing)}, unexpected={len(unexpected)}")
    if len(matched_keys) == 0:
        print("WARNING: No checkpoint keys matched the model. Please verify --arch and --weight-path.")
    elif len(missing) > 0.25 * len(model_keys):
        print(
            "WARNING: A large fraction of model parameters were not loaded. "
            "Please verify that the checkpoint matches the selected architecture."
        )
    critical_missing = [key for key in missing if key.startswith(("fc.", "classifier."))]
    if critical_missing:
        print(f"WARNING: Classifier head parameters were not loaded: {critical_missing[:4]}")
    print(f"Loaded pretrained weights from: {weight_path}")


def set_quant_state_robust(module, weight_quant: bool = False, act_quant: bool = False) -> None:
    if hasattr(module, "set_quant_state"):
        module.set_quant_state(weight_quant, act_quant)
    for child in module.children():
        set_quant_state_robust(child, weight_quant, act_quant)


def matches_head_name(full_name: str, head_name: str) -> bool:
    return full_name == head_name or full_name.endswith(f".{head_name}")


def is_fc_head(full_name: str) -> bool:
    return matches_head_name(full_name, "fc")


def is_classifier_head(full_name: str) -> bool:
    return matches_head_name(full_name, "classifier")


def should_skip_for_hma(full_name: str, protect_head: bool) -> bool:
    if is_fc_head(full_name):
        return True
    if protect_head and is_classifier_head(full_name):
        return True
    return False


def should_skip_reconstruction(full_name: str, protect_head: bool) -> bool:
    if not protect_head:
        return False
    return is_fc_head(full_name) or is_classifier_head(full_name)


def extract_hybrid_metrics(
    q_model,
    fp_model,
    clean_fp_model,
    cali_data,
    cali_target,
    batch_size: int,
    threshold: float,
    protect_head: bool,
    hessian_samples: int,
    act_quant: bool,
):
    device = next(fp_model.parameters()).device
    metric_dict = {}
    stage_metrics = {}

    print("\n[Phase 1] Estimating HMA routing metrics")
    reset_peak_memory_stats(device)
    hessian_started = time.perf_counter()

    clean_fp_model = clean_fp_model.to(device).eval()
    for param in clean_fp_model.parameters():
        param.requires_grad = True

    clean_fp_model.zero_grad()
    logits = clean_fp_model(cali_data[:batch_size].to(device))
    loss = F.cross_entropy(logits, cali_target[:batch_size].to(device))

    conv_params = {}
    for name, module in clean_fp_model.named_modules():
        if isinstance(module, nn.Conv2d) and not should_skip_for_hma(name, protect_head):
            conv_params["model." + name] = module.weight

    grads = torch.autograd.grad(loss, list(conv_params.values()), create_graph=True, allow_unused=True)

    used_conv_params, used_grads = {}, []
    for (name, param), grad in zip(conv_params.items(), grads):
        if grad is not None:
            used_conv_params[name] = param
            used_grads.append(grad)

    max_iter = hessian_samples
    trace_dict = {name: 0.0 for name in used_conv_params.keys()}
    for _ in range(max_iter):
        vectors = [
            torch.randint_like(param, high=2, device=device, dtype=torch.float32) * 2 - 1
            for param in used_conv_params.values()
        ]
        grad_vector_product = sum(torch.sum(grad * vector) for grad, vector in zip(used_grads, vectors))
        hvp = torch.autograd.grad(
            grad_vector_product,
            list(used_conv_params.values()),
            retain_graph=True,
            allow_unused=True,
        )
        for idx, name in enumerate(used_conv_params.keys()):
            if hvp[idx] is not None:
                trace_dict[name] += torch.sum(hvp[idx] * vectors[idx]).item()

    for name in trace_dict:
        trace_dict[name] = abs(trace_dict[name] / max_iter)

    # Release the second-order autograd graph before measuring perturbation memory.
    del logits, loss, grads, used_grads, vectors, grad_vector_product, hvp
    synchronize_device(device)
    stage_metrics["hessian"] = {
        "wall_time_seconds": time.perf_counter() - hessian_started,
        "peak_memory_mb": peak_memory_mb(device),
        "flops": None,
        "flops_scope": "not estimated by the current profiler",
    }
    empty_device_cache(device)

    def evaluate_topology(cur_model, cur_fp_model, prefix=""):
        for (name, module), (_, fp_module) in zip(cur_model.named_children(), cur_fp_model.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if should_skip_for_hma(full_name, protect_head):
                continue

            if isinstance(module, (BaseQuantBlock, QuantModule)):
                block_traces = [value for key, value in trace_dict.items() if full_name in key]
                hessian_score = max(block_traces) if block_traces else 0.0

                cached_inps = get_init(q_model, module, cali_data, batch_size=batch_size, keep_gpu=True)
                cur_inp = cached_inps[:batch_size].to(device)

                with torch.no_grad():
                    fp_out = fp_module(cur_inp)

                set_quant_state_robust(module, True, act_quant)
                with torch.no_grad():
                    _ = module(cur_inp)

                for sub_module in module.modules():
                    if isinstance(sub_module, QuantModule) and sub_module.weight_quantizer.delta is None:
                        sub_module.weight_quantizer.init_quantization_scale(
                            sub_module.weight,
                            channel_wise=True,
                        )

                with torch.no_grad():
                    q_out = module(cur_inp)

                perturb_score = F.mse_loss(q_out, fp_out).item()
                set_quant_state_robust(module, False, False)

                metric_dict[full_name] = {"hessian": hessian_score, "perturb": perturb_score}
            else:
                evaluate_topology(module, fp_module, full_name)

    reset_peak_memory_stats(device)
    perturbation_started = time.perf_counter()
    with torch.no_grad():
        evaluate_topology(q_model, fp_model)
    synchronize_device(device)
    stage_metrics["perturbation"] = {
        "wall_time_seconds": time.perf_counter() - perturbation_started,
        "peak_memory_mb": peak_memory_mb(device),
        "flops": None,
        "flops_scope": "not estimated by the current profiler",
    }

    h_vals = [metrics["hessian"] for metrics in metric_dict.values()]
    p_vals = [metrics["perturb"] for metrics in metric_dict.values()]
    h_min, h_max = (min(h_vals), max(h_vals)) if h_vals else (0.0, 1.0)
    p_min, p_max = (min(p_vals), max(p_vals)) if p_vals else (0.0, 1.0)

    routing_table = {}
    print(f"\n{'-' * 86}")
    print(
        f"{'Block Name':<22} | {'Curvature':<10} | {'Perturbation':<12} | "
        f"{'Hybrid':<8} | {'Route'}"
    )
    print(f"{'-' * 86}")

    for name, metrics in metric_dict.items():
        h_norm = (metrics["hessian"] - h_min) / (h_max - h_min + 1e-8)
        p_norm = (metrics["perturb"] - p_min) / (p_max - p_min + 1e-8)
        hybrid_score = max(h_norm, p_norm)
        is_sensitive = hybrid_score >= threshold
        route = "Sensitive" if is_sensitive else "Robust"
        routing_table[name] = is_sensitive
        print(f"{name:<22} | {h_norm:<10.4f} | {p_norm:<12.4f} | {hybrid_score:<8.4f} | {route}")

    print(f"{'-' * 86}")
    return routing_table, stage_metrics


def selected_route_from_payload(payload, key: str):
    routes = payload.get("route_selections", payload)
    if key not in routes:
        raise KeyError(f"Route {key!r} not found. Available routes: {', '.join(sorted(routes))}")
    route = routes[key]
    selected = route.get("selected") if isinstance(route, dict) else route
    if not isinstance(selected, list) or not all(isinstance(name, str) for name in selected):
        raise TypeError(f"Route {key!r} must be a list or contain a string-list 'selected' field.")
    if len(selected) != len(set(selected)):
        raise ValueError(f"Route {key!r} contains duplicate block names.")
    return set(selected)


def load_routing_table(q_model, json_path: str, key: str):
    with open(json_path, "r", encoding="utf-8") as stream:
        selected = selected_route_from_payload(json.load(stream), key)

    routing_table = {}

    def visit(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, (BaseQuantBlock, QuantModule)):
                routing_table[full_name] = full_name in selected
            else:
                visit(child, full_name)

    visit(q_model)
    unknown = selected.difference(routing_table)
    if unknown:
        raise ValueError(f"Route {key!r} contains unknown blocks: {', '.join(sorted(unknown))}")
    print(f"Loaded route {key!r}: {len(selected)}/{len(routing_table)} blocks use the sensitive budget.")
    return routing_table


def apply_routing_policy(routing_table, policy: str, seed: int):
    if policy == "hma":
        return routing_table
    names = sorted(routing_table)
    k = sum(routing_table.values())
    if policy == "inverted":
        selected = {name for name, sensitive in routing_table.items() if not sensitive}
        selected = set(sorted(selected)[:k])
    elif policy == "uniform":
        indices = [round(i * (len(names) - 1) / max(1, k - 1)) for i in range(k)]
        selected = {names[i] for i in indices}
    elif policy == "random":
        generator = random.Random(seed)
        selected = set(generator.sample(names, k))
    else:
        raise ValueError(f"Unknown routing policy: {policy}")
    return {name: name in selected for name in names}


def reconstruct_model(
    qnn,
    fp_model,
    routing_table,
    cali_data,
    args,
    profile_callback=None,
):
    def recon_model_topology(cur_model: nn.Module, cur_fp_model: nn.Module, prefix=""):
        for (name, module), (_, fp_module) in zip(cur_model.named_children(), cur_fp_model.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if should_skip_reconstruction(full_name, args.protect_head):
                print(f"[Recon] {full_name:<24} | skipped head module")
                continue

            cur_kwargs = dict(
                cali_data=cali_data,
                batch_size=args.batch_size,
                weight=args.recon_weight,
                lr=args.recon_lr,
                b_range=(args.b_start, args.b_end),
                warmup=args.warmup,
                opt_mode=args.opt_mode,
                input_prob=args.input_prob,
                keep_gpu=True,
                asym=args.asym,
                act_quant=args.act_quant,
                use_infonce=False,
                infonce_lambda=0.0,
                infonce_tau=args.infonce_tau,
                profile_callback=profile_callback,
                profile_label=full_name,
            )

            is_sensitive = routing_table.get(full_name, True)
            if is_sensitive:
                cur_kwargs["iters"] = args.iters_sensitive
                cur_kwargs["use_infonce"] = False
                cur_kwargs["infonce_lambda"] = 0.0
            else:
                cur_kwargs["iters"] = args.iters_robust
                cur_kwargs["use_infonce"] = not args.disable_infonce
                cur_kwargs["infonce_lambda"] = args.infonce_lambda if cur_kwargs["use_infonce"] else 0.0

            if is_fc_head(full_name):
                cur_kwargs["iters"] = args.iters_robust
                cur_kwargs["use_infonce"] = False
                cur_kwargs["infonce_lambda"] = 0.0

            if isinstance(module, (QuantModule, BaseQuantBlock)):
                status = f"ON (lambda={cur_kwargs['infonce_lambda']})" if cur_kwargs["use_infonce"] else "OFF"
                print(f"[Recon] {full_name:<24} | iters={cur_kwargs['iters']:<6} | InfoNCE={status}")

                if isinstance(module, QuantModule):
                    valid_keys = inspect.signature(layer_reconstruction).parameters.keys()
                    safe_kwargs = {key: value for key, value in cur_kwargs.items() if key in valid_keys}
                    layer_reconstruction(qnn, fp_model, module, fp_module, **safe_kwargs)
                else:
                    valid_keys = inspect.signature(block_reconstruction).parameters.keys()
                    safe_kwargs = {key: value for key, value in cur_kwargs.items() if key in valid_keys}
                    block_reconstruction(qnn, fp_model, module, fp_module, **safe_kwargs)
            else:
                recon_model_topology(module, fp_module, full_name)

    recon_model_topology(qnn, fp_model)


def save_summary(args, summary) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"hmadcc_{args.arch}_W{args.n_bits_w}A{args.n_bits_a}_seed{args.seed}.json",
    )
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"Saved summary to: {output_path}")


def save_quantized_checkpoint(qnn: nn.Module, args, path: str) -> None:
    """Save the calibrated quantized state for an independent forward benchmark."""
    state_dict = {key: value.detach().cpu() for key, value in qnn.state_dict().items()}
    checkpoint_model = copy.deepcopy(qnn).cpu()
    payload = {
        "format": "hma_dcc_quantized_v1",
        "arch": args.arch,
        "weight_bits": args.n_bits_w,
        "activation_bits": args.n_bits_a,
        "seed": args.seed,
        "model": checkpoint_model,
        "state_dict": state_dict,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)
    print(f"Saved quantized checkpoint to: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="HMA-DCC ImageNet post-training quantization")
    parser.add_argument("--arch", default="resnet18", choices=sorted(MODEL_REGISTRY.keys()))
    parser.add_argument("--data-dir", required=True, help="ImageNet root used when --train-dir/--val-dir are omitted.")
    parser.add_argument("--train-dir", help="Optional explicit ImageNet training directory.")
    parser.add_argument("--val-dir", help="Optional explicit ImageNet validation directory.")
    parser.add_argument("--calibration-manifest", help="Optional ordered list of calibration images relative to --train-dir.")
    parser.add_argument("--validation-manifest", help="Optional ordered list of validation images relative to --val-dir.")
    parser.add_argument("--weight-path", required=True, help="Path to the pretrained FP32 checkpoint.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated summaries.")
    parser.add_argument("--save-quantized-checkpoint", help="Optional path for the calibrated quantized state_dict.")
    parser.add_argument("--n-bits-w", default=2, type=int, help="Weight quantization bit-width.")
    parser.add_argument("--n-bits-a", default=2, type=int, help="Activation quantization bit-width.")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-calibration-samples", default=1024, type=int)
    parser.add_argument("--seed", default=1005, type=int)
    parser.add_argument("--hma-threshold", default=0.4, type=float, help="Hard-routing threshold for HMA scores.")
    parser.add_argument("--iters-sensitive", default=20000, type=int)
    parser.add_argument("--iters-robust", default=5000, type=int)
    parser.add_argument("--infonce-lambda", default=0.1, type=float)
    parser.add_argument("--infonce-tau", default=0.1, type=float)
    parser.add_argument("--disable-infonce", action="store_true", help="Disable InfoNCE while keeping routing and budgets fixed.")
    parser.add_argument("--routing-json", help="Reuse archived route selections instead of recomputing HMA metrics.")
    parser.add_argument("--routing-key", help="Selection key under route_selections in --routing-json.")
    parser.add_argument("--routing-policy", choices=("hma", "uniform", "random", "inverted"), default="hma")
    parser.add_argument("--hessian-samples", default=20, type=int)
    parser.add_argument("--recon-weight", default=0.01, type=float)
    parser.add_argument("--recon-lr", default=4e-5, type=float)
    parser.add_argument("--b-start", default=20, type=float)
    parser.add_argument("--b-end", default=2, type=float)
    parser.add_argument("--warmup", default=0.2, type=float)
    parser.add_argument("--opt-mode", default="mse", type=str)
    parser.add_argument("--input-prob", default=0.5, type=float)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Calibration device (paper protocol default: cuda); auto, mps, and cpu are explicit smoke-test modes.",
    )
    quant_group = parser.add_mutually_exclusive_group()
    quant_group.add_argument("--asym", dest="asym", action="store_true", default=True)
    quant_group.add_argument("--symmetric", dest="asym", action="store_false")
    act_group = parser.add_mutually_exclusive_group()
    act_group.add_argument("--act-quant", dest="act_quant", action="store_true", default=True)
    act_group.add_argument("--disable-act-quant", dest="act_quant", action="store_false")
    parser.add_argument(
        "--protect-head",
        action="store_true",
        help="Skip classifier heads during reconstruction. Disabled by default to match the hard-routing baseline.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.hessian_samples <= 0:
        raise ValueError("--hessian-samples must be positive.")
    if bool(args.routing_json) != bool(args.routing_key):
        raise ValueError("--routing-json and --routing-key must be provided together.")
    seed_all(args.seed)

    device = resolve_device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)

    print("HMA-DCC ImageNet PTQ")
    print(f"Architecture: {args.arch}")
    print(f"Quantization: W{args.n_bits_w}A{args.n_bits_a}")
    print(f"Calibration samples: {args.num_calibration_samples}")
    print(f"HMA hard-routing threshold: {args.hma_threshold}")
    print(f"Hessian estimation samples: {args.hessian_samples}")
    print(f"Head protection: {'enabled' if args.protect_head else 'disabled'}")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("WARNING: Non-CUDA runs are for local functional validation; resource metrics are not paper-comparable.")

    train_loader, test_loader = build_imagenet_data(
        batch_size=args.batch_size,
        workers=args.workers,
        data_path=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        calibration_manifest=args.calibration_manifest,
        validation_manifest=args.validation_manifest,
    )
    if args.calibration_manifest and len(train_loader.dataset) != args.num_calibration_samples:
        raise ValueError(
            "--calibration-manifest must contain exactly --num-calibration-samples images "
            f"({len(train_loader.dataset)} != {args.num_calibration_samples})."
        )

    cnn = build_model(args.arch)
    load_pretrained_weights(cnn, args.weight_path)
    clean_fp_model = copy.deepcopy(cnn)
    fp_model_base = copy.deepcopy(cnn)
    cnn = cnn.to(device).eval()
    fp_model_base = fp_model_base.to(device).eval()

    wq_params = {"n_bits": args.n_bits_w, "channel_wise": True, "scale_method": "mse"}
    aq_params = {
        "n_bits": args.n_bits_a,
        "channel_wise": False,
        "scale_method": "mse",
        "leaf_param": True,
        "prob": args.input_prob,
    }

    fp_model = QuantModel(model=fp_model_base, weight_quant_params=wq_params, act_quant_params=aq_params, is_fusing=False)
    fp_model.set_quant_state(False, False)

    print("\n[Step 0] Evaluating the FP32 reference model")
    fp32_acc = validate_model(test_loader, fp_model, device)

    qnn = QuantModel(model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.set_first_last_layer_to_8bit()
    qnn.disable_network_output_quantization()

    cali_data, cali_target, calibration_paths = get_train_samples(
        train_loader,
        num_samples=args.num_calibration_samples,
        ordered_manifest=bool(args.calibration_manifest),
    )
    calibration_count = int(cali_data.size(0))
    calibration_sampling = "ordered_manifest" if args.calibration_manifest else "loader_native_seeded"
    set_weight_quantize_params(qnn)

    calibration_started = time.perf_counter()
    if args.routing_json:
        routing_table = load_routing_table(qnn, args.routing_json, args.routing_key)
        stage_metrics = {
            "routing": {
                "source": "archived_route",
                "hessian": None,
                "perturbation": None,
            }
        }
    else:
        routing_table, routing_stage_metrics = extract_hybrid_metrics(
            qnn,
            fp_model,
            clean_fp_model,
            cali_data,
            cali_target,
            batch_size=args.batch_size,
            threshold=args.hma_threshold,
            protect_head=args.protect_head,
            hessian_samples=args.hessian_samples,
            act_quant=args.act_quant,
        )
        stage_metrics = {"routing": routing_stage_metrics}
    routing_table = apply_routing_policy(routing_table, args.routing_policy, args.seed)

    builtins.GLOBAL_CALIBRATION_FLOPS = 0.0
    reset_peak_memory_stats(device)
    reconstruction_started = time.perf_counter()
    memory_bank_metrics = {"event_count": 0, "wall_time_seconds": 0.0}
    block_cost_components = {}

    def record_profile_event(name: str, payload) -> None:
        if name == "memory_bank":
            memory_bank_metrics["event_count"] += 1
            memory_bank_metrics["wall_time_seconds"] += payload
        elif name == "flop_components":
            label = payload.get("label")
            if not label:
                raise ValueError("FLOPs profile callback did not provide a block label.")
            block_cost_components[label] = payload
        else:
            raise ValueError(f"Unknown calibration profile event: {name}")

    print("\n[Step 1] Running HMA-DCC block-wise calibration")
    reconstruct_model(qnn, fp_model, routing_table, cali_data, args, record_profile_event)

    synchronize_device(device)
    wall_clock_time = time.perf_counter() - reconstruction_started
    peak_memory = peak_memory_mb(device)
    total_flops = getattr(builtins, "GLOBAL_CALIBRATION_FLOPS", 0.0)
    route_cost_profile = {}
    for label, components in block_cost_components.items():
        base_per_iteration = components["base_flops_per_iteration"]
        route_cost_profile[label] = {
            "high_budget_flops": base_per_iteration * args.iters_sensitive,
            "low_budget_flops": base_per_iteration * args.iters_robust,
            "base_flops_per_iteration": base_per_iteration,
            "tail_flops_per_iteration": components["tail_flops_per_iteration"],
            "cost_scope": "reconstruction-module FLOPs only; valid for route matching only when InfoNCE is disabled",
        }
    stage_metrics["memory_bank"] = {
        **memory_bank_metrics,
        "peak_memory_mb": None,
        "peak_memory_scope": "interleaved with reconstruction; report the reconstruction-stage peak instead",
        "flops": None,
        "flops_scope": "not separately estimated by the current profiler",
    }
    stage_metrics["reconstruction"] = {
        "wall_time_seconds": wall_clock_time,
        "peak_memory_mb": peak_memory,
        "flops": total_flops,
        "flops_scope": "estimated phase-2 reconstruction objective only",
    }
    routing_peaks = [
        value.get("peak_memory_mb")
        for value in stage_metrics["routing"].values()
        if isinstance(value, dict) and value.get("peak_memory_mb") is not None
    ]
    stage_metrics["end_to_end_calibration"] = {
        "wall_time_seconds": time.perf_counter() - calibration_started,
        "peak_memory_mb": max([peak_memory, *routing_peaks]) if peak_memory is not None else None,
        "flops": None,
        "flops_scope": "not available until all calibration stages use a common FLOPs profiler",
    }

    print("\n[Step 2] Evaluating the quantized model")
    qnn.set_quant_state(weight_quant=True, act_quant=args.act_quant)
    if args.save_quantized_checkpoint:
        save_quantized_checkpoint(qnn, args, args.save_quantized_checkpoint)
    final_acc = validate_model(test_loader, qnn, device)

    acc_drop = fp32_acc - final_acc
    paper_comparable = (
        args.num_calibration_samples == 1024
        and calibration_count == 1024
        and len(test_loader.dataset) == 50000
        and args.seed == 1005
        and args.batch_size == 64
        and args.n_bits_w == 2
        and args.n_bits_a == 2
        and args.asym
        and args.act_quant
        and args.infonce_lambda == 0.1
        and args.infonce_tau == 0.1
        and args.hma_threshold == 0.4
        and args.iters_sensitive == 20000
        and args.iters_robust == 5000
        and args.hessian_samples == 20
        and args.recon_weight == 0.01
        and args.recon_lr == 4e-5
        and args.b_start == 20
        and args.b_end == 2
        and args.warmup == 0.2
        and args.opt_mode == "mse"
        and args.input_prob == 0.5
        and args.protect_head
        and device.type == "cuda"
        and args.calibration_manifest is None
        and calibration_paths is None
        and args.validation_manifest is None
        and args.routing_json is None
        and args.routing_key is None
        and args.routing_policy == "hma"
        and not args.disable_infonce
    )
    summary = {
        "protocol": "original_paper_v1" if paper_comparable else "custom_control",
        "paper_comparable": paper_comparable,
        "arch": args.arch,
        "weight_bits": args.n_bits_w,
        "activation_bits": args.n_bits_a,
        "seed": args.seed,
        "num_calibration_samples": args.num_calibration_samples,
        "batch_size": args.batch_size,
        "device_type": device.type,
        "data_dir": os.path.abspath(args.data_dir),
        "train_dir": os.path.abspath(args.train_dir) if args.train_dir else None,
        "val_dir": os.path.abspath(args.val_dir) if args.val_dir else None,
        "calibration_manifest": os.path.abspath(args.calibration_manifest) if args.calibration_manifest else None,
        "validation_manifest": os.path.abspath(args.validation_manifest) if args.validation_manifest else None,
        "calibration_sampling": calibration_sampling,
        "calibration_paths": calibration_paths,
        "validation_paths": [path for path, _ in test_loader.dataset.samples],
        "calibration_count": calibration_count,
        "validation_count": len(test_loader.dataset),
        "hma_threshold": args.hma_threshold,
        "iters_sensitive": args.iters_sensitive,
        "iters_robust": args.iters_robust,
        "infonce_lambda": args.infonce_lambda,
        "infonce_tau": args.infonce_tau,
        "infonce_enabled": not args.disable_infonce,
        "routing_json": args.routing_json,
        "routing_key": args.routing_key,
        "routing_policy": args.routing_policy,
        "hessian_samples": args.hessian_samples,
        "recon_weight": args.recon_weight,
        "recon_lr": args.recon_lr,
        "b_range": [args.b_start, args.b_end],
        "warmup": args.warmup,
        "opt_mode": args.opt_mode,
        "input_prob": args.input_prob,
        "asymmetric_quantization": args.asym,
        "activation_quantization": args.act_quant,
        "protect_head": args.protect_head,
        "fp32_top1": fp32_acc,
        "quantized_top1": final_acc,
        "accuracy_drop": acc_drop,
        "calibration_flops": total_flops,
        "calibration_flops_scope": "estimated phase-2 reconstruction objective only",
        "route_cost_profile": route_cost_profile,
        "peak_memory_mb": peak_memory,
        "wall_time_seconds": wall_clock_time,
        "stage_metrics": stage_metrics,
    }

    print("\n" + "=" * 72)
    print("HMA-DCC Result")
    print("-" * 72)
    print(f"FP32 Top-1          : {fp32_acc:.2f}%")
    print(f"Quantized Top-1     : {final_acc:.2f}%")
    print(f"Accuracy drop       : {acc_drop:.2f}%")
    print(f"Calibration FLOPs   : {total_flops / 1e12:.4f} TFLOPs")
    peak_memory_text = f"{peak_memory:.2f} MB" if peak_memory is not None else "N/A (CUDA-only metric)"
    print(f"Peak memory         : {peak_memory_text}")
    print(f"Wall time           : {wall_clock_time / 3600:.2f} h ({wall_clock_time:.2f} s)")
    print("=" * 72 + "\n")

    save_summary(args, summary)


if __name__ == "__main__":
    main()
