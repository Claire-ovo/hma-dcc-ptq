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


def get_train_samples(train_loader, num_samples: int):
    train_data, targets = [], []
    for batch in train_loader:
        train_data.append(batch[0])
        targets.append(batch[1])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples], torch.cat(targets, dim=0)[:num_samples]


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

    checkpoint = torch.load(weight_path, map_location="cpu")
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
):
    device = next(fp_model.parameters()).device
    metric_dict = {}

    print("\n[Phase 1] Estimating HMA routing metrics")

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

    max_iter = 20
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

    torch.cuda.empty_cache()

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

                set_quant_state_robust(module, True, True)
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

    with torch.no_grad():
        evaluate_topology(q_model, fp_model)

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
    return routing_table


def reconstruct_model(
    qnn,
    fp_model,
    routing_table,
    cali_data,
    args,
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
                weight=0.01,
                lr=4e-5,
                b_range=(20, 2),
                warmup=0.2,
                opt_mode="mse",
                input_prob=0.5,
                keep_gpu=True,
                asym=True,
                act_quant=True,
                use_infonce=False,
                infonce_lambda=0.0,
                infonce_tau=0.1,
            )

            is_sensitive = routing_table.get(full_name, True)
            if is_sensitive:
                cur_kwargs["iters"] = args.iters_sensitive
                cur_kwargs["use_infonce"] = False
                cur_kwargs["infonce_lambda"] = 0.0
            else:
                cur_kwargs["iters"] = args.iters_robust
                cur_kwargs["use_infonce"] = True
                cur_kwargs["infonce_lambda"] = args.infonce_lambda

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


def parse_args():
    parser = argparse.ArgumentParser(description="HMA-DCC ImageNet post-training quantization")
    parser.add_argument("--arch", default="resnet18", choices=sorted(MODEL_REGISTRY.keys()))
    parser.add_argument("--data-dir", required=True, help="Path to the ImageNet directory containing train/ and val/.")
    parser.add_argument("--weight-path", required=True, help="Path to the pretrained FP32 checkpoint.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated summaries.")
    parser.add_argument("--n-bits-w", default=2, type=int, help="Weight quantization bit-width.")
    parser.add_argument("--n-bits-a", default=2, type=int, help="Activation quantization bit-width.")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-calibration-samples", default=1024, type=int)
    parser.add_argument("--seed", default=1005, type=int)
    parser.add_argument("--hma-threshold", default=0.4, type=float, help="Hard-routing threshold for HMA scores.")
    parser.add_argument("--iters-sensitive", default=20000, type=int)
    parser.add_argument("--iters-robust", default=5000, type=int)
    parser.add_argument("--infonce-lambda", default=0.1, type=float)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--device", default="cuda", help="CUDA device used for calibration, e.g. cuda or cuda:0.")
    parser.add_argument(
        "--protect-head",
        action="store_true",
        help="Skip classifier heads during reconstruction. Disabled by default to match the hard-routing baseline.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("HMA-DCC calibration currently expects a CUDA device because reconstruction uses CUDA tensors.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if device.index is not None:
        torch.cuda.set_device(device.index)

    print("HMA-DCC ImageNet PTQ")
    print(f"Architecture: {args.arch}")
    print(f"Quantization: W{args.n_bits_w}A{args.n_bits_a}")
    print(f"Calibration samples: {args.num_calibration_samples}")
    print(f"HMA hard-routing threshold: {args.hma_threshold}")
    print(f"Head protection: {'enabled' if args.protect_head else 'disabled'}")

    train_loader, test_loader = build_imagenet_data(
        batch_size=args.batch_size,
        workers=args.workers,
        data_path=args.data_dir,
    )

    cnn = build_model(args.arch)
    load_pretrained_weights(cnn, args.weight_path)
    clean_fp_model = copy.deepcopy(cnn)
    fp_model_base = copy.deepcopy(cnn)
    cnn = cnn.to(device).eval()
    fp_model_base = fp_model_base.to(device).eval()

    wq_params = {"n_bits": args.n_bits_w, "channel_wise": True, "scale_method": "mse"}
    aq_params = {"n_bits": args.n_bits_a, "channel_wise": False, "scale_method": "mse", "leaf_param": True, "prob": 0.5}

    fp_model = QuantModel(model=fp_model_base, weight_quant_params=wq_params, act_quant_params=aq_params, is_fusing=False)
    fp_model.set_quant_state(False, False)

    print("\n[Step 0] Evaluating the FP32 reference model")
    fp32_acc = validate_model(test_loader, fp_model, device)

    qnn = QuantModel(model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.set_first_last_layer_to_8bit()
    qnn.disable_network_output_quantization()

    cali_data, cali_target = get_train_samples(train_loader, num_samples=args.num_calibration_samples)
    set_weight_quantize_params(qnn)

    routing_table = extract_hybrid_metrics(
        qnn,
        fp_model,
        clean_fp_model,
        cali_data,
        cali_target,
        batch_size=args.batch_size,
        threshold=args.hma_threshold,
        protect_head=args.protect_head,
    )

    builtins.GLOBAL_CALIBRATION_FLOPS = 0.0
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()

    print("\n[Step 1] Running HMA-DCC block-wise calibration")
    reconstruct_model(qnn, fp_model, routing_table, cali_data, args)

    wall_clock_time = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    total_flops = getattr(builtins, "GLOBAL_CALIBRATION_FLOPS", 0.0)

    print("\n[Step 2] Evaluating the quantized model")
    qnn.set_quant_state(weight_quant=True, act_quant=True)
    final_acc = validate_model(test_loader, qnn, device)

    acc_drop = fp32_acc - final_acc
    summary = {
        "arch": args.arch,
        "weight_bits": args.n_bits_w,
        "activation_bits": args.n_bits_a,
        "seed": args.seed,
        "num_calibration_samples": args.num_calibration_samples,
        "hma_threshold": args.hma_threshold,
        "iters_sensitive": args.iters_sensitive,
        "iters_robust": args.iters_robust,
        "infonce_lambda": args.infonce_lambda,
        "protect_head": args.protect_head,
        "fp32_top1": fp32_acc,
        "quantized_top1": final_acc,
        "accuracy_drop": acc_drop,
        "calibration_flops": total_flops,
        "peak_memory_mb": peak_memory,
        "wall_time_seconds": wall_clock_time,
    }

    print("\n" + "=" * 72)
    print("HMA-DCC Result")
    print("-" * 72)
    print(f"FP32 Top-1          : {fp32_acc:.2f}%")
    print(f"Quantized Top-1     : {final_acc:.2f}%")
    print(f"Accuracy drop       : {acc_drop:.2f}%")
    print(f"Calibration FLOPs   : {total_flops / 1e12:.4f} TFLOPs")
    print(f"Peak memory         : {peak_memory:.2f} MB")
    print(f"Wall time           : {wall_clock_time / 3600:.2f} h ({wall_clock_time:.2f} s)")
    print("=" * 72 + "\n")

    save_summary(args, summary)


if __name__ == "__main__":
    main()
