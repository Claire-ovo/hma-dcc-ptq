#!/usr/bin/env python3
"""Evaluate a repository FP32 checkpoint on an ImageNet validation folder."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main_hmadcc import build_model, load_pretrained_weights
from quant.device import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--weight-path", required=True)
    parser.add_argument("--validation-root", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(args.validation_root, transform)
    if len(dataset) != 50_000 or len(dataset.classes) != 1_000:
        raise ValueError(f"Expected ImageNet-1K val, found {len(dataset)} images and {len(dataset.classes)} classes")
    device = resolve_device(args.device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    model = build_model(args.arch)
    load_pretrained_weights(model, args.weight_path)
    model.to(device).eval()

    correct1 = correct5 = count = 0
    started = time.time()
    with torch.inference_mode():
        for index, (images, targets) in enumerate(loader):
            targets = targets.to(device, non_blocking=True)
            top5 = model(images.to(device, non_blocking=True)).topk(5, 1).indices
            correct1 += top5[:, :1].eq(targets[:, None]).sum().item()
            correct5 += top5.eq(targets[:, None]).sum().item()
            count += targets.numel()
            if index % 50 == 0:
                print(f"[{index}/{len(loader)}] samples={count}", flush=True)

    result = {
        "arch": args.arch,
        "checkpoint": args.weight_path,
        "device": str(device),
        "samples": count,
        "top1": 100.0 * correct1 / count,
        "top5": 100.0 * correct5 / count,
        "seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
