#!/usr/bin/env python3
"""Convert torchvision MobileNetV2 weights to this repository's flat block layout."""

import argparse
import sys
from pathlib import Path

import torch
from torchvision.models import mobilenet_v2 as torchvision_mobilenetv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.mobilenetv2 import mobilenetv2


def convert_key(key: str) -> str:
    parts = key.split(".")
    if len(parts) < 4 or parts[0] != "features":
        return key
    try:
        block_index = int(parts[1])
    except ValueError:
        return key
    if not 1 <= block_index <= 17 or parts[2] != "conv":
        return key

    tail = parts[3:]
    if block_index == 1:
        if tail[:2] == ["0", "0"]:
            mapped = ["0"] + tail[2:]
        elif tail[:2] == ["0", "1"]:
            mapped = ["1"] + tail[2:]
        elif tail[0] == "1":
            mapped = ["3"] + tail[1:]
        elif tail[0] == "2":
            mapped = ["4"] + tail[1:]
        else:
            raise KeyError(key)
    else:
        if tail[:2] == ["0", "0"]:
            mapped = ["0"] + tail[2:]
        elif tail[:2] == ["0", "1"]:
            mapped = ["1"] + tail[2:]
        elif tail[:2] == ["1", "0"]:
            mapped = ["3"] + tail[2:]
        elif tail[:2] == ["1", "1"]:
            mapped = ["4"] + tail[2:]
        elif tail[0] == "2":
            mapped = ["6"] + tail[1:]
        elif tail[0] == "3":
            mapped = ["7"] + tail[1:]
        else:
            raise KeyError(key)
    return ".".join(parts[:3] + mapped)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = torch.load(args.input, map_location="cpu", weights_only=True)
    converted = {convert_key(key): value for key, value in source.items()}
    local = mobilenetv2().eval()
    missing, unexpected = local.load_state_dict(converted, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Conversion incomplete: missing={missing}, unexpected={unexpected}")

    reference = torchvision_mobilenetv2().eval()
    reference.load_state_dict(source)
    torch.manual_seed(1005)
    sample = torch.randn(2, 3, 96, 96)
    with torch.no_grad():
        reference_logits = reference(sample)
        local_logits = local(sample)
    max_error = (reference_logits - local_logits).abs().max().item()
    mean_error = (reference_logits - local_logits).abs().mean().item()
    if max_error > 1e-4:
        raise RuntimeError(f"Converted model is not numerically equivalent: max_error={max_error}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output)
    print(
        f"Converted {len(converted)} tensors; numerical equivalence "
        f"max_error={max_error:.3e}, mean_error={mean_error:.3e}; saved to {output}"
    )


if __name__ == "__main__":
    main()
