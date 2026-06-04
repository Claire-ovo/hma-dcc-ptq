# Hybrid Metric Aware Dynamic Contrastive Calibration for Extreme Low-Bit Post-Training Quantization

This repository provides an anonymous code release for reproducing the main ImageNet post-training quantization experiments of Hybrid Metric Aware Dynamic Contrastive Calibration (HMA-DCC).

## Method Overview

HMA-DCC targets extreme low-bit post-training quantization. The method estimates block sensitivity with Hybrid Metric Awareness, which combines a Hessian trace score and an output perturbation score. A hard block-wise routing rule assigns each block to one of two calibration branches:

- Sensitive blocks use numerical reconstruction.
- Robust blocks use InfoNCE calibration.

The default public entry uses hard routing with `hybrid_score >= hma_threshold`. Continuous score routing and dynamic lambda scheduling are not enabled by default.
By default, the official entry keeps head protection disabled unless `--protect-head` is provided. Users can enable `--protect-head` to reproduce the protected-head variant or related ablation.

## Repository Structure

```text
main_hmadcc.py          Official ImageNet reproduction entry.
models/                Network definitions used by the PTQ pipeline.
data/                  ImageNet data loader utilities.
quant/                 Quantization, reconstruction, and calibration utilities.
scripts/               Example reproduction commands.
docs/                  Reproduction, method, weight, and release notes.
outputs/               Ignored output directory for generated summaries.
```

## Installation

Create a clean Python environment and install the required packages:

```bash
conda create -n hmadcc python=3.8
conda activate hmadcc
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build that matches your local CUDA driver. The reconstruction code expects CUDA.

## Dataset Preparation

Prepare ImageNet in the standard folder layout:

```text
/path/to/imagenet/
  train/
    class_000/
    class_001/
    ...
  val/
    class_000/
    class_001/
    ...
```

The repository does not include ImageNet. Pass the dataset root with `--data-dir`.

## Pretrained Weights

This repository does not include pretrained checkpoints. Download or prepare the FP32 checkpoint separately and pass it with `--weight-path`.

Supported checkpoint formats include:

- A raw PyTorch state dict.
- A dictionary with a `state_dict` entry.
- A dictionary with a `model` entry.

The loader removes a leading `module.` prefix when present and prints warnings for mismatched checkpoints.

## Quick Start

Example ResNet-18 W2A2 command:

```bash
python main_hmadcc.py \
  --arch resnet18 \
  --data-dir /path/to/imagenet \
  --weight-path /path/to/pretrained_weights/resnet18.pth \
  --output-dir outputs/resnet18_w2a2 \
  --n-bits-w 2 \
  --n-bits-a 2 \
  --batch-size 64 \
  --num-calibration-samples 1024 \
  --seed 1005 \
  --hma-threshold 0.4 \
  --iters-sensitive 20000 \
  --iters-robust 5000 \
  --infonce-lambda 0.1 \
  --workers 4 \
  --device cuda
```

## Reproducing Main Experiments

Example scripts are provided for common W2A2 runs:

```bash
bash scripts/run_resnet18_w2a2.sh
bash scripts/run_mobilenetv2_w2a2.sh
bash scripts/run_regnetx_600mf_w2a2.sh
```

Edit `DATA_DIR`, `WEIGHT_PATH`, and `OUT_DIR` in each script or override them through environment variables.
The `regnetx_600mf` option is an alias for the local `regnetx_600m` constructor.
See `docs/ablations.md` for the current ablation-code audit and reproducible ablation settings.

## Output Files

The main entry saves a JSON summary under `--output-dir`, including the architecture, bit-widths, seed, routing threshold, calibration settings, accuracy, calibration FLOPs, peak memory, and wall-clock time.

## Anonymous Review Notes

This release does not include personal identity information, ImageNet data, pretrained checkpoints, generated experiment outputs, local environment files, or IDE configuration files.

## Third-Party Code Acknowledgment

Parts of the model and PTQ infrastructure are organized from public post-training quantization implementations. HMA-DCC-specific additions are the hybrid metric estimation, hard block-wise routing, and dynamic contrastive calibration pipeline. Third-party licenses and attribution should be preserved when redistributing this code.

## Citation

If this anonymous submission is accepted, citation information will be added in the final public release.
