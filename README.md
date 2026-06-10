# Hybrid Metric Aware Dynamic Contrastive Calibration for Extreme Low-Bit Post-Training Quantization

This anonymous repository contains source code and reproduction commands for Hybrid Metric Aware Dynamic Contrastive Calibration (HMA-DCC).

## Method Overview

HMA-DCC targets extreme low-bit post-training quantization on ImageNet-1K. The method estimates block sensitivity with Hybrid Metric Awareness, which combines a Hessian trace score and an output perturbation score. A hard block-wise routing rule assigns each block to one of two calibration branches:

- Sensitive blocks use numerical reconstruction.
- Robust blocks use InfoNCE calibration.

The default public entry uses hard routing with `hybrid_score >= hma_threshold`. Continuous score routing and dynamic lambda scheduling are not enabled by default.

Head protection is disabled unless `--protect-head` is provided. Users can enable `--protect-head` to reproduce the protected-head variant or related ablation.

## Repository Structure

```text
main_hmadcc.py          Official ImageNet reproduction entry.
models/                Network definitions used by the PTQ pipeline.
data/                  ImageNet data loader utilities.
quant/                 Quantization, reconstruction, and calibration utilities.
scripts/               Example reproduction commands.
commands/              Paper table and figure command notes.
docs/                  Reproduction, environment, archival, and method notes.
outputs/               Ignored output directory for generated summaries.
```

## Environment

The paper setting uses:

- Benchmark: ImageNet-1K classification.
- Architectures: ResNet-18, ResNet-50, MobileNetV2, RegNetX-600MF, RegNetX-3.2GF, and MNASNet.
- Calibration data: 1,024 unlabeled ImageNet training images.
- Framework: PyTorch 2.5.1.
- OS: Ubuntu 22.04.4.
- CPU: Intel Core i9-10900X.
- GPU: two NVIDIA GeForce RTX 4090 GPUs.
- Evaluation metric: Top-1 accuracy on the ImageNet validation set.
- Efficiency metrics: calibration FLOPs, peak GPU memory, and wall-clock calibration time.

The inherited original README only listed `python >= 3.7.13`; the exact archival Python and CUDA versions were not recoverable from the released source snapshot. See `docs/environment.md` for the archival checklist.

## Installation

Create a clean environment and install dependencies:

```bash
conda create -n hmadcc python=<archival-python-version>
conda activate hmadcc
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build that matches the archival CUDA and driver stack. The reconstruction code expects CUDA.

## Dataset and Weights

Prepare ImageNet in the standard folder layout:

```text
/path/to/imagenet/
  train/
  val/
```

This repository does not include ImageNet or pretrained checkpoints. Pass the dataset root with `--data-dir` and a compatible FP32 checkpoint with `--weight-path`.

The pre-trained FP models in our experiment comes from BRECQ, they can be downloaded in [link](https://github.com/yhhhli/BRECQ). And modify the path of the pre-trained model in /path/to/weights.

## Quick Start

Example ResNet-18 W2A2 command:

```bash
python main_hmadcc.py \
  --arch resnet18 \
  --data-dir /path/to/imagenet \
  --weight-path /path/to/weights/resnet18.pth \
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
  --infonce-tau 0.1 \
  --hessian-samples 20 \
  --recon-weight 0.01 \
  --recon-lr 4e-5 \
  --b-start 20 \
  --b-end 2 \
  --warmup 0.2 \
  --opt-mode mse \
  --input-prob 0.5 \
  --asym \
  --act-quant \
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

Edit `DATA_DIR`, `WEIGHT_PATH`, and `OUT_DIR` in each script or override them through environment variables. Optional flags can be passed through `EXTRA_ARGS`; for example:

```bash
EXTRA_ARGS="--protect-head" bash scripts/run_resnet18_w2a2.sh
```

See `commands/main_results_commands.md` for six-architecture commands and `commands/ablation_commands.md` for clean ablation commands.

## Output Files

The main entry saves a JSON summary under `--output-dir`. The summary records the architecture, bit-widths, seed, HMA threshold, calibration settings, Top-1 accuracy, calibration FLOPs, peak memory, and wall-clock time.

## Reproducibility and Archival Notes

This anonymous repository contains source code and commands. Large datasets, pretrained weights, generated figures, generated CSV files, and internal archival results are not included.

Internal archival materials should be stored separately following `docs/internal_archive_guide.md`. Final data for paper tables and figures should follow `docs/results_archive.md`.

Baselines mentioned in the paper include AdaRound, BRECQ, QDrop, PD-Quant, and CL-Calib. Results with plus/minus values are reproduced under the same setting as HMA-DCC. Results without plus/minus values are taken from the original papers.

## Third-Party Code Acknowledgment

Parts of the model and PTQ infrastructure are organized from public post-training quantization implementations. HMA-DCC-specific additions are the hybrid metric estimation, hard block-wise routing, and contrastive calibration pipeline. Third-party licenses and attribution should be preserved when redistributing this code.

## Citation

If this anonymous submission is accepted, citation information will be added in the final public release.
