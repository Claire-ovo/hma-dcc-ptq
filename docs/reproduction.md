# Reproduction Guide

## Scope

This guide documents the anonymous public reproduction path for HMA-DCC. It does not include ImageNet, pretrained weights, generated logs, generated CSV files, or generated figures.

## Environment

Use the paper environment when reproducing the reported numbers:

- Benchmark: ImageNet-1K classification.
- Framework: PyTorch 2.5.1.
- OS: Ubuntu 22.04.4.
- CPU: Intel Core i9-10900X.
- GPU: two NVIDIA GeForce RTX 4090 GPUs.
- Original inherited Python lower bound: python >= 3.7.13.
- Python: to be filled according to the archival environment.
- CUDA: to be filled according to the archival environment.

Install dependencies with:

```bash
pip install -r requirements.txt
```

For internal archival, export the exact environment in the machine used for reproduction:

```bash
pip freeze > requirements-full.txt
```

## ImageNet Layout

Prepare ImageNet with the standard `train/` and `val/` folders:

```text
/path/to/imagenet/
  train/
  val/
```

Pass the dataset root with `--data-dir /path/to/imagenet`.

## Weight Path

Pretrained FP32 weights are not included. Pass a compatible checkpoint with `--weight-path /path/to/weights/<arch>.pth`.

## Default Main Setting

The default calibration settings follow the recovered main HMA-DCC entry:

```text
seed=1005
batch_size=64
num_calibration_samples=1024
n_bits_w=2
n_bits_a=2
hma_threshold=0.4
iters_sensitive=20000
iters_robust=5000
infonce_lambda=0.1
infonce_tau=0.1
hessian_samples=20
recon_weight=0.01
recon_lr=4e-5
b_range=(20, 2)
warmup=0.2
opt_mode=mse
input_prob=0.5
asym=True
act_quant=True
```

The hard routing rule is:

```text
is_sensitive = hybrid_score >= hma_threshold
```

## Main Command

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

## Supported Architectures

The public entry supports:

- `resnet18`
- `resnet50`
- `mobilenetv2`
- `regnetx_600mf`
- `regnetx_600m`
- `regnetx_3200m`
- `mnasnet`

`regnetx_600mf` is an alias for the local `regnetx_600m` constructor.

## Output Summary

The run writes a JSON summary to `--output-dir`. Read the following fields for paper reporting:

- `fp32_top1`
- `quantized_top1`
- `accuracy_drop`
- `calibration_flops`
- `peak_memory_mb`
- `wall_time_seconds`
- `seed`
- `hma_threshold`
- `iters_sensitive`
- `iters_robust`
- `infonce_lambda`
- `infonce_tau`
- `hessian_samples`

## Random Seeds

The single-run default seed is `1005`. Historical internal scripts also referenced the seed set `1005, 1029, 2023, 3048, 4096` for repeated runs. If reporting a mean and standard deviation, archive each seed and each individual run result.

## CUDA Requirement

Use `--device cuda` or `--device cuda:0`. CPU execution is not supported by the current reconstruction utilities.
