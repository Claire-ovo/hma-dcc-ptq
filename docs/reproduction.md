# Reproduction Guide

## Environment

Use Python 3.8 or a compatible Python version supported by your PyTorch build. Install dependencies with:

```bash
pip install -r requirements.txt
```

The calibration code expects CUDA because the reconstruction utilities keep calibration tensors on GPU.

## ImageNet Layout

Prepare ImageNet with the standard `train/` and `val/` folders:

```text
/path/to/imagenet/
  train/
  val/
```

Pass the dataset root with `--data-dir /path/to/imagenet`.

## Weight Path

Pretrained FP32 weights are not included. Pass a compatible checkpoint with `--weight-path`.

## Main Command

```bash
python main_hmadcc.py \
  --arch resnet18 \
  --data-dir /path/to/imagenet \
  --weight-path /path/to/pretrained_weights/resnet18.pth \
  --output-dir outputs/resnet18_w2a2 \
  --n-bits-w 2 \
  --n-bits-a 2
```

The default calibration settings follow the main HMA-DCC entry: `hma_threshold=0.4`, `iters_sensitive=20000`, `iters_robust=5000`, and `infonce_lambda=0.1`.

## Output Summary

The run writes a JSON summary to `--output-dir`. The summary records the architecture, bit-widths, seed, HMA threshold, calibration iterations, InfoNCE lambda, accuracy, calibration FLOPs, peak memory, and wall-clock time.

## CUDA Requirement

Use `--device cuda` or `--device cuda:0`. CPU execution is not supported by the current reconstruction utilities.

## Calibration Cost

HMA-DCC performs block-wise reconstruction and InfoNCE calibration. Runtime and memory depend on the architecture, calibration sample count, batch size, and GPU memory capacity.
