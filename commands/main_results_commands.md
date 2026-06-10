# Main Result Commands

These commands reproduce the HMA-DCC ImageNet-1K W2A2 setting for the six architectures reported in the paper. They correspond to the HMA-DCC rows in the main accuracy table and to the cross-architecture resource table.

Read the following fields from each output JSON: `quantized_top1`, `fp32_top1`, `accuracy_drop`, `calibration_flops`, `peak_memory_mb`, `wall_time_seconds`, and `seed`.

## ResNet-18

Paper mapping: main Top-1 table and W2A2 resource table.

```bash
python main_hmadcc.py \
  --arch resnet18 \
  --data-dir /path/to/imagenet \
  --weight-path /path/to/weights/resnet18.pth \
  --output-dir outputs/resnet18_w2a2 \
  --n-bits-w 2 --n-bits-a 2 \
  --batch-size 64 --num-calibration-samples 1024 --seed 1005 \
  --hma-threshold 0.4 --iters-sensitive 20000 --iters-robust 5000 \
  --infonce-lambda 0.1 --infonce-tau 0.1 \
  --hessian-samples 20 --recon-weight 0.01 --recon-lr 4e-5 \
  --b-start 20 --b-end 2 --warmup 0.2 --opt-mode mse \
  --input-prob 0.5 --asym --act-quant \
  --workers 4 --device cuda
```

## ResNet-50

Paper mapping: main Top-1 table and W2A2 resource table.

```bash
python main_hmadcc.py \
  --arch resnet50 \
  --data-dir /path/to/imagenet \
  --weight-path /path/to/weights/resnet50.pth \
  --output-dir outputs/resnet50_w2a2 \
  --n-bits-w 2 --n-bits-a 2 \
  --batch-size 64 --num-calibration-samples 1024 --seed 1005 \
  --hma-threshold 0.4 --iters-sensitive 20000 --iters-robust 5000 \
  --infonce-lambda 0.1 --infonce-tau 0.1 \
  --hessian-samples 20 --recon-weight 0.01 --recon-lr 4e-5 \
  --b-start 20 --b-end 2 --warmup 0.2 --opt-mode mse \
  --input-prob 0.5 --asym --act-quant \
  --workers 4 --device cuda
```

## MobileNetV2

Paper mapping: main Top-1 table and W2A2 resource table.

```bash
python main_hmadcc.py \
  --arch mobilenetv2 \
  --data-dir /path/to/imagenet \
  --weight-path /path/to/weights/mobilenetv2.pth \
  --output-dir outputs/mobilenetv2_w2a2 \
  --n-bits-w 2 --n-bits-a 2 \
  --batch-size 64 --num-calibration-samples 1024 --seed 1005 \
  --hma-threshold 0.4 --iters-sensitive 20000 --iters-robust 5000 \
  --infonce-lambda 0.1 --infonce-tau 0.1 \
  --hessian-samples 20 --recon-weight 0.01 --recon-lr 4e-5 \
  --b-start 20 --b-end 2 --warmup 0.2 --opt-mode mse \
  --input-prob 0.5 --asym --act-quant \
  --workers 4 --device cuda
```

## RegNetX-600MF

Paper mapping: main Top-1 table and W2A2 resource table. The public registry maps `regnetx_600mf` to the local `regnetx_600m` constructor.

```bash
python main_hmadcc.py \
  --arch regnetx_600mf \
  --data-dir /path/to/imagenet \
  --weight-path /path/to/weights/regnetx_600mf.pth \
  --output-dir outputs/regnetx_600mf_w2a2 \
  --n-bits-w 2 --n-bits-a 2 \
  --batch-size 64 --num-calibration-samples 1024 --seed 1005 \
  --hma-threshold 0.4 --iters-sensitive 20000 --iters-robust 5000 \
  --infonce-lambda 0.1 --infonce-tau 0.1 \
  --hessian-samples 20 --recon-weight 0.01 --recon-lr 4e-5 \
  --b-start 20 --b-end 2 --warmup 0.2 --opt-mode mse \
  --input-prob 0.5 --asym --act-quant \
  --workers 4 --device cuda
```

## RegNetX-3.2GF

Paper mapping: main Top-1 table and W2A2 resource table.

```bash
python main_hmadcc.py \
  --arch regnetx_3200m \
  --data-dir /path/to/imagenet \
  --weight-path /path/to/weights/regnetx_3200m.pth \
  --output-dir outputs/regnetx_3200m_w2a2 \
  --n-bits-w 2 --n-bits-a 2 \
  --batch-size 64 --num-calibration-samples 1024 --seed 1005 \
  --hma-threshold 0.4 --iters-sensitive 20000 --iters-robust 5000 \
  --infonce-lambda 0.1 --infonce-tau 0.1 \
  --hessian-samples 20 --recon-weight 0.01 --recon-lr 4e-5 \
  --b-start 20 --b-end 2 --warmup 0.2 --opt-mode mse \
  --input-prob 0.5 --asym --act-quant \
  --workers 4 --device cuda
```

## MNASNet

Paper mapping: main Top-1 table and W2A2 resource table.

```bash
python main_hmadcc.py \
  --arch mnasnet \
  --data-dir /path/to/imagenet \
  --weight-path /path/to/weights/mnasnet.pth \
  --output-dir outputs/mnasnet_w2a2 \
  --n-bits-w 2 --n-bits-a 2 \
  --batch-size 64 --num-calibration-samples 1024 --seed 1005 \
  --hma-threshold 0.4 --iters-sensitive 20000 --iters-robust 5000 \
  --infonce-lambda 0.1 --infonce-tau 0.1 \
  --hessian-samples 20 --recon-weight 0.01 --recon-lr 4e-5 \
  --b-start 20 --b-end 2 --warmup 0.2 --opt-mode mse \
  --input-prob 0.5 --asym --act-quant \
  --workers 4 --device cuda
```
