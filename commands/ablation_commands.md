# Ablation Commands

These commands use the clean public entry `main_hmadcc.py`. They do not copy historical internal ablation scripts.

Read the following fields from each output JSON: `quantized_top1`, `accuracy_drop`, `calibration_flops`, `peak_memory_mb`, `wall_time_seconds`, `hma_threshold`, `iters_sensitive`, `iters_robust`, `infonce_lambda`, and `protect_head`.

## Head Protection

Paper mapping: contrastive calibration and classification-head protection ablation.

Default setting, head protection off:

```bash
python main_hmadcc.py \
  --arch resnet18 \
  --data-dir /path/to/imagenet \
  --weight-path /path/to/weights/resnet18.pth \
  --output-dir outputs/ablation_head_off \
  --n-bits-w 2 --n-bits-a 2 \
  --batch-size 64 --num-calibration-samples 1024 --seed 1005 \
  --hma-threshold 0.4 --iters-sensitive 20000 --iters-robust 5000 \
  --infonce-lambda 0.1 --infonce-tau 0.1 \
  --hessian-samples 20 --recon-weight 0.01 --recon-lr 4e-5 \
  --b-start 20 --b-end 2 --warmup 0.2 --opt-mode mse \
  --input-prob 0.5 --asym --act-quant \
  --workers 4 --device cuda
```

Protected-head variant:

```bash
python main_hmadcc.py \
  --arch resnet18 \
  --data-dir /path/to/imagenet \
  --weight-path /path/to/weights/resnet18.pth \
  --output-dir outputs/ablation_head_on \
  --n-bits-w 2 --n-bits-a 2 \
  --batch-size 64 --num-calibration-samples 1024 --seed 1005 \
  --hma-threshold 0.4 --iters-sensitive 20000 --iters-robust 5000 \
  --infonce-lambda 0.1 --infonce-tau 0.1 \
  --hessian-samples 20 --recon-weight 0.01 --recon-lr 4e-5 \
  --b-start 20 --b-end 2 --warmup 0.2 --opt-mode mse \
  --input-prob 0.5 --asym --act-quant \
  --protect-head \
  --workers 4 --device cuda
```

## Threshold Sensitivity

Paper mapping: routing-threshold sensitivity figure.

Run one command per threshold and compare the output JSON files:

```bash
for threshold in 0.2 0.3 0.4 0.5 0.6 0.8; do
  python main_hmadcc.py \
    --arch resnet18 \
    --data-dir /path/to/imagenet \
    --weight-path /path/to/weights/resnet18.pth \
    --output-dir outputs/threshold_resnet18_${threshold} \
    --n-bits-w 2 --n-bits-a 2 \
    --batch-size 64 --num-calibration-samples 1024 --seed 1005 \
    --hma-threshold "${threshold}" --iters-sensitive 20000 --iters-robust 5000 \
    --infonce-lambda 0.1 --infonce-tau 0.1 \
    --hessian-samples 20 --recon-weight 0.01 --recon-lr 4e-5 \
    --b-start 20 --b-end 2 --warmup 0.2 --opt-mode mse \
    --input-prob 0.5 --asym --act-quant \
    --workers 4 --device cuda
done
```

## Iteration Allocation

Paper mapping: allocation strategy ablation.

Uniform 20k-style allocation through equal branch budgets:

```bash
python main_hmadcc.py \
  --arch resnet18 \
  --data-dir /path/to/imagenet \
  --weight-path /path/to/weights/resnet18.pth \
  --output-dir outputs/ablation_iters_20k_20k \
  --n-bits-w 2 --n-bits-a 2 \
  --batch-size 64 --num-calibration-samples 1024 --seed 1005 \
  --hma-threshold 0.4 --iters-sensitive 20000 --iters-robust 20000 \
  --infonce-lambda 0.1 --infonce-tau 0.1 \
  --hessian-samples 20 --recon-weight 0.01 --recon-lr 4e-5 \
  --b-start 20 --b-end 2 --warmup 0.2 --opt-mode mse \
  --input-prob 0.5 --asym --act-quant \
  --workers 4 --device cuda
```

Uniform 10k-style allocation through equal branch budgets:

```bash
python main_hmadcc.py \
  --arch resnet18 \
  --data-dir /path/to/imagenet \
  --weight-path /path/to/weights/resnet18.pth \
  --output-dir outputs/ablation_iters_10k_10k \
  --n-bits-w 2 --n-bits-a 2 \
  --batch-size 64 --num-calibration-samples 1024 --seed 1005 \
  --hma-threshold 0.4 --iters-sensitive 10000 --iters-robust 10000 \
  --infonce-lambda 0.1 --infonce-tau 0.1 \
  --hessian-samples 20 --recon-weight 0.01 --recon-lr 4e-5 \
  --b-start 20 --b-end 2 --warmup 0.2 --opt-mode mse \
  --input-prob 0.5 --asym --act-quant \
  --workers 4 --device cuda
```

Default dynamic allocation:

```bash
python main_hmadcc.py \
  --arch resnet18 \
  --data-dir /path/to/imagenet \
  --weight-path /path/to/weights/resnet18.pth \
  --output-dir outputs/ablation_iters_default \
  --n-bits-w 2 --n-bits-a 2 \
  --batch-size 64 --num-calibration-samples 1024 --seed 1005 \
  --hma-threshold 0.4 --iters-sensitive 20000 --iters-robust 5000 \
  --infonce-lambda 0.1 --infonce-tau 0.1 \
  --hessian-samples 20 --recon-weight 0.01 --recon-lr 4e-5 \
  --b-start 20 --b-end 2 --warmup 0.2 --opt-mode mse \
  --input-prob 0.5 --asym --act-quant \
  --workers 4 --device cuda
```

## InfoNCE Strength

Paper mapping: contrastive calibration ablation.

No-InfoNCE ablation:

```bash
python main_hmadcc.py \
  --arch resnet18 \
  --data-dir /path/to/imagenet \
  --weight-path /path/to/weights/resnet18.pth \
  --output-dir outputs/ablation_infonce_0 \
  --n-bits-w 2 --n-bits-a 2 \
  --batch-size 64 --num-calibration-samples 1024 --seed 1005 \
  --hma-threshold 0.4 --iters-sensitive 20000 --iters-robust 5000 \
  --infonce-lambda 0 --infonce-tau 0.1 \
  --hessian-samples 20 --recon-weight 0.01 --recon-lr 4e-5 \
  --b-start 20 --b-end 2 --warmup 0.2 --opt-mode mse \
  --input-prob 0.5 --asym --act-quant \
  --workers 4 --device cuda
```

Different InfoNCE strengths:

```bash
for lambda in 0.05 0.1 0.2; do
  python main_hmadcc.py \
    --arch resnet18 \
    --data-dir /path/to/imagenet \
    --weight-path /path/to/weights/resnet18.pth \
    --output-dir outputs/ablation_infonce_${lambda} \
    --n-bits-w 2 --n-bits-a 2 \
    --batch-size 64 --num-calibration-samples 1024 --seed 1005 \
    --hma-threshold 0.4 --iters-sensitive 20000 --iters-robust 5000 \
    --infonce-lambda "${lambda}" --infonce-tau 0.1 \
    --hessian-samples 20 --recon-weight 0.01 --recon-lr 4e-5 \
    --b-start 20 --b-end 2 --warmup 0.2 --opt-mode mse \
    --input-prob 0.5 --asym --act-quant \
    --workers 4 --device cuda
done
```

## Not Covered by the Public Entry

Historical Hessian-only and perturbation-only routing variants were found in internal scripts, but they are not exposed by the cleaned public entry. They should be archived internally if used for final paper tables.
