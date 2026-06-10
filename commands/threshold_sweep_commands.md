# Threshold Sweep Commands

Read these fields from each JSON: `hma_threshold`, `quantized_top1`, `calibration_flops`, `peak_memory_mb`, `wall_time_seconds`, and `seed`.

## ResNet-18 Threshold Sweep

Paper mapping: routing-threshold sensitivity figure.

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

## MobileNetV2 Threshold Sweep

The internal source snapshot also contained `threshold_sweep_mobilenetv2.csv`. Use the same clean command pattern if this figure or appendix result is required.

```bash
for threshold in 0.2 0.3 0.4 0.5 0.6 0.8; do
  python main_hmadcc.py \
    --arch mobilenetv2 \
    --data-dir /path/to/imagenet \
    --weight-path /path/to/weights/mobilenetv2.pth \
    --output-dir outputs/threshold_mobilenetv2_${threshold} \
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

## Internal CSV Reconstruction

For internal archival, collect the JSON outputs into a private CSV with at least:

```text
Threshold,Accuracy,Time_h,Peak_VRAM,FLOPs,Seed
```

Do not include generated CSV or PDF files in the anonymous repository unless they have been separately checked for privacy and size.
