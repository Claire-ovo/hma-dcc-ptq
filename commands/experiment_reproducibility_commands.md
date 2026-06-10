# Experiment Reproducibility Commands for HMA-DCC

This document organizes the commands corresponding to the experimental results reported in the paper **Hybrid Metric-Aware Dynamic Contrastive Calibration for Extreme Low-Bit Post-Training Quantization**.

The commands are written with anonymous, portable placeholders. Replace the paths and weight filenames with the local paths used in the reproduction environment.

## 1. Reproducibility requirements covered by this file

This document follows the experiment reproducibility and code archival requirements used in the lab:

1. Provide complete environment and version information.
2. Provide random seeds for all experiments involving randomness.
3. Avoid hard-coded hyperparameters and expose them as command-line arguments.
4. Provide the exact command or command template for each paper table or figure.

## 2. Common environment

The reproduced runs in the paper use the following environment:

```text
Operating system: Ubuntu 22.04.4
CPU: Intel Core i9-10900X
GPU: 2 x NVIDIA GeForce RTX 4090
Deep learning framework: PyTorch 2.5.1
Dataset: ImageNet-1K
Calibration set size: 1,024 unlabeled ImageNet training images unless otherwise specified
Evaluation set: ImageNet validation set
Metric: Top-1 accuracy
Resource metrics: Total FLOPs, peak GPU memory, wall-clock calibration time
```

## 3. Common variables

Use the following environment variables before running the commands.

```bash
export DATA_DIR=/path/to/imagenet
export WEIGHTS_DIR=/path/to/pretrained_weights
export OUT_DIR=outputs/reproduction
mkdir -p ${OUT_DIR}
```

Expected ImageNet layout:

```text
${DATA_DIR}/train/...
${DATA_DIR}/val/...
```

## 4. Common HMA-DCC hyperparameters

Unless otherwise specified, use the following settings:

```text
Calibration samples: 1024
Batch size: 64
Seed for a single representative run: 1005
Seeds for mean and standard deviation: 1005, 1029, 2023, 3048, 4096
Weight bit-width: set by each table or figure
Activation bit-width: set by each table or figure
HMA routing threshold: 0.4
Sensitive block iterations: 20000
Robust block iterations: 5000
InfoNCE lambda: 0.1
InfoNCE temperature: 0.1
Hessian samples: 20
Reconstruction learning rate: 4e-5
Rounding regularization weight: 0.01
Rounding b_start: 20
Rounding b_end: 2
Warmup ratio: 0.2
Optimization mode: mse
Input probability: 0.5
Quantization mode: asymmetric
Activation quantization: enabled
Head protection: optional, disabled unless explicitly enabled
```

The paper defines the sensitive set using the boundary condition:

```text
hybrid_score >= hma_threshold
```

---

# Commands by paper table and figure

## Table I. Top-1 accuracy on ImageNet-1K under different W/A bit-widths

### Target result

Architectures:

```text
resnet18
resnet50
mobilenetv2
regnetx_600mf
regnetx_3200m
mnasnet
```

Bit-width settings:

```text
W4A4
W2A4
W4A2
W2A2
```

Seeds:

```text
1005 1029 2023 3048 4096
```

### HMA-DCC reproduced results

Run all HMA-DCC entries in Table I:

```bash
export ARCHS="resnet18 resnet50 mobilenetv2 regnetx_600mf regnetx_3200m mnasnet"
export BITS="4 4;2 4;4 2;2 2"
export SEEDS="1005 1029 2023 3048 4096"

for arch in ${ARCHS}; do
  for bit_pair in "4 4" "2 4" "4 2" "2 2"; do
    set -- ${bit_pair}
    w_bits=$1
    a_bits=$2
    for seed in ${SEEDS}; do
      python main_hmadcc.py \
        --data-dir ${DATA_DIR} \
        --weight-path ${WEIGHTS_DIR}/${arch}.pth.tar \
        --arch ${arch} \
        --n-bits-w ${w_bits} \
        --n-bits-a ${a_bits} \
        --seed ${seed} \
        --calib-size 1024 \
        --batch-size 64 \
        --hma-threshold 0.4 \
        --iters-sensitive 20000 \
        --iters-robust 5000 \
        --infonce-lambda 0.1 \
        --infonce-tau 0.1 \
        --hessian-samples 20 \
        --recon-lr 4e-5 \
        --recon-weight 0.01 \
        --b-start 20 \
        --b-end 2 \
        --warmup 0.2 \
        --opt-mode mse \
        --input-prob 0.5 \
        --asym \
        --act-quant \
        --output-dir ${OUT_DIR}/table1/hmadcc/${arch}_W${w_bits}A${a_bits}_seed${seed} \
        2>&1 | tee ${OUT_DIR}/table1/hmadcc/${arch}_W${w_bits}A${a_bits}_seed${seed}.log
    done
  done
done
```

---

## Table II. Calibration efficiency and accuracy on ResNet-18 under W2A2

### Target result

Table II reports Total FLOPs, peak VRAM, wall-clock time, Top-1 accuracy, and accuracy drop on ResNet-18 W2A2.

### HMA-DCC command

```bash
python main_hmadcc.py \
  --data-dir ${DATA_DIR} \
  --weight-path ${WEIGHTS_DIR}/resnet18.pth.tar \
  --arch resnet18 \
  --n-bits-w 2 \
  --n-bits-a 2 \
  --seed 1005 \
  --calib-size 1024 \
  --batch-size 64 \
  --hma-threshold 0.4 \
  --iters-sensitive 20000 \
  --iters-robust 5000 \
  --infonce-lambda 0.1 \
  --infonce-tau 0.1 \
  --hessian-samples 20 \
  --recon-lr 4e-5 \
  --recon-weight 0.01 \
  --b-start 20 \
  --b-end 2 \
  --warmup 0.2 \
  --opt-mode mse \
  --input-prob 0.5 \
  --asym \
  --act-quant \
  --output-dir ${OUT_DIR}/table2/hmadcc_resnet18_w2a2 \
  2>&1 | tee ${OUT_DIR}/table2/hmadcc_resnet18_w2a2.log
```

---

## Fig. 2. Hybrid metric visualization across architectures

### Target result

Fig. 2 contains layer-wise trends and decision scatter plots for the normalized Hessian-trace score and normalized physical perturbation score.

Architectures used in the final paper:

```text
resnet18
resnet50
mobilenetv2
mnasnet
```

### Commands

```bash
for arch in resnet18 resnet50 mobilenetv2 mnasnet; do
  CUDA_VISIBLE_DEVICES=0 python main_hmadcc.py \
    --arch ${arch} \
    --data_path ${DATA_DIR} \
    --n_bits_w 2 \
    --n_bits_a 2 \
    --num_samples 1024 \
    --seed 1005 \
    2>&1 | tee ${OUT_DIR}/fig2/${arch}_sensitivity.log
done
```

---

## Table III. Ablation on sensitivity metric type for ResNet-18 W2A2

### Target result

Table III compares three routing metrics on ResNet-18 W2A2:

```text
Pure Hessian
Pure Perturbation
Hybrid Fusion
```

### Commands

Pure Hessian:

```bash
CUDA_VISIBLE_DEVICES=0 python main_hmadcc.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  --metric_type hessian \
  2>&1 | tee ${OUT_DIR}/table3/resnet18_w2a2_metric_hessian.log
```

Pure Perturbation:

```bash
CUDA_VISIBLE_DEVICES=0 python main_hmadcc.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  --metric_type perturb \
  2>&1 | tee ${OUT_DIR}/table3/resnet18_w2a2_metric_perturb.log
```

Hybrid Fusion:

```bash
CUDA_VISIBLE_DEVICES=0 python main_hmadcc.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  --metric_type hybrid \
  2>&1 | tee ${OUT_DIR}/table3/resnet18_w2a2_metric_hybrid.log
```

---

## Table IV. Ablation of guiding metrics across architectures under W2A2

### Target result

Table IV compares metric sources across architectures:

```text
Baseline / Uniform Allocation
Pure Hessian
Pure Perturbation
Hybrid / HMA-DCC
```

Architectures:

```text
resnet18
resnet50
mobilenetv2
mnasnet
```

### Commands

```bash
for arch in resnet18 resnet50 mobilenetv2 mnasnet; do
  for metric in baseline hessian perturb hybrid; do
    CUDA_VISIBLE_DEVICES=0 python main_hmadcc.py \
      --data_path ${DATA_DIR} \
      --arch ${arch} \
      --n_bits_w 2 \
      --n_bits_a 2 \
      --seed 1005 \
      --metric_type ${metric} \
      2>&1 | tee ${OUT_DIR}/table4/${arch}_W2A2_metric_${metric}.log
  done
done
```

---

## Table V. Ablation of allocation strategies for ResNet-18 W2A2

### Target result

Table V compares:

```text
Fixed 20k
Fixed 10k
Dynamic Routing / HMA-DCC
```

### Commands

Fixed 20k:

```bash
CUDA_VISIBLE_DEVICES=0 python main_hmadcc.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  --allocation_type fixed_20k \
  2>&1 | tee ${OUT_DIR}/table5/resnet18_w2a2_fixed_20k.log
```

Fixed 10k:

```bash
CUDA_VISIBLE_DEVICES=0 python main_hmadcc.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  --allocation_type fixed_10k \
  2>&1 | tee ${OUT_DIR}/table5/resnet18_w2a2_fixed_10k.log
```

Dynamic Routing:

```bash
CUDA_VISIBLE_DEVICES=0 python main_hmadcc.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  --metric_type hybrid \
  2>&1 | tee ${OUT_DIR}/table5/resnet18_w2a2_dynamic_routing.log
```

---

## Table VI. Ablation of contrastive calibration and classification-head protection

### Target result

Table VI compares:

```text
No InfoNCE, head protection off
Hard-threshold InfoNCE, head protection off
Hard-threshold InfoNCE, head protection on
```

### Commands

No InfoNCE, head protection off:

```bash
CUDA_VISIBLE_DEVICES=0 python main_hmadcc.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  --infonce_type none \
  --disable_fc_protection \
  2>&1 | tee ${OUT_DIR}/table6/resnet18_w2a2_no_infonce_head_off.log
```

Hard-threshold InfoNCE, head protection off:

```bash
CUDA_VISIBLE_DEVICES=0 python main_hmadcc.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  --infonce_type hard \
  --disable_fc_protection \
  2>&1 | tee ${OUT_DIR}/table6/resnet18_w2a2_hard_head_off.log
```

Hard-threshold InfoNCE, head protection on:

```bash
CUDA_VISIBLE_DEVICES=0 python main_hmadcc.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  --infonce_type hard \
  2>&1 | tee ${OUT_DIR}/table6/resnet18_w2a2_hard_head_on.log
```

---

## Fig. 3. Sensitivity to routing threshold

### Target result

Fig. 3 reports the effect of routing threshold `eta` on Top-1 accuracy and calibration time for ResNet-18 W2A2.

Thresholds:

```text
0.2 0.3 0.4 0.5 0.6 0.8
```

### Sweep command

```bash
CUDA_VISIBLE_DEVICES=0 python main_hmadcc.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  --sweep_thresholds \
  --thresholds 0.2 0.3 0.4 0.5 0.6 0.8 \
  2>&1 | tee ${OUT_DIR}/fig3/resnet18_w2a2_threshold_sweep.log
```

---

## Table VII. Cross-architecture performance and resource consumption under W2A2

### Target result

Table VII reports calibration FLOPs, peak VRAM, time, FP32 accuracy, W2A2 accuracy, and accuracy drop for the following architectures:

```text
resnet50
regnetx_3200m
regnetx_600mf
mnasnet
mobilenetv2
```

### Commands

ResNet-50:

```bash
CUDA_VISIBLE_DEVICES=0 python main_hmadcc.py \
  --data_path ${DATA_DIR} \
  --arch resnet50 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  2>&1 | tee ${OUT_DIR}/table7/resnet50_w2a2.log
```

RegNetX-3.2GF:

```bash
CUDA_VISIBLE_DEVICES=0 python main_hmadcc.py \
  --data_path ${DATA_DIR} \
  --arch regnetx_3200m \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  2>&1 | tee ${OUT_DIR}/table7/regnetx_3200m_w2a2.log
```

RegNetX-600MF:

```bash
CUDA_VISIBLE_DEVICES=0 python main_hmadcc.py \
  --data_path ${DATA_DIR} \
  --arch regnetx_600m \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  2>&1 | tee ${OUT_DIR}/table7/regnetx_600m_w2a2.log
```

MNASNet:

```bash
CUDA_VISIBLE_DEVICES=0 python main_hmadcc.py \
  --data_path ${DATA_DIR} \
  --arch mnasnet \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  2>&1 | tee ${OUT_DIR}/table7/mnasnet_w2a2.log
```

MobileNetV2:

```bash
CUDA_VISIBLE_DEVICES=0 python main_hmadcc.py \
  --data_path ${DATA_DIR} \
  --arch mobilenetv2 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  2>&1 | tee ${OUT_DIR}/table7/mobilenetv2_w2a2.log
```

---

## Table VIII. Data sensitivity analysis on ResNet-18 under W2A2

### Target result

Table VIII reports the effect of calibration set size on ResNet-18 W2A2.

Calibration sample sizes:

```text
1024
512
256
128
```

### Commands

1024 samples:

```bash
CUDA_VISIBLE_DEVICES=0 python main_hmadcc.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  2>&1 | tee ${OUT_DIR}/table8/resnet18_w2a2_samples1024.log
```

512 samples:

```bash
CUDA_VISIBLE_DEVICES=0 python main_hmadcc.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  --num_samples 512 \
  2>&1 |
 tee ${OUT_DIR}/table8/resnet18_w2a2_samples512.log
```

256 samples:

```bash
CUDA_VISIBLE_DEVICES=0 python main_hmadcc.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  --num_samples 256 \
  2>&1 | tee ${OUT_DIR}/table8/resnet18_w2a2_samples256.log
```

128 samples:

```bash
CUDA_VISIBLE_DEVICES=0 python main_hmadcc.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  --num_samples 128 \
  2>&1 | tee ${OUT_DIR}/table8/resnet18_w2a2_samples128.log
```

---

## Fig. 4. t-SNE visualization of feature representations

### Target result

Fig. 4 visualizes feature representations from the FP32 model and the HMA-DCC W2A2 model on a 10-class ImageNet subset.

### Command

```bash
CUDA_VISIBLE_DEVICES=0 python main_hmadcc.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  2>&1 | tee ${OUT_DIR}/fig4/resnet18_w2a2_tsne.log
```

---
