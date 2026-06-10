# Experiment Reproducibility Commands for HMA-DCC

This document organizes the commands corresponding to the experimental results reported in the paper **Hybrid Metric-Aware Dynamic Contrastive Calibration for Extreme Low-Bit Post-Training Quantization**. It is intended for the `commands/` directory of the code archive.

The commands are written with anonymous, portable placeholders. Replace the paths and weight filenames with the local paths used in the reproduction environment.

## 1. Reproducibility requirements covered by this file

This document follows the experiment reproducibility and code archival requirements used in the lab:

1. Provide complete environment and version information.
2. Provide random seeds for all experiments involving randomness.
3. Avoid hard-coded hyperparameters and expose them as command-line arguments.
4. Provide the exact command or command template for each paper table or figure.
5. Archive intermediate logs and final data used in each table or figure.
6. Keep `README.md`, `requirements.txt`, code, commands, results, figures, and paper files together in the archive.

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
Resource metrics: calibration FLOPs, peak GPU memory, wall-clock calibration time
```

The exact Python and CUDA versions should be archived from the final reproduction environment:

```bash
python --version
nvcc --version
nvidia-smi
pip freeze > requirements.txt
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

## 5. Output fields to archive for every run

For each command, archive the following files:

```text
stdout.log
summary.json
seed.txt or config.json
requirements.txt
nvidia-smi.txt
```

Each `summary.json` should include at least:

```text
method
architecture
dataset
weight_bit
activation_bit
seed
calibration_samples
batch_size
hma_threshold
iters_sensitive
iters_robust
infonce_lambda
infonce_tau
calibration_flops_tflops
peak_vram_mb
wall_clock_time_h
top1_acc
acc_drop
```

For multi-seed tables, also archive the per-seed raw results and the script used to compute mean and standard deviation.

---

# Commands by paper table and figure

## Fig. 1. Method overview

Fig. 1 is a method overview diagram. It is not generated from a numerical experiment and has no runtime command.

Archive requirement:

```text
figures/Fig1_method_overview.pdf or figures/Fig1_method_overview.png
```

---

## Table I. Top-1 accuracy on ImageNet-1K under different W/A bit-widths

### Target result

Table I reports Top-1 accuracy for multiple methods, architectures, and bit-widths. Results with `±` are reproduced under the same setting as HMA-DCC. Results without `±` are taken from the original papers and do not require reproduction commands in this code archive.

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

Extract the reported values from each run:

```text
summary.json -> top1_acc
summary.json -> calibration_flops_tflops
summary.json -> peak_vram_mb
summary.json -> wall_clock_time_h
```

Then compute the mean and standard deviation over the five seeds:

```bash
python tools/aggregate_table1.py \
  --input-root ${OUT_DIR}/table1/hmadcc \
  --output-csv results/final_data/table1_hmadcc_mean_std.csv
```

If `tools/aggregate_table1.py` is not included in the anonymous release, archive the internal script and the resulting CSV under:

```text
results/final_data/table1_hmadcc_mean_std.csv
results/intermediate_data/table1_hmadcc_per_seed/
```

### BRECQ reproduced baseline entries

The BRECQ entries with `±` were reproduced in an internal BRECQ-compatible repository. The command template is:

```bash
export ARCHS="resnet18 resnet50 mobilenetv2 regnetx_600mf regnetx_3200m mnasnet"
export SEEDS="1005 1029 2023 3048 4096"

for arch in ${ARCHS}; do
  for bit_pair in "4 4" "2 4" "4 2" "2 2"; do
    set -- ${bit_pair}
    w_bits=$1
    a_bits=$2
    for seed in ${SEEDS}; do
      CUDA_VISIBLE_DEVICES=0 python main_imagenet.py \
        --data_path ${DATA_DIR} \
        --arch ${arch} \
        --n_bits_w ${w_bits} \
        --n_bits_a ${a_bits} \
        --seed ${seed} \
        --batch_size 64 \
        --iters_w 20000 \
        2>&1 | tee ${OUT_DIR}/table1/brecq/${arch}_W${w_bits}A${a_bits}_seed${seed}.log
    done
  done
done
```

Archive:

```text
results/final_data/table1_brecq_mean_std.csv
results/intermediate_data/table1_brecq_per_seed/
```

### PD-Quant reproduced baseline entries

The PD-Quant entries with `±` were reproduced in an internal PD-Quant-compatible repository. The command template is:

```bash
export ARCHS="resnet18 resnet50 mobilenetv2 regnetx_600mf regnetx_3200m mnasnet"
export SEEDS="1005 1029 2023 3048 4096"

for arch in ${ARCHS}; do
  for bit_pair in "4 4" "2 4" "4 2" "2 2"; do
    set -- ${bit_pair}
    w_bits=$1
    a_bits=$2
    for seed in ${SEEDS}; do
      CUDA_VISIBLE_DEVICES=0 python main_imagenet.py \
        --data_path ${DATA_DIR} \
        --arch ${arch} \
        --n_bits_w ${w_bits} \
        --n_bits_a ${a_bits} \
        --seed ${seed} \
        --batch_size 64 \
        --iters_w 20000 \
        2>&1 | tee ${OUT_DIR}/table1/pdquant/${arch}_W${w_bits}A${a_bits}_seed${seed}.log
    done
  done
done
```

Archive:

```text
results/final_data/table1_pdquant_mean_std.csv
results/intermediate_data/table1_pdquant_per_seed/
```

### CL-Calib reproduced baseline entries

CL-Calib was reproduced on top of the PD-Quant-style codebase because the official CL-Calib implementation was not available. The command template is:

```bash
export ARCHS="resnet18 resnet50 mobilenetv2 regnetx_600mf regnetx_3200m mnasnet"
export SEEDS="1005 1029 2023 3048 4096"

for arch in ${ARCHS}; do
  for bit_pair in "4 4" "2 4" "4 2" "2 2"; do
    set -- ${bit_pair}
    w_bits=$1
    a_bits=$2
    for seed in ${SEEDS}; do
      CUDA_VISIBLE_DEVICES=0 python main_imagenet.py \
        --data_path ${DATA_DIR} \
        --arch ${arch} \
        --n_bits_w ${w_bits} \
        --n_bits_a ${a_bits} \
        --seed ${seed} \
        --batch_size 64 \
        --iters_w 20000 \
        --use_cl_calib \
        --cl_lambda 1.0 \
        --cl_tau 0.1 \
        2>&1 | tee ${OUT_DIR}/table1/clcalib/${arch}_W${w_bits}A${a_bits}_seed${seed}.log
    done
  done
done
```

Archive:

```text
results/final_data/table1_clcalib_mean_std.csv
results/intermediate_data/table1_clcalib_per_seed/
```

---

## Table II. Calibration efficiency and accuracy on ResNet-18 under W2A2

### Target result

Table II reports calibration FLOPs, peak VRAM, wall-clock time, Top-1 accuracy, and accuracy drop on ResNet-18 W2A2.

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

Read the values from:

```text
${OUT_DIR}/table2/hmadcc_resnet18_w2a2/summary.json
```

Fields:

```text
calibration_flops_tflops
peak_vram_mb
wall_clock_time_h
top1_acc
acc_drop
```

### BRECQ baseline command

```bash
CUDA_VISIBLE_DEVICES=0 python main_imagenet.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  --batch_size 64 \
  --iters_w 20000 \
  2>&1 | tee ${OUT_DIR}/table2/brecq_resnet18_w2a2.log
```

### QDrop baseline command

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python qdrop/solver/main_imagenet.py \
  --config exp/w2a4/r18/config.yaml \
  --data_path ${DATA_DIR} \
  --n_bits_w 2 \
  --n_bits_a 2 \
  2>&1 | tee ${OUT_DIR}/table2/qdrop_resnet18_w2a2.log
```

### PD-Quant baseline command

```bash
CUDA_VISIBLE_DEVICES=0 python main_imagenet.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  --batch_size 64 \
  --iters_w 20000 \
  2>&1 | tee ${OUT_DIR}/table2/pdquant_resnet18_w2a2.log
```

### CL-Calib baseline command

```bash
CUDA_VISIBLE_DEVICES=0 python main_imagenet.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  --batch_size 64 \
  --iters_w 20000 \
  --use_cl_calib \
  --cl_lambda 1.0 \
  --cl_tau 0.1 \
  2>&1 | tee ${OUT_DIR}/table2/clcalib_resnet18_w2a2.log
```

### AdaQuant baseline

The AdaQuant row is a baseline row used for comparison. If reproduced internally, archive its command, log, and final CSV under:

```text
results/intermediate_data/table2_adaquant_resnet18_w2a2/
results/final_data/table2_efficiency_resnet18_w2a2.csv
```

If the row is taken from a prior internal result rather than rerun with the current code, mark it explicitly in the internal archive metadata.

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
  CUDA_VISIBLE_DEVICES=0 python sensitivity_analyzer.py \
    --arch ${arch} \
    --data_path ${DATA_DIR} \
    --n_bits_w 2 \
    --n_bits_a 2 \
    --num_samples 1024 \
    --seed 1005 \
    2>&1 | tee ${OUT_DIR}/fig2/${arch}_sensitivity.log
done
```

Expected intermediate outputs:

```text
metrics_resnet18.csv
metrics_resnet50.csv
metrics_mobilenetv2.csv
metrics_mnasnet.csv
```

Expected figure outputs:

```text
sensitivity_line_chart_resnet18.pdf
sensitivity_line_chart_resnet50.pdf
sensitivity_line_chart_mobilenetv2.pdf
sensitivity_line_chart_mnasnet.pdf
scatter_decision_resnet18.png
scatter_decision_resnet50.png
scatter_decision_mobilenetv2.png
scatter_decision_mnasnet.png
Fig2_hybrid_metric_visualization.pdf
```

Archive:

```text
results/intermediate_data/fig2_hybrid_metrics/
results/final_data/fig2_hybrid_metrics_final.csv
figures/Fig2_hybrid_metric_visualization.pdf
figures/figure_generation_scripts/plot_fig2.py
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
CUDA_VISIBLE_DEVICES=0 python main_ablation.py \
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
CUDA_VISIBLE_DEVICES=0 python main_ablation.py \
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
CUDA_VISIBLE_DEVICES=0 python main_ablation.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  --metric_type hybrid \
  2>&1 | tee ${OUT_DIR}/table3/resnet18_w2a2_metric_hybrid.log
```

Expected outputs:

```text
calibration_flops_tflops
peak_vram_mb
wall_clock_time_h
top1_acc
```

Archive:

```text
results/intermediate_data/table3_metric_ablation/
results/final_data/table3_metric_ablation_resnet18_w2a2.csv
```

Note: The final paper value for the Pure Hessian row should be verified against the final saved log or CSV before internal archival submission.

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
    CUDA_VISIBLE_DEVICES=0 python main_ablation.py \
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

If `main_ablation.py` does not support `--metric_type baseline`, use the equivalent uniform-allocation flag from the internal code, or record the exact command in the final internal archive.

Expected output:

```text
Top-1 accuracy for each architecture and metric source
```

Archive:

```text
results/intermediate_data/table4_cross_arch_metric_ablation/
results/final_data/table4_cross_arch_metric_ablation.csv
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
CUDA_VISIBLE_DEVICES=0 python main_ablation.py \
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
CUDA_VISIBLE_DEVICES=0 python main_ablation.py \
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
CUDA_VISIBLE_DEVICES=0 python main_ablation.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  --metric_type hybrid \
  2>&1 | tee ${OUT_DIR}/table5/resnet18_w2a2_dynamic_routing.log
```

Anonymous-release equivalent for Dynamic Routing:

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
  --output-dir ${OUT_DIR}/table5/resnet18_w2a2_dynamic_routing_release
```

Archive:

```text
results/intermediate_data/table5_allocation_ablation/
results/final_data/table5_allocation_ablation_resnet18_w2a2.csv
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
CUDA_VISIBLE_DEVICES=0 python main_ablation.py \
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
CUDA_VISIBLE_DEVICES=0 python main_ablation.py \
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
CUDA_VISIBLE_DEVICES=0 python main_ablation.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  --infonce_type hard \
  2>&1 | tee ${OUT_DIR}/table6/resnet18_w2a2_hard_head_on.log
```

Anonymous-release equivalent for head protection on:

```bash
python main_hmadcc.py \
  --data-dir ${DATA_DIR} \
  --weight-path ${WEIGHTS_DIR}/resnet18.pth.tar \
  --arch resnet18 \
  --n-bits-w 2 \
  --n-bits-a 2 \
  --seed 1005 \
  --hma-threshold 0.4 \
  --iters-sensitive 20000 \
  --iters-robust 5000 \
  --infonce-lambda 0.1 \
  --infonce-tau 0.1 \
  --protect-head \
  --output-dir ${OUT_DIR}/table6/resnet18_w2a2_hard_head_on_release
```

Archive:

```text
results/intermediate_data/table6_infonce_head_ablation/
results/final_data/table6_infonce_head_ablation_resnet18_w2a2.csv
```

Note: The final paper row for hard-threshold InfoNCE with head protection should be checked against the final saved log or threshold-sweep CSV before internal archival submission.

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
CUDA_VISIBLE_DEVICES=0 python main_ablation.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  --sweep_thresholds \
  2>&1 | tee ${OUT_DIR}/fig3/resnet18_w2a2_threshold_sweep.log
```

If the script requires explicit thresholds, use:

```bash
CUDA_VISIBLE_DEVICES=0 python main_ablation.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  --sweep_thresholds \
  --thresholds 0.2 0.3 0.4 0.5 0.6 0.8 \
  2>&1 | tee ${OUT_DIR}/fig3/resnet18_w2a2_threshold_sweep.log
```

### Plot command

```bash
python plot_rescue.py \
  --input-csv threshold_sweep_resnet18.csv \
  --output-pdf figures/Fig3_threshold_sensitivity.pdf
```

If `plot_rescue.py` does not accept CLI arguments in the internal version, run:

```bash
python plot_rescue.py
```

Expected outputs:

```text
threshold_sweep_resnet18.csv
threshold_sweep_plot_resnet18.pdf
Fig3_threshold_sensitivity.pdf
```

Archive:

```text
results/intermediate_data/fig3_threshold_sweep/
results/final_data/fig3_threshold_sweep_resnet18_w2a2.csv
figures/Fig3_threshold_sensitivity.pdf
figures/figure_generation_scripts/plot_fig3.py
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

### Historical internal commands

ResNet-50:

```bash
CUDA_VISIBLE_DEVICES=0 python main_generalization.py \
  --data_path ${DATA_DIR} \
  --arch resnet50 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  2>&1 | tee ${OUT_DIR}/table7/resnet50_w2a2.log
```

RegNetX-3.2GF:

```bash
CUDA_VISIBLE_DEVICES=0 python main_generalization.py \
  --data_path ${DATA_DIR} \
  --arch regnetx_3200m \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  2>&1 | tee ${OUT_DIR}/table7/regnetx_3200m_w2a2.log
```

RegNetX-600MF:

```bash
CUDA_VISIBLE_DEVICES=0 python main_generalization.py \
  --data_path ${DATA_DIR} \
  --arch regnetx_600m \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  2>&1 | tee ${OUT_DIR}/table7/regnetx_600m_w2a2.log
```

MNASNet:

```bash
CUDA_VISIBLE_DEVICES=0 python main_generalization.py \
  --data_path ${DATA_DIR} \
  --arch mnasnet \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  2>&1 | tee ${OUT_DIR}/table7/mnasnet_w2a2.log
```

MobileNetV2:

```bash
CUDA_VISIBLE_DEVICES=0 python main_generalization.py \
  --data_path ${DATA_DIR} \
  --arch mobilenetv2 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  2>&1 | tee ${OUT_DIR}/table7/mobilenetv2_w2a2.log
```

### Anonymous-release equivalent commands

```bash
for arch in resnet50 regnetx_3200m regnetx_600mf mnasnet mobilenetv2; do
  python main_hmadcc.py \
    --data-dir ${DATA_DIR} \
    --weight-path ${WEIGHTS_DIR}/${arch}.pth.tar \
    --arch ${arch} \
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
    --output-dir ${OUT_DIR}/table7/${arch}_w2a2 \
    2>&1 | tee ${OUT_DIR}/table7/${arch}_w2a2.log
done
```

Archive:

```text
results/intermediate_data/table7_cross_arch_w2a2/
results/final_data/table7_cross_arch_w2a2.csv
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

### Historical internal commands

1024 samples:

```bash
CUDA_VISIBLE_DEVICES=0 python main_generalization.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  2>&1 | tee ${OUT_DIR}/table8/resnet18_w2a2_samples1024.log
```

512 samples:

```bash
CUDA_VISIBLE_DEVICES=0 python main_generalization.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  --num_samples 512 \
  2>&1 | tee ${OUT_DIR}/table8/resnet18_w2a2_samples512.log
```

256 samples:

```bash
CUDA_VISIBLE_DEVICES=0 python main_generalization.py \
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
CUDA_VISIBLE_DEVICES=0 python main_generalization.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  --num_samples 128 \
  2>&1 | tee ${OUT_DIR}/table8/resnet18_w2a2_samples128.log
```

### Anonymous-release equivalent commands

```bash
for samples in 1024 512 256 128; do
  python main_hmadcc.py \
    --data-dir ${DATA_DIR} \
    --weight-path ${WEIGHTS_DIR}/resnet18.pth.tar \
    --arch resnet18 \
    --n-bits-w 2 \
    --n-bits-a 2 \
    --seed 1005 \
    --calib-size ${samples} \
    --batch-size 64 \
    --hma-threshold 0.4 \
    --iters-sensitive 20000 \
    --iters-robust 5000 \
    --infonce-lambda 0.1 \
    --infonce-tau 0.1 \
    --output-dir ${OUT_DIR}/table8/resnet18_w2a2_samples${samples} \
    2>&1 | tee ${OUT_DIR}/table8/resnet18_w2a2_samples${samples}.log
done
```

Archive:

```text
results/intermediate_data/table8_data_sensitivity/
results/final_data/table8_data_sensitivity_resnet18_w2a2.csv
```

---

## Fig. 4. t-SNE visualization of feature representations

### Target result

Fig. 4 visualizes feature representations from the FP32 model and the HMA-DCC W2A2 model on a 10-class ImageNet subset.

### Command

```bash
CUDA_VISIBLE_DEVICES=0 python analyze_tsne_alpha.py \
  --data_path ${DATA_DIR} \
  --arch resnet18 \
  --n_bits_w 2 \
  --n_bits_a 2 \
  --seed 1005 \
  2>&1 | tee ${OUT_DIR}/fig4/resnet18_w2a2_tsne.log
```

Expected outputs:

```text
tsne_comparison.pdf
alpha_distribution.pdf
```

The paper uses:

```text
tsne_comparison.pdf
```

Archive:

```text
results/intermediate_data/fig4_tsne/
results/final_data/fig4_tsne_feature_data.csv
figures/Fig4_tsne_comparison.pdf
figures/figure_generation_scripts/analyze_tsne_alpha.py
```

---

# Final archival checklist

Before submitting the archive, check the following items.

## Code and environment

```text
README.md exists.
requirements.txt exists and is generated from the final environment.
All important hyperparameters are exposed through CLI arguments.
No dataset path, server path, user name, author name, or private IP is hard-coded in source code.
All stochastic experiments support --seed.
```

## Commands

```text
This file is saved as commands/experiment_reproducibility_commands.md.
Every table and figure in the paper has a corresponding section.
Every reproduced result has a command or command template.
Copied-from-paper baselines are explicitly marked.
Internal-only baseline commands are archived separately if they are not part of the anonymous release.
```

## Results

```text
Per-run stdout logs are saved.
Per-run summary.json files are saved.
Per-seed raw results are saved for all mean ± std entries.
Final CSV files used to generate tables are saved.
Intermediate CSV files used to generate figures are saved.
Plotting scripts are saved.
The final paper PDF is saved in the paper/ directory for internal archive only.
```

Suggested directory structure:

```text
HuangXinyi_ICDM2026/
├── README.md
├── requirements.txt
├── code/
├── commands/
│   └── experiment_reproducibility_commands.md
├── results/
│   ├── intermediate_data/
│   └── final_data/
├── figures/
│   ├── figure_generation_scripts/
│   ├── Fig1_method_overview.pdf
│   ├── Fig2_hybrid_metric_visualization.pdf
│   ├── Fig3_threshold_sensitivity.pdf
│   └── Fig4_tsne_comparison.pdf
└── paper/
    └── paper.pdf
```
