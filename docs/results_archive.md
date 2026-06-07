# Results Archive

This anonymous repository does not include generated results. Internal archival should preserve all final data and intermediate data needed to reproduce the paper tables and figures.

## Required Internal Artifacts

Save the following artifacts in the private archive:

- Final data for each paper table and figure.
- Seed used for each run.
- Top-1 accuracy for each run.
- Calibration FLOPs.
- Peak GPU memory.
- Wall-clock calibration time.
- Threshold sweep CSV files.
- Metric ablation CSV files.
- Figure source data.
- Generated figures.
- Original paper PDF for internal use only.

## JSON Summary Fields

`main_hmadcc.py` writes one JSON summary per run. Archive at least these fields:

- `arch`
- `weight_bits`
- `activation_bits`
- `seed`
- `num_calibration_samples`
- `hma_threshold`
- `iters_sensitive`
- `iters_robust`
- `infonce_lambda`
- `infonce_tau`
- `hessian_samples`
- `recon_weight`
- `recon_lr`
- `b_range`
- `warmup`
- `opt_mode`
- `input_prob`
- `asymmetric_quantization`
- `activation_quantization`
- `protect_head`
- `fp32_top1`
- `quantized_top1`
- `accuracy_drop`
- `calibration_flops`
- `peak_memory_mb`
- `wall_time_seconds`

## Historical Source Data

The internal source scan found small historical CSV files named `metrics_<arch>_W2A2.csv` and `threshold_sweep_<arch>.csv`, plus generated PDFs and PNGs. These files were not copied into the anonymous repository. If they are used for the final paper, keep sanitized copies in the private `results/` and `figures/` directories.
