# Method Overview

## HMA Phase

Hybrid Metric Awareness estimates block-level sensitivity before reconstruction. Each quantized block is assigned to a calibration branch by a hard routing rule.

## Hessian Trace Estimation

The curvature score is estimated with Hutchinson-style Hessian trace approximation on convolution weights. Blocks with high curvature are treated as more sensitive to quantization.

## Perturbation Score

The perturbation score measures the output discrepancy between full-precision and quantized block outputs on calibration data.

## Hard Routing

For each block, HMA-DCC normalizes the Hessian trace score and perturbation score, then computes:

```text
hybrid_score = max(normalized_hessian, normalized_perturbation)
```

The default routing rule is:

```text
is_sensitive = hybrid_score >= hma_threshold
```

The default threshold is `0.4`.

## Sensitive Branch

Sensitive blocks use numerical reconstruction without InfoNCE. The default iteration count is `20000`.

## Robust Branch with InfoNCE

Robust blocks use reconstruction with InfoNCE calibration. The default iteration count is `5000`, and the default InfoNCE weight is `0.1`.

## Optional Head Protection

The `--protect-head` flag skips `fc` and `classifier` heads during reconstruction. It is disabled by default to match the hard-routing baseline. Users can enable it for the protected-head variant or related ablation.
