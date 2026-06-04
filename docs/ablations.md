# Ablation Notes

This document records the current ablation-code audit for the anonymous release. Historical ablation entry scripts are not copied into this repository because they contain local paths, non-anonymous logs, `eval()`-based model construction, generated artifact logic, or exploratory variants that are not safe as public entries.

## Reproducible Options in the Public Entry

The official entry keeps the main HMA-DCC configuration unchanged. The following controlled variants can be reproduced through `main_hmadcc.py` without changing the main defaults:

- Protected-head variant: add `--protect-head`.
- Threshold study: change `--hma-threshold` explicitly.
- Calibration budget study: change `--iters-sensitive` or `--iters-robust` explicitly.
- InfoNCE strength study: change `--infonce-lambda` explicitly.

The example scripts support optional arguments through `EXTRA_ARGS`. For example:

```bash
EXTRA_ARGS="--protect-head" bash scripts/run_resnet18_w2a2.sh
```

## Historical Ablation Dimensions

The original experimental workspace included scripts for the following dimensions:

- Metric source: hybrid, Hessian-only, or perturbation-only routing scores.
- Routing threshold sweeps.
- InfoNCE branch variants, including disabled or fixed-weight InfoNCE.
- Protected-head variants.
- Fisher-guided, entropy-guided, and fast local InfoNCE exploratory variants.

Only the first four dimensions are directly related to the HMA-DCC ablation space. The Fisher-guided, entropy-guided, and fast local InfoNCE scripts are treated as exploratory code and are not included in this anonymous release.

## Current Release Decision

No historical ablation script is copied verbatim. A cleaned ablation entry can be added later if needed, but it should reuse the official model registry, checkpoint loading, data-path arguments, and HMA-DCC calibration functions from `main_hmadcc.py`.

The main method remains hard-routed HMA-DCC with:

```text
is_sensitive = hybrid_score >= hma_threshold
```

Continuous score routing and dynamic lambda scheduling are not default methods in this release.
