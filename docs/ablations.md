# Ablation Notes

This anonymous release does not directly include historical high-risk ablation scripts. The original ablation scripts came from an internal experimental workspace and contain local paths, old log parsing, generated artifact code, and exploratory branches. They are therefore not copied verbatim into this public repository.

## Clean Ablations Supported by `main_hmadcc.py`

The public entry keeps the main HMA-DCC configuration unchanged. The following controlled variants can be reproduced through CLI flags:

- Head protection on or off: use `--protect-head` to enable it. It is off by default.
- Threshold sensitivity: change `--hma-threshold`.
- Robust and sensitive iteration allocation: change `--iters-sensitive` and `--iters-robust`.
- InfoNCE strength: change `--infonce-lambda`.
- No-InfoNCE ablation: set `--infonce-lambda 0`.

The example scripts support optional arguments through `EXTRA_ARGS`. For example:

```bash
EXTRA_ARGS="--protect-head" bash scripts/run_resnet18_w2a2.sh
```

## Historical Source Mapping

The internal source scan found these ablation-related files:

- `main_ablation.py`: threshold sweep and metric-source ablations. In sweep mode it wrote `threshold_sweep_<arch>.csv` and `threshold_sweep_plot_<arch>.pdf`.
- `main_ablation_ptq.py`: metric-source, allocation, InfoNCE, and head-protection ablations, including a continuous InfoNCE option that is not part of the public default method.
- `main_plan_e_ablations.py`: anchor and two-stage exploratory ablations.
- `sensitivity_analyzer.py`: metric CSV and sensitivity line chart generation.
- `plot_scatter.py`, `plot_pareto.py`, `plot_rescue.py`, and `analyze_tsne_alpha.py`: historical figure-generation scripts.

These files are recorded as provenance only. They are not copied into the anonymous release because they are not clean public entries.

## Paper Table and Figure Coverage

The clean public ablation commands cover:

- Table V style allocation comparisons through `--iters-sensitive` and `--iters-robust`.
- Table VI style InfoNCE and head-protection comparisons through `--infonce-lambda` and `--protect-head`.
- Fig. 3 style threshold sensitivity through repeated runs with different `--hma-threshold` values.

Historical figure-generation scripts were not included in the anonymous release. If internal archival requires exact paper figures, store the sanitized historical scripts, source CSV files, generated figures, and paper PDF in the separate internal archive described in `docs/internal_archive_guide.md`.

## Default Routing Rule

The main method remains hard-routed HMA-DCC:

```text
is_sensitive = hybrid_score >= hma_threshold
```

Continuous score routing and dynamic lambda scheduling are not default methods in this release.
