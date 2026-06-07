# Figure Generation Notes

Historical figure-generation scripts were found in the internal source directory, but they are not included in this anonymous release. Several scripts contain local paths, old result constants, generated artifact logic, or exploratory branches.

## Historical Sources Found

- `sensitivity_analyzer.py`: generated `metrics_<arch>_W2A2.csv` and `sensitivity_line_chart_<arch>.pdf`.
- `plot_scatter.py`: read `metrics_<arch>_W2A2.csv` and generated `scatter_decision_<arch>.png`.
- `main_ablation.py --sweep_thresholds`: generated `threshold_sweep_<arch>.csv` and `threshold_sweep_plot_<arch>.pdf`.
- `plot_rescue.py`: cleaned threshold sweep CSV values and regenerated threshold sweep PDFs.
- `plot_pareto.py`: generated `pareto_frontier.pdf` from historical hard-coded result values.
- `analyze_tsne_alpha.py`: generated `alpha_distribution.pdf` and `tsne_comparison.pdf`.

## Public Release Decision

No public figure-generation command is provided here because the sanitized anonymous release does not include the historical CSV files, generated figures, or historical plotting scripts.

For internal archival, preserve sanitized copies of:

- Figure source CSV files.
- Figure-generation scripts.
- Generated PDF and PNG figures.
- Paper figure-to-source-data mapping.

Do not store generated figures or source CSV files in the anonymous repository unless they have been separately checked for privacy and size.
