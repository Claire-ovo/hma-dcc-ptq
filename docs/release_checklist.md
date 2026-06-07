# Anonymous Release Checklist

- No local absolute paths.
- No pretrained weights.
- No datasets.
- No IDE configuration files.
- No personal identity information.
- No generated experiment outputs.
- Only one top-level main entry: `main_hmadcc.py`.
- No historical failed experiment entries in the top-level directory.
- No local package archives.
- No private deployment information.
- `--protect-head` is optional and disabled by default.
- HMA hard routing uses `hybrid_score >= hma_threshold`.
- Historical figure and ablation scripts are documented but not copied.
