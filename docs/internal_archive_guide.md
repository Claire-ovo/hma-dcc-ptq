# Internal Archive Guide

The anonymous GitHub repository and the internal laboratory archive should be handled separately.

The anonymous repository should not contain names, private paths, server addresses, datasets, pretrained weights, generated result files, or the paper PDF.

For internal archival, create a separate non-anonymous directory outside this repository. Use placeholders in public documentation and replace them only in the private archive:

```text
<name>+<venue>/
  README.md
  requirements.txt
  requirements-full.txt
  code/
  commands/
  results/
  figures/
  paper/
```

Recommended contents:

- `README.md`: environment, seeds, dataset source, checkpoint source, and exact reproduction scope.
- `requirements.txt`: public dependency list.
- `requirements-full.txt`: exact output of `pip freeze > requirements-full.txt`.
- `code/`: sanitized code snapshot used for the final experiments.
- `commands/`: commands for every paper table and figure.
- `results/`: raw logs, JSON summaries, CSV source data, and per-seed outputs.
- `figures/`: generated paper figures and sanitized figure-generation scripts.
- `paper/`: original submitted paper PDF for internal use only.

Do not create a directory with a real name inside this anonymous release.
