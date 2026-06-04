# Pretrained Weights

## Not Included

This anonymous release does not include pretrained weights or checkpoints. Users must provide a compatible FP32 checkpoint through `--weight-path`.

## Expected Format

The loader supports:

- A raw PyTorch state dict.
- A dictionary with a `state_dict` entry.
- A dictionary with a `model` entry.

## Prefix Handling

If checkpoint keys start with `module.`, the prefix is removed before loading.

## Mismatched Checkpoints

Weights are loaded with non-strict key matching to support common checkpoint wrappers. The loader prints warnings when:

- No checkpoint keys match the model.
- A large fraction of model parameters are missing.
- Classifier head parameters are missing.

If these warnings appear, verify that `--arch` and `--weight-path` refer to the same architecture.
