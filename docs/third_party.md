# Third-Party Code Notes

This repository includes model definitions and PTQ infrastructure organized from public post-training quantization implementations. The release should preserve third-party license terms and attribution when redistributed.

The HMA-DCC-specific components are:

- Hybrid metric estimation with Hessian trace and perturbation scores.
- Hard block-wise routing based on the hybrid score.
- Dynamic contrastive calibration for routed robust blocks.
- The official anonymous ImageNet reproduction entry.

The base PTQ utilities should not be described as entirely original code.
