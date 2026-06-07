# Environment

## Paper Environment

The paper reports the following environment and evaluation setup:

- Benchmark: ImageNet-1K classification.
- Architectures: ResNet-18, ResNet-50, MobileNetV2, RegNetX-600MF, RegNetX-3.2GF, and MNASNet.
- Calibration data: 1,024 unlabeled ImageNet training images.
- Framework: PyTorch 2.5.1.
- OS: Ubuntu 22.04.4.
- CPU: Intel Core i9-10900X.
- GPU: two NVIDIA GeForce RTX 4090 GPUs.
- Evaluation metric: Top-1 accuracy on the ImageNet validation set.
- Efficiency metrics: calibration FLOPs, peak GPU memory, and wall-clock calibration time.

## Versions Still Required for Internal Archival

The inherited original README only listed a lower bound, `python >= 3.7.13`. The source snapshot and paper text did not provide exact Python and CUDA versions.

- Python: to be filled according to the archival environment.
- CUDA: to be filled according to the archival environment.

Before internal archiving, run the following command in the exact environment used for reproduction:

```bash
pip freeze > requirements-full.txt
```

Store `requirements-full.txt` in the internal archival package, not necessarily in the anonymous GitHub repository.

## Public Requirements File

`requirements.txt` lists the main dependencies needed by the public source tree. Only PyTorch is pinned to the version found in the paper. Other exact package versions should be recovered from the archival environment with `pip freeze`.

## Baseline Result Sources

Baselines mentioned in the paper include AdaRound, BRECQ, QDrop, PD-Quant, and CL-Calib. Results with plus/minus values are reproduced under the same setting as HMA-DCC. Results without plus/minus values are taken from the original papers.
