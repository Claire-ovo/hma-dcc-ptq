import os
from pathlib import Path

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def _apply_manifest(dataset, manifest_path: str, dataset_root: str):
    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    entries = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not entries:
        raise ValueError(f"Manifest is empty: {manifest}")

    by_path = {str(Path(path).resolve()): (path, target) for path, target in dataset.samples}
    by_name = {}
    for path, target in dataset.samples:
        name = Path(path).name
        if name in by_name:
            raise ValueError(f"Ambiguous image basename in dataset: {name}")
        by_name[name] = (path, target)

    selected = []
    seen = set()
    root = Path(dataset_root)
    for entry in entries:
        candidate = Path(entry)
        key = str((candidate if candidate.is_absolute() else root / candidate).resolve())
        sample = by_path.get(key) or by_name.get(candidate.name)
        if sample is None:
            raise ValueError(f"Manifest entry is not present in {dataset_root}: {entry}")
        if sample[0] in seen:
            raise ValueError(f"Manifest contains a duplicate image: {entry}")
        seen.add(sample[0])
        selected.append(sample)

    dataset.samples = selected
    dataset.imgs = selected
    dataset.targets = [target for _, target in selected]


def build_imagenet_data(
    data_path: str = '',
    input_size: int = 224,
    batch_size: int = 64,
    workers: int = 4,
    train_dir: str = '',
    val_dir: str = '',
    calibration_manifest: str = '',
    validation_manifest: str = '',
):
    print('==> Using Pytorch Dataset')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    traindir = train_dir or os.path.join(data_path, 'train')
    valdir = val_dir or os.path.join(data_path, 'val')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize,
    ]))
    if calibration_manifest:
        _apply_manifest(train_dataset, calibration_manifest, traindir)
    if validation_manifest:
        _apply_manifest(val_dataset, validation_manifest, valdir)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=not calibration_manifest,
        num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    return train_loader, val_loader
