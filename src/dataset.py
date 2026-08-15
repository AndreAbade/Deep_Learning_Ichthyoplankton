import io
import json
import os
import random
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm


def _worker_init_fn(worker_id: int) -> None:
    # Suppress PIL EXIF warnings in each DataLoader worker process.
    # These warnings are harmless (pixel data loads correctly) but flood
    # stderr and cause I/O contention when num_workers > 1.
    warnings.filterwarnings(
        "ignore",
        message="Corrupt EXIF data",
        category=UserWarning,
        module="PIL",
    )

CLASSES = [
    "Anostomidae",
    "Characiformes",
    "Eggs",
    "Pimelodidae",
    "Prochilodontidae",
    "Siluriformes",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def split_dataset(dataset_dir: Path, data_dir: Path, ratios=(0.70, 0.15, 0.15), seed: int = 42) -> dict:
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1"
    rng = random.Random(seed)

    split_indices = {}
    splits = ("train", "val", "test")

    for cls in CLASSES:
        cls_dir = dataset_dir / cls
        files = sorted(f.name for f in cls_dir.iterdir() if f.suffix.lower() == ".jpg")
        rng.shuffle(files)

        n = len(files)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])

        partitions = {
            "train": files[:n_train],
            "val": files[n_train : n_train + n_val],
            "test": files[n_train + n_val :],
        }
        split_indices[cls] = {s: partitions[s] for s in splits}

        for split in splits:
            dest_dir = data_dir / split / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for fname in partitions[split]:
                link = dest_dir / fname
                if not link.exists():
                    link.symlink_to(cls_dir / fname)

    return split_indices


def save_split_indices(split_indices: dict, log_dir: Path) -> None:
    out = log_dir / "global_results" / "split_indices.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(split_indices, f, indent=2)


def save_class_indices(log_dir: Path) -> None:
    out = log_dir / "global_results" / "class_indices.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
    with open(out, "w") as f:
        json.dump(class_to_idx, f, indent=2)


def get_train_transform(input_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.1, contrast=0.2),
        transforms.RandomResizedCrop(size=input_size, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform(input_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class _RamCachedDataset(Dataset):
    """Reads all JPEG bytes into RAM once, serves from memory every epoch.

    Eliminates repeated disk reads on slow storage (HDD/USB). Uses raw bytes
    (not decoded tensors) so augmentation still applies fresh each epoch.
    Memory cost: ~2–3 MB per image (JPEG), well within the available 107 GB RAM.
    """

    def __init__(self, samples: list, transform, num_threads: int = 8):
        self.samples = samples          # list of (path, class_idx)
        self.transform = transform
        self._byte_cache: list[bytes] = [b""] * len(samples)
        self._load_all(num_threads)

    def _load_one(self, idx: int) -> None:
        path, _ = self.samples[idx]
        try:
            with open(path, "rb") as f:
                self._byte_cache[idx] = f.read()
        except OSError:
            self._byte_cache[idx] = b""

    def _load_all(self, num_threads: int) -> None:
        n = len(self.samples)
        print(f"  [cache] Loading {n} images into RAM with {num_threads} threads…")
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            list(tqdm(pool.map(self._load_one, range(n)), total=n,
                      desc="  preload", leave=False, dynamic_ncols=True))
        total_mb = sum(len(b) for b in self._byte_cache) / 1024 / 1024
        print(f"  [cache] Done — {total_mb:.0f} MB in RAM")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        raw = self._byte_cache[index]
        _, target = self.samples[index]
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            # Fallback for corrupted entry
            img = Image.open(io.BytesIO(self._byte_cache[(index + 1) % len(self)])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target


class _SafeImageFolder(datasets.ImageFolder):
    """ImageFolder that skips corrupted images instead of crashing."""

    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception:
            alt = (index + 1) % len(self)
            return super().__getitem__(alt)


def _make_loader(
    root: Path,
    transform: transforms.Compose,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    ram_cache: bool = True,
) -> DataLoader:
    # Build ImageFolder first to get the samples list and validate class order
    base = _SafeImageFolder(root=str(root), transform=transform)
    assert list(base.classes) == sorted(CLASSES), (
        f"Class mismatch: {base.classes} vs {CLASSES}"
    )
    # Remap to our fixed CLASSES order (alphabetical == our order, but explicit)
    fixed_idx = {cls: i for i, cls in enumerate(CLASSES)}
    samples = [(path, fixed_idx[base.classes[orig_idx]])
               for path, orig_idx in base.samples]

    if ram_cache:
        dataset = _RamCachedDataset(samples, transform, num_threads=num_workers or 4)
    else:
        base.samples = samples
        base.targets = [t for _, t in samples]
        dataset = base

    use_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=_worker_init_fn,
        persistent_workers=use_workers,
        prefetch_factor=2 if use_workers else None,
    )


def get_loaders(
    data_dir: Path,
    batch_size: int,
    input_size: int = 224,
    num_workers: int = 8,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = _make_loader(
        data_dir / "train",
        get_train_transform(input_size),
        batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = _make_loader(
        data_dir / "val",
        get_eval_transform(input_size),
        batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = _make_loader(
        data_dir / "test",
        get_eval_transform(input_size),
        batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


def get_filenames_from_loader(loader: DataLoader) -> list[str]:
    return [Path(p).name for p, _ in loader.dataset.samples]
