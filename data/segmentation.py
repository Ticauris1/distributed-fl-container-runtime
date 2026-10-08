"""Manifest-backed paired image/mask loading for binary segmentation."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_segmentation_manifest(manifest_path, require_approved=True):
    manifest_path = Path(manifest_path).expanduser().resolve()
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Segmentation manifest is empty: {manifest_path}")
    required = {
        "sample_id", "split", "image_path", "mask_path", "is_cracked",
        "source_group_id", "mask_status",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Manifest is missing required fields: {sorted(missing)}")
    approved = {"approved_manual", "approved_zero_from_uncracked_label"}
    if require_approved:
        pending = [row["sample_id"] for row in rows if row["mask_status"] not in approved]
        if pending:
            raise ValueError(
                f"Manifest contains {len(pending)} unapproved masks, e.g. {pending[:5]}"
            )
    return manifest_path, rows


class SegmentationDataset(Dataset):
    """Load paired RGB images and binary masks with synchronized transforms."""

    def __init__(
        self, manifest_path, split, image_size=256, augment=False,
        augmentation_profile=None,
    ):
        self.manifest_path, all_rows = read_segmentation_manifest(manifest_path)
        self.root = self.manifest_path.parent
        self.rows = [row for row in all_rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No segmentation samples found for split={split!r}")
        self.split = split
        self.image_size = int(image_size)
        if augmentation_profile is None:
            augmentation_profile = "baseline" if augment else "none"
        self.augmentation_profile = str(augmentation_profile).strip().lower()
        if self.augmentation_profile not in {"none", "baseline", "moderate", "strong"}:
            raise ValueError(
                f"Unknown augmentation profile: {self.augmentation_profile}"
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = self.root / row["image_path"]
        mask_path = self.root / row["mask_path"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(mask_path) as source:
            mask = source.convert("L")
        size = [self.image_size, self.image_size]
        image = TF.resize(image, size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)
        if self.augmentation_profile != "none":
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if random.random() < 0.5:
                image, mask = TF.vflip(image), TF.vflip(mask)
            rotations = random.randrange(4)
            if rotations:
                angle = 90 * rotations
                image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
        if self.augmentation_profile in {"moderate", "strong"}:
            limit = 10.0 if self.augmentation_profile == "moderate" else 18.0
            scale_range = (0.95, 1.05) if self.augmentation_profile == "moderate" else (0.88, 1.12)
            angle = random.uniform(-limit, limit)
            scale = random.uniform(*scale_range)
            image = TF.affine(
                image, angle, [0, 0], scale, [0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
            )
            mask = TF.affine(
                mask, angle, [0, 0], scale, [0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
            )
            jitter = 0.15 if self.augmentation_profile == "moderate" else 0.25
            image = TF.adjust_brightness(image, random.uniform(1.0 - jitter, 1.0 + jitter))
            image = TF.adjust_contrast(image, random.uniform(1.0 - jitter, 1.0 + jitter))
            image = TF.adjust_saturation(image, random.uniform(1.0 - jitter, 1.0 + jitter))
            blur_probability = 0.15 if self.augmentation_profile == "moderate" else 0.25
            if random.random() < blur_probability:
                image = TF.gaussian_blur(image, kernel_size=[3, 3], sigma=[0.1, 1.0])
        if self.augmentation_profile == "strong":
            top, left, height, width = RandomResizedCrop.get_params(
                image, scale=(0.80, 1.0), ratio=(0.90, 1.10)
            )
            image = TF.resized_crop(
                image, top, left, height, width, size,
                interpolation=InterpolationMode.BILINEAR,
            )
            mask = TF.resized_crop(
                mask, top, left, height, width, size,
                interpolation=InterpolationMode.NEAREST,
            )
        image_tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        if self.augmentation_profile == "strong" and random.random() < 0.30:
            image_tensor = image_tensor + torch.randn_like(image_tensor) * 0.02
        # Ground-truth masks may be stored as JPEGs (for example CCSS-DATA),
        # where compression turns nominal 0/255 pixels into small non-zero
        # background values and near-white foreground values.  A midpoint
        # threshold preserves binary PNG masks while rejecting JPEG artifacts.
        mask_array = (np.asarray(mask, dtype=np.uint8) >= 128).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        return image_tensor, mask_tensor


def _nested_stratified_indices(dataset, fraction, seed):
    """Select a deterministic, class-balanced prefix for learning curves."""
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"train_fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0:
        return list(range(len(dataset)))
    groups = {False: [], True: []}
    for index, row in enumerate(dataset.rows):
        cracked = str(row["is_cracked"]).strip().lower() in {"1", "true", "yes"}
        groups[cracked].append(index)
    target_total = max(1, round(len(dataset) * fraction))
    false_count = round(target_total * len(groups[False]) / len(dataset))
    requested_counts = {False: false_count, True: target_total - false_count}
    selected = []
    for offset, (label, indices) in enumerate(groups.items()):
        ordered = list(indices)
        random.Random(int(seed) + offset).shuffle(ordered)
        count = min(len(ordered), requested_counts[label])
        selected.extend(ordered[:count])
    return sorted(selected)


def build_segmentation_loaders(
    manifest_path, image_size, batch_size, workers=0, seed=42,
    train_fraction=1.0, evaluation_manifest_path=None,
    augmentation_profile="baseline",
):
    train_dataset = SegmentationDataset(
        manifest_path, "train", image_size, True,
        augmentation_profile=augmentation_profile,
    )
    evaluation_manifest_path = evaluation_manifest_path or manifest_path
    val_dataset = SegmentationDataset(
        evaluation_manifest_path, "val", image_size, False
    )
    test_dataset = SegmentationDataset(
        evaluation_manifest_path, "test", image_size, False
    )
    train_indices = _nested_stratified_indices(train_dataset, train_fraction, seed)
    selected_train_dataset = Subset(train_dataset, train_indices)
    generator = torch.Generator().manual_seed(int(seed))
    common = {
        "batch_size": int(batch_size),
        "num_workers": int(workers),
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(selected_train_dataset, shuffle=True, generator=generator, **common),
        DataLoader(val_dataset, shuffle=False, **common),
        DataLoader(test_dataset, shuffle=False, **common),
    )
