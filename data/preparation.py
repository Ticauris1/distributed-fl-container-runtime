"""Explicit dataset preparation entry points."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets import ImageDataset
from .discovery import find_image_files
from .loaders import tensors_to_flat_numpy


@dataclass
class SourceData:
    source_name: str
    class_labels: list[str]
    num_classes: int
    train_paths: list[str]
    train_labels: np.ndarray
    val_paths: list[str]
    val_labels: np.ndarray
    test_paths: list[str]
    test_labels: np.ndarray
    traditional_data: dict[str, np.ndarray] | None = None

    def deep_data(self) -> dict[str, Any]:
        return {
            "train_paths": self.train_paths,
            "train_labels": self.train_labels,
            "val_paths": self.val_paths,
            "val_labels": self.val_labels,
            "test_paths": self.test_paths,
            "test_labels": self.test_labels,
            "num_classes": self.num_classes,
            "class_labels": self.class_labels,
        }


@dataclass
class PreparedData:
    sources: dict[str, SourceData]
    datasets_to_run: dict[str, str]

    @property
    def all_dataloaders_info(self):
        return {name: source.deep_data() for name, source in self.sources.items()}

    @property
    def all_numpy_data(self):
        return {
            name: source.traditional_data
            for name, source in self.sources.items()
            if source.traditional_data is not None
        }

    @property
    def source_class_labels(self):
        return {name: source.class_labels for name, source in self.sources.items()}


def select_datasets(datasets, run_on_dataset):
    if run_on_dataset == "Both":
        return dict(datasets)
    if run_on_dataset in datasets:
        return {run_on_dataset: datasets[run_on_dataset]}
    return {}


def _limit_geometric_v2(image_paths, string_labels, max_images_per_class):
    class_items = {}
    for image_path, label in zip(image_paths, string_labels):
        class_items.setdefault(label, []).append(image_path)
    limited_paths, limited_labels = [], []
    for label in sorted(class_items):
        original_paths = class_items[label]
        selected_paths = sorted(original_paths, key=lambda path: str(path).lower())[
            :max_images_per_class
        ]
        limited_paths.extend(selected_paths)
        limited_labels.extend([label] * len(selected_paths))
        print(
            f"[Geometric V2] {label}: using {len(selected_paths):,} of "
            f"{len(original_paths):,} images"
        )
    print(f"[Geometric V2] Total selected images: {len(limited_paths):,}")
    return limited_paths, limited_labels


def prepare_source(
    source_name,
    source_path,
    *,
    run_mode,
    pixel_img_size,
    batch_size,
    device,
    random_state,
    geom2_max_images_per_class,
):
    print(f"\n{'=' * 28} PREPARING SOURCE: {source_name} {'=' * 28}")
    image_paths, string_labels = find_image_files(source_path)
    if not image_paths:
        print(f"⚠️ No images found for {source_name}; skipping.")
        return None
    if source_name == "Geometric V2":
        image_paths, string_labels = _limit_geometric_v2(
            image_paths, string_labels, geom2_max_images_per_class
        )
    encoder = LabelEncoder()
    int_labels = encoder.fit_transform(string_labels)
    class_labels = list(encoder.classes_)
    print(f"[{source_name}] Classes: {class_labels}")
    print(f"[{source_name}] Images before splitting: {len(image_paths):,}")
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths,
        int_labels,
        test_size=0.15,
        random_state=random_state,
        stratify=int_labels,
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=0.15 / 0.85,
        random_state=random_state,
        stratify=train_val_labels,
    )
    print(
        f"[{source_name}] Split sizes — Train: {len(train_paths):,}, "
        f"Validation: {len(val_paths):,}, Test: {len(test_paths):,}"
    )
    traditional_data = None
    if run_mode in ("traditional", "both"):
        print(f"[{source_name}] Flattening pixels for traditional machine-learning models.")
        pixel_transform = transforms.Compose(
            [transforms.Resize((pixel_img_size, pixel_img_size)), transforms.ToTensor()]
        )
        X_train, y_train = tensors_to_flat_numpy(
            DataLoader(
                ImageDataset(train_paths, train_labels, transform=pixel_transform),
                batch_size=batch_size,
                num_workers=0,
            ),
            device,
        )
        X_test, y_test = tensors_to_flat_numpy(
            DataLoader(
                ImageDataset(test_paths, test_labels, transform=pixel_transform),
                batch_size=batch_size,
                num_workers=0,
            ),
            device,
        )
        traditional_data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
    else:
        print(f"[{source_name}] Skipping pixel flattening because RUN_MODE='{run_mode}'.")
    return SourceData(
        source_name=source_name,
        class_labels=class_labels,
        num_classes=len(class_labels),
        train_paths=train_paths,
        train_labels=np.asarray(train_labels),
        val_paths=val_paths,
        val_labels=np.asarray(val_labels),
        test_paths=test_paths,
        test_labels=np.asarray(test_labels),
        traditional_data=traditional_data,
    )


def prepare_all_datasets(
    *,
    datasets,
    run_on_dataset,
    run_mode,
    pixel_img_size,
    batch_size,
    device,
    random_state,
    geom2_max_images_per_class,
):
    selected = select_datasets(datasets, run_on_dataset)
    sources = {}
    for source_name, source_path in selected.items():
        source = prepare_source(
            source_name,
            source_path,
            run_mode=run_mode,
            pixel_img_size=pixel_img_size,
            batch_size=batch_size,
            device=device,
            random_state=random_state,
            geom2_max_images_per_class=geom2_max_images_per_class,
        )
        if source is not None:
            sources[source_name] = source
    prepared = PreparedData(sources=sources, datasets_to_run=selected)
    print("Prep summary:")
    print("  numpy_data keys:     ", sorted(prepared.all_numpy_data))
    print("  dl_info keys:        ", sorted(prepared.all_dataloaders_info))
    return prepared
