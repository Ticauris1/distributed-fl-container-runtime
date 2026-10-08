"""
Reusable helpers that connect your existing research logic to distributed containers.

This file intentionally reuses:
- training.local.epoch_train
- models.factory.make_dl_model
- data.datasets.ImageDataset

The goal is to run the local client-training phase
inside a client container.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from training.local import epoch_train
from data.datasets import ImageDataset
from models.factory import make_dl_model


def get_container_device():
    """
    Docker Desktop on Mac usually cannot expose MPS into Linux containers.
    CUDA may work on Linux/NVIDIA hosts. Otherwise CPU is used.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def default_image_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def load_manifest_loader(
    manifest_path: str | os.PathLike,
    batch_size: int,
    image_size: int = 224,
    shuffle: bool = True,
):
    """
    Manifest CSV format:
        image_path,label
        /app/data/class_x/img001.jpg,0
    """
    df = pd.read_csv(manifest_path)
    if "image_path" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"{manifest_path} must contain columns: image_path,label"
        )

    paths = df["image_path"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    ds = ImageDataset(
        image_paths=paths,
        labels=labels,
        transform=default_image_transform(image_size),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_one_client_round(
    *,
    client_id: int,
    round_id: int,
    model_name: str,
    num_classes: int,
    global_state_path: str,
    client_manifest_path: str,
    output_update_path: str,
    batch_size: int,
    local_epochs: int,
    lr: float,
    image_size: int = 224,
    pretrained: bool = False,
    preserve_local_classifier: bool = False,
    local_classifier_state_path: str | None = None,
) -> Tuple[str, int, float, float]:
    """
    Train one client for one FL round using your current research logic.

    Returns:
        update_path, num_samples, last_loss, last_accuracy
    """
    device = get_container_device()

    loader = load_manifest_loader(
        manifest_path=client_manifest_path,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
    )

    model = make_dl_model(
        model_name=model_name,
        num_classes=num_classes,
        device=device,
        pretrained=pretrained,
    )

    global_state = torch.load(global_state_path, map_location=device)
    classifier_prefix = (
        "head." if hasattr(model, "head")
        else "classifier." if hasattr(model, "classifier")
        else None
    )
    if preserve_local_classifier:
        if classifier_prefix is None:
            raise ValueError(
                "paper_celm requires a model exposing a head or classifier module"
            )
        retained_path = Path(local_classifier_state_path)
        if retained_path.is_file():
            retained = torch.load(retained_path, map_location=device)
            global_state.update(retained)
    model.load_state_dict(global_state, strict=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    last_loss, last_acc = 0.0, 0.0
    for _ in range(local_epochs):
        last_loss, last_acc = epoch_train(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

    output_update_path = str(output_update_path)
    Path(output_update_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_update_path)
    if preserve_local_classifier:
        retained_path = Path(local_classifier_state_path)
        retained_path.parent.mkdir(parents=True, exist_ok=True)
        classifier_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
            if key.startswith(classifier_prefix)
        }
        if not classifier_state:
            raise ValueError(f"No classifier keys found for prefix {classifier_prefix!r}")
        torch.save(classifier_state, retained_path)

    return output_update_path, len(loader.dataset), float(last_loss), float(last_acc)
