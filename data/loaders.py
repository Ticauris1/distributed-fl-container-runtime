"""DataLoader construction and tensor conversion helpers."""

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from config import DEVICE, DIST_INDEPENDENT, DIST_LABEL_SKEW, DIST_QTY_SKEW
from .datasets import FeatureDataset, ImageDataset
from .partitioning import (
    make_iid_stratified_indices,
    make_label_skew_dirichlet_indices,
    make_quantity_skew_indices,
)
from .validation import assert_disjoint, leak_report


def _split_server_val_from_train(
    train_idx, val_frac=0.1, random_state=42, stratify_y=None
):
    if stratify_y is None:
        train, server_val = train_test_split(
            np.array(train_idx),
            test_size=val_frac,
            random_state=random_state,
            shuffle=True,
        )
    else:
        train, server_val = train_test_split(
            np.array(train_idx),
            test_size=val_frac,
            random_state=random_state,
            shuffle=True,
            stratify=np.asarray(stratify_y)[train_idx],
        )
    return train.tolist(), server_val.tolist()


def _make_client_splits_indices(y, n_clients, stratified=True, random_state=42):
    y = np.asarray(y)
    all_indices = np.arange(len(y))
    if stratified:
        counts = np.bincount(y, minlength=int(y.max()) + 1)
        if counts[counts > 0].min() >= n_clients:
            splitter = StratifiedKFold(
                n_splits=n_clients, shuffle=True, random_state=random_state
            )
            splits = [test for _, test in splitter.split(np.zeros(len(y)), y)]
        else:
            print(
                f"⚠️ Not enough samples per class for n_splits={n_clients}; "
                "using random chunks."
            )
            rng = np.random.RandomState(random_state)
            splits = np.array_split(rng.permutation(all_indices), n_clients)
    else:
        splits = np.array_split(all_indices, n_clients)
    return [np.asarray(split, dtype=int).tolist() for split in splits]


def build_fed_loaders_dl(
    train_paths,
    train_labels,
    val_paths,
    val_labels,
    test_paths,
    test_labels,
    train_tf,
    val_tf,
    batch_size,
    num_clients,
    dist=DIST_INDEPENDENT,
    random_state=42,
    label_skew_alpha=0.3,
    qty_skew_concentration=0.7,
    min_client_size=100,
):
    y_train = np.asarray(train_labels)
    if dist == DIST_INDEPENDENT:
        # Thesis-comparison path: reproduce the original project's
        # StratifiedKFold client assignment exactly.
        splits = _make_client_splits_indices(
            y_train,
            n_clients=num_clients,
            stratified=True,
            random_state=random_state,
        )
    elif dist == DIST_LABEL_SKEW:
        splits = make_label_skew_dirichlet_indices(
            y_train, num_clients, label_skew_alpha, random_state, min_client_size
        )
    elif dist == DIST_QTY_SKEW:
        splits = make_quantity_skew_indices(
            y_train, num_clients, qty_skew_concentration, random_state
        )
    else:
        raise ValueError(f"Unknown federated distribution: {dist}")
    splits = [np.asarray(split, dtype=int).tolist() for split in splits]
    client_dataloaders = []
    for split in splits:
        paths = [train_paths[index] for index in split]
        labels = [train_labels[index] for index in split]
        loader = DataLoader(
            ImageDataset(paths, labels, transform=train_tf),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        client_dataloaders.append({"train": loader})
    server_val_loader = DataLoader(
        ImageDataset(val_paths, val_labels, transform=val_tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        ImageDataset(test_paths, test_labels, transform=val_tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    client_sets = [set(train_paths[index] for index in split) for split in splits]
    for left in range(len(client_sets)):
        for right in range(left + 1, len(client_sets)):
            assert_disjoint(
                f"Client {left + 1}", client_sets[left], f"Client {right + 1}", client_sets[right]
            )
        assert_disjoint(f"Client {left + 1}", client_sets[left], "Server-VAL", val_paths)
        assert_disjoint(f"Client {left + 1}", client_sets[left], "TEST", test_paths)
    assert_disjoint("Server-VAL", val_paths, "TEST", test_paths)
    leak_report(
        splits,
        list(range(len(val_paths))),
        list(range(len(test_paths))),
        "(DL indices: TRAIN shards / SERVER-VAL / TEST)",
    )
    ledger = {"client_indices": splits, "server_val_indices": "provided_val"}
    return client_dataloaders, server_val_loader, test_loader, ledger


def build_fed_loaders_trad(
    X_train, y_train, X_test, y_test, batch_size, num_clients, stratified=True, random_state=42
):
    initial_splits = _make_client_splits_indices(
        y_train, num_clients, stratified, random_state
    )
    flat_indices = np.concatenate([np.array(split) for split in initial_splits]).tolist()
    flat_indices, server_val_indices = _split_server_val_from_train(
        flat_indices, 0.1, random_state, np.asarray(y_train)
    )
    flat_indices = np.array(flat_indices)
    sizes = [len(split) for split in initial_splits]
    offsets = np.cumsum([0] + sizes[:-1])
    splits = [
        flat_indices[offset : offset + size].tolist()
        for offset, size in zip(offsets, sizes)
        if size > 0
    ]
    client_dataloaders = []
    for split in splits:
        loader = DataLoader(
            FeatureDataset(np.asarray(X_train)[split], np.asarray(y_train)[split]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        client_dataloaders.append({"train": loader})
    server_val = (
        np.asarray(X_train)[server_val_indices],
        np.asarray(y_train)[server_val_indices],
    )
    for left in range(len(splits)):
        for right in range(left + 1, len(splits)):
            assert_disjoint(f"Client {left + 1}", splits[left], f"Client {right + 1}", splits[right])
        assert_disjoint(f"Client {left + 1}", splits[left], "Server-VAL", server_val_indices)
    leak_report(splits, server_val_indices, np.arange(len(X_test)), "(Traditional indices relative to TRAIN/TEST)")
    return client_dataloaders, server_val


def tensors_to_flat_numpy(loader, device, normalize=True):
    rows, labels = [], []
    with torch.no_grad():
        for images, batch_labels in tqdm(loader, desc="Flattening to numpy", leave=False):
            images = images.to(device, non_blocking=True).float()
            if normalize and images.max() > 1.0:
                images = images / 255.0
            batch_size = images.shape[0]
            rows.append(images.view(batch_size, -1).cpu().numpy().astype(np.float32))
            labels.append(
                batch_labels.cpu().numpy()
                if isinstance(batch_labels, torch.Tensor)
                else np.asarray(batch_labels)
            )
    return np.concatenate(rows, axis=0), np.concatenate(labels, axis=0).astype(np.int64)


def build_client_loaders_manual(
    train_ds,
    y_train,
    n_clients,
    dist,
    batch_size,
    seed,
    label_skew_alpha=0.3,
    qty_skew_concentration=0.7,
    min_client_size=4,
):
    y = np.asarray(y_train)
    if dist == DIST_INDEPENDENT:
        splits = make_iid_stratified_indices(y, n_clients, seed)
    elif dist == DIST_LABEL_SKEW:
        splits = make_label_skew_dirichlet_indices(y, n_clients, label_skew_alpha, seed, min_client_size)
    elif dist == DIST_QTY_SKEW:
        splits = make_quantity_skew_indices(y, n_clients, qty_skew_concentration, seed, True)
    else:
        raise ValueError(f"Unknown dist: {dist}")
    pin_memory = DEVICE.type == "cuda"
    return [
        {
            "train": DataLoader(
                Subset(train_ds, indices),
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=pin_memory,
            )
        }
        for indices in splits
    ]
