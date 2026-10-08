"""Federated client partition algorithms."""

import numpy as np


def make_iid_stratified_indices(y, n_clients, seed=42):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    idx_by_class = {c: np.where(y == c)[0] for c in np.unique(y)}
    for class_id in idx_by_class:
        rng.shuffle(idx_by_class[class_id])
    client_idxs = [[] for _ in range(n_clients)]
    for idxs in idx_by_class.values():
        splits = np.array_split(idxs, n_clients)
        for client_id in range(n_clients):
            client_idxs[client_id].extend(splits[client_id].tolist())
    for client_id in range(n_clients):
        rng.shuffle(client_idxs[client_id])
    return [np.array(indices, dtype=int) for indices in client_idxs]


def make_label_skew_dirichlet_indices(y, n_clients, alpha=0.3, seed=42, min_size=4):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    classes = np.unique(y)
    idx_by_class = {c: np.where(y == c)[0] for c in classes}
    for class_id in classes:
        rng.shuffle(idx_by_class[class_id])
    while True:
        client_idxs = [[] for _ in range(n_clients)]
        for class_id in classes:
            idxs = idx_by_class[class_id]
            proportions = rng.dirichlet(alpha * np.ones(n_clients))
            split_sizes = (proportions * len(idxs)).astype(int)
            difference = len(idxs) - np.sum(split_sizes)
            for index in range(abs(difference)):
                split_sizes[index % n_clients] += 1 if difference > 0 else -1
            start = 0
            for client_id in range(n_clients):
                end = start + split_sizes[client_id]
                client_idxs[client_id].extend(idxs[start:end].tolist())
                start = end
        if min(len(indices) for indices in client_idxs) >= min_size:
            for client_id in range(n_clients):
                rng.shuffle(client_idxs[client_id])
            return [np.array(indices, dtype=int) for indices in client_idxs]


def make_quantity_skew_indices(
    y, n_clients, concentration=0.7, seed=42, guarantee_all_classes=True
):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    classes = np.unique(y)
    n_classes = len(classes)
    total_samples = len(y)
    if guarantee_all_classes:
        if total_samples < n_clients * n_classes:
            raise ValueError(
                "Cannot guarantee every class for every client: "
                f"{total_samples} total samples is less than {n_clients} clients × "
                f"{n_classes} classes."
            )
        for class_id in classes:
            class_count = int(np.sum(y == class_id))
            if class_count < n_clients:
                raise ValueError(
                    f"Cannot guarantee class {class_id} for all {n_clients} clients "
                    f"because it has only {class_count} samples."
                )
    base_size = n_classes if guarantee_all_classes else 0
    remaining_samples = total_samples - (n_clients * base_size)
    proportions = rng.dirichlet(concentration * np.ones(n_clients))
    extra_sizes_float = proportions * remaining_samples
    extra_sizes = np.floor(extra_sizes_float).astype(int)
    remainder = remaining_samples - int(extra_sizes.sum())
    if remainder > 0:
        fractional_parts = extra_sizes_float - extra_sizes
        extra_order = np.argsort(-fractional_parts)
        for index in range(remainder):
            extra_sizes[extra_order[index % n_clients]] += 1
    target_sizes = base_size + extra_sizes
    client_idxs = [[] for _ in range(n_clients)]
    remaining_indices = []
    if guarantee_all_classes:
        for class_id in classes:
            class_indices = np.where(y == class_id)[0]
            rng.shuffle(class_indices)
            for client_id in range(n_clients):
                client_idxs[client_id].append(int(class_indices[client_id]))
            remaining_indices.extend(class_indices[n_clients:].tolist())
    else:
        remaining_indices = np.arange(total_samples).tolist()
    rng.shuffle(remaining_indices)
    current_sizes = np.array([len(indices) for indices in client_idxs], dtype=int)
    remaining_capacity = target_sizes - current_sizes
    if np.any(remaining_capacity < 0):
        raise RuntimeError(
            "A client target size is smaller than the required minimum number of classes."
        )
    start = 0
    for client_id in range(n_clients):
        end = start + int(remaining_capacity[client_id])
        client_idxs[client_id].extend(remaining_indices[start:end])
        start = end
    if start != len(remaining_indices):
        raise RuntimeError("Not all remaining samples were assigned to clients.")
    for client_id in range(n_clients):
        rng.shuffle(client_idxs[client_id])
    return [np.asarray(indices, dtype=int) for indices in client_idxs]
