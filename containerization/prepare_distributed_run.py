"""
Prepare a distributed FL runtime from your existing config/data logic.

This creates:
- runtime/plan.json
- runtime/manifests/client_1.csv ... client_N.csv
- runtime/manifests/server_val.csv
- runtime/manifests/test.csv

Important:
This script is the bridge from main.py/config.py to containers.
It reads config values and writes file manifests so each client container can
load only its assigned shard.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import os
import sys


CONTAINERIZATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONTAINERIZATION_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np #type: ignore
import pandas as pd #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.preprocessing import LabelEncoder #type: ignore

import config
from data.discovery import find_image_files
from data.partitioning import (
    make_iid_stratified_indices,
    make_label_skew_dirichlet_indices,
    make_quantity_skew_indices,
)


def _get_selected_dataset():
    if config.RUN_ON_DATASET == "Both":
        # Pick the first configured dataset for distributed runs.
        source_name, source_path = next(iter(config.DATASETS.items()))
    else:
        source_name = config.RUN_ON_DATASET
        source_path = config.DATASETS[source_name]
    return source_name, source_path


def _split_train_val_test(paths, labels, random_state):
    """
    Uses the same simple logic expected by the research pipeline:
    train / temp, then val / test.
    These percentages match data.preparation.
    """
    paths = np.asarray(paths)
    labels = np.asarray(labels)

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths,
        labels,
        test_size=0.30,
        random_state=random_state,
        stratify=labels,
        shuffle=True,
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_labels,
        shuffle=True,
    )

    return (
        train_paths.tolist(),
        train_labels.tolist(),
        val_paths.tolist(),
        val_labels.tolist(),
        test_paths.tolist(),
        test_labels.tolist(),
    )


def _make_client_indices(y_train, n_clients, dist, random_state):
    if dist == config.DIST_INDEPENDENT:
        return make_iid_stratified_indices(
            y_train,
            n_clients=n_clients,
            seed=random_state,
        )

    if dist == config.DIST_LABEL_SKEW:
        return make_label_skew_dirichlet_indices(
            y_train,
            n_clients=n_clients,
            alpha=getattr(config, "LABEL_SKEW_ALPHA", 0.3),
            seed=random_state,
            min_size=getattr(config, "MIN_CLIENT_SIZE", 100),
        )

    if dist == config.DIST_QTY_SKEW:
        return make_quantity_skew_indices(
            y_train,
            n_clients=n_clients,
            concentration=getattr(config, "QTY_SKEW_CONCENTRATION", 0.7),
            seed=random_state,
        )

    raise ValueError(f"Unknown distribution: {dist}")


def to_container_path(p):
    """
    Convert host paths created during preparation into paths visible inside Docker.

    Host example:
        /path/to/project/data/augmented_cotton_dataset_v2/0/img.jpg

    Container example:
        /app/data/augmented_cotton_dataset_v2/0/img.jpg
    """
    p = str(p)

    marker = "/data/"
    if marker in p:
        rel = p.split(marker, 1)[1]
        return "/app/data/" + rel

    return p


def write_manifest(path, image_paths, labels):
    container_paths = [to_container_path(p) for p in image_paths]

    df = pd.DataFrame(
        {
            "image_path": container_paths,
            "label": labels,
        }
    )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    runtime_dir = Path(os.getenv("RUNTIME_DIR", str(PROJECT_ROOT / "runtime")))
    manifests_dir = runtime_dir / "manifests"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    source_name, source_path = _get_selected_dataset()
    image_paths, string_labels = find_image_files(source_path)

    if not image_paths:
        raise RuntimeError(f"No images found at {source_path}")

    le = LabelEncoder()
    labels = le.fit_transform(string_labels)
    class_labels = list(le.classes_)
    num_classes = len(class_labels)

    random_state = int(getattr(config, "RANDOM_STATE", 42))
    n_clients = int(getattr(config, "N_CLIENTS", 2))
    if hasattr(config, "FED_CLIENTS"):
        n_clients = int(config.FED_CLIENTS)

    # Override the configured client count through the distributed environment if needed.
    n_clients = int(os.getenv("DISTRIBUTED_NUM_CLIENTS", n_clients))

    dist = os.getenv(
        "DISTRIBUTED_DIST",
        getattr(config, "DIST_LABEL_SKEW", "Non-Identically Distributed (Label Skew)"),
    )

    (
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        test_paths,
        test_labels,
    ) = _split_train_val_test(image_paths, labels, random_state)

    client_splits = _make_client_indices(
        y_train=np.asarray(train_labels),
        n_clients=n_clients,
        dist=dist,
        random_state=random_state,
    )

    for i, split in enumerate(client_splits, start=1):
        split = np.asarray(split, dtype=int)
        write_manifest(
            manifests_dir / f"client_{i}.csv",
            [train_paths[j] for j in split],
            [int(train_labels[j]) for j in split],
        )

    write_manifest(manifests_dir / "server_val.csv", val_paths, [int(x) for x in val_labels])
    write_manifest(manifests_dir / "test.csv", test_paths, [int(x) for x in test_labels])

    model_name = os.getenv("DISTRIBUTED_MODEL", None)
    if model_name is None:
        deep_models = list(getattr(config, "DEEP_MODELS", ["efficientnet_b0"]))
        model_name = deep_models[0]

    client_weight_method = os.getenv(
        "CLIENT_WEIGHT_METHOD", getattr(config, "CLIENT_WEIGHT_METHOD", "sample_size")
    ).lower()
    paper_warmup_rounds = getattr(config, "PAPER_CELM_WARMUP_ROUNDS", None)
    if paper_warmup_rounds is None:
        paper_warmup_rounds = max(
            1,
            math.ceil(
                int(getattr(config, "FED_ROUNDS", 10))
                * float(getattr(config, "PAPER_CELM_WARMUP_FRACTION", 0.05))
            ),
        )
    clients_per_round = int(getattr(config, "FED_CLIENTS_PER_ROUND", n_clients))
    if client_weight_method == "paper_celm":
        clients_per_round = n_clients
    plan = {
        "source_name": source_name,
        "source_path": str(source_path),
        "model_name": model_name,
        "num_classes": num_classes,
        "class_labels": class_labels,
        "n_clients": n_clients,
        "distribution": dist,
        "rounds": int(getattr(config, "FED_ROUNDS", 10)),
        "local_epochs": int(getattr(config, "FED_LOCAL_EPOCHS", 1)),
        "lr": float(getattr(config, "FED_LEARNING_RATE", 1e-3)),
        "clients_per_round": clients_per_round,
        "batch_size": int(getattr(config, "BATCH_SIZE", 16)),
        "server_method": os.getenv("SERVER_METHOD", "FedAvg"),
        "client_weight_method": client_weight_method,
        "celm_warmup_rounds": int(getattr(config, "CELM_WARMUP_ROUNDS", 1)),
        "celm_warmup_fraction": float(getattr(config, "CELM_WARMUP_FRACTION", 0.05)),
        "celm_steps": int(getattr(config, "CELM_LM_STEPS", 25)),
        "celm_lr": float(getattr(config, "CELM_LM_LEARNING_RATE", 1e-2)),
        "celm_l2_penalty": float(getattr(config, "CELM_L2_PENALTY", 1e-3)),
        "celm_ema_beta": float(getattr(config, "CELM_EMA_BETA", 0.5)),
        "paper_celm_warmup_rounds": int(paper_warmup_rounds),
        "paper_celm_warmup_fraction": float(
            getattr(config, "PAPER_CELM_WARMUP_FRACTION", 0.05)
        ),
        "paper_celm_steps": int(getattr(config, "PAPER_CELM_LM_STEPS", 200)),
        "paper_celm_lr": float(
            getattr(config, "PAPER_CELM_LM_LEARNING_RATE", 1e-2)
        ),
        "paper_celm_l2_penalty": float(
            getattr(config, "PAPER_CELM_L2_PENALTY", 1e-3)
        ),
        "paper_celm_ema_beta": float(
            getattr(config, "PAPER_CELM_EMA_BETA", 0.5)
        ),
        "random_state": random_state,
        "train_agg_mode": getattr(config, "TRAIN_AGG_MODE", "weight_avg"),
        "privacy_agg_enabled": bool(getattr(config, "USE_PRIVACY_AWARE_AGGREGATION", False)),
        "runtime_dir": str(runtime_dir),
        "manifests_dir": str(manifests_dir),
    }

    with open(runtime_dir / "plan.json", "w") as f:
        json.dump(plan, f, indent=2)

    print(f"[prepare] wrote {runtime_dir / 'plan.json'}")
    print(f"[prepare] wrote manifests for {n_clients} clients")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
