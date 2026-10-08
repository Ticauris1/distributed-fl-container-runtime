"""Active centralized and federated segmentation experiment maps.

Edit the ENABLED lists to run a small batch without rewriting the complete,
paper-facing experiment design.
"""

import os
from pathlib import Path

from experiment_setup import (
    CENTRALIZED_SEGMENTATION_RUNS,
    FEDERATED_SEGMENTATION_RUNS,
    SEGMENTATION_AUGMENTATION,
    SEGMENTATION_BATCH_SIZE,
    SEGMENTATION_EARLY_STOPPING_PATIENCE,
    SEGMENTATION_EPOCHS,
    SEGMENTATION_FEDERATED_CLIENT_FRACTION,
    SEGMENTATION_FEDERATED_LEARNING_RATE,
    SEGMENTATION_FEDERATED_LOCAL_EPOCHS,
    SEGMENTATION_FEDERATED_ROUNDS,
    SEGMENTATION_FEDERATED_STUDY_NAME,
    SEGMENTATION_IMAGE_SIZE,
    SEGMENTATION_LEARNING_RATE,
    SEGMENTATION_LOSS,
    SEGMENTATION_POSITIVE_CLASS_WEIGHT,
    SEGMENTATION_SEEDS,
    SEGMENTATION_WEIGHT_DECAY,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAVE_DIR = PROJECT_ROOT / "results"
CCSS_DATASET_ROOT = Path(
    os.environ.get(
        "CCSS_DATASET_ROOT",
        str(PROJECT_ROOT / "data" / "CCSS-DATA-V4-withVal"),
    )
).expanduser()
CCSS_SEGMENTATION_MANIFEST = (
    PROJECT_ROOT / "segmentation_pilot" / "ccss_benchmark_manifest.csv"
)

# Shared centralized defaults: resolution upscaling, matched optimization, and
# validation-only threshold selection across every encoder.
SEGMENTATION_MANIFEST = CCSS_SEGMENTATION_MANIFEST
SEGMENTATION_RESULT_DIR = SAVE_DIR / "segmentation"
SEGMENTATION_MODELS = ["unetplusplus"]
SEGMENTATION_MASK_THRESHOLD = 0.75
SEGMENTATION_ENCODER_WEIGHTS = "imagenet"
SEGMENTATION_NUM_WORKERS = 0
SEGMENTATION_PREVIEW_COUNT = 24
SEGMENTATION_EXPERIMENT_SEEDS = list(SEGMENTATION_SEEDS)
SEGMENTATION_THRESHOLD_RANGE = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90]

SEGMENTATION_STRUCTURED_STUDY_NAME = "ccss_vgg16_seed52_62_60e"
SEGMENTATION_STRUCTURED_TRAIN_MANIFEST = CCSS_SEGMENTATION_MANIFEST
SEGMENTATION_STRUCTURED_EVALUATION_MANIFEST = CCSS_SEGMENTATION_MANIFEST

_CENTRAL_ENCODERS = (
    "densenet121",
    "efficientnet_b3",
    "vgg16",
    "resnet50",
    "mobilenetv2_100",
    "inception_v3",
)

SEGMENTATION_STRUCTURED_VARIANTS = [
    {
        "name": f"unetplusplus_{encoder}_384px",
        "model": "unetplusplus",
        "encoder": encoder,
        "augmentation": SEGMENTATION_AUGMENTATION,
        "learning_rate": SEGMENTATION_LEARNING_RATE,
        "scheduler": "none",
        "loss": SEGMENTATION_LOSS,
        "positive_class_weight": SEGMENTATION_POSITIVE_CLASS_WEIGHT,
        "attention": "none",
        "encoder_freeze_epochs": 0,
        "encoder_learning_rate": SEGMENTATION_LEARNING_RATE,
        "gate_policies": ["no_gate"],
    }
    for encoder in _CENTRAL_ENCODERS
]

# Run one encoder batch at a time. Add names from SEGMENTATION_STRUCTURED_VARIANTS
# as resources allow; the complete map remains fixed for reproducibility.
SEGMENTATION_STRUCTURED_ENABLED_VARIANTS = list(CENTRALIZED_SEGMENTATION_RUNS)

SEGMENTATION_GATE_POLICIES = {"no_gate": {"type": "none"}}

# The final value should be changed only after the three-seed centralized study.
BEST_SEGMENTATION_ENCODER = "efficientnet_b3"
SEGMENTATION_FEDERATED_MANIFEST = CCSS_SEGMENTATION_MANIFEST
SEGMENTATION_FEDERATED_MODEL = "unetplusplus"
SEGMENTATION_FEDERATED_ENCODER = BEST_SEGMENTATION_ENCODER
SEGMENTATION_FEDERATED_AUGMENTATION = SEGMENTATION_AUGMENTATION
SEGMENTATION_FEDERATED_CLIENTS = 2
SEGMENTATION_FEDERATED_SEEDS = list(SEGMENTATION_SEEDS)


def _federated_variant(encoder, clients, distribution, weighting):
    distribution_label = "iid" if distribution == "iid" else "crack_skew"
    return {
        "name": (
            f"{encoder}_clients{clients}_{distribution_label}_"
            f"fedavg_{weighting}_{SEGMENTATION_FEDERATED_ROUNDS}r_"
            f"{SEGMENTATION_FEDERATED_LOCAL_EPOCHS}e"
        ),
        "manifest": CCSS_SEGMENTATION_MANIFEST,
        "evaluation_manifest": CCSS_SEGMENTATION_MANIFEST,
        "encoder": encoder,
        "clients": clients,
        "distribution": distribution,
        "method": "FedAvg",
        "weighting": weighting,
        "rounds": SEGMENTATION_FEDERATED_ROUNDS,
        "local_epochs": SEGMENTATION_FEDERATED_LOCAL_EPOCHS,
        "loss": SEGMENTATION_LOSS,
        "positive_class_weight": SEGMENTATION_POSITIVE_CLASS_WEIGHT,
        "attention": "none",
    }


_MATCHED_ENCODER_VARIANTS = [
    _federated_variant(encoder, 2, "iid", "sample_size")
    for encoder in _CENTRAL_ENCODERS
]

_CLIENT_SCALING_VARIANTS = [
    _federated_variant("efficientnet_b3", clients, distribution, weighting)
    for clients in (3, 4)
    for distribution in ("iid", "crack_area_skew")
    for weighting in ("sample_size", "paper_celm")
]

_FOUR_CLIENT_ENCODER_COMPARISON_VARIANTS = [
    _federated_variant(encoder, 4, distribution, weighting)
    for encoder in ("efficientnet_b3", "vgg16")
    for distribution in ("iid", "crack_area_skew")
    for weighting in ("sample_size", "paper_celm")
]

_CKKS_TWO_CLIENT_VARIANTS = [
    _federated_variant("efficientnet_b3", 2, distribution, weighting)
    for distribution in ("iid", "crack_area_skew")
    for weighting in ("sample_size", "paper_celm")
]

SEGMENTATION_FEDERATED_VARIANTS = list({
    variant["name"]: variant
    for variant in (
        _MATCHED_ENCODER_VARIANTS
        + _CLIENT_SCALING_VARIANTS
        + _FOUR_CLIENT_ENCODER_COMPARISON_VARIANTS
        + _CKKS_TWO_CLIENT_VARIANTS
    )
}.values())

# experiment_setup.py selects the current focused batch from this complete map.
SEGMENTATION_FEDERATED_ENABLED_VARIANTS = list(FEDERATED_SEGMENTATION_RUNS)

SEGMENTATION_FEDADAM_SERVER_LR = 1e-3
SEGMENTATION_FEDADAM_BETA1 = 0.9
SEGMENTATION_FEDADAM_BETA2 = 0.999
SEGMENTATION_FEDADAM_EPS = 1e-8
SEGMENTATION_FEDERATED_CENTRALIZED_SUMMARY = (
    SEGMENTATION_RESULT_DIR
    / "centralized"
    / SEGMENTATION_STRUCTURED_STUDY_NAME
    / "structured_comparison_summary.csv"
)
