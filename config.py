"""Project configuration and explicit runtime initialization."""

import os
from pathlib import Path
import random
import sys
import numpy as np  # type: ignore
import torch  # type: ignore

from experiment_setup import (
    CLASSIFICATION_ENABLED_VARIANTS,
    CLASSIFICATION_SEEDS,
    RUN_TASK,
    TABLEAU_AUTO_REFRESH,
)
from experiment_configs.classification import COMPARISON_VARIANTS
from experiment_configs.security import *  # noqa: F401,F403
from experiment_configs.segmentation import *  # noqa: F401,F403

USE_RANDOM_SEED = False
RANDOM_STATE = int.from_bytes(os.urandom(4), "little") if USE_RANDOM_SEED else 42
EXPERIMENT_SEEDS = list(CLASSIFICATION_SEEDS)
ON_COLAB = "google.colab" in sys.modules
ON_MAC = sys.platform == "darwin"


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
PROJECT_ROOT = Path(__file__).resolve().parent
ORIG_ROOT = PROJECT_ROOT / "data" / "Cotton Disease"
GEOM_ROOT = PROJECT_ROOT / "data" / "augmented_cotton_dataset_v2"
GEOM_ROOT_2 = Path(
    os.environ.get(
        "GEOM_ROOT_2",
        str(PROJECT_ROOT / "data" / "combined_images"),
    )
).expanduser()
GEOM2_MAX_IMAGES_PER_CLASS = 1400
SAVE_DIR = PROJECT_ROOT / "results"
EXP_BASE_DIR = SAVE_DIR / "experiments"
MODEL_SAVE_DIR = PROJECT_ROOT / "saved_models"
CACHE_DIR = PROJECT_ROOT / "sk_cache"
DATASETS = {"Original": ORIG_ROOT, "Geometric": GEOM_ROOT, "Geometric V2": GEOM_ROOT_2}

# Select one focused workflow. Variant maps live in experiment_configs/.
RUN_MODE = "deep"
RUN_ON_DATASET = "Geometric"
RUN_FEDERATED_LEARNING = True
DEEP_MODELS = ["densenet121"]

TRADITIONAL_MODELS = ["SVM", "Naïve Bayes", "KNN", "Random Forest"]

_classification_names = {variant[2] for variant in COMPARISON_VARIANTS}
_unknown_classification_names = (
    set(CLASSIFICATION_ENABLED_VARIANTS) - _classification_names
)
if _unknown_classification_names:
    raise ValueError(
        "Unknown classification experiment names: "
        f"{sorted(_unknown_classification_names)}"
    )
COMPARISON_VARIANTS = [
    variant
    for variant in COMPARISON_VARIANTS
    if variant[2] in set(CLASSIFICATION_ENABLED_VARIANTS)
]

# Classification training.
NUM_EPOCHS = 50
BATCH_SIZE = 64
BASE_LEARNING_RATE = 1e-3
FREEZE_BACKBONE = True
UNFREEZE_SCHEDULE = {10: 1, 20: 2}
FED_ROUNDS = 50
FED_LOCAL_EPOCHS = 2
FED_LEARNING_RATE = 1e-4
FED_SERVER_LEARNING_RATE = 1e-3
FEDPROX_LEARNING_RATE = 1e-4
FEDPROX_MU = 0.01
FED_COLD_START = False
USE_STRATIFIED_CLIENT_SPLITS = True
FED_CLIENTS_PER_ROUND = 2
FED_CLIENTS = 2
LABEL_SKEW_ALPHA = 0.05

DIST_INDEPENDENT = "Independent and Identically Distributed"
DIST_LABEL_SKEW = "Non-Identically Distributed (Label Skew)"
DIST_QTY_SKEW = "Non-Identically Distributed (Quantity Skew)"
DIST_DEFS = {
    DIST_INDEPENDENT: "Each client has data randomly sampled from the same overall distribution.",
    DIST_LABEL_SKEW: "Each client has data from only a subset of labels.",
    DIST_QTY_SKEW: "All clients see all labels, but the amount of data per client varies.",
}
TRAIN_AGG_MODE = "weight_avg"
EVAL_AGG_MODE = "weight_avg"
PREDICTION_WEIGHT_MODE = "data_size"
CLIENT_WEIGHT_METHOD = "sample_size"

# Traditional ML and hybrid inference.
USE_FLATTENED_PIXELS_FOR_TRAD = True
PIXEL_IMG_SIZE = 32
SVM_BACKEND = "kcpa"
KNN_WIDE_GRID = True
PRUNE_KNN_BY_CV = False
SCORING = {"acc": "accuracy", "f1": "f1_macro", "nll": "neg_log_loss"}
REFIT_METRIC = "f1"
GRID_N_JOBS = -1
USE_HYBRID_INFERENCE = False
HYBRID_THRESHOLD_HIGH = 0.60
HYBRID_POSITIVE_CLASS_INDEX = [0, 1]
HYBRID_EVAL_SPLIT = "test"
HYBRID_SAVE_DETAILS = True
USE_BINARY_HYBRID_EVAL = False
HYBRID_POSITIVE_CLASSES = [0, 1]
USE_HYBRID_BAYES_OPT = True
HYBRID_BO_THRESHOLD_MIN = 0.75
HYBRID_BO_THRESHOLD_MAX = 0.995
HYBRID_BO_N_INIT = 10
HYBRID_BO_N_ITER = 40
HYBRID_BO_MAX_ACC_DROP = 0.01
HYBRID_BO_FALLBACK_THRESHOLD = 0.80
LIGHTWEIGHT_MODEL_NAME = "paper_lightweight"
LIGHTWEIGHT_CHECKPOINT_DIR = MODEL_SAVE_DIR / "paper_lightweight"
LIGHTWEIGHT_EPOCHS = 125

# Privacy and optional CKKS aggregation.
USE_PRIVACY_AWARE_AGGREGATION = False
PRIVACY_AGG_TARGET = "classifier_bias"
PRIVACY_AGG_SCALE = 10000
PRIVACY_AGG_MODULUS = 1000003
PRIVACY_AGG_SEED = 123
USE_CKKS_AGGREGATION = False
CKKS_CLASSIFICATION_TARGET = "classifier_bias"
CKKS_SEGMENTATION_TARGET = "segmentation_head"
CKKS_POLY_MODULUS_DEGREE = 8192
CKKS_COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
CKKS_GLOBAL_SCALE = 2**40

EXPERIMENT_CLIENT_COUNTS = [5]
EXPERIMENT_DISTRIBUTIONS = [DIST_LABEL_SKEW]
EXPERIMENT_PARTICIPATION_RATES = [1.0]
EXPERIMENT_METHODS = ["FedAvg"]


def enabled_variants(variants, enabled_names):
    enabled = set(enabled_names)
    known = {variant["name"] for variant in variants}
    unknown = enabled - known
    if unknown:
        raise ValueError(f"Unknown enabled variant names: {sorted(unknown)}")
    selected = [variant for variant in variants if variant["name"] in enabled]
    if not selected:
        raise ValueError("At least one experiment variant must be enabled")
    return selected


def validate_config():
    if USE_PRIVACY_AWARE_AGGREGATION and USE_CKKS_AGGREGATION:
        raise ValueError("Modular masking and CKKS cannot be enabled together")
    if USE_CKKS_AGGREGATION:
        if CKKS_CLASSIFICATION_TARGET not in {"classifier_bias", "classifier_head"}:
            raise ValueError(f"Unsupported CKKS classification target: {CKKS_CLASSIFICATION_TARGET}")
        if CKKS_SEGMENTATION_TARGET not in {"segmentation_head_bias", "segmentation_head"}:
            raise ValueError(f"Unsupported CKKS segmentation target: {CKKS_SEGMENTATION_TARGET}")
        try:
            import tenseal  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "USE_CKKS_AGGREGATION=True requires `pip install tenseal`; "
                "plaintext fallback is disabled"
            ) from error

    supported = {"classification", "segmentation_centralized", "segmentation_federated"}
    if RUN_TASK not in supported:
        raise ValueError(f"Unsupported RUN_TASK: {RUN_TASK}")

    if RUN_TASK.startswith("segmentation_"):
        from models.segmentation import SEGMENTATION_MODEL_BUILDERS, normalize_segmentation_model_name
        if not SEGMENTATION_MANIFEST.is_file():
            raise FileNotFoundError(f"Segmentation manifest not found: {SEGMENTATION_MANIFEST}")
        if min(SEGMENTATION_IMAGE_SIZE, SEGMENTATION_BATCH_SIZE, SEGMENTATION_EPOCHS) <= 0:
            raise ValueError("Segmentation image size, batch size, and epochs must be positive")
        if normalize_segmentation_model_name(SEGMENTATION_FEDERATED_MODEL) not in SEGMENTATION_MODEL_BUILDERS:
            raise ValueError(f"Unsupported segmentation model: {SEGMENTATION_FEDERATED_MODEL}")

    if RUN_TASK == "segmentation_centralized":
        selected = enabled_variants(
            SEGMENTATION_STRUCTURED_VARIANTS, SEGMENTATION_STRUCTURED_ENABLED_VARIANTS
        )
        for variant in selected:
            required = {"name", "model", "encoder", "augmentation", "learning_rate", "gate_policies"}
            missing = required - set(variant)
            if missing:
                raise ValueError(f"Centralized variant missing keys: {sorted(missing)}")

    if RUN_TASK == "segmentation_federated":
        selected = enabled_variants(
            SEGMENTATION_FEDERATED_VARIANTS, SEGMENTATION_FEDERATED_ENABLED_VARIANTS
        )
        for variant in selected:
            required = {"name", "distribution", "method", "weighting"}
            missing = required - set(variant)
            if missing:
                raise ValueError(f"Federated variant missing keys: {sorted(missing)}")
            if variant["distribution"] not in {"iid", "crack_area_skew"}:
                raise ValueError(f"Unsupported segmentation distribution: {variant['distribution']}")
            if variant["weighting"] not in {"sample_size", "uniform", "paper_celm"}:
                raise ValueError(f"Unsupported segmentation weighting: {variant['weighting']}")

    if RUN_MODE not in {"deep", "traditional", "both"}:
        raise ValueError(f"Unsupported RUN_MODE: {RUN_MODE}")
    if RUN_ON_DATASET != "Both" and RUN_ON_DATASET not in DATASETS:
        raise ValueError(f"Unknown RUN_ON_DATASET: {RUN_ON_DATASET}")
    if not EXPERIMENT_SEEDS or len(set(EXPERIMENT_SEEDS)) != len(EXPERIMENT_SEEDS):
        raise ValueError("EXPERIMENT_SEEDS must be a non-empty unique list")


def initialize_runtime():
    validate_config()
    set_seed(RANDOM_STATE)
    for directory in (SAVE_DIR, EXP_BASE_DIR, MODEL_SAVE_DIR, CACHE_DIR, SEGMENTATION_RESULT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def print_runtime_summary():
    print(f"[seed] RANDOM_STATE = {RANDOM_STATE}")
    print("=" * 60)
    print("🚀 CONFIGURATION SUMMARY")
    print(f"  Task: {RUN_TASK.upper()} | Device: {DEVICE.type.upper()}")
    if RUN_TASK == "segmentation_centralized":
        selected = enabled_variants(
            SEGMENTATION_STRUCTURED_VARIANTS, SEGMENTATION_STRUCTURED_ENABLED_VARIANTS
        )
        print(
            f"  Study: {SEGMENTATION_STRUCTURED_STUDY_NAME} | "
            f"Image: {SEGMENTATION_IMAGE_SIZE}x{SEGMENTATION_IMAGE_SIZE} | "
            f"Batch: {SEGMENTATION_BATCH_SIZE} | Epochs: {SEGMENTATION_EPOCHS}"
        )
        print(f"  Seeds: {SEGMENTATION_EXPERIMENT_SEEDS}")
        for variant in selected:
            print(f"    {variant['name']}: {variant['model']}/{variant['encoder']}")
    elif RUN_TASK == "segmentation_federated":
        selected = enabled_variants(
            SEGMENTATION_FEDERATED_VARIANTS, SEGMENTATION_FEDERATED_ENABLED_VARIANTS
        )
        print(
            f"  Study: {SEGMENTATION_FEDERATED_STUDY_NAME} | "
            f"Image: {SEGMENTATION_IMAGE_SIZE}x{SEGMENTATION_IMAGE_SIZE} | "
            f"Seeds: {SEGMENTATION_FEDERATED_SEEDS}"
        )
        print(
            f"  CKKS: {'enabled' if USE_CKKS_AGGREGATION else 'disabled'} | "
            f"Target: {CKKS_SEGMENTATION_TARGET if USE_CKKS_AGGREGATION else 'none'}"
        )
        for variant in selected:
            print(
                f"    {variant['name']}: clients={variant['clients']} | "
                f"distribution={variant['distribution']} | "
                f"server={variant['method']} | weighting={variant['weighting']}"
            )
    else:
        print(f"  Mode: {RUN_MODE.upper()} | Dataset: {RUN_ON_DATASET}")
        print(f"  Deep Models: {', '.join(DEEP_MODELS)}")
        print(f"  Experiment seeds: {EXPERIMENT_SEEDS}")
    print("=" * 60)
