"""User-facing experiment control panel.

Edit this file for routine experiment selection and training changes. Full
variant construction and validation remain in experiment_configs/.
"""

# Workflow: classification, segmentation_centralized, or segmentation_federated.
RUN_TASK = "segmentation_federated"

# Refresh results/tableau automatically after a completed experiment batch.
TABLEAU_AUTO_REFRESH = True

# Matched replication seeds.
CLASSIFICATION_SEEDS = [42, 52, 62]
SEGMENTATION_SEEDS = [42]

# Classification runs to execute. Names map to experiment_configs/classification.py.
CLASSIFICATION_ENABLED_VARIANTS = [
    "densenet121_fedavg_uniform",
    "densenet121_fedavg_paper_celm",
]

# Shared centralized segmentation training settings.
SEGMENTATION_IMAGE_SIZE = 384
SEGMENTATION_BATCH_SIZE = 2
SEGMENTATION_EPOCHS = 60
SEGMENTATION_LEARNING_RATE = 1e-4
SEGMENTATION_WEIGHT_DECAY = 1e-4
# Match the full 100-epoch centralized budget. The runner still checkpoints
# the best validation epoch, but it will not terminate this study early.
SEGMENTATION_EARLY_STOPPING_PATIENCE = 100
SEGMENTATION_AUGMENTATION = "moderate"
SEGMENTATION_LOSS = "weighted_bce_dice"
SEGMENTATION_POSITIVE_CLASS_WEIGHT = 5.0

# Add or remove names to run encoder comparisons in manageable batches.
CENTRALIZED_SEGMENTATION_RUNS = [
    "unetplusplus_vgg16_384px",
]

# Shared federated segmentation settings.
SEGMENTATION_FEDERATED_STUDY_NAME = (
    "ccss_four_client_paper_celm_iid_skew_30r_2e"
)
SEGMENTATION_FEDERATED_ROUNDS = 30
SEGMENTATION_FEDERATED_LOCAL_EPOCHS = 2
SEGMENTATION_FEDERATED_CLIENT_FRACTION = 1.0
SEGMENTATION_FEDERATED_LEARNING_RATE = 1e-4

# Four-client Paper-CELM comparison matched to the completed sample-size runs.
FEDERATED_SEGMENTATION_RUNS = [
    "vgg16_clients4_iid_fedavg_paper_celm_30r_2e",
    "vgg16_clients4_crack_skew_fedavg_paper_celm_30r_2e",
    "efficientnet_b3_clients4_iid_fedavg_paper_celm_30r_2e",
    "efficientnet_b3_clients4_crack_skew_fedavg_paper_celm_30r_2e",
]
