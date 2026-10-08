"""Security and contribution-estimation experiment settings."""

PAPER_CELM_WARMUP_ROUNDS = None
PAPER_CELM_WARMUP_FRACTION = 0.05
PAPER_CELM_LM_STEPS = 100
PAPER_CELM_LM_LEARNING_RATE = 1e-2
PAPER_CELM_L2_PENALTY = 1e-3
PAPER_CELM_EMA_BETA = 0.5

# Legacy CELM remains readable for old checkpoints, but new studies should use
# paper_celm so the experiment name states which implementation was evaluated.
CELM_WARMUP_ROUNDS = 3
CELM_WARMUP_FRACTION = 0.05
CELM_LM_STEPS = 25
CELM_LM_LEARNING_RATE = 1e-2
CELM_L2_PENALTY = 1e-3
CELM_EMA_BETA = 0.5

