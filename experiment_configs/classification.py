"""Active classification comparison map.

Tuple fields are: model, privacy enabled, run name, contribution weighting,
and federated server method.
"""

COMPARISON_VARIANTS = [
    ("densenet121", False, "densenet121_fedavg_uniform", "uniform", "FedAvg"),
    (
        "densenet121",
        False,
        "densenet121_fedavg_paper_celm",
        "paper_celm",
        "FedAvg",
    ),
]

