"""Local, federated, evaluation, and aggregation APIs."""

from .aggregation import FedAdamServer, average_weights
from .evaluation import epoch_eval, epoch_eval_fed, evaluate_client_ensemble_on_loader
from .federated import federated_training
from .fedprox import federated_prox_training
from .local import GradualUnfreezer, epoch_train
from .traditional import federated_training_traditional_ensemble, evaluate_federated_ensemble

__all__ = [
    "FedAdamServer", "GradualUnfreezer", "average_weights", "epoch_eval",
    "epoch_eval_fed", "epoch_train", "evaluate_client_ensemble_on_loader",
    "evaluate_federated_ensemble", "federated_prox_training", "federated_training",
    "federated_training_traditional_ensemble",
]
