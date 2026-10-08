"""Federated loss and accuracy history plots."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from .common import save_figure


def plot_federated_loss_accuracy(fed_history, client_histories, save_dir, model_name, source_name, method=None, distribution=None, dpi=200):
    title_parts = [model_name, source_name] + ([str(method)] if method else []) + ([str(distribution)] if distribution else [])
    title_suffix = " | ".join(title_parts)
    paths = []
    for metric in ("loss", "accuracy"):
        figure, axis = plt.subplots(figsize=(8, 5), dpi=dpi)
        server_values = fed_history.get(metric, [])
        if server_values:
            axis.plot(np.arange(1, len(server_values) + 1), server_values, marker="o", linewidth=2.2, label=f"Server {metric.title()}")
        for client_id, history in (client_histories or {}).items():
            values = history.get(metric, [])
            if values:
                axis.plot(np.arange(1, len(values) + 1), values, "--", marker=".", alpha=0.75, label=f"Client {client_id + 1} {metric.title()}")
        axis.set_title(f"Federated {metric.title()}\n{title_suffix}")
        axis.set_xlabel("Communication Round")
        axis.set_ylabel(metric.title())
        if metric == "accuracy": axis.set_ylim(0, 1.05)
        axis.grid(True, linestyle=":", alpha=0.6)
        axis.legend(frameon=True)
        path = Path(save_dir) / f"{source_name}_{model_name}_{method or 'fed'}_{metric}_curve.png"
        paths.append(save_figure(figure, path, dpi))
    return tuple(paths)


def plot_federated_learning_curves(global_history, client_histories, save_dir, model_name, source_name):
    return plot_federated_loss_accuracy(global_history, client_histories, save_dir, model_name, source_name)
