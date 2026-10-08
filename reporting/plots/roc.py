"""Server and client ROC plots."""

from itertools import cycle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

from .common import panel_title, save_figure


def draw_roc_on_ax(axis, y_true, y_prob, title, class_labels=None, **kwargs):
    del kwargs
    if y_true is None or y_prob is None:
        axis.text(0.5, 0.5, "No data", ha="center", va="center")
        axis.set_title(title)
        return
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if not np.isfinite(y_prob).all():
        invalid = int((~np.isfinite(y_prob)).sum())
        axis.text(
            0.5, 0.5,
            f"ROC unavailable\n{invalid} non-finite probabilities",
            ha="center", va="center",
        )
        axis.set_title(title)
        axis.set_axis_off()
        return
    classes = np.unique(y_true)
    binary = label_binarize(y_true, classes=classes)
    if binary.ndim == 1 or binary.shape[1] == 1:
        binary = np.column_stack([1 - binary.ravel(), binary.ravel()])
    colors = cycle(plt.cm.tab10.colors)
    for index, color in zip(range(min(binary.shape[1], y_prob.shape[1])), colors):
        false_positive, true_positive, _ = roc_curve(binary[:, index], y_prob[:, index])
        label = class_labels[index] if class_labels and index < len(class_labels) else f"Class {index}"
        axis.plot(false_positive, true_positive, color=color, label=f"{label} (AUC={auc(false_positive, true_positive):.2f})")
    axis.plot([0, 1], [0, 1], "k--")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="lower right", fontsize=8, frameon=False)


def _panel(server_y_true, server_y_prob, client_pairs, path, class_labels=None, vertical=False, dpi=180):
    count = 1 + len(client_pairs)
    rows, columns = (count, 1) if vertical else (1, count)
    figure, axes = plt.subplots(rows, columns, figsize=((6.8 if vertical else 5 * count), (4.8 * count if vertical else 5)), dpi=dpi, squeeze=False)
    flat = axes.flatten()
    draw_roc_on_ax(flat[0], server_y_true, server_y_prob, "Server (Global) ROC", class_labels)
    for index, (labels, probabilities) in enumerate(client_pairs, start=1):
        draw_roc_on_ax(flat[index], labels if labels is not None else server_y_true, probabilities, f"Client {index} ROC", class_labels)
    figure.suptitle(panel_title(len(client_pairs)))
    figure.tight_layout()
    return save_figure(figure, path)


def plot_col_rocs(experiment, save_dir, dpi=180, filename=None, **kwargs):
    del kwargs
    path = Path(save_dir) / (filename or f"{experiment.get('source_name','Dataset')}_{experiment.get('model_name','Model')}_panel_ROC_server_and_clients_col.png")
    return _panel(experiment["server_y_true"], experiment["server_y_prob"], experiment.get("client_pairs", []), path, experiment.get("class_labels"), True, dpi)


def plot_grid_rocs(server_y_true, server_y_prob, client_pairs, save_dir, model_name, source_name, class_labels=None, dpi=180, filename=None, **kwargs):
    del kwargs
    path = Path(save_dir) / (filename or f"{source_name}_{model_name}_panel_ROC_server_and_clients_grid.png")
    return _panel(server_y_true, server_y_prob, client_pairs, path, class_labels, False, dpi)


def plot_fed_server_roc_cm(model_name, source_name, y_true, y_prob, cm, class_labels, out_dir, **kwargs):
    from .confusion import draw_confusion_matrix
    del kwargs
    figure, axis = plt.subplots(figsize=(5.2, 4.5), dpi=200)
    draw_roc_on_ax(axis, y_true, y_prob, f"{model_name} — {source_name} (Server ROC)", class_labels)
    save_figure(figure, Path(out_dir) / f"{source_name}_{model_name}_FedAvg_server_ROC.png")
    figure, axis = plt.subplots(figsize=(4.8, 4.4), dpi=200)
    draw_confusion_matrix(axis, cm, f"{model_name} — {source_name} (Server CM)", class_labels)
    save_figure(figure, Path(out_dir) / f"{source_name}_{model_name}_FedAvg_server_CM.png")
