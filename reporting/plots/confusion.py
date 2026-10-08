"""Standalone and server/client confusion-matrix plots."""

from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from .common import panel_title, save_figure


def draw_confusion_matrix(axis, matrix, title, class_labels=None, cmap="Blues"):
    image = axis.imshow(matrix, interpolation="nearest", cmap=cmap)
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Count")
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    if class_labels is not None:
        ticks = np.arange(len(class_labels))
        axis.set_xticks(ticks, class_labels, rotation=45, ha="right")
        axis.set_yticks(ticks, class_labels)
    threshold = matrix.max() / 2 if matrix.size and matrix.max() else 0.5
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{int(matrix[row, column])}", ha="center", va="center",
                      color="white" if matrix[row, column] > threshold else "black")


def _panel(server_cm, client_cms, save_path, class_labels=None, vertical=False, dpi=180):
    matrices = [server_cm] + list(client_cms)
    count = len(matrices)
    rows, columns = (count, 1) if vertical else (1, count)
    figure, axes = plt.subplots(rows, columns, figsize=((6 if vertical else 5 * count), (4.8 * count if vertical else 5)), dpi=dpi, squeeze=False)
    flat = axes.flatten()
    for index, matrix in enumerate(matrices):
        draw_confusion_matrix(flat[index], matrix, "Server (Global)" if index == 0 else f"Client {index}", class_labels)
    figure.suptitle(panel_title(len(client_cms)))
    figure.tight_layout()
    return save_figure(figure, save_path)


def plot_col_cms(experiment, save_dir, dpi=180, filename=None, **kwargs):
    del kwargs
    path = Path(save_dir) / (filename or f"{experiment.get('source_name','Dataset')}_{experiment.get('model_name','Model')}_panel_confusion_matrices_col.png")
    return _panel(experiment["server_cm"], experiment.get("client_cms", []), path, experiment.get("class_labels"), True, dpi)


def plot_grid_cms(server_cm, client_cms, save_dir, model_name, source_name, class_labels=None, dpi=180, filename=None, **kwargs):
    del kwargs
    path = Path(save_dir) / (filename or f"{source_name}_{model_name}_panel_confusion_matrices.png")
    return _panel(server_cm, client_cms, path, class_labels, False, dpi)


def plot_final_confusion_matrix(y_true, y_pred, save_dir, model_name, source_name, method=None, distribution=None, class_labels=None, dpi=220):
    del distribution
    labels = list(range(len(class_labels))) if class_labels is not None else sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    figure, axis = plt.subplots(figsize=(6, 5), dpi=dpi)
    draw_confusion_matrix(axis, matrix, f"Final Test Confusion Matrix\n{model_name}\n{source_name}", class_labels)
    path = Path(save_dir) / f"{source_name}_{model_name}_{method or 'fed'}_final_confusion_matrix.png"
    figure.tight_layout()
    return save_figure(figure, path)


draw_cm_on_ax = draw_confusion_matrix
