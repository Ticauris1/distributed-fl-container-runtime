"""Shared plot saving, titles, and axis helpers."""

from pathlib import Path
import matplotlib.pyplot as plt


def save_figure(figure, path, dpi=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(figure)
    print(f"[saved] {path}")
    return str(path)


def panel_title(client_count, resource_label=None):
    title = " | ".join(["Server"] + [f"Client {index}" for index in range(1, client_count + 1)])
    return f"{title} ({resource_label})" if resource_label else title


def format_axis(axis, xlabel=None, ylabel=None, title=None, grid=True):
    if xlabel: axis.set_xlabel(xlabel)
    if ylabel: axis.set_ylabel(ylabel)
    if title: axis.set_title(title)
    if grid: axis.grid(True, linestyle=":", alpha=0.5)


_assert_and_log_save = save_figure
