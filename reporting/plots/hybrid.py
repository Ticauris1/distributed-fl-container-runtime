"""Hybrid optimization, routing, timing, and comparison plots."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from .common import save_figure


def plot_hybrid_routing_bar(lightweight_count, heavy_count, save_path, title="Hybrid Routing Usage"):
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.bar(["Lightweight", "Heavy"], [lightweight_count, heavy_count])
    axis.set(title=title, ylabel="Samples")
    return save_figure(figure, save_path, 200)


def plot_hybrid_timing_bar(heavy_only_time, hybrid_time, save_path, title="Average Inference Time"):
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.bar(["Heavy Only", "Hybrid"], [heavy_only_time, hybrid_time])
    axis.set(title=title, ylabel="Seconds / sample")
    return save_figure(figure, save_path, 200)


def plot_hybrid_history(history_df, save_path, selected_threshold=None, title="Bayesian Optimization of Hybrid Threshold"):
    if history_df is None or len(history_df) == 0:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history_df["threshold_high"], history_df["accuracy"], marker="o")
    axes[1].plot(history_df["threshold_high"], history_df["avg_time_per_sample"], marker="o")
    for axis, ylabel in zip(axes, ("accuracy", "avg_time_per_sample")):
        if selected_threshold is not None: axis.axvline(selected_threshold, linestyle="--")
        axis.set(xlabel="threshold_high", ylabel=ylabel)
        axis.grid(True)
    figure.suptitle(title)
    return save_figure(figure, save_path, 200)


def plot_hybrid_bo_pareto(history_df, save_path, title="Optimized Thresholds for Hybrid Model"):
    if history_df is None or len(history_df) == 0:
        return None
    data = history_df.copy().reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(8, 6))
    scatter = axis.scatter(data["avg_time_per_sample"], data["accuracy"], c=np.arange(len(data)), alpha=0.4, cmap="viridis", edgecolors="none", s=40, label="Trial Points (BO Iterations)")
    ordered = data.sort_values("avg_time_per_sample")
    pareto_rows, best_accuracy = [], -1.0
    for _, row in ordered.iterrows():
        if row["accuracy"] > best_accuracy:
            pareto_rows.append(row)
            best_accuracy = row["accuracy"]
    if pareto_rows:
        x_values = [row["avg_time_per_sample"] for row in pareto_rows]
        y_values = [row["accuracy"] for row in pareto_rows]
        # Intentionally no point markers on the selected/Pareto series.
        axis.step(x_values, y_values, where="post", linestyle="--", color="red", linewidth=2.5, label="Pareto Frontier")
    axis.set(xlabel="Processing Time (sec/sample)", ylabel="Accuracy", title=title)
    axis.grid(True, linestyle=":", alpha=0.6)
    axis.legend(loc="lower right")
    figure.colorbar(scatter, ax=axis, label="BO Iteration Number (Progression)")
    return save_figure(figure, save_path, 250)


def plot_hybrid_comparison_bars(comparison_df, save_path, title="Heavy vs Lightweight vs Hybrid Comparison"):
    if comparison_df is None or len(comparison_df) == 0:
        return None
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, column in zip(axes, ("accuracy", "f1_score", "avg_time_per_sample")):
        axis.bar(comparison_df["mode"], comparison_df[column])
        axis.set_title(column.replace("_", " ").title())
        axis.tick_params(axis="x", rotation=20)
    figure.suptitle(title)
    return save_figure(figure, save_path, 200)


plot_hybrid_threshold_tradeoff = plot_hybrid_bo_pareto
