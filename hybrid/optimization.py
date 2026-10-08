"""Bayesian threshold optimization with a single PI objective."""

import numpy as np #type: ignore
import pandas as pd #type: ignore
from scipy.stats import norm #type: ignore
from sklearn.gaussian_process import GaussianProcessRegressor #type: ignore
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel #type: ignore

from config import RANDOM_STATE
from .evaluation import compute_binary_eval_metrics, evaluate_hybrid_on_loader_binary, evaluate_threshold


def calculate_pi_score(row, heavy_baseline_acc=None, k=10):
    del heavy_baseline_acc
    return row["accuracy"] * np.exp(-k * row["avg_time_per_sample"])


def threshold_objective(row, heavy_only_accuracy, max_acc_drop=0.01, k=10):
    """Minimize 1 - PI, with a constraint penalty below the accuracy floor."""
    penalty = 50.0 if row["accuracy"] < heavy_only_accuracy - max_acc_drop else 0.0
    return 1.0 - calculate_pi_score(row, heavy_only_accuracy, k) + penalty


def optimize_hybrid_threshold_bayes(
    lightweight_model, heavy_model, server_val_loader, device,
    threshold_min=0.10, threshold_max=0.95, n_init=10, n_iter=25,
    max_acc_drop=0.01, binary_mode=True, positive_classes=None,
    random_state=RANDOM_STATE,
):
    rng = np.random.RandomState(random_state)
    heavy_output = evaluate_hybrid_on_loader_binary(
        heavy_model, heavy_model, server_val_loader, device, -1.0, positive_classes
    )
    heavy_baseline = compute_binary_eval_metrics(
        heavy_output["y_true"], heavy_output["y_pred"], heavy_output["probs"]
    )["accuracy"]
    rows = []
    for iteration in range(n_init + n_iter):
        if iteration < n_init:
            threshold = float(rng.uniform(threshold_min, threshold_max))
        else:
            history = pd.DataFrame(rows)
            tested = history[["threshold_high"]].values
            objectives = history["objective"].values
            kernel = (ConstantKernel(1.0, (1e-3, 1e3)) *
                      Matern(length_scale=0.1, length_scale_bounds=(1e-3, 1e2), nu=2.5) +
                      WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1)))
            process = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, normalize_y=True, n_restarts_optimizer=10).fit(tested, objectives)
            grid = np.linspace(threshold_min, threshold_max, 1000).reshape(-1, 1)
            mean, std = process.predict(grid, return_std=True)
            improvement = objectives.min() - mean
            z_score = improvement / (std + 1e-9)
            expected_improvement = improvement * norm.cdf(z_score) + std * norm.pdf(z_score)
            expected_improvement[std == 0] = 0.0
            threshold = float(grid[np.argmax(expected_improvement), 0])
        row = evaluate_threshold(threshold, lightweight_model, heavy_model, server_val_loader, device, binary_mode, positive_classes)
        row["pi_score"] = calculate_pi_score(row, heavy_baseline, k=10)
        row["objective"] = threshold_objective(row, heavy_baseline, max_acc_drop, k=10)
        rows.append(row)
    history = pd.DataFrame(rows).sort_values("threshold_high").reset_index(drop=True)
    #best = history.loc[history["pi_score"].idxmax()]
    best = history.loc[history["objective"].idxmin()]
    return float(best["threshold_high"]), history, heavy_baseline


_hybrid_objective_from_row = threshold_objective
