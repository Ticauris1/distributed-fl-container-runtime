"""End-to-end execution of one deep-learning experiment."""

from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
import numpy as np #type: ignore
import pandas as pd #type: ignore
import torch #type: ignore
from sklearn.metrics import classification_report, confusion_matrix #type: ignore
from torch.utils.data import DataLoader #type: ignore

from data.datasets import ImageDataset
from data.loaders import build_fed_loaders_dl
from hybrid.evaluation import (
    compute_binary_eval_metrics, compute_eval_metrics, evaluate_heavy_only_on_loader,
    evaluate_hybrid_on_loader, evaluate_hybrid_on_loader_binary,
)
from hybrid.optimization import calculate_pi_score, optimize_hybrid_threshold_bayes
from models.factory import get_transforms
from models.lightweight import LightweightCNN, get_ready_lightweight_model, verify_model_complexity
from privacy.integration import PrivacyAggregationError
from reporting.metrics import macro_prec_recall_from_report, safe_auc_multiclass
from reporting.outputs import (
    append_client_summary_row, append_summary_row, initialize_client_summary_csv,
    save_metadata, save_model_checkpoint, save_predictions,
)
from reporting.plots.confusion import plot_col_cms, plot_final_confusion_matrix, plot_grid_cms
from reporting.plots.hybrid import plot_hybrid_bo_pareto, plot_hybrid_history
from reporting.plots.roc import plot_col_rocs, plot_grid_rocs
from reporting.plots.training import plot_federated_loss_accuracy
from training.federated import federated_training
from training.fedprox import federated_prox_training
from .definitions import experiment_directory


def _package_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"


def _summarize_privacy_rounds(rounds, enabled):
    successful = [row for row in rounds if row.get("status") == "success"]
    failed = [row for row in rounds if row.get("status") == "failed"]
    if enabled:
        status = "success" if successful and len(successful) == len(rounds) else "failed"
    else:
        status = "disabled"
    total_seconds = float(sum(row.get("total_seconds", 0.0) for row in successful))
    return {
        "enabled": bool(enabled),
        "status": status,
        "rounds_recorded": len(rounds),
        "successful_rounds": len(successful),
        "failed_rounds": len(failed),
        "fallback_used": any(bool(row.get("fallback_used")) for row in rounds),
        "total_seconds": total_seconds,
        "mean_seconds": total_seconds / len(successful) if successful else 0.0,
        "max_abs_error": max((row.get("max_abs_error", 0.0) for row in successful), default=0.0),
        "mean_abs_error": (
            float(np.mean([row.get("mean_abs_error", 0.0) for row in successful]))
            if successful else 0.0
        ),
        "plaintext_bytes": int(sum(row.get("plaintext_bytes", 0) for row in successful)),
        "protected_bytes": int(sum(row.get("protected_bytes", 0) for row in successful)),
        "size_expansion_ratio": (
            float(np.mean([row["size_expansion_ratio"] for row in successful
                           if row.get("size_expansion_ratio") is not None]))
            if any(row.get("size_expansion_ratio") is not None for row in successful) else None
        ),
        "rounds": rounds,
    }


def run_deep_experiment(
    spec, source_data, model_name, config, summary_csv,
    privacy_agg_enabled=None, variant_name=None, client_weight_method=None,
    experiment_seed=None,
):
    privacy_enabled = (
        config.USE_PRIVACY_AWARE_AGGREGATION
        if privacy_agg_enabled is None else privacy_agg_enabled
    )
    run_name = variant_name or model_name
    seed = config.RANDOM_STATE if experiment_seed is None else int(experiment_seed)
    weight_method = client_weight_method or config.CLIENT_WEIGHT_METHOD
    experiment_dir = (
        experiment_directory(config.EXP_BASE_DIR, spec, source_data.source_name)
        / run_name / f"seed_{seed}"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Experiment save dir] {experiment_dir}")
    client_summary_csv = initialize_client_summary_csv(experiment_dir / "_client_summary.csv")
    train_transform, validation_transform = get_transforms(model_name)
    client_loaders, server_val_loader, test_loader, index_ledger = build_fed_loaders_dl(
        source_data.train_paths, source_data.train_labels, source_data.val_paths,
        source_data.val_labels, source_data.test_paths, source_data.test_labels,
        train_transform, validation_transform, config.BATCH_SIZE, spec.clients,
        spec.distribution, seed, label_skew_alpha=config.LABEL_SKEW_ALPHA,
        qty_skew_concentration=0.7, min_client_size=100,
    )
    if config.FED_COLD_START:
        lightweight_model = LightweightCNN(source_data.num_classes).to(config.DEVICE)
        print("🆕 Initialized lightweight model with random weights (Cold Start).")
    else:
        paper_train_loader = DataLoader(
            ImageDataset(
                source_data.train_paths,
                source_data.train_labels,
                transform=train_transform,
            ),
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )
        lightweight_model = get_ready_lightweight_model(
            source_data.num_classes,
            config.DEVICE,
            train_loader=paper_train_loader,
            val_loader=server_val_loader,
            checkpoint_root=config.LIGHTWEIGHT_CHECKPOINT_DIR,
            dataset_name=source_data.source_name,
            epochs=config.LIGHTWEIGHT_EPOCHS,
        )
    verify_model_complexity(lightweight_model)
    metadata = {
        "random_state": seed, "experiment_seed": seed, "clients": spec.clients,
        "distribution": spec.distribution,
        "distribution_definition": config.DIST_DEFS.get(spec.distribution, ""),
        "participation": spec.participation, "method": spec.method,
        "client_shard_sizes": [len(group["train"].dataset) for group in client_loaders],
        "server_val_size": len(server_val_loader.dataset), "test_size": len(test_loader.dataset),
        "index_ledger": index_ledger,
        "model_name": model_name,
        "comparison_variant": run_name,
        "privacy_agg_enabled": privacy_enabled,
        "privacy_agg_target": config.PRIVACY_AGG_TARGET,
        "client_weight_method": weight_method,
        "celm_settings": {
            "mode": weight_method,
            "warmup_rounds": (
                config.PAPER_CELM_WARMUP_ROUNDS if weight_method == "paper_celm"
                else config.CELM_WARMUP_ROUNDS
            ),
            "warmup_fraction": (
                config.PAPER_CELM_WARMUP_FRACTION if weight_method == "paper_celm"
                else config.CELM_WARMUP_FRACTION
            ),
            "logit_maximization_steps": (
                config.PAPER_CELM_LM_STEPS if weight_method == "paper_celm"
                else config.CELM_LM_STEPS
            ),
            "learning_rate": (
                config.PAPER_CELM_LM_LEARNING_RATE if weight_method == "paper_celm"
                else config.CELM_LM_LEARNING_RATE
            ),
            "l2_penalty": (
                config.PAPER_CELM_L2_PENALTY if weight_method == "paper_celm"
                else config.CELM_L2_PENALTY
            ),
            "ema_beta": (
                config.PAPER_CELM_EMA_BETA if weight_method == "paper_celm"
                else config.CELM_EMA_BETA
            ),
        },
        "run_status": "training",
        "privacy_aggregation": {
            "enabled": bool(privacy_enabled),
            "status": "pending" if privacy_enabled else "disabled",
            "fallback_used": False,
        },
        "library_versions": {
            "torch": _package_version("torch"),
            "torchvision": _package_version("torchvision"),
            "timm": _package_version("timm"),
        },
    }
    save_metadata(experiment_dir, metadata)
    common_training_args = dict(
        model_name=model_name, source_name=source_data.source_name,
        client_dataloaders=client_loaders, server_val_loader=server_val_loader,
        test_loader=test_loader, num_classes=source_data.num_classes, device=config.DEVICE,
        rounds=config.FED_ROUNDS, local_epochs=config.FED_LOCAL_EPOCHS,
        clients_per_round=spec.clients_per_round, lr=config.FED_LEARNING_RATE,
        pretrained=not config.FED_COLD_START,
    )
    try:
        if spec.method.strip().lower() == "fedprox":
            if weight_method.strip().lower() != "sample_size":
                raise ValueError(
                    "CELM weighting is currently implemented for FedAvg/FedAdam, "
                    "not FedProx. Use client_weight_method='sample_size' for FedProx."
                )
            fedprox_args = {
                **common_training_args,
                "lr": config.FEDPROX_LEARNING_RATE,
            }
            training_result = federated_prox_training(
                **fedprox_args,
                mu=config.FEDPROX_MU,
                privacy_agg_enabled=privacy_enabled,
                privacy_agg_target=config.PRIVACY_AGG_TARGET,
            )
        else:
            training_result = federated_training(
                **common_training_args,
                server_method=spec.method,
                server_lr=config.FED_SERVER_LEARNING_RATE,
                server_beta1=0.9, server_beta2=0.999, server_eps=1e-8,
                train_agg_mode=config.TRAIN_AGG_MODE, eval_agg_mode=config.EVAL_AGG_MODE,
                prediction_weight_mode=config.PREDICTION_WEIGHT_MODE,
                privacy_agg_enabled=privacy_enabled,
                privacy_agg_target=config.PRIVACY_AGG_TARGET,
                client_weight_method=weight_method,
                celm_warmup_rounds=config.CELM_WARMUP_ROUNDS,
                celm_warmup_fraction=config.CELM_WARMUP_FRACTION,
                celm_steps=config.CELM_LM_STEPS,
                celm_lr=config.CELM_LM_LEARNING_RATE,
                celm_l2_penalty=config.CELM_L2_PENALTY,
                celm_ema_beta=config.CELM_EMA_BETA,
                celm_seed=seed,
                paper_celm_warmup_rounds=config.PAPER_CELM_WARMUP_ROUNDS,
                paper_celm_warmup_fraction=config.PAPER_CELM_WARMUP_FRACTION,
                paper_celm_steps=config.PAPER_CELM_LM_STEPS,
                paper_celm_lr=config.PAPER_CELM_LM_LEARNING_RATE,
                paper_celm_l2_penalty=config.PAPER_CELM_L2_PENALTY,
                paper_celm_ema_beta=config.PAPER_CELM_EMA_BETA,
            )
    except Exception as error:
        metadata["run_status"] = "failed"
        metadata["error"] = {"type": type(error).__name__, "message": str(error)}
        metadata["privacy_aggregation"] = {
            "enabled": bool(privacy_enabled),
            "status": "failed" if isinstance(error, PrivacyAggregationError) else "not_completed",
            "fallback_used": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        save_metadata(experiment_dir, metadata)
        raise
    federated_model, fed_history, client_histories, _, _, client_pairs = training_result
    privacy_rounds = fed_history.get("privacy_aggregation", [])
    privacy_summary = _summarize_privacy_rounds(privacy_rounds, privacy_enabled)
    metadata["run_status"] = "trained"
    metadata["privacy_aggregation"] = privacy_summary
    save_metadata(experiment_dir, metadata)
    pd.DataFrame(privacy_rounds).to_csv(
        experiment_dir / "privacy_aggregation_rounds.csv", index=False
    )
    client_weighting_rounds = fed_history.get("client_weighting", [])
    pd.DataFrame(client_weighting_rounds).to_csv(
        experiment_dir / "client_weighting_rounds.csv", index=False
    )
    class_labels = source_data.class_labels
    binary_mode = config.USE_BINARY_HYBRID_EVAL
    if binary_mode:
        heavy_output = evaluate_hybrid_on_loader_binary(
            federated_model, federated_model, test_loader, config.DEVICE, -1.0,
            config.HYBRID_POSITIVE_CLASSES,
        )
        heavy_metrics = compute_binary_eval_metrics(heavy_output["y_true"], heavy_output["y_pred"], heavy_output["probs"])
    else:
        heavy_output = evaluate_heavy_only_on_loader(federated_model, test_loader, config.DEVICE)
        heavy_metrics = compute_eval_metrics(heavy_output["y_true"], heavy_output["y_pred"], heavy_output["probs"])
    if binary_mode:
        lightweight_output = evaluate_hybrid_on_loader_binary(
            lightweight_model, lightweight_model, test_loader, config.DEVICE, -1.0,
            config.HYBRID_POSITIVE_CLASSES,
        )
        lightweight_metrics = compute_binary_eval_metrics(lightweight_output["y_true"], lightweight_output["y_pred"], lightweight_output["probs"])
    else:
        lightweight_output = evaluate_heavy_only_on_loader(lightweight_model, test_loader, config.DEVICE)
        lightweight_metrics = compute_eval_metrics(lightweight_output["y_true"], lightweight_output["y_pred"], lightweight_output["probs"])
    final_output, final_metrics = heavy_output, heavy_metrics
    selected_threshold = config.HYBRID_BO_THRESHOLD_MAX
    if config.USE_HYBRID_INFERENCE and config.EVAL_AGG_MODE == "weight_avg":
        if config.USE_HYBRID_BAYES_OPT:
            selected_threshold, history, heavy_baseline = optimize_hybrid_threshold_bayes(
                lightweight_model, federated_model, server_val_loader, config.DEVICE,
                config.HYBRID_BO_THRESHOLD_MIN, config.HYBRID_BO_THRESHOLD_MAX,
                config.HYBRID_BO_N_INIT, config.HYBRID_BO_N_ITER,
                config.HYBRID_BO_MAX_ACC_DROP, binary_mode,
                config.HYBRID_POSITIVE_CLASSES, seed,
            )
            history["performance_index"] = history.apply(lambda row: calculate_pi_score(row, heavy_baseline), axis=1)
            history.to_csv(experiment_dir / f"{source_data.source_name}_{model_name}_hybrid_bo_history.csv", index=False)
            history[["threshold_high", "accuracy", "avg_time_per_sample", "performance_index"]].to_csv(
                experiment_dir / f"{source_data.source_name}_{model_name}_table_v_data.csv", index=False
            )
            plot_hybrid_history(history, experiment_dir / f"{source_data.source_name}_{model_name}_hybrid_bo_history.png", selected_threshold)
            plot_hybrid_bo_pareto(history, experiment_dir / f"{source_data.source_name}_{model_name}_hybrid_bo_pareto.png")
        final_output = (
            evaluate_hybrid_on_loader_binary(lightweight_model, federated_model, test_loader, config.DEVICE, selected_threshold, config.HYBRID_POSITIVE_CLASSES)
            if binary_mode else
            evaluate_hybrid_on_loader(lightweight_model, federated_model, test_loader, config.DEVICE, selected_threshold)
        )
        final_metrics = (compute_binary_eval_metrics if binary_mode else compute_eval_metrics)(
            final_output["y_true"], final_output["y_pred"], final_output["probs"]
        )
        pd.DataFrame([
            {"mode": "lightweight_only", "accuracy": lightweight_metrics["accuracy"], "avg_time_per_sample": lightweight_output["avg_time_per_sample"]},
            {"mode": "heavy_only", "accuracy": heavy_metrics["accuracy"], "avg_time_per_sample": heavy_output["avg_time_per_sample"]},
            {"mode": "hybrid", "accuracy": final_metrics["accuracy"], "avg_time_per_sample": final_output["avg_time_per_sample"]},
        ]).to_csv(experiment_dir / f"{source_data.source_name}_{model_name}_hybrid_comparison.csv", index=False)
        if final_output["heavy_count"] == 0:
            print(
                "[HYBRID WARNING] No test samples were routed to the heavy model; "
                "hybrid accuracy equals lightweight-only performance and must not "
                "be interpreted as heavy-model performance."
            )
    y_true, y_pred, probabilities = (np.asarray(final_output[key]) for key in ("y_true", "y_pred", "probs"))
    plot_final_confusion_matrix(y_true, y_pred, experiment_dir, model_name, source_data.source_name,
                                spec.method, spec.distribution,
                                ["Negative", "Positive"] if binary_mode else class_labels)
    multiclass_output = evaluate_heavy_only_on_loader(federated_model, test_loader, config.DEVICE)
    server_labels = np.asarray(multiclass_output["y_true"])
    server_probabilities = np.asarray(multiclass_output["probs"])
    server_cm = confusion_matrix(server_labels, server_probabilities.argmax(axis=1), labels=list(range(source_data.num_classes)))
    client_cms = [confusion_matrix(labels, probs.argmax(axis=1), labels=list(range(source_data.num_classes))) for labels, probs in client_pairs]
    panel = {"server_cm": server_cm, "client_cms": client_cms, "server_y_true": server_labels,
             "server_y_prob": server_probabilities, "client_pairs": client_pairs,
             "class_labels": class_labels, "model_name": model_name, "source_name": source_data.source_name}
    plot_col_cms(panel, experiment_dir); plot_grid_cms(server_cm, client_cms, experiment_dir, model_name, source_data.source_name, class_labels)
    plot_col_rocs(panel, experiment_dir); plot_grid_rocs(server_labels, server_probabilities, client_pairs, experiment_dir, model_name, source_data.source_name, class_labels)
    plot_federated_loss_accuracy(fed_history, client_histories, experiment_dir, model_name, source_data.source_name, spec.method, spec.distribution)
    for client_id, (labels, probs) in enumerate(client_pairs, start=1):
        predictions = probs.argmax(axis=1)
        report = classification_report(labels, predictions, output_dict=True, zero_division=0)
        precision, recall, f1 = macro_prec_recall_from_report(report)
        append_client_summary_row(client_summary_csv, {
            "experiment": spec.experiment_id, "experiment_seed": seed,
            "source": source_data.source_name,
            "model_name": model_name, "client_id": client_id,
            "accuracy": float((predictions == labels).mean()), "precision": precision,
            "recall": recall, "f1_score": f1,
            "auc_score": safe_auc_multiclass(labels, probs, source_data.num_classes),
        })
    predictions_path = experiment_dir / f"{source_data.source_name}_{model_name}_test_predictions.csv"
    checkpoint_path = experiment_dir / f"{source_data.source_name}_{model_name}_fed_final.pth"
    save_predictions(predictions_path, y_true, y_pred)
    save_model_checkpoint(checkpoint_path, federated_model)
    append_summary_row(summary_csv, {
        "experiment": spec.experiment_id, "experiment_seed": seed,
        "comparison_variant": run_name, "source": source_data.source_name,
        "model_name": f"{model_name}-train[{config.TRAIN_AGG_MODE}]-eval[{config.EVAL_AGG_MODE}]",
        "clients": spec.clients, "distribution": spec.distribution, "participation": spec.participation,
        "method": spec.method, "eval_mode": "hybrid" if config.USE_HYBRID_INFERENCE else "standard",
        "lightweight_model_name": config.LIGHTWEIGHT_MODEL_NAME if config.USE_HYBRID_INFERENCE else "None",
        "heavy_model_name": model_name, "hybrid_threshold_low": 0.1,
        "hybrid_threshold_high": selected_threshold,
        "hybrid_lightweight_count": final_output["lightweight_count"] if config.USE_HYBRID_INFERENCE else 0,
        "hybrid_heavy_count": final_output["heavy_count"], "hybrid_heavy_fraction": final_output["heavy_fraction"],
        "avg_time_per_sample": final_output["avg_time_per_sample"],
        "client_weight_method": weight_method,
        "client_weighting_total_seconds": float(sum(
            row.get("estimation_seconds", 0.0) for row in client_weighting_rounds
        )),
        "privacy_agg_enabled": privacy_enabled,
        "privacy_agg_target": config.PRIVACY_AGG_TARGET,
        "privacy_agg_status": privacy_summary["status"],
        "privacy_agg_rounds": privacy_summary["rounds_recorded"],
        "privacy_agg_failures": privacy_summary["failed_rounds"],
        "privacy_agg_fallback_used": privacy_summary["fallback_used"],
        "privacy_agg_total_seconds": privacy_summary["total_seconds"],
        "privacy_agg_mean_seconds": privacy_summary["mean_seconds"],
        "privacy_agg_max_abs_error": privacy_summary["max_abs_error"],
        "privacy_agg_mean_abs_error": privacy_summary["mean_abs_error"],
        "privacy_agg_plaintext_bytes": privacy_summary["plaintext_bytes"],
        "privacy_agg_protected_bytes": privacy_summary["protected_bytes"],
        "privacy_agg_size_expansion_ratio": privacy_summary["size_expansion_ratio"],
        **final_metrics, "save_dir": str(experiment_dir),
    })
    metadata["run_status"] = "complete"
    save_metadata(experiment_dir, metadata)
    return {"model": federated_model, "metrics": final_metrics, "save_dir": experiment_dir}
