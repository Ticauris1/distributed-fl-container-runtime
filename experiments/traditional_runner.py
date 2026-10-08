"""End-to-end execution of one traditional-ML experiment."""

from pathlib import Path
import numpy as np #type: ignore
from sklearn.metrics import classification_report, confusion_matrix #type: ignore

from data.loaders import build_fed_loaders_trad
from reporting.metrics import macro_prec_recall_from_report, safe_auc_multiclass
from reporting.outputs import (
    append_client_summary_row, append_summary_row, initialize_client_summary_csv,
    save_metadata, save_predictions,
)
from reporting.plots.confusion import plot_col_cms, plot_grid_cms
from reporting.plots.roc import plot_col_rocs, plot_grid_rocs
from training.traditional import evaluate_federated_ensemble, federated_training_traditional_ensemble
from .definitions import experiment_directory, subset_clients_for_participation


def _client_probability_pairs(client_models, features):
    pairs = []
    for model in client_models:
        try:
            pairs.append((None, model.predict_proba(features)))
        except Exception:
            pairs.append((None, None))
    return pairs


def run_traditional_experiment(spec, source_data, model_name, model_template, config, summary_csv):
    if source_data.traditional_data is None:
        raise ValueError(f"No traditional data prepared for {source_data.source_name}")
    experiment_dir = experiment_directory(config.EXP_BASE_DIR, spec, source_data.source_name) / model_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Experiment save dir] {experiment_dir}")
    client_summary_csv = initialize_client_summary_csv(experiment_dir / "_client_summary.csv")
    data = source_data.traditional_data
    client_loaders, server_val = build_fed_loaders_trad(
        data["X_train"], data["y_train"], data["X_test"], data["y_test"],
        config.BATCH_SIZE, spec.clients, random_state=config.RANDOM_STATE,
    )
    active_loaders = subset_clients_for_participation(client_loaders, spec.participation, config.RANDOM_STATE)
    save_metadata(experiment_dir, {
        "clients": spec.clients, "distribution": spec.distribution,
        "distribution_definition": config.DIST_DEFS.get(spec.distribution, ""),
        "participation": spec.participation, "method": spec.method,
        "client_shard_sizes": [len(group["train"].dataset) for group in active_loaders],
        "server_val_size": len(server_val[1]), "test_size": len(data["X_test"]),
    })
    client_models, _ = federated_training_traditional_ensemble(model_template, active_loaders, server_val)
    test_features, test_labels = data["X_test"], data["y_test"]
    for client_id, model in enumerate(client_models, start=1):
        predictions = model.predict(test_features)
        probabilities = model.predict_proba(test_features) if hasattr(model, "predict_proba") else None
        report = classification_report(test_labels, predictions, target_names=source_data.class_labels, output_dict=True, zero_division=0)
        precision, recall, f1 = macro_prec_recall_from_report(report)
        append_client_summary_row(client_summary_csv, {
            "experiment": spec.experiment_id, "source": source_data.source_name,
            "model_name": f"{model_name}-FedAvg", "client_id": client_id,
            "accuracy": float((predictions == test_labels).mean()), "precision": precision,
            "recall": recall, "f1_score": f1,
            "auc_score": safe_auc_multiclass(test_labels, probabilities, source_data.num_classes),
        })
    weights = np.asarray([len(group["train"].dataset) for group in active_loaders], dtype=float)
    predictions, probabilities = evaluate_federated_ensemble(
        client_models, test_features, y_test=test_labels,
        class_labels=source_data.class_labels, show_k=3, weights=weights,
    )
    server_cm = confusion_matrix(test_labels, predictions, labels=list(range(source_data.num_classes)))
    client_cms = [confusion_matrix(test_labels, model.predict(test_features), labels=list(range(source_data.num_classes))) for model in client_models]
    client_pairs = _client_probability_pairs(client_models, test_features)
    panel = {"server_cm": server_cm, "client_cms": client_cms,
             "server_y_true": test_labels, "server_y_prob": probabilities,
             "client_pairs": client_pairs, "class_labels": source_data.class_labels,
             "model_name": model_name, "source_name": source_data.source_name}
    plot_col_cms(panel, experiment_dir)
    plot_grid_cms(server_cm, client_cms, experiment_dir, model_name, source_data.source_name, source_data.class_labels)
    plot_col_rocs(panel, experiment_dir)
    plot_grid_rocs(test_labels, probabilities, client_pairs, experiment_dir, model_name, source_data.source_name, source_data.class_labels)
    report = classification_report(test_labels, predictions, target_names=source_data.class_labels, output_dict=True, zero_division=0)
    precision, recall, f1 = macro_prec_recall_from_report(report)
    auc_score = safe_auc_multiclass(test_labels, probabilities, source_data.num_classes)
    save_predictions(experiment_dir / f"{source_data.source_name}_{model_name}_test_predictions.csv", test_labels, predictions)
    append_summary_row(summary_csv, {
        "experiment": spec.experiment_id, "source": source_data.source_name,
        "model_name": f"{model_name}-FedAvg", "clients": spec.clients,
        "distribution": spec.distribution, "participation": spec.participation,
        "method": spec.method, "eval_mode": "standard", "accuracy": float((predictions == test_labels).mean()),
        "precision": precision, "recall": recall, "f1_score": f1,
        "auc_score": auc_score, "save_dir": str(experiment_dir),
    })
    return {"client_models": client_models, "metrics": {"accuracy": float((predictions == test_labels).mean()),
            "precision": precision, "recall": recall, "f1_score": f1, "auc_score": auc_score},
            "save_dir": experiment_dir}
