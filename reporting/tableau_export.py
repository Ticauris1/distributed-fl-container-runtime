"""Consolidate active and archived experiment artifacts for Tableau dashboards."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "tableau"


def _read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows, preferred_fields=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(preferred_fields)
    seen = set(fields)
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        if not fields:
            return
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _relative(path, root):
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _context(path, results_root):
    relative = _relative(path, results_root)
    parts = Path(relative).parts
    archived = "archive" in parts
    return {
        "source_file": relative,
        "archived": archived,
        "result_status": "archived" if archived else "active",
    }


def _experiment_id(row, source_file):
    parts = [
        row.get("training_type", "unknown"),
        row.get("variant_name", row.get("dataset_label", "unnamed")),
        f"seed_{row.get('seed', 'unknown')}",
    ]
    return "::".join(str(value) for value in parts) + f"::{source_file}"


def _should_include(path, results_root, include_archived):
    return include_archived or "archive" not in Path(_relative(path, results_root)).parts


def collect_test_metrics(results_root, include_archived=False):
    rows = []
    for path in sorted(results_root.rglob("test_metrics.json")):
        if not _should_include(path, results_root, include_archived):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        context = _context(path, results_root)
        payload.update(context)
        payload["experiment_id"] = _experiment_id(payload, context["source_file"])
        payload["artifact_type"] = "test_metrics"
        rows.append(payload)
    return rows


def _find_parent_metrics(history_path):
    candidate = history_path.parent / "test_metrics.json"
    if not candidate.is_file():
        return {}
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def collect_histories(
    results_root, filename, index_name, include_archived=False
):
    rows = []
    for path in sorted(results_root.rglob(filename)):
        if not _should_include(path, results_root, include_archived):
            continue
        context = _context(path, results_root)
        metrics = _find_parent_metrics(path)
        base = {
            **context,
            "training_type": metrics.get("training_type", ""),
            "variant_name": metrics.get("variant_name", path.parent.parent.name),
            "model": metrics.get("model", ""),
            "encoder": metrics.get("encoder", ""),
            "seed": metrics.get("seed", ""),
            "method": metrics.get("method", ""),
            "weighting": metrics.get("weighting", ""),
            "client_distribution": metrics.get("client_distribution", ""),
            "clients": metrics.get("clients", ""),
        }
        base["experiment_id"] = _experiment_id(base, context["source_file"])
        for position, row in enumerate(_read_csv(path), start=1):
            item = {**base, **row}
            item.setdefault(index_name, position)
            rows.append(item)
    return rows


def collect_client_weights(round_rows):
    output = []
    for row in round_rows:
        clients = str(row.get("selected_clients", "")).split(";")
        weights = str(row.get("aggregation_weights", "")).split(";")
        samples = str(row.get("selected_client_samples", "")).split(";")
        if not clients or not weights or len(clients) != len(weights):
            continue
        for position, (client, weight) in enumerate(zip(clients, weights)):
            if not client or not weight:
                continue
            output.append({
                "experiment_id": row["experiment_id"],
                "variant_name": row.get("variant_name", ""),
                "seed": row.get("seed", ""),
                "round": row.get("round", ""),
                "client_id": client,
                "client_position": position + 1,
                "client_samples": samples[position] if position < len(samples) else "",
                "aggregation_weight": weight,
                "weighting": row.get("weighting", ""),
                "weighting_phase": row.get("weighting_phase", ""),
                "client_distribution": row.get("client_distribution", ""),
                "clients": row.get("clients", ""),
                "result_status": row.get("result_status", ""),
                "source_file": row.get("source_file", ""),
            })
    return output


def collect_classification_summaries(results_root, include_archived=False):
    rows = []
    candidates = list(results_root.rglob("_summary.csv"))
    for path in sorted(candidates):
        if not _should_include(path, results_root, include_archived):
            continue
        context = _context(path, results_root)
        for row in _read_csv(path):
            normalized = {
                "training_type": "classification",
                "variant_name": row.get("comparison_variant", ""),
                "model": row.get("model_name", ""),
                "encoder": row.get("model_name", ""),
                "seed": row.get("experiment_seed", ""),
                "method": row.get("method", ""),
                "weighting": row.get("client_weight_method", ""),
                "client_distribution": row.get("distribution", ""),
                "clients": row.get("clients", ""),
                "dataset": row.get("source", ""),
            }
            item = {
                **normalized, **row, **context,
                "artifact_type": "classification_summary",
            }
            item["experiment_id"] = _experiment_id(item, context["source_file"])
            rows.append(item)
    return rows


def collect_source_catalog(
    results_root, excluded_directory=None, include_archived=False
):
    rows = []
    supported = {".csv", ".json", ".pth", ".png", ".jpg", ".jpeg"}
    for path in sorted(results_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in supported:
            continue
        if not _should_include(path, results_root, include_archived):
            continue
        if excluded_directory is not None:
            try:
                path.resolve().relative_to(Path(excluded_directory).resolve())
                continue
            except ValueError:
                pass
        stat = path.stat()
        rows.append({
            **_context(path, results_root),
            "artifact_extension": path.suffix.lower(),
            "artifact_bytes": stat.st_size,
            "modified_timestamp": int(stat.st_mtime),
        })
    return rows


def export_tableau_data(
    results_root=DEFAULT_RESULTS_ROOT,
    output_dir=DEFAULT_OUTPUT_DIR,
    include_archived=False,
):
    results_root = Path(results_root)
    output_dir = Path(output_dir)
    metrics = collect_test_metrics(results_root, include_archived)
    rounds = collect_histories(
        results_root, "round_history.csv", "round", include_archived
    )
    epochs = collect_histories(
        results_root, "training_history.csv", "epoch", include_archived
    )
    weights = collect_client_weights(rounds)
    classification = collect_classification_summaries(
        results_root, include_archived
    )
    catalog = collect_source_catalog(
        results_root, output_dir, include_archived
    )

    outputs = {
        "segmentation_experiments.csv": metrics,
        "federated_rounds.csv": rounds,
        "centralized_epochs.csv": epochs,
        "client_weights.csv": weights,
        "classification_experiments.csv": classification,
        "artifact_catalog.csv": catalog,
    }
    preferred = (
        "experiment_id", "result_status", "training_type", "variant_name",
        "model", "encoder", "seed", "method", "weighting",
        "client_distribution", "clients", "rounds", "local_epochs",
        "configured_epochs", "epochs_trained", "total_local_epochs",
        "image_size", "batch_size",
        "best_epoch", "best_round", "selected_threshold",
        "validation_dice", "test_loss", "training_seconds",
        "mean_round_seconds", "checkpoint_size_mb",
        "dice", "iou", "pixel_precision", "pixel_recall", "specificity",
        "mean_cracked_sample_dice", "empty_mask_accuracy", "source_file",
    )
    for filename, rows in outputs.items():
        _write_csv(output_dir / filename, rows, preferred if "experiments" in filename else ())

    quality = [
        {"table": filename, "row_count": len(rows), "status": "ready" if rows else "empty"}
        for filename, rows in outputs.items()
    ]
    _write_csv(output_dir / "data_quality.csv", quality, ("table", "row_count", "status"))
    return {filename: len(rows) for filename, rows in outputs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include-archived",
        action="store_true",
        help="Include results/archive in historical Tableau exports.",
    )
    args = parser.parse_args()
    counts = export_tableau_data(
        args.results_root, args.output_dir, args.include_archived
    )
    for filename, count in counts.items():
        print(f"[tableau-export] {filename}: {count} rows")
    print(f"[tableau-export] output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
