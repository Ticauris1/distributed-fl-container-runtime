"""Federated-learning experiment orchestration entry point."""

import config
from data.preparation import prepare_all_datasets
from experiments.deep_runner import run_deep_experiment
from experiments.definitions import build_experiment_grid
from experiments.traditional_runner import run_traditional_experiment
from reporting.outputs import initialize_summary_csv
from reporting.multiseed import write_multiseed_summary
from traditional.pipelines import build_traditional_models


def prepare_datasets():
    return prepare_all_datasets(
        datasets=config.DATASETS,
        run_on_dataset=config.RUN_ON_DATASET,
        run_mode=config.RUN_MODE,
        pixel_img_size=config.PIXEL_IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        device=config.DEVICE,
        random_state=config.RANDOM_STATE,
        geom2_max_images_per_class=config.GEOM2_MAX_IMAGES_PER_CLASS,
    )


def build_experiments():
    return build_experiment_grid(
        config.EXPERIMENT_CLIENT_COUNTS,
        config.EXPERIMENT_DISTRIBUTIONS,
        config.EXPERIMENT_PARTICIPATION_RATES,
        config.EXPERIMENT_METHODS,
    )


def finish_run(result):
    """Refresh Tableau extracts after a completed experiment workflow."""
    if getattr(config, "TABLEAU_AUTO_REFRESH", False):
        from reporting.tableau_export import export_tableau_data

        counts = export_tableau_data()
        print(
            "[tableau-export] refreshed "
            + ", ".join(f"{name}={count}" for name, count in counts.items())
        )
    return result


def main():
    base_random_state = config.RANDOM_STATE
    config.initialize_runtime()
    config.print_runtime_summary()

    if config.RUN_TASK == "segmentation_centralized":
        from experiments.segmentation_structured_comparison_runner import (
            run_structured_segmentation_comparison,
        )

        return finish_run(run_structured_segmentation_comparison(config))

    if config.RUN_TASK == "segmentation_federated":
        from experiments.segmentation_federated_runner import (
            run_federated_segmentation_comparison,
        )

        return finish_run(run_federated_segmentation_comparison(config))

    prepared_data = prepare_datasets()
    if not prepared_data.sources:
        raise RuntimeError("No configured datasets could be prepared.")
    if config.RUN_MODE == "traditional" and not prepared_data.all_numpy_data:
        raise RuntimeError("RUN_MODE='traditional' but no traditional data was prepared.")

    experiments = build_experiments()
    summary_csv = config.EXP_BASE_DIR / "_summary.csv"
    initialize_summary_csv(summary_csv)

    traditional_model_map = {}
    if config.RUN_MODE in {"traditional", "both"}:
        _, _, traditional_model_map = build_traditional_models(
            random_state=config.RANDOM_STATE,
            cache_dir=config.CACHE_DIR,
            features=None,
            svm_backend=config.SVM_BACKEND,
        )

    results = []
    for experiment in experiments:
        print("\n" + "=" * 100)
        print(
            f"Experiment #{experiment.experiment_id} | Clients={experiment.clients} | "
            f"Distribution={experiment.distribution} | "
            f"Participation={experiment.participation:.0%} | Method={experiment.method}"
        )
        print("=" * 100)

        for source_data in prepared_data.sources.values():
            if config.RUN_FEDERATED_LEARNING and config.RUN_MODE in {"deep", "both"}:
                variants = getattr(config, "COMPARISON_VARIANTS", None)
                if variants is None:
                    variants = [
                        (name, config.USE_PRIVACY_AWARE_AGGREGATION, name)
                        for name in config.DEEP_MODELS
                    ]
                for experiment_seed in config.EXPERIMENT_SEEDS:
                    print(f"\n[Replication] seed={experiment_seed}")
                    for variant in variants:
                        if len(variant) == 3:
                            model_name, privacy_enabled, variant_name = variant
                            client_weight_method = config.CLIENT_WEIGHT_METHOD
                            variant_method = None
                        elif len(variant) == 4:
                            model_name, privacy_enabled, variant_name, client_weight_method = variant
                            variant_method = None
                        elif len(variant) == 5:
                            (
                                model_name, privacy_enabled, variant_name,
                                client_weight_method, variant_method,
                            ) = variant
                        else:
                            raise ValueError(
                                "Each COMPARISON_VARIANTS entry must be "
                                "(model, privacy, name), (model, privacy, name, "
                                "client_weight_method), or (model, privacy, name, "
                                "client_weight_method, server_method)."
                            )
                        if (
                            variant_method is not None
                            and variant_method.strip().lower()
                            != experiment.method.strip().lower()
                        ):
                            continue
                        # Both variants receive exactly the same seed. This
                        # matches initialization, sharding, client selection,
                        # DataLoader shuffling, and Bayesian search randomness.
                        config.RANDOM_STATE = experiment_seed
                        config.set_seed(experiment_seed)
                        results.append(
                            run_deep_experiment(
                                experiment,
                                source_data,
                                model_name,
                                config,
                                summary_csv,
                                privacy_agg_enabled=privacy_enabled,
                                variant_name=variant_name,
                                client_weight_method=client_weight_method,
                                experiment_seed=experiment_seed,
                            )
                        )

            if (
                config.RUN_FEDERATED_LEARNING
                and config.RUN_MODE in {"traditional", "both"}
                and source_data.traditional_data is not None
            ):
                for model_name in config.TRADITIONAL_MODELS:
                    results.append(
                        run_traditional_experiment(
                            experiment,
                            source_data,
                            model_name,
                            traditional_model_map[model_name]["model"],
                            config,
                            summary_csv,
                        )
                    )

    config.RANDOM_STATE = base_random_state
    multiseed_summary = write_multiseed_summary(summary_csv)
    print(f"\n✅ Completed {len(results)} experiment run(s).")
    print(f"Summary: {summary_csv}")
    if multiseed_summary is not None:
        print(f"Multi-seed summary: {multiseed_summary}")
    return results


if __name__ == "__main__":
    main()
