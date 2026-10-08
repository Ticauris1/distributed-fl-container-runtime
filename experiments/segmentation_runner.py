"""Centralized U-Net++ segmentation experiment runner."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import torch

from data.segmentation import build_segmentation_loaders
from models.segmentation import (
    make_segmentation_model,
    normalize_segmentation_encoder_name,
    normalize_segmentation_model_name,
)
from reporting.segmentation import save_history, save_metrics, save_prediction_panels
from training.segmentation import evaluate_segmentation, make_segmentation_loss, train_segmentation_epoch


def run_segmentation_condition(
    config, model_name, encoder_name, train_fraction=1.0,
    experiment_name="centralized", train_manifest=None,
    evaluation_manifest=None, dataset_label=None,
    augmentation_profile="baseline", learning_rate=None,
    experiment_seed=None, variant_name=None, scheduler_name="none",
    encoder_freeze_epochs=0, encoder_learning_rate=None,
    loss_name="bce_dice", positive_class_weight=5.0,
    attention="none", cbam_reduction=16, cbam_spatial_kernel_size=7,
):
    training_started = perf_counter()
    experiment_seed = (
        config.RANDOM_STATE if experiment_seed is None else int(experiment_seed)
    )
    config.set_seed(experiment_seed)
    train_manifest = train_manifest or config.SEGMENTATION_MANIFEST
    evaluation_manifest = evaluation_manifest or train_manifest
    # Rebuild loaders for every model so each comparison receives the same
    # seeded training order and augmentations.
    train_loader, val_loader, test_loader = build_segmentation_loaders(
        train_manifest,
        config.SEGMENTATION_IMAGE_SIZE,
        config.SEGMENTATION_BATCH_SIZE,
        config.SEGMENTATION_NUM_WORKERS,
        experiment_seed,
        train_fraction=train_fraction,
        evaluation_manifest_path=evaluation_manifest,
        augmentation_profile=augmentation_profile,
    )
    print(
        f"[segmentation] train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}"
    )
    normalized_name = normalize_segmentation_model_name(model_name)
    normalized_encoder = normalize_segmentation_encoder_name(encoder_name)
    if experiment_name == "centralized":
        output_dir = (
            Path(config.SEGMENTATION_RESULT_DIR)
            / f"centralized_{normalized_name}_{normalized_encoder}"
        )
    else:
        output_dir = (
            Path(config.SEGMENTATION_RESULT_DIR)
            / experiment_name
            / f"{normalized_name}_{normalized_encoder}"
        )
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = make_segmentation_model(
        model_name,
        config.DEVICE,
        encoder_name,
        config.SEGMENTATION_ENCODER_WEIGHTS,
        attention=attention,
        cbam_reduction=cbam_reduction,
        cbam_spatial_kernel_size=cbam_spatial_kernel_size,
    )
    criterion = make_segmentation_loss(
        loss_name, positive_class_weight=positive_class_weight
    )
    learning_rate = (
        config.SEGMENTATION_LEARNING_RATE
        if learning_rate is None else float(learning_rate)
    )
    encoder_freeze_epochs = int(encoder_freeze_epochs)
    if encoder_freeze_epochs < 0:
        raise ValueError("Encoder freeze epochs cannot be negative")
    encoder_learning_rate = (
        learning_rate
        if encoder_learning_rate is None else float(encoder_learning_rate)
    )
    if encoder_learning_rate <= 0:
        raise ValueError("Encoder learning rate must be positive")
    encoder_parameters = list(model.encoder.parameters())
    encoder_parameter_ids = {id(parameter) for parameter in encoder_parameters}
    decoder_parameters = [
        parameter for parameter in model.parameters()
        if id(parameter) not in encoder_parameter_ids
    ]
    if encoder_freeze_epochs:
        for parameter in encoder_parameters:
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": encoder_learning_rate},
            {"params": decoder_parameters, "lr": learning_rate},
        ],
        weight_decay=config.SEGMENTATION_WEIGHT_DECAY,
    )
    scheduler_name = str(scheduler_name).strip().lower()
    if scheduler_name == "none":
        scheduler = None
    elif scheduler_name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6,
        )
    elif scheduler_name == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(config.SEGMENTATION_EPOCHS), eta_min=1e-6,
        )
    else:
        raise ValueError(f"Unsupported segmentation scheduler: {scheduler_name}")
    best_path = checkpoint_dir / f"best_{normalized_name}.pth"
    best_dice = float("-inf")
    best_epoch = 0
    patience = 0
    history = []
    for epoch in range(1, config.SEGMENTATION_EPOCHS + 1):
        encoder_frozen = epoch <= encoder_freeze_epochs
        if epoch == encoder_freeze_epochs + 1 and encoder_freeze_epochs:
            for parameter in encoder_parameters:
                parameter.requires_grad_(True)
        encoder_lr_used = optimizer.param_groups[0]["lr"]
        decoder_lr_used = optimizer.param_groups[1]["lr"]
        train_loss = train_segmentation_epoch(
            model, train_loader, optimizer, criterion, config.DEVICE,
            freeze_encoder=encoder_frozen,
        )
        validation = evaluate_segmentation(
            model, val_loader, criterion, config.DEVICE,
            config.SEGMENTATION_MASK_THRESHOLD,
        )
        if scheduler_name == "reduce_on_plateau":
            scheduler.step(validation["loss"])
        elif scheduler is not None:
            scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": validation["loss"],
            "val_dice": validation["dice"],
            "val_iou": validation["iou"],
            "val_cracked_sample_dice": validation["mean_cracked_sample_dice"],
            "val_empty_mask_accuracy": validation["empty_mask_accuracy"],
            "learning_rate": decoder_lr_used,
            "encoder_learning_rate": encoder_lr_used,
            "decoder_learning_rate": decoder_lr_used,
            "encoder_frozen": encoder_frozen,
            "next_encoder_learning_rate": optimizer.param_groups[0]["lr"],
            "next_decoder_learning_rate": optimizer.param_groups[1]["lr"],
        }
        history.append(row)
        print(
            f"[segmentation] epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={validation['loss']:.4f} val_dice={validation['dice']:.4f} "
            f"val_iou={validation['iou']:.4f}"
        )
        if validation["dice"] > best_dice + 1e-8:
            best_dice = validation["dice"]
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if patience >= config.SEGMENTATION_EARLY_STOPPING_PATIENCE:
                print(f"[segmentation] early stopping at epoch {epoch}")
                break
    save_history(output_dir / "training_history.csv", history)
    model.load_state_dict(torch.load(best_path, map_location=config.DEVICE, weights_only=True))
    test_metrics, probabilities, _ = evaluate_segmentation(
        model, test_loader, criterion, config.DEVICE,
        config.SEGMENTATION_MASK_THRESHOLD, return_predictions=True,
    )
    test_metrics.update({
        "model": normalized_name,
        "encoder": normalized_encoder,
        "seed": experiment_seed,
        "train_fraction": float(train_fraction),
        "dataset_label": dataset_label or Path(train_manifest).parent.name,
        "variant_name": variant_name or "default",
        "augmentation_profile": augmentation_profile,
        "learning_rate": learning_rate,
        "encoder_learning_rate": encoder_learning_rate,
        "encoder_freeze_epochs": encoder_freeze_epochs,
        "scheduler": scheduler_name,
        "loss_name": loss_name,
        "positive_class_weight": float(positive_class_weight),
        "attention": str(attention),
        "cbam_reduction": int(cbam_reduction),
        "cbam_spatial_kernel_size": int(cbam_spatial_kernel_size),
        "training_manifest": str(Path(train_manifest).resolve()),
        "evaluation_manifest": str(Path(evaluation_manifest).resolve()),
        "best_validation_dice": best_dice,
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "training_type": "centralized",
        "configured_epochs": int(config.SEGMENTATION_EPOCHS),
        "image_size": int(config.SEGMENTATION_IMAGE_SIZE),
        "batch_size": int(config.SEGMENTATION_BATCH_SIZE),
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(val_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "training_seconds": perf_counter() - training_started,
        "checkpoint_size_mb": best_path.stat().st_size / (1024 ** 2),
    })
    save_metrics(output_dir / "test_metrics.json", test_metrics)
    save_prediction_panels(
        test_loader.dataset,
        probabilities,
        output_dir / "test_predictions",
        config.SEGMENTATION_MASK_THRESHOLD,
        config.SEGMENTATION_PREVIEW_COUNT,
    )
    print(f"[segmentation] test metrics: {test_metrics}")
    print(f"[segmentation] results: {output_dir}")
    return test_metrics


def run_centralized_segmentation(config):
    results = []
    for model_name in config.SEGMENTATION_MODELS:
        for encoder_name in config.DEEP_MODELS:
            print(
                f"[segmentation] starting model={model_name} "
                f"encoder={encoder_name}"
            )
            results.append(run_segmentation_condition(config, model_name, encoder_name))
    return results
