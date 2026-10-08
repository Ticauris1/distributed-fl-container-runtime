"""Traditional-model federated training and ensemble evaluation."""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import confusion_matrix


def federated_training_traditional_ensemble(model_template, client_dataloaders, server_val):
    print(f"Starting federated training (ensemble) for {type(model_template).__name__}...")
    features_val, labels_val = server_val
    client_models, client_cms = [], []
    for index, loader_group in enumerate(client_dataloaders):
        print(f"  Training client {index + 1}/{len(client_dataloaders)}...")
        model = clone(model_template)
        feature_batches, label_batches = [], []
        for features, labels in loader_group["train"]:
            feature_batches.append(features.numpy())
            label_batches.append(labels.numpy())
        if not feature_batches:
            continue
        features = np.concatenate(feature_batches)
        labels = np.concatenate(label_batches)
        model.fit(features, labels)
        client_models.append(model)
        client_cms.append(confusion_matrix(labels_val, model.predict(features_val), labels=np.unique(labels_val)))
    print("Federated ensemble training complete.")
    return client_models, client_cms


def federated_training_traditional_linear(model_template, client_dataloaders, rounds=10):
    print(f"Starting federated training (linear) for {type(model_template).__name__}...")
    global_model = clone(model_template)
    sample_features, sample_labels = next(iter(client_dataloaders[0]["train"]))
    global_model.fit(sample_features.numpy(), sample_labels.numpy())
    for round_number in range(rounds):
        clients = []
        print(f"  Round {round_number + 1}/{rounds}...")
        for loader_group in client_dataloaders:
            model = clone(global_model)
            feature_batches, label_batches = [], []
            for features, labels in loader_group["train"]:
                feature_batches.append(features.numpy())
                label_batches.append(labels.numpy())
            if not feature_batches:
                continue
            model.fit(np.concatenate(feature_batches), np.concatenate(label_batches))
            clients.append(model)
        if clients:
            global_model.coef_ = np.mean([model.coef_ for model in clients], axis=0)
            global_model.intercept_ = np.mean([model.intercept_ for model in clients], axis=0)
    print("Federated training for linear model complete.")
    return global_model


def get_proba_and_classes(model, features):
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
        return probabilities, np.asarray(getattr(model, "classes_", np.arange(probabilities.shape[1])))
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=float)
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)
            classes = np.asarray([0, 1])
        else:
            classes = np.arange(scores.shape[1])
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= np.clip(probabilities.sum(axis=1, keepdims=True), 1e-12, None)
        return probabilities, classes
    raise AttributeError("Model must implement predict_proba or decision_function.")


def align_probs_to_classes(probabilities, model_classes, target_classes):
    target_index = {class_id: index for index, class_id in enumerate(target_classes)}
    aligned = np.zeros((probabilities.shape[0], len(target_classes)), dtype=float)
    for model_index, class_id in enumerate(model_classes):
        target = target_index.get(class_id)
        if target is not None:
            aligned[:, target] = probabilities[:, model_index]
    return aligned


def evaluate_federated_ensemble(client_models, X_test, y_test=None, class_labels=None, explain=True, k=3, weights=None, **kwargs):
    k = kwargs.get("show_k", k)
    explain = kwargs.get("verbose", explain)
    if not client_models:
        return None, None
    raw = [get_proba_and_classes(model, X_test) for model in client_models]
    class_union = None
    for _, classes in raw:
        class_union = classes if class_union is None else np.unique(np.concatenate([class_union, classes]))
    if class_labels:
        class_union = np.arange(len(class_labels))
    probabilities = [align_probs_to_classes(values, classes, class_union) for values, classes in raw]
    client_count = len(probabilities)
    sample_count, class_count = probabilities[0].shape
    if weights is None:
        normalized_weights = np.full(client_count, 1.0 / client_count)
    else:
        normalized_weights = np.asarray(weights, dtype=float).reshape(-1)
        if normalized_weights.shape != (client_count,):
            raise ValueError(f"weights must have shape {(client_count,)}, got {normalized_weights.shape}")
        normalized_weights /= normalized_weights.sum() or 1.0
    average = np.tensordot(normalized_weights, np.stack(probabilities), axes=(0, 0))
    predictions = np.argmax(average, axis=1)
    if explain:
        print("\n[Server aggregation]")
        print(f"  Clients: {client_count} | After alignment: (N={sample_count}, C={class_count})")
        print(f"  Weights: {np.round(normalized_weights, 4).tolist()}  (sum={float(normalized_weights.sum()):.4f})")
        if y_test is not None:
            print(f"  Quick accuracy on provided y_test: {float((predictions == np.asarray(y_test)).mean()) * 100:.2f}%")
    return predictions, average


_get_proba_and_classes = get_proba_and_classes
_align_probs_to_classes = align_probs_to_classes
