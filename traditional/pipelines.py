"""Explicit traditional-model construction, grids, and CV helpers."""

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import KernelPCA, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import ParameterGrid, cross_validate, learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler
from sklearn.svm import LinearSVC, SVC


class SafePCA(PCA):
    def fit(self, features, labels=None):
        if isinstance(self.n_components, (int, np.integer)):
            self.n_components = min(self.n_components, features.shape[0], features.shape[1])
        return super().fit(features, labels)


def _feature_steps(features):
    if features is None:
        return []
    return [features] if isinstance(features, tuple) and len(features) == 2 else [("feats", features)]


def build_traditional_models(random_state, cache_dir, features=None, svm_backend="svc"):
    if svm_backend == "kcpa":
        svm_backend = "kpca"
    memory = Memory(location=str(cache_dir), verbose=0) if cache_dir else None
    pipelines = {
        "KNN": Pipeline([
            ("vt", VarianceThreshold(1e-4)), ("scaler", StandardScaler()),
            ("pca", PCA(n_components=384, svd_solver="randomized", random_state=random_state)),
            ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance", metric="cosine", algorithm="brute", leaf_size=30, n_jobs=-1)),
        ], memory=memory),
        "Random Forest": Pipeline([
            *_feature_steps(features), ("vt", VarianceThreshold(1e-4)),
            ("pca", PCA(n_components=128, svd_solver="randomized", random_state=random_state)),
            ("clf", RandomForestClassifier(n_jobs=-1, class_weight="balanced_subsample", random_state=random_state)),
        ], memory=memory),
        "Naïve Bayes": Pipeline([
            *_feature_steps(features), ("vt", VarianceThreshold(1e-4)),
            ("gaussify", PowerTransformer(method="yeo-johnson", standardize=True)),
            ("pca", PCA(n_components=64, whiten=True, svd_solver="randomized", random_state=random_state)),
            ("clf", GaussianNB()),
        ], memory=memory),
    }
    if svm_backend == "svc":
        pipelines["SVM"] = Pipeline([
            ("vt", VarianceThreshold(1e-4)), ("scaler", StandardScaler()),
            ("pca", PCA(n_components=128, whiten=True, svd_solver="randomized", random_state=random_state)),
            ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", cache_size=1024, tol=1e-3, shrinking=True, random_state=random_state)),
        ], memory=memory)
    elif svm_backend == "linearsvc_calibrated":
        pipelines["SVM"] = Pipeline([
            ("vt", VarianceThreshold(1e-4)), ("scaler", StandardScaler()),
            ("pca", PCA(n_components=64, svd_solver="randomized", random_state=random_state)),
            ("clf", CalibratedClassifierCV(estimator=LinearSVC(dual=False, class_weight="balanced", tol=1e-3, max_iter=8000, random_state=random_state), method="sigmoid", cv=3)),
        ], memory=memory)
    elif svm_backend == "kpca":
        pipelines["SVM"] = Pipeline([
            ("vt", VarianceThreshold(1e-4)), ("scaler", StandardScaler()),
            ("kpca", KernelPCA(kernel="rbf", random_state=random_state)),
            ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state)),
        ], memory=memory)
    else:
        raise ValueError("svm_backend must be 'svc', 'linearsvc_calibrated', or 'kpca'.")
    # Templates used by the federated traditional runner preserve the original
    # config.py model definitions; the richer pipelines above remain available
    # for the explicit CV/grid-search workflow.
    model_map = {
        "LogReg": {"model": LogisticRegression(random_state=random_state, max_iter=1000)},
        "SVM": {"model": SVC(random_state=random_state, probability=True)},
        "Random Forest": {"model": RandomForestClassifier(random_state=random_state)},
        "KNN": {"model": KNeighborsClassifier()},
        "Naïve Bayes": {"model": GaussianNB()},
    }
    return pipelines, build_parameter_grids(random_state, svm_backend), model_map


def build_parameter_grids(random_state, svm_backend="svc", knn_wide_grid=True):
    if svm_backend == "kcpa":
        svm_backend = "kpca"
    knn_micro = [{"pca__n_components": [32, 0.90, 0.95, 0.99], "clf__n_neighbors": [3, 5, 7, 11], "clf__weights": ["distance"], "clf__metric": ["cosine"], "clf__algorithm": ["brute"], "clf__leaf_size": [15, 30, 60]}]
    knn_wide = [
        {"pca__n_components": [32, 0.90, 0.95, 0.99], "clf__n_neighbors": [3, 5, 7, 11, 15], "clf__weights": ["uniform", "distance"], "clf__algorithm": ["auto"]},
        {"pca__n_components": [32, 0.90, 0.95, 0.99], "clf__n_neighbors": [3, 5, 7, 11, 15], "clf__weights": ["uniform", "distance"], "clf__algorithm": ["auto", "brute"]},
    ]
    grids = {
        "KNN": knn_wide if knn_wide_grid else knn_micro,
        "Random Forest": {"pca__n_components": [32, 0.90, 0.95, 0.99], "clf__n_estimators": [300, 600], "clf__max_depth": [None, 40], "clf__min_samples_split": [2, 5], "clf__min_samples_leaf": [1, 2], "clf__max_features": ["sqrt", 0.5]},
        "Naïve Bayes": {"gaussify": [PowerTransformer(), QuantileTransformer(output_distribution="normal", subsample=200000, random_state=random_state)], "pca__whiten": [True, False], "pca__n_components": [32, 0.90, 0.95, 0.99], "clf__var_smoothing": np.logspace(-10, -1, 12)},
    }
    if svm_backend == "svc":
        grids["SVM"] = [{"pca__n_components": [32, 0.90, 0.95, 0.99], "clf__C": [1, 3, 10, 30], "clf__gamma": ["scale", 3e-3, 1e-2]}]
    elif svm_backend == "linearsvc_calibrated":
        grids["SVM"] = [{"pca__n_components": [32, 0.90, 0.95, 0.99], "clf__estimator__C": [0.25, 1.0, 4.0, 16.0]}]
    else:
        grids["SVM"] = [{"kpca__kernel": ["rbf"], "kpca__n_components": [192, 256, 320], "kpca__gamma": [1e-5, 2e-5, 3e-5, 5e-5], "clf__kernel": ["linear"], "clf__C": [4, 6, 8, 10, 12]}]
    return grids


def generate_learning_curve_history(estimator, features, labels, cv=3):
    sizes, train_scores, validation_scores = learning_curve(estimator, features, labels, cv=cv, scoring="accuracy", n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    return [{"epoch": size, "train_acc": np.mean(train_scores[index]), "val_acc": np.mean(validation_scores[index])} for index, size in enumerate(sizes)]


def train_sklearn_with_pipeline_on_pixels(model_key, X_train, y_train, X_test, y_test, *, pipelines, parameter_grids, cv=3, scoring=None, refit="f1", n_jobs=-1, error_score="raise"):
    scoring = scoring or {"acc": "accuracy", "f1": "f1_macro", "nll": "neg_log_loss"}
    pipeline, grid = pipelines[model_key], parameter_grids.get(model_key, [])
    grids = [grid] if isinstance(grid, dict) else grid
    candidates = [params for candidate in grids for params in ParameterGrid(candidate)]
    refit_key = {"accuracy": "acc", "acc": "acc", "f1": "f1", "f1_macro": "f1", "neg_log_loss": "nll", "nll": "nll"}.get(refit, "f1")
    rows, best_score, best_params = [], -np.inf, None
    for params in candidates:
        try:
            result = cross_validate(clone(pipeline).set_params(**params), X_train, y_train, scoring=scoring, cv=cv, n_jobs=n_jobs, error_score=error_score)
        except Exception:
            continue
        acc = np.mean(result.get("test_acc", [np.nan])); f1 = np.mean(result.get("test_f1", [np.nan])); nll = -np.mean(result.get("test_nll", [np.nan]))
        rows.append({"params": params, "mean_test_acc": acc, "mean_test_f1": f1, "mean_test_nll": nll})
        score = {"acc": acc, "f1": f1, "nll": -nll if np.isfinite(nll) else -np.inf}[refit_key]
        if np.isfinite(score) and score > best_score:
            best_score, best_params = score, params
    best = clone(pipeline if best_params is None else pipeline.set_params(**best_params)).fit(X_train, y_train)
    predictions = best.predict(X_test)
    try:
        probabilities = best.predict_proba(X_test)
        test_nll = log_loss(y_test, probabilities, labels=np.unique(y_train))
    except Exception:
        probabilities, test_nll = None, None
    metrics = {"acc": accuracy_score(y_test, predictions), "f1": f1_score(y_test, predictions, average="macro"), "nll": test_nll}
    return best, probabilities, y_test, pd.DataFrame(rows), metrics
