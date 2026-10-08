"""Shared experiment metric helpers."""

import numpy as np #type: ignore
from sklearn.metrics import roc_auc_score #type: ignore
from sklearn.preprocessing import label_binarize #type: ignore


def safe_auc_multiclass(y_true, probabilities, n_classes):
    try:
        if probabilities is None:
            return np.nan
        binary = label_binarize(y_true, classes=list(range(n_classes)))
        if binary.shape[1] < 2:
            return np.nan
        return roc_auc_score(binary, probabilities, average="macro", multi_class="ovr")
    except Exception:
        return np.nan


def macro_prec_recall_from_report(report_dict):
    try:
        macro = report_dict.get("macro avg", {})
        return macro.get("precision", np.nan), macro.get("recall", np.nan), macro.get("f1-score", np.nan)
    except Exception:
        return np.nan, np.nan, np.nan


_safe_auc_multiclass = safe_auc_multiclass
_macro_prec_recall_from_report = macro_prec_recall_from_report
