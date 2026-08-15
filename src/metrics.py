from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def specificity_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    specificities = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        denom = tn + fp
        specificities.append(tn / denom if denom > 0 else 0.0)
    return float(np.mean(specificities))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loss: float | None = None,
    y_prob: np.ndarray | None = None,
) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "specificity_macro": specificity_macro(y_true, y_pred),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    if y_prob is not None:
        from sklearn.metrics import roc_auc_score
        try:
            metrics["macro_auc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
        except ValueError:
            metrics["macro_auc"] = float("nan")
    if loss is not None:
        metrics["loss"] = float(loss)
    return metrics
