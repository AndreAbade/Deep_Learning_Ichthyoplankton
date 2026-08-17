from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import CLASSES
from metrics import compute_metrics
from plots import generate_model_plots
from utils import ensure_dir


CNN_NAMES = {
    "DenseNet121", "DenseNet169", "DenseNet201",
    "VGG16", "VGG19",
    "ResNet50", "ResNet50V2", "ResNet101", "ResNet101V2", "ResNet152", "ResNet152V2",
    "ResNeXt50",
    "InceptionV3", "Xception", "InceptionResNetV2",
    "NASNetLarge",
    "MobileNetV2", "MobileNetV3Large",
    "EfficientNetV2B0", "EfficientNetV2M",
    "NFNet", "ConvNeXt",
}


def select_top2_cnn(log_dir: Path, model_families: dict) -> tuple[str, str]:
    """Pick the two best CNNs from distinct families by VALIDATION accuracy.

    Selection must never look at the test set: choosing members on test
    performance and then reporting the ensemble's test score would make that
    score an optimistic, in-sample estimate. The ranking is therefore taken
    from each model's best-epoch validation metrics, not from ranking_models.csv
    (which holds test metrics only).
    """
    candidates = []
    for name in sorted(CNN_NAMES):
        best = log_dir / name / "metrics" / "best_epoch_metrics.json"
        if not best.exists():
            continue  # not trained in this run
        with open(best) as f:
            val_ba = json.load(f).get("val_balanced_accuracy")
        if val_ba is not None:
            candidates.append((float(val_ba), name))

    selected = []
    used_families = set()
    for val_ba, m in sorted(candidates, reverse=True):
        fam = model_families.get(m, m)
        if fam not in used_families:
            selected.append(m)
            used_families.add(fam)
            print(f"[Ensemble] {m}: val_balanced_accuracy = {val_ba:.4f}")
        if len(selected) == 2:
            break

    if len(selected) < 2:
        raise RuntimeError(f"Could not find 2 CNNs from different families. Found: {selected}")

    print(f"[Ensemble] Models: {selected[0]} + {selected[1]}")
    return selected[0], selected[1]


def build_ensemble(log_dir: Path, ranking_csv: Path, model_families: dict) -> None:
    # ranking_csv is kept in the signature for call-site compatibility but is
    # deliberately unused: selection is validation-based (see select_top2_cnn).
    m1, m2 = select_top2_cnn(log_dir, model_families)

    y_prob1 = np.load(log_dir / m1 / "predictions" / "y_prob.npy")
    y_prob2 = np.load(log_dir / m2 / "predictions" / "y_prob.npy")
    y_true = np.load(log_dir / m1 / "predictions" / "y_true.npy")

    ensemble_prob = (y_prob1 + y_prob2) / 2.0
    ensemble_pred = np.argmax(ensemble_prob, axis=1)

    ens_dir = log_dir / "Ensemble_CNN"
    pred_dir = ensure_dir(ens_dir / "predictions")
    metrics_dir = ensure_dir(ens_dir / "metrics")
    fig_dir = ensure_dir(ens_dir / "figures")

    np.save(pred_dir / "y_true.npy", y_true)
    np.save(pred_dir / "y_pred.npy", ensemble_pred)
    np.save(pred_dir / "y_prob.npy", ensemble_prob)

    # Copy filenames from first model
    src_filenames = log_dir / m1 / "predictions" / "filenames.csv"
    if src_filenames.exists():
        import shutil
        shutil.copy(src_filenames, pred_dir / "filenames.csv")

    test_m = compute_metrics(y_true, ensemble_pred, y_prob=ensemble_prob)
    test_m["ensemble_models"] = [m1, m2]
    with open(metrics_dir / "test_metrics.json", "w") as f:
        json.dump(test_m, f, indent=2)

    # Figures
    from plots import plot_confusion_matrix, plot_roc_multiclass, plot_radar_model
    plot_confusion_matrix(y_true, ensemble_pred, fig_dir, "Ensemble_CNN")
    plot_roc_multiclass(y_true, ensemble_prob, fig_dir, "Ensemble_CNN")
    plot_radar_model(test_m, fig_dir, "Ensemble_CNN")

    print(f"[Ensemble_CNN] Metrics: {test_m}")
