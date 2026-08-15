from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm import tqdm

from dataset import CLASSES, get_loaders, get_filenames_from_loader
from metrics import compute_metrics
from models import build_model
from utils import ensure_dir, get_device, set_seed


def evaluate_model(model_cfg: dict, config: dict, base_dir: Path) -> dict:
    set_seed(config["project"]["seed"])
    device = get_device()

    name = model_cfg["name"]
    backend = model_cfg["backend"]
    key = model_cfg["key"]
    batch_size = model_cfg["batch_size"]
    input_size = model_cfg["input_size"][0]

    log_dir = base_dir / "log" / name
    ckpt_path = log_dir / "checkpoints" / "best_model.pt"
    pred_dir = ensure_dir(log_dir / "predictions")
    metrics_dir = ensure_dir(log_dir / "metrics")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    data_dir = base_dir / config["paths"]["data_dir"]
    _, _, test_loader = get_loaders(
        data_dir, batch_size, input_size, config["training"]["num_workers"]
    )

    model = build_model(name, backend, key, num_classes=config["project"]["num_classes"])
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    all_preds, all_probs, all_targets = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc=f"Eval {name}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_probs.append(probs)
            all_preds.append(preds)
            all_targets.append(targets.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs, axis=0)
    avg_loss = total_loss / len(test_loader.dataset)

    # Save arrays
    np.save(pred_dir / "y_true.npy", y_true)
    np.save(pred_dir / "y_pred.npy", y_pred)
    np.save(pred_dir / "y_prob.npy", y_prob)

    # Save filenames
    filenames = get_filenames_from_loader(test_loader)
    with open(pred_dir / "filenames.csv", "w", newline="") as f:
        csv.writer(f).writerows([["filename"]] + [[fn] for fn in filenames])

    # Compute and save metrics (y_prob enables Macro AUC)
    test_m = compute_metrics(y_true, y_pred, avg_loss, y_prob=y_prob)
    with open(metrics_dir / "test_metrics.json", "w") as f:
        json.dump(test_m, f, indent=2)

    # Classification report
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    report_rows = []
    for label, vals in report.items():
        if isinstance(vals, dict):
            report_rows.append({"class": label, **vals})
    with open(metrics_dir / "classification_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1-score", "support"])
        w.writeheader()
        w.writerows(report_rows)

    print(f"[{name}] Test metrics: {test_m}")
    return test_m
