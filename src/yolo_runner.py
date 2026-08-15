from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

# ImageNet stats used by get_loaders — needed to undo normalisation before YOLO inference
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

from dataset import CLASSES, get_filenames_from_loader, get_loaders
from metrics import compute_metrics
from plots import generate_model_plots
from utils import ensure_dir, save_environment


MODEL_NAME = "YOLO26N_CLS"


def train_yolo(model_cfg: dict, config: dict, base_dir: Path) -> None:
    key = model_cfg["key"]
    batch_size = model_cfg["batch_size"]
    input_size = model_cfg["input_size"][0]

    log_dir = base_dir / "log" / MODEL_NAME
    ensure_dir(log_dir)
    save_environment(log_dir)

    data_dir = base_dir / config["paths"]["data_dir"]
    tc = config["training"]

    model = YOLO(key)
    model.train(
        data=str(data_dir),
        epochs=tc["epochs"],
        imgsz=input_size,
        batch=batch_size,
        lr0=tc["max_lr"],
        lrf=tc["base_lr"] / tc["max_lr"],
        momentum=tc["momentum"],
        weight_decay=tc["weight_decay"],
        patience=0,  # disabled — fixed epochs for fair comparison
        project=str(log_dir / "ultralytics_run"),
        name="train",
        exist_ok=True,
        seed=config["project"]["seed"],
        task="classify",
        verbose=True,
    )

    # Copy best weights to standard location
    best_src = log_dir / "ultralytics_run" / "train" / "weights" / "best.pt"
    ckpt_dir = ensure_dir(log_dir / "checkpoints")
    if best_src.exists():
        shutil.copy(best_src, ckpt_dir / "best_model.pt")
        last_src = log_dir / "ultralytics_run" / "train" / "weights" / "last.pt"
        if last_src.exists():
            shutil.copy(last_src, ckpt_dir / "last_model.pt")


def evaluate_yolo(model_cfg: dict, config: dict, base_dir: Path) -> dict:
    input_size = model_cfg["input_size"][0]
    batch_size = model_cfg["batch_size"]

    log_dir = base_dir / "log" / MODEL_NAME
    ckpt_path = log_dir / "checkpoints" / "best_model.pt"
    pred_dir = ensure_dir(log_dir / "predictions")
    metrics_dir = ensure_dir(log_dir / "metrics")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"YOLO checkpoint not found: {ckpt_path}")

    data_dir = base_dir / config["paths"]["data_dir"]
    model = YOLO(str(ckpt_path))

    _, _, test_loader = get_loaders(
        data_dir, batch_size, input_size, config["training"]["num_workers"]
    )

    all_probs, all_preds, all_targets = [], [], []

    for inputs, targets in test_loader:
        # Undo ImageNet normalisation → [0, 1] float, as expected by YOLO
        inputs_raw = (inputs * _IMAGENET_STD + _IMAGENET_MEAN).clamp(0.0, 1.0)
        results = model.predict(inputs_raw, verbose=False, task="classify")
        for r, t in zip(results, targets.numpy()):
            probs = np.array(r.probs.data.cpu().numpy(), dtype=np.float32)
            all_probs.append(probs)
            all_preds.append(int(np.argmax(probs)))
            all_targets.append(int(t))

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.stack(all_probs)

    np.save(pred_dir / "y_true.npy", y_true)
    np.save(pred_dir / "y_pred.npy", y_pred)
    np.save(pred_dir / "y_prob.npy", y_prob)

    filenames = get_filenames_from_loader(test_loader)
    with open(pred_dir / "filenames.csv", "w", newline="") as f:
        csv.writer(f).writerows([["filename"]] + [[fn] for fn in filenames])

    test_m = compute_metrics(y_true, y_pred, y_prob=y_prob)
    with open(metrics_dir / "test_metrics.json", "w") as f:
        json.dump(test_m, f, indent=2)

    print(f"[{MODEL_NAME}] Test metrics: {test_m}")
    return test_m
