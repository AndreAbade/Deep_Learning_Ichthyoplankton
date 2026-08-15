from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import get_loaders
from metrics import compute_metrics
from models import build_model, freeze_backbone, unfreeze_backbone, wrap_multigpu
from utils import ensure_dir, get_device, save_config_used, save_environment, set_seed



def _run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    scheduler,
    device: torch.device,
    training: bool,
    desc: str = "",
) -> tuple[float, np.ndarray, np.ndarray]:
    model.train(training)
    total_loss = 0.0
    all_preds, all_targets = [], []

    phase = "train" if training else "val"
    pbar = tqdm(loader, desc=f"{desc} {phase}", leave=False, dynamic_ncols=True)

    with torch.set_grad_enabled(training):
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)

            # InceptionV3 returns (output, aux) during training
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, targets)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            all_targets.append(targets.cpu().numpy())

    n = len(loader.dataset)
    return total_loss / n, np.concatenate(all_preds), np.concatenate(all_targets)


def train_model(model_cfg: dict, config: dict, base_dir: Path) -> None:
    set_seed(config["project"]["seed"])
    device = get_device()

    name = model_cfg["name"]
    backend = model_cfg["backend"]
    key = model_cfg["key"]
    batch_size = model_cfg["batch_size"]
    input_size = model_cfg["input_size"][0]

    log_dir = base_dir / "log" / name
    ckpt_dir = ensure_dir(log_dir / "checkpoints")
    metrics_dir = ensure_dir(log_dir / "metrics")

    save_config_used({**config, "model": model_cfg}, log_dir)
    save_environment(log_dir)

    data_dir = base_dir / config["paths"]["data_dir"]
    train_loader, val_loader, _ = get_loaders(
        data_dir, batch_size, input_size, config["training"]["num_workers"]
    )

    model = build_model(name, backend, key, num_classes=config["project"]["num_classes"])
    model = model.to(device)

    tc = config["training"]
    freeze_epochs = tc["freeze_epochs"]
    total_epochs = tc["epochs"]
    monitor = tc["monitor"]
    base_lr = tc["base_lr"]
    max_lr = tc["max_lr"]
    step_size_up = tc["step_size_up"]

    criterion = nn.CrossEntropyLoss()

    def make_optimizer_and_scheduler(lr_base: float):
        opt = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr_base,
            momentum=tc["momentum"],
            weight_decay=tc["weight_decay"],
        )
        sched = torch.optim.lr_scheduler.CyclicLR(
            opt,
            base_lr=lr_base,
            max_lr=max_lr,
            step_size_up=step_size_up * len(train_loader),
            mode=tc["mode"],
            cycle_momentum=True,
        )
        return opt, sched

    history: list[dict] = []
    best_val_monitor = -float("inf")
    best_metrics: dict = {}

    # Phase 1: frozen backbone
    freeze_backbone(model, backend, key)
    model = wrap_multigpu(model)
    optimizer, scheduler = make_optimizer_and_scheduler(max_lr)  # higher LR for head only

    for epoch in range(1, total_epochs + 1):
        if epoch == freeze_epochs + 1:
            # Phase 2: unfreeze backbone and rebuild optimizer with lower LR
            raw = model.module if isinstance(model, nn.DataParallel) else model
            unfreeze_backbone(raw)
            optimizer, scheduler = make_optimizer_and_scheduler(base_lr)

        ep_tag = f"[{name}] Ep {epoch:03d}/{total_epochs}"
        t0 = time.time()
        train_loss, train_pred, train_true = _run_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, training=True, desc=ep_tag
        )
        val_loss, val_pred, val_true = _run_epoch(
            model, val_loader, criterion, None, None, device, training=False, desc=ep_tag
        )

        train_m = compute_metrics(train_true, train_pred, train_loss)
        val_m = compute_metrics(val_true, val_pred, val_loss)

        row = {"epoch": epoch, "time_s": round(time.time() - t0, 1)}
        for k, v in train_m.items():
            row[f"train_{k}"] = round(v, 6)
        for k, v in val_m.items():
            row[f"val_{k}"] = round(v, 6)
        history.append(row)

        monitor_val = val_m[monitor.replace("val_", "")]
        improved = monitor_val > best_val_monitor

        print(
            f"[{name}] Ep {epoch:03d}/{total_epochs} | "
            f"train_loss={train_m['loss']:.4f} val_loss={val_m['loss']:.4f} | "
            f"val_ba={val_m['balanced_accuracy']:.4f} val_f1={val_m['f1_macro']:.4f}"
            + (" ✓" if improved else "")
        )

        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(raw_model.state_dict(), ckpt_dir / "last_model.pt")

        if improved:
            best_val_monitor = monitor_val
            torch.save(raw_model.state_dict(), ckpt_dir / "best_model.pt")
            best_metrics = {**row, "monitor_value": monitor_val}

    # Save history CSV
    if history:
        keys = list(history[0].keys())
        with open(metrics_dir / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(history)

    # Save best epoch metrics
    with open(metrics_dir / "best_epoch_metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    # Save hyperparameters (LR range, batch, epochs, etc.)
    hyperparams = {
        "name": name,
        "backend": backend,
        "key": key,
        "total_epochs": total_epochs,
        "batch_size": batch_size,
        "input_size": input_size,
        "optimizer": "SGD",
        "momentum": tc["momentum"],
        "weight_decay": tc["weight_decay"],
        "scheduler": "CyclicLR",
        "scheduler_mode": tc["mode"],
        "freeze_epochs": freeze_epochs,
        "lr_phase1_fixed": max_lr,       # Fase 1: backbone congelado, LR fixo no max
        "lr_phase2_min": base_lr,        # Fase 2: backbone livre, LR cicla entre min e max
        "lr_phase2_max": max_lr,
        "step_size_up_batches": step_size_up * len(train_loader),
        "n_train_samples": len(train_loader.dataset),
        "n_val_samples": len(val_loader.dataset),
        "monitor": monitor,
    }
    with open(metrics_dir / "hyperparams.json", "w") as f:
        json.dump(hyperparams, f, indent=2)
