"""Overfitting/underfitting analysis, hyperparameter table, and dataset summary."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import ensure_dir


# ──────────────────────────────────────────────────────────────────────────────
# Dataset split summary table
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset_table(data_dir: Path, out_dir: Path,
                        total_epochs: int = 100) -> pd.DataFrame:
    """Count images per class per split; compute augmented total (online aug).

    Online augmentation (RandomAffine + Flip + Rotation + ColorJitter +
    RandomResizedCrop) is applied fresh every epoch, so the total number of
    distinct augmented presentations = train_count × total_epochs.

    Saves:
      - dataset_summary.csv
      - dataset_summary_table.png  (publication-ready figure)
    """
    from dataset import CLASSES

    splits = ("train", "val", "test")
    rows: list[dict] = []

    for cls in CLASSES:
        counts: dict[str, int] = {}
        for split in splits:
            split_dir = data_dir / split / cls
            if split_dir.exists():
                counts[split] = sum(
                    1 for f in split_dir.iterdir()
                    if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
                )
            else:
                counts[split] = 0

        total = sum(counts[s] for s in splits)
        aug_total = counts["train"] * total_epochs
        rows.append({
            "class":     cls,
            "train":     counts["train"],
            "val":       counts["val"],
            "test":      counts["test"],
            "total":     total,
            "aug_total": aug_total,
        })

    df = pd.DataFrame(rows)

    # Totals row
    totals = {
        "class":     "TOTAL",
        "train":     df["train"].sum(),
        "val":       df["val"].sum(),
        "test":      df["test"].sum(),
        "total":     df["total"].sum(),
        "aug_total": df["aug_total"].sum(),
    }
    df_full = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

    ensure_dir(out_dir)
    csv_path = out_dir / "dataset_summary.csv"
    df_full.to_csv(csv_path, index=False)
    print(f"[Dataset] Summary saved: {csv_path}")

    # ── figure ────────────────────────────────────────────────────────────────
    _plot_dataset_table(df_full, out_dir, total_epochs)

    return df_full


def _plot_dataset_table(df: pd.DataFrame, out_dir: Path,
                        total_epochs: int) -> None:
    """Render a publication-ready table figure."""
    col_labels = ["Class", "Train", "Val", "Test", "Total", f"Aug Total\n(train×{total_epochs} epochs)"]
    col_keys   = ["class", "train", "val", "test", "total", "aug_total"]

    n_rows = len(df)
    fig_h  = 0.42 * n_rows + 1.4
    fig, ax = plt.subplots(figsize=(13, max(3.5, fig_h)))
    ax.axis("off")

    cell_data = [[str(df.loc[i, k]) for k in col_keys] for i in range(n_rows)]

    # Format numbers with thousand separators (except "class" column)
    for r_idx in range(n_rows):
        for c_idx, key in enumerate(col_keys[1:], start=1):
            v = df.loc[r_idx, key]
            cell_data[r_idx][c_idx] = f"{int(v):,}"

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.55)

    # Header style
    header_color = "#2c3e6b"
    for col in range(len(col_labels)):
        cell = tbl[0, col]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")

    # Totals row (last row) — distinct style
    totals_row = n_rows - 1
    for col in range(len(col_labels)):
        cell = tbl[totals_row + 1, col]   # +1 because row 0 is header
        cell.set_facecolor("#dce6f4")
        cell.set_text_props(fontweight="bold")

    # Alternate row colours
    for row in range(1, n_rows):  # skip totals row (handled above)
        for col in range(len(col_labels)):
            cell = tbl[row, col]
            if row % 2 == 0:
                cell.set_facecolor("#f7f9fc")
            else:
                cell.set_facecolor("white")

    # Class column bold
    for row in range(1, n_rows + 1):
        tbl[row, 0].set_text_props(fontweight="bold")

    aug_transforms = (
        "RandomAffine(translate=0.15)  |  RandomHorizontalFlip(p=0.5)  |  "
        "RandomRotation(20°)  |  ColorJitter  |  RandomResizedCrop(224)"
    )
    ax.set_title(
        "Dataset Split Summary\n"
        f"Augmentation (online, per epoch): {aug_transforms}",
        fontsize=10, pad=16, loc="left",
    )

    plt.tight_layout()
    fig_path = out_dir / "dataset_summary_table.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Dataset] Table figure saved: {fig_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Tabela de hiperparâmetros
# ──────────────────────────────────────────────────────────────────────────────

def build_hyperparams_table(log_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Agrega hyperparams.json de cada modelo em uma tabela CSV."""
    rows = []
    for model_dir in sorted(log_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "global_results":
            continue
        hp_path = model_dir / "metrics" / "hyperparams.json"
        if not hp_path.exists():
            continue
        with open(hp_path) as f:
            hp = json.load(f)

        # Registra métricas finais de treino (melhor época) se disponíveis
        best_path = model_dir / "metrics" / "best_epoch_metrics.json"
        best_val_ba = None
        if best_path.exists():
            with open(best_path) as f:
                bm = json.load(f)
            best_val_ba = bm.get("val_balanced_accuracy") or bm.get("monitor_value")

        rows.append({
            "model": hp.get("name", model_dir.name),
            "backend": hp.get("backend", ""),
            "total_epochs": hp.get("total_epochs"),
            "freeze_epochs": hp.get("freeze_epochs"),
            "batch_size": hp.get("batch_size"),
            "input_size": hp.get("input_size"),
            "optimizer": hp.get("optimizer"),
            "momentum": hp.get("momentum"),
            "weight_decay": hp.get("weight_decay"),
            "scheduler": hp.get("scheduler"),
            "scheduler_mode": hp.get("scheduler_mode"),
            "lr_phase1_fixed": hp.get("lr_phase1_fixed"),
            "lr_phase2_min": hp.get("lr_phase2_min"),
            "lr_phase2_max": hp.get("lr_phase2_max"),
            "step_size_up_batches": hp.get("step_size_up_batches"),
            "monitor": hp.get("monitor"),
            "n_train_samples": hp.get("n_train_samples"),
            "n_val_samples": hp.get("n_val_samples"),
            "best_val_ba": round(best_val_ba, 4) if best_val_ba is not None else None,
        })

    ensure_dir(out_dir)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "hyperparams_table.csv", index=False)
    print(f"[Hyperparams] Table saved: {out_dir / 'hyperparams_table.csv'}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Análise de over/underfitting
# ──────────────────────────────────────────────────────────────────────────────

def _val_loss_slope(df: pd.DataFrame, frac: float = 0.30) -> float:
    """Slope da val_loss no último `frac` das épocas (positivo → aumentando)."""
    n = len(df)
    last_n = max(3, int(n * frac))
    tail = df["val_loss"].values[-last_n:]
    x = np.arange(last_n, dtype=float)
    return float(np.polyfit(x, tail, 1)[0])


def _classify_fit(acc_gap: float, final_val_acc: float, slope: float) -> str:
    """
    Classifica o ajuste do modelo com base em três sinais:
      acc_gap   — diferença treino_acc - val_acc na última época
      final_val_acc — acurácia de validação na última época
      slope     — tendência da val_loss no final do treino

    Regras (em ordem de prioridade):
      overfitting      : gap > 10 pp  E val_loss subindo
      mild_overfitting : gap entre 5–10 pp  OU (gap > 10 pp E val_loss estável)
      underfitting     : val_acc < 70% E gap < 5 pp
      good_fit         : demais casos
    """
    if acc_gap > 0.10 and slope > 1e-4:
        return "overfitting"
    if acc_gap > 0.05 or (acc_gap > 0.10 and slope <= 1e-4):
        return "mild_overfitting"
    if final_val_acc < 0.70 and acc_gap < 0.05:
        return "underfitting"
    return "good_fit"


def analyze_overfitting(log_dir: Path, out_dir: Path) -> pd.DataFrame:
    """
    Para cada modelo com history.csv, calcula indicadores de over/underfitting
    e salva overfitting_analysis.csv em out_dir.
    """
    rows = []
    for model_dir in sorted(log_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "global_results":
            continue
        hist_path = model_dir / "metrics" / "history.csv"
        if not hist_path.exists():
            continue

        df = pd.read_csv(hist_path)
        if len(df) < 3:
            continue

        needed = {"train_accuracy", "val_accuracy", "train_loss", "val_loss"}
        if not needed.issubset(df.columns):
            continue

        final = df.iloc[-1]
        final_train_acc = float(final["train_accuracy"])
        final_val_acc   = float(final["val_accuracy"])
        final_train_loss = float(final["train_loss"])
        final_val_loss   = float(final["val_loss"])

        acc_gap  = final_train_acc  - final_val_acc
        loss_gap = final_val_loss   - final_train_loss
        slope    = _val_loss_slope(df)

        best_val_acc   = float(df["val_accuracy"].max())
        best_epoch     = int(df["val_accuracy"].idxmax()) + 1
        best_val_ba    = float(df["val_balanced_accuracy"].max()) \
                         if "val_balanced_accuracy" in df.columns else None

        # Indicadores extras
        train_acc_delta = final_train_acc - float(df.iloc[0]["train_accuracy"])
        val_acc_delta   = final_val_acc   - float(df.iloc[0]["val_accuracy"])

        fit_status = _classify_fit(acc_gap, final_val_acc, slope)

        rows.append({
            "model": model_dir.name,
            "fit_status": fit_status,
            "final_train_acc": round(final_train_acc, 4),
            "final_val_acc":   round(final_val_acc,   4),
            "acc_gap":         round(acc_gap,          4),
            "final_train_loss": round(final_train_loss, 4),
            "final_val_loss":   round(final_val_loss,   4),
            "loss_gap":         round(loss_gap,          4),
            "val_loss_trend_slope": round(slope,         6),
            "best_val_acc":    round(best_val_acc,    4),
            "best_val_ba":     round(best_val_ba, 4) if best_val_ba is not None else None,
            "best_epoch":      best_epoch,
            "total_epochs":    len(df),
            "train_acc_delta": round(train_acc_delta, 4),
            "val_acc_delta":   round(val_acc_delta,   4),
        })

    ensure_dir(out_dir)
    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_dir / "overfitting_analysis.csv", index=False)
    print(f"[Overfitting] Analysis saved: {out_dir / 'overfitting_analysis.csv'}")
    return result_df
