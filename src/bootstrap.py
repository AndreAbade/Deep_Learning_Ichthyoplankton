from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from metrics import compute_metrics
from plots import (plot_bootstrap_heatmap, plot_violin_bootstrap,
                   plot_bootstrap_ci_bars, plot_bootstrap_dual_bars,
                   plot_bar_comparison)
from utils import ensure_dir


def run_bootstrap(
    log_dir: Path,
    out_dir: Path,
    n_iterations: int = 1000,
    seed: int = 42,
    metrics_to_eval: list[str] | None = None,
    model_families: dict | None = None,
) -> None:
    if metrics_to_eval is None:
        metrics_to_eval = ["accuracy", "f1_macro", "balanced_accuracy",
                           "precision_macro", "recall_macro", "specificity_macro"]

    rng = np.random.default_rng(seed)
    rows = []

    model_dirs = [d for d in log_dir.iterdir() if d.is_dir() and d.name != "global_results"]

    for model_dir in tqdm(sorted(model_dirs), desc="Bootstrap"):
        pred_dir = model_dir / "predictions"
        y_true_path = pred_dir / "y_true.npy"
        y_pred_path = pred_dir / "y_pred.npy"
        if not (y_true_path.exists() and y_pred_path.exists()):
            continue

        y_true = np.load(y_true_path)
        y_pred = np.load(y_pred_path)
        n = len(y_true)

        for it in range(n_iterations):
            idx = rng.integers(0, n, size=n)
            m = compute_metrics(y_true[idx], y_pred[idx])
            row = {"model": model_dir.name, "iteration": it}
            for key in metrics_to_eval:
                row[key] = m.get(key, np.nan)
            rows.append(row)

    if not rows:
        print("[Bootstrap] No model predictions found.")
        return

    ensure_dir(out_dir)
    out_csv = out_dir / "bootstrap_results.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[Bootstrap] Saved {len(df)} rows to {out_csv}")

    ranking_csv = out_dir / "ranking_models.csv"
    actual_csv  = ranking_csv if ranking_csv.exists() else None

    for metric in ["accuracy", "f1_macro", "balanced_accuracy",
                   "precision_macro", "recall_macro", "specificity_macro"]:
        if metric in df.columns:
            plot_bootstrap_heatmap(out_csv, metric, out_dir)
            plot_violin_bootstrap(out_csv, metric, out_dir,
                                  actual_csv=actual_csv)
            plot_bootstrap_ci_bars(out_csv, metric, out_dir,
                                   actual_csv=actual_csv,
                                   n_boot=n_iterations)

    # Combined F1 + Accuracy chart (fig6 style)
    plot_bootstrap_dual_bars(out_csv, out_dir,
                             actual_csv=actual_csv,
                             n_boot=n_iterations)

    # Re-generate bar comparison with CI now that bootstrap is done
    if ranking_csv.exists():
        plot_bar_comparison(ranking_csv, out_dir,
                            bootstrap_csv=out_csv,
                            n_boot=n_iterations,
                            model_families=model_families)
