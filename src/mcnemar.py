from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from tqdm import tqdm

from plots import plot_mcnemar_heatmap, plot_network_graph
from utils import ensure_dir


def run_mcnemar(log_dir: Path, out_dir: Path,
                model_families: dict | None = None) -> None:
    model_dirs = sorted([
        d for d in log_dir.iterdir()
        if d.is_dir() and d.name != "global_results"
        and (d / "predictions" / "y_true.npy").exists()
        and (d / "predictions" / "y_pred.npy").exists()
    ])

    # Load all predictions
    preds = {}
    y_true_ref = None
    for d in model_dirs:
        y_true = np.load(d / "predictions" / "y_true.npy")
        y_pred = np.load(d / "predictions" / "y_pred.npy")
        preds[d.name] = (y_true, y_pred)
        y_true_ref = y_true

    models = list(preds.keys())
    rows = []

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            ma, mb = models[i], models[j]
            y_true, pred_a = preds[ma]
            _, pred_b = preds[mb]

            correct_a = pred_a == y_true
            correct_b = pred_b == y_true

            b = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
            c = int(np.sum(~correct_a & correct_b))   # A wrong, B right

            table = np.array([[0, b], [c, 0]])  # we only need b and c
            # Use (b+c) contingency for chi2 McNemar
            contingency = np.array([[np.sum(correct_a & correct_b), b],
                                     [c, np.sum(~correct_a & ~correct_b)]])
            result = mcnemar(contingency, exact=False, correction=True)

            rows.append({
                "model_a": ma,
                "model_b": mb,
                "b": b,
                "c": c,
                "statistic": round(result.statistic, 6),
                "p_value": round(result.pvalue, 8),
                "significant_0_05": result.pvalue < 0.05,
            })

    ensure_dir(out_dir)
    out_csv = out_dir / "mcnemar_results.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[McNemar] {len(df)} pairs saved to {out_csv}")

    plot_mcnemar_heatmap(out_csv, out_dir)
    ranking_csv = out_dir / "ranking_models.csv"
    plot_network_graph(out_csv, out_dir,
                       ranking_csv=ranking_csv if ranking_csv.exists() else None,
                       model_families=model_families)
