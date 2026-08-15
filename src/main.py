"""
Benchmark entry point.

Usage examples:
    python src/main.py --mode split_data
    python src/main.py --mode train_one --model DenseNet121
    python src/main.py --mode train_all
    python src/main.py --mode evaluate_all
    python src/main.py --mode ensemble
    python src/main.py --mode bootstrap
    python src/main.py --mode mcnemar
    python src/main.py --mode report
    python src/main.py --mode full_pipeline
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Make src/ importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from utils import ensure_dir, load_yaml, set_seed

BASE_DIR = Path(__file__).parent.parent


def load_configs() -> tuple[dict, dict]:
    config = load_yaml(BASE_DIR / "configs" / "config.yaml")
    models_cfg = load_yaml(BASE_DIR / "configs" / "models.yaml")
    return config, models_cfg


def all_model_cfgs(models_cfg: dict) -> list[dict]:
    return (
        models_cfg.get("cnn_models", [])
        + models_cfg.get("transformer_models", [])
        + models_cfg.get("yolo_models", [])
    )


def record_failure(log_dir: Path, model_name: str, err: Exception) -> None:
    failed_csv = log_dir / "global_results" / "failed_models.csv"
    ensure_dir(failed_csv.parent)
    file_exists = failed_csv.exists()
    with open(failed_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "error_message", "traceback", "datetime"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "model": model_name,
            "error_message": str(err),
            "traceback": traceback.format_exc(),
            "datetime": datetime.now().isoformat(),
        })
    print(f"[FAILED] {model_name}: {err}")


def build_ranking(log_dir: Path, out_dir: Path) -> None:
    import pandas as pd
    rows = []
    for model_dir in sorted(log_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "global_results":
            continue
        metrics_path = model_dir / "metrics" / "test_metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        row = {"model": model_dir.name}
        for k, v in m.items():
            if isinstance(v, (int, float)):
                row[f"test_{k}"] = round(v, 6)
        rows.append(row)

    if not rows:
        print("[Ranking] No test metrics found.")
        return

    df = pd.DataFrame(rows)
    sort_cols = [c for c in ["test_balanced_accuracy", "test_f1_macro", "test_accuracy"] if c in df.columns]
    df = df.sort_values(sort_cols, ascending=False).reset_index(drop=True)
    ensure_dir(out_dir)
    df.to_csv(out_dir / "ranking_models.csv", index=False)

    summary = df.to_dict(orient="records")
    with open(out_dir / "global_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Ranking] {len(df)} models ranked. Saved to {out_dir / 'ranking_models.csv'}")


# ──────────────────────────────────────────────────────────────────────────────
# Mode handlers
# ──────────────────────────────────────────────────────────────────────────────

def mode_split_data(config: dict, **_) -> None:
    from dataset import (
        save_class_indices,
        save_split_indices,
        split_dataset,
    )
    from overfitting import build_dataset_table

    dataset_dir = BASE_DIR / config["paths"]["dataset_dir"]
    data_dir    = BASE_DIR / config["paths"]["data_dir"]
    log_dir     = BASE_DIR / config["paths"]["log_dir"]
    out_dir     = log_dir / "global_results"
    ratios      = (config["split"]["train"], config["split"]["val"], config["split"]["test"])
    seed        = config["project"]["seed"]

    print(f"Splitting dataset with ratios {ratios} and seed={seed} ...")
    split_indices = split_dataset(dataset_dir, data_dir, ratios, seed)
    save_split_indices(split_indices, log_dir)
    save_class_indices(log_dir)

    build_dataset_table(data_dir, out_dir,
                        total_epochs=config["training"]["epochs"])
    print("Split complete.")


def mode_train_one(config: dict, models_cfg: dict, model_name: str, **_) -> None:
    from train import train_model
    from yolo_runner import train_yolo

    log_dir = BASE_DIR / config["paths"]["log_dir"]
    cfg_list = all_model_cfgs(models_cfg)
    matches = [m for m in cfg_list if m["name"] == model_name]
    if not matches:
        raise ValueError(f"Model '{model_name}' not found in models.yaml")
    model_cfg = matches[0]

    set_seed(config["project"]["seed"])
    if model_cfg["backend"] == "ultralytics":
        train_yolo(model_cfg, config, BASE_DIR)
    else:
        train_model(model_cfg, config, BASE_DIR)


def mode_train_all(config: dict, models_cfg: dict, force: bool = False, **_) -> None:
    from train import train_model
    from yolo_runner import train_yolo

    log_dir = BASE_DIR / config["paths"]["log_dir"]
    for model_cfg in all_model_cfgs(models_cfg):
        name = model_cfg["name"]
        ckpt = log_dir / name / "checkpoints" / "best_model.pt"
        if not force and ckpt.exists():
            print(f"[Skip] {name}: checkpoint already exists, use --force to retrain")
            continue
        try:
            set_seed(config["project"]["seed"])
            if model_cfg["backend"] == "ultralytics":
                train_yolo(model_cfg, config, BASE_DIR)
            else:
                train_model(model_cfg, config, BASE_DIR)
        except Exception as e:
            record_failure(log_dir, name, e)


def mode_evaluate_all(config: dict, models_cfg: dict, **_) -> None:
    from evaluate import evaluate_model
    from plots import generate_model_plots
    from yolo_runner import evaluate_yolo

    log_dir = BASE_DIR / config["paths"]["log_dir"]
    out_dir = log_dir / "global_results"

    for model_cfg in all_model_cfgs(models_cfg):
        name = model_cfg["name"]
        ckpt = log_dir / name / "checkpoints" / "best_model.pt"
        if not ckpt.exists():
            print(f"[Skip] {name}: no checkpoint found")
            continue
        try:
            if model_cfg["backend"] == "ultralytics":
                evaluate_yolo(model_cfg, config, BASE_DIR)
            else:
                evaluate_model(model_cfg, config, BASE_DIR)
            generate_model_plots(name, log_dir)
        except Exception as e:
            record_failure(log_dir, name, e)

    build_ranking(log_dir, out_dir)

    from plots import (plot_top5_radar, plot_bar_comparison,
                       plot_fit_scatter, plot_fit_heatmap,
                       plot_learning_curves_grid, plot_parallel_coords)
    from overfitting import (analyze_overfitting, build_hyperparams_table,
                              build_dataset_table)

    model_families = models_cfg.get("model_families", {})
    ranking_csv    = out_dir / "ranking_models.csv"
    if ranking_csv.exists():
        plot_top5_radar(ranking_csv, out_dir)
        plot_bar_comparison(ranking_csv, out_dir, model_families=model_families)
        plot_parallel_coords(ranking_csv, out_dir, model_families=model_families)

    # Dataset split summary + augmentation table
    data_dir = BASE_DIR / config["paths"]["data_dir"]
    build_dataset_table(data_dir, out_dir,
                        total_epochs=config["training"]["epochs"])

    # Hyperparameter table
    build_hyperparams_table(log_dir, out_dir)

    # Overfitting / underfitting analysis and visualisations
    fit_csv = out_dir / "overfitting_analysis.csv"
    analyze_overfitting(log_dir, out_dir)
    if fit_csv.exists():
        plot_fit_scatter(fit_csv, out_dir)
        plot_fit_heatmap(fit_csv, out_dir)
        plot_learning_curves_grid(log_dir, fit_csv, out_dir)

    # Bootstrap analysis — violin, heatmap and CI bar plots
    print("=== Bootstrap analysis ===")
    from bootstrap import run_bootstrap
    bs_cfg = config.get("bootstrap", {})
    run_bootstrap(
        log_dir, out_dir,
        n_iterations=bs_cfg.get("n_iterations", 1000),
        seed=bs_cfg.get("random_state", 42),
        metrics_to_eval=bs_cfg.get("metrics"),
        model_families=model_families,
    )

    # McNemar pairwise test — heatmap and network graph
    print("=== McNemar analysis ===")
    from mcnemar import run_mcnemar
    run_mcnemar(log_dir, out_dir, model_families=model_families)


def mode_ensemble(config: dict, models_cfg: dict, **_) -> None:
    from ensemble import build_ensemble
    from plots import (plot_top5_radar, plot_bar_comparison, plot_parallel_coords)
    from bootstrap import run_bootstrap
    from mcnemar import run_mcnemar

    log_dir = BASE_DIR / config["paths"]["log_dir"]
    out_dir = log_dir / "global_results"
    ranking_csv = out_dir / "ranking_models.csv"

    if not ranking_csv.exists():
        print("[Ensemble] Run evaluate_all first to generate ranking_models.csv")
        return

    model_families = models_cfg.get("model_families", {})
    build_ensemble(log_dir, ranking_csv, model_families)
    build_ranking(log_dir, out_dir)

    # Regenerate global plots now that Ensemble_CNN is in the ranking
    print("[Ensemble] Regenerating global plots with Ensemble_CNN included...")
    if ranking_csv.exists():
        plot_top5_radar(ranking_csv, out_dir)
        plot_bar_comparison(ranking_csv, out_dir, model_families=model_families)
        plot_parallel_coords(ranking_csv, out_dir, model_families=model_families)

    bs_cfg = config.get("bootstrap", {})
    run_bootstrap(
        log_dir, out_dir,
        n_iterations=bs_cfg.get("n_iterations", 1000),
        seed=bs_cfg.get("random_state", 42),
        metrics_to_eval=bs_cfg.get("metrics"),
        model_families=model_families,
    )
    run_mcnemar(log_dir, out_dir, model_families=model_families)


def mode_bootstrap(config: dict, models_cfg: dict, **_) -> None:
    from bootstrap import run_bootstrap

    log_dir        = BASE_DIR / config["paths"]["log_dir"]
    out_dir        = log_dir / "global_results"
    bs_cfg         = config.get("bootstrap", {})
    model_families = models_cfg.get("model_families", {})
    run_bootstrap(
        log_dir,
        out_dir,
        n_iterations=bs_cfg.get("n_iterations", 1000),
        seed=bs_cfg.get("random_state", 42),
        metrics_to_eval=bs_cfg.get("metrics"),
        model_families=model_families,
    )


def mode_mcnemar(config: dict, models_cfg: dict, **_) -> None:
    from mcnemar import run_mcnemar

    log_dir        = BASE_DIR / config["paths"]["log_dir"]
    out_dir        = log_dir / "global_results"
    model_families = models_cfg.get("model_families", {})
    run_mcnemar(log_dir, out_dir, model_families=model_families)


def mode_report(config: dict, **_) -> None:
    from report import generate_report

    log_dir = BASE_DIR / config["paths"]["log_dir"]
    generate_report(log_dir, config)


def mode_full_pipeline(config: dict, models_cfg: dict, force: bool = False, **_) -> None:
    print("=== Step 1/7: Split data ===")
    mode_split_data(config)
    print("=== Step 2/7: Train all ===")
    mode_train_all(config, models_cfg, force=force)
    print("=== Step 3/7: Evaluate all ===")
    mode_evaluate_all(config, models_cfg)
    print("=== Step 4/7: Ensemble ===")
    mode_ensemble(config, models_cfg)
    print("=== Step 5/7: Bootstrap ===")
    mode_bootstrap(config)
    print("=== Step 6/7: McNemar ===")
    mode_mcnemar(config)
    print("=== Step 7/7: Report ===")
    mode_report(config)
    print("=== Pipeline complete ===")


# ──────────────────────────────────────────────────────────────────────────────

MODES = {
    "split_data": mode_split_data,
    "train_one": mode_train_one,
    "train_all": mode_train_all,
    "evaluate_all": mode_evaluate_all,
    "ensemble": mode_ensemble,
    "bootstrap": mode_bootstrap,
    "mcnemar": mode_mcnemar,
    "report": mode_report,
    "full_pipeline": mode_full_pipeline,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fish benchmark pipeline")
    parser.add_argument("--mode", required=True, choices=list(MODES))
    parser.add_argument("--model", default=None, help="Model name for train_one mode")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (overrides config.yaml)")
    parser.add_argument("--force", action="store_true",
                        help="Retrain models even if best_model.pt already exists")
    args = parser.parse_args()

    config, models_cfg = load_configs()
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    MODES[args.mode](config=config, models_cfg=models_cfg, model_name=args.model, force=args.force)


if __name__ == "__main__":
    main()
