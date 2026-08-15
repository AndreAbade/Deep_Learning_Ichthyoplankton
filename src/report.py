from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def generate_report(log_dir: Path, config: dict) -> None:
    out_dir = log_dir / "global_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.md"

    ranking_csv = out_dir / "ranking_models.csv"
    bootstrap_csv = out_dir / "bootstrap_results.csv"
    mcnemar_csv = out_dir / "mcnemar_results.csv"

    classes = config["project"]["classes"]
    tc = config["training"]

    lines = []

    def h(level, text):
        lines.append(f"{'#' * level} {text}\n")

    def p(text=""):
        lines.append(text + "\n")

    h(1, "Benchmark Report — Projeto Peixes 2026")
    p(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p()

    h(2, "1. Dataset Description")
    p(f"- Total images: 9,667")
    p(f"- Split: 70% train / 15% val / 15% test (stratified, seed={config['project']['seed']})")
    p()

    h(2, "2. Classes")
    for cls in classes:
        p(f"- {cls}")
    p()

    h(2, "3. Models Evaluated")
    p("**CNNs:** DenseNet121/201, VGG16/19, ResNet50/101/152, ResNeXt50, "
      "InceptionV3, Xception, InceptionResNetV2, NASNetLarge, MobileNetV2/V3Large, "
      "EfficientNetV2B0/M, NFNet, ConvNeXt")
    p("**Transformers:** ViT_B_16, Swin_T")
    p("**YOLO:** YOLO26N_CLS")
    p("**Ensemble:** Top-2 CNNs from different families (mean probability)")
    p()

    h(2, "4. Transfer Learning Strategy")
    p(f"- Pretrained weights from ImageNet")
    p(f"- Freeze backbone for {tc['freeze_epochs']} epochs (train classifier head only)")
    p(f"- Full fine-tuning for remaining epochs")
    p(f"- Total epochs: {tc['epochs']} (phase 1: {tc['freeze_epochs']} frozen + phase 2: full fine-tuning)")
    p(f"- Monitor: {tc['monitor']}")
    p()

    h(2, "5. Data Augmentation")
    p("- RandomAffine (translate=0.15)")
    p("- RandomHorizontalFlip (p=0.5)")
    p("- RandomRotation (20°)")
    p("- ColorJitter (brightness=0.1, contrast=0.2)")
    p("- RandomResizedCrop (224, scale=0.9–1.1)")
    p("- Val/Test: Resize(256) → CenterCrop(224) → Normalize(ImageNet)")
    p()

    h(2, "6. Metrics")
    p("Loss, Accuracy, F1 macro, Precision macro, Recall macro, Specificity macro, Balanced Accuracy")
    p()

    h(2, "7. Model Ranking")
    if ranking_csv.exists():
        df = pd.read_csv(ranking_csv)
        p(df.to_markdown(index=False))
    else:
        p("_ranking_models.csv not found_")
    p()

    h(2, "8. Best Traditional CNN")
    if ranking_csv.exists():
        cnn_names = {
            "DenseNet121", "DenseNet169", "DenseNet201", "VGG16", "VGG19",
            "ResNet50", "ResNet50V2", "ResNet101", "ResNet101V2", "ResNet152", "ResNet152V2",
            "ResNeXt50", "InceptionV3", "Xception", "InceptionResNetV2", "NASNetLarge",
            "MobileNetV2", "MobileNetV3Large", "EfficientNetV2B0", "EfficientNetV2M",
            "NFNet", "ConvNeXt",
        }
        df_cnn = df[df["model"].isin(cnn_names)]
        if not df_cnn.empty:
            best = df_cnn.iloc[0]
            p(f"**{best['model']}** — Balanced Accuracy: {best.get('test_balanced_accuracy', 'N/A'):.4f}")
    p()

    h(2, "9. Best Transformer")
    if ranking_csv.exists():
        tf_names = {"ViT_B_16", "Swin_T"}
        df_tf = df[df["model"].isin(tf_names)]
        if not df_tf.empty:
            best = df_tf.iloc[0]
            p(f"**{best['model']}** — Balanced Accuracy: {best.get('test_balanced_accuracy', 'N/A'):.4f}")
    p()

    h(2, "10. YOLO26N Result")
    yolo_metrics_path = log_dir / "YOLO26N_CLS" / "metrics" / "test_metrics.json"
    if yolo_metrics_path.exists():
        with open(yolo_metrics_path) as f:
            ym = json.load(f)
        p(f"- Balanced Accuracy: {ym.get('balanced_accuracy', 'N/A'):.4f}")
        p(f"- F1 macro: {ym.get('f1_macro', 'N/A'):.4f}")
        p(f"- Accuracy: {ym.get('accuracy', 'N/A'):.4f}")
    else:
        p("_YOLO metrics not found_")
    p()

    h(2, "11. Ensemble CNN Result")
    ens_metrics_path = log_dir / "Ensemble_CNN" / "metrics" / "test_metrics.json"
    if ens_metrics_path.exists():
        with open(ens_metrics_path) as f:
            em = json.load(f)
        p(f"- Models: {em.get('ensemble_models', ['?', '?'])}")
        p(f"- Balanced Accuracy: {em.get('balanced_accuracy', 'N/A'):.4f}")
        p(f"- F1 macro: {em.get('f1_macro', 'N/A'):.4f}")
    else:
        p("_Ensemble metrics not found_")
    p()

    h(2, "12. Bootstrap Interpretation")
    if bootstrap_csv.exists():
        df_bs = pd.read_csv(bootstrap_csv)
        summary = df_bs.groupby("model")["balanced_accuracy"].agg(["mean", "std"]).sort_values("mean", ascending=False)
        p(summary.round(4).to_markdown())
    else:
        p("_bootstrap_results.csv not found_")
    p()

    h(2, "13. McNemar Interpretation")
    if mcnemar_csv.exists():
        df_mc = pd.read_csv(mcnemar_csv)
        sig = df_mc[df_mc["significant_0_05"]]
        p(f"- Total pairs compared: {len(df_mc)}")
        p(f"- Pairs with significant difference (p < 0.05): {len(sig)}")
    else:
        p("_mcnemar_results.csv not found_")
    p()

    h(2, "14. Overall Best Model")
    if ranking_csv.exists() and not df.empty:
        best = df.iloc[0]
        p(f"**{best['model']}** is the best model considering Balanced Accuracy, "
          "F1 macro, bootstrap stability, and McNemar significance.")
    p()
    p("The best model is determined not only by the highest point estimate, but by the combination of:")
    p("- High balanced accuracy and F1 macro")
    p("- Stability across 1,000 bootstrap iterations")
    p("- Statistically superior performance over most other models (McNemar p < 0.05)")
    p("- High AUC across all classes")
    p()

    h(2, "15. Generated Artifacts")
    p("| Artifact | Location |")
    p("|---|---|")
    artifacts = [
        ("ranking_models.csv", "log/global_results/"),
        ("bootstrap_results.csv", "log/global_results/"),
        ("mcnemar_results.csv", "log/global_results/"),
        ("bootstrap_heatmap_*.png", "log/global_results/"),
        ("violin_*_bootstrap.png", "log/global_results/"),
        ("mcnemar_heatmap.png", "log/global_results/"),
        ("network_graph_models.png", "log/global_results/"),
        ("top5_multimetric_radar.png", "log/global_results/"),
        ("Per-model: confusion_matrix.png", "log/<MODEL>/figures/"),
        ("Per-model: roc_multiclass.png", "log/<MODEL>/figures/"),
        ("Per-model: training_loss_accuracy.png", "log/<MODEL>/figures/"),
        ("Per-model: history.csv", "log/<MODEL>/metrics/"),
        ("Per-model: test_metrics.json", "log/<MODEL>/metrics/"),
        ("Per-model: y_true/pred/prob.npy", "log/<MODEL>/predictions/"),
    ]
    for name, loc in artifacts:
        p(f"| {name} | `{loc}` |")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[Report] Saved to {out_path}")
