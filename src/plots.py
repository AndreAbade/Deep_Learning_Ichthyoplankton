from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from dataset import CLASSES

FIG_DPI = 600

# ── dark style constants (fig1) ──────────────────────────────────────────────
_DARK_BG    = "#0d1117"
_DARK_PANEL = "#161b22"
_DARK_TEXT  = "#e6edf3"
_DARK_GRID  = "#30363d"

# ── architecture family → colour (used in network graph, parallel coords) ───
_FAMILY_COLORS = {
    "DenseNet":    "#4c8ef5",
    "VGG":         "#56c16e",
    "ResNet":      "#f5a623",
    "ResNeXt":     "#e8784a",
    "Inception":   "#c678dd",
    "Xception":    "#c678dd",
    "NASNet":      "#e06c75",
    "MobileNet":   "#61afef",
    "EfficientNet":"#98c379",
    "NFNet":       "#e5c07b",
    "ConvNeXt":    "#56b6c2",
    "Transformer": "#abb2bf",
    "YOLO":        "#e8524a",
    "Ensemble":    "#f0c040",
    "CNN":         "#888888",
}

_FIT_COLORS = {
    "good_fit":         "#2ca02c",
    "mild_overfitting": "#ff7f0e",
    "overfitting":      "#d62728",
    "underfitting":     "#1f77b4",
}
_FIT_LABELS = {
    "good_fit":         "Good fit",
    "mild_overfitting": "Mild overfitting",
    "overfitting":      "Overfitting",
    "underfitting":     "Underfitting",
}


def _model_family(model_name: str, model_families: dict | None) -> str:
    if model_families and model_name in model_families:
        return model_families[model_name]
    if "YOLO" in model_name:
        return "YOLO"
    if model_name in ("ViT_B_16", "Swin_T"):
        return "Transformer"
    if "Ensemble" in model_name:
        return "Ensemble"
    return "CNN"


# ──────────────────────────────────────────────────────────────────────────────
# Per-model plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          fig_dir: Path, model_name: str) -> None:
    for normalize, suffix in [(False, ""), (True, "_normalized")]:
        cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)
        fig, ax = plt.subplots(figsize=(8, 7))
        fmt = ".2f" if normalize else "d"
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{model_name} — Confusion Matrix{'(Normalized)' if normalize else ''}")
        plt.tight_layout()
        fig.savefig(fig_dir / f"confusion_matrix{suffix}.png", dpi=FIG_DPI)
        plt.close(fig)


def plot_roc_multiclass(y_true: np.ndarray, y_prob: np.ndarray,
                        fig_dir: Path, model_name: str) -> None:
    n_classes = len(CLASSES)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    fpr_all, tpr_all = {}, {}

    for i, (cls, col) in enumerate(zip(CLASSES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        fpr_all[i] = fpr
        tpr_all[i] = tpr
        ax.plot(fpr, tpr, color=col, lw=1.5, label=f"{cls} (AUC={roc_auc:.3f})")

    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    micro_auc = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, "k--", lw=2, label=f"Micro-avg (AUC={micro_auc:.3f})")

    all_fpr = np.unique(np.concatenate([fpr_all[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_all[i], tpr_all[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, "b-", lw=2, label=f"Macro-avg (AUC={macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], "gray", lw=1, linestyle=":")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} — ROC Multiclass (OvR)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(fig_dir / "roc_multiclass.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_training_curves(history_csv: Path, fig_dir: Path, model_name: str) -> None:
    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss")
    axes[0].plot(df["epoch"], df["val_loss"],   label="Val Loss")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train_accuracy"], label="Train Acc")
    axes[1].plot(df["epoch"], df["val_accuracy"],   label="Val Acc")
    if "val_f1_macro" in df.columns:
        axes[1].plot(df["epoch"], df["val_f1_macro"], label="Val F1 macro", linestyle="--")
    if "val_balanced_accuracy" in df.columns:
        axes[1].plot(df["epoch"], df["val_balanced_accuracy"], label="Val Bal. Acc", linestyle=":")
    axes[1].set_title(f"{model_name} — Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(fig_dir / "training_loss_accuracy.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_radar_model(test_metrics: dict, fig_dir: Path, model_name: str) -> None:
    keys   = ["accuracy", "f1_macro", "precision_macro", "recall_macro",
              "specificity_macro", "balanced_accuracy", "macro_auc"]
    labels = ["Accuracy", "F1 macro", "Precision", "Recall",
              "Specificity", "Bal. Acc.", "Macro AUC"]
    values = [test_metrics.get(k, 0) for k in keys]

    angles  = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, "b-o", linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title(f"{model_name} — Radar Metrics", pad=20)
    plt.tight_layout()
    fig.savefig(fig_dir / "radar_metrics.png", dpi=FIG_DPI)
    plt.close(fig)


def generate_model_plots(model_name: str, log_dir: Path) -> None:
    pred_dir    = log_dir / model_name / "predictions"
    metrics_dir = log_dir / model_name / "metrics"
    fig_dir     = log_dir / model_name / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.load(pred_dir / "y_true.npy")
    y_pred = np.load(pred_dir / "y_pred.npy")
    y_prob = np.load(pred_dir / "y_prob.npy")

    plot_confusion_matrix(y_true, y_pred, fig_dir, model_name)
    plot_roc_multiclass(y_true, y_prob, fig_dir, model_name)

    history_csv = metrics_dir / "history.csv"
    if history_csv.exists():
        plot_training_curves(history_csv, fig_dir, model_name)

    import json
    with open(metrics_dir / "test_metrics.json") as f:
        test_m = json.load(f)
    plot_radar_model(test_m, fig_dir, model_name)


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap plots  (fig1 and fig2 style)
# ──────────────────────────────────────────────────────────────────────────────

def plot_violin_bootstrap(bootstrap_csv: Path, metric: str, out_dir: Path,
                          actual_csv: Path | None = None) -> None:
    """Violin plot with white background suitable for publication.

    Dark horizontal bar = median, thick grey bar = 95% CI,
    coloured circle = actual test value (from actual_csv / ranking_models.csv).
    """
    df = pd.read_csv(bootstrap_csv)
    if metric not in df.columns:
        return
    order = (df.groupby("model")[metric].mean()
               .sort_values(ascending=False).index.tolist())

    actual: dict[str, float] = {}
    if actual_csv is not None and Path(actual_csv).exists():
        rdf = pd.read_csv(actual_csv)
        col = f"test_{metric}"
        if col in rdf.columns:
            actual = dict(zip(rdf["model"], rdf[col]))

    palette = sns.color_palette("tab20", len(order))
    fig, ax = plt.subplots(figsize=(max(12, len(order) * 0.72), 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    sns.violinplot(data=df, x="model", y=metric, order=order,
                   ax=ax, inner=None, cut=0, hue="model", legend=False,
                   palette=palette, linewidth=0.8, width=0.75)

    for i, mdl in enumerate(order):
        vals   = df[df["model"] == mdl][metric].dropna().values
        ci_low  = float(np.percentile(vals, 2.5))
        ci_high = float(np.percentile(vals, 97.5))
        median  = float(np.median(vals))

        # 95% CI bar (thick, dark grey)
        ax.plot([i, i], [ci_low, ci_high], color="#333333",
                linewidth=5, solid_capstyle="round", zorder=4)
        # median (black horizontal tick)
        ax.plot([i - 0.16, i + 0.16], [median, median], color="black",
                linewidth=2.2, solid_capstyle="round", zorder=5)
        # actual test value (red dot)
        if mdl in actual:
            ax.scatter([i], [actual[mdl]], color="#e03030", s=55, zorder=6,
                       edgecolors="black", linewidths=0.8)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8, color="black")
    ax.tick_params(colors="black", labelsize=8)
    ax.set_xlabel("Model", fontsize=10, color="black")
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10, color="black")
    ax.set_title(
        f"Bootstrap Distribution — {metric.replace('_',' ').title()}\n"
        "red circle = actual test value  |  bar = 95% CI  |  tick = median",
        fontsize=11, color="black", pad=12)
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
    ax.yaxis.grid(True, color="#eeeeee", linewidth=0.6)
    ax.set_axisbelow(True)

    legend_handles = [
        Line2D([0], [0], color="#333333", lw=4, label="95% CI"),
        Line2D([0], [0], color="black", lw=2, label="Median"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#e03030",
               markeredgecolor="black", markersize=8, lw=0,
               label="Actual test value"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
              facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    fig.savefig(out_dir / f"violin_{metric}_bootstrap.png",
                dpi=FIG_DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_bootstrap_heatmap(bootstrap_csv: Path, metric: str, out_dir: Path) -> None:
    """Heatmap with mean + [CI_low, CI_high] in each cell (fig2 style)."""
    df = pd.read_csv(bootstrap_csv)
    if metric not in df.columns:
        return

    stats = (df.groupby("model")[metric]
               .agg(mean="mean",
                    ci_low=lambda x: np.percentile(x, 2.5),
                    ci_high=lambda x: np.percentile(x, 97.5))
               .reset_index()
               .sort_values("mean", ascending=False))

    n = len(stats)
    mat   = stats["mean"].values.reshape(1, n)
    annot = np.array([
        f"{r['mean']:.3f}\n[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        for _, r in stats.iterrows()
    ]).reshape(1, n)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.7), 2.8))
    sns.heatmap(mat, annot=annot, fmt="",
                xticklabels=stats["model"].tolist(),
                yticklabels=[metric.replace("_", " ").title()],
                cmap="YlGnBu", ax=ax,
                linewidths=0.6, linecolor="white",
                vmin=0, vmax=1,
                annot_kws={"size": 7, "va": "center"})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_title(
        f"Bootstrap Summary — {metric.replace('_',' ').title()}  (mean + 95% CI)",
        fontsize=10)
    plt.tight_layout()
    fig.savefig(out_dir / f"bootstrap_heatmap_{metric}.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_bootstrap_ci_bars(bootstrap_csv: Path, metric: str, out_dir: Path,
                           actual_csv: Path | None = None,
                           n_boot: int = 0) -> None:
    """Vertical bar chart with 95% bootstrap CI per model (fig6 style)."""
    df = pd.read_csv(bootstrap_csv)
    if metric not in df.columns:
        return

    stats = (df.groupby("model")[metric]
               .agg(mean="mean",
                    ci_low=lambda x: np.percentile(x, 2.5),
                    ci_high=lambda x: np.percentile(x, 97.5))
               .reset_index()
               .sort_values("mean", ascending=False))

    n = len(stats)
    x = np.arange(n)
    yerr = [stats["mean"] - stats["ci_low"], stats["ci_high"] - stats["mean"]]

    fig, ax = plt.subplots(figsize=(max(10, n * 0.62), 6))
    ax.bar(x, stats["mean"], yerr=yerr, align="center",
           color="#4c72b0", alpha=0.82, ecolor="black",
           capsize=4, error_kw={"linewidth": 1.2}, width=0.6)

    if actual_csv and Path(actual_csv).exists():
        rdf = pd.read_csv(actual_csv)
        col = f"test_{metric}"
        if col in rdf.columns:
            actual = dict(zip(rdf["model"], rdf[col]))
            for xi, mdl in zip(x, stats["model"]):
                if mdl in actual:
                    ax.scatter(xi, actual[mdl], color="white", zorder=5,
                               edgecolors="black", s=45, linewidths=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(stats["model"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10)
    ax.set_ylim(0, 1.05)
    n_boot_str = f"n_boot={n_boot}  |  " if n_boot else ""
    ax.set_title(
        f"Bootstrap 95% CI — {metric.replace('_',' ').title()}\n"
        f"({n_boot_str}error bars centred on bootstrap mean)",
        fontsize=11)
    ax.axhline(stats["mean"].max(), color="crimson", linestyle="--",
               linewidth=0.9, alpha=0.7)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(out_dir / f"bootstrap_ci_{metric}.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_bootstrap_dual_bars(bootstrap_csv: Path, out_dir: Path,
                              actual_csv: Path | None = None,
                              n_boot: int = 0) -> None:
    """Vertical grouped bars: F1 Macro + Accuracy with 95% bootstrap CI (fig6 style)."""
    df = pd.read_csv(bootstrap_csv)
    metrics = [m for m in ("f1_macro", "accuracy") if m in df.columns]
    if not metrics:
        return

    # aggregate per model per metric
    agg: dict[str, pd.DataFrame] = {}
    for m in metrics:
        agg[m] = (df.groupby("model")[m]
                    .agg(mean="mean",
                         ci_low=lambda x: np.percentile(x, 2.5),
                         ci_high=lambda x: np.percentile(x, 97.5))
                    .reset_index())

    primary = "f1_macro" if "f1_macro" in agg else metrics[0]
    models  = agg[primary].sort_values("mean", ascending=False)["model"].tolist()
    n       = len(models)

    # load actual values
    actual: dict[str, dict[str, float]] = {m: {} for m in metrics}
    if actual_csv and Path(actual_csv).exists():
        rdf = pd.read_csv(actual_csv)
        for m in metrics:
            col = f"test_{m}"
            if col in rdf.columns:
                actual[m] = dict(zip(rdf["model"], rdf[col]))

    width  = 0.35
    x      = np.arange(n)
    colors = {"f1_macro": "#4c72b0", "accuracy": "#55a868"}
    labels = {"f1_macro": "F1 Macro",  "accuracy": "Accuracy"}
    offsets = {"f1_macro": -width / 2, "accuracy": width / 2}

    fig, ax = plt.subplots(figsize=(max(12, n * 0.72), 6))
    ax.set_facecolor("white")

    for m in metrics:
        stat_m = agg[m].set_index("model").loc[models]
        means  = stat_m["mean"].values
        yerr   = [means - stat_m["ci_low"].values,
                  stat_m["ci_high"].values - means]
        ax.bar(x + offsets[m], means, width, yerr=yerr,
               label=labels[m], color=colors[m], alpha=0.85,
               ecolor="black", capsize=3,
               error_kw={"linewidth": 1.0})
        for xi, mdl in zip(x, models):
            v = actual[m].get(mdl)
            if v is not None:
                ax.scatter(xi + offsets[m], v, color="white", zorder=5,
                           edgecolors="black", s=35, linewidths=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0, 1.08)
    n_boot_str = f"n_boot={n_boot}  |  " if n_boot else ""
    ax.set_title(
        f"F1 Macro and Accuracy with 95% Bootstrap CI\n"
        f"({n_boot_str}error bars centred on bootstrap mean)",
        fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(out_dir / "bootstrap_f1_acc_bars.png", dpi=FIG_DPI)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# McNemar plots  (fig3 style)
# ──────────────────────────────────────────────────────────────────────────────

def plot_mcnemar_heatmap(mcnemar_csv: Path, out_dir: Path) -> None:
    """Pairwise McNemar −log10(p) heatmap (fig3 style)."""
    df = pd.read_csv(mcnemar_csv)
    models = sorted(set(df["model_a"].tolist() + df["model_b"].tolist()))
    n   = len(models)
    idx = {m: i for i, m in enumerate(models)}
    mat = np.full((n, n), np.nan)

    for _, row in df.iterrows():
        i, j = idx[row["model_a"]], idx[row["model_b"]]
        val  = -np.log10(float(row["p_value"]) + 1e-300)
        mat[i, j] = val
        mat[j, i] = val

    fig, ax = plt.subplots(figsize=(max(10, n * 0.55), max(8, n * 0.55)))
    mask = np.isnan(mat)
    np.fill_diagonal(mask, True)
    disp = np.where(mask, 0, mat)

    annot = np.full((n, n), "", dtype=object)
    for r in range(n):
        for c in range(n):
            if not mask[r, c]:
                annot[r, c] = f"{disp[r, c]:.1f}"

    sns.heatmap(disp, annot=annot, fmt="",
                xticklabels=models, yticklabels=models,
                cmap="Reds", ax=ax, mask=mask,
                linewidths=0.3, linecolor="white",
                cbar_kws={"label": "−log₁₀(p-value)"})

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    ax.set_title("McNemar Pairwise Test — −log₁₀(p-value)\n"
                 "(higher = more significantly different)", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / "mcnemar_heatmap.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Network graph  (fig4 style)
# ──────────────────────────────────────────────────────────────────────────────

def plot_network_graph(mcnemar_csv: Path, out_dir: Path,
                       p_threshold: float = 0.05,
                       ranking_csv: Path | None = None,
                       model_families: dict | None = None) -> None:
    """Statistical significance network graph (fig4 style).

    Nodes: colour = architecture family, size ∝ F1 Score.
    Edges: red = p<0.001  |  orange = p<0.05  |  light-blue = n.s.
    """
    try:
        import networkx as nx
    except ImportError:
        print("[NetworkGraph] networkx not installed — skipping.")
        return

    df = pd.read_csv(mcnemar_csv)
    models = sorted(set(df["model_a"].tolist() + df["model_b"].tolist()))

    # F1 scores for node sizing
    f1_scores: dict[str, float] = {}
    if ranking_csv is not None and Path(ranking_csv).exists():
        rdf = pd.read_csv(ranking_csv)
        if "test_f1_macro" in rdf.columns:
            f1_scores = dict(zip(rdf["model"], rdf["test_f1_macro"]))

    G = nx.Graph()
    G.add_nodes_from(models)
    for _, row in df.iterrows():
        G.add_edge(row["model_a"], row["model_b"],
                   p_value=float(row["p_value"]))

    # Significance is encoded redundantly (hue + dash pattern + width) because
    # hue alone is not distinguishable under red-green colour vision deficiency.
    # Hues are from the Okabe-Ito colour-blind-safe palette.
    edge_colors, edge_styles, edge_widths = [], [], []
    for u, v, data in G.edges(data=True):
        p = data["p_value"]
        if p < 0.001:
            edge_colors.append("#d55e00"); edge_styles.append("solid");  edge_widths.append(2.0)
        elif p < 0.05:
            edge_colors.append("#0072b2"); edge_styles.append("dashed"); edge_widths.append(1.5)
        else:
            edge_colors.append("#999999"); edge_styles.append("dotted"); edge_widths.append(0.8)

    node_colors = [_FAMILY_COLORS.get(_model_family(m, model_families), "#888888")
                   for m in models]
    node_sizes  = [350 + 2800 * f1_scores.get(m, 0.5) for m in models]

    node_labels = {}
    for m in models:
        f1 = f1_scores.get(m)
        node_labels[m] = f"{m}\nF1={f1:.3f}" if f1 is not None else m

    fig, ax = plt.subplots(figsize=(18, 13))
    ax.set_facecolor("white")
    pos = nx.spring_layout(G, seed=42, k=2.8)

    nx.draw_networkx_nodes(G, pos, nodelist=models,
                           node_color=node_colors, node_size=node_sizes,
                           alpha=0.92, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, style=edge_styles,
                           width=edge_widths, alpha=0.75, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=6, ax=ax)

    # p-value edge legend
    edge_legend = [
        Line2D([0], [0], color="#d55e00", lw=2.5, ls="solid",  label="p < 0.001 (highly sig.)"),
        Line2D([0], [0], color="#0072b2", lw=2.0, ls="dashed", label="p < 0.05 (significant)"),
        Line2D([0], [0], color="#999999", lw=1.5, ls="dotted", label="p ≥ 0.05 (n.s.)"),
    ]
    # family node legend
    families_present = sorted({_model_family(m, model_families) for m in models})
    node_legend = [
        mpatches.Patch(color=_FAMILY_COLORS.get(f, "#888"), label=f)
        for f in families_present
    ]
    leg1 = ax.legend(handles=edge_legend, loc="upper left",  fontsize=8, title="Edges")
    ax.add_artist(leg1)
    ax.legend(handles=node_legend, loc="lower left", fontsize=8, title="Node family")

    ax.set_title(
        "Statistical Significance Network Graph (McNemar)\n"
        "Nodes: colour = architecture family  |  Size = F1 Score\n"
        "Edges: red = p<0.001  |  orange = p<0.05  |  light blue = n.s.",
        fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_dir / "network_graph_models.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Top-5 radar  (fig5 style)
# ──────────────────────────────────────────────────────────────────────────────

def plot_top5_radar(ranking_csv: Path, out_dir: Path, top_n: int = 5) -> None:
    """Multi-metric radar: top-N models, F1 value in legend (fig5 style)."""
    df = pd.read_csv(ranking_csv)
    sort_col = "test_balanced_accuracy" if "test_balanced_accuracy" in df.columns else df.columns[1]
    df = df.sort_values(sort_col, ascending=False).head(top_n)

    metric_cols   = ["test_accuracy", "test_f1_macro", "test_precision_macro",
                     "test_recall_macro", "test_specificity_macro",
                     "test_balanced_accuracy", "test_macro_auc"]
    metric_labels = ["Accuracy", "F1 Macro", "Precision", "Recall",
                     "Specificity", "Bal. Acc.", "Macro AUC"]

    avail = [(k, l) for k, l in zip(metric_cols, metric_labels) if k in df.columns]
    if len(avail) < 3:
        return
    keys, labels = zip(*avail)
    keys, labels = list(keys), list(labels)

    angles  = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))

    for (_, row), col in zip(df.iterrows(), colors):
        vals  = [row.get(k, 0) for k in keys]
        vals += vals[:1]
        f1  = row.get("test_f1_macro")
        lbl = f"{row['model']} (F1={f1:.3f})" if f1 is not None else row["model"]
        ax.plot(angles, vals, "-o", linewidth=2, color=col, label=lbl, markersize=4)
        ax.fill(angles, vals, alpha=0.09, color=col)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(color="gray", linewidth=0.5, alpha=0.5)
    ax.set_title(f"Multi-Metric Radar Chart: Top {top_n} Models",
                 fontsize=12, fontweight="bold", pad=28)
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.18), fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "top5_multimetric_radar.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Global bar comparison  (fig6 style — vertical, F1 + Accuracy)
# ──────────────────────────────────────────────────────────────────────────────

def plot_bar_comparison(ranking_csv: Path, out_dir: Path,
                        bootstrap_csv: Path | None = None,
                        n_boot: int = 0,
                        model_families: dict | None = None) -> None:
    """Vertical grouped bars: F1 Macro + Accuracy, sorted by F1.

    If bootstrap_csv is provided, error bars show 95% CI centred on bootstrap mean.
    Family separators shown as dashed vertical lines.
    """
    df = pd.read_csv(ranking_csv)
    sort_col = "test_f1_macro" if "test_f1_macro" in df.columns else "test_balanced_accuracy"
    if sort_col not in df.columns:
        return
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    models = df["model"].tolist()
    n      = len(models)

    metrics     = [c for c in ("test_f1_macro", "test_accuracy") if c in df.columns]
    metric_lbls = {"test_f1_macro": "F1 Macro", "test_accuracy": "Accuracy"}
    bar_colors  = {"test_f1_macro": "#4c72b0", "test_accuracy": "#55a868"}

    # CI from bootstrap
    ci_data: dict[str, dict[str, tuple[float, float]]] = {m: {} for m in metrics}
    if bootstrap_csv and Path(bootstrap_csv).exists():
        bdf = pd.read_csv(bootstrap_csv)
        for tc in metrics:
            raw = tc.replace("test_", "")
            if raw in bdf.columns:
                for mdl in models:
                    vals = bdf[bdf["model"] == mdl][raw].dropna().values
                    if len(vals) > 0:
                        ci_data[tc][mdl] = (float(np.percentile(vals, 2.5)),
                                            float(np.percentile(vals, 97.5)))

    width   = 0.35
    x       = np.arange(n)
    offsets = {metrics[0]: -width / 2} if len(metrics) >= 1 else {}
    if len(metrics) >= 2:
        offsets[metrics[1]] = width / 2

    fig, ax = plt.subplots(figsize=(max(14, n * 0.76), 6.5))

    for met in metrics:
        means = [df[df["model"] == mdl][met].values[0] if mdl in df["model"].values else 0
                 for mdl in models]
        yerr_lo = [means[i] - ci_data[met].get(mdl, (means[i], means[i]))[0]
                   for i, mdl in enumerate(models)]
        yerr_hi = [ci_data[met].get(mdl, (means[i], means[i]))[1] - means[i]
                   for i, mdl in enumerate(models)]
        has_ci  = any(ci_data[met])

        ax.bar(x + offsets.get(met, 0), means,
               width if len(metrics) > 1 else 0.55,
               yerr=[yerr_lo, yerr_hi] if has_ci else None,
               label=metric_lbls.get(met, met),
               color=bar_colors.get(met, "steelblue"),
               alpha=0.82, ecolor="black",
               capsize=3, error_kw={"linewidth": 1.0})

    # family separator lines
    if model_families:
        prev_fam = None
        for i, mdl in enumerate(models):
            fam = _model_family(mdl, model_families)
            if fam != prev_fam and i > 0:
                ax.axvline(i - 0.5, color="dimgray", linestyle="--", linewidth=0.8, alpha=0.7)
            prev_fam = fam

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0, 1.08)
    n_boot_str = f"  (n_boot={n_boot})" if n_boot else ""
    ax.set_title(f"Global Model Comparison{n_boot_str}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(out_dir / "bar_comparison_models.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Parallel coordinates  (fig7 style)
# ──────────────────────────────────────────────────────────────────────────────

def plot_parallel_coords(ranking_csv: Path, out_dir: Path,
                         model_families: dict | None = None,
                         top_n: int = 3) -> None:
    """Parallel coordinates: all metrics per model.

    Gold lines = top-N (by F1).  Opacity ∝ F1.  Colour = architecture family.
    """
    df = pd.read_csv(ranking_csv)

    metric_cols   = ["test_accuracy", "test_f1_macro", "test_precision_macro",
                     "test_recall_macro", "test_specificity_macro"]
    metric_labels = ["Accuracy", "F1 Macro", "Precision", "Recall", "Specificity"]

    if "test_balanced_accuracy" in df.columns:
        metric_cols.append("test_balanced_accuracy")
        metric_labels.append("Bal. Acc.")

    avail = [(c, l) for c, l in zip(metric_cols, metric_labels) if c in df.columns]
    if len(avail) < 2:
        return
    cols, labels = zip(*avail)
    cols, labels = list(cols), list(labels)
    n_axes = len(cols)

    sort_col = "test_f1_macro" if "test_f1_macro" in df.columns else cols[0]
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    f1_vals = df[sort_col].values
    f1_min, f1_max = f1_vals.min(), f1_vals.max()
    f1_range = f1_max - f1_min if f1_max != f1_min else 1.0

    top_indices = set(range(min(top_n, len(df))))

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, row in df.iterrows():
        vals      = [row.get(c, 0) for c in cols]
        is_top    = i in top_indices
        model_nm  = row["model"]
        fam       = _model_family(model_nm, model_families)
        f1        = f1_vals[i]
        alpha     = 0.25 + 0.65 * (f1 - f1_min) / f1_range

        if is_top:
            color, alpha, lw, zorder = "#f0c040", 1.0, 2.8, 10
        else:
            color = _FAMILY_COLORS.get(fam, "#888888")
            lw, zorder = 0.9, 2

        for j in range(n_axes - 1):
            ax.plot([j, j + 1], [vals[j], vals[j + 1]],
                    color=color, alpha=alpha, linewidth=lw, zorder=zorder,
                    solid_capstyle="round")

        if is_top:
            rank = i + 1
            ax.annotate(
                f"#{rank} {model_nm}  F1={f1:.3f}",
                xy=(n_axes - 1, vals[-1]),
                xytext=(n_axes - 1 + 0.08, vals[-1]),
                fontsize=7.5, color="#f0c040", fontweight="bold",
                va="center", clip_on=False)

    # axis lines + labels
    ax.set_xticks(range(n_axes))
    ax.set_xticklabels(labels, fontsize=10)
    for j in range(n_axes):
        ax.axvline(j, color="dimgray", linewidth=0.9, alpha=0.6, zorder=1)

    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.1, n_axes - 0.55)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(
        "Parallel Coordinates: All Metrics per Model\n"
        f"Gold lines = Top {top_n}  |  Opacity = F1 Score  |  Colour = Architecture Family",
        fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    # legend — architecture families
    seen_fams = sorted({_model_family(m, model_families)
                        for m in df["model"]})
    fam_handles = [
        Line2D([0], [0], color=_FAMILY_COLORS.get(f, "#888"), lw=2, label=f)
        for f in seen_fams
    ]
    fam_handles.append(
        Line2D([0], [0], color="#f0c040", lw=2.5,
               label=f"Top {top_n} (Overall F1)"))
    ax.legend(handles=fam_handles, loc="lower right", fontsize=7.5,
              ncol=2, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_dir / "parallel_coords_models.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Overfitting / underfitting visualisations
# ──────────────────────────────────────────────────────────────────────────────

def plot_coverage_curve(pred_dir: Path, out_dir: Path, model_name: str) -> dict:
    """Selective classification: accuracy as a function of coverage.

    Ranks test predictions by their maximum class probability and reports the
    accuracy attainable if the least confident fraction is deferred to a
    specialist. This is what an abstain-and-escalate workflow would deliver,
    measured on predictions that already exist; nothing is retrained.
    """
    y_true = np.load(pred_dir / "y_true.npy")
    y_pred = np.load(pred_dir / "y_pred.npy")
    y_prob = np.load(pred_dir / "y_prob.npy")

    conf = y_prob.max(axis=1)
    correct = (y_true == y_pred)[np.argsort(-conf)]
    n = len(correct)

    cov = np.arange(1, n + 1) / n
    acc = np.cumsum(correct) / np.arange(1, n + 1)
    first_error = int(np.argmax(np.cumsum(~correct) > 0))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(cov * 100, acc * 100, color="#1f77b4", lw=2.2)
    ax.axhline(acc[-1] * 100, color="#888888", ls=":", lw=1.2)
    ax.annotate(f"full coverage: {acc[-1] * 100:.2f}%",
                xy=(100, acc[-1] * 100), xytext=(-6, -14),
                textcoords="offset points", ha="right", fontsize=9, color="#555555")
    ax.axvline(first_error / n * 100, color="#d55e00", ls="--", lw=1.4)
    ax.annotate(f"error-free up to {first_error / n * 100:.1f}% coverage",
                xy=(first_error / n * 100, 99.0), xytext=(8, 0),
                textcoords="offset points", fontsize=9, color="#d55e00")

    ax.set_xlabel("Coverage (% of test specimens classified automatically)")
    ax.set_ylabel("Accuracy on the accepted subset (%)")
    ax.set_xlim(50, 100.5)
    ax.set_ylim(min(98.0, acc[len(acc) // 2] * 100 - 0.5), 100.15)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "coverage_accuracy_curve.png", dpi=FIG_DPI)
    plt.close(fig)

    rows = {}
    for c in (1.00, 0.99, 0.98, 0.97, 0.95, 0.90, 0.85):
        k = int(round(c * n))
        rows[c] = (k, float(correct[:k].mean()), int((~correct[:k]).sum()))
    return {"model": model_name, "n": n, "error_free_coverage": first_error / n,
            "table": rows}


def plot_fit_scatter(overfitting_csv: Path, out_dir: Path) -> None:
    from adjustText import adjust_text

    df = pd.read_csv(overfitting_csv)
    fig, ax = plt.subplots(figsize=(10, 9))

    texts = []
    for status, grp in df.groupby("fit_status"):
        color = _FIT_COLORS.get(status, "gray")
        label = _FIT_LABELS.get(status, status)
        ax.scatter(grp["final_train_acc"], grp["final_val_acc"],
                   c=color, label=label, s=130,
                   edgecolors="white", linewidth=0.8, zorder=3)
        for _, row in grp.iterrows():
            texts.append(ax.text(row["final_train_acc"], row["final_val_acc"],
                                 row["model"], fontsize=7, color="black"))

    lo = min(df["final_train_acc"].min(), df["final_val_acc"].min()) - 0.02
    hi = max(df["final_train_acc"].max(), df["final_val_acc"].max()) + 0.02
    lims = [lo, hi]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Train = Val (ideal)")
    ax.fill_between(lims, lims, [hi + 0.05, hi + 0.05],
                    alpha=0.04, color="blue", label="Potential underfitting zone")
    ax.fill_between(lims, [lo - 0.05, lo - 0.05], lims,
                    alpha=0.04, color="red", label="Overfitting zone")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Training Accuracy (last epoch)", fontsize=10)
    ax.set_ylabel("Validation Accuracy (last epoch)", fontsize=10)
    ax.set_title("Fit Diagnosis — Training vs Validation Accuracy",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                expand=(1.2, 1.4), force_text=(0.3, 0.5))

    plt.tight_layout()
    fig.savefig(out_dir / "fit_scatter.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_fit_heatmap(overfitting_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(overfitting_csv).sort_values("acc_gap", ascending=False)

    cols = {
        "final_train_acc":      "Train Acc",
        "final_val_acc":        "Val Acc",
        "acc_gap":              "Acc Gap\n(train−val)",
        "loss_gap":             "Loss Gap\n(val−train)",
        "val_loss_trend_slope": "Val Loss\nTrend",
        "best_val_acc":         "Best\nVal Acc",
        "best_epoch":           "Best\nEpoch",
    }
    available = {k: v for k, v in cols.items() if k in df.columns}
    mat = df.set_index("model")[list(available.keys())]

    mat_norm = (mat - mat.min()) / (mat.max() - mat.min() + 1e-9)
    for col in ("acc_gap", "loss_gap", "val_loss_trend_slope"):
        if col in mat_norm.columns:
            mat_norm[col] = 1 - mat_norm[col]

    fig, ax = plt.subplots(
        figsize=(max(9, len(available) * 1.4), max(6, len(df) * 0.48)))
    sns.heatmap(mat_norm,
                annot=mat.values, fmt=".3f",
                xticklabels=list(available.values()),
                yticklabels=df["model"].tolist(),
                cmap="RdYlGn", ax=ax,
                linewidths=0.4, linecolor="white",
                cbar_kws={"label": "Normalized score (green = better)"})
    ax.set_title("Fit Heatmap — Diagnostic Indicators per Model",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(out_dir / "fit_heatmap.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_learning_curves_grid(log_dir: Path, overfitting_csv: Path,
                               out_dir: Path) -> None:
    fit_df = pd.read_csv(overfitting_csv).set_index("model")

    history_dirs = sorted([
        d for d in log_dir.iterdir()
        if d.is_dir() and d.name != "global_results"
        and (d / "metrics" / "history.csv").exists()
    ])
    if not history_dirs:
        return

    n     = len(history_dirs)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 4.6, nrows * 3.3))
    axes = np.array(axes).flatten()

    bg_map = {
        "good_fit":         "#f0fff0",
        "mild_overfitting": "#fff5e6",
        "overfitting":      "#fff0f0",
        "underfitting":     "#f0f5ff",
    }

    for ax, model_dir in zip(axes, history_dirs):
        name = model_dir.name
        df   = pd.read_csv(model_dir / "metrics" / "history.csv")

        status     = fit_df.loc[name, "fit_status"] if name in fit_df.index else "unknown"
        edge_color = _FIT_COLORS.get(status, "gray")
        ax.set_facecolor(bg_map.get(status, "white"))

        ax.plot(df["epoch"], df["train_accuracy"],
                color="#1f77b4", linewidth=1.4, label="Train Acc")
        ax.plot(df["epoch"], df["val_accuracy"],
                color="#ff7f0e", linewidth=1.4, label="Val Acc")
        if "val_balanced_accuracy" in df.columns:
            ax.plot(df["epoch"], df["val_balanced_accuracy"],
                    color="#2ca02c", linewidth=1.0, linestyle="--", label="Val BA")

        for spine in ax.spines.values():
            spine.set_edgecolor(edge_color)
            spine.set_linewidth(2)

        status_lbl = _FIT_LABELS.get(status, status)
        ax.set_title(f"{name}\n[{status_lbl}]",
                     fontsize=8, fontweight="bold", color=edge_color)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Epoch", fontsize=7)
        ax.set_ylabel("Accuracy", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(True, alpha=0.25)

    for ax in axes[len(history_dirs):]:
        ax.set_visible(False)

    # No figure title: Elsevier requires the caption's brief title not to be
    # displayed on the artwork itself.
    plt.tight_layout()
    fig.savefig(out_dir / "learning_curves_grid.png",
                dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
