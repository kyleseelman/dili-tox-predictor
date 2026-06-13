"""
Publication-quality visualizations for DILI prediction.

Inspired by Axiom Bio's Translate platform:
  - Mechanistic radar charts (toxicity mechanism fingerprints)
  - ROC/PR curves with model comparison
  - Compound-level risk comparison (stop vs advance)
  - Feature importance decomposition by mechanism category
  - Structure-toxicity landscape (descriptor-space embedding)
  - Dose-response-style predicted probability calibration
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE

AXIOM_COLORS = {
    "primary": "#1a1a2e",
    "accent": "#e94560",
    "safe": "#16c79a",
    "danger": "#e94560",
    "warning": "#f5a623",
    "neutral": "#a0a0b0",
    "bg": "#0f0f1a",
    "grid": "#2a2a3e",
    "text": "#e0e0e8",
    "models": ["#e94560", "#f5a623", "#16c79a", "#4a90d9", "#b07cd8"],
}

sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": AXIOM_COLORS["bg"],
    "figure.facecolor": AXIOM_COLORS["primary"],
    "text.color": AXIOM_COLORS["text"],
    "axes.labelcolor": AXIOM_COLORS["text"],
    "xtick.color": AXIOM_COLORS["text"],
    "ytick.color": AXIOM_COLORS["text"],
    "grid.color": AXIOM_COLORS["grid"],
    "axes.edgecolor": AXIOM_COLORS["grid"],
    "font.family": "sans-serif",
})


def plot_roc_pr_curves(
    results: dict,
    y_test: np.ndarray,
    output_dir: Path,
):
    """
    Side-by-side ROC and Precision-Recall curves for all models.
    Key drug safety metrics: sensitivity vs specificity trade-off.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(16, 7), facecolor=AXIOM_COLORS["primary"])

    colors = AXIOM_COLORS["models"]
    for i, (name, res) in enumerate(results.items()):
        probs = res["test_probs"]
        y = res.get("test_labels", y_test[:len(probs)])
        color = colors[i % len(colors)]

        fpr, tpr, _ = roc_curve(y, probs)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=color, lw=2.5, label=f"{name} (AUC={roc_auc:.3f})")

        precision, recall, _ = precision_recall_curve(y, probs)
        pr_auc = auc(recall, precision)
        ax_pr.plot(recall, precision, color=color, lw=2.5, label=f"{name} (AUC={pr_auc:.3f})")

    ax_roc.plot([0, 1], [0, 1], "--", color=AXIOM_COLORS["neutral"], lw=1, alpha=0.5)
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax_roc.set_title("ROC Curve — Sensitivity vs Specificity", fontsize=14, fontweight="bold")
    ax_roc.legend(fontsize=10, loc="lower right", framealpha=0.8, facecolor=AXIOM_COLORS["bg"])

    baseline = y_test.mean()
    ax_pr.axhline(y=baseline, linestyle="--", color=AXIOM_COLORS["neutral"], lw=1, alpha=0.5, label=f"Baseline ({baseline:.2f})")
    ax_pr.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax_pr.set_ylabel("Precision (PPV)", fontsize=12)
    ax_pr.set_title("Precision-Recall Curve — DILI Detection", fontsize=14, fontweight="bold")
    ax_pr.legend(fontsize=10, loc="lower left", framealpha=0.8, facecolor=AXIOM_COLORS["bg"])

    fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "roc_pr_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_mechanistic_radar(
    importance_dict: dict,
    output_dir: Path,
):
    """
    Radar chart showing contribution of each hepatotoxicity mechanism
    to the model's predictions. Mirrors Axiom's multi-readout approach.
    """
    categories = list(importance_dict.keys())
    values = [importance_dict[c]["total_importance"] for c in categories]

    total = sum(values) if sum(values) > 0 else 1
    values_norm = [v / total for v in values]

    short_labels = []
    for c in categories:
        label = c.split("(")[0].strip()
        if len(label) > 18:
            label = label[:16] + "…"
        short_labels.append(label)

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_norm += values_norm[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True), facecolor=AXIOM_COLORS["primary"])
    ax.set_facecolor(AXIOM_COLORS["bg"])

    ax.fill(angles, values_norm, color=AXIOM_COLORS["accent"], alpha=0.25)
    ax.plot(angles, values_norm, color=AXIOM_COLORS["accent"], lw=2.5)
    ax.scatter(angles[:-1], values_norm[:-1], color=AXIOM_COLORS["accent"], s=80, zorder=5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(short_labels, fontsize=11, fontweight="bold")
    ax.set_yticklabels([])

    for angle, val, label in zip(angles[:-1], values_norm[:-1], short_labels):
        ax.annotate(f"{val:.1%}", xy=(angle, val), fontsize=9,
                    ha="center", va="bottom", color=AXIOM_COLORS["text"],
                    xytext=(0, 10), textcoords="offset points")

    ax.set_title("Mechanistic Decomposition of DILI Risk\n",
                 fontsize=15, fontweight="bold", pad=20)
    ax.grid(color=AXIOM_COLORS["grid"], alpha=0.3)

    path = output_dir / "mechanistic_radar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_compound_risk_comparison(
    smiles_list: list[str],
    probs: np.ndarray,
    y_true: np.ndarray,
    desc_matrix: np.ndarray,
    desc_names: list[str],
    output_dir: Path,
    top_n: int = 20,
):
    """
    Axiom-style compound comparison: side-by-side risk profiles
    for highest-risk vs lowest-risk compounds, showing which molecular
    properties differentiate them (stop vs advance decisions).
    """
    sorted_idx = np.argsort(probs)
    safest = sorted_idx[:top_n]
    riskiest = sorted_idx[-top_n:]

    key_descs = ["LogP", "TPSA", "MolWt", "NumAromaticRings", "FractionCSP3",
                 "NumRotatableBonds", "NumHDonors", "NumHAcceptors"]
    desc_idx = [desc_names.index(d) for d in key_descs if d in desc_names]
    used_names = [desc_names[i] for i in desc_idx]

    safe_vals = desc_matrix[safest][:, desc_idx].mean(axis=0)
    risk_vals = desc_matrix[riskiest][:, desc_idx].mean(axis=0)

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=AXIOM_COLORS["primary"])
    x = np.arange(len(used_names))
    width = 0.35

    bars_safe = ax.bar(x - width / 2, safe_vals, width, label=f"Safest {top_n} (Advance)",
                       color=AXIOM_COLORS["safe"], alpha=0.85, edgecolor="white", linewidth=0.5)
    bars_risk = ax.bar(x + width / 2, risk_vals, width, label=f"Riskiest {top_n} (Stop)",
                       color=AXIOM_COLORS["danger"], alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(used_names, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("Mean Property Value", fontsize=12)
    ax.set_title("Compound Risk Comparison: Advance vs Stop\nMolecular Properties Driving DILI Risk",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12, framealpha=0.8, facecolor=AXIOM_COLORS["bg"])

    for bars in [bars_safe, bars_risk]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=8, color=AXIOM_COLORS["text"])

    plt.tight_layout()
    path = output_dir / "compound_risk_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_feature_importance_bars(
    importance_result: dict,
    output_dir: Path,
    top_n: int = 25,
):
    """
    Horizontal bar chart of top feature importances, color-coded by
    mechanistic category (lipophilicity, reactivity, assays, etc.).
    """
    names = importance_result["feature_names"]
    importances = importance_result["mean_abs_shap"]
    top_idx = importance_result["top_feature_idx"][:top_n]

    top_names = [names[i] for i in top_idx]
    top_values = [importances[i] for i in top_idx]

    def get_color(name):
        n = name.lower()
        if n.startswith("fp_"):
            return "#4a90d9"
        elif any(k in n for k in ["logp", "fraction", "lipophil"]):
            return AXIOM_COLORS["danger"]
        elif any(k in n for k in ["molwt", "heavy", "ring", "aromatic"]):
            return AXIOM_COLORS["warning"]
        elif any(k in n for k in ["tpsa", "donor", "acceptor"]):
            return AXIOM_COLORS["safe"]
        elif any(k in n for k in ["charge", "valence"]):
            return "#b07cd8"
        elif any(k in n for k in ["mito", "bsep", "rat", "hepato", "shuai", "other"]):
            return "#ff6b9d"
        else:
            return AXIOM_COLORS["neutral"]

    colors = [get_color(n) for n in top_names]

    fig, ax = plt.subplots(figsize=(12, 10), facecolor=AXIOM_COLORS["primary"])
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_values[::-1], color=colors[::-1], edgecolor="white", linewidth=0.3, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.set_xlabel("Feature Importance (MDI / Gini)", fontsize=12)
    ax.set_title("Top Features Driving DILI Predictions\n", fontsize=14, fontweight="bold")

    legend_items = [
        ("Lipophilicity", AXIOM_COLORS["danger"]),
        ("Structural Complexity", AXIOM_COLORS["warning"]),
        ("Polarity", AXIOM_COLORS["safe"]),
        ("Reactivity", "#b07cd8"),
        ("In Vitro Assays", "#ff6b9d"),
        ("Structural Fingerprints", "#4a90d9"),
    ]
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for _, c in legend_items]
    labels = [l for l, _ in legend_items]
    ax.legend(handles, labels, loc="lower right", fontsize=9, framealpha=0.8, facecolor=AXIOM_COLORS["bg"])

    plt.tight_layout()
    path = output_dir / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_toxicity_landscape(
    desc_matrix: np.ndarray,
    y_true: np.ndarray,
    probs: np.ndarray,
    output_dir: Path,
    perplexity: int = 30,
):
    """
    t-SNE embedding of physicochemical descriptor space, colored by
    DILI risk — shows the chemical landscape of toxic vs safe regions.
    """
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(desc_matrix)
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(X_scaled) - 1),
                random_state=42, max_iter=1000)
    embedding = tsne.fit_transform(X_scaled)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor=AXIOM_COLORS["primary"])

    scatter1 = ax1.scatter(embedding[:, 0], embedding[:, 1], c=y_true,
                           cmap="RdYlGn_r", s=25, alpha=0.7, edgecolors="white", linewidths=0.3)
    ax1.set_title("Chemical Space — True DILI Labels\n(Red = Toxic, Green = Safe)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("t-SNE 1", fontsize=11)
    ax1.set_ylabel("t-SNE 2", fontsize=11)
    plt.colorbar(scatter1, ax=ax1, label="DILI Label", shrink=0.8)

    scatter2 = ax2.scatter(embedding[:, 0], embedding[:, 1], c=probs,
                           cmap="RdYlGn_r", s=25, alpha=0.7, edgecolors="white", linewidths=0.3)
    ax2.set_title("Chemical Space — Predicted Risk Score\n(Red = High Risk, Green = Low Risk)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("t-SNE 1", fontsize=11)
    ax2.set_ylabel("t-SNE 2", fontsize=11)
    plt.colorbar(scatter2, ax=ax2, label="Predicted P(DILI)", shrink=0.8)

    plt.tight_layout()
    path = output_dir / "toxicity_landscape.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_calibration_and_distribution(
    results: dict,
    y_test: np.ndarray,
    output_dir: Path,
):
    """
    Calibration plot + predicted probability distribution.
    Shows how well the model's confidence matches actual DILI rates
    (analogous to Axiom's dose-response calibration).
    """
    fig, (ax_cal, ax_dist) = plt.subplots(1, 2, figsize=(16, 7), facecolor=AXIOM_COLORS["primary"])
    colors = AXIOM_COLORS["models"]

    for i, (name, res) in enumerate(results.items()):
        probs = res["test_probs"]
        y = res.get("test_labels", y_test[:len(probs)])
        color = colors[i % len(colors)]

        try:
            prob_true, prob_pred = calibration_curve(y, probs, n_bins=8, strategy="quantile")
            ax_cal.plot(prob_pred, prob_true, "o-", color=color, lw=2, markersize=6, label=name)
        except Exception:
            pass

        ax_dist.hist(probs[y == 1], bins=20, alpha=0.4, color=color, density=True, label=f"{name} (DILI+)")
        ax_dist.hist(probs[y == 0], bins=20, alpha=0.2, color=color, density=True,
                     histtype="step", lw=2, linestyle="--")

    ax_cal.plot([0, 1], [0, 1], "--", color=AXIOM_COLORS["neutral"], lw=1)
    ax_cal.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax_cal.set_ylabel("Fraction of Positives", fontsize=12)
    ax_cal.set_title("Calibration Plot\n(Perfect = diagonal)", fontsize=14, fontweight="bold")
    ax_cal.legend(fontsize=9, framealpha=0.8, facecolor=AXIOM_COLORS["bg"])

    ax_dist.set_xlabel("Predicted P(DILI)", fontsize=12)
    ax_dist.set_ylabel("Density", fontsize=12)
    ax_dist.set_title("Predicted Probability Distribution\n(Solid = DILI+, Dashed = DILI−)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = output_dir / "calibration_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_model_comparison_summary(
    summary: dict,
    output_dir: Path,
):
    """
    Grouped bar chart comparing all models across key drug safety metrics.
    """
    models = list(summary.keys())
    metrics = ["AUC-ROC", "AUC-PR", "Sensitivity (Recall)", "Specificity", "MCC"]
    metric_labels = ["AUC-ROC", "AUC-PR", "Sensitivity", "Specificity", "MCC"]

    fig, ax = plt.subplots(figsize=(16, 8), facecolor=AXIOM_COLORS["primary"])
    x = np.arange(len(models))
    width = 0.15
    colors = ["#e94560", "#f5a623", "#16c79a", "#4a90d9", "#b07cd8"]

    for j, (metric, label) in enumerate(zip(metrics, metric_labels)):
        vals = [summary[m].get(metric, 0) for m in models]
        offset = (j - len(metrics) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=colors[j],
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=7, color=AXIOM_COLORS["text"])

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison — Drug Safety Metrics", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.8, facecolor=AXIOM_COLORS["bg"])

    plt.tight_layout()
    path = output_dir / "model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def generate_all_visualizations(
    results: dict,
    summary: dict,
    y_test: np.ndarray,
    importance_result: dict,
    mechanistic_result: dict,
    desc_matrix_test: np.ndarray,
    desc_names: list[str],
    smiles_test: list[str],
    best_model_name: str,
    output_dir: Path,
) -> list[Path]:
    """Generate all Axiom-style visualizations."""
    output_dir.mkdir(exist_ok=True)
    print("\nGenerating Axiom-style visualizations...")
    paths = []

    paths.append(plot_roc_pr_curves(results, y_test, output_dir))
    paths.append(plot_model_comparison_summary(summary, output_dir))
    paths.append(plot_feature_importance_bars(importance_result, output_dir))

    if mechanistic_result:
        paths.append(plot_mechanistic_radar(mechanistic_result, output_dir))

    best_probs = results[best_model_name]["test_probs"]
    best_y = results[best_model_name].get("test_labels", y_test[:len(best_probs)])
    paths.append(plot_compound_risk_comparison(
        smiles_test[:len(best_probs)], best_probs, best_y,
        desc_matrix_test[:len(best_probs)], desc_names, output_dir
    ))
    paths.append(plot_toxicity_landscape(
        desc_matrix_test[:len(best_probs)], best_y, best_probs, output_dir
    ))
    paths.append(plot_calibration_and_distribution(results, y_test, output_dir))

    print(f"\n  {len(paths)} visualizations saved to {output_dir}/")
    return paths
