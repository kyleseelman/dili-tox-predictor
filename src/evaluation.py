"""
Evaluation metrics for DILI prediction models.

Reports the metrics that matter for drug safety: sensitivity (catching
toxic compounds), specificity (not over-flagging safe ones), and
AUC-ROC/AUC-PR for overall discrimination.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    balanced_accuracy_score,
    classification_report,
    roc_curve,
    precision_recall_curve,
)


def compute_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "AUC-ROC": roc_auc_score(y_true, y_probs),
        "AUC-PR": average_precision_score(y_true, y_probs),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Sensitivity (Recall)": recall_score(y_true, y_pred),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "Precision (PPV)": precision_score(y_true, y_pred, zero_division=0),
        "NPV": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def print_metrics(name: str, metrics: dict):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  AUC-ROC:     {metrics['AUC-ROC']:.4f}")
    print(f"  AUC-PR:      {metrics['AUC-PR']:.4f}")
    print(f"  Sensitivity: {metrics['Sensitivity (Recall)']:.4f}  (catching toxic drugs)")
    print(f"  Specificity: {metrics['Specificity']:.4f}  (not over-flagging safe drugs)")
    print(f"  Precision:   {metrics['Precision (PPV)']:.4f}")
    print(f"  F1:          {metrics['F1']:.4f}")
    print(f"  MCC:         {metrics['MCC']:.4f}")
    print(f"  Confusion:   TP={metrics['TP']}  FP={metrics['FP']}  "
          f"FN={metrics['FN']}  TN={metrics['TN']}")
    print(f"{'='*60}")


def compare_models(results: dict[str, dict], y_test: np.ndarray) -> dict:
    """Compare all model results and return summary table."""
    summary = {}
    for name, res in results.items():
        probs = res["test_probs"]
        if len(probs) != len(y_test):
            y = res.get("test_labels", y_test[:len(probs)])
        else:
            y = y_test
        metrics = compute_metrics(y, probs)
        print_metrics(name, metrics)
        summary[name] = metrics

    print("\n" + "="*80)
    print(f"{'Model':<20} {'AUC-ROC':>8} {'AUC-PR':>8} {'Sens':>8} {'Spec':>8} {'MCC':>8}")
    print("-"*80)
    for name, m in summary.items():
        print(f"{name:<20} {m['AUC-ROC']:>8.4f} {m['AUC-PR']:>8.4f} "
              f"{m['Sensitivity (Recall)']:>8.4f} {m['Specificity']:>8.4f} {m['MCC']:>8.4f}")
    print("="*80)

    return summary
