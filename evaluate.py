"""
Stage 7: Model Evaluation
---------------------------
Computes accuracy, precision, recall, F1-score, and ROC-AUC for a fitted
model, and reports cross-validated F1 (mean +/- std) to demonstrate the
robustness argument made in Section 7 of the plan.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import cross_val_score


def evaluate_model(name, model, X_test, y_test) -> dict:
    """Compute the full evaluation metric set for a single fitted model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return metrics


def print_metrics(metrics: dict):
    print(f"\n--- {metrics['model']} ---")
    print(f"  Accuracy : {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall   : {metrics['recall']:.3f}")
    print(f"  F1-score : {metrics['f1_score']:.3f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.3f}")


def cross_validated_f1(estimator, X, y, cv=5):
    """Cross-validated F1 mean/std, as described in Section 7 of the plan."""
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="f1", n_jobs=-1)
    return float(np.mean(scores)), float(np.std(scores))
