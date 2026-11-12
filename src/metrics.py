# AI use: This file was created by Jenzzo Joyosa. Claude was used to assist in code optimization and comments.

"""
Issue #9 — Eval: Metrics
=========================
All evaluation metrics for the binary rain-prediction task, implemented
from scratch with NumPy only (no scikit-learn).

Rainy class is label 1 (≥ 5 mm).  Every function accepts plain Python
lists or 1-D NumPy arrays of integers {0, 1}.

Public API
----------
confusion_matrix(y_true, y_pred) → (TP, FP, FN, TN)
accuracy(y_true, y_pred)         → float  in [0, 1]
precision(y_true, y_pred)        → float  in [0, 1]
recall(y_true, y_pred)           → float  in [0, 1]
f1(y_true, y_pred)               → float  in [0, 1]
all_metrics(y_true, y_pred)      → dict
print_report(y_true, y_pred, label) → None  (formatted table to stdout)

Usage
-----
    python src/metrics.py          # runs a self-test with known values
"""

import numpy as np
import sys


# ---------------------------------------------------------------------------
# Core building block
# ---------------------------------------------------------------------------
def confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix for binary classification.

    Convention:  positive class = 1 (rainy day)
                 negative class = 0 (dry day)

    Parameters
    ----------
    y_true, y_pred : array-like of int {0, 1}

    Returns
    -------
    TP, FP, FN, TN : int
        True Positives, False Positives, False Negatives, True Negatives.
    """
    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int32).ravel()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true length {len(y_true)} != y_pred length {len(y_pred)}"
        )

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    return TP, FP, FN, TN


# ---------------------------------------------------------------------------
# Scalar metrics — all derived from the confusion matrix
# ---------------------------------------------------------------------------
def accuracy(y_true, y_pred) -> float:
    """
    Fraction of correct predictions.

        accuracy = (TP + TN) / (TP + FP + FN + TN)
    """
    TP, FP, FN, TN = confusion_matrix(y_true, y_pred)
    total = TP + FP + FN + TN
    return (TP + TN) / total if total > 0 else 0.0


def precision(y_true, y_pred) -> float:
    """
    Of all days predicted rainy, how many actually were?

        precision = TP / (TP + FP)

    Returns 0.0 when the classifier never predicts rain (TP + FP = 0).
    """
    TP, FP, FN, TN = confusion_matrix(y_true, y_pred)
    denom = TP + FP
    return TP / denom if denom > 0 else 0.0


def recall(y_true, y_pred) -> float:
    """
    Of all actual rainy days, how many did we catch?

        recall = TP / (TP + FN)

    Returns 0.0 when there are no positive examples (TP + FN = 0).
    """
    TP, FP, FN, TN = confusion_matrix(y_true, y_pred)
    denom = TP + FN
    return TP / denom if denom > 0 else 0.0


def f1(y_true, y_pred) -> float:
    """
    Harmonic mean of precision and recall (main metric for this project).

        F1 = 2 * precision * recall / (precision + recall)

    Returns 0.0 when both precision and recall are 0.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    denom = p + r
    return 2.0 * p * r / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Convenience bundle
# ---------------------------------------------------------------------------
def all_metrics(y_true, y_pred) -> dict:
    """
    Compute all metrics at once.

    Returns
    -------
    dict with keys: 'TP', 'FP', 'FN', 'TN',
                    'accuracy', 'precision', 'recall', 'f1'
    """
    TP, FP, FN, TN = confusion_matrix(y_true, y_pred)
    p = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    r = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f = 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0
    total = TP + FP + FN + TN
    acc = (TP + TN) / total if total > 0 else 0.0
    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "accuracy":  acc,
        "precision": p,
        "recall":    r,
        "f1":        f,
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------
def print_report(y_true, y_pred, label: str = "Model") -> None:
    """
    Print a compact evaluation table to stdout.

    Parameters
    ----------
    y_true  : array-like of int {0, 1}
    y_pred  : array-like of int {0, 1}
    label   : str   Name shown in the header row.
    """
    m = all_metrics(y_true, y_pred)
    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"{'=' * 50}")
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1        : {m['f1']:.4f}  ← main metric")
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    print(f"              Pred 0   Pred 1")
    print(f"  Actual 0  {m['TN']:8d} {m['FP']:8d}")
    print(f"  Actual 1  {m['FN']:8d} {m['TP']:8d}")
    print(f"{'=' * 50}\n")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Manually constructed example with known answers.
    # 10 samples: 4 rainy (1), 6 dry (0)
    # Predictions: gets 3/4 rainy right, 1 false alarm, 5/6 dry right
    #
    #  true : [1,1,1,1, 0,0,0,0,0,0]
    #  pred : [1,1,1,0, 1,0,0,0,0,0]
    #
    # TP=3  FP=1  FN=1  TN=5
    # accuracy  = (3+5)/10 = 0.80
    # precision = 3/(3+1)  = 0.75
    # recall    = 3/(3+1)  = 0.75
    # f1        = 2*0.75*0.75/(0.75+0.75) = 0.75

    y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]

    m = all_metrics(y_true, y_pred)
    print("Self-test results:")
    print(f"  TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  TN={m['TN']}")
    print(f"  accuracy  = {m['accuracy']:.4f}   expected 0.8000")
    print(f"  precision = {m['precision']:.4f}   expected 0.7500")
    print(f"  recall    = {m['recall']:.4f}   expected 0.7500")
    print(f"  f1        = {m['f1']:.4f}   expected 0.7500")

    # Verify values
    tol = 1e-9
    assert abs(m["accuracy"]  - 0.8)  < tol, "accuracy mismatch"
    assert abs(m["precision"] - 0.75) < tol, "precision mismatch"
    assert abs(m["recall"]    - 0.75) < tol, "recall mismatch"
    assert abs(m["f1"]        - 0.75) < tol, "f1 mismatch"
    print("\nAll assertions passed. ✓")

    print_report(y_true, y_pred, label="Self-test example")

    # Edge cases
    # --- all correct ---
    y_all_correct = [1, 1, 0, 0]
    m2 = all_metrics(y_all_correct, y_all_correct)
    assert m2["f1"] == 1.0, "perfect prediction should give F1=1.0"

    # --- never predict rain (zero division guard) ---
    y_true3 = [1, 1, 0, 0]
    y_pred3 = [0, 0, 0, 0]
    m3 = all_metrics(y_true3, y_pred3)
    assert m3["f1"] == 0.0, "all-negative prediction should give F1=0.0"

    print("Edge-case assertions passed. ✓")