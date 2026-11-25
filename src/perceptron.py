# AI use: This file was created by Jenzzo Joyosa. Claude was used to assist in code optimization and comments.

"""
Issue #24 — Perceptron: Classifier
====================================
Averaged Perceptron for binary rain prediction.

Built on the same pattern as knn.py and naive_bayes.py:
  - fit(X_train, y_train)   trains the model
  - predict(X)              returns labels {0, 1}

Design decisions (from the Milestone 1 proposal)
-------------------------------------------------
* Averaged Perceptron (Freund & Schapire 1999): keeps a running sum of
  all weight vectors seen during training, then divides by the total
  number of updates.  This regularises the model and usually beats the
  last-iterate Perceptron on held-out data.
* Z-score (standardisation) scaling is used as specified in the issue:
  "use the z-scores from the training years".  The scaler fitted here
  is separate from the min-max scaler in split_scale.py.
* Epochs are shuffled with a fixed seed for reproducibility:
  "shuffle each epoch with a fixed seed".
* Bias term is included as an extra weight (no need to prepend a 1-column).

Usage
-----
    python src/perceptron.py           # trains, prints validation scores
    python src/perceptron.py --epochs 15 --seed 42
"""

import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_scale import make_splits
from metrics import print_report, all_metrics

# ---------------------------------------------------------------------------
# Z-score scaler (fit on train only, applied to val/test)
# ---------------------------------------------------------------------------

def fit_zscore(X_train: np.ndarray):
    """
    Compute per-feature mean and std from X_train.

    Returns
    -------
    scaler : np.ndarray shape (2, n_features)
        Row 0 = feature means, Row 1 = feature stds.
        Stds of 0 are replaced by 1 to avoid division by zero.
    """
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    std  = np.where(std == 0, 1.0, std)
    return np.stack([mean, std], axis=0)


def apply_zscore(X: np.ndarray, scaler: np.ndarray) -> np.ndarray:
    """Apply z-score scaling: (X - mean) / std."""
    return (X - scaler[0]) / scaler[1]


# ---------------------------------------------------------------------------
# Averaged Perceptron
# ---------------------------------------------------------------------------

class AveragedPerceptron:
    """
    Binary averaged Perceptron classifier.

    Parameters
    ----------
    n_epochs : int
        Number of passes over the training data.
    lr : float
        Learning rate, the size of each weight update. The weights start at
        zero, so it only rescales them and does not change the predictions.
    seed : int
        Random seed for epoch shuffling (ensures reproducibility).

    Attributes
    ----------
    w : np.ndarray shape (n_features,)
        Final averaged weight vector (set after fit).
    b : float
        Final averaged bias term (set after fit).
    """

    def __init__(self, n_epochs: int = 10, seed: int = 42, lr: float = 0.01):
        self.n_epochs = n_epochs
        self.lr       = lr
        self.seed     = seed
        self.w        = None
        self.b        = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the averaged Perceptron.

        Parameters
        ----------
        X : np.ndarray shape (N, n_features)   scaled feature matrix
        y : np.ndarray shape (N,)              labels {0, 1}
        """
        N, n_features = X.shape

        # Convert labels from {0, 1} to {-1, +1} for the Perceptron update rule
        y_pm = np.where(y == 1, 1, -1).astype(np.float64)

        # Current weights and bias
        w = np.zeros(n_features, dtype=np.float64)
        b = 0.0

        # Accumulated (summed) weights for averaging
        w_sum = np.zeros(n_features, dtype=np.float64)
        b_sum = 0.0
        count = 0   # total number of weight vectors accumulated

        rng = np.random.default_rng(self.seed)

        for epoch in range(self.n_epochs):
            # Shuffle indices with the fixed seed advanced per epoch
            indices = rng.permutation(N)

            for i in indices:
                # Predict: sign of w·x + b
                activation = float(np.dot(w, X[i])) + b
                y_pred_pm  = 1.0 if activation >= 0 else -1.0

                # Update only on misclassification
                if y_pred_pm != y_pm[i]:
                    w = w + self.lr * y_pm[i] * X[i]
                    b = b + self.lr * y_pm[i]

                # Accumulate after every sample (not just on mistakes)
                w_sum = w_sum + w
                b_sum = b_sum + b
                count += 1

        # Averaged weights
        self.w = w_sum / count
        self.b = b_sum / count
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels for X.

        Parameters
        ----------
        X : np.ndarray shape (M, n_features)

        Returns
        -------
        y_pred : np.ndarray shape (M,)   values in {0, 1}
        """
        if self.w is None:
            raise RuntimeError("Call fit() before predict().")

        activations = X.dot(self.w) + self.b
        # Convert {-1, +1} back to {0, 1}
        return (activations >= 0).astype(np.int32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    epochs = 10
    seed   = 42
    args   = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--epochs" and i + 1 < len(args):
            epochs = int(args[i + 1])
        if arg == "--seed" and i + 1 < len(args):
            seed = int(args[i + 1])
    return epochs, seed


if __name__ == "__main__":
    epochs, seed = _parse_args()

    # --- load splits (uses the min-max scaled arrays from split_scale.py) ---
    # We re-fit a z-score scaler on top of those for the Perceptron.
    splits = make_splits(save=False)

    X_train_mm = splits["X_train"]
    X_val_mm   = splits["X_val"]
    y_train    = splits["y_train"]
    y_val      = splits["y_val"]

    # Fit z-score scaler on training set only
    zscaler      = fit_zscore(X_train_mm)
    X_train_z    = apply_zscore(X_train_mm, zscaler)
    X_val_z      = apply_zscore(X_val_mm,   zscaler)

    # --- train ---
    model = AveragedPerceptron(n_epochs=epochs, seed=seed)
    model.fit(X_train_z, y_train)

    # --- evaluate on validation set ---
    y_pred_val = model.predict(X_val_z)
    print_report(y_val, y_pred_val,
                 label=f"Perceptron (epochs={epochs}, seed={seed}) — Validation")

    # --- reproducibility check ---
    model2 = AveragedPerceptron(n_epochs=epochs, seed=seed)
    model2.fit(X_train_z, y_train)
    y_pred2 = model2.predict(X_val_z)
    same = np.array_equal(y_pred_val, y_pred2)
    print(f"Same seed gives same result: {'Yes' if same else 'No'}")