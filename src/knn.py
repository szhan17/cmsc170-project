# K-Nearest Neighbors classifier for issue #11 (builds on Exer06).
# Written by szhan17. Pamulaklakin switched the validation run to z-scores
# fitted on the training years and to the full score report.
#
# Run from the project root: python src/knn.py --k 61 --p 2

from __future__ import annotations

import argparse
import heapq
import os
import sys

import numpy as np

# this lets the script use the project's existing evaluation and split helpers.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import print_report
from split_scale import make_splits


class KNNClassifier:
    def __init__(self, k: int = 61, p: float = 2.0) -> None:
        # this checks that k and p make sense for Minkowski distance.
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer")
        if isinstance(p, bool) or not isinstance(p, (int, float, np.number)):
            raise ValueError("p must be a number greater than or equal to 1")
        if not np.isfinite(p) or p < 1:
            raise ValueError("p must be a number greater than or equal to 1")

        # this stores the model settings without asking for keyboard input.
        self.k = int(k)
        self.p = float(p)
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> KNNClassifier:
        # this converts the training features to a numeric two-dimensional array.
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        # this checks that the training features and labels line up correctly.
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty two-dimensional array")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be one-dimensional and match the rows in X")
        if self.k > len(X):
            raise ValueError("k cannot be larger than the number of training rows")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite numbers")

        # this keeps every training example and its matching label for prediction.
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # this prevents predictions before the model has training examples.
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("fit must be called before predict")

        # this treats a one-dimensional input as one feature vector.
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # this checks that prediction rows use the same features as training.
        if X.ndim != 2 or X.shape[1] != self.X_train.shape[1]:
            raise ValueError("X must have the same number of features used in fit")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite numbers")

        predictions = []
        for sample in X:
            # this computes every Minkowski distance using NumPy array arithmetic.
            distances = np.sum(
                np.abs(self.X_train - sample) ** self.p, axis=1
            ) ** (1.0 / self.p)

            # this keeps only the k closest training rows without sorting all rows.
            nearest_indices = heapq.nsmallest(
                self.k, range(len(distances)), key=distances.__getitem__
            )

            # this picks the most common neighbor label, favoring the nearest label
            # if a vote is tied.
            vote_counts = {}
            winner = self.y_train[nearest_indices[0]]
            for index in nearest_indices:
                label = self.y_train[index]
                vote_counts[label] = vote_counts.get(label, 0) + 1
                if vote_counts[label] > vote_counts[winner]:
                    winner = label
            predictions.append(winner)

        # this returns one predicted label for each input feature vector.
        return np.asarray(predictions, dtype=self.y_train.dtype)


def main() -> None:
    # this reads optional model settings from command-line arguments.
    parser = argparse.ArgumentParser(description="Run KNN on the validation split.")
    parser.add_argument("--k", type=int, default=61, help="number of neighbors")
    parser.add_argument("--p", type=float, default=2.0, help="Minkowski exponent")
    args = parser.parse_args()

    # this loads the existing time split without saving files. Distances need
    # features on the same footing, so they are z-scores from the training years.
    splits = make_splits(save=False, scaling="zscore")
    model = KNNClassifier(k=args.k, p=args.p)
    model.fit(splits["X_train"], splits["y_train"])
    predictions = model.predict(splits["X_val"])

    # this reports the validation scores without using the locked test split.
    print(f"Training samples: {len(splits['y_train'])}")
    print(f"Validation samples: {len(splits['y_val'])}")
    print_report(
        splits["y_val"], predictions,
        label=f"KNN validation (k={model.k}, p={model.p:g})",
    )


if __name__ == "__main__":
    main()
