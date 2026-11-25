# Experiment runner for issue #13. Written by Pamulaklakin, building on
# Franz's first runner (branch franz/milestone2-tasks-10-13-14).
#
# Trains each method on 2001-2017, scores it on 2018-2021 and saves the
# settings, scores, confusion matrix and run time to results/<method>.json.
# The test years are never used.
#
# Run from the project root:
#   python src/run.py                      (every method)
#   python src/run.py --method knn --k 61 --p 2

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import majority_predict, persistence_predict
from knn import KNNClassifier
from metrics import all_metrics
from naive_bayes import NaiveBayesClassifier
from perceptron import AveragedPerceptron
from split_scale import TRAIN_END_YEAR, VAL_END_YEAR, make_splits

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
METHODS = ["majority", "persistence", "knn", "naive_bayes", "perceptron"]


def run_method(method, args, splits):
    # Returns the predictions on the validation years and the settings used.
    # Each method gets the scaling it needs:
    #   KNN and the Perceptron use z-scores, because they work with distances
    #   and weighted sums. Naive Bayes only looks at the order of the values
    #   to make its bins, so min-max scaling gives the same bins as raw values.
    #   The baselines only use the labels.
    y_train, y_val = splits["minmax"]["y_train"], splits["minmax"]["y_val"]

    if method == "majority":
        return majority_predict(y_train, len(y_val)), {}

    if method == "persistence":
        return persistence_predict(y_train[-1], y_val), {}

    if method == "knn":
        data = splits["zscore"]
        model = KNNClassifier(k=args.k, p=args.p).fit(data["X_train"], data["y_train"])
        return model.predict(data["X_val"]), {"k": model.k, "p": model.p, "scaling": "zscore"}

    if method == "naive_bayes":
        data = splits["minmax"]
        model = NaiveBayesClassifier(bins=args.bins, smoothing=args.smoothing)
        model.fit(data["X_train"], data["y_train"])
        settings = {"bins": model.bins, "smoothing": model.smoothing, "scaling": "minmax"}
        return model.predict(data["X_val"]), settings

    if method == "perceptron":
        data = splits["zscore"]
        model = AveragedPerceptron(n_epochs=args.epochs, seed=args.seed, lr=args.lr)
        model.fit(data["X_train"], data["y_train"])
        settings = {"epochs": args.epochs, "learning_rate": args.lr,
                    "seed": args.seed, "scaling": "zscore"}
        return model.predict(data["X_val"]), settings

    raise ValueError(f"unknown method: {method}")


def save_result(method, settings, y_val, predictions, seconds, n_train):
    m = all_metrics(y_val, predictions)
    result = {
        "method": method,
        "settings": settings,
        "train_years": f"2001-{TRAIN_END_YEAR}",
        "validation_years": f"{TRAIN_END_YEAR + 1}-{VAL_END_YEAR}",
        "train_samples": int(n_train),
        "validation_samples": int(len(y_val)),
        "scores": {name: round(float(m[name]), 4) for name in ("accuracy", "precision", "recall", "f1")},
        "confusion_matrix": {name: int(m[name]) for name in ("TP", "FP", "FN", "TN")},
        "run_time_seconds": round(seconds, 3),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{method}.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run the methods on the validation years and save the results.")
    parser.add_argument("--method", choices=METHODS + ["all"], default="all")
    parser.add_argument("--k", type=int, default=81, help="KNN: number of neighbors (Issue #26 tuned default)")
    parser.add_argument("--p", type=float, default=2.0, help="KNN: Minkowski exponent")
    parser.add_argument("--bins", type=int, default=10, help="Naive Bayes: bins per feature")
    parser.add_argument("--smoothing", type=float, default=0.1, help="Naive Bayes: Laplace smoothing (Issue #26 tuned default)")
    parser.add_argument("--epochs", type=int, default=10, help="Perceptron: passes over the training years (tuned default)")
    parser.add_argument("--lr", type=float, default=0.01, help="Perceptron: learning rate (tuned default)")
    parser.add_argument("--seed", type=int, default=42, help="Perceptron: shuffle seed")
    args = parser.parse_args()

    methods = list(METHODS) if args.method == "all" else [args.method]
    splits = {
        "minmax": make_splits(save=False, scaling="minmax"),
        "zscore": make_splits(save=False, scaling="zscore"),
    }
    y_val = splits["minmax"]["y_val"]
    n_train = len(splits["minmax"]["y_train"])

    print(f"\n{'Method':<13} {'Accuracy':>9} {'Precision':>10} {'Recall':>7} {'F1':>7} {'Time':>8}")
    for method in methods:
        start = time.perf_counter()
        predictions, settings = run_method(method, args, splits)
        seconds = time.perf_counter() - start
        s = save_result(method, settings, y_val, predictions, seconds, n_train)["scores"]
        print(f"{method:<13} {s['accuracy']:>9.3f} {s['precision']:>10.3f} "
              f"{s['recall']:>7.3f} {s['f1']:>7.3f} {seconds:>7.1f}s")
    print(f"\nSaved to {os.path.normpath(RESULTS_DIR)}")


if __name__ == "__main__":
    main()
