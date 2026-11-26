# Experiment runner for issue #13. Written by Pamulaklakin, building on
# Franz's first runner (branch franz/milestone2-tasks-10-13-14).
#
# Trains each method on 2001-2017, scores it on 2018-2021 and saves the
# settings, scores, confusion matrix and run time to results/<method>.json.
#
# With --final (issue #32) it scores the test years 2022-2025 instead, once,
# with the tuned settings, and saves to results/test/<method>.json. Only
# --final uses the test years.
#
# Run from the project root:
#   python src/run.py                      (every method)
#   python src/run.py --method knn --k 61 --p 2
#   python src/run.py --final              (final run on the test years)

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
TEST_RESULTS_DIR = os.path.join(RESULTS_DIR, "test")
METHODS = ["majority", "persistence", "knn", "naive_bayes", "perceptron"]
# The settings that may be passed on the command line. --final refuses any
# change to them, so the test years are only ever scored with the tuned ones.
SETTINGS = ["k", "p", "bins", "smoothing", "epochs", "lr", "seed"]


def run_method(method, args, splits, part="val"):
    # Trains on 2001-2017 and returns the predictions on one part of the data
    # ("val" for 2018-2021, "test" for 2022-2025) and the settings used.
    # Each method gets the scaling it needs:
    #   KNN and the Perceptron use z-scores, because they work with distances
    #   and weighted sums. Naive Bayes only looks at the order of the values
    #   to make its bins, so min-max scaling gives the same bins as raw values.
    #   The baselines only use the labels.
    y_train, y_part = splits["minmax"]["y_train"], splits["minmax"]["y_" + part]

    if method == "majority":
        return majority_predict(y_train, len(y_part)), {}

    if method == "persistence":
        # Persistence needs the label of the day before the first day scored:
        # the last training day for validation, the last validation day for test.
        y_before = y_train[-1] if part == "val" else splits["minmax"]["y_val"][-1]
        return persistence_predict(y_before, y_part), {}

    if method == "knn":
        data = splits["zscore"]
        model = KNNClassifier(k=args.k, p=args.p).fit(data["X_train"], data["y_train"])
        return model.predict(data["X_" + part]), {"k": model.k, "p": model.p, "scaling": "zscore"}

    if method == "naive_bayes":
        data = splits["minmax"]
        model = NaiveBayesClassifier(bins=args.bins, smoothing=args.smoothing)
        model.fit(data["X_train"], data["y_train"])
        settings = {"bins": model.bins, "smoothing": model.smoothing, "scaling": "minmax"}
        return model.predict(data["X_" + part]), settings

    if method == "perceptron":
        data = splits["zscore"]
        model = AveragedPerceptron(n_epochs=args.epochs, seed=args.seed, lr=args.lr)
        model.fit(data["X_train"], data["y_train"])
        settings = {"epochs": args.epochs, "learning_rate": args.lr,
                    "seed": args.seed, "scaling": "zscore"}
        return model.predict(data["X_" + part]), settings

    raise ValueError(f"unknown method: {method}")


def save_result(method, settings, y_part, predictions, seconds, n_train, part="val", last_year=None):
    m = all_metrics(y_part, predictions)
    result = {"method": method, "settings": settings, "train_years": f"2001-{TRAIN_END_YEAR}"}
    if part == "val":
        result["validation_years"] = f"{TRAIN_END_YEAR + 1}-{VAL_END_YEAR}"
    else:
        result["test_years"] = f"{VAL_END_YEAR + 1}-{last_year}"
    result["train_samples"] = int(n_train)
    result["validation_samples" if part == "val" else "test_samples"] = int(len(y_part))
    result["scores"] = {name: round(float(m[name]), 4) for name in ("accuracy", "precision", "recall", "f1")}
    result["confusion_matrix"] = {name: int(m[name]) for name in ("TP", "FP", "FN", "TN")}
    result["run_time_seconds"] = round(seconds, 3)

    folder = RESULTS_DIR if part == "val" else TEST_RESULTS_DIR
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{method}.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run the methods on the validation years (or, with --final, the test years) and save the results.")
    parser.add_argument("--method", choices=METHODS + ["all"], default="all")
    parser.add_argument("--k", type=int, default=81, help="KNN: number of neighbors (Issue #26 tuned default)")
    parser.add_argument("--p", type=float, default=2.0, help="KNN: Minkowski exponent")
    parser.add_argument("--bins", type=int, default=10, help="Naive Bayes: bins per feature")
    parser.add_argument("--smoothing", type=float, default=0.1, help="Naive Bayes: Laplace smoothing (Issue #26 tuned default)")
    parser.add_argument("--epochs", type=int, default=10, help="Perceptron: passes over the training years (tuned default)")
    parser.add_argument("--lr", type=float, default=0.01, help="Perceptron: learning rate (tuned default)")
    parser.add_argument("--seed", type=int, default=42, help="Perceptron: shuffle seed")
    parser.add_argument("--final", action="store_true",
                        help="score the test years 2022-2025 with the tuned settings (issue #32)")
    args = parser.parse_args()

    # The final run must use the tuned settings, never ones picked after
    # looking at the test years.
    if args.final:
        changed = [name for name in SETTINGS if getattr(args, name) != parser.get_default(name)]
        if changed:
            parser.error("--final always uses the tuned settings; remove --" + ", --".join(changed))

    part = "test" if args.final else "val"
    methods = list(METHODS) if args.method == "all" else [args.method]
    splits = {
        "minmax": make_splits(save=False, scaling="minmax"),
        "zscore": make_splits(save=False, scaling="zscore"),
    }
    y_part = splits["minmax"]["y_" + part]
    n_train = len(splits["minmax"]["y_train"])
    last_year = splits["minmax"]["dates_test"][-1][:4]

    print("\nScoring the " + ("test years" if args.final else "validation years"))
    print(f"\n{'Method':<13} {'Accuracy':>9} {'Precision':>10} {'Recall':>7} {'F1':>7} {'Time':>8}")
    for method in methods:
        start = time.perf_counter()
        predictions, settings = run_method(method, args, splits, part)
        seconds = time.perf_counter() - start
        s = save_result(method, settings, y_part, predictions, seconds, n_train, part, last_year)["scores"]
        print(f"{method:<13} {s['accuracy']:>9.3f} {s['precision']:>10.3f} "
              f"{s['recall']:>7.3f} {s['f1']:>7.3f} {seconds:>7.1f}s")
    print(f"\nSaved to {os.path.normpath(TEST_RESULTS_DIR if args.final else RESULTS_DIR)}")


if __name__ == "__main__":
    main()
