"""
Issue #13 — Eval: Experiment runner and results files.

Runs the two baselines and any group classifiers registered in
src/model_registry.py. The same chronological training and validation
periods are used for every model.

Outputs:
    results/validation_results.csv
    results/validation_predictions.csv

The validation period is used during development. The test period is
intentionally excluded from this runner and should remain reserved for
the final evaluation after model selection is complete.

Classifier adapter contract:
    predict_function(X_train_scaled, y_train, X_validation_scaled)
        -> 1-D predictions containing only 0 or 1
"""

from pathlib import Path
import csv
import sys
import numpy as np

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features import load_features, RAIN_THRESHOLD_MM
from split_scale import fit_scaler, apply_scaler
from metrics import all_metrics
from baselines import (
    TRAIN_END_YEAR,
    VALIDATION_END_YEAR,
    TODAY_PRECIPITATION_INDEX,
    majority_class,
    predict_majority,
    predict_persistence,
)

RESULTS_DIR = PROJECT_ROOT / "results"


def load_train_validation(data_path=None):
    """
    Load the dataset and return only the training and validation subsets.

    Scaling parameters are fitted on training features only, then applied
    unchanged to validation features. No validation or test values are used
    to fit the scaler.
    """
    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "nasa_power_los_banos_2001_2025.csv"

    X_raw, y, dates, feature_names = load_features(str(data_path))
    years = np.asarray([int(date[:4]) for date in dates])

    train_mask = years <= TRAIN_END_YEAR
    validation_mask = (
        (years > TRAIN_END_YEAR) & (years <= VALIDATION_END_YEAR)
    )

    X_train_raw = X_raw[train_mask]
    y_train = y[train_mask]
    X_validation_raw = X_raw[validation_mask]
    y_validation = y[validation_mask]
    dates_validation = [
        date for date, keep in zip(dates, validation_mask) if keep
    ]

    if len(y_train) == 0 or len(y_validation) == 0:
        raise ValueError("Training and validation splits must both be non-empty.")

    scaler = fit_scaler(X_train_raw)
    X_train_scaled = apply_scaler(X_train_raw, scaler)
    X_validation_scaled = apply_scaler(X_validation_raw, scaler)

    return {
        "X_train_scaled": X_train_scaled,
        "y_train": y_train,
        "X_validation_scaled": X_validation_scaled,
        "X_validation_raw": X_validation_raw,
        "y_validation": y_validation,
        "dates_validation": dates_validation,
        "feature_names": feature_names,
        "scaler": scaler,
    }


def load_registered_models():
    """
    Import the optional model registry.

    Until src/model_registry.py is created from the supplied template,
    the runner evaluates the baselines only.
    """
    try:
        from model_registry import get_models
    except ModuleNotFoundError as error:
        if error.name == "model_registry":
            return {}
        raise

    models = get_models()
    if not isinstance(models, dict):
        raise TypeError("model_registry.get_models() must return a dictionary.")
    return models


def validate_predictions(model_name, predictions, expected_count):
    """Check that a model returns one binary prediction per validation row."""
    predictions = np.asarray(predictions).ravel()

    if len(predictions) != expected_count:
        raise ValueError(
            f"{model_name} returned {len(predictions)} predictions; "
            f"expected {expected_count}."
        )
    if not np.all(np.isin(predictions, [0, 1])):
        raise ValueError(f"{model_name} predictions must contain only 0 and 1.")

    return predictions.astype(np.int32)


def run_experiments(data_path=None, results_dir=None):
    """Run registered models and baselines, then write validation CSV files."""
    data = load_train_validation(data_path)
    output_dir = Path(results_dir) if results_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = data["y_validation"]
    n_validation = len(y_true)

    # Baselines use the training labels and today's RAW precipitation.
    majority_label = majority_class(data["y_train"])
    prediction_sets = {
        "Majority": predict_majority(n_validation, majority_label),
        "Persistence": predict_persistence(
            data["X_validation_raw"], threshold=RAIN_THRESHOLD_MM
        ),
    }

    # Additional classifiers are connected through a small adapter registry.
    for model_name, predict_function in load_registered_models().items():
        if not callable(predict_function):
            raise TypeError(f"Registered model {model_name} is not callable.")

        predictions = predict_function(
            data["X_train_scaled"],
            data["y_train"],
            data["X_validation_scaled"],
        )
        prediction_sets[model_name] = validate_predictions(
            model_name, predictions, n_validation
        )

    result_path = output_dir / "validation_results.csv"
    prediction_path = output_dir / "validation_predictions.csv"

    metric_columns = [
        "model", "split", "samples", "TP", "FP", "FN", "TN",
        "accuracy", "precision", "recall", "f1",
    ]

    with result_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=metric_columns)
        writer.writeheader()

        for model_name, y_pred in prediction_sets.items():
            metrics = all_metrics(y_true, y_pred)
            writer.writerow({
                "model": model_name,
                "split": "validation",
                "samples": len(y_true),
                **metrics,
            })

    with prediction_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["date", "model", "actual", "predicted"])

        for model_name, y_pred in prediction_sets.items():
            for date, actual, predicted in zip(
                data["dates_validation"], y_true, y_pred
            ):
                writer.writerow([
                    date, model_name, int(actual), int(predicted)
                ])

    print("Issue #13 — Validation experiment results")
    print(f"Training samples   : {len(data['y_train'])}")
    print(f"Validation samples : {n_validation}")
    print(f"Majority class     : {majority_label}")
    print("\nModel comparison (validation set):")
    print(
        f"{'Model':<18} {'Accuracy':>10} {'Precision':>10} "
        f"{'Recall':>10} {'F1-score':>10}"
    )

    for model_name, y_pred in prediction_sets.items():
        metric = all_metrics(y_true, y_pred)
        print(
            f"{model_name:<18} {metric['accuracy']:>10.4f} "
            f"{metric['precision']:>10.4f} {metric['recall']:>10.4f} "
            f"{metric['f1']:>10.4f}"
        )

    print(f"\nSaved metrics     : {result_path}")
    print(f"Saved predictions : {prediction_path}")
    if len(prediction_sets) == 2:
        print(
            "\nNOTE: Only the two baselines ran. Add the group's model adapters "
            "to src/model_registry.py to include the classifiers."
        )

    return prediction_sets


if __name__ == "__main__":
    run_experiments()
