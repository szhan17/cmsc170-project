"""
Issue #10 — Eval: Majority and persistence baselines.

The project predicts whether the NEXT day has at least 5 mm of rain.
The positive class is 1 (rainy); the negative class is 0 (dry).

Majority baseline:
    Always predicts the most common class in the training labels.
    A tie is resolved as class 0.

Persistence baseline:
    Uses today's observed precipitation (PRECTOTCORR, in mm) as
    tomorrow's prediction: 1 when today's precipitation >= 5 mm,
    otherwise 0.

The persistence threshold is applied to the RAW feature matrix.
Do not apply 5 mm to the scaled feature values.
"""

from pathlib import Path
import sys
import numpy as np

# Support running this file directly or importing it from the project root.
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features import RAIN_THRESHOLD_MM, load_features
from metrics import all_metrics

# The feature order is defined in src/features.py.
TODAY_PRECIPITATION_INDEX = 9

TRAIN_END_YEAR = 2017
VALIDATION_END_YEAR = 2021


def majority_class(y_train):
    """
    Find the most frequent class in the training labels.

    Parameters
    ----------
    y_train : array-like
        Training labels containing only 0 and 1.

    Returns
    -------
    int
        0 or 1. Ties are resolved as 0.
    """
    labels = np.asarray(y_train, dtype=np.int32).ravel()
    if labels.size == 0:
        raise ValueError("y_train must contain at least one label.")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Training labels must contain only 0 and 1.")

    rainy_count = int(np.sum(labels == 1))
    dry_count = int(np.sum(labels == 0))
    return 1 if rainy_count > dry_count else 0


def predict_majority(n_samples, class_label):
    """
    Produce the same majority-class prediction for every sample.

    Parameters
    ----------
    n_samples : int
        Number of samples to predict.
    class_label : int
        Class returned by majority_class().
    """
    if int(n_samples) < 0:
        raise ValueError("n_samples cannot be negative.")
    if class_label not in (0, 1):
        raise ValueError("class_label must be 0 or 1.")
    return np.full(int(n_samples), int(class_label), dtype=np.int32)


def predict_persistence(X_raw, threshold=RAIN_THRESHOLD_MM):
    """
    Predict tomorrow's class from today's raw precipitation.

    X_raw must be the unscaled matrix returned by load_features().
    PRECTOTCORR is feature index 9 and is measured in millimeters.
    """
    values = np.asarray(X_raw, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] <= TODAY_PRECIPITATION_INDEX:
        raise ValueError(
            "X_raw must be a 2-D matrix containing PRECTOTCORR at index 9."
        )
    if not np.all(np.isfinite(values[:, TODAY_PRECIPITATION_INDEX])):
        raise ValueError("Today's precipitation values must be finite.")

    return (
        values[:, TODAY_PRECIPITATION_INDEX] >= float(threshold)
    ).astype(np.int32)


def get_train_validation_data(data_path=None):
    """
    Load raw features and select the project's chronological train/validation
    periods. The test period is not returned to callers of this function.
    """
    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "nasa_power_los_banos_2001_2025.csv"

    X, y, dates, feature_names = load_features(str(data_path))
    years = np.asarray([int(date[:4]) for date in dates])

    train_mask = years <= TRAIN_END_YEAR
    validation_mask = (
        (years > TRAIN_END_YEAR) & (years <= VALIDATION_END_YEAR)
    )

    return {
        "X_train_raw": X[train_mask],
        "y_train": y[train_mask],
        "X_validation_raw": X[validation_mask],
        "y_validation": y[validation_mask],
        "dates_validation": [
            date for date, keep in zip(dates, validation_mask) if keep
        ],
        "feature_names": feature_names,
    }


def evaluate_baselines(data_path=None):
    """Evaluate both baselines on the validation period."""
    data = get_train_validation_data(data_path)
    y_train = data["y_train"]
    y_true = data["y_validation"]
    X_validation_raw = data["X_validation_raw"]

    majority_label = majority_class(y_train)
    predictions = {
        "Majority": predict_majority(len(y_true), majority_label),
        "Persistence": predict_persistence(X_validation_raw),
    }

    results = {}
    for model_name, y_pred in predictions.items():
        results[model_name] = {
            "predictions": y_pred,
            "metrics": all_metrics(y_true, y_pred),
        }

    return {
        "majority_class": majority_label,
        "y_true": y_true,
        "dates": data["dates_validation"],
        "results": results,
    }


if __name__ == "__main__":
    evaluation = evaluate_baselines()

    print("Issue #10 — Baseline evaluation")
    print(f"Majority class learned from training data: {evaluation['majority_class']}")
    print(f"Validation samples: {len(evaluation['y_true'])}")

    for name, result in evaluation["results"].items():
        metric = result["metrics"]
        print(f"\n{name}")
        print(
            f"  TP={metric['TP']}  FP={metric['FP']}  "
            f"FN={metric['FN']}  TN={metric['TN']}"
        )
        print(f"  Accuracy : {metric['accuracy']:.4f}")
        print(f"  Precision: {metric['precision']:.4f}")
        print(f"  Recall   : {metric['recall']:.4f}")
        print(f"  F1-score : {metric['f1']:.4f}")
