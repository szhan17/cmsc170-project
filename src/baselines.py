# Majority and persistence baselines for issue #10.
#
# These are reference points, not methods. A method that cannot beat them
# has not learned anything useful from the weather.
#
# Run from the project root: python src/baselines.py

import os
import sys

import numpy as np

# this lets the script use the project's existing evaluation and split helpers.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import print_report
from split_scale import make_splits


def majority_predict(y_train, n):
    # Always predict the more common training class (a tie predicts dry)
    rainy = int(np.sum(y_train))
    label = 1 if rainy > len(y_train) - rainy else 0
    return np.full(n, label, dtype=np.int32)


def persistence_predict(y_before, y):
    # Predict that tomorrow is like today. Each label says whether the next
    # day had at least 5 mm, so today's rain is the label of the day before.
    # The first day's "day before" is the last day of the previous split.
    return np.concatenate([[y_before], y[:-1]]).astype(np.int32)


def main():
    # The test years are not used
    splits = make_splits(save=False)
    y_train, y_val = splits["y_train"], splits["y_val"]

    print(f"Training samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print_report(
        y_val, majority_predict(y_train, len(y_val)),
        label="Majority baseline validation (always dry)",
    )
    print_report(
        y_val, persistence_predict(y_train[-1], y_val),
        label="Persistence baseline validation (same as today)",
    )


if __name__ == "__main__":
    main()
