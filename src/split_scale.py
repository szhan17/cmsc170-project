# AI use: This file was created by Jenzzo Joyosa. Claude was used to assist in code optimization and comments.

"""
Issue #7 — Data: Time split, scaling and test lock
====================================================
Splits the feature matrix by calendar year (never at random), fits a
min-max scaler on the training set only, and saves the processed arrays
so that the test set is locked away and not used during development.

Split boundaries (from the Milestone 1 proposal)
-------------------------------------------------
  Train      : 2001-01-01 → 2017-12-31   (6,208 samples)
  Validation : 2018-01-01 → 2021-12-31   (1,461 samples)
  Test       : 2022-01-01 → 2025-12-30   (1,461 samples)
              (last day 2025-12-30 because label = next day's rain,
               so the last sample uses 2025-12-30 → predicts 2025-12-31)

Outputs (saved to data/processed/)
------------------------------------
  X_train.npy   X_val.npy   X_test.npy   — scaled feature matrices
  y_train.npy   y_val.npy   y_test.npy   — integer label vectors
  scaler.npy                              — shape (2, 14): [min_row, range_row]

Usage
-----
    python src/split_scale.py          # prints split sizes
    python src/split_scale.py --check  # also verifies no data leakage
"""

import sys
import os
import numpy as np

# Local import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import load_features

# ---------------------------------------------------------------------------
# Split boundaries (inclusive year ranges)
# ---------------------------------------------------------------------------
TRAIN_END_YEAR = 2017
VAL_END_YEAR   = 2021
# Test: everything after VAL_END_YEAR

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "data", "processed")


# ---------------------------------------------------------------------------
# Min-max scaler (fit on train only)
# ---------------------------------------------------------------------------
def fit_scaler(X_train: np.ndarray):
    """
    Compute per-feature min and range from X_train.

    Returns
    -------
    scaler : np.ndarray shape (2, n_features)
        Row 0 = feature minimums, Row 1 = feature ranges.
        A range of 0 is replaced by 1 to avoid division by zero.
    """
    feat_min   = X_train.min(axis=0)                 # shape (n_features,)
    feat_range = X_train.max(axis=0) - feat_min      # shape (n_features,)
    # Protect against constant features
    feat_range = np.where(feat_range == 0, 1.0, feat_range)
    return np.stack([feat_min, feat_range], axis=0)  # shape (2, n_features)


def apply_scaler(X: np.ndarray, scaler: np.ndarray) -> np.ndarray:
    """
    Apply a pre-fit min-max scaler: X_scaled = (X - min) / range.
    Result is in [0, 1] for training data; test/val may fall outside.
    """
    feat_min, feat_range = scaler[0], scaler[1]
    return (X - feat_min) / feat_range


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def make_splits(save: bool = True):
    """
    Load features, split by time, scale, optionally save.

    Returns
    -------
    splits : dict with keys
        'X_train', 'X_val', 'X_test',
        'y_train', 'y_val', 'y_test',
        'dates_train', 'dates_val', 'dates_test',
        'scaler'
    """
    X, y, dates, feature_names = load_features()

    # --- time split ---
    # dates are 'YYYY-MM-DD'; extract year from first 4 chars
    years = np.array([int(d[:4]) for d in dates])

    train_mask = years <= TRAIN_END_YEAR
    val_mask   = (years > TRAIN_END_YEAR) & (years <= VAL_END_YEAR)
    test_mask  = years > VAL_END_YEAR

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    dates_train = [d for d, m in zip(dates, train_mask) if m]
    dates_val   = [d for d, m in zip(dates, val_mask)   if m]
    dates_test  = [d for d, m in zip(dates, test_mask)  if m]

    # --- fit scaler on train only ---
    scaler = fit_scaler(X_train)

    X_train_sc = apply_scaler(X_train, scaler)
    X_val_sc   = apply_scaler(X_val,   scaler)
    X_test_sc  = apply_scaler(X_test,  scaler)

    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train_sc)
        np.save(os.path.join(OUT_DIR, "X_val.npy"),   X_val_sc)
        np.save(os.path.join(OUT_DIR, "X_test.npy"),  X_test_sc)
        np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
        np.save(os.path.join(OUT_DIR, "y_val.npy"),   y_val)
        np.save(os.path.join(OUT_DIR, "y_test.npy"),  y_test)
        np.save(os.path.join(OUT_DIR, "scaler.npy"),  scaler)

    return {
        "X_train": X_train_sc, "X_val": X_val_sc, "X_test": X_test_sc,
        "y_train": y_train,    "y_val": y_val,     "y_test": y_test,
        "dates_train": dates_train,
        "dates_val":   dates_val,
        "dates_test":  dates_test,
        "scaler":      scaler,
        "feature_names": feature_names,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    check_mode = "--check" in sys.argv

    splits = make_splits(save=True)

    def _stats(name, X, y):
        n1 = int(y.sum())
        pct = 100.0 * n1 / len(y) if len(y) else 0.0
        print(f"  {name:10s}: {len(y):6d} samples  |  "
              f"rainy={n1} ({pct:.1f}%)  "
              f"X ∈ [{X.min():.3f}, {X.max():.3f}]")

    print("Split sizes and label balance:")
    _stats("train",      splits["X_train"], splits["y_train"])
    _stats("validation", splits["X_val"],   splits["y_val"])
    _stats("test",       splits["X_test"],  splits["y_test"])
    print()

    # Date boundaries
    print("Date boundaries:")
    for key in ("train", "val", "test"):
        d = splits[f"dates_{key}"]
        print(f"  {key:10s}: {d[0]}  →  {d[-1]}")
    print()

    # Scaler summary
    sc = splits["scaler"]
    print("Scaler (min, range) for first 5 features:")
    for i, name in enumerate(splits["feature_names"][:5]):
        print(f"  {name:25s}  min={sc[0,i]:9.3f}  range={sc[1,i]:9.3f}")
    print()

    if check_mode:
        # Verify that training years are strictly ≤ TRAIN_END_YEAR
        train_yrs = {int(d[:4]) for d in splits["dates_train"]}
        val_yrs   = {int(d[:4]) for d in splits["dates_val"]}
        test_yrs  = {int(d[:4]) for d in splits["dates_test"]}
        overlap_tv = train_yrs & val_yrs
        overlap_vt = val_yrs & test_yrs
        overlap_tt = train_yrs & test_yrs
        ok = not (overlap_tv or overlap_vt or overlap_tt)
        print(f"Leakage check: {'PASS ✓' if ok else 'FAIL ✗'}")
        if not ok:
            print(f"  train ∩ val  : {overlap_tv}")
            print(f"  val   ∩ test : {overlap_vt}")
            print(f"  train ∩ test : {overlap_tt}")

    print(f"Arrays saved to {os.path.abspath(OUT_DIR)}/")