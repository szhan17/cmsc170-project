# AI use: This file was created by Jenzzo Joyosa. Cladue was used to assist in code optimizatio and comments.

"""
Issue #6 — Data: Features and next-day labels
==============================================
Reads the NASA POWER CSV, extracts the 14 agreed features for each day,
and builds the next-day binary label (PRECTOTCORR >= 5 mm → 1, else 0).

Output
------
X : np.ndarray, shape (N, 14)   — feature matrix (raw, unscaled)
y : np.ndarray, shape (N,)      — next-day rain labels  {0, 1}
dates : list[str]               — ISO date strings, length N
feature_names : list[str]       — 14 column names in order

Usage
-----
    python src/features.py          # prints shape, label balance, first row
"""

import csv
import math
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = "data/nasa_power_los_banos_2001_2025.csv"
RAIN_THRESHOLD_MM = 5.0          # >= 5 mm → rainy day label

# 14 features (order is fixed across all scripts)
FEATURE_NAMES = [
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "QV2M",
    "PS",
    "WS10M",
    "CLOUD_AMT",
    "ALLSKY_SFC_SW_DWN",
    "PRECTOTCORR",          # today's rain
    "WD10M_sin",            # circular encoding of wind direction
    "WD10M_cos",
    "DOY_sin",              # circular encoding of day-of-year
    "DOY_cos",
]

# NASA POWER CSV column names we need (raw)
_RAW_COLS = [
    "T2M", "T2M_MAX", "T2M_MIN",
    "RH2M", "QV2M", "PS",
    "WS10M",
    "CLOUD_AMT",
    "ALLSKY_SFC_SW_DWN",
    "PRECTOTCORR",
    "WD10M",    # used to derive sin/cos, then dropped
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def _day_of_year(year: int, month: int, day: int) -> int:
    """Return 1-based day-of-year."""
    days_in_month = [31, 28 + _is_leap(year), 31, 30, 31, 30,
                     31, 31, 30, 31, 30, 31]
    return sum(days_in_month[:month - 1]) + day


def _doy_fraction(year: int, month: int, day: int) -> float:
    """DOY as a fraction of the full year (0 < f <= 1)."""
    doy = _day_of_year(year, month, day)
    total = 366 if _is_leap(year) else 365
    return doy / total


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------
def load_features(path: str = DATA_PATH):
    """
    Parse the NASA POWER CSV and return (X, y, dates, feature_names).

    The NASA POWER CSV has a metadata header of variable length followed
    by a row that starts with 'YEAR' (the actual column-name row).
    After that every row is one day.

    Parameters
    ----------
    path : str
        Path to the CSV file relative to the project root.

    Returns
    -------
    X     : np.ndarray shape (N, 14)   raw feature values
    y     : np.ndarray shape (N,)      next-day label (0 or 1)
    dates : list[str]                  'YYYY-MM-DD' for each sample row
    feature_names : list[str]
    """
    rows = []     # list of dicts {col: float, ...}
    dates_raw = []

    with open(path, newline="", encoding="utf-8") as fh:
        # Skip metadata lines until the header row
        header = None
        for raw_line in fh:
            stripped = raw_line.strip()
            if stripped.startswith("YEAR"):
                # This is the column-name row
                header = [c.strip() for c in stripped.split(",")]
                break

        if header is None:
            raise ValueError(
                "Could not find the data header row (expected a line "
                "starting with 'YEAR') in " + path
            )

        reader = csv.DictReader(fh, fieldnames=header)
        for row in reader:
            stripped_row = {k.strip(): v.strip() for k, v in row.items()}
            # Build date string
            yr  = int(stripped_row["YEAR"])
            mo  = int(stripped_row["MO"])
            dy  = int(stripped_row["DY"])
            date_str = f"{yr:04d}-{mo:02d}-{dy:02d}"
            dates_raw.append((date_str, yr, mo, dy))

            rec = {}
            for col in _RAW_COLS:
                rec[col] = float(stripped_row[col])
            rows.append(rec)

    n_total = len(rows)
    # We need N-1 samples (last day has no next-day label)
    N = n_total - 1

    X = np.zeros((N, len(FEATURE_NAMES)), dtype=np.float64)
    y = np.zeros(N, dtype=np.int32)
    dates_out = []

    for i in range(N):
        r = rows[i]
        _, yr, mo, dy = dates_raw[i]

        # Circular wind direction (degrees → radians)
        wd_rad = math.radians(r["WD10M"])

        # Circular day-of-year
        doy_frac = _doy_fraction(yr, mo, dy)
        doy_angle = 2.0 * math.pi * doy_frac

        X[i] = [
            r["T2M"],
            r["T2M_MAX"],
            r["T2M_MIN"],
            r["RH2M"],
            r["QV2M"],
            r["PS"],
            r["WS10M"],
            r["CLOUD_AMT"],
            r["ALLSKY_SFC_SW_DWN"],
            r["PRECTOTCORR"],
            math.sin(wd_rad),          # WD10M_sin
            math.cos(wd_rad),          # WD10M_cos
            math.sin(doy_angle),       # DOY_sin
            math.cos(doy_angle),       # DOY_cos
        ]

        # Next-day label
        next_precip = rows[i + 1]["PRECTOTCORR"]
        y[i] = 1 if next_precip >= RAIN_THRESHOLD_MM else 0

        dates_out.append(dates_raw[i][0])

    return X, y, dates_out, FEATURE_NAMES


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X, y, dates, feat_names = load_features()

    print(f"Loaded {len(dates)} samples with {X.shape[1]} features.")
    print(f"Date range : {dates[0]}  →  {dates[-1]}")
    print()

    # Label balance
    n_rain = int(y.sum())
    n_dry  = len(y) - n_rain
    print(f"Label balance:")
    print(f"  Rainy (≥5 mm next day) : {n_rain}  ({100*n_rain/len(y):.1f}%)")
    print(f"  Dry   (<5 mm next day) : {n_dry}   ({100*n_dry/len(y):.1f}%)")
    print()

    # Feature names and first-row values
    print("Features (first sample, " + dates[0] + "):")
    for name, val in zip(feat_names, X[0]):
        print(f"  {name:25s}: {val:10.4f}")
    print()
    print(f"First label (will it rain on {dates[1]}?): {y[0]}")