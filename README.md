# CMSC 170 — Milestone 2 assigned tasks (#10, #13, #14)

Project: **RAIN RAIN GO AWAY OR GO HERE**

This folder contains the work for:
- **Issue #10:** Majority and persistence baselines.
- **Issue #13:** Experiment runner and validation results files.
- **Issue #14:** Milestone 2 progress report.

The supplied Project 9 files are retained as the group's shared foundations:
`features.py`, `split_scale.py`, and `metrics.py`.

## Requirements

Use the course-allowed Python standard library and NumPy. This implementation does not use Pandas, scikit-learn, TensorFlow, or other libraries that implement classifiers.

Install NumPy if needed:

```bash
python -m pip install numpy
```

## Run Issue #10

From the project root:

```bash
python src/baselines.py
```

This evaluates Majority and Persistence on the validation period.

## Run Issue #13

```bash
python src/run_experiments.py
```

The runner writes:
- `results/validation_results.csv`
- `results/validation_predictions.csv`

The runner uses the chronological split:
- Training: 2001–2017
- Validation: 2018–2021
- Test: 2022–2025 (not evaluated by this runner)

The scaler is fitted using training features only. The validation data is transformed using those training-only parameters.

## Connect the group's classifiers

When the classifier source files are ready:

1. Copy `src/model_registry_template.py` to `src/model_registry.py`.
2. Import the group's actual classifier classes/functions in `model_registry.py`.
3. Write one adapter per classifier.
4. Each adapter must accept `(X_train_scaled, y_train, X_validation_scaled)` and return one binary prediction (0 or 1) per validation row.
5. Add those adapters to `get_models()`.

See `src/model_registry_template.py` for an example adapter structure. Do not invent or submit placeholder classifier implementations as if they were the group's actual models.

## Important

The current runner produces valid baseline results. It does not yet provide the comparison against KNN, Naive Bayes, or Perceptron because those source files have not been supplied. The report states this limitation transparently.
