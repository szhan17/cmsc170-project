# Milestone 3 — Issue #26: Tune validation settings

## Goal and evaluation protocol

This tuning experiment searches model hyperparameters using the fixed training and validation periods only. Training data: 2001–2017. Validation data: 2018–2021. The locked test period (2022 onward) is never used or scored by this tuning script.

The selection metric is **F1 for the rainy class (label 1)**. Ties are resolved by recall, then accuracy. Scalers are fitted on the training portion only.

## Search space

- **KNN:** `k` in {21, 41, 61, 81}; Minkowski `p` in {1, 2, 3}; z-score scaling. 12 configurations.
- **Naive Bayes:** bins in {5, 8, 10, 12}; Laplace smoothing in {0.1, 0.5, 1.0, 2.0}; min-max scaled features. 16 configurations.
- **Averaged Perceptron:** epochs in {5, 10, 20, 30, 50}; learning rate in {0.01, 0.05, 0.1, 0.5}; seed fixed at 42; z-score scaling. 20 configurations.

Total: **48 configurations**. This is a finite, predefined grid rather than an exhaustive search over every possible hyperparameter.

## Best configurations

See `results/tuning/best_settings.md` and `results/tuning/best_settings.json` for exact settings and metrics. The selected configurations are the highest validation-F1 settings within the tested grids.

## Output files

- `results/tuning/knn_tuning.csv`: all KNN settings tested
- `results/tuning/naive_bayes_tuning.csv`: all Naive Bayes settings tested
- `results/tuning/perceptron_tuning.csv`: all Perceptron settings tested
- `results/tuning/all_settings.csv`: combined 48-row table
- `results/tuning/best_settings.json`: selected settings and score records
- `results/tuning/best_settings.md`: human-readable best-settings table

## Important limitation

These are validation results used to select settings; they are not an unbiased final estimate of generalization. The locked test years should only be evaluated after the group freezes its settings and evaluation procedure. Repeated tuning against the same validation years may overfit that validation period.
