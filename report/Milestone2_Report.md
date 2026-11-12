# CMSC 170 AI Project — Milestone 2 Progress Report

## Project title
**RAIN RAIN GO AWAY OR GO HERE**

## 1. Project objective

The project aims to predict whether tomorrow in Los Baños, Laguna will receive at least 5 mm of precipitation, using today's weather observations from the NASA POWER dataset. The target is binary:

- `1` — next-day precipitation is at least 5 mm (rainy).
- `0` — next-day precipitation is below 5 mm (dry).

The group's proposed classifiers are K-Nearest Neighbors (KNN), Naive Bayes with binned features, and Perceptron, as stated in the project README. The baseline predictors in this report provide reference points for evaluating the classifiers.

## 2. Dataset and preparation

The dataset is NASA POWER daily point data for Los Baños, Laguna (latitude 14.17, longitude 121.24), covering January 1, 2001 through December 31, 2025. The raw file contains 9,131 daily observations. Because each sample's label is the following day's precipitation category, the final date cannot be used as an input sample; the feature loader therefore produces 9,130 labeled samples, dated January 1, 2001 through December 30, 2025.

The feature loader constructs 14 input features. These include temperature, relative humidity, specific humidity, surface pressure, wind speed, cloud amount, solar radiation, today's precipitation, circular encodings of wind direction, and circular encodings of day of year.

The chronological split is:

| Split | Years | Samples | Dates |
|---|---|---:|---|
| Training | 2001–2017 | 6,209 | 2001-01-01 to 2017-12-31 |
| Validation | 2018–2021 | 1,461 | 2018-01-01 to 2021-12-31 |
| Test | 2022–2025 | 1,460 | 2022-01-01 to 2025-12-30 |

The Min-Max scaler is fitted using training features only. The same training-derived minimums and ranges are applied to validation features. The test period is reserved and is not evaluated by the experiment runner in this milestone.

## 3. Assigned work completed

### Issue #10 — Eval: Majority and persistence baselines

Two baseline predictors were implemented in `src/baselines.py`.

**Majority baseline.** The implementation counts the two classes in the training labels and always predicts the more frequent class for each validation sample. When the class counts are tied, it predicts class `0` (dry). The majority class is determined from the training set, not from the validation labels.

**Persistence baseline.** The implementation predicts tomorrow's rainfall category using today's observed `PRECTOTCORR` value. It predicts rainy (`1`) if today's precipitation is at least 5 mm; otherwise, it predicts dry (`0`). The predictor uses the raw precipitation value in millimeters, not its scaled value, because the threshold is defined in millimeters.

### Issue #13 — Eval: Experiment runner and results files

The experiment runner is implemented in `src/run_experiments.py`. It:

1. Loads the prepared features and next-day labels.
2. Selects the chronological training and validation periods.
3. Fits the Min-Max scaler on the training features only.
4. Applies the scaler to the training and validation features.
5. Runs the Majority and Persistence baselines.
6. Provides a model-adapter interface for the group's classifiers.
7. Computes confusion-matrix counts, accuracy, precision, recall, and F1-score using the group's `src/metrics.py`.
8. Saves the validation metrics and per-sample predictions as CSV files.

Generated files:

- `results/validation_results.csv`
- `results/validation_predictions.csv`

The runner does not evaluate the test period. The KNN, Naive Bayes, and Perceptron implementations have not yet been connected because their source files are being handled by other group members. The adapter instructions are provided in `src/model_registry_template.py`.

### Issue #14 — Docs: Milestone 2 report

This document records the data split, assigned implementations, initial baseline results, current limitation, and next steps. It should be updated after the group's classifiers are integrated and evaluated.

## 4. Evaluation metrics

The project uses the following metrics, computed by the existing from-scratch implementation in `src/metrics.py`:

- **Accuracy:** proportion of all validation predictions that are correct.
- **Precision:** proportion of predicted rainy days that were actually rainy.
- **Recall:** proportion of actual rainy days correctly identified.
- **F1-score:** harmonic mean of precision and recall.

The confusion matrix is also recorded using TP, FP, FN, and TN. F1-score is useful to report alongside accuracy because the task specifically needs to identify rainy days; a model that predicts only the majority dry class can still obtain moderate accuracy while failing to detect rainy days.

## 5. Initial validation results

The following results were produced by running the baseline implementations on the validation period (2018–2021, 1,461 samples).

| Model | TP | FP | FN | TN | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Majority | 0 | 0 | 611 | 850 | 0.5818 | 0.0000 | 0.0000 | 0.0000 |
| Persistence | 423 | 188 | 188 | 662 | 0.7426 | 0.6923 | 0.6923 | 0.6923 |

The training data's majority class is dry (`0`), so the Majority baseline predicts dry for every validation sample. It achieves 58.18% accuracy but does not identify any rainy validation days, resulting in zero precision, recall, and F1-score.

The Persistence baseline achieves 74.26% accuracy and an F1-score of 0.6923. This indicates that today's precipitation category is a useful simple reference for predicting tomorrow's category in this validation period. These results are preliminary baseline results only; they do not establish whether the group's machine learning classifiers perform better.

## 6. Problems and considerations

- The persistence threshold must be applied to raw `PRECTOTCORR` values in millimeters. Applying the 5 mm threshold to a min-max-scaled feature would be incorrect.
- The project uses a time-based split rather than a random split to preserve the chronological nature of weather observations.
- The scaler must be fitted on training data only to avoid data leakage.
- The README describes 9,131 raw daily observations. The feature loader correctly produces 9,130 labeled samples because the last observation has no following day available to form a next-day target.
- The reference split file contains older sample-count comments. The actual output from the supplied feature loader and year boundaries is 6,209 training, 1,461 validation, and 1,460 test samples.
- The main classifiers have not yet been integrated into the experiment runner. Therefore, a complete model comparison is not yet available.

## 7. Plan for the next milestone

1. Obtain the KNN, Naive Bayes, and Perceptron source files from the responsible group members.
2. Write adapters in `src/model_registry.py` following `src/model_registry_template.py`.
3. Run every classifier and both baselines on the same validation period.
4. Compare all models using the same metrics and explain their differences.
5. Keep the test set reserved until model development and selection are complete.
6. Update the report with the full comparison, figures, limitations, and conclusions.

## 8. Member contributions

Update this table with the actual group members and reviewed contributions before submission.

| Member | Contribution |
|---|---|
| [Your name] | Issue #10: Majority and Persistence baselines; Issue #13: experiment runner and validation CSV outputs; Issue #14: Milestone 2 report draft |
| [Member name] | [KNN implementation / issue number] |
| [Member name] | [Naive Bayes implementation / issue number] |
| [Member name] | [Perceptron implementation / other assigned work, if applicable] |

## 9. AI tool disclosure

AI assistance was used to help review the supplied project structure, draft and explain the baseline and experiment-runner code, and prepare this report. The group should review and test the submitted code, disclose AI assistance according to the course policy, and ensure that the responsible members can explain every submitted line.

## 10. Dataset reference

National Aeronautics and Space Administration, Langley Research Center. (n.d.). *Prediction Of Worldwide Energy Resources (POWER) project*. https://power.larc.nasa.gov/

OpenAI. (2026). *ChatGPT* [Large language model]. https://chatgpt.com/

The dataset was obtained from NASA Langley Research Center's POWER project, funded through the NASA Earth Science Division, as stated in the group's README.
