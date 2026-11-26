# RAIN RAIN GO AWAY OR GO HERE
### Will Los Baños get at least 5 mm of rain tomorrow?

**CMSC 170 – Introduction to Artificial Intelligence**  
**Group FEBS**

- Enzo Joyosa
- Franz Jeremy Ellao
- Brylle Pamulaklakin
- Szhan Wayne Timosan

---

## 1. Title and Group Members

# RAIN RAIN GO AWAY OR GO HERE
### Will Los Baños get at least 5 mm of rain tomorrow?

This project is developed by Group FEBS as part of CMSC 170: Introduction to Artificial Intelligence at the University of the Philippines Los Baños.

**Group Members:**
- Enzo Joyosa
- Franz Jeremy Ellao
- Brylle Pamulaklakin
- Szhan Wayne Timosan

---

## 2. Problem Statement and Motivation

Rainfall affects everyday activities, transportation, agriculture, and planning in Los Baños, Laguna. Knowing whether rain is likely to occur the following day can help people make more informed decisions about their daily activities.

This project aims to predict whether Los Baños will receive at least 5 mm of rainfall on the following day using historical daily weather data. The problem is framed as a binary classification task: predicting whether the next day's rainfall will meet or exceed the 5 mm threshold.

We aim to compare different Artificial Intelligence and machine learning methods implemented from scratch and evaluate their performance in predicting next-day rainfall. Through this comparison, we seek to understand the strengths, limitations, and performance differences of the selected methods in handling weather-related data.

---

## 3. Dataset Description

### Source and Size

We use daily weather records from the NASA POWER project (row 4 of the instructor's list of datasets), stored as `data/nasa_power_los_banos_2001_2025.csv`.

- **Source:** NASA Langley Research Center, POWER project, Daily API v2.10.2, community `RE` (accessed November 4, 2026)
- **Location:** Los Baños, Laguna (latitude `14.17`, longitude `121.24`)
- **Period:** January 1, 2001 to December 31, 2025
- **Size:** `9,131` daily records with no missing values or date gaps. Because each day is paired with the next day's rainfall as its label, December 31, 2025 has no next day and is dropped as a sample, leaving **`9,130` labeled samples**.
- **Download URL:**
  `https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,T2M_MAX,T2M_MIN,T2MDEW,RH2M,QV2M,PS,WS2M,WS10M,WD10M,CLOUD_AMT,ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,PRECTOTCORR&community=RE&longitude=121.24&latitude=14.17&start=20010101&end=20251231&format=CSV`

### Features

Each sample has **14 features** (`src/features.py`): 10 direct daily weather measurements and 4 circular encodings derived from wind direction and calendar date.

| Feature | Description |
|---|---|
| `T2M` | Daily mean air temperature at 2 m |
| `T2M_MAX` | Daily maximum air temperature at 2 m |
| `T2M_MIN` | Daily minimum air temperature at 2 m |
| `RH2M` | Relative humidity at 2 m |
| `QV2M` | Specific humidity at 2 m |
| `PS` | Surface atmospheric pressure |
| `WS10M` | Wind speed at 10 m |
| `CLOUD_AMT` | Cloud amount |
| `ALLSKY_SFC_SW_DWN` | All-sky downward solar radiation (sunlight) |
| `PRECTOTCORR` | Today's corrected precipitation (mm) |
| `WD10M_sin`, `WD10M_cos` | Sine and cosine of 10 m wind direction (`WD10M` in radians) |
| `DOY_sin`, `DOY_cos` | Sine and cosine of day of the year (accounting for leap years) |

Three extra columns in the raw download (`T2MDEW`, `WS2M`, and `CLRSKY_SFC_SW_DWN`) are checked by the CSV loader (`src/data_loader.py`) but left out of the 14-feature matrix. Raw `WD10M` is replaced by `WD10M_sin` and `WD10M_cos`.

### Target Variable

The target is a binary next-day rain label built from tomorrow's `PRECTOTCORR`:

- **`1` (rainy):** tomorrow has at least `5.0 mm` of rain (`PRECTOTCORR >= 5.0`)
- **`0` (dry):** tomorrow has less than `5.0 mm` of rain (`PRECTOTCORR < 5.0`)

### Data Preprocessing

All loading, feature creation, splitting, and scaling were written from scratch using Python's standard library and NumPy (no pandas or scikit-learn):

1. **CSV reading and checks (`src/data_loader.py`, `src/features.py`):** Skips the NASA metadata lines before the `YEAR,MO,DY,...` header, parses rows with Python's `csv` module, and verifies there are no missing columns, `-999` missing-value markers, non-finite numbers, or skipped dates.
2. **Circular encoding (`src/features.py`):** Wind direction wraps around at 0°/360° and day of the year wraps around between December 31 and January 1. Turning both into `(sin, cos)` pairs keeps nearby angles and dates close together numerically.
3. **Next-day alignment (`src/features.py`):** Pairs day `t`'s 14 features with day `t + 1`'s rain label (`9,130` samples total).
4. **Train-only scaling (`src/split_scale.py`):** Every scaler is fitted on the training years (`2001–2017`) only and then applied to validation and test so no information leaks from future years:
   - **Z-score scaling** `(x - mean) / std` is used for KNN and the Perceptron, since distances and weighted sums break if large-scale columns dominate small-scale ones.
   - **Min-max scaling** `(x - min) / (max - min)` is used for Naive Bayes. Because Naive Bayes bins features by sorted rank order, min-max scaling keeps the exact same bin assignments as raw values.

---

## 4. Methods

We compare two simple baselines and three different machine learning methods, all implemented from scratch on the same 14 features and split.

### Baselines (`src/baselines.py`)

- **Majority baseline (always dry):** Always predicts the more common class from the training years (`0`, dry). It shows why accuracy alone is not enough: guessing dry every day already gets `58.18%` accuracy on validation (`51.99%` on test) while getting `0.0000` F1 on rainy days.
- **Persistence baseline (same as today):** Predicts that tomorrow's rain label will match today's. In code, `persistence_predict` shifts the labels by one day, taking the last label of the previous split (`y_train[-1]` for validation, `y_val[-1]` for test) as the first day's "today." Because rainy spells often last several days, persistence is a strong benchmark (`0.6923` validation F1, `0.7504` test F1).

### K-Nearest Neighbors (`src/knn.py`)

**How it works:** KNN does not fit model weights during training. Instead, it memorizes the training days. When given a new day, it computes the distance to every training day, picks the `k` closest matches, and predicts whichever class appears more often among those `k` neighbors.

**How we implemented it:** `KNNClassifier(k, p)` saves the z-scored training arrays in `fit()`. During `predict()`, it computes Minkowski distances using NumPy array arithmetic:

`d(x, x_i) = (sum(|x - x_i|^p))^(1 / p)`

Instead of sorting all `6,209` training days, it uses Python's `heapq.nsmallest` to grab only the `k` closest indices and counts their votes (breaking any tie in favor of the single closest neighbor).

### Naive Bayes (`src/naive_bayes.py`)

**How it works:** Naive Bayes uses Bayes' theorem under the assumption that the 14 features are independent given the class (rainy or dry). Since weather measurements are continuous, we cut each feature into equal-count bins so each `(feature, bin)` pair acts like a discrete token whose frequency can be counted per class.

**How we implemented it:** In `fit()`, `NaiveBayesClassifier(bins, smoothing)` sorts each feature in the training years and places `bins - 1` cut points at evenly spaced rank positions (`b * n_samples // bins`), so each bin holds roughly the same number of training days. For each class (`0` and `1`), it computes the log prior `log(n_label / n_samples)` and Laplace-smoothed log likelihoods:

`log((count + smoothing) / (n_label + smoothing * bins))`

In `predict()`, it finds the bin for each feature value using `bisect.bisect_right` and sums the log prior and 14 log likelihoods (adding logs avoids floating-point underflow). It predicts rain (`1`) when the rainy score is strictly higher than the dry score.

### Averaged Perceptron (`src/perceptron.py`)

**How it works:** The Perceptron is a linear classifier that scores each day with `w · x + b`. During training, it passes over the data for several epochs and nudges the weights and bias only when it makes a mistake. Because plain Perceptron weights can jump around near the end of training, the **Averaged Perceptron** keeps a running sum of the weight vector at every step and uses the average for prediction, which stabilizes the decision boundary.

**How we implemented it:** `AveragedPerceptron(n_epochs, seed, lr)` maps labels from `{0, 1}` to `{-1, +1}` and starts with `w = 0` and `b = 0`. At the start of each epoch, it shuffles the training rows with `np.random.default_rng(self.seed)` so results are reproducible. On each sample, if the predicted sign disagrees with the true label, it updates:

`w = w + lr * y * x` and `b = b + lr * y`

After every sample (whether a mistake happened or not), it adds `w` and `b` to `w_sum` and `b_sum`. After all epochs finish, it divides both sums by the total number of steps (`N * n_epochs`). In `predict()`, it computes `X.dot(self.w) + self.b` on z-scored features and outputs `1` whenever the result is `>= 0`.

---

## 5. Experimental Setup

### Data Splitting

We split the dataset **by calendar year, never at random** (`src/split_scale.py`). Because weather carries over from one day to the next, a random split would put neighboring days in both training and evaluation and leak information across the split.

| Split | Years | Samples | Notes |
|---|---|---:|---|
| Train | `2001–2017` | `6,209` | Used to fit models, scalers, and Naive Bayes bin edges |
| Validation | `2018–2021` | `1,461` | `611` rainy days, `850` dry days; used for model comparison and tuning |
| Test | `2022–2025` | `1,460` | `701` rainy days, `759` dry days; locked during tuning and scored once at the end |

### Training, Validation, and Test Lock

All scalers and bin edges are computed from `2001–2017` only. During development and tuning (`src/tune.py`, `src/run.py`), every model was trained on `2001–2017` and scored only on `2018–2021`. Once all settings were frozen, we ran the final evaluation on `2022–2025` once with `python src/run.py --final`, which blocks any command-line override of the tuned hyperparameters and saves the final JSON files under `results/test/`.

### Evaluation Metrics

All metrics are implemented from scratch in `src/metrics.py` with NumPy only, treating rainy days (`1`) as the positive class:

- **F1 score (main metric):** `2 * Precision * Recall / (Precision + Recall)`. We chose F1 as our primary metric because accuracy alone rewards always predicting dry (`0.5818` validation accuracy with `0.0000` F1).
- **Accuracy:** `(TP + TN) / (TP + FP + FN + TN)` — overall share of days predicted correctly.
- **Precision:** `TP / (TP + FP)` — how often a rain forecast actually rained.
- **Recall:** `TP / (TP + FN)` — how many actual rainy days the model caught.
- **Confusion matrix:** raw `TP`, `FP`, `FN`, and `TN` counts for every method.

### Hyperparameter Tuning

We ran a grid search of **48 configurations** on the validation years (`src/tune.py`), saved in `results/tuning/`. Settings were selected by highest validation F1 on the rainy class, with ties broken by recall and then accuracy.

| Method | Grid searched (`results/tuning/`) | Configs | Selected settings | Val Accuracy | Val Precision | Val Recall | Val F1 |
|---|---|---:|---|---:|---:|---:|---:|
| KNN | `k` in `{21, 41, 61, 81}`, `p` in `{1.0, 2.0, 3.0}`, z-score | 12 | `k=81`, `p=2.0`, `zscore` | `0.7337` | `0.6934` | `0.6514` | `0.6717` |
| Naive Bayes | `bins` in `{5, 8, 10, 12}`, `smoothing` in `{0.1, 0.5, 1.0, 2.0}`, min-max | 16 | `bins=10`, `smoothing=0.1`, `minmax` | `0.7050` | `0.6304` | `0.7119` | `0.6687` |
| Perceptron | `epochs` in `{5, 10, 20, 30, 50}`, `lr` in `{0.01, 0.05, 0.1, 0.5}`, `seed=42`, z-score | 20 | `epochs=10`, `lr=0.01`, `seed=42`, `zscore` | `0.7611` | `0.7266` | `0.6874` | `0.7065` |

Two notes from tuning (`results/tuning/all_settings.csv`):
- For the Perceptron, all four learning rates produced identical validation scores for each epoch count because the weights start at zero, so `lr` only rescales `(w, b)` without changing the sign of `w · x + b`.
- For Naive Bayes, smoothing had no effect at 8, 10, or 12 bins because each bin holds hundreds of training days, so small Laplace values barely shift the probabilities.

---

## 6. Results

### Validation Set (2018–2021)

| Method | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Majority (always dry) | 0.5818 | 0.0000 | 0.0000 | 0.0000 |
| Persistence (same as today) | 0.7426 | 0.6923 | 0.6923 | 0.6923 |
| KNN | 0.7337 | 0.6934 | 0.6514 | 0.6717 |
| Naive Bayes | 0.7050 | 0.6304 | 0.7119 | 0.6687 |
| **Perceptron** | **0.7611** | **0.7266** | **0.6874** | **0.7065** |

### Test Set (2022–2025)

| Method | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Majority (always dry) | 0.5199 | 0.0000 | 0.0000 | 0.0000 |
| Persistence (same as today) | 0.7603 | 0.7504 | 0.7504 | 0.7504 |
| KNN | 0.7363 | 0.7423 | 0.6904 | 0.7154 |
| Naive Bayes | 0.7336 | 0.6950 | 0.7932 | 0.7408 |
| **Perceptron** | **0.7678** | **0.7586** | **0.7575** | **0.7580** |

Figures (confusion matrices and F1 bar chart) are in the `figures/` and `figures/test/` folders of the repository.

---

## 7. Discussion

### Performance Analysis

The Perceptron achieved the highest F1 on both the validation set (0.7065) and the test set (0.7580), making it the best-performing method in this project. Its linear decision boundary, combined with averaging over all weight updates, produced a stable and well-calibrated classifier.

The Persistence baseline — predicting tomorrow's weather will be the same as today's — was surprisingly strong (validation F1 0.6923, test F1 0.7504). This reflects the autocorrelated nature of rainfall: rainy days tend to cluster together during the wet season. Any learned method must beat this baseline to be useful, and all three did on the test set.

KNN (validation F1 0.6717, test F1 0.7154) performed moderately well. Its main weakness is that it relies on distance in a 14-dimensional space, where features with different scales can dominate unless carefully normalised. It was also the slowest method (~3.7 s on validation), since it computes distances to every training sample at prediction time.

Naive Bayes (validation F1 0.6687, test F1 0.7408) had the highest recall on the test set (0.7932), meaning it caught the most actual rainy days. However, its precision was lower, indicating more false alarms. The binning of continuous features introduces information loss, and the conditional independence assumption does not hold perfectly for correlated weather variables such as temperature and humidity.

### Limitations

- The dataset covers only one location (Los Baños, Laguna). Results may not generalise to other areas.
- Features are limited to daily aggregates from NASA POWER. Sub-daily patterns (e.g. afternoon convective storms) are not captured.
- The 5 mm threshold is fixed. A different threshold would change the label balance and all reported scores.
- The Perceptron is a linear model and cannot capture non-linear interactions between features without feature engineering.

### Possible Improvements

- Tuning hyperparameters (K for KNN, number of bins for Naive Bayes, epochs for the Perceptron) on the validation set more systematically.
- Adding lag features (yesterday's rainfall, 3-day rolling mean) to give the models short-term memory.
- Trying a non-linear method such as a decision tree or multi-layer perceptron as a future extension.

---

## 8. Conclusion

This project compared three machine learning methods — K-Nearest Neighbors, Naive Bayes, and the averaged Perceptron — on the task of predicting whether Los Baños, Laguna will receive at least 5 mm of rain the following day, using 25 years of NASA POWER daily weather data (2001–2025).

All three methods were implemented from scratch using only Python's standard library, NumPy, and Matplotlib, and evaluated with a strict time-based split to prevent data leakage.

The Perceptron was the strongest method, achieving a test F1 of 0.7580, surpassing both baselines and the other two classifiers. Naive Bayes was the best at catching rainy days (highest recall), while KNN offered a middle ground. All three methods beat the majority baseline by a large margin, and all three beat the persistence baseline on the test set — confirming that learning from historical weather features adds genuine predictive value beyond simply repeating today's observation.

The project demonstrated that simple, interpretable models can perform well on weather prediction when given clean, well-engineered features and a fair evaluation setup.

---

## 9. References and Acknowledgments

### Dataset Source

NASA POWER (Prediction of Worldwide Energy Resources).  
https://power.larc.nasa.gov/

The daily weather dataset used in this project was obtained from NASA POWER:

> The data was obtained from National Aeronautics and Space Administration (NASA) Langley Research Center's Prediction Of Worldwide Energy Resources (POWER) project funded through the NASA Earth Science Division.

### AI Tool Disclosure

AI tools, including Claude (claude.ai) by Anthropic and ChatGPT by OpenAI, were used as assistance during the development of this project for guidance, troubleshooting, documentation, and writing support. All code, implementations, experimental results, and conclusions are subject to review and verification by the project members.

### Other References

*Additional references, documentation, and sources used in the implementation and analysis will be listed here.*

---

## 10. Repository and Presentation

### GitHub Repository

[CMSC 170 Project Repository](https://github.com/szhan17/cmsc170-project)

### Presentation Slides

*Presentation slides link will be added here.*