# RAIN RAIN GO AWAY OR GO HERE

CMSC 170 AI Project. From today's weather in Los Baños, predict whether tomorrow will have at least 5 mm of rain, and compare three methods written from scratch: K-Nearest Neighbors, Naive Bayes and the Perceptron.

**Group:** FEBS

**Members:** Enzo Joyosa, Franz Jeremy Ellao, Brylle Pamulaklakin, Szhan Wayne Timosan

## Setup (first time only)

```bat
python -m venv .venv
.venv\Scripts\pip install numpy
```

Only the Python standard library, NumPy and Matplotlib are allowed. Pandas, scikit-learn and any library that implements the methods are not.

## Running

Run these from the project root. Each method trains on 2001-2017 and prints accuracy, precision, recall, F1 and the confusion matrix for the validation years (2018-2021). None of them touch the locked test years.

| Command | Default settings | Time |
|---|---|---|
| `python src/knn.py` | `--k 61 --p 2` (z-scores from the training years) | about 5 s |
| `python src/naive_bayes.py` | `--bins 10 --smoothing 1` (equal-count bins from the training years) | about 1 s |
| `python -m unittest tests.test_data_loader tests.test_knn` | unit tests | under 1 s |

## Dataset

NASA POWER daily point data (row 4 of the instructor's list of datasets), saved as `data/nasa_power_los_banos_2001_2025.csv`.

- **Location:** latitude 14.17, longitude 121.24 (Los Baños, Laguna)
- **Period:** January 1, 2001 to December 31, 2025 (9,131 days, no missing values)
- **Source:** NASA Langley Research Center, POWER project, Daily API v2.10.2, community RE, accessed November 4, 2026
- **Download URL:**
  `https://power.larc.nasa.gov/api/temporal/daily/point?parameters=...&format=CSV`   ← unchanged

Citation requested by NASA:

> The data was obtained from ... NASA Earth Science Division.   ← unchanged

## Milestone 2 report: data preparation and implementation

### What was done

| Part | Issue | By | Status |
|---|---|---|---|
| CSV loader | [#5]| Timosan | Done |
| Features and next-day labels | [#6] | Joyosa | Done |
| Time split, scaling and test lock | [#7] | Joyosa | Done |
| Metrics, written from scratch | [#9]| Joyosa | Done |
| KNN | [#11]| Timosan, Pamulaklakin | Done |
| Naive Bayes | [#12]| Pamulaklakin | Done |
| Majority and persistence baselines | [#10] | Ellao | Done |



### Split and metrics

- **Split by year, not at random:** train 2001-2017 (6,209 days), validation 2018-2021 (1,461), test 2022-2025 (1,460). Rain today is correlated with rain tomorrow, so a random split would leak information between neighbouring days.
- **Scaling and bin edges come from the training years only.**
- **The test years stay locked** until every setting is frozen.
- **Main metric: F1 on the rainy class.** Always guessing "dry" already gets 58% accuracy but F1 0. We also report accuracy, precision, recall and the confusion matrix.

### Initial results (validation years 2018-2021, 611 rainy days out of 1,461)

| Method | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Always dry (majority) | 0.582 | 0.000 | 0.000 | 0.000 |
| Same as today (persistence) | 0.743 | 0.692 | 0.692 | 0.692 |
| KNN (k=61, p=2) | 0.731 | 0.686 | 0.658 | 0.672 |
| Naive Bayes (10 bins, smoothing 1) | 0.705 | 0.630 | 0.712 | 0.669 |

The two baselines were computed on the same split. Their script is #10.

### Problems met

- KNN with min-max scaling scored F1 0.626. Switching to z-scores from the training years raised it to 0.672.
- Neither method beats persistence on F1 yet, because rain carries over strongly from day to day.
- The metrics report crashed on Windows because of a non-ASCII arrow. It now prints a plain `<-`.

### Plan for Milestone 3

- Implement the Perceptron (shuffled each epoch, averaged weights)
- Finish the baselines, the experiment runner and the import check (#10, #13, #8)
- Tune k, p and the number of bins on the validation years
- Compare all methods with tables, graphs and McNemar's test

## Branch naming

Name each branch after the issue it fixes: `<issue number>-<Surname>`.

Example: `11-DelaCruz` for issue #11. Branch names can't contain spaces or `[ ]`.

```bat
git switch main
git pull
git switch -c 11-DelaCruz
git push -u origin 11-DelaCruz
```

Then open a pull request into `main` whose description starts with `Closes #11`, and ask another member to review it.
