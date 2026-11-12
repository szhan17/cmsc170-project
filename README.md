# RAIN RAIN GO AWAY OR GO HERE

CMSC 170 AI Project. From today's weather in Los Baños, predict whether tomorrow will have at least 5 mm of rain, and compare three methods written from scratch: K-Nearest Neighbors, Naive Bayes and the Perceptron.

**Group:** FEBS

**Members:** Enzo Joyosa, Franz Jeremy Ellao, Brylle Pamulaklakin, Szhan Wayne Timosan

## Setup (first time only)

Run these from the project root.

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```
## Running

### K-Nearest Neighbors

Run the KNN validation check from the project root:

```bat
python src/knn.py
```

The default settings are `k=61` and `p=2`. You can pass other settings without
keyboard input, for example `python src/knn.py --k 61 --p 2`. The script uses the
existing time-based split and z-scores from the training years (2001-2017), then
prints accuracy, precision, recall, F1 and the confusion matrix for the validation
years (2018-2021). It takes about 5 seconds. It does not evaluate the locked test split.

### Naive Bayes

Run the Naive Bayes validation check from the project root:

```bat
python src/naive_bayes.py
```

The default settings are 10 bins per feature and smoothing 1, for example
`python src/naive_bayes.py --bins 10 --smoothing 1`. Each feature is cut into
equal-count bins using the training years (2001-2017), then the script prints
accuracy, precision, recall, F1 and the confusion matrix for the validation years
(2018-2021). It takes about a second. It does not evaluate the locked test split.

## Dataset

NASA POWER daily point data (row 4 of the instructor's list of datasets), saved as `data/nasa_power_los_banos_2001_2025.csv`.

- **Location:** latitude 14.17, longitude 121.24 (Los Baños, Laguna)
- **Period:** January 1, 2001 to December 31, 2025 (9,131 days, no missing values)
- **Source:** NASA Langley Research Center, POWER project, Daily API v2.10.2, community RE, accessed November 4, 2026
- **Download URL:**
  `https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,T2M_MAX,T2M_MIN,T2MDEW,RH2M,QV2M,PS,WS2M,WS10M,WD10M,CLOUD_AMT,ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,PRECTOTCORR&community=RE&longitude=121.24&latitude=14.17&start=20010101&end=20251231&format=CSV`

Citation requested by NASA:

> The data was obtained from National Aeronautics and Space Administration (NASA) Langley Research Center's Prediction Of Worldwide Energy Resources (POWER) project funded through the NASA Earth Science Division.

## Progress (Milestone 1: proposal and setup)

- [x] [#1](https://github.com/OWNER/REPO/issues/1) Setup: Create the repository and settings
- [x] [#2](https://github.com/OWNER/REPO/issues/2) Setup: README, issue template and pull request template
- [x] [#3](https://github.com/OWNER/REPO/issues/3) Data: Add the dataset and the Milestone 1 record
- [x] [#4](https://github.com/OWNER/REPO/issues/4) Setup: Questions for the instructor

## Progress (Milestone 2: data and implementation)

- [x] [#5](https://github.com/OWNER/REPO/issues/5) Data: CSV loader
- [x] [#6](https://github.com/OWNER/REPO/issues/6) Data: Features and next-day labels
- [x] [#7](https://github.com/OWNER/REPO/issues/7) Data: Time split, scaling and test lock
- [x] [#8](https://github.com/OWNER/REPO/issues/8) Setup: Pull request checks
- [x] [#9](https://github.com/OWNER/REPO/issues/9) Eval: Metrics
- [x] [#10](https://github.com/OWNER/REPO/issues/10) Eval: Majority and persistence baselines
- [x] [#11](https://github.com/OWNER/REPO/issues/11) KNN: Classifier
- [x] [#12](https://github.com/OWNER/REPO/issues/12) Naive Bayes: Classifier with binned features
- [x] [#13](https://github.com/OWNER/REPO/issues/13) Eval: Experiment runner and results files
- [x] [#14](https://github.com/OWNER/REPO/issues/14) Docs: Milestone 2 report

## Branch naming

Name each branch after the issue it fixes: `<issue number>-<Surname>`.

Example: `11-DelaCruz` for issue #11. Branch names can't contain spaces or `[ ]`.

To start working on an issue:

```bat
git switch main
git pull
git switch -c 11-DelaCruz
git push -u origin 11-DelaCruz
```

Then open a pull request into `main` whose description starts with `Closes #11`, tick the issue's box above in the same pull request, and ask another member to review it.
