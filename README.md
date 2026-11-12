# RAIN RAIN GO AWAY OR GO HERE

CMSC 170 AI Project. From today's weather in Los Baños, predict whether tomorrow will have at least 5 mm of rain, and compare three methods written from scratch: K-Nearest Neighbors, Naive Bayes and the Perceptron.

**Group:** _group name_
**Members:** _Member A, Member B, Member C, Member D_
**Project page:** _link once GitHub Pages is live_
**Slides:** _link before presentation day_

## Setup (first time only)

Run these from the project root.

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

`requirements.txt` lists only `numpy` and `matplotlib`. The course allows the Python standard library, NumPy (arrays and basic linear algebra only) and Matplotlib. Pandas, scikit-learn and any library that implements the methods are not allowed.

## Running

### K-Nearest Neighbors

Run the KNN validation check from the project root:

```bat
python src/knn.py
```

The default settings are `k=61` and `p=2`. You can pass other settings without
keyboard input, for example `python src/knn.py --k 61 --p 2`. The script uses the
existing time-based split and training-only scaling, then reports validation F1.
It does not evaluate the locked test split.

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

When an issue's pull request is merged, tick its box and add who did it and who reviewed it, for example `(DelaCruz, reviewed by Reyes)`.

- [ ] [#1](https://github.com/OWNER/REPO/issues/1) Setup: Create the repository and settings
- [ ] [#2](https://github.com/OWNER/REPO/issues/2) Setup: README, issue template and pull request template
- [ ] [#3](https://github.com/OWNER/REPO/issues/3) Data: Add the dataset and the Milestone 1 record
- [ ] [#4](https://github.com/OWNER/REPO/issues/4) Setup: Questions for the instructor

## Progress (Milestone 2: data and implementation)

- [ ] [#5](https://github.com/OWNER/REPO/issues/5) Data: CSV loader
- [ ] [#6](https://github.com/OWNER/REPO/issues/6) Data: Features and next-day labels
- [ ] [#7](https://github.com/OWNER/REPO/issues/7) Data: Time split, scaling and test lock
- [ ] [#8](https://github.com/OWNER/REPO/issues/8) Setup: Pull request checks
- [ ] [#9](https://github.com/OWNER/REPO/issues/9) Eval: Metrics
- [ ] [#10](https://github.com/OWNER/REPO/issues/10) Eval: Majority and persistence baselines
- [ ] [#11](https://github.com/OWNER/REPO/issues/11) KNN: Classifier
- [ ] [#12](https://github.com/OWNER/REPO/issues/12) Naive Bayes: Classifier with binned features
- [ ] [#13](https://github.com/OWNER/REPO/issues/13) Eval: Experiment runner and results files
- [ ] [#14](https://github.com/OWNER/REPO/issues/14) Docs: Milestone 2 report

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
