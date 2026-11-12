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

Each script gets its own section here, with its command and what it prints, as its issue is merged.

## Dataset

NASA POWER daily point data (row 4 of the instructor's list of datasets), saved as `data/nasa_power_los_banos_2001_2025.csv`.

- **Location:** latitude 14.17, longitude 121.24 (Los Baños, Laguna)
- **Period:** January 1, 2001 to December 31, 2025 (9,131 days, no missing values)
- **Source:** NASA Langley Research Center, POWER project, Daily API v2.10.2, community RE, accessed November 4, 2026
- **Download URL:**
  `https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,T2M_MAX,T2M_MIN,T2MDEW,RH2M,QV2M,PS,WS2M,WS10M,WD10M,CLOUD_AMT,ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,PRECTOTCORR&community=RE&longitude=121.24&latitude=14.17&start=20010101&end=20251231&format=CSV`

Citation requested by NASA:

> The data was obtained from National Aeronautics and Space Administration (NASA) Langley Research Center's Prediction Of Worldwide Energy Resources (POWER) project funded through the NASA Earth Science Division.

## Project structure

The number after each file is the issue that creates it. A file whose issue isn't merged yet doesn't exist yet.

```text
cmsc170-project/
├── README.md                       setup, how to run, dataset, progress checklists
├── requirements.txt                numpy and matplotlib only (#2)
├── .gitignore                      keeps .venv/, __pycache__/ and data/processed/ out of git (#2)
├── .github/
│   ├── ISSUE_TEMPLATE/task.md      new issues start with Problem, Fix, Done when (#2)
│   ├── pull_request_template.md    review checklist for every pull request (#2)
│   └── workflows/                  banned-import check on every pull request (#8)
├── data/
│   ├── nasa_power_los_banos_2001_2025.csv   the dataset, never edited by hand (#3)
│   └── processed/                  arrays made by src/split_scale.py, not committed
├── src/                            all project code, run from the project root
│   ├── data_loader.py              reads the CSV into one record per day (#5)
│   ├── features.py                 14 features per day and tomorrow's label (#6)
│   ├── split_scale.py              split by year, scaling, test lock (#7)
│   ├── metrics.py                  confusion matrix, accuracy, precision, recall, F1 (#9)
│   ├── baselines.py                majority and persistence (#10)
│   ├── knn.py                      K-Nearest Neighbors (#11)
│   ├── naive_bayes.py              Naive Bayes over binned features (#12)
│   ├── perceptron.py               averaged Perceptron (Milestone 3)
│   └── run.py                      runs one method and saves its scores (#13)
├── tests/                          one test file per module in src/, e.g. test_data_loader.py
├── tools/
│   └── check_imports.py            fails if src/ imports a banned library (#8)
├── results/                        one JSON file per run, the source of every reported number (#13)
├── reports/
│   └── milestone2.md               Milestone 2 progress report (#14)
└── docs/                           GitHub Pages project page (later milestone)
```

Rules that keep this tidy:

- Method and data code goes in `src/` only, and nothing in `src/` imports a banned library.
- Generated files go in `results/` (committed) or `data/processed/` (not committed), never next to the code.
- Your virtual environment `.venv/` stays on your computer. Everyone builds their own with the Setup commands above.

## Progress (Milestone 1: proposal and setup)

When an issue's pull request is merged, tick its box and add who did it and who reviewed it, for example `(DelaCruz, reviewed by Reyes)`.

- [x] [#2](https://github.com/szhan17/cmsc170-project/issues/2) Setup: README links, templates and requirements.txt (Pamulaklakin, reviewed by _Surname_)
- [x] [#3](https://github.com/szhan17/cmsc170-project/issues/3) Data: Add the dataset (Joyosa, merged in #17)

## Progress (Milestone 2: data and implementation)

- [ ] [#5](https://github.com/szhan17/cmsc170-project/issues/5) Data: CSV loader
- [ ] [#6](https://github.com/szhan17/cmsc170-project/issues/6) Data: Features and next-day labels
- [ ] [#7](https://github.com/szhan17/cmsc170-project/issues/7) Data: Time split, scaling and test lock
- [ ] [#8](https://github.com/szhan17/cmsc170-project/issues/8) Setup: Pull request checks
- [ ] [#9](https://github.com/szhan17/cmsc170-project/issues/9) Eval: Metrics
- [ ] [#10](https://github.com/szhan17/cmsc170-project/issues/10) Eval: Majority and persistence baselines
- [ ] [#11](https://github.com/szhan17/cmsc170-project/issues/11) KNN: Classifier
- [ ] [#12](https://github.com/szhan17/cmsc170-project/issues/12) Naive Bayes: Classifier with binned features
- [ ] [#13](https://github.com/szhan17/cmsc170-project/issues/13) Eval: Experiment runner and results files
- [ ] [#14](https://github.com/szhan17/cmsc170-project/issues/14) Docs: Milestone 2 report

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
