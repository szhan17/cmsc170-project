# Task log

Who did what, taken from the README milestone reports, the merged pull requests and the commit history of `main`. Surnames follow the branch naming rule.

## Milestone 1: proposal and setup

| Task | By |
|---|---|
| Repository created, first commit | Timosan |
| README with project overview and setup instructions | Pamulaklakin |
| README updated with group member details | Timosan |

## Milestone 2: data and implementation

| Issue | Task | By | Pull request |
|---|---|---|---|
| #5 | CSV loader, default dataset path fix | Timosan | #18 |
| #6 | Features and next-day labels | Joyosa | #17 |
| #7 | Time split, scaling and test lock | Joyosa | #17 |
| #9 | Metrics, written from scratch | Joyosa | #17 |
| #10 | Majority and persistence baselines | Ellao (committed by Pamulaklakin) | |
| #11 | KNN classifier, z-scores for KNN | Timosan, Pamulaklakin | #22 |
| #12 | Naive Bayes over equal-count bins | Pamulaklakin | #23 |
| #20 | Remove the committed `.venv` folder | Pamulaklakin | #21 |

## Milestone 3: comparison

| Issue | Task | By | Pull request |
|---|---|---|---|
| #13 | Experiment runner and results files | Pamulaklakin, from Ellao's first version | #27 |
| #24 | Perceptron: first version, then merged into one shared Perceptron | Joyosa, Ellao, Pamulaklakin | #29, #31 |
| #25 | Comparison table, graphs and unit tests | Timosan | #28 |
| #26 | Tuning on the validation years | Ellao | #30 |

## Milestone 4: full results

| Issue | Task | By | Pull request |
|---|---|---|---|
| #32 | Final run on the test years | Pamulaklakin | #37 |
| #33 | Project page (`docs/index.md`) | Ellao | #38 |
| #34 | Project page dataset, methods, experimental setup, figures and fixes | Timosan | |
| #35 | Project page results, discussion, conclusion, task log and README fixes | Joyosa | #39 |

## AI tool use

AI tools (Claude by Anthropic, ChatGPT by OpenAI, and Gemini via Antigravity) were used during the project for debugging, code comments, and documentation review. Joyosa also noted Claude use at the top of `features.py`, `split_scale.py`, `metrics.py`, and `perceptron.py`. Every file and result was run and checked by the group.