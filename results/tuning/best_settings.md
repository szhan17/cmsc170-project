# Issue #26 — Best validation settings

Selection criterion: highest validation F1 for rainy class; ties resolved by recall, then accuracy. Test years were not evaluated.

| Method | Best settings | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| knn | `{'k': 81, 'p': 2.0, 'scaling': 'zscore'}` | 0.7337 | 0.6934 | 0.6514 | 0.6717 |
| naive_bayes | `{'bins': 10, 'smoothing': 0.1, 'scaling': 'minmax'}` | 0.7050 | 0.6304 | 0.7119 | 0.6687 |
| perceptron | `{'epochs': 10, 'learning_rate': 0.01, 'seed': 42, 'scaling': 'zscore'}` | 0.7611 | 0.7266 | 0.6874 | 0.7065 |
