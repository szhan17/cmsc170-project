# Naive Bayes classifier over binned features for issue #12 (builds on Exer04).
# Written by Pamulaklakin.
#
# Exer04 counted words in spam and ham emails. Weather values are continuous,
# so each feature is first cut into bins, and each (feature, bin) pair is then
# treated like a word: count it per class, smooth with Laplace, and multiply.
#
# Run from the project root: python src/naive_bayes.py --bins 10 --smoothing 1

import argparse
import bisect
import math
import os
import sys

import numpy as np

# this lets the script use the project's existing evaluation and split helpers.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import print_report
from split_scale import make_splits


class NaiveBayesClassifier:
    def __init__(self, bins=10, smoothing=1.0):
        # Number of bins per feature, and the Laplace k from Exer04
        self.bins = int(bins)
        self.smoothing = float(smoothing)
        self.cuts = []            # cut points of each feature
        self.log_prior = {}       # log P(class)
        self.log_likelihood = {}  # log P(bin | class), one list per feature

    def bin_of(self, feature, value):
        # Which bin a value falls in: the number of cut points at or below it
        return bisect.bisect_right(self.cuts[feature], value)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        n_samples, n_features = X.shape

        # Equal-count bins from the training years: sort each feature and cut
        # it every n / bins values, so each bin holds about the same number of days
        self.cuts = []
        for feature in range(n_features):
            values = sorted(X[:, feature])
            cuts = []
            for b in range(1, self.bins):
                cuts.append(values[b * n_samples // self.bins])
            self.cuts.append(cuts)

        for label in (0, 1):
            rows = X[y == label]
            n_label = len(rows)

            # Prior, like Exer04's spam_emails / total_emails, but as a log
            self.log_prior[label] = math.log(n_label / n_samples)

            # Bag of (feature, bin) "words" for this class
            self.log_likelihood[label] = []
            for feature in range(n_features):
                counts = [0] * self.bins
                for value in rows[:, feature]:
                    counts[self.bin_of(feature, value)] += 1

                # Laplace smoothing as in Exer04: (count + k) / (total + k * size).
                # Every day adds exactly one word per feature, so the total for a
                # feature is n_label, and its "dictionary" is its bins
                denominator = n_label + self.smoothing * self.bins
                self.log_likelihood[label].append(
                    [math.log((c + self.smoothing) / denominator) for c in counts]
                )
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        predictions = []
        for row in X:
            # Exer04 multiplied Decimal probabilities to avoid underflow.
            # Adding logs gives the same winner with plain floats
            scores = {}
            for label in (0, 1):
                score = self.log_prior[label]
                for feature, value in enumerate(row):
                    score += self.log_likelihood[label][feature][self.bin_of(feature, value)]
                scores[label] = score
            # Rain only if its score is strictly higher (a tie predicts dry)
            predictions.append(1 if scores[1] > scores[0] else 0)
        return np.array(predictions, dtype=np.int32)


def main():
    # Settings come from the command line, not keyboard input
    parser = argparse.ArgumentParser(description="Run Naive Bayes on the validation split.")
    parser.add_argument("--bins", type=int, default=10, help="bins per feature")
    parser.add_argument("--smoothing", type=float, default=1.0, help="Laplace smoothing value")
    args = parser.parse_args()

    # Bins depend only on the order of the values, so the scaling choice
    # does not change the result. The test years are not used
    splits = make_splits(save=False)
    model = NaiveBayesClassifier(bins=args.bins, smoothing=args.smoothing)
    model.fit(splits["X_train"], splits["y_train"])
    predictions = model.predict(splits["X_val"])

    print(f"Training samples: {len(splits['y_train'])}")
    print(f"Validation samples: {len(splits['y_val'])}")
    print_report(
        splits["y_val"], predictions,
        label=f"Naive Bayes validation (bins={model.bins}, smoothing={model.smoothing:g})",
    )


if __name__ == "__main__":
    main()
