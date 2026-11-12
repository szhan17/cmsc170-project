import unittest

import numpy as np

from src.knn import KNNClassifier


class KNNClassifierTests(unittest.TestCase):
    def test_predict_uses_each_feature_and_majority_vote(self) -> None:
        # this creates examples with three features to avoid a fixed feature count.
        X_train = np.array(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [4.0, 0.0, 0.0],
            ]
        )
        y_train = np.array([0, 1, 1, 0])
        model = KNNClassifier(k=3, p=2).fit(X_train, y_train)

        # this checks that the two nearest labels win the three-neighbor vote.
        prediction = model.predict(np.array([[0.0, 0.0, 0.0]]))

        np.testing.assert_array_equal(prediction, [1])

    def test_predict_accepts_one_feature_vector(self) -> None:
        # this checks that a single input row returns one predicted label.
        model = KNNClassifier(k=1).fit(
            np.array([[0.0, 0.0], [3.0, 3.0]]),
            np.array([0, 1]),
        )

        prediction = model.predict(np.array([2.9, 3.0]))

        np.testing.assert_array_equal(prediction, [1])

    def test_predict_requires_fit(self) -> None:
        # this checks that using an untrained model gives a clear error.
        with self.assertRaisesRegex(RuntimeError, "fit must be called"):
            KNNClassifier().predict(np.array([[0.0, 0.0]]))

    def test_rejects_invalid_settings(self) -> None:
        # this checks that invalid neighbor counts and distance exponents fail early.
        with self.assertRaisesRegex(ValueError, "positive integer"):
            KNNClassifier(k=0)
        with self.assertRaisesRegex(ValueError, "greater than or equal to 1"):
            KNNClassifier(p=0.5)


if __name__ == "__main__":
    unittest.main()
