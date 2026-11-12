import unittest
import numpy as np

from src.baselines import majority_class, predict_majority, predict_persistence


class TestBaselines(unittest.TestCase):
    def test_majority_class_zero(self):
        self.assertEqual(majority_class([0, 0, 1]), 0)

    def test_majority_class_one(self):
        self.assertEqual(majority_class([1, 1, 0]), 1)

    def test_majority_tie_resolves_to_zero(self):
        self.assertEqual(majority_class([0, 1]), 0)

    def test_majority_prediction_length_and_values(self):
        np.testing.assert_array_equal(
            predict_majority(4, 1),
            np.array([1, 1, 1, 1], dtype=np.int32)
        )

    def test_persistence_uses_five_mm_threshold(self):
        X = np.zeros((4, 14), dtype=float)
        X[:, 9] = [0.0, 4.99, 5.0, 12.0]
        np.testing.assert_array_equal(
            predict_persistence(X),
            np.array([0, 0, 1, 1], dtype=np.int32)
        )

    def test_persistence_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            predict_persistence(np.zeros((3, 5)))

    def test_majority_rejects_empty_labels(self):
        with self.assertRaises(ValueError):
            majority_class([])


if __name__ == "__main__":
    unittest.main()
