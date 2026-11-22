import json
import os
import tempfile
import unittest

from src.compare import (
    format_markdown_table,
    format_text_table,
    generate_comparison,
    load_results,
)


def make_result(method, f1):
    # this makes a small results file with the same keys src/run.py saves.
    return {
        "method": method,
        "train_years": "2001-2017",
        "validation_years": "2018-2021",
        "scores": {"accuracy": 0.7, "precision": 0.6, "recall": 0.5, "f1": f1},
        "confusion_matrix": {"TP": 10, "FP": 5, "FN": 4, "TN": 20},
        "run_time_seconds": 0.5,
    }


class CompareTests(unittest.TestCase):
    def make_folder(self, results):
        # this writes the results to a temporary folder that is deleted afterwards.
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        for result in results:
            path = os.path.join(temporary.name, result["method"] + ".json")
            with open(path, "w", encoding="utf-8") as file:
                json.dump(result, file)
        return temporary.name

    def test_load_results_puts_baselines_first(self):
        # this checks the order does not depend on the file names.
        results = []
        for name in ["perceptron", "naive_bayes", "knn", "persistence", "majority"]:
            results.append(make_result(name, 0.5))
        folder = self.make_folder(results)

        methods = []
        for result in load_results(folder):
            methods.append(result["method"])

        self.assertEqual(methods, ["majority", "persistence", "knn", "naive_bayes", "perceptron"])

    def test_load_results_rejects_empty_folder(self):
        # this checks an empty folder gives a clear error.
        with self.assertRaisesRegex(ValueError, "no result files"):
            load_results(self.make_folder([]))

    def test_load_results_rejects_missing_keys(self):
        # this checks a file without scores is not accepted.
        bad = make_result("knn", 0.5)
        del bad["scores"]

        with self.assertRaisesRegex(ValueError, "missing key 'scores'"):
            load_results(self.make_folder([bad]))

    def test_tables_list_every_method_and_score(self):
        # this checks both tables show every method with its scores.
        results = [make_result("majority", 0.0), make_result("knn", 0.6717)]

        for table in [format_text_table(results), format_markdown_table(results)]:
            self.assertIn("Majority (always dry)", table)
            self.assertIn("KNN", table)
            self.assertIn("0.6717", table)
            self.assertIn("0.7000", table)

    def test_generate_comparison_saves_table_and_graphs(self):
        # this checks one call makes the table and all three graphs.
        try:
            import matplotlib
        except ImportError:
            self.skipTest("matplotlib is not installed")
        folder = self.make_folder([make_result("majority", 0.0), make_result("knn", 0.6)])
        figures = os.path.join(folder, "figures")

        table, saved = generate_comparison(folder, figures)

        names = []
        for path in saved:
            names.append(os.path.basename(path))
            self.assertGreater(os.path.getsize(path), 0)
        names.sort()
        self.assertEqual(
            names,
            ["comparison_table.md", "confusion_matrices.png", "f1_comparison.png", "metrics_comparison.png"],
        )


if __name__ == "__main__":
    unittest.main()
