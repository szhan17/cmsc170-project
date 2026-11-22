# Comparison table and graphs for issue #25.
#
# This reads every results/<method>.json file (made by src/run.py), prints one
# table with accuracy, precision, recall and F1 for each method and both
# baselines, and saves the table and the Matplotlib graphs to figures/.
#
# Run from the project root: python src/compare.py

import argparse
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DEFAULT_FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# this is the order rows and bars are shown in, baselines first.
METHOD_ORDER = ["majority", "persistence", "knn", "naive_bayes", "perceptron"]
DISPLAY_NAMES = {
    "majority": "Majority (always dry)",
    "persistence": "Persistence (same as today)",
    "knn": "KNN",
    "naive_bayes": "Naive Bayes",
    "perceptron": "Perceptron",
}
BASELINES = ["majority", "persistence"]
METRIC_KEYS = ["accuracy", "precision", "recall", "f1"]
CONFUSION_KEYS = ["TP", "FP", "FN", "TN"]


def get_display_name(method):
    # this turns a name like naive_bayes into a readable label.
    if method in DISPLAY_NAMES:
        return DISPLAY_NAMES[method]
    return method.replace("_", " ").title()


def check_result(result, file_path):
    # this makes sure a results file has everything the table and graphs need.
    if "method" not in result:
        raise ValueError(file_path + ": missing key 'method'")

    if "scores" not in result:
        raise ValueError(file_path + ": missing key 'scores'")
    for key in METRIC_KEYS:
        if key not in result["scores"]:
            raise ValueError(file_path + ": 'scores' is missing " + key)

    if "confusion_matrix" not in result:
        raise ValueError(file_path + ": missing key 'confusion_matrix'")
    for key in CONFUSION_KEYS:
        if key not in result["confusion_matrix"]:
            raise ValueError(file_path + ": 'confusion_matrix' is missing " + key)


def order_results(results):
    # this puts the known methods first, in METHOD_ORDER.
    ordered = []
    for name in METHOD_ORDER:
        for result in results:
            if result["method"] == name:
                ordered.append(result)

    # this adds any other method after them, sorted by name.
    extra_names = []
    for result in results:
        if result["method"] not in METHOD_ORDER:
            extra_names.append(result["method"])
    extra_names.sort()
    for name in extra_names:
        for result in results:
            if result["method"] == name:
                ordered.append(result)
    return ordered


def load_results(results_dir=DEFAULT_RESULTS_DIR):
    # this stops early if the results folder does not exist.
    if not os.path.isdir(results_dir):
        raise FileNotFoundError("results folder not found: " + str(results_dir))

    # this reads every json file in the folder and checks it.
    results = []
    for file_name in sorted(os.listdir(results_dir)):
        if not file_name.endswith(".json"):
            continue
        file_path = os.path.join(results_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            result = json.load(file)
        check_result(result, file_path)
        results.append(result)

    if len(results) == 0:
        raise ValueError(str(results_dir) + ": no result files found, run src/run.py first")
    return order_results(results)


def format_time(result):
    # this shows the run time, or a dash if the file does not have one.
    if "run_time_seconds" in result:
        return "%.3f" % result["run_time_seconds"]
    return "-"


def format_text_table(results):
    # this builds the rows as lists of text, starting with the header.
    rows = [["Method", "Accuracy", "Precision", "Recall", "F1", "Time (s)"]]
    for result in results:
        row = [get_display_name(result["method"])]
        for key in METRIC_KEYS:
            row.append("%.4f" % result["scores"][key])
        row.append(format_time(result))
        rows.append(row)

    # this finds how wide each column needs to be.
    widths = [0, 0, 0, 0, 0, 0]
    for row in rows:
        for i in range(6):
            if len(row[i]) > widths[i]:
                widths[i] = len(row[i])

    # this builds each line, with the method name on the left and numbers on the right.
    lines = []
    for row_number in range(len(rows)):
        row = rows[row_number]
        line = row[0].ljust(widths[0])
        for i in range(1, 6):
            line = line + "  " + row[i].rjust(widths[i])
        lines.append(line)

        # this puts a dashed line under the header.
        if row_number == 0:
            dashes = "-" * widths[0]
            for i in range(1, 6):
                dashes = dashes + "  " + "-" * widths[i]
            lines.append(dashes)
    return "\n".join(lines)


def format_markdown_table(results):
    # this builds the same table in Markdown for the README and project page.
    lines = []
    lines.append("| Method | Accuracy | Precision | Recall | F1 | Time (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for result in results:
        line = "| " + get_display_name(result["method"]) + " |"
        for key in METRIC_KEYS:
            line = line + " %.4f |" % result["scores"][key]
        line = line + " " + format_time(result) + " |"
        lines.append(line)
    return "\n".join(lines)


def save_markdown_table(results, figures_dir):
    # this writes the Markdown table with a heading about the years used.
    heading = "Comparison on the validation years"
    first = results[0]
    if "train_years" in first and "validation_years" in first:
        heading = heading + " " + first["validation_years"]
        heading = heading + " (trained on " + first["train_years"] + ")"

    path = os.path.join(figures_dir, "comparison_table.md")
    with open(path, "w", encoding="utf-8") as file:
        file.write("## " + heading + "\n\n")
        file.write(format_markdown_table(results) + "\n")
    return path


def load_pyplot():
    # this loads Matplotlib without opening a window, so it works anywhere.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib is needed for the graphs: pip install matplotlib")
    return plt


def plot_f1_comparison(results, figures_dir):
    plt = load_pyplot()

    # this collects the label, F1 and color of every bar.
    names = []
    f1_scores = []
    colors = []
    for result in results:
        names.append(get_display_name(result["method"]))
        f1_scores.append(result["scores"]["f1"])
        # baselines are grey and our methods are blue, so they are easy to tell apart.
        if result["method"] in BASELINES:
            colors.append("#9e9e9e")
        else:
            colors.append("#1f77b4")

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = list(range(len(results)))
    ax.bar(positions, f1_scores, color=colors)

    # this writes each F1 value on top of its bar.
    for i in range(len(results)):
        ax.text(i, f1_scores[i] + 0.01, "%.3f" % f1_scores[i], ha="center")

    # this draws the persistence F1 as a line, since it is the score to beat.
    for result in results:
        if result["method"] == "persistence":
            ax.axhline(result["scores"]["f1"], color="black", linestyle="--",
                       linewidth=1, label="Persistence F1")
            ax.legend()

    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 on the rainy class")
    ax.set_title("F1 by method (grey = baselines)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(figures_dir, "f1_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_metrics_comparison(results, figures_dir):
    plt = load_pyplot()

    names = []
    for result in results:
        names.append(get_display_name(result["method"]))

    fig, ax = plt.subplots(figsize=(11, 5))
    bar_width = 0.2
    # this shifts each metric left or right so the four bars sit side by side.
    shifts = [-0.3, -0.1, 0.1, 0.3]

    for m in range(len(METRIC_KEYS)):
        key = METRIC_KEYS[m]
        x_values = []
        y_values = []
        for i in range(len(results)):
            x_values.append(i + shifts[m])
            y_values.append(results[i]["scores"][key])
        ax.bar(x_values, y_values, bar_width, label=key.capitalize())

    ax.set_xticks(list(range(len(results))))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Accuracy, precision, recall and F1 by method")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(figures_dir, "metrics_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion_matrices(results, figures_dir):
    plt = load_pyplot()

    # this uses up to 3 columns and adds rows until every method has a cell.
    columns = 3
    if len(results) < 3:
        columns = len(results)
    rows = 1
    while rows * columns < len(results):
        rows = rows + 1

    fig, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 4 * rows), squeeze=False)

    cell_names = [["TN", "FP"], ["FN", "TP"]]
    for i in range(rows * columns):
        ax = axes[i // columns][i % columns]

        # this hides the empty cells when the methods do not fill the grid.
        if i >= len(results):
            ax.axis("off")
            continue

        result = results[i]
        cm = result["confusion_matrix"]
        # actual is on the rows and predicted is on the columns.
        grid = [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]]
        biggest = max(max(grid[0]), max(grid[1]))
        ax.imshow(grid, cmap="Blues")

        # this writes the name and count in each cell.
        for r in range(2):
            for c in range(2):
                # white text on dark cells keeps the numbers readable.
                if grid[r][c] > biggest / 2:
                    text_color = "white"
                else:
                    text_color = "black"
                ax.text(c, r, cell_names[r][c] + "\n" + str(grid[r][c]),
                        ha="center", va="center", color=text_color)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Dry (0)", "Rainy (1)"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Dry (0)", "Rainy (1)"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(get_display_name(result["method"]) + "\nF1 = %.3f" % result["scores"]["f1"])

    fig.tight_layout()
    path = os.path.join(figures_dir, "confusion_matrices.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_comparison(results_dir=DEFAULT_RESULTS_DIR, figures_dir=DEFAULT_FIGURES_DIR):
    # this makes the table and every graph in one go.
    results = load_results(results_dir)
    os.makedirs(figures_dir, exist_ok=True)

    saved = []
    saved.append(save_markdown_table(results, figures_dir))
    saved.append(plot_f1_comparison(results, figures_dir))
    saved.append(plot_metrics_comparison(results, figures_dir))
    saved.append(plot_confusion_matrices(results, figures_dir))
    return format_text_table(results), saved


def main():
    # this reads optional folders from the command line.
    parser = argparse.ArgumentParser(description="Make the comparison table and graphs.")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="folder with the result json files")
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR, help="folder for the table and graphs")
    args = parser.parse_args()

    table, saved = generate_comparison(args.results_dir, args.figures_dir)
    print(table)
    print("\nSaved:")
    for path in saved:
        print("  " + path)


if __name__ == "__main__":
    main()
