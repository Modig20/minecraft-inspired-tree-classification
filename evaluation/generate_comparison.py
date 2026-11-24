import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def generate(results_dir):
    results_dir = Path(results_dir)
    source = results_dir / "evaluation_results.json"
    if not source.exists():
        raise FileNotFoundError("Evaluation results missing. Run evaluate_models.py first.")
    data = json.loads(source.read_text(encoding="utf-8"))
    table = pd.DataFrame(data["models"])
    table.to_csv(results_dir / "model_comparison.csv", index=False)
    with (results_dir / "model_comparison.md").open("w", encoding="utf-8") as handle:
        handle.write(table.to_markdown(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(table["model"], table["accuracy"], color=["#4C78A8", "#F58518", "#54A24B"][:len(table)])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Test Accuracy Comparison")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(results_dir / "accuracy_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.22
    for offset, (column, label) in enumerate([
        ("accuracy", "Accuracy"), ("precision", "Precision"),
        ("recall", "Recall"), ("f1_score", "F1"),
    ]):
        ax.bar([i + offset * width for i in range(len(table))],
               table[column], width=width, label=label)
    ax.set_xticks([i + width * 1.5 for i in range(len(table))], table["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Classification Metrics")
    fig.tight_layout()
    fig.savefig(results_dir / "metrics_comparison.png", dpi=150)
    plt.close(fig)

    if "prediction_time_per_image_seconds" in table:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(table["model"], table["prediction_time_per_image_seconds"] * 1000)
        ax.set_ylabel("Milliseconds per image")
        ax.set_title("Prediction Speed")
        plt.xticks(rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(results_dir / "prediction_speed_comparison.png", dpi=150)
        plt.close(fig)
    print(table.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    generate(args.results_dir)


if __name__ == "__main__":
    main()
