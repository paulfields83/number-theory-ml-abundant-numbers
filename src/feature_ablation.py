from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from train_models import load_training_sample
from utils import (
    RESEARCH_FEATURE_SETS,
    binary_metric_row,
    binary_target,
    ensure_directories,
    safe_stratify_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare feature sets with sigma_ratio removed.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    return parser.parse_args()


def plot_metric(metrics: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    ordered = metrics.sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(ordered["feature_set"], ordered[metric], color="#59A14F")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_directories(args.figures_dir, args.results_dir)
    sample = load_training_sample(args.data_dir, args.sample_size, args.random_state)
    y = binary_target(sample["label"])

    metric_rows = []
    for feature_set, columns in RESEARCH_FEATURE_SETS.items():
        X = sample[columns].copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=safe_stratify_target(y),
        )
        model = HistGradientBoostingClassifier(
            max_iter=140,
            learning_rate=0.08,
            random_state=args.random_state,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metric_rows.append(
            binary_metric_row(
                y_test,
                y_pred,
                {
                    "feature_set": feature_set,
                    "features": ",".join(columns),
                    "model": "hist_gradient_boosting",
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                },
            )
        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.results_dir / "feature_ablation_metrics.csv", index=False)
    plot_metric(
        metrics,
        "accuracy",
        args.figures_dir / "feature_ablation_accuracy.png",
        "Feature ablation: accuracy",
    )
    plot_metric(
        metrics,
        "f1_abundant",
        args.figures_dir / "feature_ablation_f1.png",
        "Feature ablation: abundant F1",
    )

    print(f"Evaluated {len(metrics)} feature sets on a {len(sample):,}-row sample.")
    print(metrics[["feature_set", "accuracy", "f1_abundant", "balanced_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
