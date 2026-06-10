from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from utils import (
    RESEARCH_FEATURE_SETS,
    binary_metric_row,
    binary_target,
    ensure_directories,
    find_feature_files,
    read_feature_chunk,
)


FEATURE_COLUMNS = RESEARCH_FEATURE_SETS["full_without_sigma_ratio"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train on a low range and test on a higher range.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--train-max-n", type=int, default=100_000)
    parser.add_argument("--test-min-n", type=int, default=100_001)
    parser.add_argument("--test-max-n", type=int, default=1_000_000)
    parser.add_argument("--max-test-rows", type=int, default=200_000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_ranges(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["n", "label", *FEATURE_COLUMNS]
    train_parts = []
    test_parts = []
    for path in find_feature_files(args.data_dir):
        frame = read_feature_chunk(path, columns=columns)
        train = frame[frame["n"] <= args.train_max_n]
        test = frame[(frame["n"] >= args.test_min_n) & (frame["n"] <= args.test_max_n)]
        if not train.empty:
            train_parts.append(train)
        if not test.empty:
            test_parts.append(test)

    train_frame = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=columns)
    test_frame = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=columns)
    if args.max_test_rows > 0 and len(test_frame) > args.max_test_rows:
        test_frame = test_frame.sample(n=args.max_test_rows, random_state=args.random_state)
    return train_frame.reset_index(drop=True), test_frame.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    ensure_directories(args.figures_dir, args.results_dir)
    train_frame, test_frame = load_ranges(args)
    if train_frame.empty or test_frame.empty:
        raise ValueError(
            "Cross-range experiment needs rows in both ranges. Generate or point --data-dir to a 1,000,000-row dataset."
        )

    y_train = binary_target(train_frame["label"])
    y_test = binary_target(test_frame["label"])
    models = {
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.08,
            random_state=args.random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=160,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=args.random_state,
        ),
    }

    rows = []
    for model_name, model in models.items():
        model.fit(train_frame[FEATURE_COLUMNS], y_train)
        y_pred = model.predict(test_frame[FEATURE_COLUMNS])
        rows.append(
            binary_metric_row(
                y_test,
                y_pred,
                {
                    "model": model_name,
                    "features": ",".join(FEATURE_COLUMNS),
                    "train_range": f"1-{args.train_max_n}",
                    "test_range": f"{args.test_min_n}-{args.test_max_n}",
                    "train_rows": len(train_frame),
                    "test_rows": len(test_frame),
                },
            )
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.results_dir / "cross_range_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    x = np.arange(len(metrics))
    width = 0.34
    ax.bar(x - width / 2, metrics["accuracy"], width=width, label="accuracy", color="#4C78A8")
    ax.bar(x + width / 2, metrics["f1_abundant"], width=width, label="F1 abundant", color="#F58518")
    ax.set_xticks(x, labels=metrics["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("score")
    ax.set_title("Cross-range generalization")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.figures_dir / "cross_range_comparison.png", dpi=160)
    plt.close(fig)

    print(f"Train rows: {len(train_frame):,}; test rows: {len(test_frame):,}")
    print(metrics[["model", "accuracy", "f1_abundant", "balanced_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
