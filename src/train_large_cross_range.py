from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from sample_large_dataset import filter_range, proportional_allocation
from utils import (
    RESEARCH_FEATURE_SETS,
    binary_metric_row,
    binary_target,
    ensure_directories,
    find_feature_files,
    read_feature_chunk,
)


FEATURE_COLUMNS = RESEARCH_FEATURE_SETS["full_without_sigma_ratio"]


def parse_range(value: str) -> tuple[int, int]:
    start_text, end_text = value.split(":", 1)
    start = int(start_text)
    end = int(end_text)
    if start > end:
        raise ValueError(f"Invalid range {value}: start must be <= end")
    return start, end


def parse_ranges(value: str) -> list[tuple[int, int]]:
    return [parse_range(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Large cross-range generalization experiment.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument("--train-max-n", type=int, default=100_000)
    parser.add_argument("--test-ranges", type=str, default="100001:1000000,1000001:10000000")
    parser.add_argument("--sample-per-range", type=int, default=200_000)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_training_rows(files: list[Path], train_max_n: int) -> pd.DataFrame:
    columns = ["n", "label", *FEATURE_COLUMNS]
    pieces = []
    for path in files:
        frame = read_feature_chunk(path, columns=columns)
        selected = frame[frame["n"] <= train_max_n]
        if not selected.empty:
            pieces.append(selected)
    if not pieces:
        raise ValueError(f"No training rows found for n <= {train_max_n}")
    return pd.concat(pieces, ignore_index=True)


def count_range_rows(files: list[Path], start: int, end: int) -> list[int]:
    counts = []
    for path in files:
        frame = filter_range(read_feature_chunk(path, columns=["n"]), start, end)
        counts.append(len(frame))
    return counts


def sample_range_rows(
    files: list[Path],
    start: int,
    end: int,
    sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    counts = count_range_rows(files, start, end)
    allocation = proportional_allocation(counts, sample_size)
    columns = ["n", "label", *FEATURE_COLUMNS]
    pieces = []
    for file_index, (path, take) in enumerate(zip(files, allocation)):
        if take == 0:
            continue
        frame = filter_range(read_feature_chunk(path, columns=columns), start, end)
        pieces.append(
            frame.sample(
                n=min(take, len(frame)),
                random_state=random_state + file_index,
            )
        )
    if not pieces:
        raise ValueError(f"No test rows found for range {start}-{end}")
    return pd.concat(pieces, ignore_index=True).sample(frac=1.0, random_state=random_state)


def save_plot(metrics: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for model_name, group in metrics.groupby("model", sort=False):
        ax.plot(group["test_range"], group["f1_abundant"], marker="o", label=model_name)
    ax.set_xlabel("test range")
    ax.set_ylabel("F1 abundant")
    ax.set_ylim(0, 1.02)
    ax.set_title("Large cross-range generalization")
    ax.tick_params(axis="x", rotation=20)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "large_cross_range_f1.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.sample_per_range < 1:
        raise ValueError("--sample-per-range must be positive")
    ensure_directories(args.results_dir, args.figures_dir)
    files = find_feature_files(args.data_dir)
    test_ranges = parse_ranges(args.test_ranges)

    train_frame = load_training_rows(files, args.train_max_n)
    y_train = binary_target(train_frame["label"])
    models = {
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.08,
            random_state=args.random_state,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=160,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=args.random_state,
        ),
    }
    for model in models.values():
        model.fit(train_frame[FEATURE_COLUMNS], y_train)

    rows = []
    for start, end in test_ranges:
        test_frame = sample_range_rows(files, start, end, args.sample_per_range, args.random_state)
        y_test = binary_target(test_frame["label"])
        for model_name, model in models.items():
            y_pred = model.predict(test_frame[FEATURE_COLUMNS])
            rows.append(
                binary_metric_row(
                    y_test,
                    y_pred,
                    {
                        "model": model_name,
                        "train_range": f"1-{args.train_max_n}",
                        "test_range": f"{start}-{end}",
                        "train_rows": len(train_frame),
                        "test_rows": len(test_frame),
                    },
                )
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.results_dir / "large_cross_range_metrics.csv", index=False)
    save_plot(metrics, args.figures_dir)
    print(metrics[["model", "train_range", "test_range", "accuracy", "f1_abundant", "balanced_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
