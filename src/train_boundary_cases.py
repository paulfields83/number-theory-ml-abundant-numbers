from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

from utils import (
    RESEARCH_FEATURE_SETS,
    binary_metric_row,
    binary_target,
    ensure_directories,
    find_feature_files,
    read_feature_chunk,
    safe_stratify_target,
    slugify,
)


BOUNDARY_WINDOWS = [(1.8, 2.2), (1.9, 2.1), (1.95, 2.05)]
FEATURE_COLUMNS = RESEARCH_FEATURE_SETS["full_without_sigma_ratio"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classifiers on sigma_ratio boundary cases.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--max-rows-per-window", type=int, default=150_000)
    return parser.parse_args()


def load_window(data_dir: Path, lower: float, upper: float, max_rows: int, random_state: int) -> pd.DataFrame:
    columns = ["label", "sigma_ratio", *FEATURE_COLUMNS]
    parts = []
    for path in find_feature_files(data_dir):
        frame = read_feature_chunk(path, columns=columns)
        selected = frame[(frame["sigma_ratio"] >= lower) & (frame["sigma_ratio"] <= upper)]
        if not selected.empty:
            parts.append(selected)
    if not parts:
        return pd.DataFrame(columns=columns)
    result = pd.concat(parts, ignore_index=True)
    if max_rows > 0 and len(result) > max_rows:
        result = result.sample(n=max_rows, random_state=random_state)
    return result.reset_index(drop=True)


def save_confusion(y_true: pd.Series, y_pred: np.ndarray, window_name: str, figures_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=["non_abundant", "abundant"])
    display = ConfusionMatrixDisplay(cm, display_labels=["non_abundant", "abundant"])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Boundary {window_name}")
    fig.tight_layout()
    fig.savefig(figures_dir / f"confusion_matrix_boundary_{slugify(window_name)}.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_directories(args.figures_dir, args.results_dir)
    metric_rows = []

    for lower, upper in BOUNDARY_WINDOWS:
        frame = load_window(args.data_dir, lower, upper, args.max_rows_per_window, args.random_state)
        window_name = f"{lower:.2f}_{upper:.2f}"
        y = binary_target(frame["label"]) if not frame.empty else pd.Series(dtype=object)
        if frame.empty or y.nunique() < 2 or y.value_counts().min() < 2:
            print(f"Skipping boundary {lower}-{upper}: not enough class variation.")
            continue
        X_train, X_test, y_train, y_test = train_test_split(
            frame[FEATURE_COLUMNS],
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=safe_stratify_target(y),
        )
        model = HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.08,
            random_state=args.random_state,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        save_confusion(y_test, y_pred, window_name, args.figures_dir)
        counts = y.value_counts()
        metric_rows.append(
            binary_metric_row(
                y_test,
                y_pred,
                {
                    "sigma_ratio_min": lower,
                    "sigma_ratio_max": upper,
                    "model": "hist_gradient_boosting",
                    "features": ",".join(FEATURE_COLUMNS),
                    "window_rows": len(frame),
                    "window_abundant": int(counts.get("abundant", 0)),
                    "window_non_abundant": int(counts.get("non_abundant", 0)),
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                },
            )
        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.results_dir / "boundary_case_metrics.csv", index=False)
    if not metrics.empty:
        labels = [f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(metrics["sigma_ratio_min"], metrics["sigma_ratio_max"])]
        x = np.arange(len(metrics))
        fig, ax = plt.subplots(figsize=(7, 4.6))
        ax.bar(x, metrics["accuracy"], color="#72B7B2", label="accuracy")
        ax.plot(x, metrics["f1_abundant"], marker="o", color="#E45756", label="F1 abundant")
        ax.set_xticks(x, labels=labels)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("sigma_ratio window used only for sampling")
        ax.set_ylabel("score")
        ax.set_title("Boundary case classification")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.figures_dir / "boundary_case_accuracy.png", dpi=160)
        plt.close(fig)

    print(f"Evaluated {len(metrics)} boundary windows.")
    if not metrics.empty:
        print(metrics[["sigma_ratio_min", "sigma_ratio_max", "window_rows", "accuracy", "f1_abundant"]].to_string(index=False))


if __name__ == "__main__":
    main()
