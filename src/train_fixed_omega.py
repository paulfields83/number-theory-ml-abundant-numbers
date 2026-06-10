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
    binary_metric_row,
    binary_target,
    ensure_directories,
    find_feature_files,
    read_feature_chunk,
    safe_stratify_target,
    slugify,
)


FEATURE_COLUMNS = ["log_n", "tau_n", "Omega_n", "min_prime_factor", "max_prime_factor"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train binary classifiers within fixed omega(n) groups.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--omegas", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--max-rows-per-omega", type=int, default=150_000)
    return parser.parse_args()


def load_fixed_omega_data(data_dir: Path, omegas: list[int], max_rows: int, random_state: int) -> dict[int, pd.DataFrame]:
    pieces = {omega: [] for omega in omegas}
    columns = ["label", "omega_n", *FEATURE_COLUMNS]
    for path in find_feature_files(data_dir):
        frame = read_feature_chunk(path, columns=columns)
        for omega in omegas:
            group = frame[frame["omega_n"] == omega]
            if not group.empty:
                pieces[omega].append(group)

    result: dict[int, pd.DataFrame] = {}
    for omega, frames in pieces.items():
        if not frames:
            continue
        group = pd.concat(frames, ignore_index=True)
        if max_rows > 0 and len(group) > max_rows:
            group = group.sample(n=max_rows, random_state=random_state + omega)
        result[omega] = group.reset_index(drop=True)
    return result


def save_confusion(y_true: pd.Series, y_pred: np.ndarray, omega: int, figures_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=["non_abundant", "abundant"])
    display = ConfusionMatrixDisplay(cm, display_labels=["non_abundant", "abundant"])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Fixed omega={omega}")
    fig.tight_layout()
    fig.savefig(figures_dir / f"confusion_matrix_fixed_omega_{omega}.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_directories(args.figures_dir, args.results_dir)
    groups = load_fixed_omega_data(args.data_dir, args.omegas, args.max_rows_per_omega, args.random_state)
    metric_rows = []

    for omega, frame in sorted(groups.items()):
        y = binary_target(frame["label"])
        if y.nunique() < 2 or y.value_counts().min() < 2:
            print(f"Skipping omega={omega}: not enough class variation.")
            continue
        X_train, X_test, y_train, y_test = train_test_split(
            frame[FEATURE_COLUMNS],
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=safe_stratify_target(y),
        )
        model = HistGradientBoostingClassifier(
            max_iter=140,
            learning_rate=0.08,
            random_state=args.random_state + omega,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        save_confusion(y_test, y_pred, omega, args.figures_dir)
        counts = y.value_counts()
        metric_rows.append(
            binary_metric_row(
                y_test,
                y_pred,
                {
                    "omega_n": omega,
                    "model": "hist_gradient_boosting",
                    "features": ",".join(FEATURE_COLUMNS),
                    "group_rows": len(frame),
                    "group_abundant": int(counts.get("abundant", 0)),
                    "group_non_abundant": int(counts.get("non_abundant", 0)),
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                },
            )
        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.results_dir / "fixed_omega_metrics.csv", index=False)
    if not metrics.empty:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        labels = [f"omega={omega}" for omega in metrics["omega_n"]]
        x = np.arange(len(metrics))
        ax.bar(x, metrics["accuracy"], color="#4C78A8", label="accuracy")
        ax.plot(x, metrics["f1_abundant"], marker="o", color="#E45756", label="F1 abundant")
        ax.set_xticks(x, labels=labels)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("score")
        ax.set_title("Fixed omega experiment")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.figures_dir / "fixed_omega_accuracy.png", dpi=160)
        plt.close(fig)

    print(f"Evaluated {len(metrics)} fixed-omega groups.")
    if not metrics.empty:
        print(metrics[["omega_n", "group_rows", "accuracy", "f1_abundant", "balanced_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
