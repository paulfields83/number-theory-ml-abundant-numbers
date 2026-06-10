from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import binary_metric_row, binary_target, ensure_directories, find_feature_files, read_feature_chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate simple rule baselines for abundant numbers.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def predict_rules(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "omega_ge_4": frame["omega_n"].to_numpy() >= 4,
        "omega_ge_5": frame["omega_n"].to_numpy() >= 5,
        "tau_ge_12": frame["tau_n"].to_numpy() >= 12,
        "tau_ge_24": frame["tau_n"].to_numpy() >= 24,
        "log_omega_combo": (frame["omega_n"].to_numpy() >= 4)
        | ((frame["omega_n"].to_numpy() >= 3) & (frame["log_n"].to_numpy() >= np.log(10_000))),
    }


def main() -> None:
    args = parse_args()
    ensure_directories(args.figures_dir, args.results_dir)
    files = find_feature_files(args.data_dir)
    rows: list[pd.DataFrame] = []
    for path in files:
        rows.append(read_feature_chunk(path, columns=["label", "log_n", "omega_n", "tau_n"]))
    frame = pd.concat(rows, ignore_index=True)
    y_true = binary_target(frame["label"])

    metric_rows = []
    for rule_name, abundant_mask in predict_rules(frame).items():
        y_pred = pd.Series(
            np.where(abundant_mask, "abundant", "non_abundant"),
            index=frame.index,
            name="prediction",
        )
        metric_rows.append(binary_metric_row(y_true, y_pred, {"rule": rule_name}))

    metrics = pd.DataFrame(metric_rows).sort_values("f1_abundant", ascending=False)
    metrics.to_csv(args.results_dir / "rule_baselines.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(metrics))
    width = 0.36
    ax.bar(x - width / 2, metrics["accuracy"], width=width, label="accuracy", color="#4C78A8")
    ax.bar(x + width / 2, metrics["f1_abundant"], width=width, label="F1 abundant", color="#F58518")
    ax.set_xticks(x, labels=metrics["rule"], rotation=25, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("score")
    ax.set_title("Rule baseline comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.figures_dir / "rule_baseline_comparison.png", dpi=160)
    plt.close(fig)

    print(f"Evaluated {len(metrics)} rule baselines on {len(frame):,} rows.")
    print(metrics[["rule", "accuracy", "precision_abundant", "recall_abundant", "f1_abundant"]].to_string(index=False))


if __name__ == "__main__":
    main()
