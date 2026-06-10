from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import LABEL_ORDER, ensure_directories, find_feature_files, read_feature_chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze generated number-theory class distributions.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--hist-bins", type=int, default=80)
    return parser.parse_args()


def save_class_distribution(counts: Counter, total: int, figures_dir: Path, results_dir: Path) -> None:
    rows = []
    for label in LABEL_ORDER:
        count = int(counts[label])
        rows.append({"label": label, "count": count, "share": count / total if total else 0.0})
    summary = pd.DataFrame(rows)
    summary.to_csv(results_dir / "distribution_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary["label"], summary["count"], color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_title("Class distribution")
    ax.set_xlabel("class")
    ax.set_ylabel("count")
    ax.ticklabel_format(axis="y", style="plain")
    fig.tight_layout()
    fig.savefig(figures_dir / "class_distribution.png", dpi=160)
    plt.close(fig)


def save_sigma_histogram(
    files: list[Path],
    sigma_min: float,
    sigma_max: float,
    bins_count: int,
    figures_dir: Path,
) -> None:
    if sigma_min == sigma_max:
        sigma_min -= 0.5
        sigma_max += 0.5
    bins = np.linspace(sigma_min, sigma_max, bins_count + 1)
    hist = np.zeros(bins_count, dtype=np.int64)

    for path in files:
        frame = read_feature_chunk(path, columns=["sigma_ratio"])
        hist += np.histogram(frame["sigma_ratio"].to_numpy(), bins=bins)[0]

    centers = (bins[:-1] + bins[1:]) / 2
    width = bins[1] - bins[0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(centers, hist, width=width, color="#72B7B2", align="center")
    ax.set_title("Distribution of sigma(n) / n")
    ax.set_xlabel("sigma_ratio")
    ax.set_ylabel("count")
    ax.ticklabel_format(axis="y", style="plain")
    fig.tight_layout()
    fig.savefig(figures_dir / "sigma_ratio_histogram.png", dpi=160)
    plt.close(fig)


def save_omega_abundant_rate(
    omega_counts: dict[int, int],
    omega_abundant: dict[int, int],
    figures_dir: Path,
    results_dir: Path,
) -> None:
    rows = []
    for omega in sorted(omega_counts):
        total = omega_counts[omega]
        abundant = omega_abundant.get(omega, 0)
        rows.append(
            {
                "omega_n": omega,
                "total": total,
                "abundant": abundant,
                "abundant_share": abundant / total if total else 0.0,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(results_dir / "omega_abundant_rate.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frame["omega_n"], frame["abundant_share"], marker="o", color="#E45756")
    ax.set_title("Abundant share by omega(n)")
    ax.set_xlabel("omega_n")
    ax.set_ylabel("abundant share")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "omega_abundant_rate.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_directories(args.figures_dir, args.results_dir)
    files = find_feature_files(args.data_dir)

    counts: Counter = Counter()
    omega_counts: defaultdict[int, int] = defaultdict(int)
    omega_abundant: defaultdict[int, int] = defaultdict(int)
    sigma_min = np.inf
    sigma_max = -np.inf
    total = 0

    for path in files:
        frame = read_feature_chunk(path, columns=["label", "sigma_ratio", "omega_n"])
        total += len(frame)
        counts.update(frame["label"].astype(str))
        sigma_values = frame["sigma_ratio"].to_numpy()
        sigma_min = min(sigma_min, float(np.nanmin(sigma_values)))
        sigma_max = max(sigma_max, float(np.nanmax(sigma_values)))

        grouped = frame.assign(is_abundant=frame["label"].astype(str) == "abundant").groupby("omega_n")
        for omega, group in grouped:
            omega_key = int(omega)
            omega_counts[omega_key] += int(len(group))
            omega_abundant[omega_key] += int(group["is_abundant"].sum())

    save_class_distribution(counts, total, args.figures_dir, args.results_dir)
    save_sigma_histogram(files, sigma_min, sigma_max, args.hist_bins, args.figures_dir)
    save_omega_abundant_rate(omega_counts, omega_abundant, args.figures_dir, args.results_dir)

    print(f"Analyzed {total:,} rows from {len(files)} chunk file(s).")
    for label in LABEL_ORDER:
        print(f"{label}: {counts[label]:,}")
    print(f"Results written to {args.results_dir}")
    print(f"Figures written to {args.figures_dir}")


if __name__ == "__main__":
    main()
