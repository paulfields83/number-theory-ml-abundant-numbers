from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils import ensure_directories, find_feature_files, read_feature_chunk


DEFAULT_RANGE_BINS = [
    (1, 100_000),
    (100_001, 1_000_000),
    (1_000_001, 10_000_000),
    (10_000_001, 50_000_000),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk-safe distribution analysis for large datasets.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def update_group_counts(
    frame: pd.DataFrame,
    column: str,
    totals: defaultdict[int, int],
    abundant: defaultdict[int, int],
) -> None:
    grouped = frame.groupby(column, observed=True)["label"].agg(
        total="size",
        abundant=lambda values: int((values.astype(str) == "abundant").sum()),
    )
    for value, row in grouped.iterrows():
        key = int(value)
        totals[key] += int(row["total"])
        abundant[key] += int(row["abundant"])


def update_range_counts(frame: pd.DataFrame, range_counts: dict[str, Counter]) -> None:
    for start, end in DEFAULT_RANGE_BINS:
        name = f"{start}-{end}"
        selected = frame[(frame["n"] >= start) & (frame["n"] <= end)]
        if selected.empty:
            continue
        range_counts[name].update(selected["label"].astype(str))


def group_frame(column: str, totals: dict[int, int], abundant: dict[int, int]) -> pd.DataFrame:
    rows = []
    for value in sorted(totals):
        total = int(totals[value])
        abundant_count = int(abundant.get(value, 0))
        rows.append(
            {
                column: value,
                "total": total,
                "abundant": abundant_count,
                "abundant_share": abundant_count / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def save_omega_plot(omega_frame: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(omega_frame["omega_n"], omega_frame["abundant_share"], marker="o", color="#E45756")
    ax.set_xlabel("omega_n")
    ax.set_ylabel("abundant share")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Large dataset abundant share by omega(n)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "large_omega_abundant_rate.png", dpi=160)
    plt.close(fig)


def save_range_plot(range_frame: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(range_frame["range"], range_frame["abundant_share"], color="#4C78A8")
    ax.set_xlabel("n range")
    ax.set_ylabel("abundant share")
    ax.set_ylim(0, 1.02)
    ax.set_title("Abundant share by numeric range")
    ax.tick_params(axis="x", rotation=20)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "large_range_abundant_share.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_directories(args.results_dir, args.figures_dir)
    files = find_feature_files(args.data_dir)

    total_rows = 0
    label_counts: Counter = Counter()
    omega_totals: defaultdict[int, int] = defaultdict(int)
    omega_abundant: defaultdict[int, int] = defaultdict(int)
    big_omega_totals: defaultdict[int, int] = defaultdict(int)
    big_omega_abundant: defaultdict[int, int] = defaultdict(int)
    range_counts: dict[str, Counter] = {
        f"{start}-{end}": Counter() for start, end in DEFAULT_RANGE_BINS
    }

    for path in files:
        frame = read_feature_chunk(path, columns=["n", "label", "omega_n", "Omega_n"])
        total_rows += len(frame)
        label_counts.update(frame["label"].astype(str))
        update_group_counts(frame, "omega_n", omega_totals, omega_abundant)
        update_group_counts(frame, "Omega_n", big_omega_totals, big_omega_abundant)
        update_range_counts(frame, range_counts)
        print(f"Scanned {path.name}: {len(frame):,} rows")

    abundant_count = int(label_counts.get("abundant", 0))
    summary = pd.DataFrame(
        [
            {
                "total_rows": total_rows,
                "deficient": int(label_counts.get("deficient", 0)),
                "perfect": int(label_counts.get("perfect", 0)),
                "abundant": abundant_count,
                "abundant_share": abundant_count / total_rows if total_rows else 0.0,
            }
        ]
    )
    omega_frame = group_frame("omega_n", omega_totals, omega_abundant)
    big_omega_frame = group_frame("Omega_n", big_omega_totals, big_omega_abundant)
    range_rows = []
    for range_name, counts in range_counts.items():
        total = sum(counts.values())
        abundant = int(counts.get("abundant", 0))
        range_rows.append(
            {
                "range": range_name,
                "total": total,
                "deficient": int(counts.get("deficient", 0)),
                "perfect": int(counts.get("perfect", 0)),
                "abundant": abundant,
                "abundant_share": abundant / total if total else 0.0,
            }
        )
    range_frame = pd.DataFrame(range_rows)

    summary.to_csv(args.results_dir / "large_distribution_summary.csv", index=False)
    omega_frame.to_csv(args.results_dir / "large_omega_abundant_rate.csv", index=False)
    big_omega_frame.to_csv(args.results_dir / "large_Omega_abundant_rate.csv", index=False)
    range_frame.to_csv(args.results_dir / "large_range_summary.csv", index=False)
    if not omega_frame.empty:
        save_omega_plot(omega_frame, args.figures_dir)
    if not range_frame.empty:
        save_range_plot(range_frame, args.figures_dir)

    print(summary.to_string(index=False))
    print(f"Results written to {args.results_dir}")
    print(f"Figures written to {args.figures_dir}")


if __name__ == "__main__":
    main()
