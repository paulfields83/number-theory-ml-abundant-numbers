from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import binary_target, ensure_directories, find_feature_files, read_feature_chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a sampled Parquet dataset from chunked features.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument("--output-path", type=Path, default=Path("data/samples/sample_10m.parquet"))
    parser.add_argument("--sample-size", type=int, default=500_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-n", type=int, default=None)
    parser.add_argument("--max-n", type=int, default=None)
    parser.add_argument("--stratify-label", dest="stratify_label", action="store_true", default=True)
    parser.add_argument("--no-stratify-label", dest="stratify_label", action="store_false")
    return parser.parse_args()


def filter_range(frame: pd.DataFrame, min_n: int | None, max_n: int | None) -> pd.DataFrame:
    if min_n is not None:
        frame = frame[frame["n"] >= min_n]
    if max_n is not None:
        frame = frame[frame["n"] <= max_n]
    return frame


def proportional_allocation(counts: list[int], target: int) -> list[int]:
    total = sum(counts)
    if target <= 0 or total <= 0:
        return [0 for _ in counts]
    target = min(target, total)
    raw = [count * target / total for count in counts]
    allocation = [min(count, int(value)) for count, value in zip(counts, raw)]
    remaining = target - sum(allocation)
    order = sorted(range(len(counts)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    while remaining > 0:
        progressed = False
        for index in order:
            if allocation[index] < counts[index]:
                allocation[index] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    return allocation


def target_label_counts(label_counts: dict[str, int], sample_size: int, stratify_label: bool) -> dict[str, int]:
    available = sum(label_counts.values())
    sample_size = min(sample_size, available)
    if not stratify_label:
        labels = sorted(label_counts)
        allocation = proportional_allocation([label_counts[label] for label in labels], sample_size)
        return dict(zip(labels, allocation))

    abundant_available = label_counts.get("abundant", 0)
    non_available = label_counts.get("non_abundant", 0)
    abundant_target = min(abundant_available, sample_size // 2)
    non_target = min(non_available, sample_size - abundant_target)
    remaining = sample_size - abundant_target - non_target
    if remaining > 0 and abundant_target < abundant_available:
        add = min(remaining, abundant_available - abundant_target)
        abundant_target += add
        remaining -= add
    if remaining > 0 and non_target < non_available:
        add = min(remaining, non_available - non_target)
        non_target += add
    return {"abundant": abundant_target, "non_abundant": non_target}


def count_chunks(files: list[Path], min_n: int | None, max_n: int | None) -> pd.DataFrame:
    rows = []
    for file_index, path in enumerate(files):
        frame = filter_range(read_feature_chunk(path, columns=["n", "label"]), min_n, max_n)
        labels = binary_target(frame["label"]) if not frame.empty else pd.Series(dtype=object)
        rows.append(
            {
                "file_index": file_index,
                "path": path,
                "total": len(frame),
                "abundant": int((labels == "abundant").sum()),
                "non_abundant": int((labels == "non_abundant").sum()),
            }
        )
    return pd.DataFrame(rows)


def build_chunk_plan(counts: pd.DataFrame, targets: dict[str, int]) -> pd.DataFrame:
    plan = counts[["file_index", "path"]].copy()
    for label in ["abundant", "non_abundant"]:
        plan[f"{label}_sample"] = proportional_allocation(counts[label].astype(int).tolist(), targets.get(label, 0))
    return plan


def main() -> None:
    args = parse_args()
    if args.sample_size < 1:
        raise ValueError("--sample-size must be positive")
    ensure_directories(args.output_path.parent)
    files = find_feature_files(args.data_dir)

    counts = count_chunks(files, args.min_n, args.max_n)
    label_counts = {
        "abundant": int(counts["abundant"].sum()),
        "non_abundant": int(counts["non_abundant"].sum()),
    }
    targets = target_label_counts(label_counts, args.sample_size, args.stratify_label)
    plan = build_chunk_plan(counts, targets)

    pieces = []
    for _, row in plan.iterrows():
        needed = int(row["abundant_sample"]) + int(row["non_abundant_sample"])
        if needed == 0:
            continue
        frame = filter_range(read_feature_chunk(row["path"]), args.min_n, args.max_n).copy()
        frame["_binary_label"] = binary_target(frame["label"])
        for label in ["abundant", "non_abundant"]:
            take = int(row[f"{label}_sample"])
            if take == 0:
                continue
            subset = frame[frame["_binary_label"] == label]
            pieces.append(
                subset.sample(
                    n=min(take, len(subset)),
                    random_state=args.random_state + int(row["file_index"]) + (0 if label == "abundant" else 10_000),
                )
            )

    if not pieces:
        raise ValueError("No rows were sampled. Check --data-dir and range filters.")
    sample = pd.concat(pieces, ignore_index=True)
    if len(sample) > args.sample_size:
        sample = sample.sample(n=args.sample_size, random_state=args.random_state)
    sample = sample.drop(columns=["_binary_label"]).sample(frac=1.0, random_state=args.random_state)
    sample.to_parquet(args.output_path, index=False)

    print(f"Available rows: {sum(label_counts.values()):,}")
    print(f"Label counts: {label_counts}")
    print(f"Sample targets: {targets}")
    print(f"Wrote {len(sample):,} rows to {args.output_path}")


if __name__ == "__main__":
    main()
