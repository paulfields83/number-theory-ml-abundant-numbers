from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from number_theory_features import (
    NUMBA_AVAILABLE,
    build_feature_frame,
    build_spf,
    estimate_chunk_memory_mb,
    estimate_spf_memory_mb,
    load_external_omega_chunk,
)
from utils import ensure_directories

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable

    def tqdm(iterable, **_: object):
        return iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate chunked number-theory feature datasets."
    )
    parser.add_argument("--max-n", type=int, required=True, help="Generate features for 1..max_n.")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing chunk files.")
    parser.add_argument(
        "--no-external-omega",
        action="store_true",
        help="Ignore data/omega_values.csv or data/omega_values.parquet when present.",
    )
    return parser.parse_args()


def chunk_ranges(max_n: int, chunk_size: int):
    for start in range(1, max_n + 1, chunk_size):
        end = min(start + chunk_size - 1, max_n)
        yield start, end


def parse_existing_chunk_ranges(output_dir: Path) -> list[tuple[int, int, Path]]:
    pattern = re.compile(r"features_(\d+)_(\d+)\.(parquet|csv)$")
    ranges = []
    for path in sorted(output_dir.glob("features_*_*.*")):
        match = pattern.match(path.name)
        if match:
            ranges.append((int(match.group(1)), int(match.group(2)), path))
    return ranges


def range_fully_covered(start: int, end: int, ranges: list[tuple[int, int, Path]]) -> bool:
    cursor = start
    for existing_start, existing_end, _ in sorted(ranges):
        if existing_end < cursor:
            continue
        if existing_start > cursor:
            return False
        cursor = max(cursor, existing_end + 1)
        if cursor > end:
            return True
    return False


def range_overlaps(start: int, end: int, ranges: list[tuple[int, int, Path]]) -> list[Path]:
    return [
        path
        for existing_start, existing_end, path in ranges
        if existing_start <= end and start <= existing_end
    ]


def estimate_parquet_disk_mb(row_count: int) -> float:
    return row_count * 90 / (1024**2)


def main() -> None:
    args = parse_args()
    if args.max_n < 1:
        raise ValueError("--max-n must be positive")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")

    ensure_directories(args.output_dir, args.data_dir)
    print(f"Generating features for n=1..{args.max_n:,}")
    print(f"Chunk size: {args.chunk_size:,}")
    print(f"Numba acceleration: {'enabled' if NUMBA_AVAILABLE else 'not available'}")
    print(f"Estimated SPF memory: {estimate_spf_memory_mb(args.max_n):,.1f} MiB")
    print(f"Estimated per-chunk array memory: {estimate_chunk_memory_mb(args.chunk_size):,.1f} MiB")
    print(f"Estimated Parquet disk usage: {estimate_parquet_disk_mb(args.max_n):,.1f} MiB")
    if not NUMBA_AVAILABLE and args.max_n > 1_000_000:
        print("Warning: large runs are intended for numba; install requirements first.")

    existing_ranges = parse_existing_chunk_ranges(args.output_dir)
    if existing_ranges:
        covered_rows = sum(end - start + 1 for start, end, _ in existing_ranges)
        print(f"Found {len(existing_ranges):,} existing chunk file(s), covering about {covered_rows:,} rows.")

    started = time.perf_counter()
    spf = build_spf(args.max_n)
    print(f"SPF sieve built in {time.perf_counter() - started:,.2f}s")

    ranges = list(chunk_ranges(args.max_n, args.chunk_size))
    missing_ranges = [
        (start, end)
        for start, end in ranges
        if args.overwrite or not range_fully_covered(start, end, existing_ranges)
    ]
    print(f"Desired chunks: {len(ranges):,}; chunks needing generation: {len(missing_ranges):,}")

    for start, end in tqdm(ranges, desc="chunks"):
        output_path = args.output_dir / f"features_{start:08d}_{end:08d}.parquet"
        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing chunk: {output_path}")
            continue
        if not args.overwrite and range_fully_covered(start, end, existing_ranges):
            print(f"Skipping covered range {start:,}-{end:,}; an existing chunk already contains it.")
            continue
        overlaps = range_overlaps(start, end, existing_ranges)
        if overlaps and not args.overwrite:
            overlap_names = ", ".join(path.name for path in overlaps[:3])
            raise RuntimeError(
                f"Refusing to create partially overlapping chunk {start:,}-{end:,}. "
                f"Overlaps include: {overlap_names}. Use a consistent chunk size or clean data/chunks."
            )

        external_omega = None
        if not args.no_external_omega:
            external_omega = load_external_omega_chunk(args.data_dir, start, end)
            if external_omega is not None:
                print(f"Using external omega_n values for {start:,}-{end:,}")

        chunk_started = time.perf_counter()
        frame = build_feature_frame(start, end, spf, external_omega=external_omega)
        frame.to_parquet(output_path, index=False)
        elapsed = time.perf_counter() - chunk_started
        print(f"Wrote {output_path} ({len(frame):,} rows) in {elapsed:,.2f}s")

    print(f"Done in {time.perf_counter() - started:,.2f}s")


if __name__ == "__main__":
    main()
