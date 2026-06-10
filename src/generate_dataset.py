from __future__ import annotations

import argparse
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
    if not NUMBA_AVAILABLE and args.max_n > 1_000_000:
        print("Warning: large runs are intended for numba; install requirements first.")

    started = time.perf_counter()
    spf = build_spf(args.max_n)
    print(f"SPF sieve built in {time.perf_counter() - started:,.2f}s")

    ranges = list(chunk_ranges(args.max_n, args.chunk_size))
    for start, end in tqdm(ranges, desc="chunks"):
        output_path = args.output_dir / f"features_{start:08d}_{end:08d}.parquet"
        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing chunk: {output_path}")
            continue

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
