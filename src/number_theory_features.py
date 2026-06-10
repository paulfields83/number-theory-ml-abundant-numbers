from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from utils import LABEL_ORDER

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - used only when numba is unavailable
    NUMBA_AVAILABLE = False
    njit = None


def estimate_spf_memory_mb(limit: int) -> float:
    return (limit + 1) * np.dtype(np.int32).itemsize / (1024**2)


def estimate_chunk_memory_mb(chunk_size: int) -> float:
    bytes_per_row = 8 + 4 + 8 + 8 + 4 + 1 + 1 + 4 + 4 + 8
    return chunk_size * bytes_per_row / (1024**2)


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _prime_capacity(limit: int) -> int:
        if limit < 64:
            return limit + 1
        return int(1.5 * limit / math.log(limit)) + 4096

    @njit(cache=True)
    def _build_spf_numba(limit: int) -> np.ndarray:
        spf = np.zeros(limit + 1, dtype=np.int32)
        if limit >= 1:
            spf[1] = 1

        primes = np.empty(_prime_capacity(limit), dtype=np.int32)
        prime_count = 0

        for i in range(2, limit + 1):
            if spf[i] == 0:
                spf[i] = i
                if prime_count >= primes.size:
                    bigger = np.empty(primes.size * 2, dtype=np.int32)
                    for k in range(primes.size):
                        bigger[k] = primes[k]
                    primes = bigger
                primes[prime_count] = i
                prime_count += 1

            j = 0
            while j < prime_count:
                p = primes[j]
                x = p * i
                if x > limit or p > spf[i]:
                    break
                spf[x] = p
                j += 1

        return spf

    @njit(cache=True)
    def _factor_range_numba(start: int, end: int, spf: np.ndarray):
        count = end - start + 1
        n_values = np.empty(count, dtype=np.int64)
        sigma_values = np.empty(count, dtype=np.int64)
        tau_values = np.empty(count, dtype=np.uint32)
        omega_values = np.empty(count, dtype=np.uint8)
        big_omega_values = np.empty(count, dtype=np.uint8)
        min_prime_values = np.empty(count, dtype=np.int32)
        max_prime_values = np.empty(count, dtype=np.int32)
        label_codes = np.empty(count, dtype=np.uint8)

        for offset in range(count):
            n = start + offset
            x = n
            sigma = 1
            tau = 1
            omega = 0
            big_omega = 0
            min_prime = 0
            max_prime = 0

            while x > 1:
                p = spf[x]
                if min_prime == 0:
                    min_prime = p
                max_prime = p

                exponent = 0
                p_power = 1
                geometric_sum = 1
                while x % p == 0:
                    x //= p
                    exponent += 1
                    p_power *= p
                    geometric_sum += p_power

                sigma *= geometric_sum
                tau *= exponent + 1
                omega += 1
                big_omega += exponent

            if sigma > 2 * n:
                label_code = 2
            elif sigma == 2 * n:
                label_code = 1
            else:
                label_code = 0

            n_values[offset] = n
            sigma_values[offset] = sigma
            tau_values[offset] = tau
            omega_values[offset] = omega
            big_omega_values[offset] = big_omega
            min_prime_values[offset] = min_prime
            max_prime_values[offset] = max_prime
            label_codes[offset] = label_code

        return (
            n_values,
            sigma_values,
            tau_values,
            omega_values,
            big_omega_values,
            min_prime_values,
            max_prime_values,
            label_codes,
        )


def build_spf(limit: int) -> np.ndarray:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if NUMBA_AVAILABLE:
        return _build_spf_numba(limit)
    return _build_spf_python(limit)


def _build_spf_python(limit: int) -> np.ndarray:
    spf = np.zeros(limit + 1, dtype=np.int32)
    if limit >= 1:
        spf[1] = 1

    root = int(math.isqrt(limit))
    for p in range(2, root + 1):
        if spf[p] != 0:
            continue
        spf[p] = p
        multiples = np.arange(p * p, limit + 1, p, dtype=np.int64)
        unset = spf[multiples] == 0
        spf[multiples[unset]] = p

    missing = spf == 0
    missing[0] = False
    spf[missing] = np.nonzero(missing)[0].astype(np.int32)
    return spf


def _factor_one_python(n: int, spf: np.ndarray) -> tuple[int, int, int, int, int, int, int]:
    x = n
    sigma = 1
    tau = 1
    omega = 0
    big_omega = 0
    min_prime = 0
    max_prime = 0

    while x > 1:
        p = int(spf[x])
        if min_prime == 0:
            min_prime = p
        max_prime = p

        exponent = 0
        p_power = 1
        geometric_sum = 1
        while x % p == 0:
            x //= p
            exponent += 1
            p_power *= p
            geometric_sum += p_power

        sigma *= geometric_sum
        tau *= exponent + 1
        omega += 1
        big_omega += exponent

    if sigma > 2 * n:
        label_code = 2
    elif sigma == 2 * n:
        label_code = 1
    else:
        label_code = 0

    return sigma, tau, omega, big_omega, min_prime, max_prime, label_code


def _factor_range_python(start: int, end: int, spf: np.ndarray):
    count = end - start + 1
    n_values = np.arange(start, end + 1, dtype=np.int64)
    sigma_values = np.empty(count, dtype=np.int64)
    tau_values = np.empty(count, dtype=np.uint32)
    omega_values = np.empty(count, dtype=np.uint8)
    big_omega_values = np.empty(count, dtype=np.uint8)
    min_prime_values = np.empty(count, dtype=np.int32)
    max_prime_values = np.empty(count, dtype=np.int32)
    label_codes = np.empty(count, dtype=np.uint8)

    for offset, n in enumerate(n_values):
        (
            sigma_values[offset],
            tau_values[offset],
            omega_values[offset],
            big_omega_values[offset],
            min_prime_values[offset],
            max_prime_values[offset],
            label_codes[offset],
        ) = _factor_one_python(int(n), spf)

    return (
        n_values,
        sigma_values,
        tau_values,
        omega_values,
        big_omega_values,
        min_prime_values,
        max_prime_values,
        label_codes,
    )


def build_feature_frame(
    start: int,
    end: int,
    spf: np.ndarray,
    external_omega: np.ndarray | None = None,
) -> pd.DataFrame:
    if start < 1 or end < start:
        raise ValueError("Use a valid positive range with start <= end")

    if NUMBA_AVAILABLE:
        arrays = _factor_range_numba(start, end, spf)
    else:
        arrays = _factor_range_python(start, end, spf)

    (
        n_values,
        sigma_values,
        tau_values,
        omega_values,
        big_omega_values,
        min_prime_values,
        max_prime_values,
        label_codes,
    ) = arrays

    if external_omega is not None:
        if len(external_omega) != len(omega_values):
            raise ValueError("external_omega length does not match the requested chunk")
        omega_values = external_omega.astype(np.uint8, copy=False)

    labels = np.asarray(LABEL_ORDER, dtype=object)[label_codes]
    frame = pd.DataFrame(
        {
            "n": n_values,
            "log_n": np.log(n_values.astype(np.float64)).astype(np.float32),
            "sigma_n": sigma_values,
            "sigma_ratio": sigma_values.astype(np.float64) / n_values,
            "tau_n": tau_values,
            "omega_n": omega_values,
            "Omega_n": big_omega_values,
            "min_prime_factor": min_prime_values,
            "max_prime_factor": max_prime_values,
            "label": pd.Categorical(labels, categories=LABEL_ORDER),
        }
    )
    return frame


def load_external_omega_chunk(data_dir: Path | str, start: int, end: int) -> np.ndarray | None:
    data_path = Path(data_dir)
    parquet_path = data_path / "omega_values.parquet"
    csv_path = data_path / "omega_values.csv"
    needed_index = pd.RangeIndex(start, end + 1, name="n")

    if parquet_path.exists():
        try:
            omega_frame = pd.read_parquet(
                parquet_path,
                columns=["n", "omega_n"],
                filters=[("n", ">=", start), ("n", "<=", end)],
            )
        except TypeError:
            omega_frame = pd.read_parquet(parquet_path, columns=["n", "omega_n"])
            omega_frame = omega_frame[(omega_frame["n"] >= start) & (omega_frame["n"] <= end)]
        return _align_external_omega(omega_frame, needed_index)

    if csv_path.exists():
        pieces = []
        for chunk in pd.read_csv(csv_path, usecols=["n", "omega_n"], chunksize=1_000_000):
            piece = chunk[(chunk["n"] >= start) & (chunk["n"] <= end)]
            if not piece.empty:
                pieces.append(piece)
        if pieces:
            return _align_external_omega(pd.concat(pieces, ignore_index=True), needed_index)

    return None


def _align_external_omega(omega_frame: pd.DataFrame, needed_index: pd.RangeIndex) -> np.ndarray | None:
    if omega_frame.empty:
        return None
    aligned = omega_frame.drop_duplicates("n").set_index("n").reindex(needed_index)["omega_n"]
    if aligned.isna().any():
        return None
    return aligned.to_numpy(dtype=np.uint8)
