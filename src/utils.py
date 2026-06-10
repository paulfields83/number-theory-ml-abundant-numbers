from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import pandas as pd


LABEL_ORDER = ["deficient", "perfect", "abundant"]

BASE_FEATURE_COLUMNS = [
    "log_n",
    "tau_n",
    "omega_n",
    "Omega_n",
    "min_prime_factor",
    "max_prime_factor",
]

WITH_SIGMA_RATIO_COLUMNS = BASE_FEATURE_COLUMNS + ["sigma_ratio"]


def ensure_directories(*paths: Path | str) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def find_feature_files(data_dir: Path | str) -> list[Path]:
    data_path = Path(data_dir)
    files = sorted(data_path.glob("features_*.parquet"))
    if not files:
        files = sorted(data_path.glob("features_*.csv"))
    if not files:
        raise FileNotFoundError(f"No feature chunks found under {data_path}")
    return files


def read_feature_chunk(path: Path | str, columns: Iterable[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=list(columns) if columns is not None else None)
    if path.suffix == ".csv":
        return pd.read_csv(path, usecols=list(columns) if columns is not None else None)
    raise ValueError(f"Unsupported chunk format: {path}")


def slugify(*parts: object) -> str:
    raw = "_".join(str(part) for part in parts)
    raw = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").lower()
    return raw or "item"
