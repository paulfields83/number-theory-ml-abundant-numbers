from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


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

RESEARCH_FEATURE_SETS = {
    "full_without_sigma_ratio": [
        "log_n",
        "tau_n",
        "omega_n",
        "Omega_n",
        "min_prime_factor",
        "max_prime_factor",
    ],
    "no_sigma_no_tau": [
        "log_n",
        "omega_n",
        "Omega_n",
        "min_prime_factor",
        "max_prime_factor",
    ],
    "prime_structure_only": [
        "omega_n",
        "Omega_n",
        "min_prime_factor",
        "max_prime_factor",
    ],
    "size_only": ["log_n"],
    "omega_only": ["omega_n"],
}


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


def binary_target(labels: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(labels.astype(str) == "abundant", "abundant", "non_abundant"),
        index=labels.index,
        name="binary_label",
    )


def binary_metric_row(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    prefix: dict[str, object] | None = None,
) -> dict[str, object]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        pos_label="abundant",
        average="binary",
        zero_division=0,
    )
    weighted = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=["non_abundant", "abundant"])
    row: dict[str, object] = {
        "rows": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_abundant": precision,
        "recall_abundant": recall,
        "f1_abundant": f1,
        "f1_weighted": weighted[2],
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
    if prefix:
        return {**prefix, **row}
    return row


def safe_stratify_target(y: pd.Series) -> pd.Series | None:
    counts = y.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return None
    return y
