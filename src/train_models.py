from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import (
    BASE_FEATURE_COLUMNS,
    LABEL_ORDER,
    WITH_SIGMA_RATIO_COLUMNS,
    ensure_directories,
    find_feature_files,
    read_feature_chunk,
    slugify,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sampled classifiers on generated chunks.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--rf-estimators", type=int, default=120)
    parser.add_argument("--hgb-iterations", type=int, default=120)
    return parser.parse_args()


def load_training_sample(data_dir: Path, sample_size: int, random_state: int) -> pd.DataFrame:
    if sample_size < 1:
        raise ValueError("--sample-size must be positive")

    files = find_feature_files(data_dir)
    columns = sorted(set(["n", "label", *WITH_SIGMA_RATIO_COLUMNS]))
    per_file_quota = max(1, int(np.ceil(sample_size / len(files))))
    perfect_parts = []
    sample_parts = []

    for file_index, path in enumerate(files):
        frame = read_feature_chunk(path, columns=columns)
        label_text = frame["label"].astype(str)
        perfect = frame[label_text == "perfect"]
        non_perfect = frame[label_text != "perfect"]
        if not perfect.empty:
            perfect_parts.append(perfect)

        if len(non_perfect) <= per_file_quota:
            sample_parts.append(non_perfect)
        else:
            sample_parts.append(
                non_perfect.sample(
                    n=per_file_quota,
                    random_state=random_state + file_index,
                    replace=False,
                )
            )

    pieces = perfect_parts + sample_parts
    if not pieces:
        raise ValueError("No training rows were loaded")

    sample = pd.concat(pieces, ignore_index=True).drop_duplicates(subset=["n"])
    if len(sample) > sample_size:
        perfect_sample = sample[sample["label"].astype(str) == "perfect"]
        other_sample = sample[sample["label"].astype(str) != "perfect"]
        keep_other = max(0, sample_size - len(perfect_sample))
        if len(other_sample) > keep_other:
            other_sample = other_sample.sample(n=keep_other, random_state=random_state)
        sample = pd.concat([perfect_sample, other_sample], ignore_index=True)

    return sample.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def build_models(random_state: int, rf_estimators: int, hgb_iterations: int):
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1_000,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=rf_estimators,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=hgb_iterations,
            learning_rate=0.08,
            random_state=random_state,
        ),
    }


def make_split(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int):
    counts = y.value_counts()
    stratify = y if len(counts) > 1 and counts.min() >= 2 else None
    if stratify is None:
        warnings.warn("Class counts are too small for a stratified split; using an unstratified split.")
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
    task: str,
    feature_set: str,
    model_name: str,
    figures_dir: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{task} / {model_name}")
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figures_dir / f"confusion_matrix_{slugify(task, feature_set, model_name)}.png", dpi=160)
    plt.close(fig)


def compute_importance(model, X_eval: pd.DataFrame, y_eval: pd.Series, random_state: int) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)

    if hasattr(model, "named_steps") and "logisticregression" in model.named_steps:
        classifier = model.named_steps["logisticregression"]
        coefficients = np.asarray(classifier.coef_, dtype=float)
        return np.mean(np.abs(coefficients), axis=0)

    max_eval = min(3_000, len(X_eval))
    eval_X = X_eval.sample(n=max_eval, random_state=random_state) if len(X_eval) > max_eval else X_eval
    eval_y = y_eval.loc[eval_X.index]
    result = permutation_importance(
        model,
        eval_X,
        eval_y,
        n_repeats=3,
        random_state=random_state,
        n_jobs=-1,
    )
    return np.asarray(result.importances_mean, dtype=float)


def save_feature_importance(
    model,
    feature_names: list[str],
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    task: str,
    feature_set: str,
    model_name: str,
    figures_dir: Path,
    random_state: int,
) -> None:
    importances = compute_importance(model, X_eval, y_eval, random_state)
    order = np.argsort(importances)
    fig_height = max(3.5, 0.42 * len(feature_names))
    fig, ax = plt.subplots(figsize=(7, fig_height))
    ax.barh(np.asarray(feature_names)[order], importances[order], color="#59A14F")
    ax.set_title(f"Feature importance: {task} / {model_name}")
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(figures_dir / f"feature_importance_{slugify(task, feature_set, model_name)}.png", dpi=160)
    plt.close(fig)


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    labels: list[str],
    task: str,
    feature_set: str,
    model_name: str,
    args: argparse.Namespace,
) -> dict[str, object] | None:
    if y_train.nunique() < 2:
        warnings.warn(f"Skipping {task}/{model_name}: training split has fewer than two classes.")
        return None

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    weighted = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    macro = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, labels=labels, zero_division=0)

    report_path = args.results_dir / f"classification_report_{slugify(task, feature_set, model_name)}.txt"
    report_path.write_text(report, encoding="utf-8")

    save_confusion_matrix(y_test, y_pred, labels, task, feature_set, model_name, args.figures_dir)
    save_feature_importance(
        model,
        list(X_train.columns),
        X_test,
        y_test,
        task,
        feature_set,
        model_name,
        args.figures_dir,
        args.random_state,
    )

    return {
        "task": task,
        "feature_set": feature_set,
        "model": model_name,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_weighted": weighted[0],
        "recall_weighted": weighted[1],
        "f1_weighted": weighted[2],
        "precision_macro": macro[0],
        "recall_macro": macro[1],
        "f1_macro": macro[2],
    }


def run_task(
    sample: pd.DataFrame,
    y: pd.Series,
    labels: list[str],
    task: str,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    metrics = []
    feature_sets = {
        "with_sigma_ratio": WITH_SIGMA_RATIO_COLUMNS,
        "without_sigma_ratio": BASE_FEATURE_COLUMNS,
    }

    for feature_set, feature_columns in feature_sets.items():
        X = sample[feature_columns].copy()
        X_train, X_test, y_train, y_test = make_split(
            X, y, test_size=args.test_size, random_state=args.random_state
        )
        models = build_models(args.random_state, args.rf_estimators, args.hgb_iterations)
        for model_name, model in models.items():
            result = evaluate_model(
                model,
                X_train,
                X_test,
                y_train,
                y_test,
                labels,
                task,
                feature_set,
                model_name,
                args,
            )
            if result is not None:
                metrics.append(result)

    return metrics


def main() -> None:
    args = parse_args()
    ensure_directories(args.figures_dir, args.results_dir)
    sample = load_training_sample(args.data_dir, args.sample_size, args.random_state)
    print(f"Loaded training sample: {len(sample):,} rows")
    print(sample["label"].astype(str).value_counts().to_string())

    all_metrics: list[dict[str, object]] = []
    multi_y = sample["label"].astype(str)
    present_labels = [label for label in LABEL_ORDER if label in set(multi_y)]
    all_metrics.extend(run_task(sample, multi_y, present_labels, "multi_class", args))

    binary_y = np.where(sample["label"].astype(str) == "abundant", "abundant", "non_abundant")
    binary_y = pd.Series(binary_y, index=sample.index, name="binary_label")
    all_metrics.extend(run_task(sample, binary_y, ["non_abundant", "abundant"], "binary", args))

    metrics_frame = pd.DataFrame(all_metrics)
    metrics_frame.to_csv(args.results_dir / "metrics.csv", index=False)
    print(f"Wrote metrics for {len(metrics_frame)} model runs to {args.results_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
