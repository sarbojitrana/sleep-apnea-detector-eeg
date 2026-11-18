"""EEG feature extraction and window-level apnea SVM utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis, skew
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from pywt import wavedec

EEG_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
STATISTICS = ("energy", "mobility", "kurtosis", "skewness", "iqr", "mad")
FEATURE_NAMES = tuple(
    f"{band}_{statistic}"
    for band in EEG_BANDS
    for statistic in STATISTICS
)
PAPER_FEATURES = (
    "beta_energy",
    "delta_kurtosis",
    "delta_mobility",
    "beta_mobility",
)


def extract_features(samples: Sequence[float] | np.ndarray, sampling_rate: float) -> dict[str, float]:
    """Extract six statistical features from each of five EEG frequency bands."""
    signal = np.asarray(samples, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("An EEG window must contain one channel of samples.")
    if not np.isfinite(signal).all():
        raise ValueError("EEG samples must all be finite numbers.")
    if not np.isfinite(sampling_rate) or sampling_rate <= 90:
        raise ValueError("Sampling rate must be greater than 90 Hz for the configured bands.")
    if len(signal) < max(64, int(np.ceil(sampling_rate * 1.5))):
        raise ValueError("EEG window is too short; provide at least 1.5 seconds of samples.")
    if np.std(signal) <= np.finfo(np.float64).eps:
        raise ValueError("EEG window is constant and does not contain a usable signal.")

    features: dict[str, float] = {}
    for band, (low, high) in EEG_BANDS.items():
        sos = butter(
            4,
            (low, high),
            btype="bandpass",
            fs=sampling_rate,
            output="sos",
        )
        filtered = sosfiltfilt(sos, signal)
        coefficient_blocks = wavedec(filtered, "db4", mode="symmetric", level=3)
        coefficients = np.concatenate(coefficient_blocks)
        block_mobilities = [
            np.sqrt(np.var(np.diff(block)) / np.var(block))
            for block in coefficient_blocks
            if len(block) > 1 and np.var(block) > np.finfo(np.float64).eps
        ]
        values = {
            "energy": float(np.sum(np.square(coefficients))),
            "mobility": float(np.mean(block_mobilities)) if block_mobilities else 0.0,
            "kurtosis": float(kurtosis(coefficients, fisher=True, bias=False)),
            "skewness": float(skew(coefficients, bias=False)),
            "iqr": float(np.percentile(coefficients, 75) - np.percentile(coefficients, 25)),
            "mad": float(np.median(np.abs(coefficients - np.median(coefficients)))),
        }
        for statistic, value in values.items():
            if not np.isfinite(value):
                raise ValueError(f"Could not calculate {band} {statistic} for this EEG window.")
            features[f"{band}_{statistic}"] = value
    return features


def _sample_columns(frame: pd.DataFrame) -> list[str]:
    indexed_columns: list[tuple[int, str]] = []
    for column in frame.columns:
        if column.startswith("sample_"):
            suffix = column.removeprefix("sample_")
            if suffix.isdigit():
                indexed_columns.append((int(suffix), column))
    indexed_columns.sort()
    if not indexed_columns or [index for index, _ in indexed_columns] != list(
        range(len(indexed_columns))
    ):
        raise ValueError(
            "Training CSV needs contiguous sample columns named sample_0, sample_1, ..."
        )
    return [column for _, column in indexed_columns]


def prepare_training_data(
    frame: pd.DataFrame,
    sampling_rate: float | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    """Validate windowed training CSV and calculate features for every row."""
    if "label" not in frame.columns:
        raise ValueError("Training CSV must include a 'label' column.")
    sample_columns = _sample_columns(frame)
    if frame.empty:
        raise ValueError("Training CSV does not contain any EEG windows.")

    if "sampling_rate" in frame.columns:
        rates = pd.to_numeric(frame["sampling_rate"], errors="coerce").to_numpy()
        if not np.isfinite(rates).all() or np.any(rates != rates[0]):
            raise ValueError("Use one finite sampling_rate value consistently across the CSV.")
        data_rate = float(rates[0])
        if sampling_rate is not None and not np.isclose(data_rate, sampling_rate):
            raise ValueError("The sampling rate argument conflicts with the CSV sampling_rate.")
    elif sampling_rate is not None:
        data_rate = float(sampling_rate)
    else:
        raise ValueError("Add a 'sampling_rate' column or provide a sampling rate.")

    sample_values = frame[sample_columns].apply(pd.to_numeric, errors="coerce").to_numpy()
    if not np.isfinite(sample_values).all():
        raise ValueError("Training CSV sample columns must contain only finite numbers.")
    if frame["label"].isna().any():
        raise ValueError("Training CSV labels cannot be empty.")
    labels = frame["label"].astype(str).str.strip().to_numpy()
    if any(not label for label in labels):
        raise ValueError("Training CSV labels cannot be empty.")

    feature_rows = [
        extract_features(row, data_rate)
        for row in sample_values
    ]
    features = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
    if "subject_id" in frame.columns:
        if frame["subject_id"].isna().any():
            raise ValueError("Training CSV subject_id values cannot be empty.")
        groups = frame["subject_id"].astype(str).to_numpy()
        if any(not group.strip() for group in groups):
            raise ValueError("Training CSV subject_id values cannot be empty.")
    else:
        groups = None
    return features, labels, groups


def _holdout_indices(
    labels: np.ndarray,
    groups: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    if len(np.unique(labels)) < 2:
        raise ValueError("Training requires at least two distinct apnea labels.")
    counts = pd.Series(labels).value_counts()
    if (counts < 2).any():
        raise ValueError("Each label needs at least two EEG windows for held-out evaluation.")

    if groups is None:
        try:
            train_indices, test_indices = train_test_split(
                np.arange(len(labels)),
                test_size=0.25,
                random_state=42,
                stratify=labels,
            )
        except ValueError as exc:
            raise ValueError(f"Could not create a stratified holdout: {exc}") from exc
        return train_indices, test_indices, "Stratified window holdout (subject IDs not provided)"

    if len(np.unique(groups)) < 4:
        raise ValueError(
            "Subject-wise evaluation needs at least four distinct subject_id values."
        )
    splitter = GroupShuffleSplit(n_splits=50, test_size=0.25, random_state=42)
    for train_indices, test_indices in splitter.split(np.zeros(len(labels)), labels, groups):
        if (
            len(np.unique(labels[train_indices])) == len(np.unique(labels))
            and len(np.unique(labels[test_indices])) == len(np.unique(labels))
        ):
            return train_indices, test_indices, "Subject-wise holdout (GroupShuffleSplit)"
    raise ValueError(
        "Could not make a subject-wise holdout containing every label in both sets. "
        "Add subjects with examples from each label."
    )


def train_svm(
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray | None,
    selected_features: Sequence[str] = FEATURE_NAMES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate by a held-out split, then fit a final SVM on all supplied windows."""
    feature_names = tuple(selected_features)
    if not feature_names or any(name not in features.columns for name in feature_names):
        raise ValueError("Select one or more valid EEG features.")
    if (
        len(features) != len(labels)
        or (groups is not None and len(groups) != len(labels))
        or len(labels) == 0
    ):
        raise ValueError("Training requires matching, non-empty feature and label rows.")
    if not np.isfinite(features.loc[:, feature_names].to_numpy()).all():
        raise ValueError("Feature table contains non-finite values.")

    train_indices, test_indices, split_method = _holdout_indices(labels, groups)
    estimator = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", random_state=42),
    )
    x = features.loc[:, feature_names].to_numpy()
    y = np.asarray(labels)
    estimator.fit(x[train_indices], y[train_indices])
    predictions = estimator.predict(x[test_indices])
    class_names = sorted(np.unique(y).tolist())
    report: dict[str, Any] = {
        "split_method": split_method,
        "train_windows": int(len(train_indices)),
        "test_windows": int(len(test_indices)),
        "classes": class_names,
        "features": list(feature_names),
        "accuracy": float(accuracy_score(y[test_indices], predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y[test_indices], predictions)),
        "precision_macro": float(
            precision_score(y[test_indices], predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y[test_indices], predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y[test_indices], predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y[test_indices], predictions, labels=class_names
        ).tolist(),
    }

    estimator.fit(x, y)
    return {"estimator": estimator, "features": feature_names, "classes": class_names}, report


def split_signal(
    samples: Sequence[float] | np.ndarray,
    sampling_rate: float,
    window_seconds: float,
    overlap_seconds: float,
) -> list[np.ndarray]:
    """Split a continuous EEG channel into complete, optionally overlapping windows."""
    signal = np.asarray(samples, dtype=np.float64)
    if signal.ndim != 1 or not np.isfinite(signal).all():
        raise ValueError("EEG input must be a one-dimensional array of finite samples.")
    if not np.isfinite(sampling_rate) or sampling_rate <= 90:
        raise ValueError("Sampling rate must be greater than 90 Hz.")
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("Window duration must be a positive number of seconds.")
    if (
        not np.isfinite(overlap_seconds)
        or overlap_seconds < 0
        or overlap_seconds >= window_seconds
    ):
        raise ValueError("Overlap must be non-negative and shorter than the window.")

    window_size = int(round(window_seconds * sampling_rate))
    step_size = int(round((window_seconds - overlap_seconds) * sampling_rate))
    if window_size < max(64, int(np.ceil(sampling_rate * 1.5))):
        raise ValueError("Window duration must be at least 1.5 seconds.")
    if step_size < 1:
        raise ValueError("Window overlap is too close to the window duration.")
    if len(signal) < window_size:
        raise ValueError(
            f"Signal contains {len(signal)} samples; {window_size} are needed for one window."
        )
    return [
        signal[start : start + window_size]
        for start in range(0, len(signal) - window_size + 1, step_size)
    ]


def predict_signal(
    model: dict[str, Any],
    samples: Sequence[float] | np.ndarray,
    sampling_rate: float,
    window_seconds: float,
    overlap_seconds: float = 0.0,
) -> pd.DataFrame:
    windows = split_signal(samples, sampling_rate, window_seconds, overlap_seconds)
    feature_rows = [extract_features(window, sampling_rate) for window in windows]
    feature_frame = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
    selected = feature_frame.loc[:, model["features"]]
    estimator = model["estimator"]
    predictions = estimator.predict(selected.to_numpy())
    decision_scores = np.asarray(estimator.decision_function(selected.to_numpy()))
    if len(estimator.classes_) == 2:
        decision_scores = np.column_stack((-decision_scores, decision_scores))
    result: dict[str, Any] = {
        "window": np.arange(1, len(windows) + 1),
        "start_seconds": np.arange(len(windows)) * (window_seconds - overlap_seconds),
        "prediction": predictions,
    }
    for index, label in enumerate(estimator.classes_):
        result[f"decision_score_{label}"] = decision_scores[:, index]
    return pd.DataFrame(result)
