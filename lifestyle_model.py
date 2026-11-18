"""Inference and training for the linked six-input lifestyle screening model."""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "lifestyle_svm.json"
DATASET_URL = (
    "https://github.com/sarbojitrana/sleep-apnea-detector/blob/main/"
    "data/Sleep_health_and_lifestyle_dataset.csv"
)
SCHEMA_VERSION = 1
FEATURES = (
    "Gender",
    "Age",
    "Sleep Duration",
    "BMI Category",
    "Heart Rate",
    "Daily Steps",
)
INPUT_COLUMNS = {
    "gender": "Gender",
    "age": "Age",
    "sleep_duration": "Sleep Duration",
    "bmi": "BMI Category",
    "heart_rate": "Heart Rate",
    "daily_steps": "Daily Steps",
}
BMI_MAP = {"normal": 0, "normal weight": 0, "overweight": 1, "obese": 2}
BMI_LABELS = {0: "Normal", 1: "Overweight", 2: "Obese"}
GENDER_MAP = {"female": 0, "male": 1}
NUMERIC_LIMITS = {
    "age": (18, 100),
    "sleep_duration": (0.5, 16),
    "heart_rate": (30, 220),
    "daily_steps": (0, 100000),
}
SYMPTOMS = ("loud_snoring", "witnessed_pauses", "daytime_sleepiness")
PARAM_GRID = [
    {"svc__kernel": ["linear"], "svc__C": [0.01, 0.1, 1, 10]},
    {
        "svc__kernel": ["rbf"],
        "svc__C": [0.1, 1, 10],
        "svc__gamma": ["scale", 0.01, 0.1],
    },
]


def _encode(value: Any, mapping: dict[str, int], field: str) -> int:
    if not isinstance(value, str) or value.strip().lower() not in mapping:
        raise ValueError(f"{field} must be one of: {', '.join(mapping)}")
    return mapping[value.strip().lower()]


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("Expected a mapping containing the six required inputs.")
    missing = set(INPUT_COLUMNS) - record.keys()
    unknown = record.keys() - set(INPUT_COLUMNS) - set(SYMPTOMS)
    if missing:
        raise ValueError(f"Missing required inputs: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"Unknown inputs: {', '.join(sorted(unknown))}")

    clean = dict(record)
    _encode(clean["gender"], GENDER_MAP, "gender")
    clean["gender"] = clean["gender"].strip().title()
    clean["bmi"] = BMI_LABELS[_encode(clean["bmi"], BMI_MAP, "bmi")]
    for name, (low, high) in NUMERIC_LIMITS.items():
        value = clean[name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite number.")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or not low <= value <= high:
            raise ValueError(f"{name} must be a finite number between {low} and {high}.")
        if name in ("age", "daily_steps") and value != int(value):
            raise ValueError(f"{name} must be a whole number.")
        clean[name] = float(value)
    for name in SYMPTOMS:
        if name in clean and not isinstance(clean[name], bool):
            raise ValueError(f"{name} must be true or false.")
    return clean


def _feature_row(record: dict[str, Any]) -> np.ndarray:
    clean = validate_record(record)
    return np.asarray(
        [
            GENDER_MAP[clean["gender"].lower()],
            clean["age"],
            clean["sleep_duration"],
            BMI_MAP[clean["bmi"].lower()],
            clean["heart_rate"],
            clean["daily_steps"],
        ],
        dtype=np.float64,
    )


def build_estimator() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(class_weight="balanced", random_state=42)),
        ]
    )


def _profile_groups(features: np.ndarray) -> np.ndarray:
    return pd.factorize(pd.MultiIndex.from_arrays(features.T))[0]


def train_and_save(dataset_path: str | Path, model_path: str | Path = MODEL_PATH) -> dict[str, Any]:
    """Train the source project's grouped-CV-tuned SVM and export a portable JSON bundle."""
    data_path = Path(dataset_path)
    frame = pd.read_csv(data_path, keep_default_na=False)
    required = set(FEATURES) | {"Sleep Disorder"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")
    if frame.empty or not frame["Sleep Disorder"].isin(
        ["None", "Insomnia", "Sleep Apnea"]
    ).all():
        raise ValueError("Dataset must have non-empty labels: None, Insomnia, or Sleep Apnea.")

    rows = []
    for row in frame.to_dict("records"):
        rows.append(
            _feature_row(
                {
                    "gender": row["Gender"],
                    "age": float(row["Age"]),
                    "sleep_duration": float(row["Sleep Duration"]),
                    "bmi": row["BMI Category"],
                    "heart_rate": float(row["Heart Rate"]),
                    "daily_steps": float(row["Daily Steps"]),
                }
            )
        )
    x = np.vstack(rows)
    y = (frame["Sleep Disorder"] == "Sleep Apnea").astype(int).to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError("Dataset needs both apnea-labelled and other-labelled rows.")
    groups = _profile_groups(x)
    unique_groups = len(np.unique(groups))
    if unique_groups < 3:
        raise ValueError("Dataset needs at least three distinct input profiles for grouped CV.")

    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    search = GridSearchCV(
        build_estimator(),
        PARAM_GRID,
        cv=cv,
        scoring=make_scorer(fbeta_score, beta=2, zero_division=0),
        n_jobs=1,
        error_score="raise",
    )
    search.fit(x, y, groups=groups)
    estimator: Pipeline = search.best_estimator_
    scaler: StandardScaler = estimator.named_steps["scaler"]
    svm: SVC = estimator.named_steps["svc"]

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_dataset": DATASET_URL,
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(y)),
        "unique_input_profiles": int(unique_groups),
        "label": "Sleep Apnea vs other source-dataset labels",
        "positive_class": 1,
        "decision_threshold": 0.0,
        "best_params": {
            key.removeprefix("svc__"): value for key, value in search.best_params_.items()
        },
        "features": list(FEATURES),
        "feature_bounds": {
            name: [float(np.min(x[:, i])), float(np.max(x[:, i]))]
            for i, name in enumerate(FEATURES)
        },
        "categorical_values": {
            "Gender": sorted(np.unique(x[:, 0]).astype(int).tolist()),
            "BMI Category": sorted(np.unique(x[:, 3]).astype(int).tolist()),
        },
        "cv_f2_mean": float(search.best_score_),
        "cv_note": (
            "Three-fold stratified grouped cross-validation on the full source dataset; "
            "development metric only, not a held-out test estimate."
        ),
    }
    bundle = {
        "metadata": metadata,
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "svm": {
            "kernel": svm.kernel,
            "gamma": float(svm._gamma),
            "support_vectors": svm.support_vectors_.tolist(),
            "dual_coef": svm.dual_coef_[0].tolist(),
            "intercept": float(svm.intercept_[0]),
            "classes": svm.classes_.astype(int).tolist(),
        },
    }
    output = Path(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, separators=(",", ":"), allow_nan=False) + "\n")
    return metadata


def load_model(model_path: str | Path = MODEL_PATH) -> dict[str, Any]:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Pre-trained lifestyle model was not found: {path}")
    bundle = json.loads(path.read_text())
    metadata = bundle.get("metadata", {})
    if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("features") != list(
        FEATURES
    ):
        raise ValueError("Lifestyle model schema is invalid or incompatible.")
    if metadata.get("positive_class") != 1 or bundle.get("svm", {}).get("classes") != [0, 1]:
        raise ValueError("Lifestyle model class encoding is invalid.")
    return bundle


def _decision_function(bundle: dict[str, Any], row: np.ndarray) -> float:
    scaler = bundle["scaler"]
    svm = bundle["svm"]
    scaled = (row - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])
    support_vectors = np.asarray(svm["support_vectors"])
    if svm["kernel"] == "rbf":
        squared_distance = np.maximum(
            np.sum(scaled**2) + np.sum(support_vectors**2, axis=1) - 2 * support_vectors @ scaled,
            0.0,
        )
        kernel_values = np.exp(-svm["gamma"] * squared_distance)
    elif svm["kernel"] == "linear":
        kernel_values = support_vectors @ scaled
    else:
        raise ValueError(f"Unsupported SVM kernel in model: {svm['kernel']}")
    return float(kernel_values @ np.asarray(svm["dual_coef"]) + svm["intercept"])


def predict(bundle: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    clean = validate_record(record)
    row = _feature_row(clean)
    metadata = bundle["metadata"]
    warnings = []
    categorical = metadata["categorical_values"]
    for index, key in enumerate(FEATURES):
        low, high = metadata["feature_bounds"][key]
        value = float(row[index])
        supported = (
            int(value) in categorical[key]
            if key in categorical
            else low <= value <= high
        )
        if not supported:
            warnings.append(f"{key} is outside this model's training-data support ({low:g}–{high:g}).")

    reported_symptoms = [
        name.replace("_", " ")
        for name in SYMPTOMS
        if clean.get(name, False)
    ]
    margin = None if warnings else _decision_function(bundle, row)
    positive = None if margin is None else margin >= metadata["decision_threshold"]
    status = (
        "insufficient_evidence"
        if positive is None
        else "screen_positive"
        if positive
        else "lower_model_signal"
    )
    follow_up = bool(reported_symptoms or positive is True)
    return {
        "status": status,
        "screen_positive": positive,
        "decision_margin": margin,
        "reported_symptoms": reported_symptoms,
        "warnings": warnings,
        "follow_up_recommended": follow_up,
        "next_step": (
            "Discuss these findings and symptoms with a clinician; ask whether a sleep assessment is appropriate."
            if follow_up
            else "A lower model signal does not rule out apnea. Seek clinical advice if symptoms or concerns persist."
            if not warnings
            else "The model cannot assess these inputs. Check the measurements and seek clinical advice if concerned."
        ),
        "disclaimer": (
            "This is a research screening estimate based on a small lifestyle dataset, not a "
            "diagnosis. It does not measure breathing events or clinical severity."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the six-input lifestyle SVM.")
    parser.add_argument("--data", type=Path, required=True, help="Source lifestyle dataset CSV")
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    arguments = parser.parse_args()
    result = train_and_save(arguments.data, arguments.output)
    print(f"Saved lifestyle model: {arguments.output}")
    print(f"Rows: {result['training_rows']}; grouped-CV development F2: {result['cv_f2_mean']:.3f}")
