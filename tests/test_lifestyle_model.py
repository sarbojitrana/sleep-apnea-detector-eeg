import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lifestyle_model import (
    FEATURES,
    _decision_function,
    load_model,
    predict,
    validate_record,
)


def _trusted_test_bundle() -> tuple[dict, Pipeline]:
    x = np.asarray(
        [
            [0, 30, 7.0, 0, 68, 7000],
            [1, 45, 6.0, 2, 78, 4500],
            [0, 55, 5.8, 1, 75, 6000],
            [1, 35, 7.8, 0, 70, 9000],
        ],
        dtype=float,
    )
    y = np.asarray([0, 1, 1, 0])
    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=2, gamma=0.1, class_weight="balanced")),
        ]
    ).fit(x, y)
    scaler = estimator.named_steps["scaler"]
    svm = estimator.named_steps["svc"]
    bundle = {
        "metadata": {
            "schema_version": 1,
            "features": list(FEATURES),
            "positive_class": 1,
            "decision_threshold": 0.0,
            "feature_bounds": {
                name: [float(x[:, i].min()), float(x[:, i].max())]
                for i, name in enumerate(FEATURES)
            },
            "categorical_values": {"Gender": [0, 1], "BMI Category": [0, 1, 2]},
        },
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "svm": {
            "kernel": svm.kernel,
            "gamma": float(svm._gamma),
            "support_vectors": svm.support_vectors_.tolist(),
            "dual_coef": svm.dual_coef_[0].tolist(),
            "intercept": float(svm.intercept_[0]),
            "classes": [0, 1],
        },
    }
    return bundle, estimator


def test_portable_rbf_decision_matches_sklearn() -> None:
    bundle, estimator = _trusted_test_bundle()
    row = np.asarray([0, 40, 6.8, 1, 72, 6500], dtype=float)

    expected = estimator.decision_function(row.reshape(1, -1))[0]
    actual = _decision_function(bundle, row)

    assert actual == pytest.approx(expected)


def test_portable_linear_decision_matches_sklearn() -> None:
    bundle, estimator = _trusted_test_bundle()
    svm = estimator.named_steps["svc"]
    svm.set_params(kernel="linear")
    estimator.fit(
        np.asarray(
            [
                [0, 30, 7.0, 0, 68, 7000],
                [1, 45, 6.0, 2, 78, 4500],
                [0, 55, 5.8, 1, 75, 6000],
                [1, 35, 7.8, 0, 70, 9000],
            ],
            dtype=float,
        ),
        np.asarray([0, 1, 1, 0]),
    )
    scaler = estimator.named_steps["scaler"]
    svm = estimator.named_steps["svc"]
    bundle["scaler"] = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    bundle["svm"] = {
        "kernel": svm.kernel,
        "gamma": float(svm._gamma),
        "support_vectors": svm.support_vectors_.tolist(),
        "dual_coef": svm.dual_coef_[0].tolist(),
        "intercept": float(svm.intercept_[0]),
        "classes": [0, 1],
    }
    row = np.asarray([0, 40, 6.8, 1, 72, 6500], dtype=float)

    assert _decision_function(bundle, row) == pytest.approx(
        estimator.decision_function(row.reshape(1, -1))[0]
    )


def test_input_validation_requires_all_six_fields() -> None:
    with pytest.raises(ValueError, match="Missing required inputs"):
        validate_record({"gender": "Female"})


def test_bundled_lifestyle_model_loads_and_returns_supported_screen() -> None:
    model = load_model()
    result = predict(
        model,
        {
            "gender": "Female",
            "age": 40,
            "bmi": "Normal",
            "sleep_duration": 7.0,
            "heart_rate": 70,
            "daily_steps": 7000,
        },
    )

    assert result["status"] in {"screen_positive", "lower_model_signal"}
    assert isinstance(result["screen_positive"], bool)
    assert result["decision_margin"] is not None
