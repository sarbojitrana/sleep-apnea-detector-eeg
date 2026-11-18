import numpy as np
import pandas as pd
import pytest

from eeg_apnea import (
    FEATURE_NAMES,
    extract_features,
    predict_signal,
    prepare_training_data,
    split_signal,
    train_svm,
)


def _window(frequency: float, phase: float, sampling_rate: int = 128) -> np.ndarray:
    time = np.arange(sampling_rate * 4) / sampling_rate
    return np.sin(2 * np.pi * frequency * time + phase)


def test_extract_features_returns_all_band_statistics() -> None:
    features = extract_features(_window(10, 0), 128)

    assert tuple(features) == FEATURE_NAMES
    assert np.isfinite(list(features.values())).all()


@pytest.mark.parametrize(
    ("samples", "sampling_rate", "message"),
    [
        (np.ones(512), 128, "constant"),
        (np.ones(512), 90, "greater than 90"),
        (np.array([0.0, np.nan, 1.0]), 128, "finite"),
    ],
)
def test_extract_features_rejects_unusable_input(
    samples: np.ndarray,
    sampling_rate: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        extract_features(samples, sampling_rate)


def test_train_and_predict_uses_subject_holdout() -> None:
    rows = []
    for label, frequency in (("normal", 10), ("apnea", 2)):
        for subject in range(6):
            for window in range(2):
                samples = _window(frequency, phase=subject * 0.1 + window * 0.2)
                rows.append(
                    {
                        "label": label,
                        "subject_id": f"{label}-{subject}",
                        "sampling_rate": 128,
                        **{f"sample_{i}": value for i, value in enumerate(samples)},
                    }
                )

    features, labels, groups = prepare_training_data(pd.DataFrame(rows))
    model, report = train_svm(features, labels, groups)
    prediction = predict_signal(model, _window(2, 0), 128, 4)

    assert report["split_method"].startswith("Subject-wise")
    assert set(report["classes"]) == {"apnea", "normal"}
    assert prediction.loc[0, "prediction"] == "apnea"
    assert prediction.loc[0, "decision_score_apnea"] > 0


def test_prepare_training_data_requires_sampling_rate() -> None:
    frame = pd.DataFrame(
        [{"label": "normal", **{f"sample_{i}": value for i, value in enumerate(_window(10, 0))}}]
    )

    with pytest.raises(ValueError, match="sampling_rate"):
        prepare_training_data(frame)


def test_split_signal_rejects_overlap_that_rounds_to_zero_step() -> None:
    signal = _window(10, 0)

    with pytest.raises(ValueError, match="too close"):
        split_signal(signal, 128, window_seconds=2, overlap_seconds=1.999)
