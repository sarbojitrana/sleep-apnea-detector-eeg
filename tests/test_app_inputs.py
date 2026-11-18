import numpy as np
import pytest

from app_inputs import random_supported_record, summarize_signal, synthetic_signal
from lifestyle_model import load_model, predict


def test_random_profile_is_inside_bundled_model_support() -> None:
    model = load_model()
    record = random_supported_record(model, seed=7)
    result = predict(model, record)

    assert result["status"] in {"screen_positive", "lower_model_signal"}
    assert record["age"] in range(27, 60)
    assert 5.8 <= record["sleep_duration"] <= 8.5
    assert record["heart_rate"] in range(65, 87)
    assert 3000 <= record["daily_steps"] <= 10000


@pytest.mark.parametrize(
    ("kind", "sampling_rate", "duration"),
    (("EEG", 125, 30), ("ECG", 250, 30), ("SpO2", 1, 30)),
)
def test_synthetic_signals_are_labeled_and_summarizable(
    kind: str,
    sampling_rate: int,
    duration: int,
) -> None:
    samples = synthetic_signal(kind, sampling_rate, duration)
    summary = summarize_signal(samples, sampling_rate, kind)

    assert len(samples) == sampling_rate * duration
    assert summary["channel"] == kind
    assert summary["duration_seconds"] == pytest.approx(duration)


def test_spo2_summary_rejects_non_percentage_values() -> None:
    with pytest.raises(ValueError, match="between 50 and 100"):
        summarize_signal(np.asarray([0.96, 0.97, 0.95]), 1, "SpO2")


def test_eeg_demo_requires_feature_extraction_sample_rate() -> None:
    with pytest.raises(ValueError, match="greater than 90"):
        synthetic_signal("EEG", 50, 30)
