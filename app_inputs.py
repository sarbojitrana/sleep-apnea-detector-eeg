"""Helpers for supported demo profiles and clearly synthetic sensor previews."""

from __future__ import annotations

from typing import Any

import numpy as np

from lifestyle_model import BMI_LABELS, GENDER_MAP


def random_supported_record(
    model: dict[str, Any],
    seed: int | None = None,
) -> dict[str, Any]:
    """Create lifestyle inputs within the bundled model's observed feature support."""
    rng = np.random.default_rng(seed)
    metadata = model["metadata"]
    bounds = metadata["feature_bounds"]
    gender_code = int(rng.choice(metadata["categorical_values"]["Gender"]))
    bmi_code = int(rng.choice(metadata["categorical_values"]["BMI Category"]))
    age_low, age_high = (int(value) for value in bounds["Age"])
    sleep_low, sleep_high = bounds["Sleep Duration"]
    heart_low, heart_high = (int(value) for value in bounds["Heart Rate"])
    steps_low, steps_high = (int(value) for value in bounds["Daily Steps"])
    step_units_low = int(np.ceil(steps_low / 100))
    step_units_high = int(np.floor(steps_high / 100))
    record = {
        "gender": next(name.title() for name, code in GENDER_MAP.items() if code == gender_code),
        "age": int(rng.integers(age_low, age_high + 1)),
        "bmi": BMI_LABELS[bmi_code],
        "sleep_duration": round(float(rng.uniform(sleep_low, sleep_high)), 1),
        "heart_rate": int(rng.integers(heart_low, heart_high + 1)),
        "daily_steps": int(rng.integers(step_units_low, step_units_high + 1) * 100),
        "loud_snoring": False,
        "witnessed_pauses": False,
        "daytime_sleepiness": False,
    }
    return record


def example_supported_record() -> dict[str, Any]:
    return {
        "gender": "Female",
        "age": 40,
        "bmi": "Normal",
        "sleep_duration": 7.0,
        "heart_rate": 70,
        "daily_steps": 7000,
        "loud_snoring": False,
        "witnessed_pauses": False,
        "daytime_sleepiness": False,
    }


def synthetic_signal(
    kind: str,
    sampling_rate: float,
    duration_seconds: float,
    seed: int = 42,
) -> np.ndarray:
    """Generate an unmistakably synthetic signal for UI plumbing tests and previews."""
    if kind not in {"EEG", "ECG", "SpO2"}:
        raise ValueError("Signal kind must be EEG, ECG, or SpO2.")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("Sampling rate must be a positive finite value.")
    if not np.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("Signal duration must be a positive finite value.")
    if kind == "EEG" and sampling_rate <= 90:
        raise ValueError("EEG demo sampling rate must be greater than 90 Hz.")

    sample_count = int(round(sampling_rate * duration_seconds))
    if sample_count < 2 or sample_count > 1_000_000:
        raise ValueError("Demo signal duration and sampling rate produce an unsupported sample count.")
    time_axis = np.arange(sample_count) / sampling_rate
    rng = np.random.default_rng(seed)

    if kind == "EEG":
        return (
            np.sin(2 * np.pi * 2.0 * time_axis)
            + 0.6 * np.sin(2 * np.pi * 10.0 * time_axis)
            + 0.25 * np.sin(2 * np.pi * 20.0 * time_axis)
            + 0.05 * rng.normal(size=sample_count)
        )
    if kind == "ECG":
        heart_period = 60.0 / 72.0
        phase = np.mod(time_axis, heart_period)
        qrs = np.exp(-0.5 * ((phase - 0.12) / 0.018) ** 2)
        p_wave = 0.12 * np.exp(-0.5 * ((phase - 0.04) / 0.018) ** 2)
        t_wave = 0.25 * np.exp(-0.5 * ((phase - 0.32) / 0.06) ** 2)
        return qrs - 0.18 * np.exp(-0.5 * ((phase - 0.095) / 0.012) ** 2) + p_wave + t_wave

    return np.clip(
        96.0 + 0.4 * np.sin(2 * np.pi * time_axis / 18.0) + 0.15 * rng.normal(size=sample_count),
        92.0,
        99.0,
    )


def summarize_signal(
    samples: np.ndarray,
    sampling_rate: float,
    kind: str,
) -> dict[str, float | int | str]:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"{kind} samples must be a finite one-dimensional signal.")
    if len(values) > 1_000_000:
        raise ValueError(f"{kind} CSV cannot exceed 1,000,000 samples.")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError(f"{kind} sampling rate must be positive and finite.")
    if kind == "EEG" and sampling_rate <= 90:
        raise ValueError("EEG sampling rate must be greater than 90 Hz.")
    if kind == "SpO2" and (np.min(values) < 50 or np.max(values) > 100):
        raise ValueError("SpO2 values must be between 50 and 100 percent.")
    return {
        "channel": kind,
        "samples": int(len(values)),
        "sampling_rate_hz": float(sampling_rate),
        "duration_seconds": float(len(values) / sampling_rate),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values)),
    }
