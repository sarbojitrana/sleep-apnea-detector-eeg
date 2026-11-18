"""Capture newline-delimited EEG samples from a configurable FPGA serial port."""

from __future__ import annotations

import time
from math import isfinite


def capture_samples(
    port: str,
    baud_rate: int,
    duration_seconds: float,
    max_samples: int = 2_000_000,
) -> list[float]:
    """Read one decimal EEG sample per newline until the requested duration elapses."""
    if not port.strip():
        raise ValueError("Enter the FPGA serial port name.")
    if baud_rate <= 0:
        raise ValueError("Baud rate must be a positive integer.")
    if not isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("Capture duration must be a positive number of seconds.")
    if max_samples <= 0:
        raise ValueError("Maximum sample count must be a positive integer.")

    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("Serial capture requires pyserial; install the project requirements.") from exc

    samples: list[float] = []
    deadline = time.monotonic() + duration_seconds
    try:
        with serial.Serial(port.strip(), baudrate=baud_rate, timeout=0.2) as connection:
            while time.monotonic() < deadline:
                line = connection.readline()
                if not line:
                    continue
                try:
                    value = float(line.decode("ascii").strip())
                except (UnicodeDecodeError, ValueError) as exc:
                    raise ValueError(
                        "FPGA serial output must be one ASCII numeric sample per line."
                    ) from exc
                if not (float("-inf") < value < float("inf")):
                    raise ValueError("FPGA serial output contained a non-finite sample.")
                samples.append(value)
                if len(samples) > max_samples:
                    raise ValueError("Serial capture exceeded the configured sample limit.")
    except OSError as exc:
        raise RuntimeError(f"Could not read FPGA serial port {port!r}: {exc}") from exc

    if not samples:
        raise ValueError("No EEG samples were received from the FPGA during the capture.")
    return samples
