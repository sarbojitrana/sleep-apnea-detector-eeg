"""Read one decoded 15-feature EEG window per ASCII CSV line from the FPGA."""
from __future__ import annotations
import csv
from math import isfinite
import time

import pandas as pd
from eeg_apnea import FEATURE_NAMES, validate_features


def parse_feature_line(line: bytes | str) -> dict[str, float]:
    try:
        text = line.decode('ascii') if isinstance(line, bytes) else line
        parts = next(csv.reader([text.strip()]))
        if len(parts) != len(FEATURE_NAMES):
            raise ValueError('wrong field count')
        row = dict(zip(FEATURE_NAMES, (float(value) for value in parts)))
    except (UnicodeDecodeError, ValueError, csv.Error) as exc:
        raise ValueError('FPGA output must contain exactly 15 comma-separated ASCII numbers per line.') from exc
    return validate_features(pd.DataFrame([row])).iloc[0].to_dict()


def capture_features(port: str, baud_rate: int, duration_seconds: float,
                     max_windows: int = 1000) -> pd.DataFrame:
    """Bounded capture; an optional exact FEATURE_NAMES header is accepted."""
    if not port.strip():
        raise ValueError('Enter the FPGA serial port name.')
    if not isinstance(baud_rate, int) or baud_rate <= 0:
        raise ValueError('Baud rate must be a positive integer.')
    if not isfinite(duration_seconds) or not 0 < duration_seconds <= 600:
        raise ValueError('Capture duration must be between 0 and 600 seconds.')
    if not isinstance(max_windows, int) or not 1 <= max_windows <= 1000:
        raise ValueError('Maximum window count must be between 1 and 1,000.')
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError('Serial capture requires pyserial; install the project requirements.') from exc
    rows = []
    try:
        with serial.Serial(port.strip(), baudrate=baud_rate, timeout=.2) as connection:
            deadline = time.monotonic() + duration_seconds
            pending = b''
            while time.monotonic() < deadline and len(rows) < max_windows:
                pending += connection.read_until(b'\n', size=4097 - len(pending))
                if len(pending) > 4096:
                    raise ValueError('FPGA feature frame exceeds 4,096 bytes.')
                if not pending.endswith(b'\n'):
                    continue
                line, pending = pending.strip(), b''
                if not line:
                    continue
                if line == ','.join(FEATURE_NAMES).encode('ascii'):
                    continue
                rows.append(parse_feature_line(line))
            if pending.strip():
                raise ValueError('Capture ended with an incomplete FPGA feature frame; no partial frame was accepted.')
    except OSError as exc:
        raise RuntimeError(f'Could not read FPGA serial port {port!r}: {exc}') from exc
    if not rows:
        raise ValueError('No complete EEG feature windows were received from the FPGA.')
    return pd.DataFrame(rows, columns=FEATURE_NAMES)
