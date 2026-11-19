import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from eeg_apnea import EXAMPLE_VALUES, FEATURE_NAMES
from fpga_serial import capture_features, parse_feature_line


def test_parser_requires_full_finite_feature_vector():
    row = parse_feature_line(','.join(map(str, EXAMPLE_VALUES)).encode())
    assert tuple(row) == FEATURE_NAMES
    assert row['E_A5'] == 125.4
    for line in ('1,2', 'x,' * 14 + '2', ','.join(['nan'] * 15)):
        with pytest.raises(ValueError):
            parse_feature_line(line)


def test_capture_accumulates_partial_reads_and_accepts_header(monkeypatch):
    line = ','.join(map(str, EXAMPLE_VALUES)).encode() + b'\n'
    class Connection:
        def __init__(self, *args, **kwargs):
            self.parts = iter([','.join(FEATURE_NAMES).encode() + b'\n', line[:20], line[20:]])
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read_until(self, *args, **kwargs): return next(self.parts)
    monkeypatch.setitem(sys.modules, 'serial', SimpleNamespace(Serial=Connection))
    result = capture_features('/dev/fake', 115200, 1., max_windows=1)
    pd.testing.assert_frame_equal(result, pd.DataFrame([EXAMPLE_VALUES], columns=FEATURE_NAMES))


def test_capture_rejects_empty_port_without_touching_hardware():
    with pytest.raises(ValueError, match='port'):
        capture_features('', 115200, 1.)
