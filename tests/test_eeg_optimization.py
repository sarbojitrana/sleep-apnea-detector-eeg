import json

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from eeg_apnea import (REFERENCE_CONTRACT,
                       load_model, predict_features, synthetic_features, train_svm, transform_features)
from optimize_eeg_svm import subject_class_weights


@pytest.mark.parametrize('mode', ['relative', 'log_relative', 'centered_log'])
def test_relative_energy_transform_ignores_subject_amplitude_and_batch_context(mode):
    values = synthetic_features(seed=7, windows=4).to_numpy()
    louder = values.copy()
    louder[:, 0::3] *= 100
    np.testing.assert_allclose(transform_features(values, mode), transform_features(louder, mode), atol=1e-10)
    np.testing.assert_allclose(transform_features(values, mode)[:1], transform_features(values[:1], mode))
    np.testing.assert_array_equal(values, synthetic_features(seed=7, windows=4).to_numpy())


def test_subject_weights_do_not_let_larger_subjects_dominate():
    groups = np.array(['a'] * 8 + ['b'] * 4 + ['c'] * 3)
    labels = np.array([0] * 6 + [1] * 2 + [0, 0, 1, 1] + [0, 0, 0])
    weights = subject_class_weights(labels, groups)
    totals = [weights[groups == subject].sum() for subject in np.unique(groups)]
    np.testing.assert_allclose(totals, [5, 5, 5])
    assert weights[(groups == 'a') & (labels == 0)].sum() == pytest.approx(2.5)
    assert weights[(groups == 'a') & (labels == 1)].sum() == pytest.approx(2.5)


def test_relative_transform_robust_scaler_portable_model_roundtrip(tmp_path):
    rows = []
    for subject in range(10):
        part = synthetic_features(seed=subject, windows=12)
        part['subject_id'] = f'subject-{subject}'
        part['label'] = np.arange(12) % 2
        part.loc[part.label == 1, 'E_D3'] *= 5
        rows.append(part)
    frame = pd.concat(rows, ignore_index=True)
    model = train_svm(frame, REFERENCE_CONTRACT, 'unit-test fixture', param_grid=[{
        'transform__mode': ['relative'], 'scale': [RobustScaler()],
        'select__k': ['all'], 'svc__C': [1], 'svc__kernel': ['rbf'],
    }])
    path = tmp_path / 'relative_svm.json'
    path.write_text(json.dumps(model, allow_nan=False))
    restored = load_model(path)
    assert restored['input_transform'] == 'relative'
    assert restored['scaler_type'] == 'RobustScaler'
    pd.testing.assert_frame_equal(predict_features(model, frame), predict_features(restored, frame))
