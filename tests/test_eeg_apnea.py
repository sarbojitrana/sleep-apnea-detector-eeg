import copy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.svm import SVC

from eeg_apnea import (FEATURE_NAMES, REFERENCE_CONTRACT, extract_features, load_model,
                       predict_features, prepare_training_data, reference_filter,
                       synthetic_features, train_svm, validate_features)
from prepare_eeg_dataset import annotation_label


@pytest.fixture(scope='module')
def trained():
    # Synthetic fixtures test software behavior only, not real apnea performance.
    rows = []
    for subject in range(10):
        data = synthetic_features(seed=subject, windows=16)
        data['subject_id'] = f'fixture-{subject}'
        data['label'] = np.arange(16) % 2
        data.loc[data['label'] == 1, 'E_D3'] *= 5
        rows.append(data)
    frame = pd.concat(rows, ignore_index=True)
    grid = [{'svc__kernel': ['rbf'], 'svc__C': [1], 'select__k': [5],
             'transform__logarithmic': [True]}]
    model = train_svm(frame, REFERENCE_CONTRACT, 'unit-test fixture only', param_grid=grid)
    return frame, model


def test_feature_contract_is_ordered_and_validation_never_drops_bad_rows():
    frame = synthetic_features(windows=3)
    assert list(validate_features(frame[frame.columns[::-1]])) == list(FEATURE_NAMES)
    frame.loc[1, 'M_A5'] = np.nan
    with pytest.raises(ValueError, match='finite'):
        validate_features(frame)
    with pytest.raises(ValueError, match='Missing'):
        validate_features(frame.drop(columns='K_D2'))
    with pytest.raises(ValueError, match='unique'):
        validate_features(pd.concat([frame, frame[['E_A5']]], axis=1))


def test_invalid_features_and_training_provenance_are_rejected():
    frame = synthetic_features(windows=6)
    frame.loc[0, 'E_A5'] = -1
    with pytest.raises(ValueError, match='non-negative'):
        validate_features(frame)
    frame = synthetic_features(windows=6)
    frame['label'] = [0, 1] * 3
    with pytest.raises(ValueError, match='subject_id'):
        prepare_training_data(frame)
    frame['subject_id'] = [str(i) for i in range(6)]
    frame['source'] = 'synthetic demo'
    with pytest.raises(ValueError, match='Synthetic'):
        prepare_training_data(frame)


def test_subject_splits_and_scaler_exclude_test_subjects(trained):
    frame, model = trained
    report = model['report']
    dev_subjects, test_subjects = set(report['train_subjects']), set(report['test_subjects'])
    assert not dev_subjects & test_subjects
    for fold in report['cv_folds']:
        assert not set(fold['train_subjects']) & set(fold['validation_subjects'])
        assert set(fold['train_subjects']) | set(fold['validation_subjects']) <= dev_subjects
    dev = frame.loc[frame.subject_id.isin(dev_subjects), FEATURE_NAMES].to_numpy()
    np.testing.assert_allclose(model['mean'], np.log1p(dev).mean(axis=0))
    assert len(model['selected_indices']) == 5


def test_portable_scores_match_sklearn_and_roundtrip(trained, tmp_path):
    frame, model = trained
    dev = frame.subject_id.isin(model['report']['train_subjects'])
    values = frame.loc[:, FEATURE_NAMES].to_numpy()
    standardized = ((np.log1p(values) - model['mean']) / model['scale'])[:, model['selected_indices']]
    estimator = SVC(kernel='rbf', C=1).fit(standardized[dev], frame.label[dev])
    path = tmp_path / 'eeg.json'
    path.write_text(json.dumps(model, allow_nan=False))
    restored = load_model(path)
    result = predict_features(restored, frame)
    np.testing.assert_allclose(result.decision_margin, estimator.decision_function(standardized), atol=1e-8)
    np.testing.assert_array_equal(result.prediction, estimator.predict(standardized))
    reordered = predict_features(restored, frame[list(FEATURE_NAMES)[::-1]])
    pd.testing.assert_frame_equal(result, reordered)


def test_corrupt_model_rejected(trained, tmp_path):
    model = copy.deepcopy(trained[1])
    model['scale'][0] = 0
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(model))
    with pytest.raises(ValueError, match='scale'):
        load_model(path)


def test_reference_features_use_haar_and_pearson_kurtosis():
    # Alternating blocks of four samples concentrate in D3; all descriptors finite.
    signal = np.tile([1.] * 4 + [-1.] * 8 + [1.] * 4, 469)[:7500]
    features = extract_features(signal)
    assert tuple(features) == FEATURE_NAMES
    assert features['E_D3'] > 7400
    assert features['K_D3'] == pytest.approx(1.0, abs=.01)
    assert features['M_D3'] > 1.99
    with pytest.raises(ValueError, match='Constant'):
        extract_features(np.ones(7500))
    # Causal filtering must not let later samples change earlier outputs.
    samples = np.zeros(15000)
    samples[8000] = 1
    assert not reference_filter(samples)[:8000].any()


@pytest.mark.parametrize(('note', 'label'), [
    ('2 X X', 1), ('R OA', 1), ('1 CAA', 1), ('3', 0), ('2 A', 0),
    ('2 HA', None), ('1 H', None), ('W X', None), ('MT', None), ('2 UNKNOWN', None),
])
def test_dataset_label_mapping(note, label):
    assert annotation_label(note) == label


def test_random_demo_is_reproducible_but_changes_with_seed(trained):
    first = synthetic_features(seed=1, model=trained[1])
    pd.testing.assert_frame_equal(first, synthetic_features(seed=1, model=trained[1]))
    assert not first.equals(synthetic_features(seed=2, model=trained[1]))
    assert 'label' not in first
    validate_features(first)
