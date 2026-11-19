"""FPGA feature contract, offline training reference, and portable EEG SVM."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter, sosfilt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'eeg_svm.json'
BANDS = ('A5', 'D5', 'D4', 'D3', 'D2')
FEATURE_NAMES = tuple(f'{stat}_{band}' for band in BANDS for stat in ('E', 'M', 'K'))
EXAMPLE_VALUES = (125.4, .42, 3.1, 87.2, .61, 2.8, 64.8, .75, 3.5,
                  112.6, .93, 4.2, 38.5, 1.12, 3.0)
REFERENCE_CONTRACT = {
    'sampling_rate_hz': 250.0, 'window_seconds': 30.0, 'channel': 'EEG (C3-O1)',
    'amplitude_unit': 'uV',
    'filter': 'Butterworth prototype order 4 (bandpass order 8), 0.5-45 Hz, causal SOS, continuous per recording, zero initial state',
    'filter_sos': butter(4, (.5, 45), btype='bandpass', fs=250, output='sos').tolist(),
    'wavelet': 'haar', 'level': 5, 'boundary': 'symmetric',
    'coefficient_counts': {'A5': 235, 'D5': 235, 'D4': 469, 'D3': 938, 'D2': 1875},
    'energy': 'sum of squared coefficients (uV^2)',
    'mobility': 'sqrt(population_variance(diff(coefficients)) / population_variance(coefficients)); no sample-rate multiplier',
    'kurtosis': 'Pearson population kurtosis (Gaussian=3), no excess or bias correction',
    'degenerate_band': 'mobility=0 and kurtosis=0 when variance is zero',
    'transport': 'decoded floating-point physical features, no prior normalization; unrounded',
    'fpga_numerical_parity': 'Not yet validated against RTL / fixed-point output',
}


class FeatureTransform(TransformerMixin, BaseEstimator):
    """Fixed element-wise transform; its choice is tuned inside development CV."""
    def __init__(self, logarithmic=False, mode=None):
        self.logarithmic = logarithmic
        self.mode = mode

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        mode = self.mode or ('log1p' if self.logarithmic else 'identity')
        return transform_features(np.asarray(x), mode)


TRANSFORMS = ('identity', 'log1p', 'relative', 'log_relative', 'centered_log')


def transform_features(values: np.ndarray, mode: str) -> np.ndarray:
    """Window-local transforms; never estimate statistics from another window."""
    if mode not in TRANSFORMS:
        raise ValueError(f'Unknown EEG input transform: {mode}')
    if mode == 'identity':
        return values.copy()
    transformed = np.log1p(values)
    if mode in ('relative', 'log_relative', 'centered_log'):
        energy = values[:, 0::3]
        # Rescale before summation to avoid overflow from large finite inputs.
        energy = energy / energy.max(axis=1, keepdims=True)
        fractions = energy / energy.sum(axis=1, keepdims=True)
        if mode == 'relative':
            transformed[:, 0::3] = fractions
        else:
            log_fraction = np.log(np.maximum(fractions, 1e-12))
            transformed[:, 0::3] = (log_fraction - log_fraction.mean(axis=1, keepdims=True)
                                     if mode == 'centered_log' else log_fraction)
    return transformed


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Require all named features; reorder them without dropping invalid rows."""
    if frame.columns.duplicated().any():
        raise ValueError('EEG feature columns must have unique names.')
    missing = set(FEATURE_NAMES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing FPGA features: {', '.join(sorted(missing))}")
    if frame.empty or len(frame) > 100_000:
        raise ValueError('Provide between 1 and 100,000 EEG feature windows.')
    values = frame.loc[:, FEATURE_NAMES].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError('All 15 FPGA features must be finite numbers; no rows are discarded.')
    if (values < 0).any().any():
        raise ValueError('Energy, mobility, and Pearson kurtosis must be non-negative.')
    if (values[[f'E_{band}' for band in BANDS]].sum(axis=1) <= 0).any():
        raise ValueError('An EEG window with zero energy in every band is unusable.')
    return values.astype(float)


def synthetic_features(seed: int | None = 42, windows: int = 6, model: dict | None = None) -> pd.DataFrame:
    """Invented parameters, independent of labels and never used for training."""
    if not 1 <= windows <= 1000:
        raise ValueError('Demo window count must be between 1 and 1,000.')
    rng = np.random.default_rng(seed)
    if model is not None and 'demo_feature_profiles' in model:
        # Unlabelled development examples supply plausible correlated scales only.
        profiles = np.asarray(model['demo_feature_profiles'])
        values = profiles[rng.integers(0, len(profiles), windows)] * rng.lognormal(0, .05, (windows, 15))
    else:
        values = np.asarray(EXAMPLE_VALUES) * rng.lognormal(0, .25, (windows, 15))
    values[:, 2::3] = np.maximum(1.0, values[:, 2::3])
    return pd.DataFrame(values, columns=FEATURE_NAMES)


def reference_filter(samples: np.ndarray) -> np.ndarray:
    """Offline causal reference: call once per recording, before windowing."""
    values = np.asarray(samples, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Reference EEG must be a finite, single-channel signal in uV.')
    return sosfilt(np.asarray(REFERENCE_CONTRACT['filter_sos']), values)


def extract_features(filtered_window: np.ndarray) -> dict[str, float]:
    """Haar reference for one already-filtered 30 s / 250 Hz EEG window."""
    values = np.asarray(filtered_window, dtype=float)
    if values.shape != (7500,) or not np.isfinite(values).all():
        raise ValueError('Reference extraction needs 7,500 finite, already-filtered samples.')
    if np.ptp(values) == 0:
        raise ValueError('Constant EEG window is unusable.')
    blocks = pywt.wavedec(values, 'haar', level=5, mode='symmetric')[:5]
    result = {}
    for band, block in zip(BANDS, blocks):
        centered = block - block.mean()
        variance = float(np.mean(centered ** 2))
        result[f'E_{band}'] = float(block @ block)
        result[f'M_{band}'] = float(np.sqrt(np.var(np.diff(block)) / variance)) if variance > 0 else 0.0
        result[f'K_{band}'] = float(np.mean(centered ** 4) / variance ** 2) if variance > 0 else 0.0
    return result


def prepare_training_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not {'label', 'subject_id'}.issubset(frame.columns):
        raise ValueError('Training CSV requires label (0=normal, 1=apnea) and subject_id.')
    if 'source' in frame and frame['source'].astype(str).str.contains('synthetic|demo', case=False).any():
        raise ValueError('Synthetic/demo rows cannot be used for EEG training.')
    features = validate_features(frame)
    labels = pd.to_numeric(frame['label'], errors='coerce').to_numpy()
    if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError('Training needs both binary labels: 0=normal, 1=apnea.')
    if frame['subject_id'].isna().any():
        raise ValueError('Training subject_id cannot be missing.')
    groups = frame['subject_id'].astype(str).str.strip().to_numpy()
    if any(not value for value in groups) or len(np.unique(groups)) < 6:
        raise ValueError('Training needs at least six non-empty, distinct subject_id values.')
    duplicates = features.assign(subject_id=groups).groupby(list(FEATURE_NAMES))['subject_id'].nunique()
    if (duplicates > 1).any():
        raise ValueError('Identical feature vectors occur across subjects; resolve potential leakage first.')
    return features, labels.astype(int), groups


def _group_folds(x, y, groups, max_splits=5):
    """Deterministic split using only labels/groups, never model scores."""
    for count in range(min(max_splits, len(np.unique(groups))), 1, -1):
        splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=42)
        folds = list(splitter.split(x, y, groups))
        if all(len(np.unique(y[a])) == len(np.unique(y[b])) == 2 for a, b in folds):
            return folds
    raise ValueError('Not enough subjects containing both classes for subject-wise evaluation.')


def train_svm(frame: pd.DataFrame, contract: dict[str, Any], source: str,
              param_grid: list[dict] | None = None) -> dict[str, Any]:
    """Tune on development subjects; evaluate once and retain the evaluated model."""
    if not source.strip():
        raise ValueError('Document the real labeled EEG dataset source.')
    if not set(REFERENCE_CONTRACT).issubset(contract):
        raise ValueError('Feature contract must document the full extraction and scaling convention.')
    features, y, groups = prepare_training_data(frame)
    x = features.to_numpy()
    dev, test = _group_folds(x, y, groups)[0]
    folds = _group_folds(x[dev], y[dev], groups[dev], max_splits=4)
    pipeline = Pipeline([('transform', FeatureTransform()), ('scale', StandardScaler()), ('select', SelectKBest(f_classif)),
                         ('svc', SVC(cache_size=512))])
    if param_grid is None:
        common = {'transform__logarithmic': [False, True], 'select__k': [5, 10, 'all'], 'svc__C': [.1, 1, 10],
                  'svc__class_weight': [None, 'balanced']}
        param_grid = [{**common, 'svc__kernel': ['linear']},
                      {**common, 'svc__kernel': ['rbf'], 'svc__gamma': ['scale', .01, .1]}]
    started = perf_counter()
    search = GridSearchCV(pipeline, param_grid, cv=folds, scoring='balanced_accuracy',
                          n_jobs=1, error_score='raise', return_train_score=False)
    search.fit(x[dev], y[dev])
    fitted = search.best_estimator_
    predictions = fitted.predict(x[test])
    margins = fitted.decision_function(x[test])
    tn, fp, fn, tp = confusion_matrix(y[test], predictions, labels=[0, 1]).ravel()
    scaler, selector, svc = (fitted.named_steps[key] for key in ('scale', 'select', 'svc'))
    report = {
        'split_method': 'Subject-disjoint fixed holdout; grouped CV within development subjects only',
        'train_windows': len(dev), 'test_windows': len(test),
        'train_subjects': sorted(np.unique(groups[dev]).tolist()),
        'test_subjects': sorted(np.unique(groups[test]).tolist()),
        'cv_folds': [{'train_subjects': sorted(np.unique(groups[dev][a]).tolist()),
                      'validation_subjects': sorted(np.unique(groups[dev][b]).tolist())} for a, b in folds],
        'test_class_counts': {'normal': int(tn + fp), 'apnea': int(tp + fn)},
        'accuracy': float(accuracy_score(y[test], predictions)),
        'balanced_accuracy': float(balanced_accuracy_score(y[test], predictions)),
        'sensitivity': float(recall_score(y[test], predictions, zero_division=0)),
        'specificity': float(tn / (tn + fp)),
        'precision': float(precision_score(y[test], predictions, zero_division=0)),
        'f1': float(f1_score(y[test], predictions, zero_division=0)),
        'roc_auc': float(roc_auc_score(y[test], margins)),
        'average_precision': float(average_precision_score(y[test], margins)),
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
        'majority_baseline_accuracy': float(max(tn + fp, tp + fn) / len(test)),
        'development_cv_balanced_accuracy': float(search.best_score_),
        'development_cv_std': float(search.cv_results_['std_test_score'][search.best_index_]),
        'best_parameters': {key: str(value) if key == 'scale' else value.__name__ if callable(value) else value
                            for key, value in search.best_params_.items()},
        'training_seconds': float(perf_counter() - started),
        'support_vectors': int(len(svc.support_vectors_)),
        'development_cv_candidates': len(search.cv_results_['params']),
        'holdout_subject_metrics': [{
            'subject': subject,
            'windows': int(np.sum(groups[test] == subject)),
            'accuracy': float(accuracy_score(y[test][groups[test] == subject], predictions[groups[test] == subject])),
            'confusion_matrix': confusion_matrix(y[test][groups[test] == subject], predictions[groups[test] == subject], labels=[0, 1]).tolist(),
        } for subject in np.unique(groups[test])],
        'evaluation_note': 'Window-level point estimates on a small cohort; correlated windows are not independent patients. Test subjects were not used for tuning or refitting.',
    }
    artifact = {
        'schema_version': 1, 'features': list(FEATURE_NAMES), 'classes': [0, 1],
        'input_transform': (fitted.named_steps['transform'].mode or
                            ('log1p' if fitted.named_steps['transform'].logarithmic else 'identity')),
        'selected_indices': np.flatnonzero(selector.get_support()).tolist(),
        'mean': (scaler.mean_ if hasattr(scaler, 'mean_') else scaler.center_).tolist(),
        'scale': scaler.scale_.tolist(),
        'scaler_type': type(scaler).__name__,
        'kernel': svc.kernel, 'gamma': float(svc._gamma),
        'support_vectors': svc.support_vectors_.tolist(),
        'dual_coef': svc.dual_coef_[0].tolist(), 'intercept': float(svc.intercept_[0]),
        'feature_min': x[dev].min(axis=0).tolist(), 'feature_max': x[dev].max(axis=0).tolist(),
        'demo_feature_profiles': x[dev][np.random.default_rng(42).choice(len(dev), min(24, len(dev)), replace=False)].tolist(),
        'contract': contract, 'source': source, 'created_at': datetime.now(timezone.utc).isoformat(),
        'training_data_sha256': hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
        'report': report,
    }
    portable = predict_features(artifact, features.iloc[test])
    if not np.allclose(portable['decision_margin'], margins, atol=1e-7):
        raise RuntimeError('Portable SVM scores disagree with the evaluated estimator.')
    timings = []
    for _ in range(30):
        started = perf_counter()
        predict_features(artifact, features.iloc[test[:1]])
        timings.append(1000 * (perf_counter() - started))
    report['single_window_latency_ms_median'] = float(np.median(timings))
    report['single_window_latency_ms_p95'] = float(np.percentile(timings, 95))
    return artifact


def predict_features(model: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    values = validate_features(frame).to_numpy()
    selected = np.asarray(model['selected_indices'], dtype=int)
    transformed = transform_features(values, model['input_transform'])
    standardized = ((transformed - np.asarray(model['mean'])) / np.asarray(model['scale']))[:, selected]
    support = np.asarray(model['support_vectors'])
    dual = np.asarray(model['dual_coef'])
    scores = []
    for start in range(0, len(standardized), 128):
        block = standardized[start:start + 128]
        if model['kernel'] == 'linear':
            margin = block @ (dual @ support) + model['intercept']
        else:
            distances = np.maximum(0, (block ** 2).sum(axis=1)[:, None]
                                   + (support ** 2).sum(axis=1) - 2 * block @ support.T)
            margin = np.exp(-model['gamma'] * distances) @ dual + model['intercept']
        scores.extend(margin.tolist())
    scores = np.asarray(scores)
    if not np.isfinite(scores).all():
        raise ValueError("Feature magnitudes exceed the model's numerical range.")
    return pd.DataFrame({
        'window': np.arange(1, len(values) + 1), 'prediction': (scores > 0).astype(int),
        'classification': np.where(scores > 0, 'Apnea', 'Normal'), 'decision_margin': scores,
        'outside_training_range': ((values < np.asarray(model['feature_min']))
                                   | (values > np.asarray(model['feature_max']))).any(axis=1),
    })


def load_model(path: Path = MODEL_PATH) -> dict[str, Any]:
    model = json.loads(path.read_text())
    try:
        if model['schema_version'] != 1 or model['features'] != list(FEATURE_NAMES) or model['classes'] != [0, 1]:
            raise ValueError('Incompatible EEG model schema or feature order.')
        if model['input_transform'] not in TRANSFORMS:
            raise ValueError('Invalid EEG model input transform.')
        selected = np.asarray(model['selected_indices'])
        if selected.ndim != 1 or not len(selected) or not np.issubdtype(selected.dtype, np.integer):
            raise ValueError('Invalid selected EEG features.')
        if len(set(selected)) != len(selected) or np.any((selected < 0) | (selected >= 15)):
            raise ValueError('Invalid selected EEG feature indices.')
        for key in ('mean', 'scale', 'feature_min', 'feature_max'):
            array = np.asarray(model[key], dtype=float)
            if array.shape != (15,) or not np.isfinite(array).all():
                raise ValueError(f'Invalid EEG model {key}.')
        if np.any(np.asarray(model['scale']) <= 0) or model['kernel'] not in ('linear', 'rbf'):
            raise ValueError('Invalid EEG model scale or kernel.')
        support = np.asarray(model['support_vectors'], dtype=float)
        dual = np.asarray(model['dual_coef'], dtype=float)
        if support.ndim != 2 or support.shape[1] != len(selected) or dual.shape != (len(support),) or not len(support):
            raise ValueError('Invalid EEG model support vectors.')
        if not np.isfinite(support).all() or not np.isfinite(dual).all() or not np.isfinite([model['gamma'], model['intercept']]).all() or model['gamma'] <= 0:
            raise ValueError('Non-finite EEG model parameters.')
        if not set(REFERENCE_CONTRACT).issubset(model['contract']) or not isinstance(model['report'], dict):
            raise ValueError('Missing EEG feature contract or evaluation.')
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError('Incomplete EEG model artifact.') from exc
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the 15-feature EEG SVM from real labeled rows.')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    artifact = train_svm(pd.read_csv(args.data), json.loads(args.contract.read_text()), args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, allow_nan=False))
    print(json.dumps(artifact['report'], indent=2))
