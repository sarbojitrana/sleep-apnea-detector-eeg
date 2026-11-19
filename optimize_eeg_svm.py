"""Reproducible development-only EEG experiments; never changes the deployed model.

This separates trying improvements from the previously inspected holdout. Synthetic
rows are rejected and every CV fold holds out complete subjects. Search/threshold
scores are development scores, not an independent performance estimate.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, make_scorer, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

from eeg_apnea import FEATURE_NAMES, MODEL_PATH, FeatureTransform, _group_folds, load_model, prepare_training_data


def mobility(x, y): return np.tile([0., 1., 0.], 5)
def energy(x, y): return np.tile([1., 0., 0.], 5)
def kurtosis(x, y): return np.tile([0., 0., 1.], 5)
def energy_mobility(x, y): return np.tile([1., 1., 0.], 5)
def mobility_kurtosis(x, y): return np.tile([0., 1., 1.], 5)


def subject_class_weights(labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Equal total weight per subject/class, using training subjects only."""
    if len(labels) != len(groups) or not len(labels):
        raise ValueError('Weights require matching non-empty labels and subject IDs.')
    result = np.zeros(len(labels))
    for subject in np.unique(groups):
        subject_labels = np.unique(labels[groups == subject])
        for label in subject_labels:
            mask = (groups == subject) & (labels == label)
            result[mask] = len(labels) / (len(np.unique(groups)) * len(subject_labels) * mask.sum())
    return result


def classification_metrics(y, predictions):
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {key: float(value) for key, value in dict(
        accuracy=accuracy_score(y, predictions), balanced_accuracy=balanced_accuracy_score(y, predictions),
        precision=precision_score(y, predictions, zero_division=0),
        sensitivity=recall_score(y, predictions, zero_division=0), specificity=tn / max(1, tn + fp),
        f1=f1_score(y, predictions, zero_division=0)).items()}


def worst_metric(y, predictions):
    return min(classification_metrics(y, predictions).values())


def serializable_params(params):
    return {key: value.__name__ if callable(value) else str(value) if key == 'scale' else value
            for key, value in params.items()}


def summarize_search(search):
    scores = search.cv_results_['mean_test_balanced_accuracy']
    return [{
        'balanced_accuracy': float(scores[index]),
        'parameters': serializable_params(search.cv_results_['params'][index]),
        **{metric: float(search.cv_results_[f'mean_test_{metric}'][index])
           for metric in ('precision', 'sensitivity', 'f1', 'worst_metric')
           if f'mean_test_{metric}' in search.cv_results_},
    } for index in np.argsort(-scores)[:12]]


def run_experiments(frame, baseline, candidates=144, jobs=2):
    # Filter before validation, feature transformations, weighting or model fitting.
    dev = frame.loc[frame.subject_id.isin(baseline['report']['train_subjects'])].reset_index(drop=True)
    x_frame, y, groups = prepare_training_data(dev)
    x = x_frame.to_numpy()
    folds = _group_folds(x, y, groups, max_splits=4)
    pipeline = Pipeline([('transform', FeatureTransform()), ('scale', StandardScaler()),
                         ('select', SelectKBest(f_classif)), ('svc', SVC(cache_size=256))])
    params = {
        'transform__mode': ['log1p', 'relative', 'log_relative', 'centered_log'],
        'scale': [StandardScaler(), RobustScaler(), RobustScaler(quantile_range=(10, 90))],
        'select__k': [5, 10, 'all'], 'svc__C': [.01, .1, 1, 10, 100],
        'svc__gamma': [.001, .01, .1, 'scale'], 'svc__kernel': ['rbf'],
        'svc__class_weight': [None, 'balanced', {0: 1, 1: 1.5}, {0: 1, 1: 2}],
    }
    search = RandomizedSearchCV(pipeline, params, n_iter=candidates, random_state=42, cv=folds,
        n_jobs=jobs, scoring={'balanced_accuracy': 'balanced_accuracy', 'worst_metric': make_scorer(worst_metric)},
        refit='balanced_accuracy', error_score='raise')
    search.fit(x, y)
    margins = cross_val_predict(search.best_estimator_, x, y, cv=folds, method='decision_function', n_jobs=jobs)
    thresholds = np.linspace(-1, 1, 161)
    threshold = thresholds[np.argmax([worst_metric(y, margins > value) for value in thresholds])]
    common = {'transform__mode': ['log1p', 'relative'], 'svc__kernel': ['rbf'],
              'svc__C': [.01, .1, 1], 'svc__gamma': [.001, .01, .1]}
    grid = [{**common, 'select__score_func': [func], 'select__k': [count]} for func, count in (
        (f_classif, 'all'), (mobility, 5), (energy, 5), (kurtosis, 5),
        (energy_mobility, 10), (mobility_kurtosis, 10))]
    weighted = GridSearchCV(pipeline, grid, cv=folds, n_jobs=jobs,
        scoring={'balanced_accuracy': 'balanced_accuracy',
                 'precision': make_scorer(precision_score, zero_division=0),
                 'sensitivity': make_scorer(recall_score, zero_division=0),
                 'f1': make_scorer(f1_score, zero_division=0)},
        refit='balanced_accuracy', error_score='raise')
    weighted.fit(x, y, svc__sample_weight=subject_class_weights(y, groups))
    best = max(float(search.best_score_), float(weighted.best_score_))
    return {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'development_subjects': sorted(np.unique(groups).tolist()),
        'excluded_holdout_subjects': baseline['report']['test_subjects'],
        'baseline_development_cv_balanced_accuracy': baseline['report']['development_cv_balanced_accuracy'],
        'preprocessing_search': summarize_search(search),
        'subject_balanced_search': summarize_search(weighted),
        'candidate_counts': {'preprocessing': candidates, 'subject_balanced': 108},
        'best_development_cv_balanced_accuracy': best,
        'development_cv_change': best - baseline['report']['development_cv_balanced_accuracy'],
        'evaluation_note': 'Development tuning scores require validation on new independent subjects.',
        'pooled_oof_default': classification_metrics(y, margins > 0),
        'threshold_experiment': {'threshold': float(threshold), 'pooled_oof_tuned': classification_metrics(y, margins > threshold),
                                'note': 'Threshold fitted to OOF development scores; these are not unbiased evaluation metrics.'},
        'deployment': 'Existing deployed model retained; no test-score-driven model replacement.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/processed/slpdb/eeg_features.csv'))
    parser.add_argument('--baseline', type=Path, default=MODEL_PATH)
    parser.add_argument('--output', type=Path, default=Path('reports/eeg_optimization.json'))
    parser.add_argument('--candidates', type=int, default=144)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    result = run_experiments(pd.read_csv(args.data), load_model(args.baseline), args.candidates, args.jobs)
    result['baseline_model_sha256'] = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
    result['training_csv_sha256'] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))
