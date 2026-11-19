"""Reproduce EEG training features from the original MIT-BIH PSG data.

The runtime application never processes raw EEG. This offline Python reference
stands in for the FPGA until the two feature implementations have been compared.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from eeg_apnea import FEATURE_NAMES, REFERENCE_CONTRACT, extract_features, reference_filter

SOURCE = 'https://physionet.org/content/slpdb/1.0.0/'
FILES_URL = 'https://physionet.org/files/slpdb/1.0.0/'
# C3-O1 montage only. slp45/slp41 have no apnea annotations. Other montages excluded.
RECORDS = ('slp03', 'slp04', 'slp14', 'slp16', 'slp48', 'slp59', 'slp60', 'slp61', 'slp66', 'slp67x')
APNEA_CODES = {'OA', 'X', 'CA', 'CAA'}
SLEEP_CODES = {'1', '2', '3', '4', 'R'}
KNOWN_CODES = SLEEP_CODES | APNEA_CODES | {'W', 'H', 'HA', 'L', 'LA', 'A', 'MT'}


def annotation_label(note: str) -> int | None:
    """0=normal sleep, 1=apnea; skip wake, hypopnea-only and unknown epochs."""
    codes = set(note.replace('\x00', '').split())
    if not codes or codes - KNOWN_CODES or not codes.intersection(SLEEP_CODES):
        return None
    if 'W' in codes or 'MT' in codes:
        return None
    if codes.intersection(APNEA_CODES):
        return 1
    if codes.intersection({'H', 'HA'}):
        return None
    return 0


def download(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    def one(name):
        target = raw_dir / name
        if target.exists():
            return
        partial = target.with_suffix(target.suffix + '.part')
        with urlopen(FILES_URL + name, timeout=120) as response, partial.open('wb') as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        partial.replace(target)
        print(f'Downloaded {name}', flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, [f'{record}.{ext}' for record in RECORDS for ext in ('hea', 'st', 'dat')]))


def prepare(raw_dir: Path, output_dir: Path) -> None:
    import wfdb  # Offline dependency only; see requirements-training.txt.
    rows, manifests = [], []
    for name in RECORDS:
        path = str(raw_dir / name)
        header = wfdb.rdheader(path)
        if header.fs != 250 or 'EEG (C3-O1)' not in header.sig_name:
            raise ValueError(f'{name}: EEG channel or sample rate differs from the feature contract.')
        channel = header.sig_name.index('EEG (C3-O1)')
        recording = wfdb.rdrecord(path, channels=[channel])
        units = recording.units[0]
        factors = {'mV': 1000.0, 'uV': 1.0, 'V': 1_000_000.0}
        if units not in factors:
            raise ValueError(f'Unknown EEG unit: {units}')
        filtered = reference_filter(recording.p_signal[:, 0] * factors[units])
        annotations = wfdb.rdann(path, 'st')
        counts = Counter()
        previous_end = 0
        for start, note in zip(annotations.sample, annotations.aux_note):
            label = annotation_label(note)
            if label is None:
                counts['excluded_annotation'] += 1
                continue
            start = int(start)
            end = start + 7500
            # Exclude startup filter transient and incomplete windows.
            if start < 7500 or end > len(filtered):
                counts['excluded_boundary'] += 1
                continue
            if start < previous_end:
                raise ValueError(f'{name}: overlapping annotated windows.')
            previous_end = end
            try:
                features = extract_features(filtered[start:end])
            except ValueError:
                counts['excluded_unusable'] += 1
                continue
            rows.append({'subject_id': name, 'record_id': name, 'start_sample': start,
                         'label': label, 'source': 'MIT-BIH PSG real EEG', **features})
            counts['apnea' if label else 'normal'] += 1
        manifests.append({'record': name, 'counts': dict(counts), 'files': {
            ext: hashlib.sha256((raw_dir / f'{name}.{ext}').read_bytes()).hexdigest()
            for ext in ('hea', 'st', 'dat')}})
        print(name, dict(counts), flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / 'eeg_features.csv', index=False)
    contract = {**REFERENCE_CONTRACT,
                'label_rule': 'Sleep epochs only. OA/X/CA/CAA=1; sleep without respiratory codes=0. Hypopnea-only, wake/movement/unknown and startup epochs excluded.',
                'source_annotation_alignment': 'Each .st annotation applies to the following 30 seconds; use its exact sample index.'}
    (output_dir / 'feature_contract.json').write_text(json.dumps(contract, indent=2))
    (output_dir / 'provenance.json').write_text(json.dumps({
        'source': SOURCE, 'records': manifests,
        'selection': 'All C3-O1 records with apnea annotations; montage and annotation availability decided before model evaluation.',
        'excluded_records': {'slp45': 'Apnea annotations unavailable', 'slp41': 'Apnea annotations unavailable and different montage',
                             'slp01a/slp01b/slp02a/slp02b/slp32/slp37': 'Different EEG montage'},
        'attribution': 'Moody G. MIT-BIH Polysomnographic Database v1.0.0. PhysioNet (1999). doi:10.13026/C23K5S. Ichimaru Y, Moody GB. Psychiatry Clin Neurosci 53:175-177 (1999).',
    }, indent=2))
    # A real, unlabelled feature example for demonstrating the transport format.
    pd.DataFrame(rows[:3]).loc[:, FEATURE_NAMES].to_csv('examples/fpga_features_recorded.csv', index=False)
    print(f'Saved {len(rows)} real EEG windows from {len(RECORDS)} subjects.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=Path('data/raw/slpdb'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed/slpdb'))
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if args.download:
        download(args.raw_dir)
    prepare(args.raw_dir, args.output_dir)
