"""AIM author's synthetic marginal TV error on training-derived validation.

The fixed public eight-bin, all-pair workload matches the common-numeric AIM
adapter. Validation replaces the author's training-data reference for selection;
this is not an author-faithful categorical benchmark or a formal DP claim.
"""

from __future__ import annotations

from array import array
import csv
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import re


SCRATCH = Path('/mnt/fast-scratch/dope-benchmark')
BINS = 8


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def safe(path: Path) -> Path:
    path = Path(path)
    if (not path.is_absolute() or not path.resolve(strict=True).is_relative_to(SCRATCH)
            or 'evaluator' in path.parts or 'evaluator' in path.resolve().parts
            or path.name == 'test.csv' or path.is_symlink()
            or any(p.is_symlink() for p in path.parents)):
        raise ValueError('AIM native evidence outside training-derived scope')
    return path


def histograms(path: Path, width: int) -> tuple[list[array], int]:
    if path.stat().st_size > 512_000_000:
        raise ValueError('AIM native table exceeds bounded input size')
    pairs = list(combinations(range(width), 2))
    counts = [array('Q', [0]) * (BINS * BINS) for _ in pairs]
    rows = 0
    try:
        with path.open(newline='') as stream:
            for fields in csv.reader(stream):
                try:
                    values = [float(field) for field in fields]
                except (ValueError, OverflowError):
                    raise ValueError('invalid AIM native numeric table') from None
                if (len(values) != width or not all(math.isfinite(v) and 0 <= v <= 1
                                                   for v in values)):
                    raise ValueError('invalid AIM native numeric table')
                codes = [min(int(v * BINS), BINS - 1) for v in values]
                for count, (left, right) in zip(counts, pairs):
                    count[codes[left] * BINS + codes[right]] += 1
                rows += 1
    except (UnicodeError, csv.Error):
        raise ValueError('invalid AIM native numeric table') from None
    if rows == 0:
        raise ValueError('AIM native table has no usable rows')
    return counts, rows


def measure(worker: Path, synthetic: Path, manifest_sha256: str,
            synthetic_sha256: str, epsilon: float, sample_seed: int = 101) -> dict:
    """Minimize mean pairwise TV at one fixed epsilon; never use shared utility."""
    if (isinstance(epsilon, bool) or epsilon not in (1, 4, 10)
            or type(sample_seed) is not int or sample_seed != 101
            or not all(isinstance(h, str) and re.fullmatch('[0-9a-f]{64}', h)
                       for h in (manifest_sha256, synthetic_sha256))):
        raise ValueError('AIM native request differs from frozen objective')
    worker = safe(worker)
    manifest_path = safe(worker / 'worker-manifest.json')
    if (worker / 'test.csv').exists() or sha(manifest_path) != manifest_sha256:
        raise ValueError('AIM worker identity changed')
    manifest = json.loads(manifest_path.read_text())
    projection_path = safe(worker / 'projection.json')
    validation = safe(worker / 'validation.csv')
    synthetic = safe(synthetic)
    if (sha(projection_path) != manifest['projection_sha256']
            or sha(validation) != manifest['projected_hashes']['validation']
            or sha(synthetic) != synthetic_sha256):
        raise ValueError('AIM native input identity changed')
    projection = json.loads(projection_path.read_text())
    features = projection['output_features']
    if (type(features) is not int or not 1 <= features <= 127
            or projection['task'] not in ('binary', 'regression')):
        raise ValueError('AIM native representation is inapplicable')
    real, real_rows = histograms(validation, features + 1)
    generated, generated_rows = histograms(synthetic, features + 1)
    errors = [0.5 * math.fsum(abs(x / real_rows - y / generated_rows)
                             for x, y in zip(a, b)) for a, b in zip(real, generated)]
    value = math.fsum(errors) / len(errors)
    return {
        'name': 'author_mean_synthetic_marginal_tv', 'direction': 'minimize',
        'method': 'AIM', 'objective': 'author_mean_synthetic_marginal_tv',
        'value': value, 'implementation_sha256': sha(Path(__file__)),
        'partition': 'validation', 'partition_source': 'official_training_derived',
        'validation_sha256': manifest['projected_hashes']['validation'],
        'synthetic_sha256': synthetic_sha256, 'sample_seed': sample_seed,
        'epsilon': epsilon, 'domain_bins': BINS, 'workload_pairs': len(errors),
        'reference_rows': real_rows, 'synthetic_rows': generated_rows,
        'representation': 'common_numeric_fixed_public_eight_bin_adapter_outputs',
        'shared_kpi_used_for_selection': False, 'native_values_cross_method_ranking': False,
        'formal_dp_claim': False, 'official_tests_opened': False,
        'mfs_v2': None, 'ptf_v1': None,
    }
