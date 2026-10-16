"""Fixed validation-only pilot fidelity on authenticated existing TabSyn samples."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

DECLARATION_SHA256 = '686742975a461e59ac486997a637ab0e817d89141159a1da0f48adfb6eaeaec0'
SAMPLES = Path('/home/ubuntu/dope-scratch-x1/dope-rf-tabsyn-sample-v1')
WORKERS = Path('/mnt/fast-scratch/dope-benchmark/s3-v1/prepared/worker')


def authenticated(path, expected):
    if path.name == 'test.csv' or path.is_symlink():
        raise ValueError('official test or redirected input refused')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('fidelity input hash differs')
    return raw


def fidelity(validation, synthetic):
    import numpy as np
    from scipy.stats import ks_2samp

    if validation.ndim != 2 or synthetic.ndim != 2 or validation.shape[1] != synthetic.shape[1] or validation.shape[1] < 2 or not np.isfinite(validation).all() or not np.isfinite(synthetic).all():
        raise ValueError('fidelity table dimensions or finite-value check failed')
    ks = [ks_2samp(validation[:, i], synthetic[:, i]).statistic for i in range(validation.shape[1])]
    corr_real = np.nan_to_num(np.corrcoef(validation, rowvar=False))
    corr_synth = np.nan_to_num(np.corrcoef(synthetic, rowvar=False))
    off_diagonal = ~np.eye(validation.shape[1], dtype=bool)
    return {'marginal_ks_mean': float(np.mean(ks)),
            'pair_correlation_fidelity': float(1 - np.abs(corr_real - corr_synth)[off_diagonal].mean() / 2)}


def score_cell(job):
    import io
    import importlib.util
    from importlib.machinery import SourceFileLoader
    import numpy as np

    cell, declaration, implementation = job
    dataset, size, seed = cell['dataset'], cell['size'], cell['sample_seed']
    validation_path = WORKERS / dataset / 'validation.csv'
    sample_path = SAMPLES / dataset / f'sample-{seed}-{size}n.csv'
    if validation_path.resolve() != WORKERS.resolve() / dataset / 'validation.csv' or sample_path.resolve() != SAMPLES.resolve() / dataset / f'sample-{seed}-{size}n.csv':
        raise ValueError('redirected fidelity partition refused')
    validation_sha = declaration['validation_sha256_by_dataset'][dataset]
    validation = np.loadtxt(io.BytesIO(authenticated(validation_path, validation_sha)), delimiter=',', ndmin=2)
    sample = np.loadtxt(io.BytesIO(authenticated(sample_path, cell['synthetic_sha256'])), delimiter=',', skiprows=1, ndmin=2)
    if len(validation) != cell['rows']['validation'] or len(sample) != cell['rows']['synthetic']:
        raise ValueError('fidelity row counts differ from bound utility receipt')
    captured = authenticated(implementation, declaration['expanded_implementation_sha256'])
    loader = SourceFileLoader('fixed_expanded_metrics', str(implementation))
    loader.get_code = lambda _name: loader.source_to_code(captured, str(implementation))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    expanded = importlib.util.module_from_spec(spec)
    loader.exec_module(expanded)
    detector = expanded.c2st(validation, sample, 'catboost')
    alpha = expanded.alpha_beta(validation, sample)
    return {'dataset': dataset, 'size': size, 'sample_seed': seed,
            'official_tests_opened': False, 'formal_dp': False,
            'synthetic_sha256': cell['synthetic_sha256'], 'validation_sha256': validation_sha,
            'status': 'ok', **fidelity(validation, sample),
            'c2st_catboost_auc': detector['value'], 'c2st_status': detector['status'],
            'c2st_rows_per_class': detector.get('rows_per_class'),
            'c2st_reason': detector.get('reason'),
            'alpha_beta_status': alpha['status'], 'alpha_beta_reason': alpha.get('reason'),
            'alpha_precision': (alpha.get('value') or {}).get('alpha_precision'),
            'beta_recall': (alpha.get('value') or {}).get('beta_recall')}


def main():
    from concurrent.futures import ProcessPoolExecutor
    import catboost
    import numpy as np
    import scipy
    import sklearn

    parser = argparse.ArgumentParser()
    parser.add_argument('--declaration', required=True, type=Path)
    parser.add_argument('--ledger', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--implementation', required=True, type=Path)
    args = parser.parse_args()
    if os.uname().nodename != 'xbabe1':
        raise ValueError('fidelity row computation is restricted to xbabe1')
    declaration = json.loads(authenticated(args.declaration, DECLARATION_SHA256))
    if declaration['official_tests_opened'] is not False or declaration['formal_dp'] is not False:
        raise ValueError('official test or formal privacy flag refused')
    if declaration['runtime'] != {'catboost': catboost.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__}:
        raise ValueError('diagnostic runtime differs from the committed declaration')
    authenticated(args.implementation, declaration['expanded_implementation_sha256'])
    raw = authenticated(args.ledger, declaration['scalar_ledger_sha256'])
    cells = [json.loads(line) for line in raw.splitlines()]
    seen = set()
    for cell in cells:
        dataset, size, seed = cell['dataset'], cell['size'], cell['sample_seed']
        key = (dataset, size, seed)
        if cell['official_tests_opened'] is not False or key in seen or not re.fullmatch('[0-9a-f]{16}', dataset) or size not in (1, 4) or seed not in (101, 211, 307):
            raise ValueError('fidelity cell identity or official test flag refused')
        seen.add(key)
    jobs = [(cell, declaration, args.implementation) for cell in cells]
    with ProcessPoolExecutor(max_workers=declaration['workers']) as pool:
        records = []
        for record in pool.map(score_cell, jobs):
            records.append(record)
            if len(records) % 30 == 0:
                print('finished diagnostic cells', len(records), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = ''.join(json.dumps(c, sort_keys=True, separators=(',', ':')) + '\n' for c in records)
    args.out.write_text(text)
    meta = {'format': 'dope-wave2-tabsyn-fidelity-cells-v1', 'cell_count': len(records),
            'official_tests_opened': False, 'formal_dp': False,
            'declaration_sha256': DECLARATION_SHA256,
            'producer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'ledger_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'numpy': np.__version__, 'scipy': scipy.__version__, 'catboost': catboost.__version__,
            'scikit-learn': sklearn.__version__,
            'expanded_implementation_sha256': declaration['expanded_implementation_sha256']}
    args.out.with_suffix('.meta.json').write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')
    print('TabSyn fidelity cells', len(records), meta['ledger_sha256'])


if __name__ == '__main__':
    main()
