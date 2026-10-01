"""Common validation outcomes for the frozen GaussianCopula pilot matrix."""

import hashlib
import json
from pathlib import Path

from research.benchmark import pilot_metrics

ROOT = Path('/mnt/fast-scratch/dope-benchmark/pilot-24h')
RUN = ROOT / 'copula-matrix-v1'
LOCK = RUN / 'validation-v2.lock.json'
PARENT = RUN / 'validation.lock.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')


def main():
    lock = json.loads(LOCK.read_text())
    if (lock['script_sha256'] != sha(Path(__file__))
            or lock['metric_sha256'] != sha(Path(pilot_metrics.__file__))
            or lock['matrix_sha256'] != sha(RUN / 'matrix.lock.json')
            or lock['parent_validation_lock_sha256'] != sha(PARENT)):
        raise ValueError('copula validation source changed')
    matrix = json.loads((RUN / 'matrix.lock.json').read_text())
    if len(matrix['jobs']) != 12:
        raise ValueError('copula matrix incomplete')
    fit_paths = list((RUN / 'results').glob('*/fit-receipt.json'))
    attempt_paths = list((RUN / 'results').glob('*/fit-attempts/attempt-*.json'))
    for entry in matrix['jobs']:
        job_path = Path(entry['job_path'])
        if sha(job_path) != entry['job_sha256']:
            raise ValueError('copula job changed')
        job = json.loads(job_path.read_text())
        worker = Path(job['worker_dir'])
        manifest = json.loads((worker / 'worker-manifest.json').read_text())
        if (manifest['dataset_id'] != entry['dataset']
                or sha(worker / 'train.csv') != manifest['projected_hashes']['train']
                or sha(worker / 'validation.csv') != manifest['projected_hashes']['validation']):
            raise ValueError('copula worker changed')
        matches = []
        for path in attempt_paths:
            attempt = json.loads(path.read_text())
            identity = attempt['fit_identity']
            if (identity['dataset'] == entry['dataset']
                    and identity['method'] == 'GaussianCopula'
                    and identity['fit_seed'] == entry['seed']
                    and identity['configuration']['kind'] == entry['kind']):
                matches.append((path, attempt))
        if len(matches) != 1:
            raise ValueError(f'copula fit attempt missing or ambiguous: {entry["name"]}')
        attempt_path, attempt = matches[0]
        fit_path = attempt_path.parents[1] / 'fit-receipt.json'
        if attempt['status'] != 'ok':
            if fit_path.exists():
                raise ValueError('failed copula fit has success receipt')
            failure = {'format': 'dope-24h-copula-validation-failure', 'version': 2,
                       'job_sha256': entry['job_sha256'], 'dataset': entry['dataset'],
                       'kind': entry['kind'], 'fit_seed': entry['seed'],
                       'stage': 'fit', 'status': attempt['status'],
                       'fit_attempt_sha256': sha(attempt_path),
                       'validation_lock_sha256': sha(LOCK),
                       'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None}
            failure_path = RUN / 'validation-failures-v2' / f"{entry['name']}.json"
            if failure_path.exists():
                if json.loads(failure_path.read_text()) != failure:
                    raise ValueError('copula fit failure evidence changed')
            else:
                once(failure_path, failure)
            print(json.dumps({'job': entry['name'], 'status': attempt['status']}), flush=True)
            continue
        if fit_path not in fit_paths or json.loads(fit_path.read_text()) != attempt:
            raise ValueError('copula fit receipt changed')
        fit = attempt
        identity = fit['fit_identity']
        selection_sha = job['configuration'].get('selection_sha256')
        if (fit['fit_key'] != fit_path.parent.name
                or identity['configuration'].get('selection_sha256') != selection_sha
                or identity['sample_seeds'] != job['sample_seeds']
                or identity['size_multipliers'] != job['size_multipliers']):
            raise ValueError('copula fit identity changed')
        if sum(item['bytes'] for item in fit['artifact_inventory']) != fit['artifact_bytes']:
            raise ValueError('copula artifact bytes changed')
        for item in fit['artifact_inventory']:
            if sha(fit_path.parent / 'artifact' / item['path']) != item['sha256']:
                raise ValueError('copula artifact changed')
        expected = {(seed, manifest['train_rows'] * multiple)
                    for seed in job['sample_seeds']
                    for multiple in job['size_multipliers']}
        attempts = {}
        for path in (fit_path.parent / 'sample-attempts').glob('*/attempt-0001.json'):
            receipt = json.loads(path.read_text())
            key = (receipt.get('sample_seed'), receipt.get('row_count'))
            if (receipt.get('format') != 'dope-benchmark-sample-receipt'
                    or receipt.get('fit_key') != fit['fit_key']
                    or receipt.get('run_key') != path.parent.name
                    or receipt.get('status') not in ('ok', 'timeout', 'failed')
                    or key not in expected or key in attempts):
                raise ValueError('copula sample attempt changed')
            attempts[key] = (path, receipt)
        if set(attempts) != expected:
            raise ValueError(f'copula sample attempt matrix incomplete: {entry["name"]}')
        samples = {}
        for path in fit_path.parent.glob('*.receipt.json'):
            receipt = json.loads(path.read_text())
            if receipt.get('format') == 'dope-benchmark-sample-receipt':
                key = (receipt['sample_seed'], receipt['row_count'])
                if (key not in attempts or attempts[key][1] != receipt
                        or receipt['status'] != 'ok' or key in samples):
                    raise ValueError('copula successful sample receipt changed')
                samples[key] = (path, receipt)
        if set(samples) != {key for key, (_, row) in attempts.items()
                            if row['status'] == 'ok'}:
            raise ValueError('copula successful sample receipt missing')
        failures = [{'sample_seed': key[0], 'row_count': key[1],
                     'status': row['status'], 'error_type': row.get('error_type'),
                     'sample_attempt_sha256': sha(path)}
                    for key, (path, row) in sorted(attempts.items())
                    if row['status'] != 'ok']
        if failures:
            failure = {'format': 'dope-24h-copula-validation-failure', 'version': 2,
                       'job_sha256': entry['job_sha256'], 'dataset': entry['dataset'],
                       'kind': entry['kind'], 'fit_seed': entry['seed'],
                       'stage': 'sample', 'status': 'partial_or_complete_sample_failure',
                       'fit_receipt_sha256': sha(fit_path), 'failed_samples': failures,
                       'validation_lock_sha256': sha(LOCK),
                       'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None}
            failure_path = RUN / 'validation-failures-v2' / f"{entry['name']}.json"
            if failure_path.exists():
                if json.loads(failure_path.read_text()) != failure:
                    raise ValueError('copula sample failure evidence changed')
            else:
                once(failure_path, failure)
            print(json.dumps({'job': entry['name'], 'sample_failures': len(failures)}),
                  flush=True)
        task = json.loads((worker / 'projection.json').read_text())['task']
        for seed in job['sample_seeds']:
            for multiple in lock['metric_multipliers']:
                key = (seed, manifest['train_rows'] * multiple)
                if key not in samples:
                    continue
                receipt_path, sample = samples[key]
                data_path = receipt_path.with_suffix('').with_suffix('.csv')
                if sha(data_path) != sample['sample_sha256']:
                    raise ValueError('copula sample changed')
                result = RUN / 'validation-metrics' / fit['fit_key'] / f'{sample["run_key"]}.json'
                if result.exists():
                    prior = json.loads(result.read_text())
                    if (prior['sample_receipt_sha256'] != sha(receipt_path)
                            or prior['metric_source_sha256'] != lock['metric_sha256']):
                        raise ValueError('copula validation result changed')
                    continue
                metric = pilot_metrics.measure(worker / 'train.csv', worker / 'validation.csv',
                                               data_path, task, seed)
                once(result, {'format': 'dope-24h-copula-validation', 'version': 1,
                              'job_sha256': entry['job_sha256'],
                              'fit_receipt_path': str(fit_path),
                              'fit_receipt_sha256': sha(fit_path),
                              'sample_receipt_path': str(receipt_path),
                              'sample_receipt_sha256': sha(receipt_path),
                              'metric_source_sha256': lock['metric_sha256'],
                              'validation_lock_sha256': sha(LOCK),
                              'dataset': entry['dataset'], 'method': 'GaussianCopula',
                              'kind': entry['kind'], 'fit_seed': entry['seed'],
                              'sample_seed': seed, 'size_multiplier': multiple,
                              'artifact_bytes': fit['artifact_bytes'], 'metrics': metric,
                              'validation_only': True, 'official_tests_opened': False,
                              'mfs_v2': None, 'ptf_v1': None})
                print(json.dumps({'job': entry['name'], 'seed': seed, 'size': multiple,
                                  'catboost_retention': metric['utility']['catboost']['retention']}),
                      flush=True)


if __name__ == '__main__':
    main()
