"""Measure sealed SDV samples on the shared validation auditors, without fitting.

The caller supplies a frozen research round and admits a CPU slot. An external
monitor must charge the entire process, including integrity checks, to 600s.
This executable neither imports SDV nor loads a model, and never selects a trial.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import time


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def safe(path, base):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True).is_relative_to(base)
            and not any(p.is_symlink() for p in (path, *path.parents))
            and 'evaluator' not in path.parts and path.name != 'test.csv',
            'evidence escaped the sealed worker scope')
    return path


def inventory(root, expected, base):
    root = safe(root, base)
    require(not any(p.is_symlink() for p in root.rglob('*'))
            and {str(p) for p in root.rglob('*') if p.is_file()} == set(expected),
            'source inventory acquired an alias or unbound file')
    for path, value in expected.items():
        require(sha(safe(path, base)) == value, 'source inventory drifted')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(root, expected, request):
    # This precedes every shared-runtime initializer and metric import.
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU environment required')
    root = Path(root)
    base = root.parent
    safe(root, base)
    require(sha(safe(root / 'round.lock.json', base)) == expected, 'round drifted')
    lock = read(root / 'round.lock.json')
    inventory(root / 'source', lock['source_files'], base)
    require(lock['gpu_operations_enabled'] is False and lock['new_generator_fits_started'] == 0
            and lock['native_selection_changed'] is False and lock['official_tests_opened'] is False
            and all(lock[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')),
            'research scope acquired an execution or gated claim')
    request = safe(request, base)
    require(request.is_relative_to(root / 'attempts'), 'request escaped the round')
    req = read(request)
    job = req['job']
    require(req['round_sha256'] == expected
            and sum(digest(j) == digest(job) for j in lock['jobs']) == 1
            and type(job['fit_seed']) is int and job['fit_seed'] == 11
            and job['final'] is False and job['track'] == 'common-numeric'
            and job['method'] in ('CTGAN', 'TVAE'), 'request differs from the frozen job')
    parent = safe(lock['parent_receipt_lock_path'], base)
    require(sha(parent) == lock['parent_receipt_lock_sha256'], 'sealed parent drifted')
    custody = read(parent)

    def evidence(path, value):
        path = safe(path, base)
        require(custody['refs'].get(str(path)) == value and sha(path) == value,
                'parent evidence differs from its immutable receipt lock')
        return path

    reconciliation = safe(lock['parent_reconciliation_path'], base)
    require(sha(reconciliation) == custody['reconciliation_sha256']
            == lock['parent_reconciliation_sha256'], 'native selection receipt drifted')
    report = read(reconciliation)
    require(report['complete_native_matrix'] is True and report['new_trials_closed'] == 704
            and report['shared_kpi_used_for_selection'] is False
            and report['sampling_success_used_for_selection'] is False,
            'native selection was incomplete or used shared outcomes')
    cells = [r for r in report['default_native_cells']
             if r['dataset'] == job['dataset'] and r['method'] == job['method']]
    require(len(cells) == 1 and cells[0]['all_four_trials_closed'] is True,
            'native selection lineage is missing or duplicated')
    bindings = job['selection_bindings']
    require(isinstance(bindings, list) and bindings and len(bindings) == len(set(bindings))
            and set(bindings).issubset({'author_default', 'native_selected'}), 'selection binding changed')
    for name in bindings:
        trial = cells[0]['default_trial' if name == 'author_default' else 'native_selected_trial']
        require(trial is not None and trial['job_sha256'] == job['original_job_sha256']
                and trial['receipt_sha256'] == job['native_receipt_sha256']
                and trial['common_samples_complete'] is True, 'shared results changed the native winner')
    original_round = read(evidence(parent.parent / 'round.lock.json', custody['round_sha256']))
    inventory(parent.parent / 'source', original_round['source_files'], base)
    for path, value in original_round['source_files'].items():
        evidence(path, value)
    receipt_path = evidence(job['native_receipt_path'], job['native_receipt_sha256'])
    receipt = read(receipt_path)
    original = receipt['job']
    require(receipt['status'] == 'ok' and digest(original) == job['original_job_sha256']
            and receipt['round_sha256'] == custody['round_sha256']
            and type(original['fit_seed']) is int and original['fit_seed'] == 11
            and original['final'] is False and original['method'] == job['method']
            and original['dataset'] == job['dataset'] and digest(original['worker']) == digest(job['worker']),
            'original fit or projected lineage changed')
    for name, value in receipt['evidence_files'].items():
        require(Path(name).name == name, 'original evidence name escaped its attempt')
        evidence(receipt_path.parent / name, value)
    worker = safe(job['worker']['path'], base)
    require(set(job['worker']['files']) == {'train.csv', 'validation.csv', 'projection.json',
            'row-group-assignments.json', 'worker-manifest.json'}
            and {p.name for p in worker.iterdir()} == set(job['worker']['files'])
            and not any(p.is_symlink() for p in worker.rglob('*')), 'worker seal changed')
    for name, value in job['worker']['files'].items():
        evidence(worker / name, value)
    fit = read(evidence(receipt_path.parent / 'fit.json', job['fit_receipt_sha256']))
    artifact = safe(job['artifact_path'], base)
    require(artifact == receipt_path.parent / 'artifact'
            and fit['status'] == 'ok' and fit['job_sha256'] == job['original_job_sha256']
            and fit['artifact_bytes'] == job['artifact_bytes']
            and type(job['artifact_bytes']) is int and job['artifact_bytes'] > 0
            and job['projection_bytes_included'] is True and fit['projection_bytes_included'] is True,
            'artifact charge or fit evidence changed')
    rows = fit['artifact_inventory']
    require(sorted(r['path'] for r in rows) == ['model.json', 'model.pt', 'projection.json']
            and not any(p.is_symlink() for p in artifact.rglob('*'))
            and {p.relative_to(artifact).as_posix() for p in artifact.rglob('*') if p.is_file()}
            == {r['path'] for r in rows}, 'artifact contains an uncharged file or alias')
    for row in rows:
        path = evidence(artifact / row['path'], row['sha256'])
        require(type(row['bytes']) is int and path.stat().st_size == row['bytes'], 'artifact size changed')
    require(sum(r['bytes'] for r in rows) == job['artifact_bytes']
            and sha(artifact / 'projection.json') == job['worker']['files']['projection.json'],
            'projection bytes were omitted')
    sample = read(evidence(receipt_path.parent / 'sample.json', job['sample_receipt_sha256']))
    schedule = [{'multiplier': size, 'name': ('n' if size == 1 else '4n') + f'-seed{seed}', 'seed': seed}
                for size in (1, 4) for seed in (101, 211, 307)]
    require(sample['status'] == 'ok' and sample['sample_replay_exact'] is True
            and sample['job_sha256'] == job['original_job_sha256']
            and [r['schedule'] for r in sample['samples']] == schedule + [{'multiplier': 1, 'name': 'repeat', 'seed': 101}]
            and digest(job['samples']) == digest(sample['samples'][:6]), 'original sample schedule changed')
    require(sha(receipt_path.parent / 'repeat.csv') == sha(receipt_path.parent / 'n-seed101.csv'),
            'original exact sample replay changed')
    for row in sample['samples']:
        evidence(receipt_path.parent / (row['schedule']['name'] + '.csv'), row['sample_sha256'])
        require(row['rows'] == job['worker']['train_rows'] * row['schedule']['multiplier'],
                'sample row count differs from the projected training partition')
    for path, value in lock['runtime_lock_files'].items():
        require(sha(safe(path, base)) == value, 'shared auditor runtime drifted')
    runtime = load('sealed_shared_runtime', root / 'source/shared_runtime.py').verify()
    require(runtime['metric_sha256'] == lock['metric_source_sha256'], 'shared metric changed')
    return lock, job, worker, receipt_path.parent, runtime


def main():
    start = time.monotonic()
    root, expected, request = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    lock, job, worker, source, runtime = prepare(root, expected, request)
    metrics = load('frozen_shared_metric', Path(runtime['metric_source']))
    output = request.parent
    samples = []
    for row in job['samples']:
        item = row['schedule']
        path = source / (item['name'] + '.csv')
        metric = metrics.measure(worker / 'train.csv', worker / 'validation.csv', path, 'regression', item['seed'])
        require(metric['dependencies'] == runtime['expected_versions'] and metric['mfs_v2'] is None
                and metric['gate_profile_complete'] is False and set(metric['utility']) == {'linear', 'catboost', 'mlp'}
                and metric['rows']['synthetic'] == row['rows'], 'shared metric contract changed')
        repeated = job['metric_replay_required'] and item['multiplier'] == 4 and item['seed'] == 101
        if repeated:
            replay = metrics.measure(worker / 'train.csv', worker / 'validation.csv', path, 'regression', item['seed'])
            without_time = lambda value: {k: v for k, v in value.items() if k != 'metric_seconds'}
            require(without_time(metric) == without_time(replay), 'shared metric replay changed')
        name = item['name'] + '.metric.json'
        with (output / name).open('x') as stream:
            json.dump(metric, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
        require(sha(path) == row['sample_sha256'], 'sample changed during measurement')
        samples.append({'sample_seed': item['seed'], 'size_multiplier': item['multiplier'], 'rows': row['rows'],
                        'sample_file': str(path), 'sample_sha256': row['sample_sha256'], 'metric_file': name,
                        'metric_sha256': sha(output / name), 'sample_replay': 'original_exact',
                        'metric_replay': 'exact' if repeated else 'not_repeated'})
    prepare(root, expected, request)  # Rehash before final success and deadline checks.
    require(time.monotonic() - start < lock['metric_batch_timeout_seconds'], 'whole batch deadline exceeded')
    with (output / 'batch.json').open('x') as stream:
        json.dump({'job_sha256': digest(job), 'samples': samples, 'sample_replays_exact': True,
                   'metric_source_sha256': lock['metric_source_sha256'], 'elapsed_seconds': time.monotonic() - start,
                   'new_generator_fits_started': 0, 'new_samples_generated': 0, 'native_selection_changed': False,
                   'global_family_selected': False, 'official_tests_opened': False, 'gate_profile_complete': False,
                   'historical_runtime_closure_upgraded': False, 'mfs_v2': None, 'ptf_v1': None},
                  stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
