"""Verify SDV research receipts and select only by the frozen native objective."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

from .manifest import digest
from .score import sha256

PRIOR_ROUND_SHA = '5634d235eb8305436ca046867d96d09f562fabbd360892c580a7da3b9b811988'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def safe(path, base):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True).is_relative_to(base)
            and not any(p.is_symlink() for p in (path, *path.parents))
            and 'evaluator' not in path.parts and path.name != 'test.csv',
            'native evidence path outside worker scope')
    return path


def evidence(path, expected, base, refs):
    path = safe(path, base)
    require(sha256(path) == expected and (str(path) not in refs or refs[str(path)] == expected),
            'native evidence digest changed')
    refs[str(path)] = expected
    return path


def read(path):
    return json.loads(Path(path).read_text())


def claims(row):
    require(row['official_tests_opened'] is False and row['mfs_v2'] is None
            and row['ptf_v1'] is None, 'native receipt acquired a gated claim')


def worker_files(worker, base, refs):
    directory = safe(worker['path'], base)
    require(set(worker['files']) == {'train.csv', 'validation.csv', 'projection.json',
                                    'row-group-assignments.json', 'worker-manifest.json'}
            and {p.name for p in directory.iterdir()} == set(worker['files'])
            and not any(p.is_symlink() for p in directory.rglob('*')),
            'native worker inventory or test seal changed')
    for name, expected in worker['files'].items():
        evidence(directory / name, expected, base, refs)


def artifact(directory, inventory, charged, worker, base, refs):
    require(type(charged) is int and charged > 0 and isinstance(inventory, list)
            and sorted(r['path'] for r in inventory) == ['model.json', 'model.pt', 'projection.json'],
            'native artifact inventory or byte charge changed')
    directory = Path(directory)
    require(not any(p.is_symlink() for p in (directory, *directory.parents))
            and not any(p.is_symlink() for p in directory.rglob('*'))
            and {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
            == {r['path'] for r in inventory}, 'native artifact gained an uncharged file or alias')
    for row in inventory:
        path = evidence(directory / row['path'], row['sha256'], base, refs)
        require(type(row['bytes']) is int and path.stat().st_size == row['bytes'],
                'native artifact file size changed')
    require(charged == sum(r['bytes'] for r in inventory)
            and next(r['sha256'] for r in inventory if r['path'] == 'projection.json')
            == worker['files']['projection.json'], 'native projection byte charge or lineage changed')


def native_metric(metric, worker, objective, lock):
    require(metric['objective'] == objective['name'] == 'sdmetrics_mean_regression_r2'
            and metric['direction'] == objective['direction'] == 'maximize'
            and metric['partition'] == 'validation'
            and metric['implementation_sha256'] == objective['implementation_sha256'] == lock['adapter_sha256']
            and metric['validation_sha256'] == worker['files']['validation.csv']
            and metric['seed'] == lock['native_metric_seed'] == 1729,
            'native metric source, objective or validation lineage changed')
    components = metric['components']
    require(set(components) == {'LinearRegression', 'MLPRegressor'}
            and all(finite(v) for v in components.values()) and finite(metric['value'])
            and math.isclose(metric['value'], sum(components.values()) / 2, rel_tol=0, abs_tol=1e-14),
            'native metric aggregation changed')
    return metric['value']


def operation(mode, row, parent, lock, base, refs):
    require(type(row['new_operation_started']) is bool and finite(row['elapsed_seconds'])
            and row['elapsed_seconds'] >= 0, 'native operation cost missing')
    if row['new_operation_started'] is False:
        require(row['status'] == 'deadline_unstarted' and row['elapsed_seconds'] == 0,
                'unstarted operation acquired compute or success')
        return 0.0
    proof = parent / (mode + '.operation.json')
    require(read(proof) == row, 'native operation receipt differs')
    claims_flag = row.get('official_tests_opened')
    require(claims_flag is False and row['host'] in lock['allowed_gpu_hosts']
            and row['host'] == 'xbabe1' and type(row['timeout_seconds']) is int
            and 0 < row['timeout_seconds'] <= lock['whole_operation_timeout_seconds'] == 600,
            'native operation host or deadline changed')
    request = parent / (mode + '.request.json')
    evidence(request, row['request_sha256'], base, refs)
    evidence(parent / (mode + '.transport.log'), row['transport_log_sha256'], base, refs)
    admissions = [p for p in parent.glob(mode + '.*.admission-*.json')
                  if sha256(safe(p, base)) == row['admission_sha256']]
    require(len(admissions) == 1, 'native admission receipt missing or duplicated')
    admission = read(admissions[0])
    requested = admission['requested']
    require(admission['admitted'] is True and admission['blockers'] == []
            and admission['round_sha256'] == lock['round_sha256'] and admission['host'] == row['host']
            and admission['official_tests_opened'] is False
            and requested['host'] == row['host'] and requested['requires_gpu'] is True
            and requested['cpu_slot'] == lock['cpu_slots'][row['host']]
            and requested['ram_bytes'] == lock['roles']['gpu_worker']['ram_bytes']
            and requested['scratch_bytes'] == lock['roles']['gpu_worker']['scratch_bytes'],
            'native operation was not admitted under the frozen resources')
    monitor_path = parent / (mode + '.monitor.json')
    if not monitor_path.exists():
        require(row['status'] != 'ok', 'native success lacks whole-operation monitor')
        return row['elapsed_seconds']
    monitor = read(monitor_path)
    claims(monitor)
    require(monitor['operation'] == mode and monitor['round_sha256'] == lock['round_sha256']
            and monitor['host'] == row['host'] and monitor['foreign_processes_signaled'] is False
            and monitor['energy_attributable_to_job'] is None
            and monitor['request_sha256'] == row['request_sha256']
            and monitor['timeout_seconds'] == row['timeout_seconds'],
            'native monitor identity or ownership changed')
    evidence(parent / (mode + '.worker.log'), monitor['worker_log_sha256'], base, refs)
    if row['status'] == 'ok':
        require(monitor['status'] == 'ok' and row['exit_code'] == monitor['exit_code'] == 0
                and finite(monitor['elapsed_seconds']) and 0 <= monitor['elapsed_seconds'] <= row['timeout_seconds']
                and 0 <= monitor['peak_gpu_used_mib'] <= lock['gpu_vram_limit_mib']
                and 0 <= monitor['peak_resident_bytes'] <= requested['ram_bytes'],
                'native success lacks passing whole-operation quotas')
        if mode == 'fit':
            require(monitor['gpu_process_observed'] is True, 'native fit did not observe a GPU process')
    return row['elapsed_seconds']


def trial(job, root, lock, refs):
    """Read a closed immutable attempt; unfinished attempts produce no winner."""
    base = Path(root).parent
    key = digest(job)
    parent = Path(root) / 'attempts' / key / 'attempt-0001'
    receipt_path = parent / 'receipt.json'
    if not receipt_path.exists():
        return None
    require(job['final'] is False and job['fit_seed'] == 11 and job['track'] == 'common-numeric',
            'native job moved outside the research track')
    worker_files(job['worker'], base, refs)
    evidence(receipt_path, sha256(safe(receipt_path, base)), base, refs)
    receipt = read(receipt_path)
    claims(receipt)
    require(receipt['job'] == job and receipt['round_sha256'] == lock['round_sha256']
            and receipt['attempt'] == 1 and receipt['status'] in ('ok', 'failed')
            and receipt['counts_as_dope_win'] is False, 'native fit-job identity changed')
    require(not any(p.is_symlink() for p in parent.rglob('*'))
            and {p.name for p in parent.iterdir() if p.is_file() and p.name != 'receipt.json'}
            == set(receipt['evidence_files'])
            and {p.name for p in parent.iterdir() if p.is_dir()}
            .issubset({'artifact', 'fit.empty-cache', 'native.empty-cache', 'sample.empty-cache'})
            and not any(list(p.rglob('*')) for p in parent.glob('*.empty-cache')),
            'native attempt evidence inventory changed')
    for name, expected in receipt['evidence_files'].items():
        require(Path(name).name == name, 'native receipt gained a nested evidence path')
        evidence(parent / name, expected, base, refs)
    operations = receipt['operations']
    require(1 <= len(operations) <= 3 and all(r['status'] == 'ok' for r in operations[:-1]),
            'native operation schedule continued after a failure')
    modes = ('fit', 'native', 'sample')[:len(operations)]
    require({p.name.removesuffix('.operation.json') for p in parent.glob('*.operation.json')}
            == {mode for mode, row in zip(modes, operations) if row['new_operation_started'] is True},
            'native operation accounting omitted a phase')
    seconds = sum(operation(mode, row, parent, lock, base, refs) for mode, row in zip(modes, operations))
    for mode in modes:
        request = read(parent / (mode + '.request.json'))
        require(request['job'] == job and request['artifact'] == str(parent / 'artifact')
                and request['receipt'] == str(parent / (mode + '.json')), 'native request lineage changed')
        if mode == 'sample':
            require(request['schedule'] == lock['common_sample_schedule'], 'native sample request schedule changed')
    fit = None
    native = None
    samples_complete = False
    if operations[0]['status'] == 'ok':
        fit = read(parent / 'fit.json')
        claims(fit)
        require(fit['status'] == 'ok' and fit['job_sha256'] == key
                and fit['round_sha256'] == lock['round_sha256'] and fit['host'] == 'xbabe1'
                and fit['gpu_fit_required'] is True and fit['source_adapter_unchanged'] is True
                and fit['projection_bytes_included'] is True and fit['training_rows'] == job['worker']['train_rows']
                and 0 < fit['peak_torch_allocated_bytes'] <= lock['gpu_vram_limit_mib'] * 2**20
                and 0 < fit['peak_torch_reserved_bytes'] <= lock['gpu_vram_limit_mib'] * 2**20,
                'native fit contract or GPU evidence changed')
        artifact(parent / 'artifact', fit['artifact_inventory'], fit['artifact_bytes'], job['worker'], base, refs)
        if len(operations) >= 2 and operations[1]['status'] == 'ok':
            native = read(parent / 'native.json')
            claims(native)
            require(native['status'] == 'ok' and native['job_sha256'] == key
                    and native['round_sha256'] == lock['round_sha256']
                    and native['name'] == native['objective'] and native['shared_kpi_used_for_selection'] is False,
                    'native KPI used shared selection evidence')
            native_metric(native, job['worker'], job['native_objective'], lock)
            evidence(parent / 'native-sample.csv', native['synthetic_sha256'], base, refs)
        if len(operations) == 3 and operations[2]['status'] == 'ok':
            batch = read(parent / 'sample.json')
            claims(batch)
            require(batch['status'] == 'ok' and batch['job_sha256'] == key
                    and batch['round_sha256'] == lock['round_sha256'] and batch['sample_replay_exact'] is True
                    and [s['schedule'] for s in batch['samples']] == lock['common_sample_schedule'],
                    'native sample matrix or replay changed')
            for sample in batch['samples']:
                claims(sample)
                require(sample['job_sha256'] == key and sample['round_sha256'] == lock['round_sha256']
                        and sample['rows'] == job['worker']['train_rows'] * sample['schedule']['multiplier']
                        and sample['sha256'] == sample['sample_sha256'], 'native sample lineage changed')
                evidence(parent / (sample['schedule']['name'] + '.csv'), sample['sample_sha256'], base, refs)
            require(sha256(parent / 'n-seed101.csv') == sha256(parent / 'repeat.csv'),
                    'native sample replay mismatch')
            require(native is not None and native['synthetic_sha256'] == sha256(parent / 'n-seed101.csv'),
                    'native efficacy sample differs from its fixed-seed replay')
            samples_complete = True
    require((receipt['status'] == 'ok') == (len(operations) == 3 and all(r['status'] == 'ok' for r in operations)),
            'native trial status launders a failed operation')
    partial = []
    directory = parent / 'artifact'
    if fit is None and directory.exists():
        require(all(p.name in ('model.json', 'model.pt', 'projection.json') for p in directory.iterdir()),
                'failed native artifact gained an unexpected file')
        for path in sorted(directory.iterdir()):
            safe(path, base)
            require(path.is_file(), 'failed native artifact contains a directory')
            h = sha256(path)
            evidence(path, h, base, refs)
            partial.append({'path': path.name, 'bytes': path.stat().st_size, 'sha256': h})
    return {'dataset': job['dataset'], 'method': job['method'], 'trial_index': job['trial_index'],
            'job_sha256': key,
            'config': job['config'], 'status': receipt['status'], 'fit_seed': 11, 'kind': 'new_native_trial',
            'receipt_path': str(receipt_path), 'receipt_sha256': sha256(receipt_path),
            'artifact_path': str(directory) if directory.exists() else None,
            'artifact_bytes': fit['artifact_bytes'] if fit else sum(r['bytes'] for r in partial),
            'partial_artifact_inventory': partial, 'complete_fitted_artifact': fit is not None,
            'native_kpi': native, 'native_selection_eligible': native is not None,
            'common_samples_complete': samples_complete, 'operation_seconds': seconds,
            'historical_runtime_closure_upgraded': False, 'counts_as_dope_win': False}


def prior_trial(row, lock, worker, base, refs):
    """Reuse externally bound historical evidence without upgrading its closure."""
    path = evidence(row['receipt_path'], row['receipt_sha256'], base, refs)
    receipt = read(path)
    job = receipt['job']
    require(receipt['status'] == 'ok' and receipt['validation_only'] is True
            and receipt['round_sha256'] == PRIOR_ROUND_SHA
            and receipt['mfs_v2'] is None and receipt['ptf_v1'] is None
            and row['kind'] == 'immutable_prior_native_trial_reuse'
            and row['historical_runtime_closure_upgraded'] is False
            and row['new_gpu_fit_started'] is False and row['counts_as_dope_win'] is False
            and (job['dataset'], job['method'], job['trial'], job['config'], job['fit_seed'])
            == (row['dataset'], row['method'], row['trial_index'], row['config'], 11)
            and job['sample_seed'] == 101 and job['size_multiplier'] == 1,
            'historical native identity or scope changed')
    worker_files(worker, base, refs)
    require(job['train_rows'] == worker['train_rows']
            and all(job[key] == worker['files'][name] for name, key in
                    [('train.csv', 'train_sha256'), ('validation.csv', 'validation_sha256'),
                     ('projection.json', 'projection_sha256')]), 'historical native worker lineage changed')
    for name, expected in receipt['evidence_files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,
                'historical native evidence escaped its attempt')
        evidence(path.parent / name, expected, base, refs)
    require(Path(row['artifact_path']) == path.parent / 'artifact'
            and row['artifact_bytes'] == receipt['artifact_bytes'], 'historical native artifact changed')
    artifact(row['artifact_path'], receipt['artifact_inventory'], receipt['artifact_bytes'], worker, base, refs)
    metric = receipt['native_kpi']
    require(metric == row['native_kpi'], 'historical native KPI was rewritten')
    native_metric(metric, worker, lock['native_objectives'][row['method']]['native_objective'], lock)
    native_phase = read(path.parent / 'native.receipt.json')
    require(native_phase['status'] == 'ok' and native_phase['exit_code'] == 0
            and native_phase['child'] == metric and finite(native_phase['wall_seconds'])
            and 0 <= native_phase['wall_seconds'] <= 600, 'historical native metric operation differs')
    fit = receipt['fit']
    require(fit['status'] == 'ok' and fit['exit_code'] == 0
            and 0 < fit['peak_device_used_mib'] <= lock['gpu_vram_limit_mib']
            and 0 < fit['child']['peak_torch_allocated_bytes'] <= lock['gpu_vram_limit_mib'] * 2**20
            and 0 < fit['child']['peak_torch_reserved_bytes'] <= lock['gpu_vram_limit_mib'] * 2**20
            and finite(fit['wall_seconds']) and 0 <= fit['wall_seconds'] <= 600
            and finite(receipt['wall_seconds']) and receipt['wall_seconds'] >= fit['wall_seconds'],
            'historical native fit or operation cost changed')
    sample = receipt['sample']
    require(sample['status'] == 'ok' and sample['exit_code'] == 0
            and sample['child']['sha256'] == receipt['sampling_repeated_sha256'] == metric['synthetic_sha256'],
            'historical native sample replay changed')
    evidence(path.parent / 'sample.csv', metric['synthetic_sha256'], base, refs)
    evidence(path.parent / 'repeat.csv', metric['synthetic_sha256'], base, refs)
    return row | {'status': 'ok', 'native_selection_eligible': True, 'common_samples_complete': False,
                  'operation_seconds': receipt['wall_seconds'], 'complete_fitted_artifact': True,
                  'partial_artifact_inventory': []}


def closure(root, lock, rows, refs):
    """Require every frozen job and a clean stopped coordinator before sealing."""
    root = Path(root)
    require(len(rows) == len(lock['jobs'])
            and {r['job_sha256'] for r in rows} == {digest(j) for j in lock['jobs']},
            'native matrix is incomplete or duplicated')
    completion = read(safe(root / 'completion.json', root.parent))
    end = read(safe(root / 'coordinator-exit.json', root.parent))
    claims(completion)
    claims(end)
    require(completion['jobs'] == len(lock['jobs']) and completion['round_sha256'] == lock['round_sha256']
            and end['round_sha256'] == lock['round_sha256'] and end['exit_code'] == 0
            and finite(end['elapsed_seconds']) and end['elapsed_seconds'] >= 0,
            'native coordinator did not close cleanly')
    supervisor = read(safe(root / 'supervisor-launch.json', root.parent))
    require(supervisor['round_sha256'] == lock['round_sha256']
            and all(type(supervisor[k]) is int and supervisor[k] > 0
                    and not Path('/proc', str(supervisor[k])).exists() for k in ('pid', 'supervisor_pid')),
            'native supervisor or coordinator is still active')
    evidence(root / 'coordinator.log', end['log_sha256'], root.parent, refs)
    for name in ('completion.json', 'coordinator-exit.json', 'supervisor-launch.json'):
        path = root / name
        evidence(path, sha256(safe(path, root.parent)), root.parent, refs)
    return end['elapsed_seconds']


def select(rows, methods, expected_pairs, failures):
    """Select complete four-trial groups, retaining post-native sampling failures."""
    groups = defaultdict(list)
    failed = defaultdict(list)
    for row in rows:
        groups[(row['dataset'], row['method'])].append(row)
    for row in failures:
        failed[(row['dataset'], row['method'])].append(row)
    require(set(groups).issubset(set(expected_pairs)) and set(failed).issubset(set(expected_pairs)),
            'native selection contains an unexpected lineage')
    cells = []
    for dataset, method in sorted(expected_pairs):
        trials = groups[(dataset, method)]
        spec = methods[method]
        indices = [r['trial_index'] for r in trials]
        require(len(indices) == len(set(indices)) and all(type(i) is int and 0 <= i < 4 for i in indices),
                'native selection duplicated a tuning trial')
        require(all(r['config'] == spec['configurations'][r['trial_index']] for r in trials),
                'native tuning escaped the frozen configuration grid')
        cost = sum(r['operation_seconds'] for r in trials) + sum(r['failed_operation_seconds'] for r in failed[(dataset, method)])
        require(finite(cost) and 0 <= cost <= 43200 and len(trials) + len(failed[(dataset, method)]) <= 8,
                'native failed attempts exceeded the per-cell tuning budget')
        complete = set(indices) == set(range(4))
        eligible = [r for r in trials if r['native_selection_eligible'] is True]
        require(all(type(r['native_selection_eligible']) is bool and finite(r['operation_seconds'])
                    and r['operation_seconds'] >= 0 for r in trials)
                and all(r['artifact_bytes'] is not None and finite(r['native_kpi']['value']) for r in eligible),
                'native selection admitted an invalid KPI')
        winner = min(eligible, key=lambda r: (-r['native_kpi']['value'], r['artifact_bytes'], digest(r['config']), r['trial_index'])) if complete and eligible else None
        defaults = [r for r in trials if r['config'] == spec['default_config']]
        require(len(defaults) <= 1, 'native author default is ambiguous')
        cells.append({'dataset': dataset, 'method': method, 'all_four_trials_closed': complete,
                      'default_trial': defaults[0] if defaults else None, 'native_selected_trial': winner,
                      'status': 'selected' if winner else ('native_selection_unavailable' if complete else 'pending'),
                      'selection_direction': 'maximize', 'tuning_trials_with_previous_failures': len(trials) + len(failed[(dataset, method)]),
                      'operation_seconds_with_previous_failures': cost,
                      'shared_kpi_used_for_selection': False, 'sampling_success_used_for_selection': False,
                      'native_kpis_cross_ranked': False, 'counts_as_dope_win': False})
    return cells
