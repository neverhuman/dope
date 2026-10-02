"""Reconcile the complete disjoint S3 confirmation and native baseline validation panel."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
import math
from pathlib import Path

from .manifest import digest
from .score import sha256
from .publish_s3_matched import cell_row, summarize, schema

BASE = Path('/mnt/fast-scratch/dope-benchmark')
DOPE = BASE / 'gpu-target-s3-confirmation-v1'
SDV = BASE / 's3-native-confirmation-validation-v1'
NATIVE = BASE / 'neural-native-v1'
HERE = Path(__file__).parent
NAME = 's3-matched-confirmation-validation'
RECEIPT_LOCK = BASE / 's3-confirmation-publication-custody-v1/receipt-lock.json'
RECEIPT_LOCK_SHA256 = '2d2fe75ec7c0c88056ea02987f36352342f69ad5526c9372f85b24fd5e110949'
PINS = {
    'dope_fit': '71b835791fbe1f17f4299d18d9e2ea6923b701ffa3f89f1ad518622244e07a1a',
    'dope_validation': '977b37204e282327a620748b2750a52feea3dd0648af422f22bc20ba6f4fc969',
    'native_fit': '5634d235eb8305436ca046867d96d09f562fabbd360892c580a7da3b9b811988',
    'native_validation': '5d45eb78c0de6eb491cf90522a61aa3e4c8031380ad304fbbcd03f52c920bfd5',
}
PROFILES = ('features12_steps512', 'features12_steps2048',
            'features24_steps512', 'features24_steps2048')
CONFIGS = [('DOPE', p) for p in PROFILES] + [
    (m, c) for m in ('CTGAN', 'TVAE') for c in ('default', 'native_selected')]
AUDITORS = ('catboost', 'linear', 'mlp')
SEEDS = (101, 211, 307)
SIZES = (1, 4)


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def safe(path):
    path = Path(path)
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError('publication evidence path unavailable') from exc
    require(resolved.is_relative_to(BASE) and 'evaluator' not in resolved.parts
            and 'evaluator' not in path.parts and resolved.suffix != '.csv'
            and path.suffix != '.csv' and not path.is_symlink()
            and not any(parent.is_symlink() for parent in path.parents),
            'publication evidence path outside allowed scope')
    return path


def evidence(root, receipt):
    for name, expected in receipt.get('evidence_files', {}).items():
        path = root / name
        require(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink()
                and 'evaluator' not in path.parts, 'cell evidence path escaped')
        require(sha256(path) == expected, 'immutable cell evidence changed')


def sealed(row):
    require(row['official_tests_opened'] is False and row['mfs_v2'] is None
            and row['ptf_v1'] is None and row['new_fits_started'] == 0,
            'validation seal or claim changed')


def verify_cell(root, job, lock_sha, receipt_hashes):
    path = root / 'cells' / digest(job) / 'receipt.json'
    require(path.exists(), 'frozen matched validation matrix incomplete')
    require(str(path) in receipt_hashes and sha256(safe(path)) == receipt_hashes[str(path)],
            'frozen validation receipt changed')
    receipt = read(safe(path))
    require(receipt['job'] == job and receipt['round_sha256'] == lock_sha,
            'validation job identity changed')
    sealed(receipt)
    evidence(path.parent, receipt)
    metrics = None
    if receipt['status'] == 'ok':
        require([op['tag'] for op in receipt['operations']] == ['sample', 'repeat', 'metric'],
                'successful cell operation coverage changed')
        for op in receipt['operations']:
            require(op['status'] == 'ok' and op['exit_code'] == 0
                    and math.isfinite(op['elapsed_seconds']) and 0 <= op['elapsed_seconds'] <= 610,
                    'successful cell contains a failed or over-budget operation')
            require(sha256(safe(path.parent / (op['tag'] + '.log'))) == op['log_sha256']
                    and sha256(safe(path.parent / (op['tag'] + '.admission.json')))
                    == op['admission_sha256'], 'operation evidence changed')
            require(read(safe(path.parent / (op['tag'] + '.operation.json'))) == op,
                    'cell operation receipt differs')
        metrics = read(safe(path.parent / 'metrics.json'))
        replay = read(safe(path.parent / 'metrics-replay.json'))
        payload = lambda r: {k: v for k, v in r.items() if k != 'metric_seconds'}
        require(payload(metrics) == payload(replay) and receipt['sample_replay_exact']
                and receipt['metric_replay_exact'], 'metric replay differs')
        require(sha256(path.parent / 'sample.csv') == sha256(path.parent / 'repeat.csv'),
                'sample replay differs')
        require(metrics['mfs_v2'] is None and metrics['gate_profile_complete'] is False
                and metrics['rows']['synthetic'] == job['rows'], 'metric scope changed')
    return path, receipt, metrics


def verify_frozen_sources(locks, refs):
    """Hash frozen executable/input closures before consuming measured results."""
    checked = {}

    def check(path, expected, runtime=False, table=False, dependency=False, publish=True):
        path = Path(path)
        if table:
            safe(path.parent)
            require(path.name in ('train.csv', 'validation.csv') and not path.is_symlink(),
                    'worker table outside validation scope')
        elif dependency:
            # Frozen library distributions include demo resources. Hash their
            # bytes without parsing them; these are outside study partitions.
            safe(path.parent)
            require(path.resolve(strict=True).is_relative_to(NATIVE / 'deps')
                    and not path.is_symlink(), 'dependency path outside frozen distribution')
        elif runtime:
            # The original frozen Torch runtime has this versioned cuDNN alias.
            # Permit that same target only; its content digest remains locked.
            cudnn_alias = (path.name == 'libcudnn.so'
                           and path.resolve(strict=True) == path.with_name('libcudnn.so.9'))
            require(path.suffix != '.csv' and 'evaluator' not in path.parts
                    and 'evaluator' not in path.resolve(strict=True).parts
                    and not any(parent.is_symlink() for parent in path.parents)
                    and (not path.is_symlink() or cudnn_alias), 'runtime evidence outside source scope')
        else:
            safe(path)
        if path in checked:
            require(checked[path] == expected, 'frozen source identities disagree')
        else:
            require(sha256(path) == expected, 'frozen source or input changed')
            checked[path] = expected
        if publish:
            require(str(path) not in refs or refs[str(path)] == expected,
                    'publication reference identity changed')
            refs[str(path)] = expected

    fit, validation, native, samples = (locks[k] for k in (
        'dope_fit', 'dope_validation', 'native_fit', 'native_validation'))
    for root, lock in ((DOPE, fit), (DOPE / 'validation-v2', validation), (SDV, samples)):
        for name, expected in lock['source_files'].items():
            check(root / 'source' / name, expected)
    package = BASE / 'pilot-24h/dope-target-refinement-v1/package'
    manifest = package / 'package-manifest.json'
    require(fit['package_manifest_sha256'] == validation['package_manifest_sha256']
            == samples['metric_package_manifest_sha256'], 'metric package identities disagree')
    check(manifest, fit['package_manifest_sha256'])
    for name, expected in read(manifest)['files'].items():
        check(package / name, expected)
    check(package / 'research/benchmark/pilot_metrics.py', validation['metric_source_sha256'])
    check(Path(fit['gpu_binary_path']), fit['gpu_binary_sha256'])
    for name, expected in native['source_files'].items():
        require(samples['native_source_files'][name] == expected, 'native source identities disagree')
        check(NATIVE / 'package/research/benchmark' / name, expected)
    for root, files in ((Path(validation['metric_dependency_root']), validation['metric_dependency_files']),
                        (NATIVE / 'deps', native['dependency_files']),
                        (NATIVE / 'deps', samples['dependency_files'])):
        for name, expected in files.items():
            check(root / name, expected, dependency=True, publish=False)
    runtimes = list(fit['host_runtime_locks'].values()) + [{
        'path': str(BASE / 's3-native-matched-runtime-v1/xbabe2/runtime.lock.json'),
        'sha256': samples['runtime_sha256']}]
    for runtime in runtimes:
        check(runtime['path'], runtime['sha256'])
        for package in read(runtime['path'])['packages'].values():
            for name, expected in package['files'].items():
                check(Path(package['root']) / name, expected, runtime=True, publish=False)
    workers = list(fit['workers'].values()) + [job['worker'] for job in samples['jobs']]
    for worker in workers:
        root = Path(worker['path'])
        require(not (root / 'test.csv').exists(), 'worker contains official test partition')
        for name, expected in worker['files'].items():
            check(root / name, expected, table=name.endswith('.csv'))
    return len(checked)


def artifact_inventory(path):
    require(path.is_dir() and not path.is_symlink()
            and not any(parent.is_symlink() for parent in path.parents),
            'artifact root must be an unlinked directory')
    result = []
    for item in sorted(path.rglob('*')):
        require(not item.is_symlink(), 'artifact contains link')
        if item.is_file():
            result.append({'path': item.relative_to(path).as_posix(),
                           'bytes': item.stat().st_size, 'sha256': sha256(item)})
    return result


def verify_native_fit(path, lock, expected_sha):
    path = safe(path)
    require(sha256(path) == expected_sha, 'native fit receipt changed')
    receipt = read(path)
    require(receipt['job'] in lock['jobs'] and receipt['round_sha256'] == PINS['native_fit']
            and receipt['status'] == 'ok' and receipt['validation_only'] is True
            and receipt['mfs_v2'] is None and receipt['ptf_v1'] is None,
            'native fit identity changed')
    inventory = artifact_inventory(path.parent / 'artifact')
    evidence(path.parent, receipt)
    require(inventory == sorted(receipt['artifact_inventory'], key=lambda r: r['path'])
            and sum(r['bytes'] for r in inventory) == receipt['artifact_bytes'],
            'native artifact accounting changed')
    kpi = receipt['native_kpi']
    require(kpi['partition'] == 'validation' and kpi['direction'] == 'maximize'
            and kpi['objective'] == 'sdmetrics_mean_regression_r2'
            and set(kpi['components']) == {'LinearRegression', 'MLPRegressor'}
            and kpi['implementation_sha256'] == lock['source_files']['sdv_adapter.py']
            and kpi['validation_sha256'] == receipt['job']['validation_sha256']
            and kpi['seed'] == 1729
            and kpi['synthetic_sha256'] == receipt['sample']['child']['sha256']
            == receipt['sampling_repeated_sha256']
            and math.isfinite(kpi['value'])
            and abs(kpi['value'] - sum(kpi['components'].values()) / 2) <= 1e-14,
            'native objective changed')
    return receipt


def native_winner(attempts):
    require(len(attempts) == 4 and {r['job']['trial'] for r in attempts} == set(range(4)),
            'native trial matrix incomplete')
    require(sum(r['wall_seconds'] for r in attempts) <= 43200,
            'native tuning time cap exceeded')
    lineage = {(r['job']['dataset'], r['job']['method'], r['job']['fit_seed'],
                r['job']['train_sha256'], r['job']['validation_sha256'],
                r['job']['projection_sha256']) for r in attempts}
    require(len(lineage) == 1, 'native tuning lineage differs')
    return min(attempts, key=lambda r: (-r['native_kpi']['value'],
                                       r['artifact_bytes'], digest(r['job']['config'])))


def verify_dope_artifact(fit, job, receipt, refs):
    """A rejected GPU admission cannot supply an artifact or a trained fit."""
    charged = None
    if fit['status'] == 'ok':
        child = safe(fit['child']['receipt'])
        require(sha256(child) == fit['child']['receipt_sha256'], 'DOPE child receipt changed')
        child_receipt = read(child)
        require(child_receipt['trained_gpu_target_verified'] is True
                and child_receipt['status'] == 'ok', 'GPU training verification changed')
        refs[str(child)] = sha256(child)
        charged = child_receipt['artifact_bytes']
        model = child.with_suffix('.dpk')
        projection = safe(Path(job['worker']['path']) / 'projection.json')
        require(sha256(model) == child_receipt['artifact_sha256']
                and sha256(projection) == job['worker']['files']['projection.json']
                and model.stat().st_size + projection.stat().st_size == charged <= 10240,
                'DOPE charged artifact changed')
    else:
        require(fit['status'] == 'dispatch_failed' and fit['child'] is None
                and receipt['status'] == 'fit_unavailable',
                'unavailable DOPE fit accounting differs')
    return charged, (child, child_receipt) if fit['status'] == 'ok' else None


def completed_controllers():
    references = {}
    for root, expected in ((DOPE / 'validation-v2', PINS['dope_validation']),
                           (SDV, PINS['native_validation'])):
        path = root / 'controller-attempt-0001-exit.json'
        require(path.is_file(), 'confirmation controller not closed')
        receipt = read(safe(path))
        require(receipt['exit_code'] == 0 and receipt['round_sha256'] == expected,
                'confirmation controller failed or lineage changed')
        log = root / 'controller-attempt-0001.log'
        require(sha256(safe(log)) == receipt['log_sha256'], 'confirmation controller log changed')
        references[str(path)] = sha256(path)
        references[str(log)] = receipt['log_sha256']
    return references


def verify_execution(path, expected_sha, refs):
    """Verify declared admission sources without claiming a full runtime closure."""
    def check(member, expected):
        member = safe(member)
        require(sha256(member) == expected, 'confirmation execution dependency changed')
        require(str(member) not in refs or refs[str(member)] == expected,
                'confirmation execution identities disagree')
        refs[str(member)] = expected

    check(path, expected_sha)
    execution = read(path)
    for name, expected in execution.get('source_files', {}).items():
        check(path.parent / 'source' / name, expected)
    for member, expected in execution.get('required_files', {}).items():
        check(member, expected)
    if execution.get('owner_scan_source'):
        check(execution['owner_scan_source'], execution['owner_scan_source_sha256'])
    for member, identity in execution['recognized_owner_sources'].items():
        check(member, identity['sha256'])
    meter = execution.get('meter_lock')
    if meter:
        check(meter['path'], meter['sha256'])
        lock = read(meter['path'])
        check(Path(meter['path']).parent / 'meter.py', lock['source_sha256'])
        require(execution['du_sha256'] == lock['du_sha256'] == sha256(Path('/usr/bin/du')),
                'scratch measurement binary changed')
        refs['/usr/bin/du'] = lock['du_sha256']
    return execution


def build():
    require(sha256(HERE / 'publish_s3_matched.py') ==
            'b6dc53c2bcc65efcdc90ceb39569005fe39a934d55bd6f1211ce26db2cb1a81a',
            'shared publication helpers changed')
    controller_references = completed_controllers()
    paths = {'dope_fit': DOPE / 'round.lock.json',
             'dope_validation': DOPE / 'validation-v2/round.lock.json',
             'native_fit': NATIVE / 'round.lock.json', 'native_validation': SDV / 'round.lock.json'}
    for name, path in paths.items():
        require(sha256(safe(path)) == PINS[name], 'frozen input round changed')
    locks = {name: read(path) for name, path in paths.items()}
    require(sha256(safe(RECEIPT_LOCK)) == RECEIPT_LOCK_SHA256, 'publication receipt lock changed')
    receipt_lock = read(RECEIPT_LOCK)
    receipt_hashes = receipt_lock['reference_hashes']
    require(receipt_lock['physical_receipts'] == 270 and receipt_lock['logical_cells'] == 288,
            'publication receipt coverage changed')
    for path, expected in receipt_hashes.items():
        require(sha256(safe(path)) == expected, 'frozen publication reference changed')
    dlock, slock, fitlock, native = (locks[n] for n in (
        'dope_validation', 'native_validation', 'dope_fit', 'native_fit'))
    require(dlock['fit_round_sha256'] == slock['s3_dope_fit_round_sha256'] == PINS['dope_fit']
            and slock['native_round_sha256'] == PINS['native_fit']
            and dlock['metric_source_sha256'] == slock['metric_source_sha256'], 'comparison lineage differs')
    require(len(dlock['jobs']) == len(slock['logical_cells']) == 144 and len(slock['jobs']) == 126,
            'frozen matrix size changed')
    refs = receipt_hashes | controller_references | {str(p): sha256(p) for p in paths.values()}
    refs[str(RECEIPT_LOCK)] = RECEIPT_LOCK_SHA256
    source_closure_file_count = verify_frozen_sources(locks, refs)
    physical, fits, cells, operations, child_fits = {}, {}, [], [], {}
    continuation = DOPE / 'validation-v2/execution.lock.json'
    continuation_pin = '96b115e9a6c4f26d2e36a84c5aacbe24d203fd6817ed821ca400f453f3c628b6'
    execution = verify_execution(continuation, continuation_pin, refs)
    for root, expected_sha in ((SDV, 'bbffbe789df2a13c741919b13fa8a26ad67def46a935cb16e6e24d7c3eecdebe'),
                               (DOPE / 'transport-v3', '692ef6f313f1fd04405d8622d9dd79a2702de7d77372a15fcd4d7878b3567281')):
        filename = 'execution.lock.json' if root == SDV else 'execution-v7.lock.json'
        path = root / filename
        verify_execution(path, expected_sha, refs)
    for job in dlock['jobs']:
        path, receipt, metrics = verify_cell(DOPE / 'validation-v2', job, PINS['dope_validation'], receipt_hashes)
        refs[str(path)] = sha256(path)
        if 'validation_execution_sha256' in receipt:
            require(receipt['validation_execution_sha256'] == continuation_pin, 'validation execution differs')
        parent = safe(DOPE / 'dispatch' / job['fit_job_sha256'] / 'receipt.json')
        fit = read(parent)
        require(fit['job'] == job['fit_job'] and fit['round_sha256'] == PINS['dope_fit'],
                'DOPE fit identity changed')
        if 'fit_dispatch_sha256' in receipt:
            require(sha256(parent) == receipt['fit_dispatch_sha256'], 'DOPE fit receipt changed')
        fits[str(parent)] = fit
        refs[str(parent)] = sha256(parent)
        charged, trained = verify_dope_artifact(fit, job, receipt, refs)
        if trained is not None:
            child, child_receipt = trained
            child_fits[str(child)] = child_receipt
        cells.append(cell_row(job, receipt, metrics, 'DOPE', job['fit_job']['research_profile'],
                              charged, path))
        operations.extend(receipt.get('operations', []))
    native_fits = {}
    for job in slock['jobs']:
        path, receipt, metrics = verify_cell(SDV, job, PINS['native_validation'], receipt_hashes)
        refs[str(path)] = sha256(path)
        require(digest(job) not in physical, 'physical sample identity duplicated')
        physical[digest(job)] = (job, receipt, metrics, path)
        fp = job['fit_receipt_path']
        if fp not in native_fits:
            native_fits[fp] = verify_native_fit(fp, native, job['fit_receipt_sha256'])
        fit = native_fits[fp]
        require(fit['job'] == job['fit_job'] and fit['artifact_bytes'] == job['artifact_bytes'],
                'baseline fit binding changed')
        refs[fp] = job['fit_receipt_sha256']
        operations.extend(receipt.get('operations', []))
    used_physical = set()
    for alias in slock['logical_cells']:
        key = alias['physical_job_sha256']
        require(key in physical, 'logical sample alias missing physical receipt')
        used_physical.add(key)
        job, receipt, metrics, path = physical[key]
        selection_ref = slock['selections'][f"{job['dataset']}-{job['method']}"]
        require(selection_ref['sha256'] == alias['native_selection_sha256']
                == sha256(safe(selection_ref['path'])), 'native selection changed')
        refs[selection_ref['path']] = selection_ref['sha256']
        selection = read(selection_ref['path'])
        attempts = []
        for attempt in selection['attempts']:
            fp = attempt['path']
            if fp not in native_fits:
                native_fits[fp] = verify_native_fit(fp, native, attempt['sha256'])
            refs[fp] = attempt['sha256']
            attempts.append(native_fits[fp])
        winner = native_winner(attempts)
        require(digest(winner['job']) == selection['selected_job'], 'native winner differs')
        config = 'default' if alias['kind'] == 'default' else 'native_selected'
        require(alias['kind'] in ('default', 'native_tuned'), 'baseline configuration kind changed')
        expected = next(r for r in attempts if r['job']['trial'] == 0) if config == 'default' else winner
        require(job['fit_job'] == expected['job']
                and job['worker']['files'] == fitlock['workers'][job['dataset']]['files'],
                'baseline input or configuration differs')
        cells.append(cell_row(job, receipt, metrics, job['method'], config,
                              job['artifact_bytes'], path, expected['native_kpi']))
    require(used_physical == set(physical) and len(fits) == 24 and len(native_fits) == 48,
            'fit or sample accounting incomplete')
    summaries = summarize(cells, sorted(fitlock['workers']))
    native_publication = HERE / 'results/sdv-native-validation.json'
    native_publication_sha = '6cdbce873eb5c9df77e4e4f9649201a2da30bd3a1f0a6634a2abe2717e19ce6a'
    require(sha256(native_publication) == native_publication_sha,
            'original native tuning publication changed')
    rights_path = HERE / 'results/s3-data.lock.json'
    require(sha256(rights_path) == fitlock['s3_data_lock_sha256'], 'committed S3 rights/data lock changed')
    rights = read(rights_path)
    catalog_path = BASE / 'catalog/v1/catalog.jsonl'
    require(sha256(safe(catalog_path)) == rights['catalog_sha256'], 'source catalog changed')
    refs[str(catalog_path)] = rights['catalog_sha256']
    catalog = {r['dataset_hash']: r for r in (json.loads(line) for line in catalog_path.open())}
    entries = {r['id']: r for r in rights['entries']}
    metadata = []
    for dataset, worker in sorted(fitlock['workers'].items()):
        entry = entries[dataset]
        require(entry['status'] == 'prepared' and entry['license']['spdx'] == 'MIT'
                and entry['projected_files']['train'] == worker['files']['train.csv']
                and entry['projected_files']['validation'] == worker['files']['validation.csv'],
                'matched worker differs from rights-cleared partitions')
        require(digest(catalog[dataset]) == entry['catalog_entry_sha256'], 'catalog lineage differs')
        metadata.append({'id': dataset, 'display_name': catalog[dataset]['display_name'],
                         'train_rows': worker['cohort']['train_rows'],
                         'projected_features': worker['cohort']['projected_features'],
                         'source_catalog_rows': catalog[dataset]['shape']['rows'],
                         'license_spdx': 'MIT', 'projection_sha256': worker['files']['projection.json'],
                         'train_sha256': worker['files']['train.csv'],
                         'validation_sha256': worker['files']['validation.csv']})
    counts = Counter(r['status'] for r in cells)
    host_costs = []
    for host in sorted({f['host'] for f in fits.values()}):
        parents = [f for f in fits.values() if f['host'] == host]
        children = [f for f in child_fits.values() if f['host'] == host]
        host_costs.append({
            'host': host, 'attempted_fits': len(parents), 'successful_gpu_fits': len(children),
            'failed_admissions': sum(f['status'] != 'ok' for f in parents),
            'fit_entry_seconds': sum(f['elapsed_seconds'] for f in children),
            'dispatch_seconds': sum(f['elapsed_seconds'] for f in parents),
            'peak_observed_gpu_used_mib': max(f['peak_gpu_used_mib'] for f in children),
            'device_energy_estimate_joules_including_idle': sum(f['energy_joules_estimate'] for f in children)
            if all(f['energy_joules_estimate'] is not None for f in children) else None,
        })
    cost = {
        'dope_gpu_host_costs': host_costs,
        'dope_successful_fit_entry_seconds': sum(f['elapsed_seconds'] for f in child_fits.values()),
        'dope_failed_dispatch_seconds': sum(f['elapsed_seconds'] for f in fits.values() if f['status'] != 'ok'),
        'dope_dispatch_seconds_including_hash_and_admission_overhead': sum(f['elapsed_seconds'] for f in fits.values()),
        'dope_declared_gpu_fit_ceiling_seconds': 14400,
        'native_prior_48_trial_wall_seconds': sum(f['wall_seconds'] for f in native_fits.values()),
        'native_new_fits': 0, 'native_new_tuning_trials': 0,
        'sample_repeat_metric_operation_seconds': sum(r['elapsed_seconds'] for r in operations),
        'validation_controller_host': 'xbabe2',
        'validation_controller_wall_seconds_including_admission': sum(
            read(root / 'controller-attempt-0001-exit.json')['elapsed_seconds']
            for root in (DOPE / 'validation-v2', SDV)),
        'gpu_device_energy_estimate_joules_including_idle': sum(f['energy_joules_estimate'] for f in child_fits.values())
        if all(f['energy_joules_estimate'] is not None for f in child_fits.values()) else None,
        'peak_observed_gpu_used_mib': max(f['peak_gpu_used_mib'] for f in child_fits.values()),
        'cpu_energy_joules': None,
        'gpu_fit_hosts': sorted({f['host'] for f in child_fits.values()}),
        'limitations': 'Operation wall times include sample and metric replays and exclude admission scans. '
                      'Prior native trial wall includes fit/sample/native efficacy. Energy and earlier architecture '
                      'R&D are not fully accounted here. GPU energy is device-wide, includes idle, and is not baseline-attributed; per-cell parity is not equal total R&D spend.',
    }
    report = {
        'format': 'dope-s3-matched-confirmation-validation-panel', 'version': 1,
        'source_sha256': sha256(Path(__file__)), 'dataset_ids': sorted(fitlock['workers']),
        'dataset_metadata': metadata, 's3_data_lock_sha256': sha256(rights_path),
        'datasets': 6, 'track': 'common_numeric', 'partition': 'official_training_derived_validation',
        'fit_seeds': [11], 'sample_seeds': list(SEEDS), 'sizes': list(SIZES),
        'logical_sample_cells': 288, 'planned_physical_sample_cells': 270,
        'successful_physical_sample_cells': sum(r['status'] == 'ok' for r in cells[:144])
                                            + sum(r[1]['status'] == 'ok' for r in physical.values()),
        'logical_status_counts': dict(counts), 'all_frozen_cells_accounted': True,
        'default_native_aliases_charge_once': True,
        'validation_execution': {'original_pilot_deadline_utc': execution['original_pilot_deadline_unchanged'],
                                 'continuation_deadline_utc': execution['deadline_utc'],
                                 'original_controller_cells': sum('validation_execution_sha256' not in read(r['receipt']['path']) for r in cells[:144]),
                                 'continuation_controller_cells': sum('validation_execution_sha256' in read(r['receipt']['path']) for r in cells[:144]),
                                 'source_versioned_capacity_only': True,
                                 'original_metrics_and_sample_binary_unchanged': True},
        'native_objective_policy': 'CTGAN/TVAE use frozen author-library efficacy: mean LinearRegression and '
                                   'MLPRegressor R2, maximize on validation, then bytes/config digest. '
                                   'Common utility and DOPE KPI never select baselines; native KPIs are not '
                                   'ranked across methods.',
        'native_tuning_publication': {'path': 'research/benchmark/results/sdv-native-validation.json',
                                     'sha256': native_publication_sha,
                                     'all_original_trials': 96,
                                     'confirmation_trials_verified_here': 48,
                                     'confirmation_selections_verified_here': 12,
                                     'new_tuning_trials': 0},
        'scope': 'Single-fit-seed disjoint confirmation research, not public-core paired analysis, '
                 'five-fit stability or certification. Training cohorts are bounded subsets of official S3 training partitions; '
                 'native selection and these descriptive outcomes use training-derived validation. '
                 'All four DOPE profiles retained; production family unselected.',
        'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None, 'release_safe_l3': None,
        'production_certified': False, 'paired_superiority': None, 'counts_as_dope_win': False,
        'missing_evidence': ['shadow/privacy/static-artifact attacks', 'six production auditors and complete PTF profile',
                             'projection-only utility cost', 'product and informative-lineage coverage',
                             'public-core full frozen final matrix'],
        'source_locks': PINS | {'metric_implementation_sha256': dlock['metric_source_sha256'],
                                'shared_publication_helpers_sha256': sha256(HERE / 'publish_s3_matched.py')},
        'publication_integrity': {'receipt_lock_sha256': RECEIPT_LOCK_SHA256,
                                  'receipt_anchor_commit': receipt_lock['anchor_source_commit'],
                                  'declared_source_inventory_verified_files': source_closure_file_count,
                                  'current_full_executable_closure_verified': False,
                                  'historical_fit_executable_closure_verified': False},
        'cost': cost, 'cells': cells, 'summary': summaries,
        'immutable_references': [{'path': p, 'sha256': h} for p, h in sorted(refs.items())],
        'citation_keys': ['xu2019modeling'],
    }
    return report


def tables(report):
    out = io.StringIO()
    writer = csv.writer(out, lineterminator='\n')
    writer.writerow(['dataset', 'display_name', 'method', 'configuration', 'fit_seed', 'size_multiplier',
                     'charged_bytes', 'statuses', *[a + '_median_retention' for a in AUDITORS]])
    lines = ['# S3 matched confirmation validation', '', report['scope'], '',
             'Official tests sealed. MFS-v2 / PTF-v1 / release-safe L3: null. '
             'Bars and table values are sample medians from one fit seed, without fit uncertainty.', '',
             '| Dataset | Method/config | Size | Charged bytes | Sample statuses | CatBoost | Linear | MLP |',
             '|---|---|---:|---:|---|---:|---:|---:|']
    for row in report['summary']:
        values = [row['utility'][a]['median_retention'] for a in AUDITORS]
        name = next(d['display_name'] for d in report['dataset_metadata'] if d['id'] == row['dataset'])
        writer.writerow([row['dataset'], name, row['method'], row['configuration'], 11,
                         row['size_multiplier'], row['charged_artifact_bytes'],
                         json.dumps(row['statuses'], sort_keys=True), *values])
        formatted = ['null' if v is None else f'{v:.6f}' for v in values]
        lines.append(f"| {name} ({row['dataset']}) | {row['method']}/{row['configuration']} | {row['size_multiplier']}n | "
                     f"{row['charged_artifact_bytes']} | {json.dumps(row['statuses'], sort_keys=True)} | "
                     + ' | '.join(formatted) + ' |')
    lines.extend(['', 'Native objectives: ' + report['native_objective_policy'], '',
                  'All four declared DOPE research profiles remain visible; no production family is selected. '
                  'Baseline artifacts exceed L3 where charged bytes exceed 10,240; their common utility is unconstrained quality.', '',
                  'Failures and low-signal groups yield null medians. No failed or unavailable cell supplies a DOPE win. '
                  'Copy/control counts are evidence, without a complete privacy gate or formal DP claim.', '',
                  'Cost: ' + json.dumps(report['cost'], sort_keys=True), '',
                  'Citation keys: ' + ', '.join(report['citation_keys']), ''])
    return out.getvalue(), '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE / 'results' / (NAME + '.json'))
    parser.add_argument('--from-json', action='store_true', help='Regenerate tables from committed JSON only')
    args = parser.parse_args()
    if args.from_json:
        report = read(args.output)
        require(report['official_tests_opened'] is False and report['production_certified'] is False
                and all(report[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe_l3', 'paired_superiority'))
                and summarize(report['cells'], report['dataset_ids']) == report['summary'],
                'committed confirmation scope or reconciliation changed')
    else:
        report = build()  # Refuse incomplete matrices before writing anything.
        args.output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
        args.output.with_suffix('.schema.json').write_text(json.dumps(schema(report), sort_keys=True, indent=2) + '\n')
    table, markdown = tables(report)
    args.output.with_suffix('.csv').write_text(table)
    args.output.with_suffix('.md').write_text(markdown)
    print(args.output)


if __name__ == '__main__':
    main()
