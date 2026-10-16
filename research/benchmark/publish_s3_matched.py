"""Reconcile the complete frozen S3 discovery and native baseline validation panel."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import io
import json
import math
from pathlib import Path
from statistics import median

from .manifest import digest
from .score import sha256

BASE = Path('/mnt/fast-scratch/dope-benchmark')
DOPE = BASE / 'gpu-target-s3-discovery-v1'
SDV = BASE / 's3-native-matched-validation-v1'
NATIVE = BASE / 'neural-native-v1'
HERE = Path(__file__).parent
NAME = 's3-matched-discovery-validation'
RECEIPT_LOCK = BASE / 's3-matched-publication-custody-v1/receipt-lock.json'
RECEIPT_LOCK_SHA256 = '9b9113d026d8bb37a25f3f583b1f373b5911e261f6026006fcf010382ec4a00a'
PINS = {
    'dope_fit': '82ce9806e166e7d0c5dcd7c0bd976c1d60691fca609e9cc0999b571d7cdd9316',
    'dope_validation': '1fa7ad41a2d6058cd080b882fdb74656dc7be40bb1badf3b6dcc1b36d785875d',
    'native_fit': '5634d235eb8305436ca046867d96d09f562fabbd360892c580a7da3b9b811988',
    'native_validation': 'c58a271c0d0555892596ab469005f2c625f325fa8edb9dabd3805509d1e5f298',
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
            and path.suffix != '.csv' and not path.is_symlink(),
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
    for root, lock in ((DOPE, fit), (DOPE / 'validation-v1', validation), (SDV, samples)):
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
    require(path.is_dir() and not path.is_symlink(),
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


def cell_row(job, receipt, metrics, method, config, charged, path, native=None):
    return {
        'dataset': job['fit_job']['dataset'], 'method': method, 'configuration': config,
        'fit_seed': 11, 'sample_seed': job['sample_seed'], 'size_multiplier': job['size_multiplier'],
        'status': receipt['status'], 'charged_artifact_bytes': charged,
        'artifact_within_l3_cap': charged is not None and charged <= 10240,
        'physical_job_sha256': digest(job), 'receipt': {'path': str(path), 'sha256': sha256(path)},
        'native_validation_kpi': native, 'utility': metrics['utility'] if metrics else None,
        'exact_copy_count': metrics['copy_counts']['exact'] if metrics else None,
        'near_copy_count': metrics['copy_counts']['near'] if metrics else None,
        'real_vs_real_control_counts': metrics['real_vs_real_control_counts'] if metrics else None,
        'marginal_ks_mean': metrics['marginal_ks_mean'] if metrics else None,
        'pair_correlation_fidelity': metrics['pair_correlation_fidelity'] if metrics else None,
        'c2st_auc': metrics['c2st_auc'] if metrics else None,
        'mfs_v2': None, 'ptf_v1': None, 'release_safe_l3': None, 'counts_as_dope_win': False,
    }


def summarize(cells, datasets):
    expected = {(d, m, c, s, z) for d in datasets for m, c in CONFIGS for s in SEEDS for z in SIZES}
    actual = [(r['dataset'], r['method'], r['configuration'], r['sample_seed'],
               r['size_multiplier']) for r in cells]
    require(len(actual) == len(set(actual)) and set(actual) == expected,
            'logical matched matrix incomplete or duplicated')
    groups = defaultdict(list)
    for row in cells:
        groups[(row['dataset'], row['method'], row['configuration'], row['size_multiplier'])].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda r: r['sample_seed'])
        utility = {}
        for auditor in AUDITORS:
            values = [r['utility'][auditor].get('retention') if r['status'] == 'ok'
                      and r['utility'] is not None and r['utility'][auditor].get('informative') is True else None for r in rows]
            require(all(v is None or math.isfinite(v) for v in values), 'nonfinite common utility')
            complete = all(v is not None for v in values)
            utility[auditor] = {'complete_informative_sample_group': complete,
                                'median_retention': median(values) if complete else None,
                                'sample_retention_values': values}
        result.append({'dataset': key[0], 'method': key[1], 'configuration': key[2],
                       'size_multiplier': key[3], 'fit_seed': 11, 'sample_cells': 3,
                       'statuses': dict(Counter(r['status'] for r in rows)),
                       'charged_artifact_bytes': rows[0]['charged_artifact_bytes'], 'utility': utility})
    return result


def build():
    paths = {'dope_fit': DOPE / 'round.lock.json',
             'dope_validation': DOPE / 'validation-v1/round.lock.json',
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
    refs = receipt_hashes | {str(p): sha256(p) for p in paths.values()}
    refs[str(RECEIPT_LOCK)] = RECEIPT_LOCK_SHA256
    source_closure_file_count = verify_frozen_sources(locks, refs)
    physical, fits, cells, operations, child_fits = {}, {}, [], [], {}
    continuation = DOPE / 'validation-capacity-repair-v3/execution.lock.json'
    continuation_pin = 'da9f6f5baee2cf334484e22d4e2f9a0550dbefb2aa275d2368e61682ddfc9306'
    require(sha256(safe(continuation)) == continuation_pin, 'validation continuation changed')
    execution = read(continuation)
    refs[str(continuation)] = continuation_pin
    for name, expected in execution['source_files'].items():
        path = continuation.parent / 'source' / name
        require(sha256(safe(path)) == expected, 'continuation source changed')
        refs[str(path)] = expected
    meter = execution['meter_lock']
    require(sha256(safe(meter['path'])) == meter['sha256'], 'coordinator meter lock changed')
    meter_lock = read(meter['path'])
    meter_source = Path(meter['path']).parent / 'meter.py'
    require(sha256(safe(meter_source)) == meter_lock['source_sha256'], 'coordinator meter source changed')
    refs[meter['path']] = meter['sha256']
    refs[str(meter_source)] = meter_lock['source_sha256']
    for job in dlock['jobs']:
        path, receipt, metrics = verify_cell(DOPE / 'validation-v1', job, PINS['dope_validation'], receipt_hashes)
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
        child = safe(fit['child']['receipt'])
        require(sha256(child) == fit['child']['receipt_sha256'], 'DOPE child receipt changed')
        child_receipt = read(child)
        child_fits[str(child)] = child_receipt
        refs[str(child)] = sha256(child)
        charged = child_receipt['artifact_bytes']
        if fit['status'] == 'ok':
            model = child.with_suffix('.dpk')
            projection = safe(Path(job['worker']['path']) / 'projection.json')
            require(sha256(model) == child_receipt['artifact_sha256']
                    and sha256(projection) == job['worker']['files']['projection.json']
                    and model.stat().st_size + projection.stat().st_size == charged <= 10240,
                    'DOPE charged artifact changed')
        else:
            require(fit['status'] == 'charged_artifact_cap' and charged > 10240
                    and receipt['status'] == 'fit_unavailable', 'unavailable DOPE fit accounting differs')
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
        operations.extend(receipt['operations'])
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
    cost = {
        'dope_gpu_fit_seconds': sum(f['elapsed_seconds'] for f in child_fits.values()),
        'dope_dispatch_seconds_including_hash_and_admission_overhead': sum(f['elapsed_seconds'] for f in fits.values()),
        'dope_declared_gpu_fit_ceiling_seconds': 14400,
        'native_prior_48_trial_wall_seconds': sum(f['wall_seconds'] for f in native_fits.values()),
        'native_new_fits': 0, 'native_new_tuning_trials': 0,
        'sample_repeat_metric_operation_seconds': sum(r['elapsed_seconds'] for r in operations),
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
        'format': 'dope-s3-matched-discovery-validation-panel', 'version': 1,
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
        'scope': 'Single-fit-seed discovery research, not public-core paired analysis, five-fit stability, '
                 'confirmation or certification. Training cohorts are bounded subsets of official S3 training partitions; '
                 'native selection and these descriptive outcomes use training-derived validation. '
                 'All four DOPE profiles retained; production family unselected.',
        'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None, 'release_safe_l3': None,
        'production_certified': False, 'paired_superiority': None, 'counts_as_dope_win': False,
        'missing_evidence': ['shadow/privacy/static-artifact attacks', 'six production auditors and complete PTF profile',
                             'projection-only utility cost', 'product and informative-lineage coverage',
                             'public-core full frozen final matrix'],
        'source_locks': PINS | {'metric_implementation_sha256': dlock['metric_source_sha256']},
        'publication_integrity': {'receipt_lock_sha256': RECEIPT_LOCK_SHA256,
                                  'receipt_anchor_commit': receipt_lock['anchor_publication_commit'],
                                  'source_closure_verified_files': source_closure_file_count},
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
    lines = ['# S3 matched discovery validation', '', report['scope'], '',
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


def schema(document):
    """Freeze the complete shape and all claim/seal fields, leaving measured values typed."""
    def infer(value, name=''):
        if value is None:
            return {'type': 'null'}
        if isinstance(value, bool):
            return {'const': value}
        if isinstance(value, dict):
            return {'type': 'object', 'additionalProperties': False, 'required': sorted(value),
                    'properties': {k: infer(v, k) for k, v in value.items()}}
        if isinstance(value, list):
            # Per-cell status, bytes and utility differ, so retain all observed shapes.
            shapes = {json.dumps(infer(v), sort_keys=True): infer(v) for v in value}
            item = next(iter(shapes.values())) if len(shapes) == 1 else {'anyOf': list(shapes.values())}
            return {'type': 'array', 'minItems': len(value), 'maxItems': len(value),
                    'items': item if shapes else {}}
        if isinstance(value, int):
            return {'type': 'integer'}
        if isinstance(value, float):
            return {'type': 'number'}
        if name.endswith('sha256') or name in PINS:
            return {'type': 'string', 'pattern': '^[0-9a-f]{64}$'}
        return {'const': value} if name in ('format', 'partition', 'track', 'scope') else {'type': 'string'}
    result = infer(document)
    result['$schema'] = 'https://json-schema.org/draft/2020-12/schema'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE / 'results' / (NAME + '.json'))
    args = parser.parse_args()
    report = build()  # Refuse incomplete matrices before writing anything.
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    args.output.with_suffix('.schema.json').write_text(json.dumps(schema(report), sort_keys=True, indent=2) + '\n')
    table, markdown = tables(report)
    args.output.with_suffix('.csv').write_text(table)
    args.output.with_suffix('.md').write_text(markdown)
    print(args.output)


if __name__ == '__main__':
    main()
