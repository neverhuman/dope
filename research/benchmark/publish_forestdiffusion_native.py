"""Reconcile the complete ForestDiffusion native search and common pilot matrix."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
import math
from pathlib import Path
from statistics import median

from research.benchmark.manifest import digest
from research.benchmark.publish_forestdiffusion_source import native_mean, require, build as source_build
from research.benchmark.publish_native_neural_fivefit import evidence, sealed
from research.benchmark.score import sha256 as file_sha256

ROOT = Path('/mnt/fast-scratch/dope-benchmark/pilot-24h')
CPU = ROOT / 'forestdiffusion-native-v1'
GPU = ROOT / 'forestdiffusion-gpu-native-v1'
SELECTION_ROOT = ROOT / 'forestdiffusion-combined-native-v1'
COMMON = ROOT / 'forestdiffusion-common-validation-v1'
PINS = {
    'cpu': 'beb4dfe71edb1e03ecf96764410b63018cf91fdf3fd8cc858793038f6496a816',
    'gpu': '75a7d6ed753767007147143a8356dc7518ef0d67cc0e7222eab174b2bd230255',
    'selection': '784d2a4cf7830a917e6533718d1b25f2e3b24f0f5ed2f48ccc9e3130c1117f98',
    'common': '06dfeb4047bd7edf2b188664313ee74436350ef047e37306c05752ddec4d3b90',
}
DATASETS = ('Adult', 'California', 'News')
CONFIGURATIONS = ('cpu_author_default', 'gpu_author_default', 'native_selected')
SEEDS = (101, 211, 307)
SIZES = (1, 2, 4, 8)
AUDITORS = ('linear', 'catboost', 'mlp')
REFERENCE_PIN = '5a7605b2bbf3d46635c5d7bbeb4c922149bbd79e3bf75c6269cedd4637447dc3'


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return file_sha256(Path(path))


def ref(path):
    return {'path': str(path), 'sha256': sha256(path)}


def winner(attempts):
    require(len(attempts) == 8 and {r['job']['trial'] for r in attempts} == set(range(8)),
            'eight-trial native matrix incomplete')
    require(len({digest(r['job']['input_files']) for r in attempts}) == 1,
            'native trials use different partitions')
    require(sum(r['elapsed_seconds'] for r in attempts) <= 43200,
            'native method-dataset time cap exceeded')
    good = [r for r in attempts if r['status'] == 'ok']
    for row in good:
        native_mean(row['native_kpi'])
        require(row['native_kpi']['name'] == row['job']['objective'], 'native objective differs')
    return min(good, key=lambda r: (-r['native_kpi']['value'], r['artifact_bytes'],
                                   digest(r['job']['config'])), default=None)


def metric_payload(row):
    return {k: v for k, v in row.items() if k != 'metric_seconds'}


def summary(cells):
    expected = {(d, c, s, z) for d in DATASETS for c in CONFIGURATIONS for s in SEEDS for z in SIZES}
    keys = [(r['dataset'], r['configuration'], r['sample_seed'], r['size_multiplier']) for r in cells]
    require(len(keys) == len(set(keys)) and set(keys) == expected, 'logical common matrix incomplete')
    result = []
    for dataset in DATASETS:
        for configuration in CONFIGURATIONS:
            for size in (1, 4):
                group = [r for r in cells if (r['dataset'], r['configuration'], r['size_multiplier'])
                         == (dataset, configuration, size)]
                for auditor in AUDITORS:
                    values = [r['metrics']['utility'][auditor]['retention'] for r in group
                              if r['status'] == 'ok' and r['metrics'] is not None
                              and r['metrics']['utility'][auditor]['informative'] is True
                              and r['metrics']['utility'][auditor]['retention'] is not None]
                    require(all(math.isfinite(v) for v in values), 'nonfinite common retention')
                    complete = len(values) == len(SEEDS)
                    result.append({'dataset': dataset, 'configuration': configuration,
                                   'fit_seed': 23, 'size_multiplier': size, 'auditor': auditor,
                                   'informative_samples': len(values), 'median_retention': median(values) if complete else None,
                                   'min_retention': min(values) if complete else None,
                                   'max_retention': max(values) if complete else None,
                                   'artifact_bytes': group[0]['artifact_bytes'],
                                   'within_l3_bytes': group[0]['within_l3_bytes'],
                                   'sample_status_counts': dict(Counter(r['status'] for r in group))})
    return result


def inventory(root):
    result = []
    for p in sorted(root.rglob('*')):
        require(not p.is_symlink(), 'fit artifact symlink')
        if p.is_file():
            result.append({'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha256(p)})
    return result


def device_energy(operation):
    points = operation.get('power_observations', []) if operation else []
    if len(points) < 2:
        return None
    require(all(math.isfinite(p['elapsed_seconds']) and math.isfinite(p['watts']) and p['watts'] >= 0
                for p in points), 'invalid GPU power observations')
    require(all(a['elapsed_seconds'] <= b['elapsed_seconds'] for a, b in zip(points, points[1:])),
            'GPU power observations out of order')
    return sum((b['elapsed_seconds'] - a['elapsed_seconds']) * (a['watts'] + b['watts']) / 2
               for a, b in zip(points, points[1:]))


def matched_references(repo, workers, common):
    path = repo / 'research/benchmark/results/pilot24-target-gpu-research.json'
    require(sha256(path) == REFERENCE_PIN, 'committed matched reference changed')
    report = read(path)
    sealed(report)
    require(report['metric_source_sha256'] == common['metric_source_sha256']
            and report['worker_hashes'] == workers, 'reference projection or partitions differ')
    parent_ref = report['matched_baseline_report']
    parent_path = Path(parent_ref['path'])
    require(sha256(parent_path) == parent_ref['sha256'], 'prior neural report changed')
    parent = read(parent_path)
    sealed(parent)
    rows = []
    for row in report['eligible_cells']:
        if row['fit_seed'] != 23:
            continue
        p = Path(row['receipt']['path'])
        require(sha256(p) == row['receipt']['sha256'], 'DOPE research reference receipt changed')
        receipt = read(p)
        sealed(receipt)
        evidence(p.parent, receipt)
        metrics = row['metrics']
        if metrics is not None:
            metric_ref = row['metric_receipt']
            require(sha256(metric_ref['path']) == metric_ref['sha256']
                    and read(metric_ref['path']) == metrics, 'DOPE reference metric changed')
        rows.append({'dataset': row['dataset'], 'method': 'DOPE', 'configuration': row['profile'],
                     'fit_seed': 23, 'sample_seed': row['sample_seed'], 'size_multiplier': row['size_multiplier'],
                     'status': row['status'], 'artifact_bytes': row['artifact_bytes'],
                     'within_l3_bytes': row['artifact_bytes'] <= 10240,
                     'utility': metrics['utility'] if metrics else None,
                     'reference_receipt': row['receipt'], 'metric_replay_status': 'exact',
                     'source_unavailable': False, 'contributes_dope_win': False, 'release_safe': None})
    require(len(rows) == 72, 'DOPE research single-fit reference matrix incomplete')
    for row in report['matched_baseline_cells']:
        if row['fit_seed'] != 23:
            continue
        if row['sample_receipt_path']:
            p = Path(row['sample_receipt_path'])
            require(sha256(p) == row['sample_receipt_sha256'], 'prior baseline sample receipt changed')
            receipt = read(p)
            sealed(receipt)
            if row['status'] == 'ok':
                if 'metrics' in receipt:
                    metric = receipt['metrics']
                    require(sha256(p) == row['metric_receipt_sha256'], 'embedded baseline metric identity changed')
                else:
                    mp = p.parent / 'attempt-0001/metrics.json'
                    require(sha256(mp) == row['metric_receipt_sha256'], 'prior baseline metric changed')
                    metric = read(mp)
                require(metric['implementation_sha256'] == common['metric_source_sha256']
                        and metric['dependencies'] == common['metric_environment']
                        and metric['utility'] == row['utility']
                        and row['metric_replay_status'] in ('exact', 'prior_immutable_receipt_verified'),
                        'prior baseline common evaluator differs')
            reference = {'path': str(p), 'sha256': row['sample_receipt_sha256']}
        else:
            require(row['status'] == 'fit_unavailable', 'prior reference sample missing')
            fit = next(r for r in parent['fit_attempts'] if r['fit_receipt_sha256'] == row['fit_receipt_sha256'])
            require(sha256(fit['fit_receipt_path']) == fit['fit_receipt_sha256'] and fit['status'] != 'ok',
                    'prior failed fit changed')
            reference = {'path': fit['fit_receipt_path'], 'sha256': fit['fit_receipt_sha256']}
        rows.append({k: row[k] for k in ('dataset', 'method', 'configuration', 'fit_seed', 'sample_seed',
                                       'size_multiplier', 'status', 'artifact_bytes', 'within_l3_bytes',
                                       'utility', 'metric_replay_status')}
                    | {'reference_receipt': reference, 'source_unavailable': False,
                       'contributes_dope_win': False, 'release_safe': None})
    require(len(rows) == 120, 'prior single-fit baseline reference matrix incomplete')
    summaries = []
    for dataset, method, config in sorted({(r['dataset'], r['method'], r['configuration']) for r in rows}):
        for size in (1, 4):
            group = [r for r in rows if (r['dataset'], r['method'], r['configuration'], r['size_multiplier'])
                     == (dataset, method, config, size)]
            require(len(group) == 3 and {r['sample_seed'] for r in group} == set(SEEDS), 'reference sample schedule differs')
            for auditor in AUDITORS:
                values = [r['utility'][auditor]['retention'] for r in group if r['status'] == 'ok'
                          and r['utility'][auditor]['informative'] is True
                          and r['utility'][auditor]['retention'] is not None]
                require(all(math.isfinite(v) for v in values), 'nonfinite reference retention')
                summaries.append({'dataset': dataset, 'method': method, 'configuration': config,
                                  'fit_seed': 23, 'size_multiplier': size, 'auditor': auditor,
                                  'informative_samples': len(values),
                                  'median_retention': median(values) if len(values) == 3 else None,
                                  'min_retention': min(values) if len(values) == 3 else None,
                                  'max_retention': max(values) if len(values) == 3 else None,
                                  'artifact_bytes': group[0]['artifact_bytes'],
                                  'within_l3_bytes': group[0]['within_l3_bytes'],
                                  'sample_status_counts': dict(Counter(r['status'] for r in group))})
    return {'report': ref(path), 'cells': rows, 'summaries': summaries,
            'comparator_native_selections': report['matched_native_selections'],
            'notes': ['Same projected official-training-derived inputs, fit seed23, sample seeds101/211/307 and n/4n sizes.',
                      'All four DOPE research profiles retained; no family selected by this comparison.',
                      'Earlier CTGAN/TVAE seed23 metrics retain prior_immutable_receipt_verified status; no fresh replay claim.',
                      'Default and native-selected baseline configurations frozen before common evaluation; native values never ranked across methods.',
                      'Previously reported reference compute is not counted again as new fits or new sample operations.']}


def build(repo):
    locks = {name: read(root / 'round.lock.json') for name, root in
             (('cpu', CPU), ('gpu', GPU), ('selection', SELECTION_ROOT), ('common', COMMON))}
    for name, root in (('cpu', CPU), ('gpu', GPU), ('selection', SELECTION_ROOT), ('common', COMMON)):
        require(sha256(root / 'round.lock.json') == PINS[name], 'frozen round identity changed')
        sealed(locks[name])
    common = locks['common']
    require((COMMON / 'completion.json').exists(), 'common matrix still running')
    completion = read(COMMON / 'completion.json')
    sealed(completion)
    require(completion['round_sha256'] == PINS['common'] and completion['physical_cells'] == 12
            and completion['logical_cells'] == 108 and completion['new_fits_started'] == 0,
            'common completion identity changed')
    require(sha256(CPU / 'source/native_round.py') == locks['cpu']['source_sha256']
            and sha256(CPU / 'source/forestdiffusion_adapter.py') == locks['cpu']['adapter_source_sha256']
            and sha256(locks['cpu']['runtime_lock']['path']) == locks['cpu']['runtime_lock']['sha256']
            and sha256(GPU / 'source/gpu_native_round.py') == locks['gpu']['source_sha256']
            and sha256(GPU / 'source/resource_admission.py') == locks['gpu']['resource_admission_sha256'],
            'native frozen source or runtime changed')
    source = source_build()
    require(source == read(repo / 'research/benchmark/forestdiffusion-source.lock.json'),
            'verified source lock differs from committed evidence')
    for name, field in (('common.py', 'source_sha256'), ('forestdiffusion_adapter.py', 'adapter_sha256'),
                        ('resource_admission.py', 'resource_admission_sha256'),
                        ('inventory_hosts.py', 'inventory_hosts_sha256')):
        require(sha256(COMMON / 'source' / name) == common[field], 'common frozen source changed')
    require(sha256(SELECTION_ROOT / 'source/select.py') == locks['selection']['source_sha256']
            and common['selection_round_sha256'] == PINS['selection']
            and sha256(SELECTION_ROOT / 'completion.json') == common['selection_completion_sha256']
            and common['gpu_round_sha256'] == PINS['gpu']
            and common['new_fits_started'] == 0 and common['shared_kpi_used_for_selection'] is False,
            'common/native selection lineage changed')
    require(sha256(common['runtime_path']) == common['runtime_sha256']
            and sha256(repo / 'research/benchmark/forestdiffusion_adapter.py') == common['adapter_sha256'],
            'GPU runtime or committed adapter changed')
    attempts, by_receipt, workers = [], {}, {}
    for name, root in (('cpu', CPU), ('gpu', GPU)):
        for job in locks[name]['jobs']:
            p = root / 'jobs' / digest(job) / 'attempt-0001/receipt.json'
            row = read(p)
            sealed(row)
            evidence(p.parent, row)
            require(row['job'] == job and row['round_sha256'] == PINS[name], 'native attempt identity changed')
            worker = Path(job['worker_dir'])
            require('evaluator' not in worker.resolve().parts and not (worker / 'test.csv').exists(),
                    'native worker contains official tests')
            require(all(sha256(worker / f) == h for f, h in job['input_files'].items()), 'native worker hash changed')
            workers[job['dataset']] = job['input_files']
            require(job['fit_seed'] == 23 and job['native_auditor_fit_seeds'] == list(range(5)),
                    'native fit or auditor seeds changed')
            if row['fit_status'] == 'ok':
                fit = read(p.parent / 'fit.receipt.json')
                actual = inventory(p.parent / 'artifact')
                require(actual == fit['artifact_inventory']
                        and sum(v['bytes'] for v in actual) == fit['artifact_bytes'] == row['artifact_bytes'],
                        'complete fit artifact accounting changed')
            fit_operation = row['fit_operation']
            if fit_operation['status'] == 'ok':
                require(fit_operation['elapsed_seconds'] <= 600 and fit_operation['exit_code'] == 0,
                        'successful fit exceeds deadline')
            attempt = {**ref(p), 'job': job, 'status': row['status'], 'fit_status': row['fit_status'],
                       'native_status': row['native_status'], 'native_kpi': row['native_kpi'],
                       'artifact_bytes': row['artifact_bytes'], 'elapsed_seconds': row['elapsed_seconds'],
                       'execution': name, 'fit_peak_gpu_mib': fit_operation.get('peak_gpu_used_mib'),
                       'fit_peak_resident_bytes': fit_operation.get('peak_resident_bytes'),
                       'fit_gpu_device_energy_joules_estimate': device_energy(fit_operation),
                       'native_gpu_device_energy_joules_estimate': device_energy(row.get('native_operation')),
                       'evidence_files': row['evidence_files'],
                       'source_unavailable': False, 'contributes_dope_win': False}
            attempts.append(attempt)
            by_receipt[str(p)] = attempt
    selections = []
    for dataset in DATASETS:
        p = SELECTION_ROOT / 'selections' / (dataset + '.json')
        require(ref(p) == common['selections'][dataset], 'native selection receipt hash changed')
        selected = read(p)
        sealed(selected)
        group = [r for r in attempts if r['job']['dataset'] == dataset]
        best = winner(group)
        require(selected['round_sha256'] == PINS['selection'] and selected['shared_kpi_used_for_selection'] is False
                and selected['selected_trial'] == (best['job']['trial'] if best else None)
                and selected['tuning_trials'] == 8 and selected['new_fits_started'] == 0,
                'native winner disagrees with author objective')
        for parent in selected['attempts']:
            actual = by_receipt[parent['path']]
            require(all(parent[k] == actual[k] for k in ('sha256', 'job', 'status', 'native_kpi', 'artifact_bytes')),
                    'native selection attempt list changed')
        selections.append({'dataset': dataset, 'receipt': ref(p), 'selected_trial': selected['selected_trial'],
                           'selected_native_value': best['native_kpi']['value'] if best else None,
                           'objective': selected['objective'], 'direction': 'maximize',
                           'tuning_trials': 8, 'tuning_attempt_seconds': selected['tuning_attempt_seconds'],
                           'shared_kpi_used_for_selection': False,
                           'author_default_trials': [0, 6], 'source_unavailable': False, 'contributes_dope_win': False})
    require(len(attempts) == 24, 'native attempt matrix incomplete')
    physical = {}
    for key, job in common['physical_jobs'].items():
        p = COMMON / 'physical' / key / 'receipt.json'
        row = read(p)
        sealed(row)
        evidence(p.parent, row)
        require(row['job'] == job and row['round_sha256'] == PINS['common'] and row['new_fits_started'] == 0,
                'physical sample identity changed')
        for op in row['operations']:
            if op['status'] == 'ok':
                require(op['elapsed_seconds'] <= 600 and op['exit_code'] == 0, 'successful operation exceeds deadline')
            if op.get('peak_gpu_used_mib') is not None:
                require(op['peak_gpu_used_mib'] <= 16384, 'GPU operation memory cap exceeded')
            admission = p.parent / (op['tag'] + '.admission.json')
            require(sha256(admission) == op['admission_sha256'], 'operation admission receipt changed')
            admitted = read(admission)
            capacity, host = admitted['capacity'], admitted['host']
            require(capacity['admitted'] is True and not capacity['blockers']
                    and admitted['earlier_owners_absent'] is True
                    and capacity['reserved_ram_bytes'] <= host['memory']['MemTotal']
                    and capacity['allocated_scratch_bytes'] + capacity['reserved_scratch_bytes'] <= 200000000000
                    and host['host'] == 'xbabe1', 'aggregate common admission gap')
            if op['tag'] in ('sample', 'repeat'):
                require(not host['active_gpu_processes'] and host['gpus'][0]['memory_free_mib'] >= 17 * 1024
                        and capacity['reserved_gpu_jobs'] == 1, 'sample GPU admission gap')
        metrics = None
        if row['status'] == 'ok':
            sample, repeat = read(p.parent / 'sample.child.json'), read(p.parent / 'repeat.child.json')
            require(sample == repeat and sample['rows'] == job['rows']
                    and sha256(p.parent / 'sample.csv') == sha256(p.parent / 'repeat.csv') == sample['sample_sha256'],
                    'physical sample replay differs')
        if row['metric_status'] == 'ok':
            require(job['size_multiplier'] in (1, 4), 'metric outside frozen schedule')
            metrics = read(p.parent / 'metrics.json')
            replay = read(p.parent / 'metrics-replay.json')
            require(metric_payload(metrics) == metric_payload(replay)
                    and metrics['implementation_sha256'] == common['metric_source_sha256']
                    and metrics['dependencies'] == common['metric_environment']
                    and metrics['rows']['synthetic'] == job['rows'] and metrics['mfs_v2'] is None
                    and metrics['gate_profile_complete'] is False, 'common metric or exact replay differs')
        public_job = {k: v for k, v in job.items() if k != 'physical_job_key'}
        public_job['physical_job_sha256'] = job['physical_job_key']
        physical[key] = {**ref(p), 'job': public_job, 'status': row['status'], 'metric_status': row['metric_status'],
                         'artifact_bytes': row['artifact_bytes'], 'metrics': metrics,
                         'sample_sha256': row['evidence_files'].get('sample.csv'),
                         'operations': [{k: v for k, v in op.items() if k != 'power_observations'} for op in row['operations']],
                         'evidence_files': row['evidence_files'], 'new_fits_started': 0}
    require(len(physical) == 12, 'physical sample matrix incomplete')
    cells = []
    for job in common['jobs']:
        p = COMMON / 'cells' / digest(job) / 'receipt.json'
        row = read(p)
        sealed(row)
        parent = physical.get(job['physical_job_key'])
        expected_parent = {'path': parent['path'], 'sha256': parent['sha256']} if parent else None
        require(row['job'] == job and row['round_sha256'] == PINS['common']
                and row['physical_receipt'] == expected_parent
                and row['status'] == (parent['status'] if parent else 'fit_unavailable')
                and row['source_unavailable'] is False and row['contributes_dope_win'] is False,
                'logical alias receipt changed')
        fitted = by_receipt.get(job['fit_attempt']['path']) if job['fit_attempt'] else None
        charge = fitted['artifact_bytes'] if fitted else None
        cells.append({**ref(p), 'dataset': job['dataset'], 'configuration': job['configuration'], 'fit_seed': 23,
                      'sample_seed': job['sample_seed'], 'size_multiplier': job['size_multiplier'],
                      'status': row['status'], 'artifact_bytes': charge,
                      'within_l3_bytes': charge <= 10240 if charge is not None else None,
                      'physical_receipt': expected_parent, 'metrics': parent['metrics'] if parent else None,
                      'source_unavailable': False, 'contributes_dope_win': False, 'release_safe': None})
    summaries = summary(cells)
    matched = matched_references(repo, workers, common)
    operations = [op for p in physical.values() for op in p['operations']]
    return {'format': 'dope-forestdiffusion-native-common-validation-pilot', 'version': 1,
            'scope': 'single fit seed; three sample seeds; official-training-derived validation only',
            'source_sha256': sha256(__file__), 'fit_seeds': [23], 'sample_seeds': list(SEEDS),
            'sizes_sampled': list(SIZES), 'sizes_measured': [1, 4],
            'rounds': {name: ref(root / 'round.lock.json') for name, root in
                       (('cpu', CPU), ('gpu', GPU), ('selection', SELECTION_ROOT), ('common', COMMON))},
            'completion': ref(COMMON / 'completion.json'),
            'citation_keys': ['jolicoeurmartineau2024generating', 'xu2019modeling'],
            'worker_hashes': workers, 'fit_attempts': attempts, 'native_selections': selections,
            'matched_references': matched,
            'physical_sample_receipts': list(physical.values()), 'cells': cells, 'summaries': summaries,
            'fit_status_counts': dict(Counter(r['status'] for r in attempts)),
            'physical_status_counts': dict(Counter(r['status'] for r in physical.values())),
            'logical_status_counts': dict(Counter(r['status'] for r in cells)),
            'cost': {'benchmark_fits': 24, 'additional_sampling_fits': 0,
                     'native_attempt_seconds': sum(r['elapsed_seconds'] for r in attempts),
                     'physical_operations': len(operations),
                     'physical_operation_seconds': sum(r['elapsed_seconds'] for r in operations),
                     'native_gpu_device_energy_joules_estimate': sum(
                         (r['fit_gpu_device_energy_joules_estimate'] or 0)
                         + (r['native_gpu_device_energy_joules_estimate'] or 0) for r in attempts),
                     'native_gpu_energy_scope': 'trapezoidal recorded whole-device power windows including idle baseline; CPU energy unavailable',
                     'logical_aliases_do_not_add_fit_or_sampling_compute': True,
                     'common_energy_joules': None, 'common_energy_status': 'not_measured',
                     'common_controller_overhead_seconds': None, 'common_controller_overhead_status': 'not_measured'},
            'notes': ['Original MIT Python ForestDiffusion generator; study wrapper implements author ML objective.',
                      'CPU author default trial0 and GPU author default trial6 remain separate resource profiles.',
                      'Six CPU plus two GPU native trials per dataset, failures counted; no common-KPI selection.',
                      'California GPU default and native-selected tracks alias the same trial6 artifact and sample receipts.',
                      'Adult/News timeouts are executable-method failures, not source-unavailable exclusions or DOPE wins.',
                      'All artifact bytes charged; eligible ForestDiffusion artifacts exceed the 10240-byte L3 cap.',
                      'This single-fit validation pilot is not five-fit stability, public-core paired superiority or certification.',
                      'Operation times exclude common controller admission/hash/startup overhead; complete historical campaign R&D accounting remains unavailable.',
                      'Official tests sealed; privacy attacks, projection-only utility cost and full campaign coverage remain incomplete.'],
            'official_tests_opened': False, 'production_certified': False,
            'mfs_v2': None, 'ptf_v1': None, 'release_safe': None}


def csv_bytes(document):
    out = io.StringIO(newline='')
    keys = ('dataset', 'method', 'configuration', 'fit_seed', 'size_multiplier', 'auditor', 'informative_samples',
            'median_retention', 'min_retention', 'max_retention', 'artifact_bytes', 'within_l3_bytes')
    writer = csv.DictWriter(out, fieldnames=keys, lineterminator='\n')
    writer.writeheader()
    rows = [{**r, 'method': 'ForestDiffusion'} for r in document['summaries']]
    rows += document['matched_references']['summaries']
    writer.writerows({k: row.get(k) for k in keys} for row in rows)
    return out.getvalue().encode()


def markdown(document):
    lines = ['# ForestDiffusion native/default validation pilot', '',
             'One fit seed (23), three sample seeds, official-training-derived validation. Native ML KPIs selected before common evaluation. Official tests remain sealed; MFS-v2, PTF-v1 and release-safe scores are null.', '',
             '| Dataset | Native trial | Author ML KPI | Attempts |', '|---|---:|---:|---:|']
    for row in document['native_selections']:
        value = row['selected_native_value']
        lines.append(f"| {row['dataset']} | {row['selected_trial'] if row['selected_trial'] is not None else 'null'} | {value:.6f} | 8 |" if value is not None else f"| {row['dataset']} | null | null | 8 |")
    lines += ['', 'Native values use macro F1 for Adult and R² for California/News; they are not ranked across targets or against other methods’ native objectives.', '',
              '| Dataset | Configuration | n CatBoost retention | 4n CatBoost retention | Charged bytes |', '|---|---|---:|---:|---:|']
    for dataset in DATASETS:
        for config in CONFIGURATIONS:
            rows = [r for r in document['summaries'] if r['dataset'] == dataset and r['configuration'] == config and r['auditor'] == 'catboost']
            values = ['null' if r['median_retention'] is None else f"{r['median_retention']:.6f}" for r in rows]
            lines.append(f"| {dataset} | {config} | {values[0]} | {values[1]} | {rows[0]['artifact_bytes'] if rows[0]['artifact_bytes'] is not None else 'null'} |")
    lines += ['', f"Native attempts: `{json.dumps(document['fit_status_counts'], sort_keys=True)}`. Physical samples: `{json.dumps(document['physical_status_counts'], sort_keys=True)}`. All 108 logical cells: `{json.dumps(document['logical_status_counts'], sort_keys=True)}`.", '',
              'GPU default and native-selected California cells share unchanged artifacts/samples; aliases do not count as extra fits or repetitions. Failed defaults and unavailable native selections remain visible.', '', *[f'- {note}' for note in document['notes']], '']
    lines += ['## Matched California comparison', '',
              'Same fit seed23 and three sample seeds on identical projected official-training-derived inputs. All DOPE research profiles are retained; no winning family is selected here. Reference compute is already reported in the earlier panels. The CSV retains all three auditors and all datasets.', '',
              '| Method | Configuration | n CatBoost retention | 4n CatBoost retention | Charged bytes |', '|---|---|---:|---:|---:|']
    matched = document['matched_references']['summaries']
    for method, config in sorted({(r['method'], r['configuration']) for r in matched if r['dataset'] == 'California'}):
        rows = [r for r in matched if r['dataset'] == 'California' and r['method'] == method
                and r['configuration'] == config and r['auditor'] == 'catboost']
        values = ['null' if r['median_retention'] is None else f"{r['median_retention']:.6f}" for r in rows]
        lines.append(f"| {method} | {config} | {values[0]} | {values[1]} | {rows[0]['artifact_bytes'] if rows[0]['artifact_bytes'] is not None else 'null'} |")
    lines += ['', *[f'- {note}' for note in document['matched_references']['notes']], '']
    return '\n'.join(lines).encode()


def schema(document):
    """Rights-safe complete-panel shape; absent evidence cannot become a score."""
    def shape(value, key=''):
        if value is None:
            return {'type': 'null'}
        if isinstance(value, bool):
            return {'const': value}
        if isinstance(value, str):
            result = {'type': 'string', 'minLength': 1}
            if key.endswith('sha256') or len(value) == 64 and all(c in '0123456789abcdef' for c in value):
                result['pattern'] = '^[0-9a-f]{64}$'
            return result
        if isinstance(value, (int, float)):
            result = {'type': 'integer' if isinstance(value, int) else 'number'}
            if key in ('artifact_bytes', 'rows', 'elapsed_seconds', 'fit_seed', 'sample_seed',
                       'size_multiplier', 'informative_samples', 'bytes'):
                result['minimum'] = 0
            return result
        if isinstance(value, list):
            variants = {json.dumps(shape(r), sort_keys=True) for r in value}
            return {'type': 'array', 'minItems': len(value), 'maxItems': len(value),
                    'items': {'anyOf': [json.loads(r) for r in sorted(variants)]} if variants else {}}
        return {'type': 'object', 'additionalProperties': False, 'required': sorted(value),
                'properties': {k: shape(v, k) for k, v in value.items()}}
    result = shape(document)
    result['$schema'] = 'https://json-schema.org/draft/2020-12/schema'
    for name in ('format', 'version', 'fit_seeds', 'sample_seeds', 'sizes_sampled', 'sizes_measured'):
        result['properties'][name] = {'const': document[name]}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--schema', type=Path)
    args = parser.parse_args()
    document = build(args.repo)
    args.output.write_text(json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + '\n')
    args.output.with_suffix('.csv').write_bytes(csv_bytes(document))
    args.output.with_suffix('.md').write_bytes(markdown(document))
    if args.schema:
        args.schema.write_text(json.dumps(schema(document), sort_keys=True, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
