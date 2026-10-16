"""Complete matched S3 confirmation validation, including original Forest-Flow."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import io
import json
import math
from pathlib import Path
from statistics import median

from research.benchmark.manifest import digest
from research.benchmark.publish_s3_confirmation import build as reference_build
from research.benchmark.publish_s3_confirmation import schema
from research.benchmark.publish_s3_matched import CONFIGS as REFERENCE_CONFIGS
from research.benchmark.score import sha256

BASE = Path('/mnt/fast-scratch/dope-benchmark')
NATIVE = BASE/'forest-s3-native-v1'
COMMON = BASE/'forest-s3-shared-validation-v2'
HERE = Path(__file__).resolve().parent
RESULTS = HERE/'results'
NAME = 's3-matched-forest-confirmation-validation'
PINS = {
    'native_round': '441e5d95153a236f3811522f6716e4d0da9acdb459aa84dbc295d0e9c5c21133',
    'native_receipts': 'b9be26c9a9c76c1829ca09309873993ffafb10d959b1489abc95f79aa5075e1f',
    'common_round': '003106f3ba73ae4a4e03edf1224520ac2e390024071d055028fe1a51a098b79c',
    'common_receipts': 'bf2c70825dbeb001dfb50abc0b1180800cdbea92425f1dfd901488f37f1c6214',
}
REFERENCE_SHA = '884d43e7c35e559f398c51df78b512f1224504e6c35cabac22554f3af6bd7d82'
CONFIGS = tuple(REFERENCE_CONFIGS) + (('ForestDiffusion/Forest-Flow', 'default'),
                                    ('ForestDiffusion/Forest-Flow', 'native_selected'))
AUDITORS = ('catboost', 'linear', 'mlp')
SEEDS = (101, 211, 307)
SIZES = (1, 4)

def require(value, reason):
    if not value: raise ValueError(reason)

def read(path):
    return json.loads(Path(path).read_text())

def safe(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True).is_relative_to(BASE)
            and not any(p.is_symlink() for p in (path, *path.parents))
            and 'evaluator' not in path.parts and path.name != 'test.csv',
            'evidence path outside sealed publication scope')
    return path

def check(path, expected, refs):
    path = safe(path)
    require(sha256(path) == expected, 'frozen publication evidence changed')
    require(str(path) not in refs or refs[str(path)] == expected, 'evidence identities disagree')
    refs[str(path)] = expected

def anchored(root, expected, refs):
    path = root/'receipt-lock-v1.json'
    check(path, expected, refs)
    anchor = read(path)
    # Parent metric files and receipts cannot redefine their own digest.
    for name, h in anchor['refs'].items(): check(name, h, refs)
    reconciliation = root/'reconciliation-v1.json'
    check(reconciliation, anchor['reconciliation_sha256'], refs)
    require(read(reconciliation)['complete_matrix'] is True, 'frozen matrix incomplete')
    require(read(root/'coordinator-exit.json')['exit_code'] == 0, 'coordinator did not close')
    return anchor, read(reconciliation)

def native_inventory(lock, refs):
    path = NATIVE/'runtime-inventory.lock.json'
    check(path, lock['runtime_inventory_sha256'], refs)
    inventory = read(path)
    require(inventory['aliases'] == [], 'native runtime gained an alias')
    require(set(inventory['inventory_modes']) == set(inventory['roots'])
            and all(m == 'all' for m in inventory['inventory_modes'].values()),
            'native inventory scope changed')
    for name in inventory['roots']:
        root = safe(name)
        require(not any(p.is_symlink() for p in root.rglob('*')), 'native directory alias added')
        require({str(p) for p in root.rglob('*') if p.is_file()}
                == {p for p in inventory['files'] if Path(p).is_relative_to(root)},
                'native source or executable bytecode inventory changed')
    for name, row in inventory['files'].items():
        require(Path(name).stat().st_size == row['bytes'], 'native runtime file size changed')
        check(name, row['sha256'], refs)
    return len(inventory['files'])

def shared_inventory(lock, refs):
    runtime_path = BASE/'shared-validation-python312-preparation-v2/runtime.lock.json'
    auxiliary_path = BASE/'shared-validation-python312-auxiliary-v1/auxiliary.lock.json'
    for path in (runtime_path, auxiliary_path): check(path, lock['runtime_lock_files'][str(path)], refs)
    runtime, auxiliary = read(runtime_path), read(auxiliary_path)
    require(auxiliary['runtime_sha256'] == refs[str(runtime_path)], 'shared runtime identities disagree')
    for row in (runtime, auxiliary):
        root = safe(row['site'])
        require(not any(p.is_symlink() for p in root.rglob('*')), 'shared import alias added')
        require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} == set(row['files']),
                'shared import inventory changed')
        for name, item in row['files'].items():
            require((root/name).stat().st_size == item['bytes'], 'shared runtime file size changed')
            check(root/name, item['sha256'], refs)
    library = Path('/usr/lib/python3.12')
    require({str(p) for p in library.rglob('*') if p.is_file()} == set(auxiliary['stdlib_all_files'])
            and {str(p) for p in library.rglob('*') if p.is_symlink()} == set(auxiliary['stdlib_aliases']),
            'shared interpreter import inventory changed')
    for name, row in auxiliary['stdlib_all_files'].items():
        path = Path(name)
        require(path.stat().st_size == row['bytes'] and sha256(path) == row['sha256'],
                'shared interpreter source or bytecode changed')
    for name, row in auxiliary['stdlib_aliases'].items():
        path = Path(name)
        require(str(path.readlink()) == row['target'] and str(path.resolve(strict=True)) == row['resolved']
                and sha256(path) == row['sha256'], 'pinned interpreter alias changed')
    require(sha256(Path(runtime['python'])) == runtime['python_sha256']
            and str(Path('/usr/bin/python3').readlink()) == runtime['python_symlink']
            and not Path(runtime['absent_zip']).exists(), 'shared interpreter executable changed')
    check(runtime['metric_source'], runtime['metric_sha256'], refs)
    require(runtime['metric_sha256'] == lock['metric_source_sha256']
            and not list(safe(runtime['empty_bytecode_cache']).rglob('*')), 'shared metric runtime changed')
    return runtime

def summarize(cells, datasets):
    expected = {(d, m, c, s, z) for d in datasets for m, c in CONFIGS for s in SEEDS for z in SIZES}
    actual = [(r['dataset'], r['method'], r['configuration'], r['sample_seed'], r['size_multiplier']) for r in cells]
    require(len(actual) == len(set(actual)) and set(actual) == expected, 'matched matrix incomplete or duplicated')
    groups = defaultdict(list)
    for row in cells:
        require(row['fit_seed'] == 11 and row['counts_as_dope_win'] is False
                and all(row[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe_l3')),
                'research cell scope changed')
        groups[(row['dataset'], row['method'], row['configuration'], row['size_multiplier'])].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda r: r['sample_seed']); utility = {}
        require(len({r['charged_artifact_bytes'] for r in rows}) == 1, 'cell artifact charges disagree')
        for auditor in AUDITORS:
            values = [r['utility'][auditor]['retention'] if r['status'] == 'ok' and r['utility']
                      and r['utility'][auditor]['informative'] is True else None for r in rows]
            require(all(v is None or math.isfinite(v) for v in values), 'nonfinite shared utility')
            complete = all(v is not None for v in values)
            utility[auditor] = {'complete_informative_sample_group': complete,
                                'median_retention': median(values) if complete else None,
                                'sample_retention_values': values}
        result.append({'dataset': key[0], 'method': key[1], 'configuration': key[2], 'size_multiplier': key[3],
                       'fit_seed': 11, 'sample_cells': 3, 'statuses': dict(Counter(r['status'] for r in rows)),
                       'charged_artifact_bytes': rows[0]['charged_artifact_bytes'], 'utility': utility})
    return result

def common_operation(receipt, lock):
    require(receipt['official_tests_opened'] is False and receipt['new_generator_fits_started'] == 0
            and receipt['native_selection_changed'] is False
            and all(receipt[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')),
            'common receipt claim or test seal changed')
    operation = receipt['operation']
    if receipt['status'] == 'ok':
        require(operation['status'] == 'ok' and operation['exit_code'] == 0
                and operation['new_operation_started'] is True
                and operation['foreign_processes_signaled'] is False
                and operation['runtime_lock_files'] == lock['runtime_lock_files']
                and math.isfinite(operation['elapsed_seconds'])
                and 0 < operation['timeout_seconds'] <= 600
                and 0 <= operation['elapsed_seconds'] <= operation['timeout_seconds'] + 10
                and 0 <= operation['peak_resident_bytes_including_coordinator'] <= lock['roles']['coordinator']['ram_bytes'],
                'successful cell failed a runtime or budget gate')

def build():
    require(set(PINS) == {'native_round', 'native_receipts', 'common_round', 'common_receipts'}
            and all(isinstance(v, str) and len(v) == 64 for v in PINS.values()),
            'publication has no complete frozen anchors')
    refs = {}
    for name, root in (('native', NATIVE), ('common', COMMON)):
        check(root/'round.lock.json', PINS[name+'_round'], refs)
    native_anchor, native_report = anchored(NATIVE, PINS['native_receipts'], refs)
    common_anchor, common_report = anchored(COMMON, PINS['common_receipts'], refs)
    native = read(NATIVE/'round.lock.json'); common = read(COMMON/'round.lock.json')
    require(native_anchor['round_sha256'] == PINS['native_round']
            and common_anchor['round_sha256'] == PINS['common_round']
            and common['native_receipt_lock_sha256'] == PINS['native_receipts']
            and common['native_selections'] == {k: v | {'selection_evidence_sha256': refs[str(NATIVE/'reconciliation-v1.json')]}
                                                for k, v in native_report['selections'].items()},
            'native selections or common round bindings changed')
    for root, lock in ((NATIVE, native), (COMMON, common)):
        require(lock['official_tests_opened'] is False and lock['mfs_v2'] is None and lock['ptf_v1'] is None,
                'test seal or gated score changed')
        require({str(p) for p in (root/'source').rglob('*') if p.is_file()} == set(lock['source_files'])
                and not any(p.is_symlink() for p in (root/'source').rglob('*')), 'frozen source inventory changed')
        for name, h in lock['source_files'].items(): check(name, h, refs)
        for name, h in lock['frozen_references'].items(): check(name, h, refs)
    native_runtime_files = native_inventory(native, refs)
    runtime = shared_inventory(common, refs)
    prior_path = RESULTS/'s3-matched-confirmation-validation.json'
    require(sha256(prior_path) == REFERENCE_SHA, 'committed matched reference changed')
    prior = reference_build()
    require(prior == read(prior_path) and prior['source_locks']['metric_implementation_sha256'] == common['metric_source_sha256'],
            'matched reference or common evaluator differs')
    require(prior['dataset_ids'] == sorted(common['native_selections']), 'matched datasets differ')
    cells = list(prior['cells']); native_attempts = []; operation_cost = 0.; peaks = []; device_energy = []
    fits = {}
    for job in native['jobs']:
        key = digest(job); out = NATIVE/'attempts'/key/'attempt-0001'
        receipt = read(out/'receipt.json'); fitted = read(out/'fit.json'); objective = read(out/'native.json')
        require(receipt['job'] == job and receipt['round_sha256'] == PINS['native_round'], 'native attempt identity changed')
        require(objective['name'] == 'author_mean_ml_r2' and objective['direction'] == 'maximize'
                and objective['implementation'] == native['native_implementation']
                and objective['shared_kpi_used_for_selection'] is False
                and objective['auditor_fit_seeds'] == [0, 1, 2, 3, 4]
                and math.isfinite(objective['value']), 'native objective changed')
        scores = objective['auditor_seed_scores']
        require(set(scores) == {'linear', 'adaboost', 'random_forest', 'xgboost'}
                and all(len(v) == 5 and all(math.isfinite(x) for x in v) for v in scores.values())
                and math.isclose(objective['value'], sum(sum(v)/5 for v in scores.values())/4, abs_tol=1e-14, rel_tol=0),
                'native objective components differ')
        artifact = safe(out/'artifact')
        require(not any(p.is_symlink() for p in artifact.rglob('*'))
                and {p.relative_to(artifact).as_posix() for p in artifact.rglob('*') if p.is_file()}
                == {r['path'] for r in fitted['artifact_inventory']}, 'native artifact inventory changed')
        require(fitted['artifact_bytes'] == sum(r['bytes'] for r in fitted['artifact_inventory'])
                and any(r['path'] == 'projection.json' for r in fitted['artifact_inventory']), 'projection or bytes uncharged')
        worker = safe(job['worker']['path'])
        require(set(job['worker']['files']) == {'train.csv', 'validation.csv', 'projection.json', 'worker-manifest.json'}
                and not (worker/'test.csv').exists(), 'worker contains official test inputs')
        for name, h in job['worker']['files'].items(): check(worker/name, h, refs)
        require(sha256(artifact/'projection.json') == job['worker']['files']['projection.json'], 'artifact projection differs')
        for row in fitted['artifact_inventory']:
            require((artifact/row['path']).stat().st_size == row['bytes'], 'artifact bytes changed')
            check(artifact/row['path'], row['sha256'], refs)
        require(fitted['original_sample_parity'] == 'exact' and fitted['retained_training_row_containers'] is False
                and fitted['trained_booster_devices'] and all(d.startswith('cuda') for d in fitted['trained_booster_devices']),
                'original GPU fit contract changed')
        batch = read(out/'sample.json')
        require(batch['sample_replay_exact'] is True
                and sha256(out/'n-seed101.csv') == sha256(out/'repeat.csv')
                and [r['schedule'] for r in batch['samples']] == native['common_sample_schedule'],
                'native generator replay or sample schedule differs')
        for operation in receipt['operations']:
            if operation['new_operation_started']: operation_cost += operation['elapsed_seconds']
        for path in out.glob('*.monitor.json'):
            monitor = read(path)
            require(monitor['foreign_processes_signaled'] is False, 'foreign owner was signaled')
            peaks.append(monitor['peak_gpu_used_mib'])
            points = monitor['whole_device_power_observations']
            require(all(math.isfinite(p['seconds']) and math.isfinite(p['whole_device_watts'])
                        and p['whole_device_watts'] >= 0 for p in points)
                    and all(a['seconds'] <= b['seconds'] for a, b in zip(points, points[1:])),
                    'invalid whole-device energy observations')
            device_energy.append(sum((b['seconds']-a['seconds'])*(a['whole_device_watts']+b['whole_device_watts'])/2
                                     for a, b in zip(points, points[1:])) if len(points) >= 2 else None)
        fits[key] = (job, fitted, objective)
        native_attempts.append({'dataset': job['dataset'], 'configuration': job['configuration_name'],
                                'config': job['config'], 'fit_seed': 11, 'status': receipt['status'],
                                'native_validation_kpi': objective, 'charged_artifact_bytes': fitted['artifact_bytes'],
                                'fit_core_seconds': fitted['fit_seconds'], 'receipt': {'path': str(out/'receipt.json'),
                                                                                     'sha256': refs[str(out/'receipt.json')]}})
    require(len(fits) == 12 and len(native_report['selections']) == 6, 'native trial matrix incomplete')
    for dataset, selected in native_report['selections'].items():
        trials = [(key, row) for key, row in fits.items() if row[0]['dataset'] == dataset]
        require(len(trials) == 2, 'native grid incomplete')
        eligible = [(key, row) for key, row in trials if row[2]['value'] is not None]
        winner = min(eligible, key=lambda pair: (-pair[1][2]['value'], pair[1][1]['artifact_bytes'], digest(pair[1][0]['config'])))
        require(selected['job_sha256'] == winner[0] and selected['shared_kpi_used_for_selection'] is False,
                'native winner differs from author objective')
    physical = {digest(j): j for j in common['jobs']}; require(len(physical) == len(common['jobs']), 'duplicate physical cell')
    for cell in common['logical_cells']:
        key = cell['physical_job_sha256']; job = physical.get(key); metric = None; receipt_ref = None
        if job:
            out = COMMON/'attempts'/key/'attempt-0001'; path = out/'receipt.json'; receipt = read(path)
            require(receipt['job'] == job and receipt['round_sha256'] == PINS['common_round']
                    and receipt['new_generator_fits_started'] == 0 and receipt['native_selection_changed'] is False,
                    'common job identity changed')
            common_operation(receipt, common)
            receipt_ref = {'path': str(path), 'sha256': refs[str(path)]}; status = receipt['status']
            require(job['worker']['files'] == fits[job['fit_job_sha256']][0]['worker']['files'], 'common worker differs')
            # Compare projection/input hashes with the prior matched sample receipts.
            prior_cell = next(r for r in prior['cells'] if r['dataset'] == job['dataset'] and r['status'] == 'ok')
            previous_receipt = read(prior_cell['receipt']['path'])
            previous_worker = previous_receipt['job']['worker']
            require(previous_worker['files'] == job['worker']['files'], 'reference partitions differ')
            if status == 'ok':
                metric = read(out/'metric.json')
                require(metric['dependencies'] == runtime['expected_versions']
                        and metric['implementation_sha256'] == common['metric_source_sha256']
                        and metric['mfs_v2'] is None and metric['gate_profile_complete'] is False,
                        'shared metric implementation changed')
                if job['replay_required']:
                    replay = read(out/'metric-replay.json')
                    values = lambda r: {k: v for k, v in r.items() if k != 'metric_seconds'}
                    require(values(metric) == values(replay), 'deterministic metric replay differs')
            charged = job['artifact_bytes']; native_kpi = fits[job['fit_job_sha256']][2]
        else:
            require(cell['unavailable_reason'] and cell['counts_as_dope_win'] is False, 'missing cell without receipt')
            status = 'unavailable'; charged = None; native_kpi = None
        cells.append({'dataset': cell['dataset'], 'method': 'ForestDiffusion/Forest-Flow',
                      'configuration': 'default' if cell['kind'] == 'default' else 'native_selected',
                      'fit_seed': 11, 'sample_seed': cell['sample_seed'], 'size_multiplier': cell['size_multiplier'],
                      'status': status, 'charged_artifact_bytes': charged, 'artifact_within_l3_cap': charged is not None and charged <= 10240,
                      'physical_job_sha256': key, 'receipt': receipt_ref, 'native_validation_kpi': native_kpi,
                      'utility': metric['utility'] if metric else None, 'exact_copy_count': metric['copy_counts']['exact'] if metric else None,
                      'near_copy_count': metric['copy_counts']['near'] if metric else None,
                      'real_vs_real_control_counts': metric['real_vs_real_control_counts'] if metric else None,
                      'marginal_ks_mean': metric['marginal_ks_mean'] if metric else None,
                      'pair_correlation_fidelity': metric['pair_correlation_fidelity'] if metric else None,
                      'c2st_auc': metric['c2st_auc'] if metric else None, 'mfs_v2': None, 'ptf_v1': None,
                      'release_safe_l3': None, 'counts_as_dope_win': False})
    # Inventory locks already bind each runtime file; public references retain
    # those immutable locks rather than duplicating tens of thousands of paths.
    runtime_roots = [Path(p) for p in read(NATIVE/'runtime-inventory.lock.json')['roots']]
    runtime_roots += [Path(runtime['site']), Path(read(BASE/'shared-validation-python312-auxiliary-v1/auxiliary.lock.json')['site'])]
    public_refs = {p: h for p, h in refs.items() if not any(Path(p).is_relative_to(root) for root in runtime_roots)}
    report = {'format': 'dope-s3-matched-forest-confirmation-validation-panel', 'version': 1,
              'source_sha256': sha256(Path(__file__)), 'datasets': 6, 'dataset_ids': prior['dataset_ids'],
              'dataset_metadata': prior['dataset_metadata'], 's3_data_lock_sha256': prior['s3_data_lock_sha256'],
              'partition': prior['partition'], 'track': prior['track'], 'fit_seeds': [11], 'sample_seeds': list(SEEDS),
              'sizes': list(SIZES), 'logical_sample_cells': len(cells), 'logical_status_counts': dict(Counter(r['status'] for r in cells)),
              'all_frozen_cells_accounted': True,
              'planned_physical_sample_cells': prior['planned_physical_sample_cells'] + len(physical),
              'successful_physical_sample_cells': prior['successful_physical_sample_cells'] + common_report['physical_status_counts'].get('ok', 0),
              'default_native_aliases_charge_once': True, 'reference_publication': {'path': 'research/benchmark/results/s3-matched-confirmation-validation.json',
                                                                        'sha256': REFERENCE_SHA},
              'scope': 'Single-fit-seed matched S3 confirmation validation. Training cohorts are bounded subsets of official S3 training partitions. '
                       'These descriptive validation outcomes are not public-core paired analysis, five-fit stability or certification. '
                       'All four DOPE profiles remain visible; no production family is selected.',
              'native_objective_policy': 'Original Forest-Flow selects from its frozen two-configuration grid by the author mean regression R2 '
                                         'over linear, AdaBoost, random forest and XGBoost auditors, five auditor fit seeds. CTGAN/TVAE retain '
                                         'their prior frozen native efficacy selections. Native values are never ranked across methods.',
              'native_attempts': native_attempts, 'native_selections': native_report['selections'], 'source_locks': PINS,
              'author_faithful_paper_dependency_reproduction': False,
              'native_implementation': native['native_implementation'],
              'publication_integrity': {'native_runtime_files_verified': native_runtime_files, 'shared_versions': runtime['expected_versions'],
                                        'historical_full_executable_closure_certified': False, 'system_dynamic_library_closure_certified': False,
                                        'external_receipt_digests_verified_before_metric_reads': True},
              'cost': {'forest_new_gpu_fits': 12, 'forest_fit_core_seconds': sum(row[1]['fit_seconds'] for row in fits.values()),
                       'forest_fit_native_sample_operation_seconds': operation_cost, 'forest_gpu_host': 'xbabe1',
                       'forest_shared_evaluator_seconds': common_report['shared_evaluator_seconds'], 'shared_evaluator_host': 'xbabe2',
                       'forest_peak_gpu_used_mib': max(peaks), 'forest_whole_device_energy_estimate_joules': sum(device_energy)
                       if all(v is not None for v in device_energy) else None, 'forest_cpu_energy_joules': None,
                       'prior_comparison_new_fits': 0, 'prior_comparison_new_tuning_trials': 0,
                       'limitations': 'Prior immutable metrics add no new compute. Operation time includes hashing, generation and native auditors; '
                                      'device power includes idle and is not job-attributed. Earlier DOPE architecture research is a separate cost; '
                                      'per-cell budgets do not imply equal total research spend.'},
              'cells': cells, 'summary': summarize(cells, prior['dataset_ids']),
              'immutable_references': [{'path': p, 'sha256': h} for p, h in sorted(public_refs.items())],
              'missing_evidence': prior['missing_evidence'], 'citation_keys': prior['citation_keys'] + ['jolicoeurmartineau2024generating'],
              'campaign_complete': False, 'official_tests_opened': False, 'production_certified': False,
              'mfs_v2': None, 'ptf_v1': None, 'release_safe_l3': None, 'paired_superiority': None, 'counts_as_dope_win': False}
    return report

def tables(report):
    stream = io.StringIO(); writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['dataset', 'display_name', 'method', 'configuration', 'fit_seed', 'size_multiplier',
                     'charged_bytes', 'statuses', *[a+'_median_retention' for a in AUDITORS]])
    lines = ['# S3 matched confirmation with original Forest-Flow', '', report['scope'], '',
             'Official tests sealed. MFS-v2 / PTF-v1 / release-safe L3: null. One fit seed; sample medians have no fit uncertainty.', '',
             '| Dataset | Method/config | Size | Charged bytes | Statuses | CatBoost | Linear | MLP |',
             '|---|---|---:|---:|---|---:|---:|---:|']
    names = {r['id']: r['display_name'] for r in report['dataset_metadata']}
    for row in report['summary']:
        values = [row['utility'][a]['median_retention'] for a in AUDITORS]
        states = json.dumps(row['statuses'], sort_keys=True)
        writer.writerow([row['dataset'], names[row['dataset']], row['method'], row['configuration'], 11,
                         row['size_multiplier'], row['charged_artifact_bytes'], states, *values])
        formatted = ['null' if v is None else f'{v:.6f}' for v in values]
        lines.append(f"| {names[row['dataset']]} | {row['method']}/{row['configuration']} | {row['size_multiplier']}n | "
                     f"{row['charged_artifact_bytes']} | {states} | " + ' | '.join(formatted) + ' |')
    lines.extend(['', report['native_objective_policy'], '', 'All sampler and projection bytes are charged. '
                  'Baseline values with artifacts over 10,240 bytes describe unconstrained quality. '
                  'Copy counts and controls are incomplete privacy evidence; failed or unavailable cells never supply a DOPE win.', '',
                  'Cost: ' + json.dumps(report['cost'], sort_keys=True), '', 'Citation keys: ' + ', '.join(report['citation_keys']), ''])
    return stream.getvalue(), '\n'.join(lines)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, default=RESULTS/(NAME+'.json'))
    parser.add_argument('--from-json', action='store_true', help='Regenerate tables from committed JSON only')
    args = parser.parse_args()
    if args.from_json:
        report = read(args.output)
        require(report['official_tests_opened'] is False and report['production_certified'] is False
                and all(report[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe_l3', 'paired_superiority'))
                and summarize(report['cells'], report['dataset_ids']) == report['summary'],
                'committed matched panel scope or summary changed')
    else:
        report = build()
        args.output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n')
        args.output.with_suffix('.schema.json').write_text(json.dumps(schema(report), sort_keys=True, indent=2)+'\n')
    table, markdown = tables(report); args.output.with_suffix('.csv').write_text(table); args.output.with_suffix('.md').write_text(markdown)
    print(args.output)

if __name__ == '__main__': main()
