"""Publish complete native/default ARF common outcomes after actual closure."""
import argparse
import ast
from collections import Counter
import csv
import io
import hashlib
import json
import math
from pathlib import Path
import statistics

from . import publish_arf_population_native as native
from . import publish_dope_refinement_population as dope
from . import arf_runtime_guard as guard
from .manifest import digest
from .publish_dope_refinement_discovery import summaries
from .publish_s3_forest import shared_inventory

BASE = native.BASE
ROOT = BASE / 'arf-s3-population-shared-validation-v1'
SAMPLING = BASE / 'arf-s3-population-sampling-v1'
SAMPLE_ROUND = 'bf17d9c4b2d76736036dec82adfd98447e94f66fd8d7bb56675d4e8245012777'
NATIVE_RECEIPTS = '238a19ea24694348b0cb6acd43b5203ac75989ee6ac5c714ef86c32ff040e61d'
NATIVE_REPORT = 'af722bff624b2b5cad37eb10a314767dcbd2b00b085c44114f8ef398de9c3850'
NATIVE_PUBLICATION = '3bac30656933cac2c243b7bf6af654025a0930ed94828dd371c6fb2ba335692e'
DOPE_PUBLICATION = 'f36b952f360edeb9cb1bc777aa886efbb4a7c7d4dbe2979d33d1d6da831d3240'
NAME = 'arf-matched-population-validation'
CONFIGS = ('author_default', 'native_selected')
PROFILES = ('features12_steps2048', *dope.fits.PROFILES)
GATES = dict(dope.fits.GATES, native_selection_changed=False, native_values_ranked_across_methods=False,
             global_configuration_tuned_or_selected=False, final_five_fit_coverage_complete=False,
             privacy_attack_coverage_complete=False, system_dynamic_library_closure_certified=False)


def complete_matrix(cells):
    ids = sorted({c['dataset'] for c in cells})
    keys = [(c['dataset'], c['configuration'], c['size_multiplier'], c['sample_seed']) for c in cells]
    guard.require(len(ids) == 100 and len(keys) == len(set(keys)) == 1200
        and set(keys) == {(d, c, n, s) for d in ids for c in CONFIGS for n in dope.SIZES for s in dope.SEEDS},
        'complete native/default 1200-cell matrix required')
    for row in cells:
        guard.require(row['method'] == 'ARF' and type(row['fit_seed']) is int and row['fit_seed'] == 11
            and type(row['sample_seed']) is int and type(row['size_multiplier']) is int
            and row['projection_bytes_included'] is True and row['counts_as_dope_win'] is False
            and all(row[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')),
            'common ARF identity or gate differs')
        amount = row['charged_artifact_bytes']
        guard.require(type(amount) is int and amount > 0
            and row['l3_artifact_cap_satisfied'] is (amount <= 10240), 'complete ARF charge differs')
        if row['status'] == 'ok':
            guard.require(row['utility'] is not None and row['unavailable_reason'] is None,
                          'successful ARF sample lacks measured outcomes')
        else:
            guard.require(row['utility'] is None and row['unavailable_reason'] is not None,
                          'unavailable ARF sample hid its reason')
    return ids


def reference_cells(ids):
    directory = Path(dope.__file__).with_name('results')
    report = guard.bound(directory / (dope.NAME + '.json'), DOPE_PUBLICATION, directory)
    dope.validate_report(report)
    density = guard.bound(directory / 'density-matched-population-validation.json', dope.REFERENCE, directory)
    rows = report['cells'] + [c for c in density['cells']
                             if c['method'] == 'DOPE' and c['configuration'] == PROFILES[0]]
    guard.require(len(rows) == 1800 and {c['dataset'] for c in rows} == set(ids),
                  'complete fixed DOPE comparison views required')
    return rows


def paired(groups):
    lookup = {(g['dataset'], g['method'], g['configuration'], g['size_multiplier']): g for g in groups}
    ids = sorted({g['dataset'] for g in groups})
    result = []
    for profile in PROFILES:
        for configuration in CONFIGS:
            for size in dope.SIZES:
                for auditor in ('catboost', 'linear', 'mlp'):
                    values = []
                    for dataset in ids:
                        left = lookup[dataset, 'DOPE', profile, size]['utility'][auditor]
                        right = lookup[dataset, 'ARF', configuration, size]['utility'][auditor]
                        if left['complete_informative'] and right['complete_informative']:
                            values.append((left['median_retention'], right['median_retention']))
                    result.append(dict(configuration=profile, reference_method='ARF',
                        reference_configuration=configuration, size_multiplier=size, auditor=auditor,
                        planned_lineages=len(ids), paired_complete_informative_lineages=len(values),
                        dope_median_retention=statistics.median(x for x, _ in values) if values else None,
                        reference_median_retention=statistics.median(y for _, y in values) if values else None,
                        median_paired_difference=statistics.median(x-y for x, y in values) if values else None,
                        superiority=None))
    return result


def closed(root, lock, anchor, expected, physical_jobs, logical_cells):
    refs = anchor['refs']
    actual = native.bound(root / 'coordinator-exit.json', refs[str(root / 'coordinator-exit.json')])
    done = native.bound(root / 'completion.json', refs[str(root / 'completion.json')])
    launch = native.bound(root / 'supervisor-launch.json', refs[str(root / 'supervisor-launch.json')])
    guard.require(type(actual['exit_code']) is int and actual['exit_code'] == 0
        and actual['round_sha256'] == done['round_sha256'] == launch['round_sha256'] == expected
        and done['jobs'] == physical_jobs and done['logical_cells'] == logical_cells
        and all(not Path('/proc', str(launch[k])).exists() for k in ('pid', 'supervisor_pid')),
        'actual complete common closure required')
    native.flat(root / 'source', {Path(p).name: h for p, h in lock['source_files'].items()})
    return actual


def parent_references(anchor, refs):
    """Require every pinned parent leaf in the already authenticated flat map."""
    pending = [anchor]; visited = set()
    while pending:
        parent = pending.pop()
        for path, pin in parent['refs'].items():
            guard.require(refs.get(path) == pin, 'transitive parent evidence absent or inconsistent')
            if Path(path).name == 'receipt-lock-v1.json' and path not in visited:
                visited.add(path); pending.append(native.bound(path, pin))


def sampling_directories(out, refs):
    """Replay the verified original coordinator's literal TMPDIR declaration."""
    if not any(p.is_dir() for p in out.iterdir()): return []
    source = guard.safe(SAMPLING / 'source' / 'coordinator.py', BASE)
    guard.require(refs.get(str(source)) == guard.hash_file(source), 'sampling directory source anchor differs')
    nodes = list(ast.walk(ast.parse(source.read_bytes())))
    variables = [value.args[0].id for node in nodes if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and key.value == 'TMPDIR'
        and isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'str'
        and len(value.args) == 1 and isinstance(value.args[0], ast.Name)]
    names = [node.value.right.value for node in nodes if isinstance(node, ast.Assign)
        and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in variables and isinstance(node.value, ast.BinOp)
        and isinstance(node.value.op, ast.Div) and isinstance(node.value.left, ast.Name)
        and node.value.left.id == 'out' and isinstance(node.value.right, ast.Constant)]
    guard.require(len(variables) == len(names) == 1 and type(names[0]) is str
        and Path(names[0]).name == names[0] and names[0] not in ('', '.', '..'),
        'sampling directory declaration differs')
    return [str(out / names[0])] if (out / names[0]).exists() else []


def sampling_records(original, receipt, job, key, refs):
    """Authenticate the original batch and every declared generated sample."""
    out = SAMPLING / 'attempts' / key / 'attempt-0001'
    files = receipt['evidence_files']; inventory = dict(files, **{'receipt.json': original['receipt_sha256']})
    guard.require(all(Path(name).name == name and refs.get(str(out / name)) == pin
                      for name, pin in inventory.items()), 'original sampling output anchor differs')
    directories = sampling_directories(out, refs)
    guard.inventory(out, {str(out / n): dict(sha256=h, bytes=(out / n).stat().st_size)
                         for n, h in inventory.items()}, BASE, directories)
    if original['status'] != 'ok':
        guard.require(original['samples'] == [], 'failed sampling acquired successful records'); return
    batch = native.bound(out / 'sample-batch.json', files['sample-batch.json'])
    guard.require(batch['job_sha256'] == key and batch['round_sha256'] == SAMPLE_ROUND
        and batch['status'] == 'ok' and batch['artifact_bytes'] == job['artifact_bytes']
        and batch['projection_bytes_included'] is True
        and digest(batch) == digest(receipt['result']), 'original sample batch differs')
    records = original['samples']; identities = {(s['size_multiplier'], s['sample_seed']) for s in records}
    guard.require(len(records) == 6 and identities == {(n, s) for n in dope.SIZES for s in dope.SEEDS}
        and all(type(s['size_multiplier']) is int and type(s['sample_seed']) is int for s in records)
        and digest([{k: v for k, v in s.items() if k != 'sample_path'} for s in records])
        == digest(batch['samples']), 'original sample records differ')
    for sample in records:
        size, seed = sample['size_multiplier'], sample['sample_seed']
        path = out / f'n{size}-seed{seed}.csv'
        guard.require(sample['sample_file'] == path.name and sample['sample_path'] == str(path)
            and refs.get(str(path)) == files.get(path.name) == sample['sample_sha256']
            and type(sample['rows']) is int and sample['rows'] == job['worker']['train_rows'] * size
            and type(sample['columns']) is int and sample['columns'] == job['worker']['projected_features'] + 1
            and sample['sample_replay'] == ('exact' if (size, seed) == (1, 101) else 'not_repeated'),
            'original generated sample identity differs')
        if sample['sample_replay'] == 'exact':
            replay = out / 'n1-seed101.repeat.csv'
            guard.require(refs.get(str(replay)) == files.get(replay.name) == sample['sample_sha256'],
                          'original sampling replay differs')


def sampling_receipts(sample_lock, sample_report, refs):
    """Bind upstream reconciliation declarations before any metric decoding."""
    jobs = {digest(j): j for j in sample_lock['jobs']}
    rows = sample_report['physical_batches_evidence']; batches = {r['job_sha256']: r for r in rows}
    guard.require(len(jobs) == len(sample_lock['jobs']) == len(rows) == len(batches) == 199
        and set(batches) == set(jobs), 'original sampling batch coverage differs')
    for key, row in batches.items():
        job = jobs[key]; path = SAMPLING / 'attempts' / key / 'attempt-0001' / 'receipt.json'
        guard.require(digest(row['job']) == key and type(job['fit_seed']) is int and job['fit_seed'] == 11
            and row['receipt_path'] == str(path) and refs.get(str(path)) == row['receipt_sha256']
            and row['artifact_bytes'] == job['artifact_bytes'], 'original sampling receipt anchor differs')
        receipt = native.bound(path, row['receipt_sha256'])
        guard.require(digest(receipt['job']) == key == receipt['job_sha256']
            and receipt['round_sha256'] == SAMPLE_ROUND and receipt['status'] == row['status']
            and receipt['official_tests_opened'] is False and receipt['new_generator_fits_started'] == 0
            and receipt['native_selection_changed'] is False
            and all(receipt[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')),
            'original sampling receipt lineage differs')
        sampling_records(row, receipt, job, key, refs)
    return jobs, batches


def common_sample(row, specification, jobs, sampling_batches, refs, lock, round_pin):
    """Bind a logical cell to its physical receipt and exact frozen sample."""
    guard.require(type(row['fit_seed']) is int and row['fit_seed'] == 11
        and type(row['sample_seed']) is int and type(row['size_multiplier']) is int,
        'input common seed identity differs')
    guard.require(digest({k: row[k] for k in specification}) == digest(specification),
                  'logical common cell differs from frozen schedule')
    key = specification['physical_job_sha256']; sample = row['sample_evidence']
    if key is None:
        guard.require(row['status'] == 'sampling_unavailable' and sample is None
            and row['validation_receipt_sha256'] is None, 'unavailable sample acquired metric evidence')
        return None
    job = jobs[key]; original = sampling_batches[row['sampling_job_sha256']]
    guard.require(type(job['fit_seed']) is int and job['fit_seed'] == row['fit_seed']
        and type(original['job']['fit_seed']) is int and original['job']['fit_seed'] == row['fit_seed']
        and job['method'] == 'ARF' and job['final'] is False and job['track'] == 'common-numeric'
        and job['dataset'] == row['dataset'] and job['sampling_job_sha256'] == row['sampling_job_sha256']
        and job['artifact_bytes'] == row['charged_artifact_bytes']
        and digest(job['worker']) == digest(original['job']['worker'])
        and digest(job['frozen_samples']) == digest(original['samples'])
        and job['sampling_receipt_path'] == original['receipt_path']
        and job['sampling_receipt_sha256'] == original['receipt_sha256'], 'physical common lineage differs')
    out = ROOT / 'attempts' / key / 'attempt-0001'; path = out / 'receipt.json'
    guard.require(refs[str(path)] == row['validation_receipt_sha256'], 'validation receipt identity differs')
    receipt = native.bound(path, row['validation_receipt_sha256'])
    guard.require(digest(receipt['job']) == key and receipt['round_sha256'] == round_pin
        and receipt['status'] == row['status'], 'closed validation receipt differs')
    if row['status'] != 'ok':
        guard.require(sample is None, 'failed validation acquired metric evidence'); return None
    files = receipt['evidence_files']; batch_path = out / 'batch.json'
    guard.require(refs[str(batch_path)] == files['batch.json'], 'metric batch anchor differs')
    batch = native.bound(batch_path, files['batch.json'])
    guard.require(batch['job_sha256'] == key and batch['metric_source_sha256'] == lock['metric_source_sha256']
        and len(batch['samples']) == 6 and type(job['metric_replay_required']) is bool,
        'frozen metric batch differs')
    records = {(s['size_multiplier'], s['sample_seed']): s for s in batch['samples']}
    guard.require(len(records) == 6 and set(records) == {(n, s) for n in dope.SIZES for s in dope.SEEDS}
        and all(type(s['size_multiplier']) is int and type(s['sample_seed']) is int for s in batch['samples']),
        'metric batch sample identities differ')
    identity = row['size_multiplier'], row['sample_seed']; record = records[identity]
    frozen = {(s['size_multiplier'], s['sample_seed']): s for s in job['frozen_samples']}[identity]
    guard.require(digest({k: record[k] for k in frozen}) == digest(frozen), 'frozen sample identity differs')
    name = f'n{identity[0]}-seed{identity[1]}.metric.json'; metric_path = out / name
    guard.require(record['metric_file'] == name and refs[str(metric_path)] == files[name] == record['metric_sha256']
        and digest({k: v for k, v in sample.items() if k != 'metrics'})
        == digest(dict(record, metric_path=str(metric_path))), 'logical metric evidence differs')
    repeated = job['metric_replay_required'] and identity == (4, 101)
    guard.require(record['metric_replay'] == ('exact' if repeated else 'not_repeated'), 'metric replay identity differs')
    metric = native.bound(metric_path, record['metric_sha256'])
    guard.require(digest(sample['metrics']) == digest(metric), 'embedded metric evidence differs')
    if repeated:
        replay_path = out / f'n{identity[0]}-seed{identity[1]}.metric-replay.json'
        guard.require(refs[str(replay_path)] == files[replay_path.name], 'metric replay anchor differs')
        replay = native.bound(replay_path, files[replay_path.name])
        guard.require(digest({k: v for k, v in replay.items() if k != 'metric_seconds'})
            == digest({k: v for k, v in metric.items() if k != 'metric_seconds'}), 'metric replay differs')
    return metric


def build(receipt_pin, report_pin, auxiliary_pin):
    guard.require((ROOT / 'receipt-lock-v1.json').is_file(), 'complete common matrix not yet closed')
    anchor = native.bound(ROOT / 'receipt-lock-v1.json', receipt_pin)
    guard.require(anchor['complete_matrix'] is True and anchor['reconciliation_sha256'] == report_pin,
                  'closed common receipt digest differs')
    # Verify every transitive receipt, input and executable before decoding metrics.
    refs = {}
    for path, pin in anchor['refs'].items(): dope.fits.checked(path, pin, refs)
    parent_references(anchor, refs)
    lock = native.bound(ROOT / 'round.lock.json', anchor['round_sha256'])
    for path, pin in refs.items():
        if Path(path).name == 'round.lock.json':
            parent = native.bound(path, pin)
            if 'source_files' in parent:
                source = Path(path).parent / 'source'
                guard.require(all(Path(p).parent == source for p in parent['source_files']), 'source root differs')
                native.flat(source, {Path(p).name: h for p, h in parent['source_files'].items()})
    guard.require(lock['gpu_operations_enabled'] is False and lock['official_tests_opened'] is False
        and lock['full_campaign_admitted'] is False and lock['native_selection_changed'] is False,
        'common round acquired admission or selection')
    runtime = shared_inventory(lock, {})
    sampling_anchor = native.bound(SAMPLING / 'receipt-lock-v1.json', lock['sampling_receipt_lock_sha256'])
    parent_references(sampling_anchor, refs)
    sample_lock = native.bound(SAMPLING / 'round.lock.json', SAMPLE_ROUND)
    sample_report = native.bound(SAMPLING / 'reconciliation-v1.json', sampling_anchor['reconciliation_sha256'])
    guard.require(sampling_anchor['complete_sampling_matrix'] is True
        and sampling_anchor['round_sha256'] == SAMPLE_ROUND and sample_report['complete_sampling_matrix'] is True
        and sample_report['logical_sample_cells'] == 1200 and len(sample_lock['jobs']) == 199,
        'complete original ARF samples required')
    closed(SAMPLING, sample_lock, sampling_anchor, SAMPLE_ROUND, 199, 1200)
    original = native.build(NATIVE_RECEIPTS, NATIVE_REPORT)
    native_path = Path(native.__file__).with_name('results') / (native.NAME + '.json')
    committed = guard.bound(native_path, NATIVE_PUBLICATION, native_path.parent)
    guard.require(digest(original) == digest(committed), 'frozen native winners changed')
    native_cells = {r['dataset']: r for r in original['cells']}
    sampling_jobs, sampling_batches = sampling_receipts(sample_lock, sample_report, sampling_anchor['refs'])
    worker_views = {j['dataset']: j['worker'] for j in sample_lock['jobs']}
    fit_rows = dope.fits.bound(dope.fits.ROOT / 'reconciliation-v1.json', dope.fits.REPORT)['fit_cells']
    guard.require(all(digest(r['original_fit_job']['worker']) == digest(worker_views[r['dataset']])
                      for r in fit_rows), 'DOPE and ARF worker views differ')
    for job in sampling_jobs.values():
        native.flat(Path(job['worker']['path']), job['worker']['files'])
        native.flat(Path(job['artifact_path']), job['artifact_inventory'])
    closed(ROOT, lock, anchor, anchor['round_sha256'], len(lock['jobs']), 1200)
    auxiliary = native.bound(ROOT / 'closed-batch-output-inventory-v1.json', auxiliary_pin)
    guard.require(auxiliary['round_sha256'] == anchor['round_sha256']
        and auxiliary['pinned_before_reconciliation_metric_reads'] is True
        and auxiliary['extra_outputs_used_for_metrics_or_selection'] is False,
        'closed recursive output anchor differs')
    guard.require(set(auxiliary['batches']) == {digest(j) for j in lock['jobs']}, 'metric batch inventory coverage differs')
    for key, inv in auxiliary['batches'].items():
        out = ROOT / 'attempts' / key / 'attempt-0001'
        guard.inventory(out, {str(out / n): dict(sha256=h, bytes=Path(out / n).stat().st_size)
                        for n, h in inv['files'].items()}, BASE, [str(out / n) for n in inv['directories']])
        guard.require(inv['files']['receipt.json'] == refs[str(out / 'receipt.json')], 'metric receipt anchor differs')
    report = native.bound(ROOT / 'reconciliation-v1.json', report_pin)
    guard.require(report['complete_matrix'] is True and report['logical_sample_cells'] == 1200
        and report['native_selection_changed'] is False and report['official_tests_opened'] is False
        and all(report[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')),
        'complete common outcomes required')
    jobs = {digest(j): j for j in lock['jobs']}
    specifications = {(c['dataset'], c['configuration'], c['size_multiplier'], c['sample_seed']): c
                      for c in lock['logical_cells']}
    guard.require(len(jobs) == len(lock['jobs']) and len(specifications) == len(lock['logical_cells']) == 1200,
                  'frozen physical or logical job identities differ')
    cells = []
    for row in report['cells']:
        specification = specifications[row['dataset'], row['configuration'], row['size_multiplier'], row['sample_seed']]
        selection = native_cells[row['dataset']]
        selected_key = selection['author_default_job_sha256' if row['configuration'] == 'author_default' else 'selected_job_sha256']
        guard.require(row['fit_job_sha256'] == selected_key, 'common metrics changed native selection')
        sample_job = sampling_jobs[row['sampling_job_sha256']]
        guard.require(sample_job['fit_job_sha256'] == selected_key
            and sample_job['artifact_bytes'] == row['charged_artifact_bytes'], 'common artifact identity differs')
        metric = common_sample(row, specification, jobs, sampling_batches, refs, lock, anchor['round_sha256'])
        sample = row['sample_evidence']
        if metric is not None:
            dope.metric_check(metric, sample_job['worker'], row['size_multiplier'], lock['metric_source_sha256'])
            guard.require(metric['dependencies'] == runtime['expected_versions']
                and all(type(metric[k]) in (int, float) and math.isfinite(metric[k]) and 0 <= metric[k] <= 1
                        for k in ('marginal_ks_mean', 'pair_correlation_fidelity', 'c2st_auc')),
                'metric dependencies or fidelity differ')
        cells.append(dict(dataset=row['dataset'], method='ARF', configuration=row['configuration'], fit_seed=row['fit_seed'],
            sample_seed=row['sample_seed'], size_multiplier=row['size_multiplier'], status=row['status'],
            unavailable_reason=row['unavailable_reason'] if row['status'] == 'ok' else row['unavailable_reason'] or row['status'],
            charged_artifact_bytes=row['charged_artifact_bytes'], projection_bytes_included=True,
            l3_artifact_cap_satisfied=row['charged_artifact_bytes'] <= 10240,
            fit_job_sha256=selected_key, native_kpi=selection['default_native_kpi' if row['configuration'] == 'author_default' else 'selected_native_kpi'],
            validation_receipt_sha256=row['validation_receipt_sha256'], sample_evidence=sample,
            utility=metric['utility'] if metric else None, null_loss=metric['null_loss'] if metric else None,
            copy_counts=metric['copy_counts'] if metric else None,
            real_vs_real_control_counts=metric['real_vs_real_control_counts'] if metric else None,
            marginal_ks_mean=metric['marginal_ks_mean'] if metric else None,
            pair_correlation_fidelity=metric['pair_correlation_fidelity'] if metric else None,
            c2st_auc=metric['c2st_auc'] if metric else None,
            mfs_v2=None, ptf_v1=None, release_safe=None, superiority=None, counts_as_dope_win=False))
    ids = complete_matrix(cells); references = reference_cells(ids)
    groups, panels = summaries(cells + references)
    for path, pin in refs.items(): dope.fits.checked(path, pin, {})
    return dict(format='dope-complete-original-arf-matched-population-validation', version=1,
        source_sha256=dope.fits.sha256(Path(__file__)), s3_data_lock_sha256=dope.fits.DATA,
        scope='100 bounded official-training-derived S3 views; one fit seed, three sample seeds, n/4n. ARF default and held-out FORDE native winner remain frozen.',
        datasets=ids, cells=cells, logical_validation_cells=1200,
        physical_sampling_batches=199, physical_metric_batches=len(lock['jobs']),
        logical_status_counts=report['logical_status_counts'], physical_metric_status_counts=report['physical_status_counts'],
        sample_status_counts=sample_report['physical_status_counts'], lineage_groups=groups, summary=panels,
        paired_descriptive=paired(groups), native_objective=original['native_objective'],
        native_selection_reference=dict(path=str(native_path), sha256=NATIVE_PUBLICATION),
        dope_reference=dict(sha256=DOPE_PUBLICATION, logical_cells=1800, earlier_density_sha256=dope.REFERENCE),
        cost=dict(native_fit=original['cost'], sample_operation_seconds=sample_report['new_operation_seconds'],
            sample_scheduler_wall_seconds=sample_report['coordinator_wall_seconds'],
            metric_operation_seconds=report['new_metric_operation_seconds'],
            metric_scheduler_wall_seconds=report['coordinator_wall_seconds']),
        source_locks=dict(round=anchor['round_sha256'], receipts=receipt_pin, reconciliation=report_pin,
            batch_auxiliary=auxiliary_pin, sampling_round=SAMPLE_ROUND,
            sampling_receipts=lock['sampling_receipt_lock_sha256'], native_receipts=NATIVE_RECEIPTS,
            native_reconciliation=NATIVE_REPORT), release_safe_l3_comparison_complete=False, **GATES)


def validate_report(report):
    ids = complete_matrix(report['cells'])
    guard.require(report['datasets'] == ids and report['logical_validation_cells'] == 1200
        and report['physical_sampling_batches'] == 199 and 0 <= report['physical_metric_batches'] <= 199
        and report['source_sha256'] == dope.fits.sha256(Path(__file__))
        and report['s3_data_lock_sha256'] == dope.fits.DATA
        and report['native_selection_reference']['sha256'] == NATIVE_PUBLICATION
        and report['dope_reference']['sha256'] == DOPE_PUBLICATION
        and report['release_safe_l3_comparison_complete'] is False
        and all(report[k] is v for k, v in GATES.items()), 'committed common scope differs')
    guard.require(dict(Counter(c['status'] for c in report['cells'])) == report['logical_status_counts'],
                  'logical outcome counts differ')
    references = reference_cells(ids); groups, panels = summaries(report['cells'] + references)
    guard.require(digest(groups) == digest(report['lineage_groups']) and digest(panels) == digest(report['summary'])
        and digest(paired(groups)) == digest(report['paired_descriptive']), 'descriptive aggregates differ')
    guard.require(report['source_locks']['sampling_round'] == SAMPLE_ROUND
        and report['source_locks']['native_receipts'] == NATIVE_RECEIPTS
        and report['source_locks']['native_reconciliation'] == NATIVE_REPORT
        and report['dope_reference']['earlier_density_sha256'] == dope.REFERENCE, 'frozen reference bindings differ')
    for pin in report['source_locks'].values(): guard.digest(pin)


def table(report):
    out = io.StringIO(); writer = csv.writer(out, lineterminator='\n')
    writer.writerow(['dataset', 'method', 'configuration', 'size_multiplier', 'status_counts', 'charged_artifact_bytes',
                     'catboost_retention', 'linear_retention', 'mlp_retention'])
    for row in report['lineage_groups']:
        writer.writerow([row['dataset'], row['method'], row['configuration'], row['size_multiplier'],
            json.dumps(row['statuses'], sort_keys=True), row['charged_artifact_bytes'],
            *[row['utility'][a]['median_retention'] for a in ('catboost', 'linear', 'mlp')]])
    return out.getvalue()


def markdown(report):
    lines = ['# Complete original ARF / DOPE matched S3 validation', '',
        '100 planned bounded views from official S3 training partitions; official tests remain sealed.',
        'ARF uses immutable author defaults and separate held-out FORDE native selections [@watson2023adversarial].',
        'Common scores never retune ARF. Native values are reported per lineage and never ranked across methods.', '',
        '| DOPE research profile | ARF configuration | Paired informative lineages | DOPE retention | ARF retention | Median paired difference |',
        '|---|---|---:|---:|---:|---:|']
    for row in report['paired_descriptive']:
        if row['size_multiplier'] == 4 and row['auditor'] == 'catboost':
            values = [row[k] for k in ('configuration', 'reference_configuration', 'paired_complete_informative_lineages',
                'dope_median_retention', 'reference_median_retention', 'median_paired_difference')]
            lines.append('| ' + ' | '.join(str(x) if x is not None else 'null' for x in values) + ' |')
    return '\n'.join(lines + ['', 'Sample seeds reduce within lineage before dataset summaries; each pair may have a different available cohort.',
        'Charged ARF model and projection bytes remain visible even above10240; this is unconstrained quality, not release-safe L3 evidence.',
        'One fit seed; final five-fit/privacy/product coverage incomplete. No family selection or superiority inference.',
        'MFS-v2/PTF-v1/release-safe/superiority are null; every unavailable cell remains visible.', ''])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--receipt-lock-sha256'); parser.add_argument('--reconciliation-sha256')
    parser.add_argument('--batch-auxiliary-sha256'); parser.add_argument('--from-json', type=Path)
    parser.add_argument('--publication-sha256'); parser.add_argument('--output-directory', type=Path, default=Path(__file__).with_name('results'))
    args = parser.parse_args()
    if args.from_json:
        guard.digest(args.publication_sha256); body = args.from_json.read_bytes()
        guard.require(hashlib.sha256(body).hexdigest() == args.publication_sha256, 'committed publication changed')
        report = guard.decode(body); validate_report(report)
    else:
        report = build(args.receipt_lock_sha256, args.reconciliation_sha256, args.batch_auxiliary_sha256)
        validate_report(report)
    from .publish_s3_matched import schema
    args.output_directory.mkdir(parents=True, exist_ok=True)
    outputs = dict(json=json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n', csv=table(report), md=markdown(report))
    outputs['schema.json'] = json.dumps(schema(report), sort_keys=True, indent=2)+'\n'
    for suffix, body in outputs.items(): (args.output_directory / (NAME+'.'+suffix)).write_text(body)


if __name__ == '__main__': main()
