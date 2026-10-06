"""Publish complete native/default ARF common outcomes after actual closure."""
import argparse
import ast
from collections import Counter
import csv
import io
import hashlib
import json
import math
import os
import stat
from pathlib import Path
import statistics

from . import publish_arf_population_native as native
from . import publish_dope_refinement_population as dope
from . import arf_runtime_guard as guard
from .manifest import digest
from .publish_dope_refinement_discovery import summaries
from .publish_s3_forest import shared_inventory

BASE = native.BASE
HISTORICAL_ROUND = BASE / 'neural-native-v1/round.lock.json'
HISTORICAL_ROUND_SHA = '5634d235eb8305436ca046867d96d09f562fabbd360892c580a7da3b9b811988'
HISTORICAL_SOURCE = HISTORICAL_ROUND.parent / 'package/research/benchmark'
HISTORICAL_PROJECTION = BASE / 'target-arf-neural-source-projection-v1'
HISTORICAL_FILES = {
    'contract.json': 'ccccf3deaa8f219ea6ce14d44296c5996579a55af98a5afddf0099d513f0f894',
    'fetch_jope.py': '3264d390fb2e39450551165a3c30833cb093056a8478a0501c7b4082125d8fbf',
    'freeze_sdv_round.py': 'f00942e3f6252a2295ae8e7cf172282f1f527e77dfc120780a3ae448f5ee89b4',
    'gpu_probe.py': '3a9a7762ce3ac7d68873fa8c3ec5c6ffc4d036b19a5507cd73f3573aa9e8d667',
    'inventory_hosts.py': '73565ea6e5c510b67ac1133fd0bea74834a36132675c2c7a3f55146d5018d910',
    'manifest.py': '77be48e26ce880e391c2c771161bbd5ffeb713963bcf1af53592c003dc1ce7c9',
    'score.py': '3c81ce4ab6cdb468f0b2e54bfd668389a0b1fcdbcce68a49ab40bf12036f9968',
    'sdv_adapter.py': 'c6f4240ae5b9795672693e2d6d3908053e105885b7d34d659027772183fa541b',
    'sdv_round.py': '290fd6bb86f9d8fa8f1ba3c45c1710fad08d33103707986eeac64a9f84060b79',
}
ROOT = BASE / 'arf-s3-population-shared-validation-v1'
SAMPLING = BASE / 'arf-s3-population-sampling-v1'
SAMPLE_ROUND = 'bf17d9c4b2d76736036dec82adfd98447e94f66fd8d7bb56675d4e8245012777'
NATIVE_RECEIPTS = '238a19ea24694348b0cb6acd43b5203ac75989ee6ac5c714ef86c32ff040e61d'
NATIVE_REPORT = 'af722bff624b2b5cad37eb10a314767dcbd2b00b085c44114f8ef398de9c3850'
NATIVE_PUBLICATION = '3bac30656933cac2c243b7bf6af654025a0930ed94828dd371c6fb2ba335692e'
DOPE_PUBLICATION = 'c79804a8745197094ce1233544964872f2bd9857a93b6808d5c551b7662ce602'
NAME = 'arf-matched-population-validation'
HISTORICAL_PUBLISHER_SHA = '7ba486e01ded70cae07192fb7545340cfb3f6feb07089a72e168a3de1054fa05'
HISTORICAL_PUBLICATION_SHA = 'c69ce66e79b234bf5d1f9d56938450ba253ae39a3f6655a6fd586f890b8992f7'
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
        and count(done['jobs']) == physical_jobs and count(done['logical_cells']) == logical_cells
        and all(not Path('/proc', str(launch[k])).exists() for k in ('pid', 'supervisor_pid')),
        'actual complete common closure required')
    native.flat(root / 'source', {Path(p).name: h for p, h in lock['source_files'].items()})
    return actual


def seconds(value):
    guard.require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                  'finite nonnegative receipt cost required')
    return value


def count(value):
    guard.require(type(value) is int and value >= 0, 'canonical nonnegative count required')
    return value


def status_counts(value):
    guard.require(type(value) is dict and all(type(k) is str and k and type(v) is int and v >= 0
                  for k, v in value.items()), 'canonical status counts required')
    return value


def receipt_scope(receipt):
    guard.require(receipt['official_tests_opened'] is False
        and type(receipt['new_generator_fits_started']) is int and receipt['new_generator_fits_started'] == 0
        and receipt['native_selection_changed'] is False
        and all(receipt[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')),
        'receipt scope or claim differs')
    guard.require(receipt.get('global_family_selected', False) is False, 'receipt family selection differs')


def operation_accounting(receipt, key, pin):
    receipt_scope(receipt)
    operation = receipt['operation']
    amount = 0; started = False
    if operation is not None:
        amount = seconds(operation['elapsed_seconds']); started = operation['new_operation_started']
        guard.require(type(started) is bool and (started or amount == 0)
            and operation.get('official_tests_opened', False) is False
            and operation.get('foreign_processes_signaled', False) is False,
            'operation scope or unstarted cost differs')
    if receipt['status'] == 'ok':
        guard.require(operation is not None and started and type(operation['exit_code']) is int
            and operation['exit_code'] == 0 and operation['host'] == 'xbabe2' and amount <= 600,
            'successful receipt lacks bounded actual operation')
    return dict(job_sha256=key, receipt_sha256=pin, status=receipt['status'],
                operation_seconds=amount, new_operation_started=started)


def closure_accounting(root, refs, report, records, cost_key):
    actual = native.bound(root / 'coordinator-exit.json', refs[str(root / 'coordinator-exit.json')])
    done = native.bound(root / 'completion.json', refs[str(root / 'completion.json')])
    counts = dict(Counter(r['status'] for r in records))
    amount = sum(r['operation_seconds'] for r in records)
    wall = seconds(actual['elapsed_seconds'])
    guard.require(counts == status_counts(done['counts']) == status_counts(report['physical_status_counts'])
        and seconds(report[cost_key]) == amount and seconds(report['coordinator_wall_seconds']) == wall,
        'closed physical status or cost accounting differs')
    return amount, wall


def publication_anchor(receipt_pin, name, parent_receipt_pin):
    """Select a root-pinned immutable extension without changing historical evidence."""
    guard.require(name in ('receipt-lock-v1.json', 'receipt-lock-v2.json'), 'receipt anchor name differs')
    anchor = native.bound(ROOT / name, receipt_pin)
    if name == 'receipt-lock-v2.json':
        guard.digest(parent_receipt_pin)
        parent_path = ROOT / 'receipt-lock-v1.json'
        parent = native.bound(parent_path, parent_receipt_pin)
        guard.require(type(anchor['version']) is int and anchor['version'] == 2
            and anchor['parent_receipt_lock_path'] == str(parent_path)
            and anchor['parent_receipt_lock_sha256'] == parent_receipt_pin
            and anchor['refs'].get(str(parent_path)) == parent_receipt_pin
            and all(anchor['refs'].get(p) == h for p, h in parent['refs'].items())
            and digest({k: v for k, v in anchor.items() if k in parent and k not in ('version', 'refs')})
            == digest({k: v for k, v in parent.items() if k not in ('version', 'refs')}),
            'immutable original receipt anchor changed')
    return anchor


def parent_references(anchor, refs):
    """Require every pinned parent leaf in the already authenticated flat map."""
    pending = [anchor]; visited = set()
    while pending:
        parent = pending.pop()
        for path, pin in parent['refs'].items():
            guard.digest(pin)
            guard.require(path in refs and refs[path] == pin, 'transitive parent evidence absent or inconsistent')
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
    for pin in inventory.values(): guard.digest(pin)
    guard.require(all(type(name) is str and Path(name).name == name and str(out / name) in refs
                      and refs[str(out / name)] == pin
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
        guard.digest(sample['sample_sha256'])
        guard.require(sample['sample_file'] == path.name and sample['sample_path'] == str(path)
            and str(path) in refs and path.name in files
            and refs[str(path)] == files[path.name] == sample['sample_sha256']
            and type(sample['rows']) is int and sample['rows'] == job['worker']['train_rows'] * size
            and type(sample['columns']) is int and sample['columns'] == job['worker']['projected_features'] + 1
            and sample['sample_replay'] == ('exact' if (size, seed) == (1, 101) else 'not_repeated'),
            'original generated sample identity differs')
        if sample['sample_replay'] == 'exact':
            replay = out / 'n1-seed101.repeat.csv'
            guard.require(str(replay) in refs and replay.name in files
                          and refs[str(replay)] == files[replay.name] == sample['sample_sha256'],
                          'original sampling replay differs')


def sampling_receipts(sample_lock, sample_report, refs):
    """Bind upstream reconciliation declarations before any metric decoding."""
    jobs = {digest(j): j for j in sample_lock['jobs']}
    rows = sample_report['physical_batches_evidence']; batches = {r['job_sha256']: r for r in rows}
    guard.require(len(jobs) == len(sample_lock['jobs']) == len(rows) == len(batches) == 199
        and set(batches) == set(jobs), 'original sampling batch coverage differs')
    accounting = []
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
        accounting.append(operation_accounting(receipt, key, row['receipt_sha256']))
    return jobs, batches, accounting


def common_receipts(jobs, refs, auxiliary, round_pin):
    """Authenticate the entire physical receipt graph before opening a metric."""
    receipts = {}; accounting = []
    for key, job in jobs.items():
        out = ROOT / 'attempts' / key / 'attempt-0001'; path = out / 'receipt.json'
        pin = refs[str(path)]; receipt = native.bound(path, pin)
        guard.require(digest(receipt['job']) == key and receipt['round_sha256'] == round_pin,
                      'closed validation receipt lineage differs')
        files = receipt['evidence_files']; inventory = auxiliary['batches'][key]['files']
        for value in files.values(): guard.digest(value)
        guard.require(all(type(name) is str and Path(name).name == name and name not in ('', '.', '..')
            and name in inventory and str(out / name) in refs
            and inventory[name] == value == refs[str(out / name)]
            for name, value in files.items()), 'common receipt evidence anchor differs')
        accounting.append(dict(operation_accounting(receipt, key, pin),
            sampling_job_sha256=job['sampling_job_sha256'])); receipts[key] = receipt
    return receipts, accounting


def common_sample(row, specification, jobs, sampling_batches, refs, lock, round_pin, receipts):
    """Bind a logical cell to its physical receipt and exact frozen sample."""
    guard.require(type(row['fit_seed']) is int and row['fit_seed'] == 11
        and type(row['sample_seed']) is int and type(row['size_multiplier']) is int,
        'input common seed identity differs')
    guard.require(digest({k: row[k] for k in specification}) == digest(specification),
                  'logical common cell differs from frozen schedule')
    key = specification['physical_job_sha256']; sample = row['sample_evidence']
    guard.require(row['sampling_job_sha256'] in sampling_batches,
                  'original sampling lineage unavailable')
    original = sampling_batches[row['sampling_job_sha256']]
    if key is None:
        guard.require(row['status'] == 'sampling_unavailable' and sample is None
            and row['validation_receipt_sha256'] is None
            and original['status'] != 'ok' and original['samples'] == []
            and row['dataset'] == original['job']['dataset']
            and type(original['job']['fit_seed']) is int
            and original['job']['fit_seed'] == row['fit_seed']
            and row['charged_artifact_bytes'] == original['artifact_bytes'],
            'unavailable sample lacks failed original sampling evidence')
        return None
    job = jobs[key]
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
    receipt = receipts[key]
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
    replay_path = replay_pin = None
    if repeated:
        replay_path = out / f'n{identity[0]}-seed{identity[1]}.metric-replay.json'
        guard.require(refs[str(replay_path)] == files[replay_path.name], 'metric replay anchor differs')
        replay_pin = files[replay_path.name]
    return dict(path=metric_path, pin=record['metric_sha256'], embedded=sample['metrics'],
                replay_path=replay_path, replay_pin=replay_pin)


def decode_metric(binding):
    if binding is None: return None
    metric = native.bound(binding['path'], binding['pin'])
    guard.require(digest(binding['embedded']) == digest(metric), 'embedded metric evidence differs')
    if binding['replay_path'] is not None:
        replay = native.bound(binding['replay_path'], binding['replay_pin'])
        guard.require(digest({k: v for k, v in replay.items() if k != 'metric_seconds'})
            == digest({k: v for k, v in metric.items() if k != 'metric_seconds'}), 'metric replay differs')
    return metric


def historical_source_body(path, pin):
    """Read one authenticated regular source body; never import or execute it."""
    path = native.safe(path)
    guard.digest(pin)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        guard.require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
            and 0 < before.st_size <= 1 << 20, 'historical source body type or size differs')
        raw = stream.read((1 << 20) + 1)
        after = os.fstat(stream.fileno())
    current = path.lstat()
    guard.require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        == (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
        and after.st_nlink == current.st_nlink == 1
        and hashlib.sha256(raw).hexdigest() == pin, 'historical source body changed')
    return raw


def verify_historical_projection(binding):
    guard.require(type(binding) is dict and binding['round'] == str(HISTORICAL_ROUND)
        and binding['round_sha256'] == HISTORICAL_ROUND_SHA
        and binding['origin_root'] == str(HISTORICAL_SOURCE)
        and binding['projection_root'] == str(HISTORICAL_PROJECTION)
        and set(binding['files']) == set(HISTORICAL_FILES), 'historical copy binding differs')
    historical_source_body(HISTORICAL_ROUND, HISTORICAL_ROUND_SHA)
    for name, pin in HISTORICAL_FILES.items():
        raw = historical_source_body(HISTORICAL_SOURCE / name, pin)
        guard.require(historical_source_body(HISTORICAL_PROJECTION / name, pin) == raw,
                      'historical source projection drifted')
        row = binding['files'][name]
        guard.require(row == dict(origin=str(HISTORICAL_SOURCE / name), copy=str(HISTORICAL_PROJECTION / name),
            bytes=len(raw), sha256=pin), 'historical source copy binding changed')
    native.flat(HISTORICAL_PROJECTION, {name: dict(bytes=row['bytes'], sha256=row['sha256'])
                                 for name, row in binding['files'].items()})


def project_historical_sources(round_path, round_pin, files, refs):
    guard.require(Path(round_path) == HISTORICAL_ROUND and round_pin == HISTORICAL_ROUND_SHA
        and type(files) is dict and files == HISTORICAL_FILES
        and refs.get(str(HISTORICAL_ROUND)) == HISTORICAL_ROUND_SHA
        and all(refs.get(str(HISTORICAL_SOURCE / name)) == pin for name, pin in HISTORICAL_FILES.items()),
        'exact authenticated historical source origin required')
    historical_source_body(HISTORICAL_ROUND, HISTORICAL_ROUND_SHA)
    bodies = {name: historical_source_body(HISTORICAL_SOURCE / name, pin) for name, pin in HISTORICAL_FILES.items()}
    native.safe(HISTORICAL_PROJECTION.parent)
    if not HISTORICAL_PROJECTION.exists(): HISTORICAL_PROJECTION.mkdir(mode=0o700)
    native.safe(HISTORICAL_PROJECTION)
    guard.require(HISTORICAL_PROJECTION.is_dir() and not set(p.name for p in HISTORICAL_PROJECTION.iterdir())
        - set(HISTORICAL_FILES), 'historical projection acquired an extra member')
    binding = dict(round=str(HISTORICAL_ROUND), round_sha256=HISTORICAL_ROUND_SHA,
        origin_root=str(HISTORICAL_SOURCE), projection_root=str(HISTORICAL_PROJECTION),
        source_files_executed=False, source_cache_bodies_read=False, files={})
    for name, raw in bodies.items():
        destination = HISTORICAL_PROJECTION / name
        if not destination.exists():
            fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'wb') as stream: stream.write(raw)
        guard.require(historical_source_body(destination, HISTORICAL_FILES[name]) == raw,
                      'historical source projection drifted')
        binding['files'][name] = dict(origin=str(HISTORICAL_SOURCE / name), copy=str(destination),
                                     bytes=len(raw), sha256=HISTORICAL_FILES[name])
    verify_historical_projection(binding)
    return binding


def flat_round_sources(round_path, files, round_pin, refs):
    """Authenticate exact current roots or the explicitly bound historical origin."""
    source = Path(round_path).parent / 'source'
    guard.require(type(files) is dict and bool(files)
        and all(type(key) is str for key in files), 'source key format differs')
    keys = list(files)
    paths = [Path(key) for key in keys]
    flat = all(key not in ('', '.', '..') and key == path.name
        and '/' not in key and '\\' not in key for key, path in zip(keys, paths))
    absolute = all(path.is_absolute() and str(path) == key and path.parent == source
        and path.name not in ('', '.', '..') and '\\' not in key
        for key, path in zip(keys, paths))
    guard.require(flat or absolute, 'source root differs')
    normalized = {path.name: files[key] for key, path in zip(keys, paths)}
    guard.require(len(normalized) == len(files), 'duplicate source basename')
    if flat: return project_historical_sources(round_path, round_pin, normalized, refs)
    native.flat(source, normalized)
    return None


def build(receipt_pin, report_pin, auxiliary_pin, receipt_lock_name="receipt-lock-v1.json", parent_receipt_pin=None, historical_binding_output=None):
    guard.require((ROOT / 'receipt-lock-v1.json').is_file(), 'complete common matrix not yet closed')
    anchor = publication_anchor(receipt_pin, receipt_lock_name, parent_receipt_pin)
    guard.require(anchor['complete_matrix'] is True and anchor['reconciliation_sha256'] == report_pin,
                  'closed common receipt digest differs')
    # Verify every transitive receipt, input and executable before decoding metrics.
    refs = {}
    for path, pin in anchor['refs'].items(): dope.fits.checked(path, pin, refs)
    parent_references(anchor, refs)
    lock = native.bound(ROOT / 'round.lock.json', anchor['round_sha256'])
    projections = []
    for path, pin in refs.items():
        if Path(path).name == 'round.lock.json':
            parent = native.bound(path, pin)
            if 'source_files' in parent:
                projection = flat_round_sources(path, parent['source_files'], pin, refs)
                if projection is not None: projections.append(projection)
    guard.require(len(projections) == int(str(HISTORICAL_ROUND) in refs)
        and (not projections or historical_binding_output is not None),
        'exact historical source binding output required')
    if projections:
        projection_body = json.dumps(projections[0], sort_keys=True, indent=2, allow_nan=False)+'\n'
        with Path(historical_binding_output).open('x') as stream: stream.write(projection_body)
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
        and count(sample_report['logical_sample_cells']) == 1200 and len(sample_lock['jobs']) == 199,
        'complete original ARF samples required')
    closed(SAMPLING, sample_lock, sampling_anchor, SAMPLE_ROUND, 199, 1200)
    original = native.build(NATIVE_RECEIPTS, NATIVE_REPORT)
    native_path = Path(native.__file__).with_name('results') / (native.NAME + '.json')
    committed = guard.bound(native_path, NATIVE_PUBLICATION, native_path.parent)
    guard.require(digest(original) == digest(committed), 'frozen native winners changed')
    native_cells = {r['dataset']: r for r in original['cells']}
    sampling_jobs, sampling_batches, sample_accounting = sampling_receipts(sample_lock, sample_report, sampling_anchor['refs'])
    sample_cost, sample_wall = closure_accounting(SAMPLING, sampling_anchor['refs'], sample_report,
                                                sample_accounting, 'new_operation_seconds')
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
    guard.require(report['complete_matrix'] is True and count(report['logical_sample_cells']) == 1200
        and report['native_selection_changed'] is False and report['official_tests_opened'] is False
        and all(report[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')),
        'complete common outcomes required')
    jobs = {digest(j): j for j in lock['jobs']}
    specifications = {(c['dataset'], c['configuration'], c['size_multiplier'], c['sample_seed']): c
                      for c in lock['logical_cells']}
    guard.require(len(jobs) == len(lock['jobs']) and len(specifications) == len(lock['logical_cells']) == 1200,
                  'frozen physical or logical job identities differ')
    receipts, metric_accounting = common_receipts(jobs, refs, auxiliary, anchor['round_sha256'])
    metric_cost, metric_wall = closure_accounting(ROOT, refs, report, metric_accounting, 'new_metric_operation_seconds')
    bindings = []; seen = set()
    for row in report['cells']:
        identity = row['dataset'], row['configuration'], row['size_multiplier'], row['sample_seed']
        guard.require(identity not in seen, 'duplicate logical common identity'); seen.add(identity)
        specification = specifications[identity]
        selection = native_cells[row['dataset']]
        selected_key = selection['author_default_job_sha256' if row['configuration'] == 'author_default' else 'selected_job_sha256']
        guard.require(row['fit_job_sha256'] == selected_key, 'common metrics changed native selection')
        sample_job = sampling_jobs[row['sampling_job_sha256']]
        guard.require(sample_job['fit_job_sha256'] == selected_key
            and sample_job['artifact_bytes'] == row['charged_artifact_bytes'], 'common artifact identity differs')
        binding = common_sample(row, specification, jobs, sampling_batches, refs, lock, anchor['round_sha256'], receipts)
        bindings.append((row, selected_key, sample_job, selection, binding))
    guard.require(seen == set(specifications) and dict(Counter(r['status'] for r in report['cells']))
                  == status_counts(report['logical_status_counts']), 'complete logical status accounting differs')
    for projection in projections: verify_historical_projection(projection)
    cells = []
    for row, selected_key, sample_job, selection, binding in bindings:
        metric = decode_metric(binding)
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
            sampling_job_sha256=row['sampling_job_sha256'],
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
    for projection in projections: verify_historical_projection(projection)
    if projections:
        guard.require(Path(historical_binding_output).read_text() == projection_body, 'historical copy proof changed')
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
        physical_sampling_receipts=sample_accounting, physical_metric_receipts=metric_accounting,
        scheduler_wall_seconds_evidence=dict(sample=sample_wall, metric=metric_wall),
        cost=dict(native_fit=original['cost'], sample_operation_seconds=sample_cost,
            sample_scheduler_wall_seconds=sample_wall, metric_operation_seconds=metric_cost,
            metric_scheduler_wall_seconds=metric_wall),
        source_locks=dict(round=anchor['round_sha256'], receipts=receipt_pin, reconciliation=report_pin,
            batch_auxiliary=auxiliary_pin, sampling_round=SAMPLE_ROUND,
            sampling_receipts=lock['sampling_receipt_lock_sha256'], native_receipts=NATIVE_RECEIPTS,
            native_reconciliation=NATIVE_REPORT), release_safe_l3_comparison_complete=False, **GATES)


def public_accounting(records, expected_count):
    count(expected_count)
    guard.require(len(records) == expected_count and len({r['job_sha256'] for r in records}) == expected_count
        and len({r['receipt_sha256'] for r in records}) == expected_count, 'physical receipt accounting coverage differs')
    for row in records:
        guard.digest(row['job_sha256']); guard.digest(row['receipt_sha256'])
        guard.require(type(row['status']) is str and row['status'] and type(row['new_operation_started']) is bool
            and (row['new_operation_started'] or row['operation_seconds'] == 0), 'public operation identity differs')
        seconds(row['operation_seconds'])
    return dict(Counter(r['status'] for r in records)), sum(r['operation_seconds'] for r in records)


def frozen_native_cost():
    directory = Path(native.__file__).with_name('results')
    publication = guard.bound(directory / (native.NAME + '.json'), NATIVE_PUBLICATION, directory)
    cost = publication['cost']
    guard.require(type(cost) is dict and set(cost) == {'operation_seconds', 'scheduler_wall_seconds'},
                  'frozen native cost fields differ')
    for value in cost.values(): seconds(value)
    return cost


def validate_publisher_identity(report):
    pin = report['source_sha256']
    guard.digest(pin)
    if pin == dope.fits.sha256(Path(__file__)):
        return
    body = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    guard.require(pin == HISTORICAL_PUBLISHER_SHA
        and hashlib.sha256(body).hexdigest() == HISTORICAL_PUBLICATION_SHA,
        'publication producing source differs')


def validate_report(report):
    validate_publisher_identity(report)
    ids = complete_matrix(report['cells'])
    guard.require(report['datasets'] == ids and count(report['logical_validation_cells']) == 1200
        and count(report['physical_sampling_batches']) == 199 and count(report['physical_metric_batches']) <= 199
        and report['s3_data_lock_sha256'] == dope.fits.DATA
        and report['native_selection_reference']['sha256'] == NATIVE_PUBLICATION
        and report['dope_reference']['sha256'] == DOPE_PUBLICATION
        and report['release_safe_l3_comparison_complete'] is False
        and all(report[k] is v for k, v in GATES.items()), 'committed common scope differs')
    native_cost = report['cost']['native_fit']
    guard.require(type(native_cost) is dict
        and set(native_cost) == {'operation_seconds', 'scheduler_wall_seconds'},
        'public native cost fields differ')
    for value in native_cost.values(): seconds(value)
    guard.require(digest(native_cost) == digest(frozen_native_cost()),
                  'public native cost differs from frozen native publication')
    guard.require(dict(Counter(c['status'] for c in report['cells'])) == status_counts(report['logical_status_counts']),
                  'logical outcome counts differ')
    metric_counts, metric_seconds = public_accounting(report['physical_metric_receipts'], report['physical_metric_batches'])
    sample_counts, sample_seconds = public_accounting(report['physical_sampling_receipts'], 199)
    guard.require(metric_counts == status_counts(report['physical_metric_status_counts'])
        and sample_counts == status_counts(report['sample_status_counts'])
        and seconds(report['cost']['metric_operation_seconds']) == metric_seconds
        and seconds(report['cost']['sample_operation_seconds']) == sample_seconds
        and seconds(report['cost']['metric_scheduler_wall_seconds']) == seconds(report['scheduler_wall_seconds_evidence']['metric'])
        and seconds(report['cost']['sample_scheduler_wall_seconds']) == seconds(report['scheduler_wall_seconds_evidence']['sample']),
        'public physical status or cost accounting differs')
    metric_receipts = {r['receipt_sha256']: r for r in report['physical_metric_receipts']}
    sampling_receipts = {r['job_sha256']: r for r in report['physical_sampling_receipts']}
    for receipt in metric_receipts.values():
        key = receipt['sampling_job_sha256']; guard.digest(key)
        guard.require(key in sampling_receipts and sampling_receipts[key]['status'] == 'ok',
                      'metric receipt lacks its successful sampling batch')
    logical_receipts = set()
    for cell in report['cells']:
        key = cell['sampling_job_sha256']; guard.digest(key)
        guard.require(key in sampling_receipts, 'logical sampling batch differs')
        pin = cell['validation_receipt_sha256']
        if pin is None:
            guard.require(cell['status'] == 'sampling_unavailable'
                and sampling_receipts[key]['status'] != 'ok'
                and all(cell[k] is None for k in ('sample_evidence', 'utility', 'null_loss', 'copy_counts',
                    'real_vs_real_control_counts', 'marginal_ks_mean', 'pair_correlation_fidelity', 'c2st_auc')),
                'metric outcome lacks its physical receipt')
            continue
        guard.digest(pin)
        guard.require(pin in metric_receipts and cell['status'] != 'sampling_unavailable'
            and cell['status'] == metric_receipts[pin]['status']
            and key == metric_receipts[pin]['sampling_job_sha256'], 'logical and physical statuses differ')
        logical_receipts.add(pin)
    guard.require(logical_receipts == set(metric_receipts), 'logical receipt coverage differs')
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
    parser.add_argument('--receipt-lock-name', choices=('receipt-lock-v1.json','receipt-lock-v2.json'), default='receipt-lock-v1.json')
    parser.add_argument('--parent-receipt-lock-sha256')
    parser.add_argument('--historical-source-binding-output', type=Path)
    parser.add_argument('--publication-sha256'); parser.add_argument('--output-directory', type=Path, default=Path(__file__).with_name('results'))
    args = parser.parse_args()
    if args.from_json:
        guard.digest(args.publication_sha256); body = args.from_json.read_bytes()
        guard.require(hashlib.sha256(body).hexdigest() == args.publication_sha256, 'committed publication changed')
        report = guard.decode(body); validate_report(report)
    else:
        report = build(args.receipt_lock_sha256, args.reconciliation_sha256, args.batch_auxiliary_sha256, args.receipt_lock_name, args.parent_receipt_lock_sha256, args.historical_source_binding_output)
        validate_report(report)
    from .publish_s3_matched import schema
    args.output_directory.mkdir(parents=True, exist_ok=True)
    outputs = dict(json=json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n', csv=table(report), md=markdown(report))
    outputs['schema.json'] = json.dumps(schema(report), sort_keys=True, indent=2)+'\n'
    for suffix, body in outputs.items(): (args.output_directory / (NAME+'.'+suffix)).write_text(body)


if __name__ == '__main__': main()
