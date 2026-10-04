"""Read closed density evidence from external anchors; never execute a verifier."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import stat

BASE = Path('/mnt/fast-scratch/dope-benchmark')
ROOT = BASE / 'density-s3-population-shared-validation-v1'
ROUND = 'bbbc5039bb1f569652fd04617030a0e6caa6258a2a0f8efe0e467552d72914e6'
NATIVE = '902c68d8ffd4ef7a8ed93539947ddb7496d0c14438f7a4a0c52631d51834f6f5'
NATIVE_REPORT = 'efe8f169dbf8e1b7a68b44406ced32d910a2809f09682539facd24c86ac67187'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_identity(value):
    # Match the frozen coordinator, including numeric JSON representations.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def digest_string(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value),
            'externally frozen digest required')


def safe(path):
    p = Path(path)
    require(p.is_absolute() and p == p.resolve(strict=True) and p.is_relative_to(BASE)
            and not any(q.is_symlink() for q in (p, *p.parents))
            and 'evaluator' not in p.parts and p.name != 'test.csv',
            'unadmitted evidence path')
    return p


def owned(path, expected):
    digest_string(expected)
    p = safe(path)
    require(stat.S_ISREG(p.stat().st_mode), 'regular evidence file required')
    data = p.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, 'frozen evidence changed')
    return data


def pairs(values):
    result = {}
    for key, value in values:
        require(key not in result, 'duplicate evidence JSON key')
        result[key] = value
    return result


def bound_json(path, expected):
    data = owned(path, expected)
    try:
        return json.loads(data, object_pairs_hook=pairs,
                          parse_float=finite_float,
                          parse_constant=lambda _: require(False, 'nonfinite evidence JSON'))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError('invalid evidence JSON') from error


def finite_float(value):
    result = float(value)
    require(math.isfinite(result), 'nonfinite evidence JSON')
    return result


def verify_refs(refs):
    require(type(refs) is dict and bool(refs), 'frozen references required')
    for path, expected in refs.items():
        require(type(path) is str, 'invalid reference identity')
        digest_string(expected)
        p = safe(path)
        require(stat.S_ISREG(p.stat().st_mode), 'regular evidence file required')
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for data in iter(lambda: stream.read(1 << 20), b''):
                h.update(data)
        require(h.hexdigest() == expected, 'frozen evidence changed')


def exact_tree(root, files):
    """Reject added empty directories, aliases and special files as well as files."""
    root = safe(root)
    expected = {safe(p) for p in files}
    require(root.is_dir() and all(p.is_relative_to(root) and p != root for p in expected),
            'invalid evidence tree')
    directories = {parent for p in expected for parent in p.parents
                   if parent != root and parent.is_relative_to(root)}
    actual_files, actual_directories = set(), set()
    for p in root.rglob('*'):
        require(not p.is_symlink(), 'aliased evidence tree')
        mode = p.lstat().st_mode
        require(stat.S_ISREG(mode) or stat.S_ISDIR(mode), 'special evidence tree entry')
        (actual_directories if stat.S_ISDIR(mode) else actual_files).add(p)
    require(actual_files == expected and actual_directories == directories,
            'evidence tree inventory changed')


def claims(value):
    require(value['official_tests_opened'] is False
            and value['native_selection_changed'] is False
            and value['global_family_selected'] is False
            and value.get('counts_as_dope_win', False) is False
            and type(value['new_generator_fits_started']) is int
            and value['new_generator_fits_started'] == 0
            and all(value[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')),
            'closed density evidence acquired a claim')


def request_identity(receipt, request, key, status):
    require(digest_identity(receipt['job']) == key == digest_identity(request['job'])
            and receipt['round_sha256'] == request['round_sha256'] == ROUND
            and receipt['status'] == status, 'density request identity differs')


def closed_metadata(lock, report, completion, end):
    """Structural closure only; caller must replay metrics, cost and native selection."""
    claims(lock)
    claims(report)
    require(report['complete_matrix'] is True and report['full_campaign_admitted'] is False
            and lock['full_campaign_admitted'] is False,
            'complete unadmitted density report required')
    for key, count in (('datasets', 100), ('logical_sample_cells', 3600),
                       ('physical_batches_planned', 502), ('physical_batches_closed', 502)):
        require(type(report[key]) is int and report[key] == count, 'density coverage changed')
    require(type(end['exit_code']) is int and end['exit_code'] == 0
            and type(completion['jobs']) is int and completion['jobs'] == 502
            and type(completion['logical_cells']) is int and completion['logical_cells'] == 3600
            and all(v['round_sha256'] == ROUND for v in (report, completion, end)),
            'density coordinator closure differs')
    jobs = {digest_identity(j): j for j in lock['jobs']}
    require(len(jobs) == len(lock['jobs']) == 502, 'density physical identities differ')
    for j in jobs.values():
        require(j['final'] is False and type(j['fit_seed']) is int and j['fit_seed'] == 11
                and j['track'] == 'common-numeric'
                and digest_identity(j['sample_seeds']) == digest_identity([101, 211, 307])
                and digest_identity(j['size_multipliers']) == digest_identity([1, 4]),
                'density job schedule differs')
    batches = {b['physical_job_sha256']: b for b in report['batches']}
    require(len(batches) == len(report['batches']) == 502 and set(batches) == set(jobs),
            'density batch inventory incomplete')
    require(digest_identity(dict(Counter(b['status'] for b in batches.values())))
            == digest_identity(report['physical_status_counts']) == digest_identity(completion['counts']),
            'density closure counts differ')
    cells = report['cells']
    require(len(cells) == len(lock['logical_cells']) == 3600, 'density logical inventory incomplete')
    identities = []
    for original, row in zip(lock['logical_cells'], cells):
        require(digest_identity(original) == digest_identity({k: row[k] for k in original}),
                'frozen native cell changed')
        require(type(row['sample_seed']) is int and type(row['size_multiplier']) is int
                and type(row['fit_seed']) is int and row['fit_seed'] == 11
                and row['counts_as_dope_win'] is False, 'density cell identity differs')
        identity = (row['dataset'], row['method'], row['configuration'],
                    row['sample_seed'], row['size_multiplier'])
        require(all(type(v) is str and bool(v) for v in identity[:3]),
                'density lineage identity differs')
        identities.append(identity)
        batch = batches[row['physical_job_sha256']]
        job = jobs[row['physical_job_sha256']]
        require(row['status'] == batch['status']
                and row['validation_receipt_sha256'] == batch['receipt_sha256']
                and row['dataset'] == job['dataset'] and row['method'] == job['method']
                and digest_identity(row['artifact_bytes']) == digest_identity(job['artifact_bytes']),
                'density cell receipt differs')
    datasets = {i[0] for i in identities}
    expected = {(d, m, c, s, z) for d in datasets
                for m in ('GaussianCopula', 'independent_marginals', 'Chow-Liu')
                for c in ('default', 'native_selected') for s in (101, 211, 307) for z in (1, 4)}
    require(len(datasets) == 100 and len(set(identities)) == 3600 and set(identities) == expected,
            'complete density logical identities required')
    return jobs, batches


def load_closed_inputs(receipt_sha256, report_sha256):
    """Require both external anchors before any report metric is parsed.

    This checks persisted bytes and explicit inventories, without executing frozen
    Python, opening official tests or certifying a loaded runtime/atomic lease.
    Metric/native-selection/cost replay remains a separate publication prerequisite.
    """
    for value in (receipt_sha256, report_sha256):
        digest_string(value)
    anchor = bound_json(ROOT / 'receipt-lock-v1.json', receipt_sha256)
    require(anchor['round_sha256'] == ROUND and anchor['reconciliation_sha256'] == report_sha256
            and anchor['native_receipt_lock_sha256'] == NATIVE
            and anchor['official_tests_opened'] is False
            and anchor['mfs_v2'] is None and anchor['ptf_v1'] is None,
            'density external anchors differ')
    refs = anchor['refs']
    verify_refs(refs)  # Includes all metric evidence; no metric JSON is parsed yet.
    def read(path, expected=None):
        path = str(path)
        require(path in refs and (expected is None or refs[path] == expected),
                'unbound density publication input')
        return bound_json(path, refs[path])
    lock = read(ROOT / 'round.lock.json', ROUND)
    for group in ('frozen_references', 'source_files', 'runtime_lock_files'):
        require(all(refs.get(p) == h for p, h in lock[group].items()),
                'density declared reference missing or changed')
    require(refs.get(str(ROOT / 'runtime-inventory.lock.json')) == lock['runtime_inventory_sha256'],
            'density runtime inventory unbound')
    native_anchor = read(lock['native_receipt_lock_path'], NATIVE)
    require(native_anchor['reconciliation_sha256'] == NATIVE_REPORT,
            'density native report anchor differs')
    verify_refs(native_anchor['refs'])
    native = bound_json(Path(lock['native_receipt_lock_path']).with_name('reconciliation-v1.json'),
                        NATIVE_REPORT)
    completion, end = read(ROOT / 'completion.json'), read(ROOT / 'coordinator-exit.json')
    launch = read(ROOT / 'supervisor-launch.json')
    require(all(type(launch[k]) is int and launch[k] > 0
                and not Path('/proc', str(launch[k])).exists() for k in ('pid', 'supervisor_pid')),
            'density actors still live')
    report = bound_json(ROOT / 'reconciliation-v1.json', report_sha256)
    require(report['native_receipt_lock_sha256'] == NATIVE
            and report['native_reconciliation_sha256'] == NATIVE_REPORT,
            'density report native anchors differ')
    jobs, batches = closed_metadata(lock, report, completion, end)
    exact_tree(ROOT / 'source', lock['source_files'])
    attempt_files = []
    for key, job in jobs.items():
        out = ROOT / 'attempts' / key / 'attempt-0001'
        receipt = read(out / 'receipt.json', batches[key]['receipt_sha256'])
        claims(receipt)
        request = read(out / 'request.json')
        request_identity(receipt, request, key, batches[key]['status'])
        files = [out / 'receipt.json', *(out / n for n in receipt['evidence_files'])]
        for name, h in receipt['evidence_files'].items():
            require(Path(name).name == name and refs.get(str(out / name)) == h,
                    'density receipt evidence differs')
        exact_tree(out, files)
        attempt_files.extend(files)
        worker = Path(job['worker']['path'])
        require(all(refs.get(str(worker / n)) == h for n, h in job['worker']['files'].items()),
                'density worker input unbound')
        exact_tree(worker, [worker / n for n in job['worker']['files']])
        artifact = Path(job['artifact_path'])
        inventory = job['artifact_inventory']
        require(len(inventory) == 2 and {r['path'] for r in inventory} == {'model.json', 'projection.json'}
                and job['projection_bytes_included'] is True, 'complete density artifact charge required')
        for r in inventory:
            require(type(r['bytes']) is int and r['bytes'] >= 0
                    and len(owned(artifact / r['path'], r['sha256'])) == r['bytes'],
                    'density artifact bytes differ')
        require(type(job['artifact_bytes']) is int
                and job['artifact_bytes'] == sum(r['bytes'] for r in inventory),
                'density projection charge differs')
        exact_tree(artifact, [artifact / r['path'] for r in inventory])
    exact_tree(ROOT / 'attempts', attempt_files)
    verify_refs(refs)
    verify_refs(native_anchor['refs'])
    return {'lock': lock, 'report': report, 'native_report': native,
            'receipt_sha256': receipt_sha256, 'report_sha256': report_sha256,
            'publication_admitted': False, 'full_runtime_closure_claimed': False,
            'metric_native_and_cost_replay_required': True}
