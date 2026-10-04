"""Replay saved density metric arithmetic and repetitions without loading learners."""
import math
from pathlib import Path

from research.benchmark import density_publication_inputs as inputs

RUNTIME_PATH = inputs.BASE / 'shared-validation-python312-preparation-v2/runtime.lock.json'
RUNTIME_SHA256 = '6ee0f1b5d7bd3549a7a32452121da72cca60daf5a270d7ebae87cec599a7b4d2'
METRIC_SHA256 = '6c03892685df856f7a491ca79acb1c28fe8270ead8bbfea1ba50228cb6f0e7e3'
SEEDS = (101, 211, 307)
SIZES = (1, 4)
AUDITORS = {'catboost', 'linear', 'mlp'}
FAILURES = {'failed', 'timeout', 'ram_cap_exceeded', 'deadline_unstarted',
            'prelaunch_or_metric_failure'}
require = inputs.require


def finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def verify_metric(row, job, size, runtime):
    """Check recorded controls and the frozen evaluator's unclipped retention.

    This checks arithmetic and declarations, not the underlying learner outputs
    or validation CSV row count. Native selection, runtime and cost need replay
    separately. The caller owns these JSON bytes and their external anchors.
    """
    require(row['format'] == 'dope-benchmark-validation-pilot-metrics'
            and type(row['version']) is int and row['version'] == 1
            and row['implementation_sha256'] == runtime['metric_sha256'] == METRIC_SHA256
            and inputs.digest_identity(row['dependencies'])
            == inputs.digest_identity(runtime['expected_versions']), 'metric implementation differs')
    require(row['task'] == 'regression' and row['gate_profile_complete'] is False
            and row['mfs_v2'] is None
            and all(row.get(k) is None for k in ('ptf_v1', 'release_safe', 'superiority'))
            and row.get('counts_as_dope_win', False) is False, 'metric acquired a claim')
    rows = row['rows']
    require(set(rows) == {'train', 'validation', 'synthetic'}
            and all(type(v) is int and v >= 2 for v in rows.values())
            and type(job['worker']['train_rows']) is int
            and rows['train'] == job['worker']['train_rows']
            and type(size) is int and size in SIZES
            and rows['synthetic'] == rows['train'] * size, 'metric row declaration differs')
    require(all(finite(row[k]) and 0 <= row[k] <= 1 for k in
                ('marginal_ks_mean', 'pair_correlation_fidelity', 'c2st_auc'))
            and finite(row['metric_seconds']) and row['metric_seconds'] >= 0
            and finite(row['null_loss']) and row['null_loss'] >= 0, 'invalid metric value')
    for key, count in (('copy_counts', rows['synthetic']),
                       ('real_vs_real_control_counts', rows['validation'])):
        value = row[key]
        require(set(value) == {'exact', 'near'} and all(type(v) is int for v in value.values())
                and 0 <= value['exact'] <= value['near'] <= count, 'invalid copy control')
    require(set(row['utility']) == AUDITORS, 'complete prescribed auditors required')
    for value in row['utility'].values():
        if value.get('status') == 'failed':
            require(set(value) == {'status', 'error_type'}
                    and value['error_type'] in ('ValueError', 'RuntimeError'),
                    'invalid auditor failure')
            continue
        require(set(value) == {'trtr_loss', 'tstr_loss', 'informative', 'retention',
                               'low_signal_noninferior'}
                and all(finite(value[k]) and value[k] >= 0 for k in ('trtr_loss', 'tstr_loss')),
                'invalid utility evidence')
        gap = row['null_loss'] - value['trtr_loss']
        informative = gap >= 0.01 * abs(row['null_loss'])
        require(type(value['informative']) is bool and value['informative'] is informative,
                'informative control differs')
        if informative:
            require(gap > 0 and finite(value['retention'])
                    and value['low_signal_noninferior'] is None, 'invalid informative utility')
            expected = (row['null_loss'] - value['tstr_loss']) / gap
            require(finite(expected) and math.isclose(value['retention'], expected,
                    abs_tol=1e-12, rel_tol=1e-12), 'retention arithmetic differs')
        else:
            require(value['retention'] is None and value['low_signal_noninferior'] is
                    (value['tstr_loss'] <= value['trtr_loss'] + 0.01 * row['null_loss']),
                    'low signal control differs')
    return row


def replay_batch_metrics(job, lock, runtime, refs):
    """Read receipt-owned JSON; preserve raw non-success outcomes without metrics."""
    key = inputs.digest_identity(job)
    out = inputs.ROOT / 'attempts' / key / 'attempt-0001'
    receipt = inputs.bound_json(out / 'receipt.json', refs[str(out / 'receipt.json')])
    inputs.claims(receipt)
    evidence = receipt['evidence_files']

    def read(name):
        require(type(name) is str and Path(name).name == name and name in evidence
                and refs.get(str(out / name)) == evidence[name], 'unbound metric evidence')
        return inputs.bound_json(out / name, evidence[name])

    inputs.request_identity(receipt, read('request.json'), key, receipt['status'])
    require(type(job['metric_replay_required']) is bool, 'invalid replay requirement')
    if receipt['status'] != 'ok':
        require(receipt['status'] in FAILURES, 'unknown density outcome')
        return {'physical_job_sha256': key, 'status': receipt['status'], 'samples': []}
    batch = read('batch.json')
    require(batch['job_sha256'] == key and batch['metric_source_sha256'] == METRIC_SHA256
            == lock['metric_source_sha256'] and batch['sample_replays_exact'] is True
            and batch['gate_profile_complete'] is False
            and batch['official_tests_opened'] is False
            and batch['native_selection_changed'] is False and batch['global_family_selected'] is False
            and type(batch['new_generator_fits_started']) is int
            and batch['new_generator_fits_started'] == 0
            and all(batch.get(k) is None for k in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority'))
            and batch.get('counts_as_dope_win', False) is False, 'metric batch identity or claim differs')
    schedule = [[s['size_multiplier'], s['sample_seed']] for s in batch['samples']]
    require(inputs.digest_identity(schedule) == inputs.digest_identity(
            [[z, s] for z in SIZES for s in SEEDS]), 'complete metric schedule required')
    samples, validation_rows = [], set()
    for sample in batch['samples']:
        size, seed = sample['size_multiplier'], sample['sample_seed']
        name = f'n{size}-seed{seed}'
        metric_path, sample_path = out / (name + '.metric.json'), out / (name + '.csv')
        require(sample['metric_file'] == metric_path.name and sample['sample_file'] == sample_path.name
                and type(sample['rows']) is int and sample['rows'] == job['worker']['train_rows'] * size
                and sample['metric_sha256'] == refs.get(str(metric_path)) == evidence.get(metric_path.name)
                and sample['sample_sha256'] == refs.get(str(sample_path)) == evidence.get(sample_path.name),
                'sample metric binding differs')
        inputs.verify_refs({str(sample_path): sample['sample_sha256']})
        metric = verify_metric(read(metric_path.name), job, size, runtime)
        validation_rows.add(metric['rows']['validation'])
        sample_repeat = size == 1 and seed == 101
        require(sample['sample_replay'] == ('exact' if sample_repeat else 'not_repeated'),
                'sample repetition missing')
        repeated_sample = out / (name + '.repeat.csv')
        require((repeated_sample.name in evidence) is sample_repeat, 'sample repetition inventory differs')
        if sample_repeat:
            require(refs.get(str(repeated_sample)) == evidence[repeated_sample.name]
                    == sample['sample_sha256'], 'sample repetition differs')
            inputs.verify_refs({str(repeated_sample): sample['sample_sha256']})
        metric_repeat = job['metric_replay_required'] and size == 4 and seed == 101
        require(sample['metric_replay'] == ('exact' if metric_repeat else 'not_repeated'),
                'metric repetition missing')
        repeated_metric = out / (name + '.metric-replay.json')
        require((repeated_metric.name in evidence) is metric_repeat, 'metric repetition inventory differs')
        if metric_repeat:
            repeated = verify_metric(read(repeated_metric.name), job, size, runtime)
            comparable = lambda v: {k: x for k, x in v.items() if k != 'metric_seconds'}
            require(inputs.digest_identity(comparable(metric)) == inputs.digest_identity(comparable(repeated)),
                    'repeated metric differs')
        samples.append({'evidence': {'size_multiplier': size, 'sample_seed': seed,
                        'metric_path': str(metric_path), 'metric_sha256': sample['metric_sha256'],
                        'sample_sha256': sample['sample_sha256'], 'metric_replay': sample['metric_replay'],
                        'sample_replay': sample['sample_replay']}, 'metric': metric})
    require(len(validation_rows) == 1, 'validation row declarations differ')
    return {'physical_job_sha256': key, 'status': 'ok', 'samples': samples}


def bind_logical_metrics(report, batches):
    """Bind previously replayed physical samples to all closed logical cells."""
    statuses = {b['physical_job_sha256']: b['status'] for b in batches}
    require(len(batches) == len(statuses) == 502 and len(report['cells']) == 3600,
            'complete metric binding required')
    samples = {(b['physical_job_sha256'], s['evidence']['size_multiplier'], s['evidence']['sample_seed']): s
               for b in batches for s in b['samples']}
    logical = []
    for cell in report['cells']:
        sample = samples.get((cell['physical_job_sha256'], cell['size_multiplier'], cell['sample_seed']))
        require(cell['status'] == statuses[cell['physical_job_sha256']]
                and (sample is not None) is (cell['status'] == 'ok')
                and inputs.digest_identity(cell['sample_evidence'])
                == inputs.digest_identity(sample['evidence'] if sample else None),
                'logical metric evidence differs')
        logical.append({'physical_job_sha256': cell['physical_job_sha256'],
                        'size_multiplier': cell['size_multiplier'], 'sample_seed': cell['sample_seed'],
                        'metric': sample['metric'] if sample else None})
    return logical


def replay_closed_metrics(receipt_sha256, report_sha256):
    """Require full structural custody, then check every saved metric/control.

    No runtime is initialized and no sampler or learner is run. This does not
    admit publication until native selection, runtime and cost replay also pass.
    """
    closed = inputs.load_closed_inputs(receipt_sha256, report_sha256)
    anchor = inputs.bound_json(inputs.ROOT / 'receipt-lock-v1.json', receipt_sha256)
    refs, lock, report = anchor['refs'], closed['lock'], closed['report']
    require(lock['runtime_lock_files'].get(str(RUNTIME_PATH)) == RUNTIME_SHA256
            == refs.get(str(RUNTIME_PATH)), 'frozen metric runtime declaration differs')
    runtime = inputs.bound_json(RUNTIME_PATH, RUNTIME_SHA256)
    require(runtime['metric_sha256'] == lock['metric_source_sha256'] == METRIC_SHA256
            == refs.get(runtime['metric_source']), 'frozen metric source differs')
    inputs.verify_refs({runtime['metric_source']: METRIC_SHA256})
    batches = [replay_batch_metrics(j, lock, runtime, refs) for j in lock['jobs']]
    logical = bind_logical_metrics(report, batches)
    inputs.verify_refs(refs)
    return {'batches': batches, 'logical_metrics': logical, 'publication_admitted': False,
            'full_runtime_closure_claimed': False, 'native_selection_runtime_cost_replay_required': True}
