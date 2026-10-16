"""Replay native file custody and closed density costs without candidate execution."""
from pathlib import Path
import math
from research.benchmark import density_publication_inputs as owned
from research.benchmark import density_native_replay as native
identity = native.identity


def replay_native_files():
    """Hash the full parent inventory and bind workers and every trial artifact."""
    native.replay_native_metadata()
    root = native.ROOT
    a = owned.bound_json(root / 'receipt-lock-v1.json', owned.NATIVE)
    owned.require(a['reconciliation_sha256'] == owned.NATIVE_REPORT, 'native parent differs')
    r = owned.bound_json(root / 'reconciliation-v1.json', owned.NATIVE_REPORT)
    refs = a['refs']
    owned.verify_refs(refs)

    def read(p, h):
        owned.require(refs.get(str(p)) == h, 'native file declaration unbound')
        return owned.bound_json(p, h)
    workers, artifacts, worker_rows = ({}, {}, {})
    for c in r['default_native_cells']:
        w = c['worker']
        p = Path(w['path'])
        f = w['files']
        owned.require(set(f) == {'train.csv', 'validation.csv', 'projection.json',
                                'row-group-assignments.json', 'worker-manifest.json'},
                      'native worker inventory differs')
        owned.require(w['dataset'] == c['dataset']
                      and w['recorded_official_test_digest_read_as_metadata_only'] is True
                      and type(w['train_rows']) is int and w['train_rows'] >= 2,
                      'native worker scope differs')
        for n, h in f.items():
            owned.require(refs.get(str(p / n)) == h, 'native worker file unbound')
        owned.exact_tree(p, [p / n for n in f])
        manifest = read(p / 'worker-manifest.json', f['worker-manifest.json'])
        owned.require(manifest['dataset_id'] == c['dataset']
                      and identity(manifest['train_rows']) == identity(w['train_rows'])
                      and manifest['projected_hashes']['train'] == f['train.csv']
                      and manifest['projected_hashes']['validation'] == f['validation.csv']
                      and manifest['projection_sha256'] == f['projection.json']
                      and identity(manifest['split_hashes']) == identity(w['raw_training_derived_split_hashes']),
                      'native worker manifest differs')
        projection = read(p / 'projection.json', f['projection.json'])
        owned.require(projection['task'] == 'regression' and projection['fit_partition'] == 'train'
                      and type(projection['version']) is int and projection['version'] == 1
                      and type(w['projected_features']) is int and w['projected_features'] >= 1
                      and identity(projection['output_features']) == identity(w['projected_features']),
                      'native projection declaration differs')
        worker_rows[c['dataset']] = w
        signature = identity(w)
        owned.require(c['dataset'] not in workers or workers[c['dataset']] == signature, 'native worker changed across methods/configurations')
        workers[c['dataset']] = signature
        artifact = Path(c['artifact_path'])
        owned.require(artifact == Path(c['fit_attempt_receipt_path']).parent / 'artifact', 'native artifact/attempt path differs')
        artifact_charge(artifact, c['artifact_inventory'], c['artifact_bytes'], c['model_sha256'], f['projection.json'], refs)
        artifacts[str(artifact)] = True
    for trial in r['trials']:
        raw = read(Path(trial['attempt_receipt_path']), trial['attempt_receipt_sha256'])
        artifact = Path(trial['attempt_receipt_path']).parent / 'artifact'
        worker = worker_rows[trial['dataset']]
        owned.require(raw['identity']['train_sha256'] == worker['files']['train.csv']
                      and raw['identity']['validation_sha256'] == worker['files']['validation.csv'],
                      'native trial worker partition differs')
        artifact_charge(artifact, raw['artifact_inventory'], raw['artifact_bytes'], raw['artifact_sha256'], worker['files']['projection.json'], refs)
    owned.require(len(workers) == 100 and len(r['default_native_cells']) == 600
                  and len(r['trials']) == 1100 and len(artifacts) == 502, 'native file coverage differs')
    owned.verify_refs(refs)
    return dict(native_receipt_lock_sha256=owned.NATIVE, native_report_sha256=owned.NATIVE_REPORT,
                native_refs_hashed=len(refs), workers_verified=len(workers),
                default_native_cells_verified=600, distinct_artifacts_verified=len(artifacts),
                native_trial_artifacts_verified=1100, rows_parsed=False, models_deserialized=False,
                frozen_code_initialized=False, historical_full_runtime_closure_upgraded=False,
                declared_runtime_or_import_closure_certified=False, publication_admitted=False,
                official_tests_opened=False, sdv_v3_launched=False)


def artifact_charge(artifact, inventory, charge, model_sha, projection_sha, refs):
    owned.require(len(inventory) == 2 and {x['path'] for x in inventory} == {'model.json', 'projection.json'}
                  and type(charge) is int, 'native artifact scope differs')
    total = 0
    for row in inventory:
        p = artifact / row['path']
        owned.require(type(row['bytes']) is int and row['bytes'] >= 0
                      and refs.get(str(p)) == row['sha256'], 'native artifact declaration unbound')
        owned.require(len(owned.owned(p, row['sha256'])) == row['bytes'], 'native artifact charge differs')
        total += row['bytes']
    owned.require(charge == total and refs[str(artifact / 'model.json')] == model_sha
                  and refs[str(artifact / 'projection.json')] == projection_sha,
                  'native model or projection charge differs')
    owned.exact_tree(artifact, [artifact / x['path'] for x in inventory])


def execution_cost(receipt, original_cost, marker, batch, operation_file=None):
    """Replay saved nested timers; preserve the raw status and charge outer time once."""
    require = owned.require
    operation = receipt['operation']
    statuses = ('ok', 'failed', 'timeout', 'ram_cap_exceeded',
                'deadline_unstarted', 'prelaunch_or_metric_failure')
    require(native.number(original_cost) and original_cost >= 0,
            'invalid reported operation cost')
    require(receipt['status'] in statuses, 'unknown raw operation status')
    require(operation is None or type(operation) is dict, 'invalid execution declaration')
    if operation is not None:
        require(type(operation['new_operation_started']) is bool, 'invalid execution declaration')
    started = operation is not None and operation['new_operation_started']
    if not started:
        require(original_cost == 0 and marker is None and batch is None and operation_file is None,
                'unstarted execution cost or output')
        require(receipt['status'] != 'ok', 'unstarted success declaration')
        if operation is not None:
            require(operation['status'] == 'deadline_unstarted'
                    and native.number(operation['elapsed_seconds'])
                    and operation['elapsed_seconds'] == 0, 'unstarted operation differs')
        require(all(name == 'request.json' or
                    (name.startswith('admission-') and name.endswith('.json'))
                    for name in receipt['evidence_files']), 'unstarted operation has execution evidence')
        return dict(operation_seconds=0., started=False, raw_status=receipt['status'],
                    worker_inner_seconds=None, historical_runtime_closure_upgraded=False,
                    publication_admitted=False)
    require(marker is not None and native.number(operation['elapsed_seconds'])
            and operation['elapsed_seconds'] > 0
            and owned.digest_identity(original_cost) == owned.digest_identity(operation['elapsed_seconds']),
            'started operation cost differs')
    # The frozen coordinator uses min(600, remaining), which may be fractional.
    require(native.number(operation['timeout_seconds']) and 0 < operation['timeout_seconds'] <= 600
            and native.number(marker['timeout_seconds'])
            and owned.digest_identity(marker['timeout_seconds']) == owned.digest_identity(operation['timeout_seconds'])
            and marker['round_sha256'] == receipt['round_sha256'] == owned.ROUND
            and native.number(marker['start_monotonic']) and marker['start_monotonic'] >= 0,
            'execution marker or cap differs')
    require(operation['host'] == 'xbabe2' and operation['official_tests_opened'] is False
            and operation['foreign_processes_signaled'] is False, 'operation scope differs')
    require(operation['status'] in statuses and operation['status'] != 'deadline_unstarted'
            and (receipt['status'] == operation['status'] or
                 receipt['status'] == 'prelaunch_or_metric_failure'), 'raw operation status differs')
    if operation_file is not None:
        require(owned.digest_identity(operation) == owned.digest_identity(operation_file),
                'operation alias differs')
        require(type(operation['exit_code']) is int, 'invalid operation exit code')
    else:
        require(receipt['status'] == operation['status'] == 'prelaunch_or_metric_failure',
                'started operation record missing')
    inner = None
    if batch is not None:
        inner = batch['elapsed_seconds']
        require(native.number(inner) and inner > 0 and original_cost >= inner,
                'outer operation undercharges worker')
    if operation['status'] == 'ok':
        require(operation_file is not None and operation['exit_code'] == 0
                and inner is not None and inner <= operation['timeout_seconds'],
                'successful operation cap or exit differs')
    # Outer wall time includes worker time and measured shutdown/receipt overhead.
    # An overrun retains its failed/timeout outcome; it still costs the measured time.
    return dict(operation_seconds=original_cost, started=True, raw_status=receipt['status'],
                worker_inner_seconds=inner, historical_runtime_closure_upgraded=False,
                publication_admitted=False)


def replay_closed_costs(receipt_sha256, report_sha256):
    """Require complete closed inputs; runtime closure and publication stay unadmitted.

    The input gate must account for generated auxiliary attempt trees separately
    before this can replay the real round. No partial-matrix publication path.
    """
    closed = owned.load_closed_inputs(receipt_sha256, report_sha256)
    files = replay_native_files()
    lock, report = (closed['lock'], closed['report'])
    anchor = owned.bound_json(owned.ROOT / 'receipt-lock-v1.json', receipt_sha256)
    refs = anchor['refs']
    jobs = {owned.digest_identity(j): j for j in lock['jobs']}
    costs = []
    for row in report['batches']:
        key = row['physical_job_sha256']
        out = owned.ROOT / 'attempts' / key / 'attempt-0001'

        def read(name):
            p = out / name
            owned.require(str(p) in refs, 'unbound execution cost input')
            return owned.bound_json(p, refs[str(p)])
        receipt = read('receipt.json')
        request = read('request.json')
        owned.request_identity(receipt, request, key, row['status'])
        evidence = receipt['evidence_files']
        marker = read('operation-started.json') if 'operation-started.json' in evidence else None
        batch = read('batch.json') if 'batch.json' in evidence else None
        operation_file = read('operation.json') if 'operation.json' in evidence else None
        value = execution_cost(receipt, row['operation_seconds'], marker, batch, operation_file)
        op = receipt['operation']
        if value['started'] and operation_file is not None:
            owned.require(owned.digest_identity(op['runtime_lock_files']) == owned.digest_identity(lock['runtime_lock_files'])
                          and op['request_sha256'] == refs[str(out / 'request.json')]
                          and op['worker_log_sha256'] == refs[str(out / 'worker.log')],
                          'execution reference binding differs')
            admissions = [read(n) for n, h in evidence.items() if n.startswith('admission-') and n.endswith('.json') and (h == op['admission_sha256'])]
            owned.require(len(admissions) == 1 and admissions[0]['admitted'] is True
                          and admissions[0]['blockers'] == [] and admissions[0]['round_sha256'] == owned.ROUND
                          and admissions[0]['host'] == 'xbabe2' and admissions[0]['requested']['requires_gpu'] is False
                          and all(owned.digest_identity(admissions[0]['requested'][k]) == owned.digest_identity(v)
                                  for k, v in lock['roles']['coordinator'].items())
                          and type(admissions[0]['requested']['pid']) is int and admissions[0]['requested']['pid'] > 0,
                          'execution admission binding differs')
            peak = op['peak_resident_bytes_including_coordinator']
            cap = lock['roles']['coordinator']['ram_bytes']
            owned.require(type(peak) is int and peak >= 0 and type(cap) is int and cap > 0,
                          'execution memory declaration differs')
            if op['status'] == 'ok':
                owned.require(peak <= cap, 'successful operation exceeded memory cap')
        if batch is not None:
            owned.require(batch['job_sha256'] == key, 'worker timing identity differs')
        costs.append(value['operation_seconds'])
    total = math.fsum(costs)
    owned.require(len(costs) == len(jobs) == 502 and native.number(report['new_operation_seconds'])
                  and math.isclose(total, report['new_operation_seconds'], abs_tol=1e-9, rel_tol=1e-12)
                  and native.number(report['native_trial_operation_seconds'])
                  and math.isclose(report['native_trial_operation_seconds'],
                                   closed['native_report']['native_trial_operation_seconds'],
                                   abs_tol=1e-9, rel_tol=1e-12), 'closed operation totals differ')
    owned.verify_refs(refs)
    return dict(native_file_custody=files, physical_batches=502, operation_seconds=total,
                prior_native_trial_operation_seconds=report['native_trial_operation_seconds'],
                prior_fit_cost_multiplied_by_logical_reuse=False, raw_statuses_preserved=True,
                runtime_closure_replay_required=True, publication_admitted=False,
                official_tests_opened=False, sdv_v3_launched=False, mfs_v2=None, ptf_v1=None,
                release_safe=None, superiority=None, counts_as_dope_win=False)
