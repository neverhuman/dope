"""Complete opaque receipt graph for ARF publisher rejection controls."""
from contextlib import ExitStack, contextmanager
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from research.benchmark import publish_arf_population_matched as pub


@contextmanager
def closed_fixture(base, input_rows, reference_rows, sampling_fit_seed=11, sampling_mutation=None,
                   common_mutation=None, sampling_unavailable=None):
    assert sampling_unavailable in (None, 'genuine_timeout', 'false_unavailable')
    root, sampling = base / 'common', base / 'sampling'
    root.mkdir(); sampling.mkdir(); refs = {}

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def record(path, value):
        refs[str(path)] = write(path, value); return refs[str(path)]

    def closure(path, pin, count):
        record(path / 'coordinator-exit.json', dict(exit_code=0, round_sha256=pin, elapsed_seconds=0))
        record(path / 'completion.json', dict(round_sha256=pin, jobs=count, logical_cells=1200, counts={'ok': count}))
        record(path / 'supervisor-launch.json', dict(round_sha256=pin, pid=-1, supervisor_pid=-1))

    workers, sample_jobs, native_cells = {}, [], []
    for number in range(100):
        dataset = f'{number:016x}'; worker_path = base / 'workers' / dataset
        h = record(worker_path / 'worker-metadata.json', dict(fixture_dataset=dataset))
        worker = dict(path=str(worker_path), files={'worker-metadata.json': h}, train_rows=2, projected_features=1)
        workers[dataset] = worker; keys = []
        for configuration in range(1 if number == 0 else 2):
            key = pub.digest(dict(fixture_dataset=dataset, fixture_configuration=configuration)); keys.append(key)
            artifact = base / 'artifacts' / key
            model = record(artifact / 'model-metadata.json', dict(fixture_identifier=key))
            projection = record(artifact / 'projection-metadata.json', dict(fixture_identifier=dataset))
            inventory = {p.name: dict(sha256=refs[str(p)], bytes=p.stat().st_size) for p in artifact.iterdir()}
            sample_jobs.append(dict(dataset=dataset, fit_job_sha256=key, worker=worker,
                artifact_path=str(artifact), artifact_inventory=inventory,
                artifact_bytes=sum(v['bytes'] for v in inventory.values()), fit_seed=sampling_fit_seed))
        native_cells.append(dict(dataset=dataset, author_default_job_sha256=keys[0], selected_job_sha256=keys[-1],
            default_native_kpi={'fixture': True}, selected_native_kpi={'fixture': True}))
    assert len(sample_jobs) == 199
    native_report = dict(cells=native_cells, native_objective={'fixture': True},
                         cost={'operation_seconds': 0, 'scheduler_wall_seconds': 0})
    fit_root = base / 'dope-fit'
    fit_pin = record(fit_root / 'reconciliation-v1.json', dict(fit_cells=[
        dict(dataset=d, original_fit_job={'worker': w}) for d, w in workers.items() for _ in pub.dope.fits.PROFILES]))
    sample_source = sampling / 'source' / 'fixture.py'; record(sample_source, dict(fixture_source='sampling'))
    sample_lock = dict(jobs=sample_jobs, source_files={str(sample_source): refs[str(sample_source)]})
    sample_round = record(sampling / 'round.lock.json', sample_lock); closure(sampling, sample_round, 199)
    sampling_batches = []
    for job in sample_jobs:
        key = pub.digest(job); out = sampling / 'attempts' / key / 'attempt-0001'; samples = []
        for size in pub.dope.SIZES:
            for seed in pub.dope.SEEDS:
                path = out / f'n{size}-seed{seed}.csv'
                pin = (None if sampling_mutation == 'null_omitted_sample' and not sampling_batches
                       and (size, seed) == (4, 307)
                       else record(path, dict(fixture=True, size_multiplier=size, sample_seed=seed)))
                samples.append(dict(sample_file=path.name, sample_path=str(path), sample_sha256=pin,
                    sample_seed=seed, size_multiplier=size, rows=2*size, columns=2,
                    sample_replay='exact' if (size, seed) == (1, 101) else 'not_repeated'))
        files = {s['sample_file']: s['sample_sha256'] for s in samples if s['sample_sha256'] is not None}
        replay = out / 'n1-seed101.repeat.csv'
        replay.write_bytes((out / 'n1-seed101.csv').read_bytes())
        refs[str(replay)] = files[replay.name] = hashlib.sha256(replay.read_bytes()).hexdigest()
        batch = dict(job_sha256=key, round_sha256=sample_round, status='ok',
            artifact_bytes=job['artifact_bytes'], projection_bytes_included=True,
            samples=[{k: v for k, v in s.items() if k != 'sample_path'} for s in samples])
        files['sample-batch.json'] = record(out / 'sample-batch.json', batch)
        receipt = out / 'receipt.json'; pin = record(receipt, dict(job=job, job_sha256=key,
            status='ok', round_sha256=sample_round, result=batch, evidence_files=files,
            operation=dict(new_operation_started=True, elapsed_seconds=0, exit_code=0, host='xbabe2'),
            official_tests_opened=False, new_generator_fits_started=0, native_selection_changed=False,
            mfs_v2=None, ptf_v1=None, release_safe=None, superiority=None))
        sampling_batches.append(dict(job=job, job_sha256=key, samples=samples, status='ok',
            receipt_path=str(receipt), receipt_sha256=pin, artifact_bytes=job['artifact_bytes']))
    if sampling_mutation == 'omitted_both_sample':
        path = sampling_batches[0]['samples'][0]['sample_path']; refs.pop(path)
        Path(path).write_text('{"fixture_changed_after_pin": true}\n')
    elif sampling_mutation == 'sample_digest':
        sampling_batches[0]['samples'][0]['sample_sha256'] = '0' * 64
    elif sampling_mutation == 'receipt_digest':
        sampling_batches[0]['receipt_sha256'] = '0' * 64
    else:
        assert sampling_mutation in (None, 'original_anchor_sample_omission', 'null_omitted_sample')
    if sampling_unavailable == 'genuine_timeout':
        original = sampling_batches[0]; path = Path(original['receipt_path'])
        receipt = json.loads(path.read_bytes()); receipt.update(status='timeout', result=None)
        receipt['operation']['elapsed_seconds'] = 17
        original.update(status='timeout', samples=[], receipt_sha256=record(path, receipt))
        completion = json.loads((sampling / 'completion.json').read_bytes())
        completion['counts'] = {'ok': 198, 'timeout': 1}
        record(sampling / 'completion.json', completion)
    sample_report = dict(complete_sampling_matrix=True, logical_sample_cells=1200,
        physical_batches_evidence=sampling_batches, physical_status_counts={'ok': 199},
        new_operation_seconds=0, coordinator_wall_seconds=0)
    if sampling_unavailable == 'genuine_timeout':
        sample_report.update(physical_status_counts={'ok': 198, 'timeout': 1}, new_operation_seconds=17)
    sample_report_pin = record(sampling / 'reconciliation-v1.json', sample_report)
    sample_anchor = dict(complete_sampling_matrix=True, round_sha256=sample_round,
        reconciliation_sha256=sample_report_pin, refs=dict(refs))
    if sampling_mutation == 'original_anchor_sample_omission':
        sample_anchor['refs'].pop(sampling_batches[0]['samples'][0]['sample_path'])
    sample_anchor_pin = record(sampling / 'receipt-lock-v1.json', sample_anchor)
    metric_source = 'a' * 64; common_jobs = []
    for original in sampling_batches:
        job = original['job']
        if sampling_unavailable and original is sampling_batches[0]: continue
        common_jobs.append(dict(method='ARF', track='common-numeric', final=False, fit_seed=11,
            dataset=job['dataset'], worker=job['worker'], sampling_job_sha256=original['job_sha256'],
            sampling_receipt_path=original['receipt_path'], sampling_receipt_sha256=original['receipt_sha256'],
            artifact_bytes=job['artifact_bytes'], metric_replay_required=job['dataset'] == f'{0:016x}',
            frozen_samples=original['samples']))
    by_sample = {j['sampling_job_sha256']: j for j in common_jobs}
    by_fit = {j['fit_job_sha256']: j for j in sample_jobs}; logical = []
    for row in input_rows:
        selection = native_cells[int(row['dataset'], 16)]
        fit = selection['author_default_job_sha256' if row['configuration'] == 'author_default' else 'selected_job_sha256']
        sample_job = by_fit[fit]; sample_key = pub.digest(sample_job)
        common_key = pub.digest(by_sample[sample_key]) if sample_key in by_sample else None
        logical.append(dict(dataset=row['dataset'], configuration=row['configuration'], fit_seed=11,
            sample_seed=row['sample_seed'], size_multiplier=row['size_multiplier'], fit_job_sha256=fit,
            sampling_job_sha256=sample_key, physical_job_sha256=common_key,
            sampling_status='timeout' if sampling_unavailable == 'genuine_timeout' and common_key is None else 'ok',
            charged_artifact_bytes=sample_job['artifact_bytes'],
            unavailable_reason='timeout' if common_key is None else None, counts_as_dope_win=False))
    common_source = root / 'source' / 'fixture.py'; record(common_source, dict(fixture_source='common'))
    lock = dict(jobs=common_jobs, logical_cells=logical, source_files={str(common_source): refs[str(common_source)]},
        gpu_operations_enabled=False, official_tests_opened=False, full_campaign_admitted=False,
        native_selection_changed=False, sampling_receipt_lock_sha256=sample_anchor_pin, metric_source_sha256=metric_source)
    common_round = record(root / 'round.lock.json', lock); closure(root, common_round, len(common_jobs))
    auxiliary = dict(round_sha256=common_round, pinned_before_reconciliation_metric_reads=True,
        extra_outputs_used_for_metrics_or_selection=False, batches={})
    evidence, receipt_pins = {}, {}
    for job in common_jobs:
        key = pub.digest(job); out = root / 'attempts' / key / 'attempt-0001'; files = {}; records = []
        for frozen in job['frozen_samples']:
            size, seed = frozen['size_multiplier'], frozen['sample_seed']; retention = {101: -.2, 211: .6, 307: 1.2}[seed]
            metric = dict(implementation_sha256=metric_source, task='regression', gate_profile_complete=False, mfs_v2=None,
                rows=dict(train=2, synthetic=2*size, validation=2), null_loss=100,
                utility={a: dict(trtr_loss=50, tstr_loss=100-50*retention, informative=True, retention=retention)
                         for a in ('catboost', 'linear', 'mlp')},
                copy_counts=dict(exact=0, near=0), real_vs_real_control_counts=dict(exact=0, near=0),
                dependencies={}, marginal_ks_mean=.5, pair_correlation_fidelity=.5, c2st_auc=.5)
            path = out / f'n{size}-seed{seed}.metric.json'; pin = record(path, metric); files[path.name] = pin
            repeated = job['metric_replay_required'] and (size, seed) == (4, 101)
            if repeated:
                replay = out / f'n{size}-seed{seed}.metric-replay.json'; files[replay.name] = record(replay, metric)
            sample = dict(frozen, metric_file=path.name, metric_sha256=pin,
                metric_replay='exact' if repeated else 'not_repeated'); records.append(sample)
            evidence[key, size, seed] = dict(sample, metric_path=str(path), metrics=metric)
        batch = out / 'batch.json'; files[batch.name] = record(batch, dict(job_sha256=key,
            metric_source_sha256=metric_source, samples=records))
        receipt_value = dict(job=job, round_sha256=common_round, status='ok', evidence_files=dict(files),
            operation=dict(new_operation_started=True, elapsed_seconds=0, exit_code=0, host='xbabe2'),
            new_generator_fits_started=0, native_selection_changed=False, global_family_selected=False,
            official_tests_opened=False, mfs_v2=None, ptf_v1=None, release_safe=None, superiority=None)
        if common_mutation and not receipt_pins: common_mutation(receipt_value)
        receipt = out / 'receipt.json'; receipt_pins[key] = record(receipt, receipt_value)
        auxiliary['batches'][key] = dict(files=dict(files, **{'receipt.json': receipt_pins[key]}), directories=[])
    auxiliary_pin = record(root / 'closed-batch-output-inventory-v1.json', auxiliary)
    cells = [dict(row, method='ARF', status='sampling_unavailable' if row['physical_job_sha256'] is None else 'ok',
        projection_bytes_included=True,
        l3_artifact_cap_satisfied=True, mfs_v2=None, ptf_v1=None, release_safe=None, superiority=None,
        validation_receipt_sha256=receipt_pins[row['physical_job_sha256']] if row['physical_job_sha256'] else None,
        sample_evidence=evidence[row['physical_job_sha256'], row['size_multiplier'], row['sample_seed']]
        if row['physical_job_sha256'] else None) for row in logical]
    report = dict(complete_matrix=True, logical_sample_cells=1200, native_selection_changed=False,
        official_tests_opened=False, mfs_v2=None, ptf_v1=None, release_safe=None, superiority=None,
        cells=cells, logical_status_counts={'ok': 1200}, physical_status_counts={'ok': 199},
        new_metric_operation_seconds=0, coordinator_wall_seconds=0)
    if sampling_unavailable:
        report.update(physical_status_counts={'ok': 198},
                      logical_status_counts={'sampling_unavailable': 12, 'ok': 1188})
    native_path = Path(pub.native.__file__).with_name('results') / (pub.native.NAME + '.json')
    original_bound = pub.guard.bound; decoded_metrics = []

    def bound(path, pin, root):
        if Path(path) == native_path:
            assert pin == pub.NATIVE_PUBLICATION; return copy.deepcopy(native_report)
        if str(path).endswith('.metric.json'): decoded_metrics.append(str(path))
        return original_bound(path, pin, root)

    def run(change=None, omitted_leaf=None, mutate_leaf=False):
        current = copy.deepcopy(report)
        if change: change(current)
        report_pin = write(root / 'reconciliation-v1.json', current); common_refs = dict(refs)
        if omitted_leaf: common_refs.pop(omitted_leaf)
        if mutate_leaf: Path(omitted_leaf).write_text('{"fixture_changed_after_pin": true}\n')
        pin = write(root / 'receipt-lock-v1.json', dict(complete_matrix=True, reconciliation_sha256=report_pin,
            round_sha256=common_round, refs=common_refs))
        decoded_metrics.clear()
        result = pub.build(pin, report_pin, auxiliary_pin); pub.validate_report(result); return result

    with ExitStack() as stack:
        for obj, name, value in [(pub, 'ROOT', root), (pub, 'SAMPLING', sampling), (pub, 'BASE', base),
            (pub, 'SAMPLE_ROUND', sample_round), (pub.native, 'BASE', base), (pub.dope.fits, 'BASE', base),
            (pub.dope.fits, 'ROOT', fit_root), (pub.dope.fits, 'REPORT', fit_pin)]:
            stack.enter_context(patch.object(obj, name, value))
        stack.enter_context(patch.object(pub.native, 'build', return_value=native_report))
        stack.enter_context(patch.object(pub.guard, 'bound', side_effect=bound))
        stack.enter_context(patch.object(pub, 'shared_inventory', return_value={'expected_versions': {}}))
        stack.enter_context(patch.object(pub, 'reference_cells', return_value=reference_rows))
        yield run, next(c['sample_evidence']['sample_path'] for c in cells if c['sample_evidence']), decoded_metrics
