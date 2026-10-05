"""Publish complete accounting of existing CTGAN/TVAE samples against fixed DOPE profiles."""
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

from research.benchmark import density_publication_inputs as inputs
from research.benchmark import density_declared_runtime_replay as runtime
from research.benchmark import density_metric_replay as metrics
from research.benchmark import publish_sdv_population_native as native
from research.benchmark import sdv_matched_aggregate as common

BASE = inputs.BASE
ROOT = BASE / 'sdv-s3-population-shared-validation-v1'
ROUND = 'fa9ebc88488e45ba8bc9edcaf9584554dba84c01024cf7cfce5a7d70690ff10f'
RECEIPTS = '3635a59fb18f8654a872eb6c5c88c995d11eecd50ffc00b0da14a2acdd871766'
RECONCILIATION = '12bf30223ddafe760d6f97222f81de708ce2c6e48d0d25c994f24943ca0f4075'
REFERENCE = '5264b88a40ad5efb21d9b119789caeb13e71e911a17f1d8b57f1d8e8ac31732a'
NATIVE_RECEIPTS = 'f49440e4c66ef9c71452d3c969fe6637ec8908086fc317e75ca83d165b715c8c'
NATIVE_REPORT = 'c6ab816ec7890aa2d22aa7e77c1ee22c7e086e2b46cacae79c5d9b2b1cf40266'
NAME = 'sdv-matched-population-validation'
REPORT_SHA256 = '4dc367eb78159a0386882d23ae1f99b0aa2cc9f425d1aae499b6f3453222a44e'
AUXILIARY = '47d4d737921d5aafbf553cd5527d4b15801d0c273786b3c08f47f36a4640ec12'
RESULTS = Path(__file__).with_name('results')


def committed(path, expected):
    inputs.digest_string(expected)
    p = Path(path)
    inputs.require(p.is_file() and not any(q.is_symlink() for q in (p, *p.parents)),
                   'committed reference is not an owned regular file')
    data = p.read_bytes()
    inputs.require(hashlib.sha256(data).hexdigest() == expected, 'committed reference changed')
    return json.loads(data, object_pairs_hook=inputs.pairs, parse_float=inputs.finite_float,
                      parse_constant=lambda _: inputs.require(False, 'nonfinite report JSON'))


def shared_runtime(lock, read):
    """Rehash stored import inventories without loading packages or upgrading historical closure."""
    manifests = [inputs.bound_json(p, h) for p, h in lock['runtime_lock_files'].items()]
    original = next(m for m in manifests if 'python' in m)
    auxiliary = next(m for m in manifests if 'stdlib_all_files' in m)
    inputs.require(auxiliary['runtime_sha256'] == lock['runtime_lock_files'][str(
        BASE / 'shared-validation-python312-preparation-v2/runtime.lock.json')]
        and original['metric_sha256'] == lock['metric_source_sha256'], 'metric runtime changed')
    count = 0
    for m in manifests:
        site = inputs.safe(m['site'])
        count += len(runtime.files_under(site, {str(site / p): v for p, v in m['files'].items()}))
    library = Path('/usr/lib/python3.12')
    files = {str(p) for p in library.rglob('*') if p.is_file()}
    links = {str(p) for p in library.rglob('*') if p.is_symlink()}
    inputs.require(files == set(auxiliary['stdlib_all_files'])
                   and links == set(auxiliary['stdlib_aliases']), 'system import inventory changed')
    for p, row in auxiliary['stdlib_all_files'].items():
        runtime.rehash_system_file(p, row['sha256'], row['bytes'])
    for p, row in auxiliary['stdlib_aliases'].items():
        q = Path(p)
        inputs.require(str(q.readlink()) == row['target']
                       and str(q.resolve(strict=True)) == row['resolved'], 'system import alias changed')
    inputs.require(original['python'] == '/usr/bin/python3.12'
        and str(Path('/usr/bin/python3').readlink()) == original['python_symlink']
        and not Path(original['absent_zip']).is_symlink()
        and not Path(original['absent_zip']).exists(), 'interpreter declaration changed')
    runtime.rehash_system_file(original['python'], original['python_sha256'])
    inputs.require(not list(inputs.safe(original['empty_bytecode_cache']).rglob('*')),
                   'shared bytecode cache not empty')
    return dict(package_files_rehashed=count, system_import_files_rehashed=len(files),
        system_aliases_checked=len(links), dependencies_initialized=False,
        generator_invoked=False, historical_runtime_closure_upgraded=False,
        all_possible_dynamic_loads_and_system_closure_certified=False)


def build():
    anchor = inputs.bound_json(ROOT / 'receipt-lock-v1.json', RECEIPTS)
    inputs.require(anchor['round_sha256'] == ROUND
        and anchor['reconciliation_sha256'] == RECONCILIATION
        and anchor['native_parent_receipt_lock_sha256'] == NATIVE_RECEIPTS,
        'frozen shared receipt anchors differ')
    auxiliary = inputs.bound_json(ROOT / 'matched-auxiliary.lock-v1.json', AUXILIARY)
    inputs.require(auxiliary['receipt_lock_sha256'] == RECEIPTS and auxiliary['original_receipts_changed'] is False,
                   'auxiliary evidence anchor differs')
    for tree in auxiliary['trees'].values():
        for row in tree['files']:
            inputs.require(len(inputs.owned(row['path'], row['sha256'])) == row['bytes'], 'auxiliary bytes changed')
    refs = anchor['refs']
    inputs.verify_refs(refs)  # Frozen receipts/source/model/sample bytes before any metric parse.
    def read(path, expected=None):
        path = str(path)
        inputs.require(path in refs and (expected is None or refs[path] == expected),
                       'unbound publication reference')
        return inputs.bound_json(path, refs[path])
    lock = read(ROOT / 'round.lock.json', ROUND)
    inputs.exact_tree(ROOT / 'source', lock['source_files'])
    for group in ('source_files', 'frozen_references'):
        inputs.require(all(refs.get(p) == h for p, h in lock[group].items()),
                       'frozen source or runtime reference missing')
    runtime_proof = shared_runtime(lock, read)
    evaluator = inputs.bound_json(metrics.RUNTIME_PATH, metrics.RUNTIME_SHA256)
    report = read(ROOT / 'reconciliation-v1.json', RECONCILIATION)
    closed, end = read(ROOT / 'completion.json'), read(ROOT / 'coordinator-exit.json')
    launch = read(ROOT / 'supervisor-launch.json')
    inputs.require(all(type(launch[k]) is int and launch[k] > 0
        and not Path('/proc', str(launch[k])).exists() for k in ('pid', 'supervisor_pid')),
        'shared validation actors still live')
    inputs.require(report['complete_frozen_matrix_accounting'] is True
        and report['logical_cells'] == closed['logical_cells'] == 2400
        and report['physical_batches'] == closed['jobs'] == len(lock['jobs']) == 78
        and type(end['exit_code']) is int and end['exit_code'] == 0
        and end['round_sha256'] == ROUND
        and refs[str(ROOT / 'coordinator.log')] == end['coordinator_log_sha256'],
        'complete shared validation closure required')
    for row in (lock, report):
        inputs.require(all(row[k] is False for k in ('native_selection_changed',
            'shared_kpi_used_for_selection', 'sdv_v3_launched', 'official_tests_opened'))
            and type(row['new_generator_fits_started']) is int and row['new_generator_fits_started'] == 0
            and type(row['new_samples_generated']) is int and row['new_samples_generated'] == 0,
            'shared evaluation acquired selection or fitting authority')
    pin = read(ROOT / 'closed-receipt-pin-v1.json')
    jobs = {inputs.digest_identity(j): j for j in lock['jobs']}
    batches = {b['physical_job_sha256']: b for b in report['batches']}
    inputs.require(len(jobs) == len(batches) == 78 and set(jobs) == set(batches)
        and Counter(b['status'] for b in batches.values()) == closed['counts'] == {'ok': 78},
        'closed physical matrix differs')
    inputs.require(set(auxiliary['trees']) == set(jobs), 'auxiliary physical coverage differs')
    measured, attempt_files, attempt_directories = {}, [], set()
    seconds = 0
    for key, job in jobs.items():
        b = batches[key]; out = ROOT / 'attempts' / key / 'attempt-0001'
        receipt_path = str(out / 'receipt.json')
        inputs.require(pin['receipts'][receipt_path] == b['receipt_sha256'] == refs[receipt_path],
                       'receipt differs from its frozen pin')
        receipt = read(receipt_path)
        inputs.require(inputs.digest_identity(receipt['job']) == key
            and receipt['round_sha256'] == ROUND and receipt['status'] == b['status']
            and inputs.digest_identity(receipt['operation']) == inputs.digest_identity(b['operation']),
            'batch identity or operation differs')
        files = [out / 'receipt.json', *(out / n for n in receipt['evidence_files'])]
        inputs.require(all(Path(n).name == n and refs.get(str(out / n)) == h
            for n, h in receipt['evidence_files'].items()), 'batch evidence reference differs')
        tree = auxiliary['trees'][key]
        auxiliary_files = [inputs.safe(r['path']) for r in tree['files']]
        directories = {inputs.safe(p) for p in tree['directories']}
        inputs.require(all(p.is_relative_to(out) and p != out for p in auxiliary_files + list(directories)),
                       'auxiliary evidence escaped batch')
        files += auxiliary_files; inputs.exact_tree(out, files, directories)
        attempt_files.extend(files); attempt_directories.update(directories)
        batch = read(out / 'batch.json')
        inputs.require(batch['job_sha256'] == key and batch['sample_replays_exact'] is True
            and batch['new_generator_fits_started'] == batch['new_samples_generated'] == 0
            and batch['historical_runtime_closure_upgraded'] is False
            and batch['gate_profile_complete'] is False
            and inputs.digest_identity([(s['size_multiplier'], s['sample_seed']) for s in batch['samples']])
            == inputs.digest_identity([(z, s) for z in (1, 4) for s in (101, 211, 307)]),
            'original samples, schedule or gate claims differ')
        worker = inputs.safe(job['worker']['path'])
        inputs.exact_tree(worker, [worker / n for n in job['worker']['files']])
        fit = read(Path(job['native_receipt_path']).parent / 'fit.json', job['fit_receipt_sha256'])
        artifact = inputs.safe(job['artifact_path']); inventory = fit['artifact_inventory']
        inputs.exact_tree(artifact, [artifact / row['path'] for row in inventory])
        inputs.require(job['projection_bytes_included'] is True
            and job['artifact_bytes'] == fit['artifact_bytes'] == sum(r['bytes'] for r in inventory)
            and sorted(r['path'] for r in inventory) == ['model.json', 'model.pt', 'projection.json'],
            'complete charged artifact inventory required')
        for row in inventory:
            inputs.require(len(inputs.owned(artifact / row['path'], row['sha256'])) == row['bytes'],
                           'artifact byte count differs')
        for sample in batch['samples']:
            repeat = job['metric_replay_required'] and sample['size_multiplier'] == 4 and sample['sample_seed'] == 101
            inputs.require(sample['sample_replay'] == 'original_exact'
                and sample['metric_replay'] == ('exact' if repeat else 'not_repeated'), 'recorded repetition differs')
            metric = read(out / sample['metric_file'], sample['metric_sha256'])
            inputs.require(refs[str(Path(sample['sample_file']))] == sample['sample_sha256']
                and metric['implementation_sha256'] == lock['metric_source_sha256']
                and metric['mfs_v2'] is None and metric['gate_profile_complete'] is False,
                'metric or sample binding differs')
            metrics.verify_metric(metric, job, sample['size_multiplier'], evaluator)
            measured[key, sample['size_multiplier'], sample['sample_seed']] = metric
        seconds += receipt['operation']['elapsed_seconds']
    inputs.exact_tree(ROOT / 'attempts', attempt_files, attempt_directories)
    inputs.require(abs(seconds - report['operation_seconds']) <= 1e-9, 'shared operation cost differs')
    reference = committed(RESULTS /
                          'density-matched-population-validation.json', REFERENCE)
    density = inputs.bound_json(BASE / 'density-s3-population-shared-validation-v1/round.lock.json',
                                reference['source_locks']['density_round'])
    workers = {j['dataset']: j['worker'] for j in density['jobs']}
    for job in jobs.values():
        inputs.require(inputs.digest_identity(job['worker']) == inputs.digest_identity(workers[job['dataset']]),
                       'DOPE and neural training-derived validation views differ')
    # Recompute native winners from all bounded original trials, not shared utility or sampling success.
    native_report = native.build(NATIVE_RECEIPTS, NATIVE_REPORT)
    for lineage in native_report['dataset_lineages']:
        inputs.require(inputs.digest_identity(lineage['files']) == inputs.digest_identity(workers[lineage['dataset']]['files']),
                       'matched all-lineage file hashes differ')
    selection = {(r['dataset'], r['method']): r for r in native_report['cells']}
    cells = [c for c in reference['cells'] if c['method'] == 'DOPE']
    inputs.require(len(report['cells']) == len(lock['logical_cells']) == 2400, 'logical schedule incomplete')
    for frozen, c in zip(lock['logical_cells'], report['cells']):
        inputs.require(inputs.digest_identity(frozen) == inputs.digest_identity({k: c[k] for k in frozen}),
                       'logical native cell changed')
        selected = selection[c['dataset'], c['method']]
        kind = 'default' if c['configuration_name'] == 'author_default' else 'native_selected'
        index = selected['default_trial_index' if kind == 'default' else 'native_selected_trial_index']
        inputs.require(c['native_trial_index'] == index, 'common evaluation changed native winner')
        if c['physical_job_sha256'] is not None:
            metric = measured[c['physical_job_sha256'], c['size_multiplier'], c['sample_seed']]
            inputs.require(all(inputs.digest_identity(c[k]) == inputs.digest_identity(metric[k])
                for k in ('utility', 'copy_counts', 'real_vs_real_control_counts', 'null_loss',
                          'marginal_ks_mean', 'pair_correlation_fidelity', 'c2st_auc')),
                'published metric differs from frozen receipt')
        inputs.require(c['method_failure_inferred'] is False, 'missing cell became a method failure')
        charge = c['charged_artifact_bytes']
        row = dict(c, configuration=kind,
            artifact_within_l3_cap=None if charge is None else charge <= 10240, counts_as_dope_win=False)
        # This is the original fit's canonical job digest, not an API key or a model hash.
        row['physical_artifact_job_sha256'] = row.pop('physical_artifact_key')
        cells.append(row)
    summary = common.aggregate(cells)
    inputs.verify_refs(refs)
    inputs.bound_json(ROOT / 'receipt-lock-v1.json', RECEIPTS)
    inputs.bound_json(ROOT / 'matched-auxiliary.lock-v1.json', AUXILIARY)
    return dict(format='dope-sdv-matched-existing-sample-validation', version=1,
        scope='100 S3 lineages fully accounted; four fixed DOPE profiles versus original author-default/native-selected CTGAN and TVAE. Only existing samples are evaluated. Descriptive validation; one fit seed.',
        **summary, cells=cells, logical_status_counts=dict(Counter(c['status'] for c in cells)),
        sdv_validation_batches=report['batches'],
        sdv_available_configuration_bindings=81, sdv_unavailable_configuration_bindings=319,
        sdv_shared_status_counts=report['logical_status_counts'],
        sdv_unavailable_reason_counts=dict(Counter(c['unavailable_reason'] for c in report['cells']
                                                  if c['status'] != 'ok')),
        native_validation=native_report['cells'], native_objectives=native_report['native_objectives'],
        sdv_native_ledger_class_counts=native_report['new_outcome_class_counts'],
        sdv_scheduling_cutoff_reasons=native_report['new_scheduling_cutoff_reason_counts'],
        method_conclusions_from_scheduling_cutoffs=False, native_selection_changed=False,
        shared_kpi_used_for_selection=False, native_kpis_never_ranked_across_methods=True,
        new_generator_fits_started=0, new_samples_generated=0, sdv_v3_launched=False,
        matched_views=dict(identical_five_projected_file_hashes=100,
                           matched_available_neural_dataset_views=len({j['dataset'] for j in jobs.values()})),
        source_locks=dict(sdv_shared_round=ROUND, sdv_shared_receipts=RECEIPTS,
            sdv_shared_report=RECONCILIATION, shared_auxiliary_inventory=AUXILIARY, native_receipts=NATIVE_RECEIPTS,
            native_report=NATIVE_REPORT, unchanged_matched_reference=REFERENCE),
        runtime=runtime_proof, recorded_metric_repetition_batches=sum(j['metric_replay_required'] for j in jobs.values()),
        recorded_metric_repetition_lineages=len({j['dataset'] for j in jobs.values() if j['metric_replay_required']}),
        learners_refitted_during_publication=False, cost=dict(shared_metric_operation_seconds=seconds,
            shared_coordinator_wall_seconds=end['elapsed_seconds'], native_research=native_report['cost'],
            earlier_dope_gpu_research_and_validation=reference['cost']['previous_dope_gpu_research_and_validation'],
            energy_attributable_to_jobs=None, total_research_spend_equalized=False),
        full_campaign_admitted=False,
        missing_evidence=['five-fit final schedule', 'n/2n/4n/8n final samples',
            'complete privacy attacks', 'projection-only utility cost', 'public-core paired analysis',
            'five-lock full campaign admission', '319 unavailable SDV common sample schedules'])


def tables(report):
    expected = common.aggregate(report['cells'])
    inputs.require(all(report[k] == expected[k] for k in expected), 'matched summary differs')
    output = io.StringIO(); writer = csv.writer(output, lineterminator='\n')
    writer.writerow(['dataset', 'method', 'configuration', 'size_multiplier', 'charged_artifact_bytes',
        'statuses', *[a + '_median_retention' for a in common.AUDITORS]])
    for row in report['summary']:
        writer.writerow([row[k] for k in ('dataset', 'method', 'configuration', 'size_multiplier',
            'charged_artifact_bytes')] + [json.dumps(row['statuses'], sort_keys=True)]
            + [row['utility'][a]['median_retention'] for a in common.AUDITORS])
    lines = ['# Matched S3 neural and DOPE validation', '', report['scope'], '',
        'CTGAN/TVAE maximize their original native SDMetrics regression efficacy KPI. Common retention never selects their configurations; native KPI values are not ranked across methods.', '',
        'All 100 lineages remain in the coverage denominator. SDV has 486 measured cells and 1,914 sample-unavailable cells: 81/400 configuration bindings have complete existing samples. Native research ended with 154 successful new trials, 524 deadline-unstarted trials, one deadline-truncated trial and 25 infrastructure interruptions. Scheduling cutoffs do not establish method failures.', '',
        'Each paired number uses lineages where all three sample seeds are informative for both methods at the stated size. Availability can bias this subset. DOPE profiles are fixed research configurations; no global family or production winner is selected.', '',
        '| DOPE profile | Native baseline | Size | CatBoost paired lineages | DOPE median | Baseline median | Median paired difference |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in report['paired_descriptive']:
        if row['auditor'] == 'catboost' and row['configuration'] == 'native_selected':
            values = [row['dope_profile'], row['method'], str(row['size_multiplier']) + 'n',
                str(row['complete_paired_lineages'])] + ['null' if row[k] is None else f'{row[k]:.6g}'
                for k in ('dope_matched_median', 'baseline_matched_median', 'median_dataset_difference')]
            lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', 'JSON includes default and native-selected results, all three auditors, per-cell utility/privacy controls, native KPIs, charged model+projection bytes, costs and unavailable reasons. These are unconstrained-quality validation comparisons. Official tests remain sealed; MFS-v2/PTF-v1/release-safe L3/superiority stay null.', '']
    return output.getvalue(), '\n'.join(lines)


def main():
    import argparse
    from research.benchmark.publish_s3_matched import schema
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); report = build(); table, markdown = tables(report)
    inputs.require(hashlib.sha256((json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()).hexdigest()
                   == REPORT_SHA256, 'frozen matched report does not reproduce')
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    for suffix, data in {'.json': json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n',
                         '.schema.json': json.dumps(schema(report), sort_keys=True, indent=2) + '\n',
                         '.csv': table, '.md': markdown}.items():
        (args.output_dir / (NAME + suffix)).write_text(data)


if __name__ == '__main__': main()
