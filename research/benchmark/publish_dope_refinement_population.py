"""Prepare the complete matched validation panel; refuse unfinished rounds."""
from collections import Counter
import json
from pathlib import Path
import argparse
import csv
import io
import math

from . import publish_dope_refinement_fits as fits
from . import arf_runtime_guard as guard
from .manifest import digest
from .publish_dope_refinement_discovery import summaries, paired
from .publish_s3_forest import shared_inventory

BASE = fits.BASE
ROOT = BASE / 'dope-s3-refinement-expansion-validation-v2'
ROUND = 'b0198ffc65524d3616d9f0f062e9a6e49345b0eb4db7dfa7f727bd1612e75545'
REFERENCE = '5264b88a40ad5efb21d9b119789caeb13e71e911a17f1d8b57f1d8e8ac31732a'
FIT_PUBLICATION = 'ec330be7103ee9b63440f4e3b1b26c31592695ef2f27988b1fd73ff97a0fd91e'
AUXILIARY = '64e0ada86999b04ee3a2a050817d1d12ed2342604e5d9120f690df6e4fb2e269'
AUX_PATH = BASE / 'dope-s3-refinement-expansion-validation-publication-preparation-v1/batch-auxiliary.lock.json'
NAME = 'dope-target-refinement-population-validation'
SEEDS = (101, 211, 307)
SIZES = (1, 4)


def metric_check(value, worker, size, implementation):
    guard.require(value['implementation_sha256'] == implementation and value['task'] == 'regression'
        and value['gate_profile_complete'] is False and value['mfs_v2'] is None
        and value['rows']['train'] == worker['train_rows']
        and value['rows']['synthetic'] == worker['train_rows'] * size
        and all(type(n) is int and n > 0 for n in value['rows'].values())
        and set(value['utility']) == {'linear', 'catboost', 'mlp'}, 'common metric contract differs')
    null = value['null_loss']
    guard.require(type(null) in (int, float) and math.isfinite(null) and null >= 0, 'invalid null loss')
    for row in value['utility'].values():
        if row.get('status') == 'failed':
            guard.require(set(row) == {'status', 'error_type'} and row['error_type'] in ('ValueError', 'RuntimeError'),
                          'unknown auditor failure')
            continue
        real, synthetic = row['trtr_loss'], row['tstr_loss']
        guard.require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in (real, synthetic)),
                      'invalid auditor losses')
        informative = null - real >= .01 * abs(null)
        guard.require(row['informative'] is informative, 'informative profile differs')
        if informative:
            retention = row['retention']
            guard.require(null > real and type(retention) in (int, float) and math.isfinite(retention)
                and math.isclose(retention, (null - synthetic) / (null - real), abs_tol=1e-12, rel_tol=1e-12),
                'unclipped normalized retention differs')
        else:
            guard.require(row['retention'] is None, 'low signal acquired retention')
    for key, count in [('copy_counts', value['rows']['synthetic']), ('real_vs_real_control_counts', value['rows']['validation'])]:
        row = value[key]
        guard.require(set(row) == {'exact', 'near'} and all(type(x) is int for x in row.values())
            and 0 <= row['exact'] <= row['near'] <= count, 'copy control count differs')


def complete_matrix(cells):
    ids = {c['dataset'] for c in cells}
    keys = [(c['dataset'], c['configuration'], c['size_multiplier'], c['sample_seed']) for c in cells]
    guard.require(len(ids) == 100 and len(keys) == len(set(keys)) == 1200
        and set(keys) == {(d, p, n, s) for d in ids for p in fits.PROFILES for n in SIZES for s in SEEDS},
        'complete 1200-cell common matrix required')
    for c in cells:
        guard.require(type(c['fit_seed']) is int and c['fit_seed'] == 11
            and type(c['sample_seed']) is int and type(c['size_multiplier']) is int
            and c['method'] == 'DOPE' and c['counts_as_dope_win'] is False
            and c['projection_bytes_included'] is True
            and all(c[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')),
            'sample identity or gated claim differs')
        if c['status'] == 'ok':
            guard.require(c['utility'] is not None and type(c['charged_artifact_bytes']) is int
                and 0 < c['charged_artifact_bytes'] <= 10240, 'success without utility or byte eligibility')
        else:
            guard.require(c['utility'] is None and c['unavailable_reason'] is not None,
                          'missing sample hid its reason')
    return sorted(ids)


def reference_cells(ids):
    path = Path(fits.__file__).with_name('results') / 'density-matched-population-validation.json'
    reference = guard.bound(path, REFERENCE, path.parent)
    guard.require(len(reference['cells']) == 6000
        and {c['dataset'] for c in reference['cells']} == set(ids), 'complete matched reference required')
    return reference


def sampling_inventories(report, refs, auxiliary):
    """Reject additions to every measured batch before decoding its metrics."""
    batches = {}
    for batch in report.get('batches', []):
        root = fits.safe(ROOT / 'attempts' / batch['physical_job_sha256'] / 'attempt-0001')
        batches[root] = batch['receipt_sha256']
    for row in report['cells']:
        if row['sample_evidence']:
            root = fits.safe(Path(row['sample_evidence']['metric_path']).parent)
            pin = row['validation_receipt_sha256']
            guard.require(root not in batches or batches[root] == pin, 'batch receipt identities differ')
            batches[root] = pin
    guard.require(set(auxiliary['batches']) == {str(p) for p in batches}, 'closed batch inventory coverage differs')
    for root, pin in batches.items():
        path = str(root / 'receipt.json')
        guard.require(refs[path] == pin, 'sample receipt absent from frozen anchor')
        receipt = fits.bound(path, pin)
        files = {**receipt['evidence_files'], 'receipt.json': pin}
        inv = auxiliary['batches'][str(root)]
        guard.inventory(root, inv['files'], BASE, inv['directories'])
        guard.require({p.name for p in root.iterdir() if p.is_file()} == set(files), 'batch files differ')
        guard.require(all(inv['files'][str(root / name)]['sha256'] == h for name, h in files.items()),
                      'closed batch output identity differs')
        guard.require(all(refs[str(root / name)] == h for name, h in files.items()),
                      'batch inventory absent from frozen anchor')


def build(receipt_pin, report_pin):
    anchor = fits.bound(ROOT / 'receipt-lock-v1.json', receipt_pin)
    guard.require(anchor['round_sha256'] == ROUND and anchor['fit_receipt_lock_sha256'] == fits.RECEIPTS
        and anchor['reconciliation_sha256'] == report_pin, 'closed validation anchor differs')
    refs = {}
    for path, pin in anchor['refs'].items(): fits.checked(path, pin, refs)
    lock = fits.bound(ROOT / 'round.lock.json', ROUND)
    for path, pin in list(refs.items()):
        if Path(path).name == 'round.lock.json':
            parent = fits.bound(path, pin)
            if 'source_files' in parent:
                source = Path(path).parent / 'source'
                guard.require(all(Path(p).parent == source for p in parent['source_files']), 'source root differs')
                fits.flat(source, {Path(p).name: h for p, h in parent['source_files'].items()})
    shared_inventory(lock, {})
    actual = fits.bound(ROOT / 'coordinator-exit.json', refs[str(ROOT / 'coordinator-exit.json')])
    done = fits.bound(ROOT / 'completion.json', refs[str(ROOT / 'completion.json')])
    launched = fits.bound(ROOT / 'supervisor-launch.json', refs[str(ROOT / 'supervisor-launch.json')])
    guard.require(type(actual['exit_code']) is int and actual['exit_code'] == 0
        and actual['round_sha256'] == done['round_sha256'] == launched['round_sha256'] == ROUND
        and done['jobs'] == 112 and done['logical_cells'] == 1200
        and all(not Path('/proc', str(launched[k])).exists() for k in ('pid', 'supervisor_pid')),
        'actual complete validation closure required')
    report = fits.bound(ROOT / 'reconciliation-v1.json', report_pin)
    guard.require(report['complete_matrix'] is True and report['logical_sample_cells'] == 1200
        and report['immutable_validation_cells_reused'] == 144 and report['physical_batches_closed'] == 112
        and report['official_tests_opened'] is False and report['native_selection_changed'] is False
        and report['global_family_selected'] is False
        and all(report[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')),
        'closed validation scope differs')
    auxiliary = fits.bound(AUX_PATH, AUXILIARY)
    guard.require(auxiliary['round_sha256'] == ROUND
        and auxiliary['validation_receipt_lock_sha256'] == receipt_pin
        and auxiliary['reconciliation_sha256'] == report_pin
        and auxiliary['inventory_pinned_before_publisher_metric_decoding'] is True
        and auxiliary['extra_directory_outputs_used_for_metrics_or_selection'] is False,
        'closed batch output anchor differs')
    ledger = fits.build()
    ledger_path = Path(fits.__file__).with_name('results') / (fits.NAME + '.json')
    guard.require(digest(guard.bound(ledger_path, FIT_PUBLICATION, ledger_path.parent)) == digest(ledger),
                  'committed fit ledger differs')
    sampling_inventories(report, refs, auxiliary)
    fit_rows = {r['job_sha256']: r for r in fits.bound(fits.ROOT / 'reconciliation-v1.json', fits.REPORT)['fit_cells']}
    cells = []
    for row in report['cells']:
        sample = row.get('sample_evidence'); metric = None
        if sample:
            guard.require(refs[sample['metric_path']] == sample['metric_sha256'], 'metric absent from immutable anchor')
            metric = fits.bound(sample['metric_path'], sample['metric_sha256'])
            guard.require(metric['implementation_sha256'] == lock['metric_source_sha256']
                and metric['gate_profile_complete'] is False and metric['mfs_v2'] is None,
                'metric implementation or claim differs')
            metric_check(metric, fit_rows[row['fit_job_sha256']]['original_fit_job']['worker'], row['size_multiplier'], lock['metric_source_sha256'])
        cells.append(dict(dataset=row['dataset'], method='DOPE', configuration=row['profile'], fit_seed=row['fit_seed'],
            sample_seed=row['sample_seed'], size_multiplier=row['size_multiplier'], status=row['status'],
            unavailable_reason=(row['unavailable_reason'] or row['status']) if row['status'] != 'ok' else None,
            charged_artifact_bytes=row['charged_artifact_bytes'],
            projection_bytes_included=True, fit_job_sha256=row['fit_job_sha256'], fit_receipt_sha256=row['fit_receipt_sha256'],
            validation_receipt_sha256=row['validation_receipt_sha256'], sample_evidence=sample,
            immutable_validation_reused=row['immutable_validation_reused'],
            utility=metric['utility'] if metric else None, null_loss=metric['null_loss'] if metric else None,
            copy_counts=metric['copy_counts'] if metric else None,
            real_vs_real_control_counts=metric['real_vs_real_control_counts'] if metric else None,
            marginal_ks_mean=metric['marginal_ks_mean'] if metric else None,
            pair_correlation_fidelity=metric['pair_correlation_fidelity'] if metric else None,
            c2st_auc=metric['c2st_auc'] if metric else None,
            mfs_v2=None, ptf_v1=None, release_safe=None, superiority=None, counts_as_dope_win=False))
    ids = complete_matrix(cells)
    guard.require(dict(Counter(c['status'] for c in cells)) == report['logical_status_counts'], 'sample status accounting differs')
    reference_path = Path(fits.__file__).with_name('results') / 'density-matched-population-validation.json'
    reference = reference_cells(ids)
    comparison = reference['cells']
    guard.require(len(comparison) == 6000 and {c['dataset'] for c in comparison} == set(ids), 'complete matched reference required')
    density = fits.bound(BASE / 'density-s3-population-shared-validation-v1/round.lock.json', reference['source_locks']['density_round'])
    workers = {j['dataset']: j['worker'] for j in density['jobs']}
    for c in lock['logical_cells']:
        fit_row = fit_rows[c['fit_job_sha256']]
        guard.require(digest(fit_row['original_fit_job']['worker']) == digest(workers[c['dataset']]), 'matched worker views differ')
    groups, panels = summaries(cells + comparison)
    for path, pin in list(refs.items()): fits.checked(path, pin, refs)
    return dict(format='dope-complete-refinement-population-common-validation', version=1,
        source_sha256=fits.sha256(Path(__file__)), s3_data_lock_sha256=fits.DATA,
        scope='100 bounded official-training-derived S3 lineages; declared two-profile research, fit11/three sample seeds/n+4n. Baselines selected only by frozen native objectives.',
        datasets=ids, logical_validation_cells=1200, physical_validation_batches=112,
        logical_status_counts=report['logical_status_counts'], physical_status_counts=report['physical_status_counts'],
        immutable_validation_cells_reused=144, cells=cells, matched_reference=dict(path=str(reference_path), sha256=REFERENCE, logical_cells=6000),
        sample_replay_controls=dict(exact=sum(bool(c['sample_evidence']) and c['sample_evidence']['sample_replay'] == 'exact' for c in cells),
            not_repeated=sum(bool(c['sample_evidence']) and c['sample_evidence']['sample_replay'] == 'not_repeated' for c in cells)),
        metric_replay_controls=dict(exact=sum(bool(c['sample_evidence']) and c['sample_evidence']['metric_replay'] == 'exact' for c in cells),
            not_repeated=sum(bool(c['sample_evidence']) and c['sample_evidence']['metric_replay'] == 'not_repeated' for c in cells)),
        lineage_groups=groups, summary=panels, paired_descriptive=paired(groups, fits.PROFILES),
        fit_ledger_reference=dict(path=str(ledger_path), sha256=FIT_PUBLICATION),
        fit_cost=ledger['cost'], validation_operation_seconds=report['new_operation_seconds'],
        validation_scheduler_wall_seconds=report['coordinator_wall_seconds'],
        source_locks=dict(round=ROUND, receipts=receipt_pin, reconciliation=report_pin, density_reference=REFERENCE,
                          batch_auxiliary=AUXILIARY),
        global_configuration_tuned_or_selected=False, final_five_fit_coverage_complete=False,
        privacy_attack_coverage_complete=False, system_dynamic_library_closure_certified=False,
        **fits.GATES)


def validate_report(report):
    ids = complete_matrix(report['cells'])
    guard.require(ids == report['datasets']
        and all(report[k] is v for k, v in fits.GATES.items())
        and report['source_sha256'] == fits.sha256(Path(__file__))
        and report['source_locks']['round'] == ROUND and report['s3_data_lock_sha256'] == fits.DATA
        and report['fit_ledger_reference']['sha256'] == FIT_PUBLICATION
        and report['matched_reference']['sha256'] == REFERENCE
        and report['logical_validation_cells'] == 1200
        and report['physical_validation_batches'] == 112
        and report['immutable_validation_cells_reused'] == 144
        and all(report[k] is False for k in ('global_configuration_tuned_or_selected',
            'final_five_fit_coverage_complete', 'privacy_attack_coverage_complete',
            'system_dynamic_library_closure_certified')),
        'publication shape, provenance or claims differ')
    for pin in report['source_locks'].values(): guard.digest(pin)
    guard.require(report['source_locks']['density_reference'] == REFERENCE
        and report['source_locks']['batch_auxiliary'] == AUXILIARY
        and dict(Counter(c['status'] for c in report['cells'])) == report['logical_status_counts'],
        'publication status accounting differs')
    groups, panels = summaries(report['cells'] + reference_cells(ids)['cells'])
    guard.require(digest(report['lineage_groups']) == digest(groups)
        and digest(report['summary']) == digest(panels)
        and digest(report['paired_descriptive']) == digest(paired(groups, fits.PROFILES)),
        'descriptive aggregate replay differs')


def tables(report):
    validate_report(report)
    out = io.StringIO(); writer = csv.writer(out, lineterminator='\n')
    fields = ('dataset', 'configuration', 'fit_seed', 'sample_seed', 'size_multiplier', 'status',
              'unavailable_reason', 'charged_artifact_bytes', 'immutable_validation_reused')
    writer.writerow([*fields, 'linear_retention', 'catboost_retention', 'mlp_retention'])
    for row in report['cells']:
        writer.writerow([*[row[k] for k in fields],
            *[(row['utility'] or {}).get(a, {}).get('retention') for a in ('linear', 'catboost', 'mlp')]])
    lines = ['# DOPE population refinement: complete matched validation', '',
        '100 bounded S3 official-training-derived lineages; one fit seed, three sample seeds at n/4n. Descriptive research comparison; official tests sealed. MFS-v2/PTF-v1/release/superiority null.', '',
        f"All 1,200 declared refinement cells accounted: {report['logical_status_counts']}. There are 112 new physical batches and 144 immutable prior validation cells. Infrastructure and byte-cap reasons remain visible; unavailable cells supply no DOPE win.", '',
        'Native baseline configurations retain their frozen held-out density selection. Native KPI units are never ranked across methods. Each pair below uses identical worker views and complete informative three-seed groups; sample seeds reduce within lineage before dataset medians.', '',
        '| DOPE profile | Reference | 4n CatBoost paired N | DOPE median | Reference median | Median paired difference |',
        '|---|---|---:|---:|---:|---:|']
    def number(value): return 'null' if value is None else f'{value:.6f}'
    for row in report['paired_descriptive']:
        if row['size_multiplier'] == 4 and row['auditor'] == 'catboost':
            label = row['reference_method'] + '/' + row['reference_configuration']
            if row['reference_method'] in ('Chow-Liu', 'independent_marginals'): label += ' (study implementation)'
            lines.append(f"| {row['configuration']} | {label} | {row['paired_complete_informative_lineages']} | {number(row['dope_median_retention'])} | {number(row['reference_median_retention'])} | {number(row['median_paired_difference'])} |")
    lines += ['', 'Pair cohorts can differ; medians from different rows are not a common-cohort ranking. Negative and above-one retentions are retained. No superiority test or global family selection is claimed.', '',
        f"Exact sampler controls: {report['sample_replay_controls']}; exact metric controls: {report['metric_replay_controls']}. Each count covers measured cells only; cells marked not_repeated were measured once.", '',
        f"Validation operations {report['validation_operation_seconds']:.6f}s; validation scheduler wall {report['validation_scheduler_wall_seconds']:.6f}s. Bounded refinement fits: {report['fit_cost']['cumulative_bounded_refinement_operation_seconds']:.6f}s including prior discovery/confirmation. Earlier architecture research remains separate.", '',
        'JSON retains all auditors, fidelity, copy/near-match and real-vs-real controls, bytes, immutable receipt/metric identities and failed cells. The referenced committed density panel supplies all 6,000 unchanged per-cell reference outcomes and native KPIs. Five-fit coverage, complete privacy/profile evidence, final locks and public-core paper analysis remain unfinished.', '']
    return out.getvalue(), '\n'.join(lines)


def write_outputs(report, root):
    root.mkdir(parents=True, exist_ok=True); csv_text, markdown = tables(report)
    path = root / (NAME + '.json'); path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    path.with_suffix('.schema.json').write_text(json.dumps(fits.schema(report), sort_keys=True, indent=2) + '\n')
    path.with_suffix('.csv').write_text(csv_text); path.with_suffix('.md').write_text(markdown)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt-lock-sha256'); parser.add_argument('--reconciliation-sha256')
    parser.add_argument('--from-json', type=Path); parser.add_argument('--publication-sha256')
    parser.add_argument('--results-dir', type=Path, default=Path(__file__).with_name('results'))
    args = parser.parse_args()
    if args.from_json:
        report = guard.bound(args.from_json, args.publication_sha256, args.from_json.parent)
        validate_report(report)
    else:
        guard.digest(args.receipt_lock_sha256); guard.digest(args.reconciliation_sha256)
        report = build(args.receipt_lock_sha256, args.reconciliation_sha256)
    write_outputs(report, args.results_dir)
    print(args.results_dir / (NAME + '.json'))


if __name__ == '__main__': main()
