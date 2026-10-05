"""Reconcile complete matched density/DOPE validation from frozen evidence."""
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

from research.benchmark import density_publication_inputs as inputs
from research.benchmark import density_metric_replay as metrics
from research.benchmark import density_custody_replay as costs
from research.benchmark import density_native_replay as native
from research.benchmark import publish_dope_population_validation as dope
from research.benchmark import publish_dope_population_fits as fits
from . import density_declared_runtime_replay as runtime
from . import density_matched_aggregate as common

DOPE_RECEIPT = 'ecd88832cb9c8c0018bb55de67ded9c3eba0936eba541e799ded0def34e173bc'
DOPE_REPORT = '51e48fb3227588c0a8b1e49a64b94c1dd8b391063f94411aa33fc6bdac96004a'
NAME = 'density-matched-population-validation'
RECEIPT_SHA256 = 'b2f5964cf40d047d63dc5698c42c2370d945384f5f9567d6809d0959f4f4d81b'
RECONCILIATION_SHA256 = 'b72a1a7cc2c9298e9eb0cd1c774a995561a1ea7e51f39e269c4789532096cb09'
REPORT_SHA256 = '5264b88a40ad5efb21d9b119789caeb13e71e911a17f1d8b57f1d8e8ac31732a'


def serialized(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def regular_bytes(path):
    path = Path(path)
    inputs.require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
                   'publication file or parent must not be a link')
    return path.read_bytes()


def load_report(path, expected_sha256):
    """Check the external versioned digest before parsing committed research JSON."""
    inputs.digest_string(expected_sha256)
    inputs.require(expected_sha256 == REPORT_SHA256, 'frozen matched report required')
    data = regular_bytes(path)
    inputs.require(hashlib.sha256(data).hexdigest() == expected_sha256, 'matched report changed')
    report = json.loads(data)
    tables(report)
    return report


def matched_views(density_lock, dope_fit_lock):
    views = []
    for lock in (density_lock, dope_fit_lock):
        by_dataset = {}
        for j in lock['jobs']:
            w = j['worker']
            if j['dataset'] in by_dataset:
                inputs.require(inputs.digest_identity(by_dataset[j['dataset']])
                    == inputs.digest_identity(w), 'worker changed within method matrix')
            by_dataset[j['dataset']] = w
        views.append(by_dataset)
    inputs.require(set(views[0]) == set(views[1]) and len(views[0]) == 100,
                   'matched dataset inventory differs')
    for dataset in views[0]:
        x, y = views[0][dataset], views[1][dataset]
        for field in ('files', 'raw_training_derived_split_hashes', 'train_rows',
                      'projected_features'):
            inputs.require(inputs.digest_identity(x[field]) == inputs.digest_identity(y[field]),
                           'matched training or validation view differs')
        inputs.require(set(x['files']) == {'train.csv', 'validation.csv', 'projection.json',
                                           'row-group-assignments.json', 'worker-manifest.json'},
                       'complete numeric projection binding required')
    return dict(datasets=100, identical_five_projected_file_hashes=100,
                identical_training_derived_splits=True, official_tests_opened=False,
                raw_rows_parsed=False)


def build(receipt_sha, report_sha):
    closed = inputs.load_closed_inputs(receipt_sha, report_sha)
    measured = metrics.replay_closed_metrics(receipt_sha, report_sha)
    cost = costs.replay_closed_costs(receipt_sha, report_sha)
    native_proof = native.replay_native_metadata()
    runtime_proof = runtime.replay(closed['lock'])
    dope_fit_lock = inputs.bound_json(fits.ROOT / 'round.lock.json', fits.ROUND_SHA)
    views = matched_views(closed['lock'], dope_fit_lock)
    previous = dope.build(DOPE_RECEIPT, DOPE_REPORT)
    dope_lock = inputs.bound_json(dope.ROOT / 'round.lock.json', dope.ROUND_SHA)
    dope_jobs = {inputs.digest_identity(j): j for j in dope_lock['jobs']}
    evaluator = inputs.bound_json(metrics.RUNTIME_PATH, metrics.RUNTIME_SHA256)
    logical = {(m['physical_job_sha256'], m['size_multiplier'], m['sample_seed']): m['metric']
               for m in measured['logical_metrics']}
    cells = []
    for c in closed['report']['cells']:
        metric = logical[c['physical_job_sha256'], c['size_multiplier'], c['sample_seed']]
        row = dict(c, charged_artifact_bytes=c['artifact_bytes'],
                   utility=metric['utility'] if metric else None,
                   copy_counts=metric['copy_counts'] if metric else None,
                   real_vs_real_control_counts=metric['real_vs_real_control_counts'] if metric else None,
                   null_loss=metric['null_loss'] if metric else None,
                   marginal_ks_mean=metric['marginal_ks_mean'] if metric else None,
                   pair_correlation_fidelity=metric['pair_correlation_fidelity'] if metric else None,
                   c2st_auc=metric['c2st_auc'] if metric else None,
                   mfs_v2=None, ptf_v1=None, release_safe_l3=None,
                   paired_superiority=None, counts_as_dope_win=False)
        cells.append(row)
    for c in previous['cells']:
        row = dict(c)
        row['configuration'] = row.pop('profile')
        if row['status'] == 'ok':
            reference = row['metric_receipt']
            metric = inputs.bound_json(reference['metric_path'], reference['metric_sha256'])
            metrics.verify_metric(metric, dope_jobs[row['physical_job_sha256']],
                                  row['size_multiplier'], evaluator)
            inputs.require(inputs.digest_identity(row['utility']) == inputs.digest_identity(metric['utility']),
                           'DOPE common utility binding differs')
        cells.append(row)
    summary = common.aggregate(cells)
    native_rows = []
    for c in closed['native_report']['default_native_cells']:
        native_rows.append({k: c[k] for k in ('dataset', 'method', 'configuration',
            'native_objective', 'native_validation_kpi', 'selected_trial_index',
            'implementation_kind', 'artifact_bytes', 'fit_attempt_receipt_sha256',
            'selection_receipt_sha256')})
    anchor = inputs.bound_json(inputs.ROOT / 'receipt-lock-v1.json', receipt_sha)
    inputs.verify_refs(anchor['refs'])
    return dict(format='dope-density-matched-population-validation-panel', version=1,
        scope='100 matched S3 lineages; fixed four DOPE GPU research profiles versus default/native-selected density methods; descriptive validation only',
        **summary, cells=cells, logical_status_counts=dict(Counter(c['status'] for c in cells)),
        native_validation=native_rows, native_kpis_never_ranked_across_methods=True,
        baseline_selected_by_common_utility=False, matched_views=views,
        source_locks=dict(density_round=inputs.ROUND, density_receipts=receipt_sha,
            density_report=report_sha, native_receipts=inputs.NATIVE, native_report=inputs.NATIVE_REPORT,
            dope_round=dope.ROUND_SHA, dope_receipts=DOPE_RECEIPT, dope_report=DOPE_REPORT),
        cost=dict(density=cost, previous_dope_gpu_research_and_validation=previous['cost'],
                  prior_fit_cost_charged_once=True, total_research_spend_equalized=False),
        native_metadata_replay=native_proof,
        runtime={k: v for k, v in runtime_proof.items() if k != 'scratch_runtime_refs'},
        missing_evidence=['five-fit final schedule', 'n/2n/4n/8n final samples', 'complete privacy attacks',
                          'projection-only utility cost', 'public-core paired analysis', 'five-lock full campaign admission'],
        full_campaign_admitted=False, publication_admitted=False, sdv_v3_launched=False,
        baseline_native_selection_changed=False)


def tables(report):
    expected = common.aggregate(report['cells'])
    inputs.require(all(report[k] == expected[k] for k in expected), 'matched summary differs')
    output = io.StringIO()
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(['dataset', 'method', 'configuration', 'size_multiplier', 'charged_artifact_bytes',
                     'artifact_within_l3_cap', 'statuses', *[a + '_median_retention' for a in common.AUDITORS]])
    for g in report['summary']:
        writer.writerow([g[k] for k in ('dataset', 'method', 'configuration', 'size_multiplier',
                                       'charged_artifact_bytes', 'artifact_within_l3_cap')]
                        + [json.dumps(g['statuses'], sort_keys=True)]
                        + [g['utility'][a]['median_retention'] for a in common.AUDITORS])
    lines = ['# Matched S3 density and DOPE validation', '', report['scope'], '',
        'Baselines use their frozen native likelihood objectives. Common retention never selects baseline configurations. Native KPI values are reported per method without cross-method ranking.', '',
        'All100 dataset views match the same five training-derived numeric files. One fit seed and three sample seeds at n/4n; every failure and artifact charge remains visible. Comparisons below describe unconstrained quality. Release-safe L3/MFS-v2/PTF-v1 and superiority stay null.', '',
        '| Method | Configuration | Size | CatBoost informative /100 | Median retention | Linear informative /100 | Median retention | MLP informative /100 | Median retention |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for p in report['configuration_panels']:
        values = [p['method'], p['configuration'], str(p['size_multiplier'])]
        for a in common.AUDITORS:
            u = p['utility'][a]; value = u['median_of_lineage_sample_medians']
            values += [str(u['complete_informative_lineages']), 'null' if value is None else f'{value:.6g}']
        lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', 'Paired descriptive JSON uses common complete informative lineages and the median of dataset differences. It makes no significance, win or selected-family claim. Earlier DOPE architecture research spend and native density tuning costs remain separately reported.', '']
    return output.getvalue(), '\n'.join(lines)


def main():
    import argparse
    from .publish_s3_matched import schema
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--reconciliation-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    inputs.require(args.receipt_sha256 == RECEIPT_SHA256
                   and args.reconciliation_sha256 == RECONCILIATION_SHA256,
                   'frozen matched custody required')
    report = build(args.receipt_sha256, args.reconciliation_sha256)
    data = serialized(report)
    inputs.require(hashlib.sha256(data).hexdigest() == REPORT_SHA256,
                   'matched report does not reproduce')
    table, markdown = tables(report)
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    outputs = {'.json': data, '.schema.json': serialized(schema(report)),
               '.csv': table.encode(), '.md': markdown.encode()}
    for suffix, blob in outputs.items():
        with (args.output_dir / (NAME + suffix)).open('xb') as stream:
            stream.write(blob)
    print(json.dumps({'datasets': 100, 'cells': 6000, 'production_certified': False}))


if __name__ == '__main__':
    main()
