"""Reproduce first12 public leaves in memory; --check writes no scratch files."""
import argparse
import csv
import hashlib
import html
import io
import json
import math
from pathlib import Path

PUBLIC_SHA256 = '8e039b1d57298f10bf7c05c2f3a8b14481a60ed663c1063ed30c273854f3bdfe'
PUBLIC_BYTES = 77406
A952 = 'a952062c0f83805f6442a440a5ae15293a843349d423eaa106a9c56a20d14c0f'
METRICS = [('Pearson difference', 'fidelity', 'pairwise_pearson_difference'),
           ('Category TV', 'fidelity', 'categorical_contingency_tv'),
           ('Mixed dependence', 'fidelity', 'mixed_type_dependence'),
           ('Alpha/beta', 'alpha_beta', None), ('C2ST CatBoost', 'detection', 'catboost'),
           ('C2ST logistic', 'detection', 'logistic'), ('NNDR', 'privacy', 'nndr_fit'),
           ('Distance MIA', 'privacy', 'distance_mia'), ('DOMIAS KDE', 'privacy', 'domias_kde')]


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def pairs(items):
    value = {}
    for key, item in items:
        require(key not in value, 'duplicate_json_key')
        value[key] = item
    return value


def checked_public(path):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'public_input_alias')
    raw = path.read_bytes()
    require(len(raw) == PUBLIC_BYTES and hashlib.sha256(raw).hexdigest() == PUBLIC_SHA256,
            'public_input_receipt_changed')
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: require(False, 'nonfinite_public_json'))


def check_contract(value):
    require(value['format'] == 'common-A952-completed-validation-batch'
            and type(value['version']) is int and value['version'] == 1
            and value['evaluator_sha256'] == A952, 'public_source_identity_changed')
    require(value['batch_complete'] is True and value['cohort_complete'] is False
            and value['actual_phase_status'] == 'stopped'
            and value['frozen_queue_job_count'] == 5724
            and value['positive_frozen_queue_cell_count'] == 16
            and value['requested_batch_job_count'] == value['batch_positive_cell_count'] == 12
            and value['batch_pending_job_sha256s'] == [], 'public_coverage_changed')
    require(value['official_tests_opened'] is False and value['empirical_only'] is True
            and value['formal_dp'] is False and value['hipaa_deidentification'] is False
            and value['historical_ROOT_1ae_results_pooled'] is False
            and value['historical_flat_report_pooled'] is False, 'public_scope_changed')
    for key in ('mfs_v2', 'ptf_v1', 'release_safe_l3', 'superiority', 'elapsed_campaign_wall_seconds'):
        require(value[key] is None, 'gated_or_unmeasured_claim')
    cells = value['cells']
    require(len(cells) == 12 and len({c['physical_job_sha256'] for c in cells}) == 12,
            'physical_cell_repeated')
    require(sum(len(c['logical_aliases']) for c in cells) == 14, 'logical_alias_count_changed')
    for cell in cells:
        core = cell['diagnostics']; cost = cell['cost']
        for key in ('mfs_v2', 'ptf_v1', 'release_safe_l3', 'superiority'):
            require(core[key] is None, 'gated_or_unmeasured_claim')
        require(core['official_tests_opened'] is False and core['privacy']['empirical_only'] is True
                and core['privacy']['formal_dp'] is False and core['privacy']['hipaa_deidentification'] is False,
                'public_scope_changed')
        require(cost['core_seconds'] is None and cost['energy_joules'] is None
                and type(cost['GPU_queries']) is int and cost['GPU_queries'] == 0,
                'gated_or_unmeasured_claim')
        for key in ('operation_wall_seconds', 'evaluator_worker_seconds'):
            require(type(cost[key]) in (int, float) and math.isfinite(cost[key]) and 0 <= cost[key] < 600,
                    'public_clock_invalid')
    require(value['closed_batch_operation_wall_seconds_sum'] ==
            sum(c['cost']['operation_wall_seconds'] for c in cells), 'physical_cost_sum_changed')


def metric(cell, key, sub):
    return cell['diagnostics'][key] if sub is None else cell['diagnostics'][key][sub]


def scalar(cell, key, sub, field=None):
    row = metric(cell, key, sub)
    if row['status'] != 'ok':
        return 'unavailable:' + row['reason']
    value = row['value']; value = value[field] if field else value
    return format(value, '.6g') if isinstance(value, (int, float)) else str(value)


def render(value):
    check_contract(value)
    headers = ['physical_job_sha256', 'logical_aliases', 'pearson_difference_mean', 'alpha_precision',
               'beta_recall', 'C2ST_catboost_mean_fold_AUC', 'C2ST_logistic_mean_fold_AUC',
               'distance_MIA_AUC', 'DOMIAS_KDE_AUC', 'operation_wall_seconds', 'evaluator_worker_seconds',
               'peak_summed_process_resident_bytes']
    rows = []
    for cell in value['cells']:
        aliases = '; '.join(a['method'] + '/' + str(a['configuration']) + '/' + a['dataset'] + '/n' +
                           str(a['size_multiplier']) + '/sample' + str(a['sample_seed'])
                           for a in cell['logical_aliases'])
        rows.append([cell['physical_job_sha256'], aliases,
            scalar(cell, 'fidelity', 'pairwise_pearson_difference', 'mean'),
            scalar(cell, 'alpha_beta', None, 'alpha_precision'), scalar(cell, 'alpha_beta', None, 'beta_recall'),
            scalar(cell, 'detection', 'catboost'), scalar(cell, 'detection', 'logistic'),
            scalar(cell, 'privacy', 'distance_mia'), scalar(cell, 'privacy', 'domias_kde'),
            repr(cell['cost']['operation_wall_seconds']), repr(cell['cost']['evaluator_worker_seconds']),
            str(cell['cost']['peak_summed_process_resident_bytes'])])
    stream = io.StringIO(newline=''); writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(headers); writer.writerows(rows)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="820" viewBox="0 0 1200 820">',
        '<rect width="1200" height="820" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#152238} .small{font-size:12px} .heading{font-size:20px;font-weight:bold} .mono{font-family:monospace;font-size:12px}</style>',
        '<text x="28" y="38" class="heading">A952 frozen first-12 batch: diagnostic availability</text>',
        '<text x="28" y="64" class="small">12 authenticated physical cells / 14 logical aliases. Cohort stopped after 16/5724 successes; no ranking or gated scores.</text>']
    left, top, cw, rh = 310, 185, 90, 38
    for j, (title, key, sub) in enumerate(METRICS):
        xx = left + j*cw + cw/2
        parts.append(f'<text x="{xx}" y="{top-18}" class="small" text-anchor="end" transform="rotate(-40 {xx} {top-18})">{html.escape(title)}</text>')
    for i, cell in enumerate(value['cells']):
        y = top + i*rh
        label = cell['physical_job_sha256'][:12] + ' ' + ','.join(sorted({a['method'] for a in cell['logical_aliases']}))
        parts.append(f'<text x="28" y="{y+24}" class="mono">{html.escape(label)}</text>')
        for j, (title, key, sub) in enumerate(METRICS):
            ok = metric(cell, key, sub)['status'] == 'ok'; xx = left + j*cw
            parts.append(f'<rect x="{xx}" y="{y}" width="{cw-6}" height="{rh-6}" rx="3" fill="{"#dceeea" if ok else "#eeeeee"}"/>')
            parts.append(f'<text x="{xx+(cw-6)/2}" y="{y+21}" class="small" text-anchor="middle">{"measured" if ok else "unavailable"}</text>')
    parts += ['<text x="28" y="688" class="small">Grey marks preserve typed unavailable evidence; they are never scored as zero.</text>',
        '<text x="28" y="712" class="small">TRAIN-derived numeric kinds: binary representation TV and continuous Pearson/KS; author semantic categories remain unavailable.</text>',
        '<text x="28" y="736" class="small">Privacy diagnostics are empirical only: no formal DP, HIPAA de-identification or release-safe claim.</text>',
        '<text x="28" y="760" class="small">Exact values and per-cell costs are retained in first12-metrics-and-costs.csv and the schema-validated public JSON.</text>', '</svg>']
    readme = ['# A952 frozen first-12 common-validation batch', '',
        '**Scope:** twelve completed physical evaluations, selected by canonical job digest before START. They have fourteen logical aliases. This is a deterministic first batch, not a representative sample or method comparison. The full 5724-cell cohort stopped after sixteen positives because the declared external RAM headroom was missing; 5708 cells remain uncompleted.', '',
        'The JSON authenticates each frozen job, source/runtime/config/request, actual child exit0 and empty owned family through the existing queue adapter. Fidelity, grouped detection and empirical privacy diagnostics retain their unavailable reasons. Official TEST was not opened. MFS-v2, PTF-v1, release-safe and superiority remain null.', '',
        '**Representation:** TRAIN-only binary/continuous numeric kinds, with the target included. These are not author semantic column types. Pearson difference is the measured source mean; C2ST entries are mean fold AUCs. Alpha/beta and DOMIAS are the pinned A952 research variants. No metric is recomputed here. CSV display uses six significant digits; exact values remain in JSON.', '',
        '**Cost:** ' + repr(value['closed_batch_operation_wall_seconds_sum']) + ' seconds is the sum of twelve unique actual evaluator operation clocks, not campaign wall time. Logical aliases share those costs. Fit/sample time, preallocation waiting, other four completed cells, energy and core-seconds are excluded or null.', '',
        '## Per-cell measured diagnostics', '',
        '| Physical job | Alias methods | Pearson difference mean | Alpha / beta | C2ST CatBoost / logistic | Distance MIA / DOMIAS | Actual operation seconds |',
        '|---|---|---:|---|---|---|---:|']
    for cell, row in zip(value['cells'], rows):
        methods = ', '.join(sorted({a['method'] for a in cell['logical_aliases']}))
        readme.append('| ' + cell['physical_job_sha256'][:12] + ' | ' + methods + ' | ' + row[2] + ' | ' +
                      row[3] + ' / ' + row[4] + ' | ' + row[5] + ' / ' + row[6] + ' | ' + row[7] + ' / ' +
                      row[8] + ' | ' + format(cell['cost']['operation_wall_seconds'], '.6f') + ' |')
    readme += ['', '## Files', '',
        '- `first12.completed.json`: exact measured cells, aliases, typed unavailable diagnostics, evidence digests and costs.',
        '- `first12-metrics-and-costs.csv`: readable value/cost table with full frozen job and opaque lineage identities.',
        '- `first12-diagnostic-availability.svg`: availability only; no rankings or invented zero values.',
        '- `first12.completed.schema.json`: strict publication schema embedding the existing evaluator-core definitions.',
        '- `render.py --check`: reproduce and compare CSV, SVG and this README in memory; no private receipts or scratch files required.', '',
        'Historical ROOT222/1ae and the earlier flattened1326 report are kept separate. This slice changes no evaluation, selection, source/runtime locks, queues or certification gates.']
    return {'first12-metrics-and-costs.csv': stream.getvalue().encode(),
            'first12-diagnostic-availability.svg': ('\n'.join(parts) + '\n').encode(),
            'README.md': ('\n'.join(readme) + '\n').encode()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true')
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(); directory = args.directory
    expected = render(checked_public(directory / 'first12.completed.json'))
    for name, raw in expected.items():
        path = directory / name
        require(not any(p.is_symlink() for p in (path, *path.parents)), 'public_leaf_alias')
        if args.check:
            require(path.read_bytes() == raw, 'public_leaf_not_reproducible:' + name)
        else:
            require(not path.exists(), 'public_leaf_exists:' + name)
            with path.open('xb') as stream:
                stream.write(raw)
    print(json.dumps({'status': 'PASS', 'checked': args.check, 'leaf_files': sorted(expected)}, sort_keys=True))


if __name__ == '__main__':
    main()
