#!/usr/bin/env python3
"""Materialize pinned historical inputs and regenerate the appendix tables.

Static bibliography entries, license extracts, and the historical provenance
snapshot are source inputs. This step never scans a live benchmark filesystem.
"""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / 'docs/whitepaper/scripts'
sys.path[:0] = [str(REPO), str(SCRIPTS)]
from compute_panel import load
from research.benchmark.publish_lineage_appendix import (
    displayed_columns, lineage_caption, measured_lineage, write_longtable,
)
from research.benchmark.publish_baseline_completion import build, load_sources, render_budget_tex

STATIC = REPO / 'ops/ci/paper-inputs'
OUT = REPO / 'docs/whitepaper/generated'
LOCK_SHA256 = 'c1c7142eed329e15ebdd2cd5745817962c03d09a68786166655cc159e2f97b81'
INVENTORY_SHA256 = '504166764631dbdf5e02d116334af5cc587731fcf3efe130d62240e6dd512d64'
PREDECLARE = REPO / 'research/benchmark/review_fixes/predeclare.json'
PREDECLARE_SHA256 = '921865c374bc4f0d6dd88f3450409ea3c35a84ff9e8ce206d5772b7710dca3e5'


def authenticated(path, digest):
    if path.is_symlink():
        raise ValueError('symlinked paper input')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('paper input digest mismatch')
    return raw


def tost_protocol_tex():
    """Emit protocol constants from the original, pre-result declaration."""
    if PREDECLARE.resolve() != PREDECLARE.absolute():
        raise ValueError('redirected TOST predeclaration')
    document = json.loads(authenticated(PREDECLARE, PREDECLARE_SHA256))
    protocol = document['tost']
    if (document['format'] != 'dope-review-fix-predeclaration'
            or protocol['declared_before_computation'] is not True):
        raise ValueError('invalid TOST predeclaration')
    return (
        '% Protocol constants authenticated against the original pre-result declaration.\n'
        + r'\newcommand{\TostEquivalenceMargin}{' + str(protocol['margin_retention']) + '}\n'
        + r'\newcommand{\TostAlpha}{' + str(protocol['alpha']) + '}\n'
    )


def inventory_tex(document):
    rows = document['datasets']
    if len(rows) != document['dataset_count'] or len({r['name'] for r in rows}) != len(rows):
        raise ValueError('invalid inventory matrix')
    lines = [r'\scriptsize', r'\begin{longtable}{>{\raggedright\arraybackslash}p{5.5cm}lrrll}',
             r'\caption{All \BeyondInventory{} BeyondArena families.}\\', r'\toprule',
             r'name & task & rows & cols & license & split\\', r'\midrule', r'\endfirsthead',
             r'\multicolumn{6}{l}{\tablename\ \thetable, continued}\\', r'\toprule',
             r'name & task & rows & cols & license & split\\', r'\midrule', r'\endhead',
             r'\bottomrule', r'\endfoot']
    tasks = {'binary_classification': 'binary', 'multiclass_classification': 'multiclass', 'regression': 'regression'}
    for row in sorted(rows, key=lambda r: r['name']):
        fields = [row['name'].replace('_', r'\_\allowbreak{}'), tasks[row['task']], f"{row['rows']:,}",
                  str(row['columns']), {'LicenseRef-Public-Domain': 'public-domain', 'LicenseRef-US-Government-Works': 'us-gov'}.get(row['license_spdx'], row['license_spdx']) or 'not cleared', row['stratum']]
        lines.append(' & '.join(fields) + r'\\')
    return '\n'.join(lines + [r'\end{longtable}', r'\normalsize', ''])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = json.loads(authenticated(STATIC / 'inputs.lock.json', LOCK_SHA256))
    for name, ref in lock['files'].items():
        if Path(name).name != name:
            raise ValueError('invalid static input name')
        raw = authenticated(STATIC / name, ref['sha256'])
        if len(raw) != ref['bytes']:
            raise ValueError('static input length mismatch')
        (OUT / name).write_bytes(raw)
    (OUT / 'tost-protocol.tex').write_text(tost_protocol_tex())
    record = load('s3-lineage-record.json')
    columns = displayed_columns()
    for kind, key, name in [('Displayed profile', 'rows', 's3-2048-lineages.tex'),
                            ('Sufficiency budget', 'sufficiency_rows', 's3-8192-lineages.tex')]:
        rows = record[key]
        write_longtable(OUT / name, lineage_caption(kind, rows, columns),
                        [row for row in rows if measured_lineage(row, columns)], columns)
        path = OUT / name
        path.write_text(path.read_text().replace(r'\footnotesize', r'\scriptsize'))
    inventory = json.loads(authenticated(REPO / 'research/benchmark/results/beyondarena-s3-inventory.json', INVENTORY_SHA256))
    (OUT / 'beyondarena-inventory.tex').write_text(inventory_tex(inventory))
    lock, docs = load_sources()
    (OUT / 'baseline-budget.tex').write_text(render_budget_tex(build(lock, docs)))
    # The replay summary is a dependency of compute_panel, not a later side effect.
    from research.benchmark.mfs_v3_score import _v3_contract, evaluate_v3, PREREQUISITES
    contract = _v3_contract()
    gates = contract['release_gates']
    rows = [
        ('Scalar', r'$100\exp(\sum_i w_i\log\max(s_i,\epsilon)/\sum_i w_i)$, clipped to $[0,100]$'),
        ('Weights', ', '.join(key.replace('_', r'\_') + ' = ' + str(value) for key, value in contract['master_fitness']['weights'].items())),
        ('Epsilon', '$' + str(contract['master_fitness']['epsilon']).replace('e-06', r'\times 10^{-6}') + '$'),
        ('Bytes', r'$b\in\mathbb{Z},\ 0<b\le ' + str(gates['byte_limit_l3']) + '$'),
        ('Prerequisites', r'Real-vs-real control; required sizes; artifact sampling; locked metric implementations.'),
        ('Matched null', 'Median linear MSE of three independent-marginal candidates; candidate-index tie break.'),
        ('Near-copy floor', '$q=' + str(gates['near_copy_quantile']) + '$ of nearest-other distinct real-row squared distances; linear interpolation.'),
    ]
    text = r'\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{0.73\linewidth}@{}}' + '\n' + r'\toprule' + '\n'
    text += '\n'.join(name + ' & ' + value + r' \\' for name, value in rows)
    text += '\n' + r'\bottomrule' + '\n' + r'\end{tabular}' + '\n'
    (OUT / 'mfs-v3-conformance.tex').write_text(text)
    evidence = {key: True for key in PREREQUISITES} | {
        'encoder': 'kumo_tabular_l', 'normalizer': 'empirical_midrank_quantile_v1',
        'utility_protocol': 'train_on_synthetic_score_on_real',
        'utility_auditors': ['kumo_tabular_l', 'mitra_v2', 'tabicl2'],
        'exact_row_matches': 0, 'near_copy_ok': True, 'cleartext_absent': True,
        'membership_auc': 0.5, 'attribute_inference_advantage': 0.0, 'artifact_bytes': 1000,
        'distance': 0.0, 'd_null': 0.5, 'd_match': 0.4, 'gap_mitra': -0.1, 'gap_tabicl': -0.2,
        'mfs_components': {key: 1.0 for key in contract['master_fitness']['weights'] if key != 'representation_closeness'},
    }
    perfect = evaluate_v3(evidence)['score']
    evidence['mfs_components']['compactness'] = 0.0
    zero = evaluate_v3(evidence)['score']
    (OUT / 'mfs-v3-examples.tex').write_text(
        r'\newcommand{\MfsVThreePerfect}{' + f'{perfect:.0f}' + '}\n' +
        r'\newcommand{\MfsVThreeZeroCompact}{' + f'{zero:.2f}' + '}\n')
    import render_figures
    render_figures.draw_loss()
    print('paper inputs and appendix tables regenerated from committed pins')


if __name__ == '__main__':
    main()
