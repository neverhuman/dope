"""Configuration-aware fidelity from pinned scalar receipts, without fitting."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.benchmark.review_fixes.receipt_panel import (
    _table, fmt, ci, names_of, marginal, tex_name,
)
from research.benchmark.review_fixes.bind_review_cells import canonical, decode_cells, identity, finite, unique_identities

PREFIX = 'research/benchmark/results/'
RECEIPTS = PREFIX + 'review-fixes-receipts-v1/panel.json'
PRIVACY = PREFIX + 'review-fixes-privacy-v1/panel.json'
EXPANDED = PREFIX + 'expanded-validation-diagnostics.json'
FOLLOWON = PREFIX + 'arf-tabsyn-followon-validation-v1/panel.json'
RETAINED = PREFIX + 'tabsyn-eight-lineage-validation/panel.json'
NAMES = PREFIX + 's3-lineage-record.json'
TABSYN = PREFIX + 'review-fixes-tabsyn-v1/scalar-cells.jsonl'
DIAGNOSTICS = PREFIX + 'review-fixes-tabsyn-fidelity-v1/scalar-cells.jsonl'
META = PREFIX + 'review-fixes-tabsyn-fidelity-v1/scalar-cells.meta.json'
DECLARATION = 'research/benchmark/review_fixes/tabsyn-fidelity-predeclare.json'
PANEL = PREFIX + 'review-fixes-fidelity-v1/panel.json'
INPUTS = (RECEIPTS, PRIVACY, EXPANDED, FOLLOWON, RETAINED, NAMES, TABSYN, DIAGNOSTICS, META, DECLARATION)
SOURCE_SHA256 = {'research/benchmark/results/arf-tabsyn-followon-validation-v1/panel.json': 'e797f3ef0c1907d51caddb06c7669683afaf96274a2af5a885ab5dd7abdaf13e',
 'research/benchmark/results/expanded-validation-diagnostics.json': 'd146d9333dae187ec7c8a2f2c1d26acc15f50c988d8cd74117e834ab5249c796',
 'research/benchmark/results/review-fixes-privacy-v1/panel.json': '081e8b87dd31b08b55e107566f847cb610d33e491b91f193b668617d7c1bb463',
 'research/benchmark/results/review-fixes-receipts-v1/panel.json': 'f0ecf5929825f1f07d877d4e01714f1ebd55868db3b863482f46be0f27ffdf47',
 'research/benchmark/results/review-fixes-tabsyn-fidelity-v1/scalar-cells.jsonl': 'f7df39e9a45989fbe9cb8ff127134a3edd284ad72de3530cb63ee462fce50697',
 'research/benchmark/results/review-fixes-tabsyn-fidelity-v1/scalar-cells.meta.json': '1a7f3e892fa0610ba98138b865dff5c2d27ac3c59dc62d4a52ee27bd592de686',
 'research/benchmark/results/review-fixes-tabsyn-v1/scalar-cells.jsonl': 'b9fd4600ec54fc89b938eb627f31d6112176fc4137e040c3686c15efd17d2d62',
 'research/benchmark/results/s3-lineage-record.json': 'bbd0852c49f595315104261efa4eaa6f50db257e323b8c002aac6737a2dbe7d7',
 'research/benchmark/results/tabsyn-eight-lineage-validation/panel.json': 'ee1f26ceddf0c5ba82d6d0ce5da2d1f956a44b4a8ce9413f8f4a6d223676ccb5',
 'research/benchmark/review_fixes/tabsyn-fidelity-predeclare.json': 'e58a9ba4a659cfbac43d370d3eda1886254d6cb8327974b324317c67292744c8'}
PANEL_SHA256 = 'bc2ecd817584cd309c8cd2d06cf5b5e2d6933fb3a538e5285afad60f0a77ff00'
PRODUCER_SHA256 = '3a0f958b5b1a7b5bf1621baea6d26d48693dbc56ad2f2f8bcb5c609fad17c7a9'


def labels(method, config):
    method = {'ForestDiffusion/Forest-Flow': 'Forest-Flow',
              'independent_marginals': 'Independent marginals'}.get(method, method)
    config = {'features12_steps2048': 'DOPE profile', 'author_default': 'author default',
              'native_selected': 'native selected',
              'scaled_200_vae_1000_diffusion': 'scaled schedule'}.get(config, config)
    return tex_name(method), tex_name(config)


def summary(cells, metric, method, config, size, names, cohort):
    grouped = defaultdict(dict)
    for c in cells:
        if c['size'] != size or not finite(c.get(metric)):
            continue
        key, seed = c['dataset'], c['sample_seed']
        if seed in grouped[key]:
            raise ValueError('fidelity scalar identity is duplicated')
        grouped[key][seed] = float(c[metric])
    values = {dataset: sorted(by_seed.values())[1] for dataset, by_seed in grouped.items()
              if set(by_seed) == {101, 211, 307}}
    row = marginal(values, names, f'wave2-fidelity|{method}|{config}|{size}|{metric}|{cohort}')
    row.update(method=method, configuration=config, size=size, metric=metric, cohort=cohort)
    return row


def payload_from_inputs(inputs):
    docs = {name: json.loads(inputs[name]) for name in (RECEIPTS, PRIVACY, EXPANDED, FOLLOWON, RETAINED, NAMES, DECLARATION)}
    for name in (EXPANDED, FOLLOWON, RETAINED, NAMES, DECLARATION):
        if docs[name].get('official_tests_opened') is not False:
            raise ValueError('fidelity source official test flag refused')
    names = names_of(docs[NAMES])
    stored = docs[RECEIPTS]['fidelity_stored']
    utility_cells = decode_cells(inputs[TABSYN])
    unique_identities(utility_cells, 'tabsyn')
    utility = {identity(c, 'tabsyn'): c for c in utility_cells}
    diagnostics = decode_cells(inputs[DIAGNOSTICS])
    unique_identities(diagnostics, 'tabsyn')
    declaration = docs[DECLARATION]
    if declaration['official_tests_opened'] is not False or declaration['scalar_ledger_sha256'] != hashlib.sha256(inputs[TABSYN]).hexdigest() or declaration['validation_pin_source']['sha256'] != hashlib.sha256(inputs[FOLLOWON]).hexdigest():
        raise ValueError('fidelity declaration differs from the pinned inputs')
    meta = json.loads(inputs[META])
    if (meta.get('format') != 'dope-wave2-tabsyn-fidelity-cells-v1'
            or meta.get('official_tests_opened') is not False or meta.get('formal_dp') is not False
            or meta.get('declaration_sha256') != hashlib.sha256(inputs[DECLARATION]).hexdigest()
            or meta.get('ledger_sha256') != hashlib.sha256(inputs[DIAGNOSTICS]).hexdigest()
            or meta.get('producer_sha256') != PRODUCER_SHA256
            or meta.get('expanded_implementation_sha256') != declaration['expanded_implementation_sha256']
            or type(meta.get('cell_count')) is not int or meta['cell_count'] != len(diagnostics)
            or {key: meta.get(key) for key in declaration['runtime']} != declaration['runtime']):
        raise ValueError('TabSyn diagnostic provenance differs from frozen source or runtime')
    if len(diagnostics) != len(utility) or {identity(c, 'tabsyn') for c in diagnostics} != set(utility):
        raise ValueError('TabSyn fidelity cohort differs from bound utility cells')
    keys = {'dataset', 'size', 'sample_seed', 'official_tests_opened', 'formal_dp',
            'synthetic_sha256', 'validation_sha256', 'status', 'marginal_ks_mean', 'pair_correlation_fidelity',
            'c2st_catboost_auc', 'c2st_status', 'c2st_rows_per_class', 'c2st_reason',
            'alpha_beta_status', 'alpha_beta_reason', 'alpha_precision', 'beta_recall'}
    for c in diagnostics:
        if set(c) != keys or c['status'] != 'ok' or not finite(c['marginal_ks_mean']) or not finite(c['pair_correlation_fidelity']):
            raise ValueError('TabSyn fidelity scalar fields refused')
        if c['synthetic_sha256'] != utility[identity(c, 'tabsyn')]['synthetic_sha256'] or c['validation_sha256'] != declaration['validation_sha256_by_dataset'][c['dataset']]:
            raise ValueError('TabSyn fidelity sample or validation hash differs')
        for metric in ('marginal_ks_mean', 'pair_correlation_fidelity'):
            if not 0 <= c[metric] <= 1:
                raise ValueError('TabSyn fidelity scalar is outside its range')
        for status, reason, metrics in (('c2st_status', 'c2st_reason', ('c2st_catboost_auc',)),
                                        ('alpha_beta_status', 'alpha_beta_reason', ('alpha_precision', 'beta_recall'))):
            if c[status] == 'ok':
                if c[reason] is not None or any(not finite(c[m]) or not 0 <= c[m] <= 1 for m in metrics):
                    raise ValueError('TabSyn diagnostic success/status differs')
            elif c[status] == 'unavailable':
                if not isinstance(c[reason], str) or not c[reason] or any(c[m] is not None for m in metrics):
                    raise ValueError('TabSyn unavailable diagnostic contains a result')
            else:
                raise ValueError('TabSyn diagnostic status refused')
        if c['c2st_status'] == 'ok' and (type(c['c2st_rows_per_class']) is not int or
                c['c2st_rows_per_class'] != min(utility[identity(c, 'tabsyn')]['rows']['validation'],
                                              utility[identity(c, 'tabsyn')]['rows']['synthetic'], 800)):
            raise ValueError('TabSyn detector class-n differs from the frozen rule')
    fidelity = list(stored)
    for size in (1, 4):
        for field, metric in (('marginal_ks_mean', 'ks'), ('pair_correlation_fidelity', 'pair')):
            row = summary(diagnostics, field, 'TabSyn', 'scaled_200_vae_1000_diffusion', size, names, 'common numeric; own lineages')
            row['metric'] = metric
            fidelity.append(row)
    detectors = []
    for row in docs[PRIVACY]['summaries']:
        if row['metric'] == 'c2st_catboost_auc':
            detectors.append({**row, 'metric': 'catboost'})
    for size in (1, 4):
        row = summary(diagnostics, 'c2st_catboost_auc', 'TabSyn', 'scaled_200_vae_1000_diffusion', size, names, 'common numeric; own lineages')
        row['metric'] = 'catboost'
        detectors.append(row)
    alpha_cells = defaultdict(list)
    alpha_sizes = defaultdict(set)
    for row in stored:
        key = (row['method'], row['configuration'], 'common numeric')
        alpha_cells[key]
        alpha_sizes[key].add(row['size'])
    for c in docs[EXPANDED]['cells']:
        if c.get('fit_seed') in (None, 11):
            key = (c['method'], c['configuration'], 'common numeric')
            alpha_cells[key].append({**c, 'size': c.get('size_multiplier', 4),
                'alpha_precision': c.get('alpha_precision') if c.get('alpha_beta_status') == 'ok' else None,
                'beta_recall': c.get('beta_recall') if c.get('alpha_beta_status') == 'ok' else None})
            alpha_sizes[key].add(c.get('size_multiplier', 4))
    for c in docs[FOLLOWON]['rows']:
        if c['method'] == 'ARF' and c['fit_seed'] == 11:
            key = ('ARF', c['selection_binding'], 'common numeric')
            alpha_cells[key].append({
                **c, 'size': c['row_multiplier']})
            alpha_sizes[key].add(c['row_multiplier'])
    for c in docs[RETAINED]['cells']:
        block = c['metrics']['alpha_beta']
        if c['fit_seed'] == 11:
            key = ('TabSyn', 'scaled_200_vae_1000_diffusion', 'retained subset')
            alpha_cells[key].append({
                'dataset': c['dataset'], 'sample_seed': c['sample_seed'], 'size': c['row_multiplier'],
                **(block['value'] if block.get('status') == 'ok' else {})})
            alpha_sizes[key].add(c['row_multiplier'])
    for c in diagnostics:
        key = ('TabSyn', 'scaled_200_vae_1000_diffusion', 'common numeric')
        alpha_cells[key].append(c)
        alpha_sizes[key].add(c['size'])
    alpha = []
    for (method, config, cohort), cells in sorted(alpha_cells.items()):
        for size in sorted(alpha_sizes[(method, config, cohort)]):
            for metric in ('alpha_precision', 'beta_recall'):
                alpha.append(summary(cells, metric, method, config, size, names, cohort))
    return {'format': 'dope-review-fidelity-v2', 'official_tests_opened': False,
            'formal_dp': False, 'fidelity': fidelity, 'detectors': detectors, 'alpha_beta': alpha,
            'sources': {name: hashlib.sha256(inputs[name]).hexdigest() for name in INPUTS},
            'claims': {'superiority': None, 'mfs_v2': None, 'ptf_v1': None, 'release_safe_l3': None},
            'scope': 'Median of three sample seeds, then hierarchical lineage summary at fit seed11. Configurations and unmatched cohorts stay separate; every metric prints its own lineage count. Stored logistic C2ST is distinct from the shared grouped CatBoost protocol. Forest-Flow alpha/beta and CatBoost remain unavailable.'}


def cell(row):
    if row is None:
        return '---'
    if row['median'] is None:
        return r'\begin{tabular}[t]{@{}c@{}}unavailable \\ $n=' + str(row['n']) + r'$\end{tabular}'
    # Top-aligned, so the row label sits on the median line rather than on the count.
    return (r'\begin{tabular}[t]{@{}c@{}}' + fmt(row['median']) + r' \\ {' + ci(row) + '}'
            + r' \\ $n=' + str(row['n']) + r'$\end{tabular}')


def render(payload):
    grouped = defaultdict(dict)
    for row in payload['fidelity'] + payload['detectors']:
        key = (row['method'], row['configuration'], row['size'])
        if row['metric'] in grouped[key]:
            raise ValueError('fidelity configuration/metric row is duplicated')
        grouped[key][row['metric']] = row
    rows = []
    for (method, config, size), metrics in sorted(grouped.items()):
        m, c = labels(method, config)
        rows.append(f'{m} & {c} & {size}$n$ & ' + ' & '.join(cell(metrics.get(k)) for k in ('ks', 'pair', 'c2st', 'catboost')) + r' \\')
    text = _table('Method & Configuration & Size & Marginal KS & Pair fidelity & Logistic AUC & CatBoost AUC', rows, 'lllrrrr')
    alpha_groups = defaultdict(dict)
    for row in payload['alpha_beta']:
        alpha_groups[(row['method'], row['configuration'], row['size'], row['cohort'])][row['metric']] = row
    rows = []
    for (method, config, size, cohort), metrics in sorted(alpha_groups.items()):
        m, c = labels(method, config)
        rows.append(f'{m} & {c} & {size}$n$ & {tex_name(cohort)} & ' +
                    cell(metrics.get('alpha_precision')) + ' & ' + cell(metrics.get('beta_recall')) + r' \\')
    return text, _table(r'Method & Configuration & Size & Cohort & $\alpha$ precision & $\beta$ recall', rows, 'llllrr')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-panel', type=Path)
    parser.add_argument('--write-panel', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.write_panel:
        captured = {}
        for name in INPUTS:
            path = REPO / name
            if path.is_symlink() or path.resolve() != REPO.resolve() / name:
                raise ValueError('redirected fidelity input')
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256[name]:
                raise ValueError('fidelity input digest differs before decoding')
            captured[name] = raw
        payload = payload_from_inputs(captured)
        path = REPO / PANEL; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(canonical(payload))
    else:
        path = args.from_panel or REPO / PANEL
        if path.is_symlink() or path.resolve() != REPO / PANEL:
            raise ValueError('redirected fidelity panel')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != PANEL_SHA256:
            raise ValueError('fidelity panel digest differs before decoding')
        payload = json.loads(raw)
    fidelity, alpha = render(payload)
    outputs = {'review-fidelity.json': canonical(payload), 'review-fidelity.tex': fidelity,
               'review-alpha-beta.tex': alpha}
    for name, text in outputs.items():
        path = REPO / 'docs/whitepaper/generated' / name
        if args.check:
            if path.read_text() != text:
                raise ValueError('generated fidelity output differs from panel')
        else:
            path.write_text(text)


if __name__ == '__main__':
    main()
