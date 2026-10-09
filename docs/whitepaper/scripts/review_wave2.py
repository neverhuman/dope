"""Main-text numbers, sizes, and paired TabSyn points from bound public cells."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from research.benchmark.review_fixes.receipt_panel import (
    AUDITORS, _table, primary_map, reduce_cells, fmt, ci, tex_name,
)
from research.benchmark.review_fixes.bind_review_cells import canonical, decode_cells
from docs.whitepaper.scripts.review_tabsyn import _adapt

PREFIX = 'research/benchmark/results/'
RECEIPTS = PREFIX + 'review-fixes-receipts-v1/panel.json'
PRIVACY = PREFIX + 'review-fixes-privacy-v1/panel.json'
TABSYN = PREFIX + 'review-fixes-tabsyn-v1/panel.json'
CELLS = PREFIX + 'review-fixes-tabsyn-v1/scalar-cells.jsonl'
DENSITY = PREFIX + 'density-matched-population-validation.json'
NAMES = PREFIX + 's3-lineage-record.json'
INPUTS = (RECEIPTS, PRIVACY, TABSYN, CELLS, DENSITY, NAMES)
SOURCE_SHA256 = {'research/benchmark/results/density-matched-population-validation.json': '5264b88a40ad5efb21d9b119789caeb13e71e911a17f1d8b57f1d8e8ac31732a',
 'research/benchmark/results/review-fixes-privacy-v1/panel.json': '2caa429ae03b5bbc61b30c4ce9391670f70ec9a991e2d398974a8bf3d76942f8',
 'research/benchmark/results/review-fixes-receipts-v1/panel.json': 'f0ecf5929825f1f07d877d4e01714f1ebd55868db3b863482f46be0f27ffdf47',
 'research/benchmark/results/review-fixes-tabsyn-v1/panel.json': '8e87a4b304588f6b3c2e6cf34c2a2ee9f4b4f0ea3d282f836f14cbc23a7dbfa3',
 'research/benchmark/results/review-fixes-tabsyn-v1/scalar-cells.jsonl': 'b9fd4600ec54fc89b938eb627f31d6112176fc4137e040c3686c15efd17d2d62',
 'research/benchmark/results/s3-lineage-record.json': 'bbd0852c49f595315104261efa4eaa6f50db257e323b8c002aac6737a2dbe7d7'}


def payload_from_inputs(inputs):
    receipts = json.loads(inputs[RECEIPTS])
    privacy = json.loads(inputs[PRIVACY])
    tabsyn = json.loads(inputs[TABSYN])
    cells = decode_cells(inputs[CELLS])
    from research.benchmark.review_fixes.receipt_panel import names_of
    names = names_of(json.loads(inputs[NAMES]))
    density = json.loads(inputs[DENSITY])
    adapted, _absent = _adapt(cells, names)
    synthetic = primary_map(reduce_cells(adapted, 'TabSyn', tabsyn['configuration']))
    dope = primary_map(reduce_cells(density['cells'], 'DOPE', 'features12_steps2048'))
    teaser = []
    for auditor in AUDITORS:
        keys = sorted(set(dope[4][auditor]) & set(synthetic[4][auditor]))
        differences = [dope[4][auditor][key] - synthetic[4][auditor][key] for key in keys]
        row = next(r for r in tabsyn['contrasts'] if (r['size'], r['auditor']) == (4, auditor))
        import statistics
        if len(differences) != row['n'] or statistics.median(differences) != row['median']:
            raise ValueError('TabSyn teaser points differ from the bound paired panel')
        teaser.append({'auditor': auditor, 'lineages': keys, 'differences': differences, 'summary': row})
    own = [{**r, 'cohort': 'own'} for r in receipts['retention']]
    matched = [{**r, 'cohort': 'matched TabSyn'} for r in tabsyn['levels'] if r['cohort'] == 'matched lineages']
    return {'format': 'dope-review-wave2-paper-v1', 'official_tests_opened': False, 'formal_dp': False,
            'sources': {name: hashlib.sha256(inputs[name]).hexdigest() for name in INPUTS},
            'size_retention': own + matched, 'tabsyn_teaser': teaser,
            'tabsyn_lineages': len({c['dataset'] for c in cells}),
            'tabsyn_configuration': tabsyn['configuration'],
            'tabsyn_contrasts': tabsyn['contrasts'], 'privacy': privacy['summaries']}


def label(method, config):
    method = {'ForestDiffusion/Forest-Flow': 'Forest-Flow',
              'independent_marginals': 'Independent marginals'}.get(method, method)
    config = {'author_default': 'default', 'native_selected': 'selected',
              'features12_steps2048': 'profile',
              'scaled_200_vae_1000_diffusion': 'scaled default'}.get(config, config)
    return tex_name(method + ' / ' + config)


def retention_table(payload):
    grouped = {}
    for row in payload['size_retention']:
        key = (row['method'], row['configuration'], row['cohort'])
        entry = grouped.setdefault(key, {})
        slot = (row['size'], row['auditor'])
        if slot in entry:
            raise ValueError('size retention row is duplicated')
        entry[slot] = row
    rows = []
    order = {'DOPE': 0, 'ARF': 1, 'TabSyn': 2, 'GaussianCopula': 3, 'Chow-Liu': 4,
             'independent_marginals': 5, 'CTGAN': 6, 'TVAE': 7,
             'ForestDiffusion/Forest-Flow': 8}
    for (method, config, cohort), grid in sorted(grouped.items(), key=lambda x: (order[x[0][0]], x[0])):
        cohort_label = 'matched' if cohort == 'matched TabSyn' else 'own'
        for size in (1, 4):
            values = []
            for auditor in AUDITORS:
                row = grid[(size, auditor)]
                values.append(r'\shortstack{' + fmt(row['median']) + r' \\ {' + ci(row)
                              + ' (' + str(row['n']) + ')}}')
            rows.append(label(method, config) + ' & ' + cohort_label + ' & ' + str(size) + '$n$ & ' + ' & '.join(values) + r' \\')
    header = 'Method / configuration & Cohort & Size & CatBoost & Linear & MLP'
    return _table(header, rows, 'lllrrr')


def numbers(payload):
    values = {'WaveTabsynLineages': str(payload['tabsyn_lineages'])}
    epochs = re.fullmatch(r'scaled_(\d+)_vae_(\d+)_diffusion', payload['tabsyn_configuration'])
    if epochs is None:
        raise ValueError('TabSyn epoch configuration differs from the admitted arm')
    values['WaveTabsynVaeEpochs'], values['WaveTabsynDiffusionEpochs'] = epochs.groups()
    for auditor, name in (('catboost', 'Cb'), ('linear', 'Lin'), ('mlp', 'Mlp')):
        row = next(r for r in payload['tabsyn_contrasts'] if (r['size'], r['auditor']) == (4, auditor))
        for field, suffix in (('n', 'N'), ('median', 'Diff'), ('lo', 'Lo'), ('hi', 'Hi')):
            values['WaveTabsyn' + name + suffix] = str(row[field]) if field == 'n' else fmt(row[field])
    for metric, name in (('dcr_validation_median', 'HoldoutDcr'), ('dcr_fit_median', 'FitDcr'),
                          ('nndr_fit_median', 'Nndr'), ('distance_mia_auc', 'Mia'),
                          ('c2st_catboost_auc', 'CatboostDetection')):
        row = next(r for r in payload['privacy'] if (r['method'], r['configuration'], r['size'], r['metric']) ==
                   ('DOPE', 'features12_steps2048', 4, metric))
        values['WaveDope' + name] = fmt(row['median'])
        values['WaveDope' + name + 'Lo'] = fmt(row['lo'])
        values['WaveDope' + name + 'Hi'] = fmt(row['hi'])
        values['WaveDope' + name + 'N'] = str(row['n'])
    return ''.join('\\newcommand{\\' + name + '}{' + value + '}\n' for name, value in values.items())


def authenticated_inputs():
    captured = {}
    for name in INPUTS:
        path = REPO / name
        if path.is_symlink() or path.resolve() != REPO.resolve() / name:
            raise ValueError('redirected wave2 paper input')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256[name]:
            raise ValueError('wave2 paper input digest differs')
        captured[name] = raw
    return captured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-panel', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.from_panel.resolve() != REPO / TABSYN:
        raise ValueError('wave2 renderer requires the bound TabSyn panel')
    payload = payload_from_inputs(authenticated_inputs())
    outputs = {'review-wave2.json': canonical(payload),
               'review-wave2-numbers.tex': numbers(payload),
               'review-wave2-size-retention.tex': retention_table(payload)}
    for name, text in outputs.items():
        path = REPO / 'docs/whitepaper/generated' / name
        if args.check:
            if path.read_text() != text:
                raise ValueError('generated wave2 output differs')
        else:
            path.write_text(text)


if __name__ == '__main__':
    main()
