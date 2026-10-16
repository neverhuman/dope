"""Bind existing public privacy/TabSyn scalars; never fit or read data rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / 'research/benchmark/results'
PREDECLARE = 'research/benchmark/review_fixes/predeclare.json'
PREDECLARE_SHA256 = '921865c374bc4f0d6dd88f3450409ea3c35a84ff9e8ce206d5772b7710dca3e5'
TABSYN_FITS = 'research/benchmark/results/tabsyn-default100-scaled-cuda-fits.json'
TABSYN_REFERENCE = 'research/benchmark/results/tabsyn-eight-lineage-validation/panel.json'
TABSYN_SAMPLES = 'research/benchmark/results/review-fixes-tabsyn-v1/sample-evidence.json'
TABSYN_OPERATIONS = 'research/benchmark/results/review-fixes-tabsyn-v1/sampling-receipts.json'
SOURCE_SHA256 = {
    TABSYN_FITS: 'cb2c1c134ccd1af9865e156faee8796143c212d1f76e70ab70f7048f82a08272',
    TABSYN_REFERENCE: '3e49c1f8f99615f21abee3504afb645e6b61eaab2d88be2e279be0c0c1b91d76',
    TABSYN_SAMPLES: '274b6eef4ce95545ded969763ffffa3e1154a2d9192c1e2867dd7ddcb719e0e3',
    TABSYN_OPERATIONS: 'e09c7eb22abb15a171a321b7f4c7b80b276509963898870a6f430965f665d88d',
}
PRIVACY_BASE = {'configuration', 'dataset', 'formal_dp', 'hipaa_deidentification',
                'method', 'official_tests_opened', 'reason', 'sample_seed', 'size', 'status'}
PRIVACY_METRICS = {'c2st_catboost_auc', 'c2st_rows_per_class', 'c2st_status',
                   'dcr_fit_median', 'dcr_validation_median', 'distance_mia_auc',
                   'distance_mia_status', 'nndr_fit_median', 'rows', 'synthetic_sha256'}
TABSYN_KEYS = {'dataset', 'formal_dp', 'null_loss', 'official_tests_opened', 'rows',
               'sample_seed', 'size', 'status', 'synthetic_sha256', 'utility'}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def canonical(value):
    return json.dumps(value, indent=2, sort_keys=True) + '\n'


def decode_cells(raw):
    cells = [json.loads(line) for line in raw.splitlines()]
    if not cells or any(not isinstance(c, dict) for c in cells):
        raise ValueError('public scalar ledger requires nonempty objects')
    if any(c.get('official_tests_opened') is not False or c.get('formal_dp') is not False for c in cells):
        raise ValueError('official test or formal privacy flag refused')
    return cells


def identity(cell, kind):
    return ((cell['method'], cell['configuration']) if kind == 'privacy' else ()) + (
        cell['dataset'], cell['size'], cell['sample_seed'])


def unique_identities(cells, kind):
    identities = [identity(c, kind) for c in cells]
    if len(set(identities)) != len(identities):
        raise ValueError('public scalar identity is duplicated')
    if any(type(c['size']) is not int or c['size'] not in (1, 4)
           or type(c['sample_seed']) is not int or c['sample_seed'] not in (101, 211, 307)
           for c in cells):
        raise ValueError('public scalar size or seed differs from predeclaration')
    return set(identities)


def admit_privacy(cells, sources):
    from research.benchmark.review_fixes.privacy_fidelity import PUBLISHED

    expected = {}
    for name, method, config in PUBLISHED:
        doc = json.loads(sources['research/benchmark/results/' + name])
        for c in doc['cells']:
            if (c.get('method'), c.get('configuration')) != (method, config):
                continue
            if c.get('fit_seed') not in (None, 11) or c.get('size_multiplier') not in (1, 4) or c.get('sample_seed') not in (101, 211, 307):
                continue
            e = (c.get('metric_receipt') if method == 'DOPE' else c.get('sample_evidence')) or {}
            digest = e.get('sample_sha256')
            if not digest and isinstance(e.get('path'), str) and e['path'].endswith('.csv'):
                digest = e.get('sha256')
            key = (method, config, c['dataset'], c['size_multiplier'], c['sample_seed'])
            if key in expected:
                raise ValueError('published privacy identity is duplicated')
            expected[key] = digest
    for c in cells:
        if c.get('hipaa_deidentification') is not False:
            raise ValueError('formal privacy flag refused')
        if c.get('status') == 'ok':
            if set(c) != PRIVACY_BASE | PRIVACY_METRICS or c.get('reason') is not None:
                raise ValueError('privacy cell is outside the public scalar set')
            for field in ('c2st_catboost_auc', 'dcr_fit_median', 'dcr_validation_median', 'distance_mia_auc', 'nndr_fit_median'):
                if c[field] is not None and not finite(c[field]):
                    raise ValueError('privacy scalar is not finite')
                if c[field] is not None and (c[field] < 0 or (field in ('c2st_catboost_auc', 'distance_mia_auc', 'nndr_fit_median') and c[field] > 1)):
                    raise ValueError('privacy scalar is outside its domain')
            rows = c['rows']
            if not isinstance(rows, dict) or set(rows) != {'fit', 'synthetic', 'validation'} or any(type(v) is not int or v < 1 for v in rows.values()):
                raise ValueError('privacy row counts are invalid')
        elif c.get('status') == 'unavailable':
            if set(c) != PRIVACY_BASE or not isinstance(c['reason'], str):
                raise ValueError('unavailable privacy cell is outside the public set')
        else:
            raise ValueError('privacy cell status is outside the recorded protocol')
    actual = unique_identities(cells, 'privacy')
    if actual != set(expected):
        raise ValueError('privacy ledger does not cover the frozen published cohort')
    for c in cells:
        if c['status'] == 'ok':
            digest = expected[identity(c, 'privacy')]
            if not valid_sha(digest) or c['synthetic_sha256'] != digest:
                raise ValueError('privacy synthetic hash differs from the frozen published sample')


def admit_tabsyn(cells, binding, sources):
    declaration = json.loads(sources[PREDECLARE])['tabsyn']
    fits = json.loads(sources[TABSYN_FITS])
    if fits.get('official_tests_opened') is not False or fits.get('config_sha256') != declaration['config_sha256']:
        raise ValueError('TabSyn fit configuration or official test flag differs')
    allowed = {c['dataset_id'] for c in fits['cells'] if c.get('status') == 'fit_ok'
               and c.get('actual_fit_exit') == 0 and valid_sha((c.get('fit_complete') or {}).get('sha256'))
               and c.get('fit_seed') == 11 and c.get('config_sha256') == declaration['config_sha256']}
    gate = binding['parser_check']
    if gate.get('official_tests_opened') is not False or gate.get('match') is not True:
        raise ValueError('TabSyn parser official test flag or match refused')
    if (gate.get('dataset'), gate.get('sample_seed'), gate.get('row_multiplier')) != ('02a45777900441ce', 101, 4):
        raise ValueError('TabSyn parser identity differs from predeclaration')
    reference = json.loads(sources[TABSYN_REFERENCE])
    if reference.get('official_tests_opened') is not False:
        raise ValueError('TabSyn reference official test flag refused')
    matches = [c for c in reference['cells'] if (c.get('dataset'), c.get('sample_seed'), c.get('row_multiplier')) ==
               (gate.get('dataset'), gate.get('sample_seed'), gate.get('row_multiplier')) and c.get('method') == 'TabSyn']
    if len(matches) != 1:
        raise ValueError('TabSyn parser reference is not unique')
    expected = matches[0]['metrics']['utility']['auditors']['catboost']['retention']
    if not finite(gate.get('scored_retention')) or gate.get('expected_retention') != expected or abs(gate['scored_retention'] - expected) > 1e-9:
        raise ValueError('TabSyn parser differs from the frozen published reference')
    expected_grid = {(dataset, size, seed) for dataset in allowed
                     for size in (1, 4) for seed in (101, 211, 307)}
    sample_document = json.loads(sources[TABSYN_SAMPLES])
    if sample_document.get('official_tests_opened') is not False or sample_document.get('formal_dp') is not False or sample_document.get('fit_registry_sha256') != hashlib.sha256(sources[TABSYN_FITS]).hexdigest():
        raise ValueError('TabSyn independent sample evidence flag or fit pin differs')
    sample_cells = sample_document['cells']
    sample_grid = unique_identities(sample_cells, 'tabsyn')
    if sample_grid != expected_grid:
        raise ValueError('TabSyn independent sample evidence grid differs')
    sample_hashes = {}
    for sample in sample_cells:
        if sample.get('official_tests_opened') is not False or sample.get('formal_dp') is not False:
            raise ValueError('TabSyn independent sample evidence flag refused')
        if sample['status'] == 'ok':
            if sample['reason'] is not None or type(sample['bytes']) is not int or sample['bytes'] <= 0 or not valid_sha(sample['synthetic_sha256']):
                raise ValueError('TabSyn independent sample evidence is invalid')
            sample_hashes[identity(sample, 'tabsyn')] = sample['synthetic_sha256']
        elif sample['status'] != 'unavailable' or sample['reason'] != 'sample_csv_absent' or sample['bytes'] is not None or sample['synthetic_sha256'] is not None:
            raise ValueError('TabSyn independent unavailable sample evidence is invalid')
    operations = set()
    operation_rows = {}
    for op in binding['operations']:
        if op.get('official_tests_opened') is not False or op.get('status') != 'ok':
            raise ValueError('TabSyn sampling official test flag refused')
        key = (op['dataset'], op['size'], op['seed'])
        if key not in expected_grid or type(op['size']) is not int or type(op['seed']) is not int:
            raise ValueError('TabSyn sampling identity is outside the frozen grid')
        if key in operations:
            raise ValueError('TabSyn sampling identity is duplicated')
        if type(op['rows']) is not int or op['rows'] <= 0:
            raise ValueError('TabSyn sampling row count is invalid')
        operations.add(key)
        operation_rows[key] = op['rows']
    unavailable = set()
    for op in binding['unavailable']:
        if op.get('official_tests_opened') is not False or op.get('formal_dp') is not False:
            raise ValueError('TabSyn unavailable official test flag refused')
        key = identity(op, 'tabsyn')
        if key not in expected_grid or type(op['size']) is not int or type(op['sample_seed']) is not int or op['status'] != 'unavailable' or op['reason'] != 'sample_csv_absent':
            raise ValueError('TabSyn unavailable identity or disposition differs')
        if key in unavailable or key in operations:
            raise ValueError('TabSyn unavailable identity is duplicated or overlaps success')
        unavailable.add(key)
    if operations | unavailable != expected_grid or set(sample_hashes) != operations:
        raise ValueError('TabSyn sampling dispositions do not cover the frozen grid')
    for rejected in binding['rejected_operations']:
        if rejected['reason'] != 'foreign_dataset_identity' or rejected['receipt']['dataset'] in allowed:
            raise ValueError('TabSyn rejected sampling receipt contains an admitted identity')
    for c in cells:
        if set(c) != TABSYN_KEYS or c['status'] != 'ok' or c['dataset'] not in allowed or not valid_sha(c['synthetic_sha256']) or not finite(c['null_loss']) or c['null_loss'] < 0:
            raise ValueError('TabSyn scalar or artifact cohort refused')
        rows = c['rows']
        if not isinstance(rows, dict) or set(rows) != {'train', 'synthetic', 'validation'} or any(type(v) is not int or v < 1 for v in rows.values()) or rows['synthetic'] != rows['train'] * c['size']:
            raise ValueError('TabSyn declared synthetic size differs')
        if set(c['utility']) != {'catboost', 'linear', 'mlp'}:
            raise ValueError('TabSyn auditor set differs')
        for b in c['utility'].values():
            if set(b) != {'informative', 'low_signal_noninferior', 'retention', 'trtr_loss', 'tstr_loss'} or type(b['informative']) is not bool or (b['low_signal_noninferior'] is not None and type(b['low_signal_noninferior']) is not bool):
                raise ValueError('TabSyn utility fields refused')
            if any(not finite(b[k]) for k in ('trtr_loss', 'tstr_loss')) or (b['retention'] is not None and not finite(b['retention'])):
                raise ValueError('TabSyn utility scalar is not finite')
            if b['trtr_loss'] < 0 or b['tstr_loss'] < 0:
                raise ValueError('TabSyn utility loss is outside its domain')
            informative = c['null_loss'] - b['trtr_loss'] >= 0.01 * abs(c['null_loss'])
            if informative and c['null_loss'] - b['trtr_loss'] <= 0:
                raise ValueError('TabSyn informative denominator is invalid')
            expected_retention = ((c['null_loss'] - b['tstr_loss']) / (c['null_loss'] - b['trtr_loss'])) if informative else None
            expected_noninferior = (b['tstr_loss'] <= b['trtr_loss'] + 0.01 * c['null_loss']) if not informative else None
            if b['informative'] is not informative or b['low_signal_noninferior'] is not expected_noninferior or (expected_retention is None and b['retention'] is not None) or (expected_retention is not None and (b['retention'] is None or not math.isclose(b['retention'], expected_retention, rel_tol=1e-12, abs_tol=1e-12))):
                raise ValueError('TabSyn retention or signal flags differ from the published equation')
        key = identity(c, 'tabsyn')
        if c['synthetic_sha256'] != sample_hashes.get(key):
            raise ValueError('TabSyn synthetic hash differs from independent CSV evidence')
        if c['rows']['synthetic'] != operation_rows.get(key):
            raise ValueError('TabSyn sampling rows differ from metric rows')
    actual = unique_identities(cells, 'tabsyn')
    if actual != operations:
        raise ValueError('TabSyn metric cohort differs from successful sampling receipts')
    projected = project_tabsyn_binding(json.loads(sources[TABSYN_OPERATIONS]), sample_document)
    if canonical(binding) != canonical(projected):
        raise ValueError('TabSyn binding differs from authenticated source receipts')


def project_tabsyn_binding(receipts, samples):
    """Quarantine foreign raw identities; keep the complete genuine sample grid."""
    allowed = {c['dataset'] for c in samples['cells']}
    return {
        'parser_check': receipts['parser_check'],
        'operations': [op for op in receipts['operations'] if op['dataset'] in allowed],
        'unavailable': [c for c in samples['cells'] if c['status'] == 'unavailable'],
        'rejected_operations': [{'reason': 'foreign_dataset_identity', 'receipt': op}
                                for op in receipts['operations'] if op['dataset'] not in allowed],
    }


def replay(kind, raw, sources, binding=None):
    cells = decode_cells(raw)
    if kind == 'privacy':
        admit_privacy(cells, sources)
        from docs.whitepaper.scripts.review_privacy import payload_from_cells
        return payload_from_cells(cells, sources=sources)
    admit_tabsyn(cells, binding, sources)
    from docs.whitepaper.scripts.review_tabsyn import payload_from_cells
    return payload_from_cells(cells, binding['parser_check'], sources=sources)


def public_sources():
    from research.benchmark.review_fixes.privacy_fidelity import PUBLISHED
    from research.benchmark.review_fixes.source_pins import authenticated_published_bytes
    captured = authenticated_published_bytes(RESULTS, ['s3-lineage-record.json', *(s[0] for s in PUBLISHED)])
    sources = {'research/benchmark/results/' + n: raw for n, raw in captured.items()}
    sources[PREDECLARE] = (REPO / PREDECLARE).read_bytes()
    if hashlib.sha256(sources[PREDECLARE]).hexdigest() != PREDECLARE_SHA256:
        raise ValueError('predeclaration hash differs')
    for name in (TABSYN_FITS, TABSYN_REFERENCE, TABSYN_SAMPLES, TABSYN_OPERATIONS):
        sources[name] = (REPO / name).read_bytes()
        if hashlib.sha256(sources[name]).hexdigest() != SOURCE_SHA256[name]:
            raise ValueError('frozen TabSyn source hash differs')
    return sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', required=True, choices=('privacy', 'tabsyn'))
    parser.add_argument('--ledger', required=True, type=Path)
    parser.add_argument('--binding', type=Path)
    parser.add_argument('--panel', required=True, type=Path)
    parser.add_argument('--write-ledger', type=Path)
    parser.add_argument('--write-binding', type=Path)
    args = parser.parse_args()
    raw = args.ledger.read_bytes()
    sources = public_sources()
    binding = json.loads(args.binding.read_bytes()) if args.binding else (
        project_tabsyn_binding(json.loads(sources[TABSYN_OPERATIONS]), json.loads(sources[TABSYN_SAMPLES]))
        if args.kind == 'tabsyn' else None)
    payload = replay(args.kind, raw, sources, binding)
    if canonical(payload).encode() != args.panel.read_bytes():
        raise ValueError('scalar ledger does not reproduce the committed panel')
    if args.write_ledger:
        cells = sorted(decode_cells(raw), key=lambda c: identity(c, args.kind))
        text = ''.join(json.dumps(c, sort_keys=True, separators=(',', ':')) + '\n' for c in cells)
        args.write_ledger.write_text(text)
    if args.write_binding:
        args.write_binding.write_text(canonical(binding))
    print(args.kind + ' scalar cells reproduce the committed panel')


if __name__ == '__main__':
    main()
