"""Export recorded checkpoints and measured validation KPI inputs, without execution."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import median

from . import arf_runtime_guard as guard
from .publish_s3_matched import schema

HERE = Path(__file__).parent
RESULTS = HERE / 'results'
NAME = 'checkpoint-figure-input-index'
INPUT_SHA = '813ffe564d44995566c055f3dc3d2505d25ebf0f6178d0dfe2454414aa6999b2'
PANELS = ('dope-s3-population-validation.json', 'dope-target-refinement-population-validation.json',
          'arf-matched-population-validation.json', 'density-matched-population-validation.json')
SEEDS = (101, 211, 307)
CLAIMS = dict(scientific_execution=False, official_tests_opened=False, production_certified=False,
              complete_benchmark_matrix=False, all_requested_methods_covered=False,
              native_selection_changed=False, native_values_ranked_across_methods=False,
              current_bulk_hashes_reverified=False, panel_pooling_authorized=False,
              mfs_v2=None, ptf_v1=None, release_safe=None, superiority=None)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def bound_bytes(path, expected):
    guard.digest(expected)
    path = Path(path)
    guard.require(path.is_absolute() and path == path.resolve(strict=True)
                  and not any(p.is_symlink() for p in (path, *path.parents))
                  and path.is_file() and path.name != 'test.csv' and 'evaluator' not in path.parts,
                  'publication input must be ordinary metadata outside the sealed test root')
    body = path.read_bytes()
    guard.require(hashlib.sha256(body).hexdigest() == expected, 'frozen publication input changed')
    return body


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def grouped(panel):
    """Keep each publication separate, with all three original sample identities."""
    guard.require(panel['official_tests_opened'] is False, 'sealed test evidence is forbidden')
    groups = defaultdict(list)
    for row in panel['cells']:
        guard.require(type(row['fit_seed']) is int and type(row['sample_seed']) is int
                      and type(row['size_multiplier']) is int, 'typed sample identity required')
        guard.require(row['size_multiplier'] in (1, 4) and row['sample_seed'] in SEEDS,
                      'unexpected validation sample schedule')
        for name in ('mfs_v2', 'ptf_v1'):
            guard.require(row[name] is None, 'gated score cannot enter a validation export')
        key = (row['method'], row['dataset'], row.get('configuration', row.get('profile')),
               row['fit_seed'], row['size_multiplier'])
        groups[key].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        guard.require(sorted(r['sample_seed'] for r in rows) == list(SEEDS),
                      'missing or duplicated sample seed')
        charges = {r['charged_artifact_bytes'] for r in rows}
        guard.require(len(charges) == 1, 'checkpoint charge differs between sample seeds')
        complete = all(r['status'] == 'ok' for r in rows)
        out = dict(zip(('method', 'dataset', 'configuration', 'fit_seed', 'size_multiplier'), key))
        out.update(sample_seeds=','.join(map(str, SEEDS)), complete_three_sample_group=complete,
                   status_counts=dict(Counter(r['status'] for r in rows)),
                   charged_artifact_bytes=next(iter(charges)))
        for auditor in ('catboost', 'linear', 'mlp'):
            values = [(r.get('utility') or {}).get(auditor) for r in rows]
            for loss in ('tstr_loss', 'trtr_loss'):
                measured = complete and all(v is not None and finite(v.get(loss)) for v in values)
                guard.require(not complete or measured, 'successful group has missing finite loss')
                out[auditor + '_' + loss.replace('_loss', '_mse')] = median(v[loss] for v in values) if measured else None
            informative = complete and all(v is not None and v.get('informative') is True for v in values)
            guard.require(not informative or all(finite(v.get('retention')) for v in values),
                          'informative group has missing finite retention')
            out[auditor + '_retention'] = median(v['retention'] for v in values) if informative else None
        result.append(out)
    return result


def csv_text(rows, fields):
    out = io.StringIO(); writer = csv.DictWriter(out, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(row[k], sort_keys=True, separators=(',', ':'))
                         if isinstance(row[k], (dict, list)) else row[k] for k in fields})
    return out.getvalue().encode()


def csv_lf(body):
    """Preserve CSV field strings while normalizing exported record terminators."""
    source = io.StringIO(body.decode(), newline='')
    target = io.StringIO(newline='')
    csv.writer(target, lineterminator='\n').writerows(csv.reader(source))
    return target.getvalue().encode()


def checkpoint_rows(index, metadata_digests):
    guard.require(index['official_tests_opened'] is False and index['current_bulk_hashes_reverified'] is False
                  and index['complete_benchmark_matrix'] is False, 'checkpoint inventory scope differs')
    result = []; identities = set()
    for row in index['checkpoint_records']:
        key = (row['method'], row['dataset'], row['configuration'], row['fit_seed'])
        guard.require(key not in identities and type(row['fit_seed']) is int, 'duplicate checkpoint identity')
        identities.add(key)
        fit = row['fit_receipt']; guard.digest(fit['sha256'])
        guard.require(fit['sha256'] in metadata_digests, 'checkpoint receipt absent from authenticated metadata')
        guard.require(row['physical_checkpoint_identity'] == fit['sha256']
                      and row['current_bulk_hashes_reverified'] is False,
                      'checkpoint identity or hash-verification claim differs')
        for name in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority'):
            guard.require(row[name] is None, 'checkpoint acquired a gated score')
        cm = row['common_validation']['4']
        guard.require(row['key_error_value'] == cm['median_tstr_mse']
                      and (row['key_error_value'] is None or (finite(row['key_error_value'])
                           and cm['complete_three_sample_group'] is True
                           and cm['measured_sample_seeds'] == list(SEEDS))), 'checkpoint error group differs')
        for artifact in row['recorded_artifacts']:
            guard.digest(artifact['sha256'])
            guard.require(type(artifact['bytes']) is int and artifact['bytes'] >= 0, 'recorded artifact bytes invalid')
        native = row['native_validation_objective']
        guard.require(native.get('value') is None or finite(native['value']), 'nonfinite native objective')
        result.append(dict(method=row['method'], dataset=row['dataset'], configuration=row['configuration'],
            fit_seed=row['fit_seed'], status=row['status'], charged_artifact_bytes=row['charged_artifact_bytes'],
            key_error_metric=row['key_error_metric'], key_error_value=row['key_error_value'],
            key_error_missing_reason=row['key_error_missing_reason'], native_objective=native,
            fit_receipt_path=fit['path'], fit_receipt_sha256=fit['sha256'],
            recorded_artifacts=row['recorded_artifacts']))
    guard.require(dict(Counter(r['method'] for r in result)) == index['coverage']
                  and sum(r['key_error_value'] is not None for r in result) == index['key_error_measured_records'],
                  'checkpoint coverage differs')
    return result


def verify_checkpoint_errors(index, panels):
    """Join the listed checkpoint errors back to their committed sample cells."""
    logical, arf = defaultdict(list), defaultdict(list)
    for name in PANELS:
        for row in panels[name]['cells']:
            key = (row['method'], row['dataset'], row.get('configuration', row.get('profile')), row['fit_seed'])
            logical[key].append(row)
            if row['method'] == 'ARF': arf[row['fit_job_sha256']].append(row)
    for checkpoint in index['checkpoint_records']:
        key = (checkpoint['method'], checkpoint['dataset'], checkpoint['configuration'], checkpoint['fit_seed'])
        rows = arf[checkpoint['job_sha256']] if checkpoint['method'] == 'ARF' else logical[key]
        by_seed = {}
        for row in rows:
            if row['size_multiplier'] != 4 or row['status'] != 'ok': continue
            value = row['utility']['catboost']['tstr_loss']
            guard.require(finite(value), 'nonfinite recorded checkpoint loss')
            seed = row['sample_seed']
            guard.require(seed not in by_seed or by_seed[seed] == value, 'checkpoint metric aliases disagree')
            by_seed[seed] = value
        expected = median(by_seed.values()) if set(by_seed) == set(SEEDS) else None
        guard.require(checkpoint['key_error_value'] == expected, 'checkpoint error differs from measured cells')


def build():
    anchor_path = (RESULTS / NAME / 'inputs.lock.json').absolute()
    anchor = guard.decode(bound_bytes(anchor_path, INPUT_SHA))
    frozen = {name: bound_bytes(ref['path'], ref['sha256']) for name, ref in anchor['private_inputs'].items()}
    guard.require(all(len(frozen[name]) == ref['bytes'] for name, ref in anchor['private_inputs'].items()),
                  'frozen input byte count differs')
    readset = guard.decode(frozen['metadata_readset'])
    # Check every small metadata buffer before projecting any metrics; never open model/sample bodies.
    for path, ref in readset.items():
        body = bound_bytes(path, ref['sha256'])
        guard.require(len(body) == ref['bytes'], 'metadata byte count differs')
    index = guard.decode(frozen['checkpoint_index'])
    source_pins = {name: ref['sha256'] for name, ref in index['source_publications'].items()}
    panels = {}
    for name, expected in source_pins.items():
        panels[name] = guard.decode(bound_bytes((RESULTS / name).absolute(), expected))
    checkpoints = checkpoint_rows(index, {r['sha256'] for r in readset.values()})
    verify_checkpoint_errors(index, panels)
    figure_rows = []
    for name in PANELS:
        for row in grouped(panels[name]):
            figure_rows.append(dict(source_panel=name, source_sha256=source_pins[name], **row))
    checkpoint_fields = tuple(checkpoints[0])
    figure_fields = tuple(figure_rows[0])
    outputs = {'checkpoints.csv': csv_text(checkpoints, checkpoint_fields),
               'figure-kpis.csv': csv_text(figure_rows, figure_fields),
               'forest-key-errors.csv': csv_lf(frozen['forest_key_errors']),
               'tabddpm-historical-v8-checkpoints.csv': csv_lf(frozen['tabddpm_historical_checkpoints']),
               'tabddpm-current-operation-receipts.csv': csv_lf(frozen['tabddpm_current_operations'])}
    forest = guard.decode(frozen['forest_coverage'])
    td = guard.decode(frozen['tabddpm_partial_index'])
    # Repeat the small metadata and frozen-input checks before writing derived evidence.
    for path, ref in readset.items(): bound_bytes(path, ref['sha256'])
    for name, ref in anchor['private_inputs'].items():
        guard.require(bound_bytes(ref['path'], ref['sha256']) == frozen[name], 'input changed during projection')
    for name, expected in source_pins.items(): bound_bytes((RESULTS / name).absolute(), expected)
    report = dict(format='checkpoint-and-figure-input-publication-v1', version=1,
        scope='Recorded checkpoints and previously measured official-training-derived validation utility only.',
        loss_units='TSTR/TRTR mean squared error in each frozen numeric target representation; no cross-dataset raw-MSE ranking.',
        native_objective_units='Method-specific objectives; never ranked across methods.',
        forest_error_scope='Separate n/seed101 descriptive fragment; not the 4n/three-sample-seed schedule.',
        checkpoint_records=len(checkpoints), physical_checkpoint_identities=index['physical_checkpoint_identities'],
        checkpoint_coverage=index['coverage'], key_error_measured_records=index['key_error_measured_records'],
        figure_groups=len(figure_rows), complete_three_sample_groups=sum(r['complete_three_sample_group'] for r in figure_rows),
        forest_coverage=forest,
        tabddpm_coverage=dict(aggregate_units=td['current_TD14_aggregate_units'],
            historical_checkpoint_records=len(td['historical_recorded_providers']),
            current_operation_records=len(td['current_operations']), precise_gap=td['precise_gap'],
            current_provider_inventory_complete=False, current_model_hash_remeasured=False,
            native_scores_are_error_losses=False, no_new_scientific_work=True),
        private_inputs=anchor['private_inputs'],
        source_publications={name: dict(path='research/benchmark/results/' + name, sha256=pin) for name, pin in source_pins.items()},
        inputs_lock_sha256=INPUT_SHA, publisher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        artifact_tables={name: dict(sha256=hashlib.sha256(body).hexdigest(), bytes=len(body)) for name, body in outputs.items()},
        checkpoint_aliases_are_additional_compute=False, full_five_fit_seed_coverage=False,
        artifact_payload_bytes_read=0, remaining_methods=['TabDDPM inventory pending; full comparison incomplete',
                                                        'TabSyn shared metrics incomplete', 'Forest-Flow full panel incomplete'],
        **CLAIMS)
    return report, outputs


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output-dir', type=Path, default=RESULTS / NAME)
    root = parser.parse_args().output_dir
    report, outputs = build()
    for name, body in outputs.items(): (root / name).write_bytes(body)
    (root / 'publication.json').write_bytes(encoded(report))
    (root / 'publication.schema.json').write_bytes(encoded(schema(report)))
    print(root / 'publication.json')


if __name__ == '__main__': main()
