"""Publish the frozen scaled TabSyn default fit phase, without quality claims."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

from .publish_s3_forest import require
from .publish_s3_matched import schema as infer_schema

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
NAME = 'tabsyn-default100-scaled-cuda-fits'
CACHE = HERE.parents[1] / 'target/tabsyn-default100-fit-ledger-v1/publisher-input-custody'
CACHE_SHA = 'e961b5a62b412b26e10a9731c487e103af65fbd5c0c3632cfc1989ad0b69ff8b'
MANIFEST_SHA = '555229ad005510f621880fe59be5f91ba8de03297fc13df7ae3aca17d600e04a'
REPLAY = Path('/home/ubuntu/dope/target/work-order-b-root-readiness-v1/TabSyn100-independent-small-custody-replay.actual.v1.json')
REPLAY_SHA = '7a740feb199a4590a1eced09cb349f496f8baf07d4180a18f226b99f456e027d'
CONFIG_SHA = 'a34729383eac6783c7e754a791094054d54f1062655db9a306cd4cca296c5ebc'
AUTHOR = 'cb5ac0f74ec36ee88e7a974a393dfbef50d42da7'
ADAPTER_SHA = '456e9f718daf3a765cd44e25240a339300e3bcdc149b0f3deb4b9ad0fc4444f2'
NULL_CLAIMS = ('utility', 'native_objective', 'common_validation', 'privacy', 'mfs_v2',
               'ptf_v1', 'release_safe_l3', 'paired_superiority')
INCOMPLETE = ('full_sample_matrix_complete', 'native_selection_complete', 'common_validation_complete',
              'privacy_coverage_complete', 'full_runtime_hardware_certified', 'production_certified',
              'official_tests_opened', 'counts_as_dope_win', 'current_bulk_model_hashes_reverified',
              'book_charges_are_measured_method_cost')


def hashed(path, expected):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'custody alias rejected')
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, 'frozen custody bytes changed')
    return data


def anchored(root, expected):
    """Capture all verified bytes once; decoding never rereads the filesystem."""
    root = Path(root)
    lock = json.loads(hashed(root / 'input-custody.lock.json', expected))
    require(lock['model_bulk_copied'] is False and lock['source_bytes_executed'] is False,
            'publisher custody scope changed')
    require(not any(p.is_symlink() for p in root.rglob('*'))
            and {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
            == set(lock['files']) | {'input-custody.lock.json'}, 'custody file inventory changed')
    buffers = {}
    for name, row in lock['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe custody member')
        data = hashed(root / name, row['sha256'])
        require(type(row['bytes']) is int and len(data) == row['bytes'], 'custody byte charge changed')
        buffers[name] = data
    for group in ('metadata', 'source_files'):
        for row in lock[group].values():
            require(lock['files'][row['file']]['sha256'] == row['sha256'], 'custody reference changed')
    return lock, buffers


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'invalid measured cost')
    return value


def same_cost(left, right):
    require(math.isclose(finite(left), finite(right), rel_tol=0, abs_tol=1e-8), 'measured cost differs')


def fit_cell(row, plan, read_ref, sources):
    require(re.fullmatch('[0-9a-f]{16}', row['dataset']) is not None
            and re.fullmatch('[0-9a-f]{64}', row['lineage_sha256']) is not None,
            'nonopaque lineage identity')
    require(row['status'] == 'fit_ok' and row['scientific_receipt_status'] == 'ok'
            and type(row['actual_worker_exit']) is int and row['actual_worker_exit'] == 0
            and type(row['fit_seed']) is int and row['fit_seed'] == 11
            and row['config_sha256'] == CONFIG_SHA and row['artifact_file_sizes_rechecked'] is True,
            'incomplete or changed fit')
    request = read_ref(row['fit_request_ref'])
    complete = read_ref(row['fit_complete_ref'])
    exit_row = read_ref(row['fit_exit_ref'])
    identity = request['learning_identity']
    require(identity == plan['learning_identity'] and identity['dataset'] == row['dataset']
            and request['lineage_sha256'] == plan['lineage_sha256'] == row['lineage_sha256']
            and identity['source_commit'] == request['model_source_commit'] == AUTHOR
            and type(identity['fit_seed']) is int and identity['fit_seed'] == 11
            and request['generated_fixture'] is False and request['official_tests_opened'] is False
            and request['hardware_qualification'] is None and request['production_certified'] is False,
            'fit request identity or claims changed')
    expected_sources = {p: h for p, h in sources.items() if '/ts100-owner-direct-v1/' in p} if row['retained_first'] is True else sources
    require(type(row['retained_first']) is bool and request['source_files'] == expected_sources
            and request['config']['config_sha256'] == CONFIG_SHA
            and request['config']['vae_epochs'] == 200 and request['config']['diffusion_epochs'] == 1000
            and request['author_default_4000_10001_track_preserved'] is True
            and request['author_patience_500_preserved'] is True, 'scaled source or epoch track changed')
    require(type(exit_row['actual_worker_exit']) is int and exit_row['actual_worker_exit'] == 0
            and exit_row['foreign_signals'] == 0 and complete['status'] == 'ok'
            and complete['lineage_sha256'] == row['lineage_sha256']
            and complete['config_sha256'] == CONFIG_SHA and complete['hardware_qualification'] is None
            and complete['production_certified'] is False and complete['official_tests_opened'] is False
            and complete['mfs_v2'] is None and complete['ptf_v1'] is None,
            'fit exit or completion is incomplete')
    if 'success_receipt_ref' in exit_row:
        require(exit_row['success_receipt_ref'] == row['fit_complete_ref'], 'fit exit receipt differs')
    else:
        require(exit_row['fit_complete_receipt_present'] is True
                and exit_row['owned_termination_reason'] is None, 'retained fit did not close')
    same_cost(row['measured_whole_operation_seconds'], exit_row['whole_operation_seconds'])
    same_cost(row['measured_scientific_seconds'], complete['operation_seconds'])
    inventory = complete.get('inventory', complete['result'].get('inventory'))
    require(inventory == row['artifact_inventory']
            and row['artifact_inventory_receipt_ref'] == row['fit_complete_ref'], 'model inventory differs')
    files = inventory['files']
    expected = {'decoder-final.safetensors', 'diffusion-epoch-0.safetensors', 'diffusion.safetensors',
                'encoder-final.safetensors', 'model.json', 'numeric-state.npz', 'projection.json',
                'vae-selected.safetensors'}
    require(set(files) == expected, 'complete model or projection inventory missing')
    for item in files.values():
        require(type(item['bytes']) is int and item['bytes'] > 0
                and re.fullmatch('[0-9a-f]{64}', item['sha256']) is not None, 'invalid artifact charge')
    total = sum(x['bytes'] for x in files.values())
    require(type(row['artifact_bytes']) is int and total == row['artifact_bytes'] == inventory['artifact_bytes']
            and type(inventory['artifact_bytes']) is int and type(inventory['projection_bytes']) is int
            and inventory['projection_bytes'] == files['projection.json']['bytes']
            and identity['projection_sha256'] == files['projection.json']['sha256'], 'complete byte charge differs')
    refs = row['actual_metadata_refs']
    require(len({r['path'] for r in refs}) == len(refs), 'duplicate fit receipt')
    decoded = {Path(r['path']).name: read_ref(r) for r in refs}
    model = decoded['model.json']
    require(next(r['sha256'] for r in refs if Path(r['path']).name == 'model.json') == files['model.json']['sha256']
            and model == complete['result'].get('result', complete['result'])
            and model['complete'] is True and type(model['seed']) is int and model['seed'] == 11
            and model['source_commit'] == AUTHOR and model['config'] == request['config']
            and model['code_sha256'] == model['operation_admission_sha256'] == row['fit_request_ref']['sha256']
            and model['declared_epoch_budget'] == {'vae_epochs': 200, 'diffusion_epochs': 1000}
            and model['author_epoch_baseline'] == {'vae_epochs': 4000, 'diffusion_epochs': 10001}
            and model['generated_fixture'] is False and model['official_tests_opened'] is False
            and all(model[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe_l3', 'superiority')),
            'model receipt track or claims changed')
    cuda = decoded['worker.CUDA-initialized.receipt.json']
    require(cuda['request_sha256'] == row['fit_request_ref']['sha256']
            and cuda['scientific_body_sha256'] == ADAPTER_SHA and cuda['pid'] == exit_row['scientific_worker_pid']
            and cuda['hardware_qualification'] is None and cuda['real_TRAIN'] is True,
            'actual CUDA receipt binding changed')
    provenance = row['validation_provenance']
    require(provenance['partition'] == 'official_training_derived_validation'
            and provenance['official_tests_opened'] is False
            and provenance['worker_sha256'] == identity['worker_sha256']
            and provenance['projection_sha256'] == identity['projection_sha256'], 'validation lineage changed')
    prior = row['prior_history']['allocated_operations']
    require(all(x['measured_seconds'] is None and x['charged_seconds'] == 600 for x in prior),
            'unmeasured budget charge relabeled')
    return {'dataset_id': row['dataset'], 'lineage_sha256': row['lineage_sha256'], 'fit_seed': 11,
            'config_sha256': CONFIG_SHA, 'status': 'fit_ok', 'actual_fit_exit': 0,
            'retained_first_fit': row['retained_first'], 'charged_artifact_bytes': total,
            'model_bytes': total - inventory['projection_bytes'], 'projection_bytes': inventory['projection_bytes'],
            'artifact_files': files, 'inventory_receipt': row['artifact_inventory_receipt_ref'],
            'fit_request': row['fit_request_ref'], 'fit_complete': row['fit_complete_ref'],
            'fit_exit': row['fit_exit_ref'], 'small_receipts': refs,
            'worker_sha256': identity['worker_sha256'], 'joint_train_sha256': identity['joint_train_sha256'],
            'joint_validation_sha256': identity['joint_validation_sha256'],
            'measured_whole_seconds': row['measured_whole_operation_seconds'],
            'measured_scientific_seconds': row['measured_scientific_seconds'],
            'prior_unmeasured_allocations': len(prior), 'prior_book_charge_seconds': len(prior) * 600,
            'physical_file_sizes_read_back': True, 'current_bulk_model_hashes_reverified': False}


def summarize(cells):
    require(len(cells) == 100 and len({x['dataset_id'] for x in cells}) == 100
            and len({x['lineage_sha256'] for x in cells}) == 100, 'missing or duplicate default fit')
    require(all(x['status'] == 'fit_ok' and type(x['actual_fit_exit']) is int and x['actual_fit_exit'] == 0
                and type(x['fit_seed']) is int and x['fit_seed'] == 11 and x['config_sha256'] == CONFIG_SHA
                and type(x['charged_artifact_bytes']) is int
                and x['charged_artifact_bytes'] == x['model_bytes'] + x['projection_bytes']
                for x in cells), 'fit coverage or byte charge changed')
    require(all(type(x['retained_first_fit']) is bool for x in cells)
            and sum(x['retained_first_fit'] for x in cells) == 1, 'retained fit accounting changed')
    return {'closed_default_fits': 100,
            'new_default_fits': 99, 'retained_default_fits': 1,
            'charged_model_and_projection_bytes': sum(x['charged_artifact_bytes'] for x in cells),
            'measured_whole_seconds': sum(finite(x['measured_whole_seconds']) for x in cells),
            'measured_scientific_seconds': sum(finite(x['measured_scientific_seconds']) for x in cells),
            'prior_unmeasured_allocations': sum(x['prior_unmeasured_allocations'] for x in cells),
            'prior_unmeasured_book_charge_seconds': sum(x['prior_book_charge_seconds'] for x in cells)}


def build():
    lock, buffers = anchored(CACHE, CACHE_SHA)
    require(lock['manifest_sha256'] == MANIFEST_SHA and len(lock['metadata']) == 804
            and len(lock['source_files']) == 6, 'frozen population custody changed')
    manifest = json.loads(buffers['manifest.json'])
    require(hashlib.sha256(buffers['manifest.json']).hexdigest() == MANIFEST_SHA,
            'manifest identity changed')
    sources = {p: r['sha256'] for p, r in lock['source_files'].items()}
    require(sources == manifest['source_files'] and manifest['host'] == 'xbabe1'
            and manifest['source_commit'] == AUTHOR and manifest['config_sha256'] == CONFIG_SHA
            and manifest['fit_phase_closed'] is True
            and manifest['default_fits_all_ok'] is True and manifest['counts'] == {
                'fit_ok': 100, 'closed_unavailable': 0, 'running_or_closing': 0, 'unstarted': 0}
            and manifest['full_campaign_complete'] is False and manifest['hardware_qualification'] is None
            and manifest['official_tests_opened'] is False and manifest['production_certified'] is False
            and manifest['mfs_v2'] is None and manifest['ptf_v1'] is None
            and manifest['model_bulk_copied'] is False, 'fit phase scope or seal changed')
    require({r['remote_path']: r['sha256'] for r in manifest['metadata_custody']}
            == {p: r['sha256'] for p, r in lock['metadata'].items()}, 'receipt graph changed')
    def read_ref(ref):
        row = lock['metadata'][ref['path']]
        require(row['sha256'] == ref['sha256'], 'frozen receipt reference differs')
        return json.loads(buffers[row['file']])
    plan = read_ref(manifest['plan_ref'])
    require(read_ref(manifest['source_peer_ref'])['source_files'] == sources, 'source peer differs')
    plans = {r['dataset']: r for r in plan['lineages']}
    require(len(plans) == len(plan['lineages']) == 100, 'plan coverage changed')
    cells = [fit_cell(r, plans[r['dataset']], read_ref, sources) for r in manifest['lineages']]
    summary = summarize(cells)
    require(summary['charged_model_and_projection_bytes'] == manifest['retained_model_bytes'] == 8486920261
            and summary['prior_unmeasured_allocations'] == manifest['prior_allocations_with_unmeasured_cost'] == 35
            and summary['prior_unmeasured_book_charge_seconds'] == manifest['prior_conservative_book_charge_seconds'] == 21000,
            'population charge reconciliation differs')
    same_cost(summary['measured_whole_seconds'], manifest['measured_closed_fit_whole_seconds'])
    same_cost(summary['measured_scientific_seconds'], manifest['measured_positive_or_partial_scientific_receipt_seconds'])
    replay = json.loads(hashed(REPLAY, REPLAY_SHA))
    require(replay['manifest_ref']['sha256'] == MANIFEST_SHA and replay['actual_fit_exits_zero'] == 100
            and replay['artifact_inventory_bytes_charged'] == summary['charged_model_and_projection_bytes']
            and replay['verified_small_metadata_files'] == 804 and replay['model_bulk_rehashed_by_this_replay'] is False,
            'independent replay differs')
    return {'format': 'tabsyn-default100-scaled-cuda-fit-publication', 'version': 1,
            'scope': 'Default configuration fit phase only; 100 S3 TRAIN-derived validation lineages',
            'track': 'scaled_200_vae_1000_diffusion', 'author_default_epochs': {'vae': 4000, 'diffusion': 10001},
            'scaled_epochs': {'vae': 200, 'diffusion': 1000}, 'author_patience': 500,
            'author_commit': AUTHOR, 'config_sha256': CONFIG_SHA, 'epoch_policy_sha256': manifest['scaled_epoch_policy_sha256'],
            'publisher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'host': 'xbabe1', 'device': 'CUDA', 'fit_seed': 11,
            'partition': 'official_training_derived_validation', 'summary': summary, 'cells': cells,
            'source_files': sources, 'local_source_cache': {'path': str(CACHE / 'input-custody.lock.json'), 'sha256': CACHE_SHA},
            'immutable_references': [manifest['plan_ref'], manifest['source_peer_ref'], manifest['dispatch_amendment_ref'],
                                     {'path': str(CACHE / 'manifest.json'), 'sha256': MANIFEST_SHA},
                                     {'path': str(REPLAY), 'sha256': REPLAY_SHA}],
            'verified_small_receipts': 804, 'verified_frozen_source_files': 6,
            'current_bulk_model_hashes_reverified': False, 'physical_file_sizes_read_back': True,
            'book_charges_are_measured_method_cost': False, 'energy_attributable_to_job': None,
            'full_sample_matrix_complete': False, 'native_selection_complete': False,
            'common_validation_complete': False, 'privacy_coverage_complete': False,
            'full_runtime_hardware_certified': False, 'production_certified': False,
            'official_tests_opened': False, 'counts_as_dope_win': False,
            **{k: None for k in NULL_CLAIMS}}


def tables(report):
    require(summarize(report['cells']) == report['summary']
            and all(report[k] is False for k in INCOMPLETE)
            and all(report[k] is None for k in NULL_CLAIMS), 'publication acquired a claim')
    columns = ('dataset_id', 'lineage_sha256', 'fit_seed', 'config_sha256', 'status', 'actual_fit_exit',
               'charged_artifact_bytes', 'model_bytes', 'projection_bytes', 'measured_whole_seconds',
               'measured_scientific_seconds', 'prior_unmeasured_allocations', 'prior_book_charge_seconds')
    out = io.StringIO(); writer = csv.writer(out, lineterminator='\n'); writer.writerow(columns)
    for row in report['cells']: writer.writerow([row[k] for k in columns])
    s = report['summary']
    markdown = '\n'.join(['# TabSyn default 100 CUDA fit ledger', '',
        'Fit phase only: 100/100 actual fit exits were zero, at seed 11 on xbabe1 (99 new fits and one retained prior fit).', '',
        'This is the scaled 200 VAE / 1,000 diffusion epoch track. Author defaults remain a separate 4,000 / 10,001 track; patience 500 is preserved.', '',
        f"Measured closed fit operations: {s['measured_whole_seconds']:.6f}s. Charged model and projection bytes: {s['charged_model_and_projection_bytes']:,}.", '',
        f"Prior {s['prior_unmeasured_allocations']} allocations carry {s['prior_unmeasured_book_charge_seconds']:,} conservative seconds; these are unmeasured budget charges, not measured method runtime.", '',
        'Frozen inventories bind all model files and projection bytes. Physical file sizes were read back; this publisher does not rehash the retained 8.49 GB of bulk models.', '',
        'Later sampling batches and their infrastructure failure are outside this fit ledger. Full sampling, native selection, common validation and privacy coverage remain incomplete. Runtime/hardware and production certification are absent.', '',
        'Official tests remain sealed. Utility, native/common outcomes, privacy, MFS-v2, PTF-v1, release and superiority are null. No DOPE win is claimed.', ''])
    return out.getvalue(), markdown


def schema(report):
    result = infer_schema(report); properties = result['properties']
    for key in ('host', 'device', 'fit_seed', 'config_sha256', 'author_commit', 'scaled_epochs', 'author_default_epochs'):
        properties[key] = {'const': report[key]}
    items = properties['cells']['items']
    for shape in items.get('anyOf', [items]):
        shape['properties']['dataset_id'] = {'type': 'string', 'pattern': '^[0-9a-f]{16}$'}
        for key in ('fit_seed', 'config_sha256', 'status', 'actual_fit_exit'):
            shape['properties'][key] = {'const': report['cells'][0][key]}
    properties['source_files'] = {'const': report['source_files']}
    return result


def main():
    report = build(); table, markdown = tables(report); path = RESULTS / (NAME + '.json')
    path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    path.with_suffix('.schema.json').write_text(json.dumps(schema(report), sort_keys=True, indent=2) + '\n')
    path.with_suffix('.csv').write_text(table); path.with_suffix('.md').write_text(markdown)
    print(path)


if __name__ == '__main__': main()
