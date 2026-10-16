"""Publish lossless byte feasibility; never sample or evaluate a generator."""
from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

from . import research_container as codec
from .manifest import digest
from .publish_s3_matched import schema
from .score import sha256

ROOT = Path('/mnt/fast-scratch/dope-benchmark/dope-s3-artifact-compression-probe-v1')
HERE = Path(__file__).parent
RESULTS = HERE / 'results'
NAME = 'dope-artifact-byte-feasibility'
RECEIPT_SHA = '57e1a26d0fefad7b24d1411ad884f8168d4f813e57252f88b75a10ab9876999e'
GRID_SHA = '016db3603b170c180e5a8f28cfcfccaa3dbf60864cffaca0bba4dcaaa5f09f3e'
PARENT_ANCHORS = {
    'round.lock.json': '87d031fe6d1d49c9c86417d6f4e8995b16de507d89a8fefe8f30fc6ed20e9d64',
    'receipt-lock-v1.json': 'b0471a8936042b907c4481fb2b84e2205bedd2fbec7538bfd26d1f8d07d94461',
    'reconciliation-v1.json': '76132e62124fd1535c741df86d0e46525c64a315fdda0268e6c278923fdb23f4',
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def checked(path, expected, base):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True).is_relative_to(base)
            and not any(p.is_symlink() for p in (path, *path.parents))
            and 'evaluator' not in path.parts and path.name != 'test.csv'
            and path.is_file() and sha256(path) == expected, 'byte evidence digest or path changed')
    return path


def read(path):
    return json.loads(Path(path).read_text())


def no_claims(row):
    require(row['official_tests_opened'] is False and row['mfs_v2'] is None
            and row['ptf_v1'] is None, 'byte analysis acquired a test or score claim')


def build():
    base = ROOT.parent
    receipt_path = checked(ROOT / 'receipt-lock-v1.json', RECEIPT_SHA, base)
    receipt = read(receipt_path)
    no_claims(receipt)
    require(receipt['generator_validation_admitted'] is False and receipt['production_certified'] is False
            and receipt['analysis_grid_sha256'] == GRID_SHA, 'byte analysis acquired admission')
    # Verify the external receipt anchor before any mutable analysis metrics.
    require(not any(p.is_symlink() for p in ROOT.rglob('*'))
            and {str(p) for p in ROOT.rglob('*') if p.is_file()}
            == set(receipt['refs']) | {str(receipt_path)}, 'byte analysis inventory changed')
    for name, expected in receipt['refs'].items():
        require(Path(name).parent == ROOT, 'byte analysis reference escaped its scope')
        checked(name, expected, base)
    checked(ROOT / 'analysis-grid.lock.json', GRID_SHA, base)
    checked(ROOT / 'analysis.json', receipt['analysis_sha256'], base)
    grid, analysis = read(ROOT / 'analysis-grid.lock.json'), read(ROOT / 'analysis.json')
    no_claims(grid); no_claims(analysis)
    require(analysis['generator_validation_admitted'] is False and analysis['raw_parent_scores_rewritten'] is False
            and analysis['production_certified'] is False and analysis['counts_as_dope_win'] is False
            and analysis['release_safe_l3'] is None and analysis['new_gpu_fits'] == 0,
            'byte result acquired a generator or production claim')
    require(grid['candidate_compression_levels'] == [1, 9] and grid['candidate_count'] == 32
            and grid['artifact_cap_bytes'] == 10240 and grid['whole_container_bytes_charged'] is True
            and grid['maximum_threads'] == 1 and grid['maximum_elapsed_seconds'] == 60
            and grid['maximum_decoded_bytes_per_candidate'] == codec.DECODED_LIMIT == 65536
            and grid['generator_validation_admitted'] is False and grid['new_gpu_fits'] == 0
            and analysis['analysis_grid_sha256'] == GRID_SHA
            and analysis['codec_sha256'] == grid['codec_source_sha256']
            == sha256(ROOT / 'codec.py') == sha256(Path(codec.__file__)), 'declared byte analysis changed')
    require(type(analysis['elapsed_seconds']) in (int, float) and math.isfinite(analysis['elapsed_seconds'])
            and 0 <= analysis['elapsed_seconds'] <= 60 and analysis['host'] in ('xbabe1', 'xbabe2', 'xbabe3'),
            'byte analysis cost or host changed')
    parent = base / 'dope-s3-population-research-v2'
    for name, expected in PARENT_ANCHORS.items(): checked(parent / name, expected, base)
    parent_lock, parent_report, parent_receipt = [read(parent / n)
        for n in ('round.lock.json', 'reconciliation-v1.json', 'receipt-lock-v1.json')]
    no_claims(parent_report); no_claims(parent_receipt)
    require(parent_report['round_sha256'] == grid['parent_round_sha256'] == PARENT_ANCHORS['round.lock.json']
            and grid['parent_receipt_lock_sha256'] == PARENT_ANCHORS['receipt-lock-v1.json']
            and parent_receipt['reconciliation_sha256'] == PARENT_ANCHORS['reconciliation-v1.json'],
            'parent fit custody changed')
    failed = {r['job_sha256']: r for r in parent_report['fit_cells'] if r['status'] == 'charged_artifact_cap'}
    jobs = {digest(j): j for j in parent_lock['jobs']}
    inputs = {r['parent_job_sha256']: r for r in grid['inputs']}
    require(len(failed) == len(inputs) == 16 and set(failed) == set(inputs),
            'byte probe omitted or duplicated a failed parent')
    cells = []
    identities = set()
    for row in analysis['candidates']:
        no_claims(row)
        key, level = row['parent_job_sha256'], row['compression_level']
        require(type(level) is int and level in (1, 9) and key in inputs and (key, level) not in identities,
                'byte candidate identity missing or duplicated')
        identities.add((key, level))
        item, parent_cell, job = inputs[key], failed[key], jobs[key]
        require(row['dataset'] == item['dataset'] == parent_cell['dataset'] == job['dataset']
                and row['profile'] == item['profile'] == parent_cell['profile'] == job['research_profile'],
                'byte candidate parent lineage changed')
        model, projection = Path(item['model']), Path(item['projection'])
        require(model == Path(parent_cell['model_path'])
                and projection == Path(job['worker']['path']) / 'projection.json'
                and item['model_sha256'] == row['model_sha256'] == parent_cell['model_sha256']
                == parent_receipt['refs'][str(model)]
                and item['projection_sha256'] == row['projection_sha256']
                == job['worker']['files']['projection.json'] == parent_receipt['refs'][str(projection)],
                'owned artifact source binding changed')
        checked(model, row['model_sha256'], base); checked(projection, row['projection_sha256'], base)
        require(row['raw_model_bytes'] == model.stat().st_size
                and row['raw_projection_bytes'] == projection.stat().st_size
                and row['original_charged_bytes'] == item['original_charged_bytes'] == parent_cell['charged_artifact_bytes']
                == model.stat().st_size + projection.stat().st_size, 'parent byte charge changed')
        output = ROOT / (key + '.level' + str(level) + '.artifact')
        proof_path = output.with_suffix('.receipt.json')
        require(read(proof_path) == row and row['status'] == 'ok'
                and row['artifact_path'] == str(output), 'candidate receipt identity changed')
        checked(output, row['artifact_sha256'], base)
        blob = output.read_bytes()
        require(codec.decode(blob) == (model.read_bytes(), projection.read_bytes())
                and codec.encode(model.read_bytes(), projection.read_bytes(), level) == blob,
                'lossless byte or encoding replay differs')
        require(row['container_bytes'] == len(blob) and row['container_header_bytes'] == codec.HEADER.size
                and row['all_container_bytes_charged'] is True
                and row['exact_original_bytes_recovered'] is True and row['encoding_replay_exact'] is True
                and type(row['byte_cap_satisfied']) is bool
                and row['byte_cap_satisfied'] == (len(blob) <= 10240), 'container bytes undercharged')
        require(row['generator_admissible'] is False and row['generator_validation_complete'] is False
                and row['release_safe_l3'] is None and row['new_gpu_fits'] == 0
                and type(row['elapsed_seconds']) in (int, float) and math.isfinite(row['elapsed_seconds'])
                and 0 <= row['elapsed_seconds'] <= analysis['elapsed_seconds'], 'candidate acquired a claim or invalid cost')
        cells.append({k: row[k] for k in ('dataset', 'profile', 'parent_job_sha256', 'compression_level',
            'raw_model_bytes', 'raw_projection_bytes', 'original_charged_bytes', 'container_bytes',
            'container_header_bytes', 'byte_cap_satisfied', 'exact_original_bytes_recovered',
            'encoding_replay_exact', 'elapsed_seconds', 'artifact_sha256')}
            | {'receipt': {'path': str(proof_path), 'sha256': receipt['refs'][str(proof_path)]},
               'generator_admissible': False, 'mfs_v2': None, 'ptf_v1': None, 'release_safe_l3': None})
    require(identities == {(key, level) for key in failed for level in (1, 9)}
            and len(cells) == analysis['candidate_count'] == 32 and analysis['all_candidates_accounted'] is True
            and sum(c['byte_cap_satisfied'] for c in cells) == analysis['byte_cap_satisfied_candidates'],
            'complete declared compression grid required')
    cells.sort(key=lambda r: (r['dataset'], r['profile'], r['compression_level']))
    return {'format': 'dope-owned-artifact-byte-feasibility-publication', 'version': 1,
        'scope': 'Lossless byte inspection of 16 already fitted over-cap artifacts; no generator sampling or quality evaluation',
        'source_sha256': sha256(Path(__file__)), 'codec_sha256': sha256(Path(codec.__file__)),
        'receipt_lock_sha256': RECEIPT_SHA, 'analysis_grid_sha256': GRID_SHA,
        'parent_round_sha256': PARENT_ANCHORS['round.lock.json'],
        'cells': cells, 'candidate_count': 32, 'parent_fit_cells': 16,
        'datasets': len({r['dataset'] for r in cells}), 'all_candidates_accounted': True,
        'byte_cap_satisfied_candidates': sum(c['byte_cap_satisfied'] for c in cells),
        'cost': {'byte_analysis_elapsed_seconds': analysis['elapsed_seconds'], 'host': analysis['host'],
            'new_gpu_fits': 0, 'parent_training_cost_reused_without_relabeling': True},
        'immutable_references': [{'path': str(ROOT / name), 'sha256': value} for name, value in
            [('receipt-lock-v1.json', RECEIPT_SHA), ('analysis-grid.lock.json', GRID_SHA),
             ('analysis.json', receipt['analysis_sha256'])]],
        'raw_parent_scores_rewritten': False, 'generator_validation_complete': False,
        'product_wire_format_supported': False, 'matrix_admission': False,
        'official_tests_opened': False, 'production_certified': False,
        'counts_as_dope_win': False, 'mfs_v2': None, 'ptf_v1': None, 'release_safe_l3': None}


def tables(report):
    cells = report['cells']
    require(len(cells) == 32 and all(r['generator_admissible'] is False
            and r['mfs_v2'] is None and r['ptf_v1'] is None and r['release_safe_l3'] is None for r in cells),
            'byte table acquired a generator claim')
    fields = ['dataset', 'profile', 'compression_level', 'original_charged_bytes', 'container_bytes', 'byte_cap_satisfied']
    stream = io.StringIO(); writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(fields)
    for row in cells: writer.writerow([row[k] for k in fields])
    lines = ['# DOPE lossless artifact byte feasibility', '',
        'Byte inspection only. Original over-cap GPU fit receipts remain unchanged. Wrapped generators have not been sampled or evaluated; MFS-v2/PTF-v1/release-safe remain null.', '',
        '| Profile | Level | Candidates | Within 10,240 container bytes | Largest container |',
        '|---|---:|---:|---:|---:|']
    for profile, level in sorted({(r['profile'], r['compression_level']) for r in cells}):
        rows = [r for r in cells if (r['profile'], r['compression_level']) == (profile, level)]
        lines.append(f"| {profile} | {level} | {len(rows)} | {sum(r['byte_cap_satisfied'] for r in rows)} | {max(r['container_bytes'] for r in rows)} |")
    lines.extend(['', f'The full {codec.HEADER.size}-byte header, compressed model and complete learned projection are charged. Exact decoded model/projection hashes and repeated encoding were verified for all 32 candidates.', '',
        'This research container is not a product wire format. Generator validity, held-out utility, privacy, coverage, runtime admission and paired comparison remain pending. No DOPE win or production claim.', ''])
    return stream.getvalue(), '\n'.join(lines)


def main():
    report = build(); path = RESULTS / (NAME + '.json')
    path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    path.with_suffix('.schema.json').write_text(json.dumps(schema(report), sort_keys=True, indent=2) + '\n')
    table, text = tables(report)
    path.with_suffix('.csv').write_text(table); path.with_suffix('.md').write_text(text)
    print(path)


if __name__ == '__main__': main()
