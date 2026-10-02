"""Bind complete matched S3 validation outputs to immutable source and cell receipts."""
from __future__ import annotations

import json
from pathlib import Path

from .publish_s3_matched import NAME, require, schema, summarize
from .score import sha256

HERE = Path(__file__).parent
RESULTS = HERE / 'results'


def build():
    report = json.loads((RESULTS / (NAME + '.json')).read_text())
    require(report['official_tests_opened'] is False and report['production_certified'] is False
            and all(report[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe_l3', 'paired_superiority')),
            'research claim or test seal changed')
    require(summarize(report['cells'], report['dataset_ids']) == report['summary'],
            'committed matrix reconciliation changed')
    for reference in report['immutable_references']:
        require(sha256(Path(reference['path'])) == reference['sha256'],
                'immutable publication reference changed')
    sources = {name: sha256(HERE / name) for name in (
        'publish_s3_matched.py', 'publish_s3_matched_figure.py', 'publish_s3_matched_manifest.py',
        'manifest.py', 'score.py')}
    require(sources['publish_s3_matched.py'] == report['source_sha256'], 'publisher source changed')
    require(sha256(RESULTS / 's3-data.lock.json') == report['s3_data_lock_sha256'], 'rights lock changed')
    artifacts = {}
    for suffix in ('.json', '.schema.json', '.csv', '.md', '.svg', '.pdf'):
        path = RESULTS / (NAME + suffix)
        artifacts[path.name] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
    return {'format': 'dope-s3-matched-validation-publication-manifest', 'version': 1,
            'scope': report['scope'], 'artifacts': artifacts, 'publisher_sources': sources,
            's3_data_lock_sha256': report['s3_data_lock_sha256'],
            'immutable_references': report['immutable_references'],
            'matrix': {'datasets': 6, 'fit_seed_count': 1, 'sample_seed_count': 3,
                       'size_count': 2, 'logical_cells': 288, 'planned_physical_cells': 270,
                       'successful_physical_cells': report['successful_physical_sample_cells'],
                       'logical_status_counts': report['logical_status_counts']},
            'campaign_complete': False, 'official_tests_opened': False, 'production_certified': False,
            'mfs_v2': None, 'ptf_v1': None, 'release_safe_l3': None, 'paired_superiority': None,
            'counts_as_dope_win': False}


def main():
    document = build()
    path = RESULTS / (NAME + '.manifest.json')
    path.write_text(json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + '\n')
    path.with_suffix('.schema.json').write_text(json.dumps(schema(document), sort_keys=True, indent=2) + '\n')
    print(path)


if __name__ == '__main__':
    main()
