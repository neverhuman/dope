"""Rebuild every public byte-analysis artifact from immutable scratch proof."""
from __future__ import annotations

import json
from pathlib import Path

from .publish_artifact_bytes import NAME, RESULTS, build as build_report, require, schema, tables
from .publish_artifact_bytes_figure import render
from .score import sha256


def build():
    report = json.loads((RESULTS / (NAME + '.json')).read_text())
    require(report == build_report(), 'byte publication differs from frozen proof')
    require(json.loads((RESULTS / (NAME + '.schema.json')).read_text()) == schema(report),
            'byte publication schema differs')
    csv, markdown = tables(report)
    require((RESULTS / (NAME + '.csv')).read_text() == csv
            and (RESULTS / (NAME + '.md')).read_text() == markdown, 'byte publication tables differ')
    for kind in ('svg', 'pdf'):
        require((RESULTS / (NAME + '.' + kind)).read_bytes() == render(report, kind),
                'byte publication figure differs')
    here = Path(__file__).parent
    sources = {name: sha256(here / name) for name in ('publish_artifact_bytes.py',
        'publish_artifact_bytes_figure.py', 'publish_artifact_bytes_manifest.py', 'research_container.py',
        'publish_s3_matched.py', 'manifest.py', 'score.py')}
    return {'format': 'dope-artifact-byte-feasibility-manifest', 'version': 1,
        'publisher_sources': sources,
        'artifacts': {NAME + suffix: {'bytes': (RESULTS / (NAME + suffix)).stat().st_size,
            'sha256': sha256(RESULTS / (NAME + suffix))} for suffix in
            ('.json', '.schema.json', '.csv', '.md', '.svg', '.pdf')},
        'immutable_references': report['immutable_references'],
        'candidate_count': 32, 'parent_fit_cells': 16, 'campaign_complete': False,
        'generator_validation_complete': False, 'production_certified': False,
        'official_tests_opened': False, 'counts_as_dope_win': False,
        'mfs_v2': None, 'ptf_v1': None, 'release_safe_l3': None}


if __name__ == '__main__':
    manifest = build(); path = RESULTS / (NAME + '.manifest.json')
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    path.with_suffix('.schema.json').write_text(json.dumps(schema(manifest), sort_keys=True, indent=2) + '\n')
