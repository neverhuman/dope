"""Bind the fit-only publication and deterministic outputs to frozen custody."""
from __future__ import annotations

import json
from pathlib import Path

from .publish_dope_population_fits import NAME, RESULTS, build as build_report, require, schema, summarize, tables
from .publish_dope_population_fits_figure import render
from .score import sha256


def build():
    report = json.loads((RESULTS/(NAME+'.json')).read_text())
    require(build_report() == report and summarize(report['cells']) == report['summary'], 'fit publication differs from frozen custody')
    require(json.loads((RESULTS/(NAME+'.schema.json')).read_text()) == schema(report), 'fit schema differs')
    table, markdown = tables(report)
    require((RESULTS/(NAME+'.csv')).read_text() == table and (RESULTS/(NAME+'.md')).read_text() == markdown, 'fit tables differ')
    for kind in ('svg', 'pdf'): require((RESULTS/(NAME+'.'+kind)).read_bytes() == render(report, kind), 'fit figure differs')
    here = Path(__file__).parent
    sources = {n: sha256(here/n) for n in ('publish_dope_population_fits.py', 'publish_dope_population_fits_figure.py',
        'publish_dope_population_fits_manifest.py', 'publish_s3_forest.py', 'publish_s3_matched.py', 'manifest.py', 'score.py')}
    artifacts = {NAME+s: {'sha256': sha256(RESULTS/(NAME+s)), 'bytes': (RESULTS/(NAME+s)).stat().st_size}
                 for s in ('.json','.schema.json','.csv','.md','.svg','.pdf')}
    return {'format': 'dope-s3-population-gpu-fit-publication-manifest', 'version': 1,
            'scope': report['scope'], 'artifacts': artifacts, 'publisher_sources': sources,
            's3_data_lock_sha256': report['s3_data_lock_sha256'], 'immutable_references': report['immutable_references'],
            'closed_fit_cells': 400, 'status_counts': report['status_counts'],
            'campaign_complete': False, 'global_family_selected': False, 'official_tests_opened': False,
            'production_certified': False, 'mfs_v2': None, 'ptf_v1': None, 'release_safe_l3': None,
            'paired_superiority': None, 'counts_as_dope_win': False}


if __name__ == '__main__':
    manifest = build(); path = RESULTS/(NAME+'.manifest.json')
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False)+'\n')
    path.with_suffix('.schema.json').write_text(json.dumps(schema(manifest), sort_keys=True, indent=2)+'\n')
    print(path)
