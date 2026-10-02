"""Bind complete rights-safe ForestDiffusion outputs to immutable receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .score import sha256

HERE = Path(__file__).parent
RESULTS = HERE / 'results'
NAME = 'pilot24-forestdiffusion-native'


def build():
    report_path = RESULTS / (NAME + '.json')
    report = json.loads(report_path.read_text())
    if report['official_tests_opened'] is not False or any(report[k] is not None for k in ('mfs_v2', 'ptf_v1', 'release_safe')):
        raise ValueError('pilot test seal or score changed')
    references = {}
    def add(item):
        path, expected = item['path'], item['sha256']
        if path in references and references[path] != expected:
            raise ValueError('receipt identity collision')
        if sha256(Path(path)) != expected:
            raise ValueError('immutable publication reference changed')
        references[path] = expected
    for row in report['fit_attempts'] + report['physical_sample_receipts'] + report['cells']:
        add(row)
    for row in report['matched_references']['cells']:
        add(row['reference_receipt'])
    for item in report['rounds'].values():
        add(item)
    add(report['completion'])
    add(report['matched_references']['report'])
    for row in report['native_selections']:
        add(row['receipt'])
    sources = {}
    for name in ('publish_forestdiffusion_native.py', 'publish_forestdiffusion_native_figure.py',
                 'publish_forestdiffusion_manifest.py', 'publish_forestdiffusion_source.py',
                 'forestdiffusion_adapter.py', 'publish_native_neural_fivefit.py', 'score.py', 'manifest.py',
                 'forestdiffusion-source.lock.json'):
        sources[name] = sha256(HERE / name)
    if sources['publish_forestdiffusion_native.py'] != report['source_sha256']:
        raise ValueError('publisher source identity changed')
    artifacts = {}
    for suffix in ('.json', '.schema.json', '.csv', '.md', '.svg', '.pdf'):
        p = RESULTS / (NAME + suffix)
        artifacts[p.name] = {'sha256': sha256(p), 'bytes': p.stat().st_size}
    return {'format': 'dope-forestdiffusion-publication-manifest', 'version': 1,
            'scope': 'complete frozen single-fit validation pilot only',
            'artifacts': artifacts, 'publisher_sources': sources,
            'immutable_references': [{'path': p, 'sha256': h} for p, h in sorted(references.items())],
            'matrix': {'native_attempts': 24, 'physical_samples': 12, 'logical_samples': 108,
                       'matched_prior_reference_cells': 120},
            'campaign_complete': False, 'official_tests_opened': False, 'production_certified': False,
            'mfs_v2': None, 'ptf_v1': None, 'release_safe': None}


def schema(document):
    sha = {'type': 'string', 'pattern': '^[0-9a-f]{64}$'}
    def obj(properties):
        return {'type': 'object', 'additionalProperties': False, 'required': sorted(properties), 'properties': properties}
    artifacts = obj({name: obj({'sha256': sha, 'bytes': {'type': 'integer', 'minimum': 0}}) for name in document['artifacts']})
    reference = obj({'path': {'type': 'string', 'minLength': 1}, 'sha256': sha})
    result = obj({
        'format': {'const': document['format']}, 'version': {'const': 1}, 'scope': {'const': document['scope']},
        'artifacts': artifacts, 'publisher_sources': obj({name: sha for name in document['publisher_sources']}),
        'immutable_references': {'type': 'array', 'minItems': len(document['immutable_references']),
                                 'maxItems': len(document['immutable_references']), 'items': reference, 'uniqueItems': True},
        'matrix': {'const': document['matrix']}, 'campaign_complete': {'const': False},
        'official_tests_opened': {'const': False}, 'production_certified': {'const': False},
        'mfs_v2': {'type': 'null'}, 'ptf_v1': {'type': 'null'}, 'release_safe': {'type': 'null'},
    })
    result['$schema'] = 'https://json-schema.org/draft/2020-12/schema'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=RESULTS / (NAME + '.manifest.json'))
    parser.add_argument('--schema', type=Path)
    args = parser.parse_args()
    document = build()
    args.output.write_text(json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + '\n')
    if args.schema:
        args.schema.write_text(json.dumps(schema(document), sort_keys=True, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
