"""Regenerate publication artifacts from committed native metadata only."""
import argparse
import hashlib
import json
from pathlib import Path

from . import publish_sdv_population_native as m
from . import publish_sdv_population_native_figure as figure
from .publish_s3_matched import schema
from .score import sha256


def generate(directory, publication_sha256):
    directory = Path(directory)
    m.accounting.require(type(publication_sha256) is str
        and m.re.fullmatch('[0-9a-f]{64}', publication_sha256), 'external publication digest required')
    data = (directory / (m.NAME + '.json')).read_bytes()
    m.accounting.require(hashlib.sha256(data).hexdigest() == publication_sha256,
                         'native publication digest differs')
    report = json.loads(data)
    m.accounting.require(all(report[k] is v for k, v in m.GATES.items()),
                         'native manifest acquired a gated claim')
    for name, expected in report['publisher_sources'].items():
        m.accounting.require(Path(name).name == name and sha256(Path(m.__file__).with_name(name)) == expected,
                             'native publisher source changed')
    (directory / (m.NAME + '.csv')).write_text(m.table(report))
    (directory / (m.NAME + '.md')).write_text(m.markdown(report))
    for kind in ('svg', 'pdf'):
        (directory / (m.NAME + '.' + kind)).write_bytes(figure.render(report, kind))
    (directory / (m.NAME + '.schema.json')).write_text(json.dumps(schema(report), sort_keys=True, indent=2) + '\n')
    files = [directory / (m.NAME + '.' + suffix) for suffix in ('json', 'csv', 'md', 'svg', 'pdf', 'schema.json')]
    manifest = {'format': 'dope-sdv-population-native-publication-manifest', 'version': 1,
        'source_locks': report['source_locks'],
        'artifacts': {p.name: {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in files},
        'publisher_sources': report['publisher_sources'] | {
            Path(__file__).name: sha256(Path(__file__)),
            Path(figure.__file__).name: sha256(Path(figure.__file__)),
            'publish_s3_matched.py': sha256(Path(m.__file__).with_name('publish_s3_matched.py'))},
        'immutable_references': [
            {'path': str(m.predecessor.ROOT / filename), 'sha256': report['source_locks'][key]}
            for filename, key in (('round.lock.json', 'round'), ('receipt-lock-v1.json', 'receipts'),
                                  ('reconciliation-v1.json', 'reconciliation'))],
        'all_frozen_cells_accounted': True, **m.GATES}
    path = directory / (m.NAME + '.manifest.json')
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    path.with_name(m.NAME + '.manifest.schema.json').write_text(
        json.dumps(schema(manifest), sort_keys=True, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=m.RESULTS)
    parser.add_argument('--publication-sha256', required=True)
    args = parser.parse_args()
    generate(args.directory, args.publication_sha256)
