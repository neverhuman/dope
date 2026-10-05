"""Bind complete common outcomes to frozen scratch and committed references."""
import argparse
import json
from pathlib import Path

from . import publish_dope_refinement_population as pub
from .publish_dope_refinement_population_figure import render


def build(root):
    path = root / (pub.NAME + '.json'); report = pub.guard.decode(path.read_bytes())
    pub.validate_report(report)
    pins = report['source_locks']
    pub.guard.require(pub.digest(pub.build(pins['receipts'], pins['reconciliation'])) == pub.digest(report),
                      'common publication differs from closed evidence')
    csv_text, markdown = pub.tables(report)
    pub.guard.require((root / (pub.NAME + '.csv')).read_text() == csv_text
        and (root / (pub.NAME + '.md')).read_text() == markdown
        and json.loads((root / (pub.NAME + '.schema.json')).read_bytes()) == pub.fits.schema(report),
        'common table/schema replay differs')
    for kind in ('svg', 'pdf'):
        pub.guard.require((root / (pub.NAME + '.' + kind)).read_bytes() == render(report, kind), 'common figure replay differs')
    sources = ('publish_dope_refinement_population.py', 'publish_dope_refinement_population_figure.py',
        'publish_dope_refinement_population_manifest.py', 'publish_dope_refinement_fits.py', 'arf_runtime_guard.py',
        'publish_dope_refinement_discovery.py', 'publish_s3_forest.py', 'publish_s3_matched.py', 'manifest.py', 'score.py')
    return dict(format='dope-complete-refinement-population-publication-manifest', version=1,
        artifacts={pub.NAME + s: dict(sha256=pub.fits.sha256(root / (pub.NAME + s)), bytes=(root / (pub.NAME + s)).stat().st_size)
            for s in ('.json', '.csv', '.md', '.svg', '.pdf', '.schema.json')},
        publisher_sources={n: pub.fits.sha256(Path(__file__).with_name(n)) for n in sources},
        source_locks=report['source_locks'], matched_reference=report['matched_reference'],
        fit_ledger_reference=report['fit_ledger_reference'], logical_validation_cells=1200,
        raw_rows_samples_weights_or_detailed_logs_committed=False, **pub.fits.GATES)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--results-dir', type=Path, default=Path(__file__).with_name('results'))
    root = parser.parse_args().results_dir; value = build(root); path = root / (pub.NAME + '.manifest.json')
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    path.with_suffix('.schema.json').write_text(json.dumps(pub.fits.schema(value), sort_keys=True, indent=2) + '\n')
    print(path)


if __name__ == '__main__': main()
