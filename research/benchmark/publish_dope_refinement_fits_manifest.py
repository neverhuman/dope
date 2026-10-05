"""Regenerate and bind the complete training ledger and deterministic figures."""
import argparse
import json
from pathlib import Path

from . import publish_dope_refinement_fits as pub
from .publish_dope_refinement_fits_figure import render


def build(root):
    report = json.loads((root / (pub.NAME + '.json')).read_bytes())
    pub.guard.require(pub.build() == report, 'publication differs from closed custody')
    csv_text, markdown = pub.tables(report)
    pub.guard.require((root / (pub.NAME + '.csv')).read_text() == csv_text
        and (root / (pub.NAME + '.md')).read_text() == markdown
        and json.loads((root / (pub.NAME + '.schema.json')).read_bytes()) == pub.schema(report),
        'publication tables or schema differ')
    for kind in ('svg', 'pdf'):
        pub.guard.require((root / (pub.NAME + '.' + kind)).read_bytes() == render(report, kind), 'figure differs')
    return dict(format='dope-refinement-population-fit-publication-manifest', version=1,
        artifacts={pub.NAME + s: dict(sha256=pub.sha256(root / (pub.NAME + s)), bytes=(root / (pub.NAME + s)).stat().st_size)
            for s in ('.json', '.csv', '.md', '.svg', '.pdf', '.schema.json')},
        publisher_sources={n: pub.sha256(Path(__file__).with_name(n)) for n in (
            'publish_dope_refinement_fits.py', 'publish_dope_refinement_fits_figure.py',
            'publish_dope_refinement_fits_manifest.py', 'arf_runtime_guard.py', 'manifest.py', 'score.py', 'publish_s3_matched.py')},
        immutable_references=report['immutable_references'], status_counts=report['status_counts'],
        s3_data_lock_sha256=report['s3_data_lock_sha256'], closed_fit_cells=200,
        raw_rows_samples_weights_or_detailed_logs_committed=False, **pub.GATES)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--results-dir', type=Path, default=Path(__file__).with_name('results'))
    root = parser.parse_args().results_dir; value = build(root); path = root / (pub.NAME + '.manifest.json')
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    path.with_suffix('.schema.json').write_text(json.dumps(pub.schema(value), sort_keys=True, indent=2) + '\n')
    print(path)


if __name__ == '__main__': main()
