"""Pin rights-safe confirmation outputs to their immutable scratch evidence."""
import argparse
import json
from pathlib import Path

from research.benchmark import publish_dope_refinement_confirmation as report
from research.benchmark.publish_dope_refinement_confirmation_figure import REPORT_SHA256


def build(root):
    value = report.bound(root / (report.NAME + '.json'), REPORT_SHA256)
    csv_text, markdown = report.tables(value)
    report.require((root / (report.NAME + '.csv')).read_text() == csv_text
                   and (root / (report.NAME + '.md')).read_text() == markdown, 'tables differ from report')
    return dict(format='dope-rights-safe-refinement-confirmation-manifest', version=1,
        files={report.NAME + suffix: report.sha(root / (report.NAME + suffix))
               for suffix in ('.json', '.csv', '.md', '.svg', '.pdf', '.schema.json')},
        source_files={name: report.sha(Path(__file__).with_name(name)) for name in (
            'publish_dope_refinement_confirmation.py', 'publish_dope_refinement_confirmation_figure.py',
            'publish_dope_refinement_confirmation_manifest.py')},
        receipt_lock=dict(path=str(report.ROOT / 'receipt-lock-v1.json'), sha256=report.RECEIPTS),
        fit_receipt_lock=dict(path=str(report.FIT / 'receipt-lock-v1.json'), sha256=report.FIT_RECEIPTS),
        all_confirmation_fit_cells_accounted=True, confirmation_complete=True,
        raw_rows_samples_weights_or_detailed_logs_committed=False,
        official_tests_opened=False, mfs_v2=None, ptf_v1=None, release_safe=None, superiority=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=Path(__file__).with_name('results'))
    args = parser.parse_args()
    (args.results_dir / (report.NAME + '.manifest.json')).write_text(
        json.dumps(build(args.results_dir), sort_keys=True, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__': main()
