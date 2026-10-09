"""Rendering of the review tables from the committed panels; nothing is written."""

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.benchmark.review_fixes.receipt_tables import configured_tables  # noqa: E402

RESULTS = ROOT / 'research/benchmark/results'


def panel(directory):
    return json.loads((RESULTS / directory / 'panel.json').read_text())


class ConfiguredReceiptTableTests(unittest.TestCase):
    def test_every_arf_row_names_its_configuration(self):
        tables = configured_tables(panel('review-fixes-receipts-v1'))
        self.assertEqual(set(tables), {'review-tost.tex', 'review-rmse.tex',
                                       'review-fidelity-stored.tex', 'review-seeds-existing.tex'})
        for name, tex in tables.items():
            with self.subTest(table=name):
                self.assertIn('Configuration', tex.split('\\midrule')[0])
                arf = [line for line in tex.splitlines() if 'ARF' in line.split(' & ')[:3]]
                self.assertTrue(any('author\\_default' in line for line in arf))
                self.assertTrue(any('native\\_selected' in line for line in arf))

    def test_rows_that_cannot_be_told_apart_are_refused(self):
        payload = panel('review-fixes-receipts-v1')
        for key in ('paired', 'rmse', 'fidelity_stored', 'existing_multiseed'):
            with self.subTest(block=key):
                broken = copy.deepcopy(payload)
                row = next(item for item in broken[key] if item['method'] == 'ARF'
                           and item.get('auditor', 'catboost') == 'catboost')
                broken[key].append(dict(row))
                with self.assertRaisesRegex(ValueError, 'same identity'):
                    configured_tables(broken)


if __name__ == '__main__':
    unittest.main()
