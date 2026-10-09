"""Rendering of the review tables from the committed panels; nothing is written."""

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.benchmark.review_fixes.receipt_tables import configured_tables  # noqa: E402

RESULTS = ROOT / 'research/benchmark/results'


def load(name):
    spec = importlib.util.spec_from_file_location(f'paper_{name}', ROOT / f'docs/whitepaper/scripts/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TABSYN = load('review_tabsyn')
CHECK = load('check_paper')


def panel(directory):
    return json.loads((RESULTS / directory / 'panel.json').read_text())


def body_rows(tex):
    body = tex.split('\\endhead\n', 1)[1].split('\\bottomrule', 1)[0]
    return [line for line in body.splitlines() if line.endswith('\\\\')]


class TabSynTableTests(unittest.TestCase):
    def test_each_matched_row_is_followed_by_dope_on_the_same_lineages(self):
        tex = TABSYN.render_tabsyn(panel('review-fixes-tabsyn-v1'))
        rows = body_rows(tex)
        self.assertEqual(len(rows), 24)
        for start in range(0, 24, 4):
            labels = [row.split(' & ')[2] for row in rows[start:start + 4]]
            self.assertEqual(labels, ['TabSyn, own lineages', 'TabSyn, matched', 'DOPE, same lineages',
                                      'DOPE $-$ TabSyn'])
            counts = {row.split(' & ')[3] for row in rows[start + 1:start + 4]}
            self.assertEqual(len(counts), 1)
        self.assertIn('200 VAE epochs and 1{,}000 diffusion epochs', tex)
        self.assertIn('Holm family: three tests per size', tex)
        self.assertEqual(CHECK.table_family_failures(CHECK.strip_tex_comments(tex)), [])

    def test_missing_or_unmatched_dope_row_is_refused(self):
        payload = panel('review-fixes-tabsyn-v1')
        missing = copy.deepcopy(payload)
        missing['levels'] = [row for row in missing['levels']
                             if not (row['method'] == 'DOPE' and row['size'] == 4 and row['auditor'] == 'mlp')]
        with self.assertRaisesRegex(ValueError, 'lacks a paired DOPE row'):
            TABSYN.render_tabsyn(missing)
        shifted = copy.deepcopy(payload)
        next(row for row in shifted['levels'] if row['method'] == 'DOPE')['n'] += 1
        with self.assertRaisesRegex(ValueError, 'same lineages'):
            TABSYN.render_tabsyn(shifted)


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
