"""Anonymity failures and generated contract conformance examples."""
import json
import unittest
from pathlib import Path

import check_paper
from research.benchmark.mfs_v3_score import _v3_contract


class PackagingTests(unittest.TestCase):
    def test_body_limit_counts_text_before_references_on_the_same_page(self):
        first_ten = "body\f" * 10
        self.assertEqual(check_paper.presentation_failures(first_ten + 'R E F E R E N C E S\n'), [])
        self.assertIn('main body before References exceeds 10 pages',
                      check_paper.presentation_failures(first_ten + 'last body paragraph\nReferences\n'))
        self.assertIn('main PDF has no References heading',
                      check_paper.presentation_failures(first_ten))

    def test_frozen_negation_limit_rejects_six_but_allows_five_per_thousand(self):
        for count in (5, 6):
            text = 'word ' * (999 - count) + ' not ' * count + '\nReferences\n'
            self.assertEqual(len(text.split()), 1000)
            failed = 'main PDF negation density exceeds 5 per 1000 words'
            if count == 5:
                self.assertNotIn(failed, check_paper.presentation_failures(text))
            else:
                self.assertIn(failed, check_paper.presentation_failures(text))

    def test_nonprinting_pdf_glyphs_cannot_dilute_negation_density(self):
        text = 'word ' * 993 + ' not ' * 6 + '\nReferences\n'
        text = ' \x01 ' * 200 + text
        self.assertIn('main PDF negation density exceeds 5 per 1000 words',
                      check_paper.presentation_failures(text))

    def test_name_and_url_leaks_are_case_insensitive(self):
        for text in ('Jepson Taylor', 'alton alexander', 'NEVERHUMAN Research',
                     'https://github.com/neverhuman/dope', 'https://example.com/jepsontaylor'):
            self.assertTrue(check_paper.anonymous_leaks(text))
        self.assertFalse(check_paper.anonymous_leaks('Anonymous; https://github.com/EpistasisLab/pmlb'))

    def test_conformance_table_contains_every_contract_weight_and_floor(self):
        contract = _v3_contract()
        generated = check_paper.ROOT / 'generated'
        text = (generated / 'mfs-v3-conformance.tex').read_text()
        for key, weight in contract['master_fitness']['weights'].items():
            self.assertIn(key.replace('_', r'\_') + ' = ' + str(weight), text)
        self.assertIn('$q=' + str(contract['release_gates']['near_copy_quantile']) + '$', text)
        self.assertIn('linear interpolation', text)
        self.assertIn('Median linear MSE of three', text)
        self.assertIn(r'\newcommand{\MfsVThreePerfect}{100}', (generated / 'mfs-v3-examples.tex').read_text())
        supplement = (check_paper.ROOT / 'supplement.tex').read_text()
        self.assertIn(r'\input{generated/mfs-v3-conformance.tex}', supplement)
        self.assertIn(r'\input{generated/mfs-v3-examples.tex}', supplement)
        self.assertNotIn('100.0001', supplement)


if __name__ == '__main__':
    unittest.main()
