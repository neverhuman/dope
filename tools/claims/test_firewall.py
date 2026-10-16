"""Seeded publication violations must fail the actual public registry."""
import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'docs/whitepaper/scripts'))
import claim_registry as C
from tools.claims import render, registry


class FirewallTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / 'paper/generated/headline-results.json').read_text())
        self.status = json.loads((ROOT / 'evidence/status.json').read_text())
        self.auditors = json.loads((ROOT / 'research/transfer/registries/auditors.json').read_text())

    def failures(self):
        return '\n'.join(C.result_failures(self.data, self.auditors, self.status))

    def test_current_status_labels_are_consistent(self):
        self.assertEqual(self.failures(), '')

    def test_seeded_bad_claim_phrases_and_unproven_knowledge_fail(self):
        for phrase in ('privacy-preserving', 'minimum', 'beats TabSyn', 'validated across Kumo', '1.9 KiB cold closure', 'preserves knowledge'):
            with self.subTest(phrase=phrase):
                self.assertTrue(C.lint_public_text(phrase, 'fixture'))

    def test_hand_typed_numbers_fail_but_cli_and_citation_numbers_are_not_claims(self):
        self.assertTrue(C.lint_public_text('Retention is 0.898.', 'fixture'))
        self.assertFalse(C.lint_public_text('Run `dope sample --rows 200`. T90/D80.', 'fixture'))
        self.assertTrue(C.lint_public_text('Result is 9 out of 10.', 'fixture'))
        self.assertTrue(C.lint_public_text('Retention is `0.898`.', 'fixture'))
        self.assertTrue(C.lint_public_text('Payload is `1.9 KiB`.', 'fixture'))
        self.assertTrue(C.lint_public_text('Claim: `privacy-preserving`.', 'fixture'))
        self.assertTrue(C.lint_public_text(r'\newcommand{\Result}{0.898}', 'fixture'))

    def test_unqualified_auditor_incomplete_cohort_and_missing_gates_fail(self):
        row = self.data['results'][0]
        row['publication_role'] = 'headline'
        failures = self.failures()
        for expected in ('unqualified headline auditor', 'incomplete-cohort', 'frozen protocol'):
            self.assertIn(expected, failures)

    def test_non_sealed_final_and_mixed_cohort_claims_fail(self):
        row = self.data['results'][0]
        row['final'] = True
        row['pooled_cohort_ids'] = [row['cohort']['id'], 'unrelated-small-cohort']
        self.assertIn('requires sealed-test', self.failures())
        self.assertIn('mixed-cohort', self.failures())

    def test_historical_source_cannot_be_relabelled_as_sealed(self):
        row = self.data['results'][0]
        row.update(final=True, evaluation_split='sealed_test', evidence_status='sealed_test')
        self.status['results'][0]['evidence_status'] = 'sealed_test'
        self.assertIn('historical source cannot be relabeled', self.failures())

    def test_renderer_refuses_synthetic_fixture(self):
        import tempfile
        local_tmp = ROOT / 'target/claims-tests'
        local_tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local_tmp) as directory:
            path = Path(directory) / 'fixture.json'
            for value in (dict(self.data, citable=False), dict(self.data, fixture=True)):
                path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, 'fixture'):
                    render.load_results(path)

    def test_figure_and_card_drift_is_detected_by_same_gate(self):
        from unittest.mock import patch
        with patch.object(render, 'expected_outputs', return_value={'paper/generated/hero-frontier.png': b'seeded drift'}):
            self.assertTrue(any('figure hash drift' in f for f in C.public_failures()))


class RegistryTests(unittest.TestCase):
    def test_all_schemas_and_every_existing_lock_validate(self):
        self.assertEqual(registry.validate(), [])
        for path in (ROOT / 'research/transfer/locks').glob('*.json'):
            lock = json.loads(path.read_text())
        index = json.loads((ROOT / 'research/transfer/registries/lock-states.json').read_text())
        self.assertEqual(len(index['entries']), len(list((ROOT / 'research/transfer/locks').glob('*.json'))))
        self.assertTrue(all(row['qualification']['state'] == 'specified' for row in index['entries']))

    def test_no_state_skip_or_transition_without_evidence(self):
        entry = copy.deepcopy(json.loads((ROOT / 'research/transfer/registries/auditors.json').read_text())['entries'][0])
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            registry.advance(entry, 'qualified', 'a' * 64)
        with self.assertRaisesRegex(ValueError, 'evidence'):
            registry.advance(entry, 'source_verified', 'a' * 64)
        entry.update(source_revision='pinned-revision', code_sha256='b' * 64)
        advanced = registry.advance(entry, 'source_verified', 'a' * 64)
        self.assertEqual(advanced['qualification']['state'], 'source_verified')
        self.assertEqual(entry['qualification']['state'], 'specified')
        self.assertEqual(advanced['qualification']['admission_status'], 'unqualified')

    def test_migration_is_idempotent_and_keeps_historical_lock_bytes(self):
        import subprocess
        paths = sorted((ROOT / 'research/transfer/locks').glob('*.json'))
        original = {p: p.read_bytes() for p in paths}
        index = ROOT / 'research/transfer/registries/lock-states.json'
        before = index.read_bytes()
        subprocess.run([sys.executable, str(ROOT / 'tools/claims/registry.py'), '--migrate'], check=True, capture_output=True)
        self.assertEqual(index.read_bytes(), before)
        self.assertEqual({p: p.read_bytes() for p in paths}, original)


if __name__ == '__main__':
    unittest.main()
