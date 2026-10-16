"""Seeded publication violations must fail the actual public registry."""
import copy
import json
import sys
import unittest
from contextlib import contextmanager
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

    @contextmanager
    def public_fixture(self):
        import shutil
        import tempfile
        local_tmp = ROOT / 'target/claims-tests'
        local_tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local_tmp) as directory:
            root = Path(directory)
            for name in ('paper', 'schemas', 'research/transfer/registries',
                         'research/transfer/locks', 'research/transfer/manifests'):
                shutil.copytree(ROOT / name, root / name)
            inputs = {'README.md', 'Cargo.toml', 'pyproject.toml', 'CITATION.cff',
                      'contracts/tdl-contract-v1.json', 'evidence/status.json', C.REGISTRY['sources']['headline']}
            inputs.update(pin['path'] for pin in self.data['demo_notes']['pins'])
            for name in inputs:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, path)
            yield root

    def test_current_status_labels_are_consistent(self):
        self.assertEqual(self.failures(), '')

    def test_seeded_bad_claim_phrases_and_unproven_knowledge_fail(self):
        for phrase in ('privacy-preserving', 'minimum', 'beats TabSyn', 'validated across Kumo', '1.9 KiB cold closure', 'preserves knowledge'):
            with self.subTest(phrase=phrase):
                self.assertTrue(C.lint_public_text(phrase, 'fixture'))

    def test_hipaa_deidentified_and_mdl_claim_fixtures_fail(self):
        for phrase in ('HIPAA-safe artifacts.', 'The artifact is hipaa compliant.',
                       'These artifacts are de-identified.', 'These artifacts are deidentified.',
                       'These artifacts are de identified.', 'The MDL score is favorable.'):
            with self.subTest(phrase=phrase):
                self.assertTrue(any('forbidden claim' in failure for failure in C.lint_public_text(phrase, 'fixture')))
                self.data['demo_notes']['paragraphs'] = [phrase]
                self.assertIn('demo_notes[0]: forbidden claim', self.failures())

    def test_direct_negation_does_not_hide_an_affirmative_claim(self):
        for disclaimer in ('There is no HIPAA de-identification.',
                           'Empirical attacks are not HIPAA de-identification.',
                           'The artifact is not HIPAA-safe.', 'The artifact is not HIPAA compliant.',
                           'The artifact is not de-identified.', 'This score is not MDL.'):
            with self.subTest(disclaimer=disclaimer):
                self.assertEqual(C.lint_public_text(disclaimer, 'fixture'), [])
                for claim in ('It is HIPAA-safe.', 'It is de-identified.', 'It is MDL.'):
                    self.assertTrue(C.lint_public_text(disclaimer + ' ' + claim, 'fixture'))
        self.assertTrue(C.lint_public_text('The artifact is not only HIPAA-safe but also de-identified.', 'fixture'))
        self.assertTrue(C.lint_public_text('There is no HIPAA risk.', 'fixture'))

    def test_nested_paper_claim_fixtures_fail_the_public_gate(self):
        with self.public_fixture() as root:
            (root / 'paper/archive/fixture.tex').write_text('Historical privacy-preserving claim.')
            self.assertEqual(C.public_failures(root, check_render=False), [])
            # Unknown generated files are prose too; only authenticated renderer outputs are exempt.
            for name in ('paper/sections/deep/fixture.tex', 'paper/generated/fixture.tex'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                for phrase in ('HIPAA-safe artifacts.', 'These artifacts are de-identified.',
                               'The MDL score is favorable.', 'This is privacy-preserving.',
                               'Retention is 0.898.'):
                    with self.subTest(path=name, phrase=phrase):
                        path.write_text(phrase)
                        self.assertTrue(any(failure.startswith(name + ':')
                                            for failure in C.public_failures(root, check_render=False)))
                path.write_text('No measured claim is available.')
            self.assertEqual(C.public_failures(root, check_render=False), [])

    def test_demo_metadata_drift_and_contradictions_fail_the_public_gate(self):
        with self.public_fixture() as root:
            data_path = root / 'paper/generated/headline-results.json'
            for original, replacement, expected in (
                    ('CC BY 4.0', 'CC BY 3.0', 'license'),
                    ('DOI 10.24432/C5VW2C', 'DOI 10.24432/WRONG', 'DOI'),
                    ('three positional features', 'four positional features', 'positional feature count'),
                    ('CC BY 4.0', 'CC BY 4.0; CC BY 3.0', 'license'),
                    ('DOI 10.24432/C5VW2C', 'DOI 10.24432/C5VW2C; DOI 10.24432/WRONG', 'DOI'),
                    ('three positional features', 'three positional features and four positional features', 'positional feature count'),
                    ('CC BY 4.0', '', 'license'),
                    ('DOI 10.24432/C5VW2C', '', 'DOI'),
                    ('three positional features', '', 'positional feature count')):
                with self.subTest(replacement=replacement):
                    data = copy.deepcopy(self.data)
                    data['demo_notes']['paragraphs'] = [p.replace(original, replacement)
                                                        for p in data['demo_notes']['paragraphs']]
                    data_path.write_text(json.dumps(data))
                    failures = C.public_failures(root, check_render=False)
                    self.assertTrue(any('demo_notes: ' + expected in failure for failure in failures), failures)

    def test_demo_metadata_is_compared_with_sources_and_requires_pins(self):
        import hashlib
        with self.public_fixture() as root:
            for relative, field, value, expected in (
                    ('docs/whitepaper/receipts/native-byte-conformance-v1.json', 'license', 'CC-BY-3.0', 'license'),
                    ('docs/whitepaper/receipts/native-byte-conformance-v1.json', 'doi', '10.24432/WRONG', 'DOI'),
                    ('examples/readme-regression/INSPECT.txt', 'positional_features', 4, 'positional feature count')):
                with self.subTest(field=field):
                    path = root / relative
                    original = path.read_bytes()
                    source = json.loads(original)
                    (source['source_data'] if 'source_data' in source else source)[field] = value
                    path.write_text(json.dumps(source))
                    data = copy.deepcopy(self.data)
                    for pin in data['demo_notes']['pins']:
                        if pin['path'] == relative:
                            pin['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                    (root / 'paper/generated/headline-results.json').write_text(json.dumps(data))
                    self.assertTrue(any('demo_notes: ' + expected in failure
                                        for failure in C.public_failures(root, check_render=False)))
                    path.write_bytes(original)
            data = copy.deepcopy(self.data)
            data['demo_notes']['pins'] = []
            (root / 'paper/generated/headline-results.json').write_text(json.dumps(data))
            self.assertTrue(any('metadata source pin' in failure
                                for failure in C.public_failures(root, check_render=False)))

    def test_contract_macros_need_reproduced_bytes_before_lint_exemption(self):
        with self.public_fixture() as root:
            generator = root / 'paper/scripts/contract_macros.py'
            generator.parent.mkdir(parents=True, exist_ok=True)
            generator.write_text('def render(contract):\n    return "\\\\newcommand{\\\\ContractVersion}{%d}\\n" % contract["version"]\n')
            path = root / 'paper/generated/contract-macros.tex'
            path.write_text('\\newcommand{\\ContractVersion}{1}\n')
            self.assertEqual(C.public_failures(root, check_render=False), [])
            path.write_text('\\newcommand{\\ContractVersion}{2}\n')
            self.assertTrue(any('contract macro drift' in failure
                                for failure in C.public_failures(root, check_render=False)))
            path.write_text('\\newcommand{\\ContractVersion}{1}\n')
            contract_path = root / 'contracts/tdl-contract-v1.json'
            contract = json.loads(contract_path.read_text())
            contract['version'] = 2
            contract_path.write_text(json.dumps(contract))
            self.assertTrue(any('contract macro drift' in failure
                                for failure in C.public_failures(root, check_render=False)))
            generator.rename(generator.with_suffix('.disabled'))
            self.assertTrue(any('contract-macros.tex: hand-typed' in failure
                                for failure in C.public_failures(root, check_render=False)))

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

    def test_generated_blocks_are_admitted_but_new_prose_numbers_and_demo_drift_fail(self):
        from unittest.mock import patch
        readme_path = ROOT / 'README.md'
        candidate = "# DOPE\n\n" + (ROOT / 'paper/generated/readme-result-card.md').read_text()
        candidate += (ROOT / 'paper/generated/readme-demo-notes.md').read_text()
        original_read = Path.read_text
        def check(text):
            def read(path, *args, **kwargs):
                return text if path == readme_path else original_read(path, *args, **kwargs)
            with patch.object(Path, 'read_text', read):
                return C.public_failures(check_render=False)
        self.assertEqual(check(candidate), [])
        self.assertTrue(any('hand-typed' in f for f in check(candidate + "New result: 0.123.")))
        self.assertTrue(any('demo notes drift' in f for f in check(candidate.replace('`artifact_bytes` 151', '`artifact_bytes` 152'))))

    def test_figure_and_card_drift_is_detected_by_same_gate(self):
        from unittest.mock import patch
        outputs = render.expected_outputs()
        outputs['paper/generated/hero-frontier.png'] = b'seeded drift'
        with patch.object(render, 'expected_outputs', return_value=outputs):
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
