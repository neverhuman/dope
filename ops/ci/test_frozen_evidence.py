"""Exercise frozen evidence admission on synthetic Git repositories only."""

import copy
import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    'frozen_evidence', Path(__file__).with_name('check-frozen-evidence.py'))
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)
PATHS = (
    'research/benchmark/results/fixture/panel.json',
    'docs/whitepaper/generated/fixture.tex',
    'research/benchmark/methods.lock.json',
    'research/benchmark/review_fixes/predeclare.json',
    'research/benchmark/review_fixes/predeclare_v2.json',
    'ops/ci/paper-review-inputs.json',
)
ORIGINAL = b'{"host":"historical-host","retention":0.9}\n'
SCRUBBED = ORIGINAL.replace(b'historical-host', b'public-placeholder')


class FrozenEvidenceTests(unittest.TestCase):
    def setUp(self):
        local_tmp = Path(__file__).resolve().parents[2] / 'target/frozen-evidence-tests'
        local_tmp.mkdir(parents=True, exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=local_tmp)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.git('init', '-q')
        for name in PATHS:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(ORIGINAL)
        (self.root / '.gitignore').write_text('__pycache__/\n*.pyc\n')
        self.git('add', '.')
        self.recorded = {name: hashlib.sha256(ORIGINAL).hexdigest() for name in PATHS}

    def git(self, *args):
        subprocess.run(['git', *args], cwd=self.root, check=True, capture_output=True)

    def check(self, allowlist=None):
        return GUARD.check_tree(self.root, self.recorded, {} if allowlist is None else allowlist)

    def test_unchanged_baseline_and_generated_python_caches_pass(self):
        cache = self.root / 'research/benchmark/results/__pycache__/fixture.pyc'
        cache.parent.mkdir()
        cache.write_bytes(b'local Python cache')
        self.assertEqual(self.check(), [])

    def test_host_only_scrub_in_each_protected_area_fails(self):
        for name in PATHS:
            with self.subTest(name=name):
                path = self.root / name
                path.write_bytes(SCRUBBED)
                self.assertEqual(len(self.check()), 1)
                self.assertIn(f'{name}: frozen SHA256 drift', self.check()[0])
                path.write_bytes(ORIGINAL)

    def test_missing_file_and_rename_are_rejected(self):
        path = self.root / PATHS[0]
        renamed = path.with_name('renamed.json')
        path.rename(renamed)
        failures = '\n'.join(self.check())
        self.assertIn(f'{PATHS[0]}: frozen file missing', failures)
        self.assertIn('renamed.json: unrecorded', failures)

    def test_unrecorded_new_files_in_each_protected_area_fail(self):
        names = (
            'research/benchmark/results/new.json',
            'docs/whitepaper/generated/new.tex',
            'research/benchmark/review_fixes/predeclare_new.json',
        )
        for name in names:
            with self.subTest(name=name):
                (self.root / name).write_bytes(ORIGINAL)
                self.assertTrue(any(f'{name}: unrecorded' in f for f in self.check()))

    def test_override_requires_exact_replacement_hash_and_reason(self):
        name = PATHS[0]
        path = self.root / name
        path.write_bytes(SCRUBBED)
        allowlist = {name: {'sha256': hashlib.sha256(SCRUBBED).hexdigest(),
                            'reason': 'Approved replacement in synthetic regression fixture'}}
        self.assertEqual(self.check(allowlist), [])
        path.write_bytes(SCRUBBED + b' ')
        self.assertTrue(any('frozen SHA256 drift' in f for f in self.check(allowlist)))
        path.unlink()
        self.assertTrue(any('frozen file missing' in f for f in self.check(allowlist)))

    def test_approved_new_file_is_hash_pinned_and_must_exist(self):
        name = 'research/benchmark/results/new.json'
        allowlist = {name: {'sha256': hashlib.sha256(ORIGINAL).hexdigest(),
                            'reason': 'New separately reviewed synthetic fixture'}}
        self.assertTrue(any('frozen file missing' in f for f in self.check(allowlist)))
        (self.root / name).write_bytes(ORIGINAL)
        self.assertEqual(self.check(allowlist), [])

    def test_broad_unexplained_unpinned_or_out_of_scope_exceptions_fail(self):
        entry = {'sha256': hashlib.sha256(SCRUBBED).hexdigest(), 'reason': 'fixture'}
        cases = [
            [],
            {'research/benchmark/results/*': entry},
            {'research/benchmark/results/../methods.lock.json': entry},
            {'/research/benchmark/results/panel.json': entry},
            {'README.md': entry},
            {PATHS[0]: {'sha256': entry['sha256'], 'reason': '  '}},
            {PATHS[0]: {'reason': 'no hash'}},
            {PATHS[0]: {'sha256': None, 'reason': 'no deletion exception'}},
            {PATHS[0]: {'sha256': 'not a hash', 'reason': 'fixture'}},
        ]
        for allowlist in cases:
            with self.subTest(allowlist=allowlist), self.assertRaises(ValueError):
                self.check(allowlist)

    def test_matching_bytes_through_file_or_directory_symlink_fail(self):
        name = PATHS[0]
        path = self.root / name
        saved = self.root / 'saved.json'
        path.rename(saved)
        path.symlink_to(saved)
        self.assertTrue(any(f'{name}: frozen file missing' in f for f in self.check()))
        path.unlink()
        saved.rename(path)
        directory = path.parent
        saved_directory = self.root / 'saved-directory'
        directory.rename(saved_directory)
        directory.symlink_to(saved_directory, target_is_directory=True)
        self.assertTrue(any(f'{name}: frozen file missing' in f for f in self.check()))

    def test_tracked_files_remain_protected_if_new_ignore_rule_matches(self):
        (self.root / '.gitignore').write_text('*.json\n')
        (self.root / PATHS[0]).write_bytes(SCRUBBED)
        self.assertTrue(any(f'{PATHS[0]}: frozen SHA256 drift' in f for f in self.check()))

    def test_bulk_repin_or_manifest_coverage_removal_fails(self):
        original = GUARD.load_manifest(GUARD.MANIFEST)
        first = next(iter(original['files']))
        mutated = copy.deepcopy(original)
        mutated['files'][first] = '0' * 64
        removed = copy.deepcopy(original)
        del removed['files'][first]
        baseline = copy.deepcopy(original)
        baseline['baseline_commit'] = 'f' * 40
        path = self.root / 'tampered-manifest.json'
        for data in (mutated, removed, baseline):
            with self.subTest(data=data['baseline_commit']):
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    GUARD.load_manifest(path)

    def test_only_requested_areas_are_frozen(self):
        path = self.root / 'README.md'
        path.write_bytes(SCRUBBED)
        self.assertEqual(self.check(), [])


if __name__ == '__main__':
    unittest.main()
