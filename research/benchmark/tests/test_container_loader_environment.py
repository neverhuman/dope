"""Restricted environment and snapshot controls using only retained fake inputs."""
import builtins
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from research.benchmark import container_loader_environment as environment
from research.benchmark.tests import test_container_search_path_custody as fixture


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EnvironmentControls(unittest.TestCase):
    def setUp(self):
        self.f = fixture.SearchSnapshotControls('runTest'); self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        # Route the fixed preload check to this fixture's frozen missing file.
        guard = patch.object(environment, 'PRELOAD', str(self.f.f.absent))
        guard.start(); self.addCleanup(guard.stop)
        root = '/mnt/fast-scratch/dope-benchmark/opaque-round'
        self.proposal = {'host': 'xbabe1', 'root': root, 'entry': root + '/source/entry.py',
            'python': '/usr/bin/python3.12', 'empty_bytecode_cache': root + '/empty-cache',
            'native_library_directory': str(self.f.f.directory_alias), 'round_sha256': 'a' * 64,
            'job_sha256': 'b' * 64, 'attempt': 1,
            'request': root + '/attempts/' + 'b' * 64 + '/attempt-0001/request.json',
            'batch_cap_seconds': 600, 'execution_admitted': False, 'official_tests_opened': False,
            'gpu_operations_enabled': False, 'mfs_v2': None, 'ptf_v1': None,
            'release_safe': None, 'superiority': None}
        self.proposal_path = self.f.f.base / 'invocation.json'
        self.lock = {'format': 'dope-proposed-loader-environment-preparation', 'version': 1,
            'proposal_path': str(self.proposal_path), 'projection_path': str(self.f.manifest),
            'projection_sha256': self.f.digest, 'candidate_processes_started': 0,
            'environment_helper_sha256': sha(Path(environment.__file__)),
            'bootstrap_helper_sha256': sha(Path(environment.bootstrap.__file__)),
            'search_helper_sha256': sha(Path(environment.search.__file__)),
            'environment': [['CUDA_VISIBLE_DEVICES', ''], ['LC_ALL', 'C'],
                ['LD_LIBRARY_PATH', str(self.f.f.directory_alias)], ['MKL_NUM_THREADS', '1'],
                ['OMP_NUM_THREADS', '1'], ['OPENBLAS_NUM_THREADS', '1'], ['PATH', '/usr/bin:/bin']]}
        for key in ('actual_loader_selection_verified', 'complete_provider_search_verified',
                    'full_runtime_closure_certified', 'execution_admitted', 'official_tests_opened'):
            self.lock[key] = False
        for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority'):
            self.lock[key] = None
        self.manifest = self.f.f.base / 'environment.json'
        self.freeze_proposal(); self.freeze(); self.verify()

    def freeze_proposal(self):
        self.proposal_path.write_text(json.dumps(self.proposal, sort_keys=True))
        self.lock['proposal_sha256'] = sha(self.proposal_path)

    def freeze(self):
        self.manifest.write_text(json.dumps(self.lock, sort_keys=True)); self.digest = sha(self.manifest)

    def verify(self, now=125.0, **overrides):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'numpy', 'scipy', 'sklearn', 'torch', 'catboost'}:
                self.fail('candidate dependency initialized')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded), \
                patch('subprocess.run', side_effect=AssertionError('process started')), \
                patch('subprocess.Popen', side_effect=AssertionError('process started')), \
                patch('ctypes.CDLL', side_effect=AssertionError('library loaded')), \
                patch.object(environment.bootstrap.time, 'monotonic', return_value=now):
            return environment.verify_proposed_environment(**dict({'environment_path': str(self.manifest),
                'environment_sha256': self.digest, 'batch_started_at': 100.0}, **overrides))

    def rejected(self, **overrides):
        with self.assertRaisesRegex(ValueError, '^compressed loader environment rejected$'):
            self.verify(**overrides)

    def test_restricted_environment_directory_and_preload_absence(self):
        result = self.verify()
        self.assertEqual(result['restricted_environment_entries_verified'], 7)
        self.assertEqual(result['outer_seconds_remaining'], 575.0)
        self.assertIs(result['declared_library_directory_snapshot_verified'], True)
        self.assertIs(result['preload_absence_snapshot_verified'], True)
        for key in ('actual_loader_selection_verified', 'complete_provider_search_verified',
                    'full_runtime_closure_certified', 'execution_admitted', 'official_tests_opened'):
            self.assertIs(result[key], False)
        self.assertTrue(all(result[key] is None for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')))

    def test_digest_and_helper_mismatches_precede_plan(self):
        class Alias(str):
            pass
        with patch.object(environment.bootstrap, 'prepare_invocation', side_effect=AssertionError('plan derived')):
            self.rejected(environment_sha256=Alias(self.digest))
            for key in ('environment_helper_sha256', 'bootstrap_helper_sha256', 'search_helper_sha256'):
                old = self.lock[key]; self.lock[key] = '0' * 64
                self.freeze(); self.rejected(); self.lock[key] = old

    def test_proposal_digest_before_decode(self):
        self.proposal_path.write_bytes(b'changed opaque proposal')
        with patch.object(environment.bootstrap, 'prepare_invocation', side_effect=AssertionError('plan derived')):
            self.rejected()

    def test_changed_or_reordered_environment_before_loader_inspection(self):
        old = copy.deepcopy(self.lock['environment'])
        for entries in (old[:-1], list(reversed(old)), old + [['LD_PRELOAD', 'opaque']],
                        [['CUDA_VISIBLE_DEVICES', '0'], *old[1:]]):
            self.lock['environment'] = entries; self.freeze()
            with patch.object(environment.search, 'verify_declared_search_paths',
                              side_effect=AssertionError('loader inspected before environment rejection')):
                self.rejected()

    def test_unbound_library_directory_even_with_frozen_valid_plan(self):
        self.proposal['native_library_directory'] = str(self.f.f.base / 'unbound-library')
        self.lock['environment'][2][1] = self.proposal['native_library_directory']
        self.freeze_proposal(); self.freeze(); self.rejected()

    def test_present_preload_and_missing_absence_record_reject(self):
        self.f.f.absent.write_bytes(b'opaque preload input'); self.rejected()
        self.f.f.absent.unlink()
        del self.f.f.lock['absent'][str(self.f.f.absent)]
        self.refresh_loader_parent(); self.rejected()

    def refresh_loader_parent(self):
        self.f.f.freeze(); self.f.search['loader_sha256'] = self.f.f.digest
        self.f.lock['loader_sha256'] = self.f.f.digest
        self.f.freeze_search(); self.f.freeze()
        self.lock['projection_sha256'] = self.f.digest; self.freeze()

    def test_snapshot_and_projection_identity_drift_reject(self):
        self.f.manifest.write_bytes(b'changed owned projection'); self.rejected()

    def test_forbidden_host_retry_and_library_separators_reject(self):
        old = copy.deepcopy(self.proposal)
        for key, value in [('host', 'xbabe0'), ('attempt', True), ('attempt', 1.0),
                           ('native_library_directory', '/opaque/a:/opaque/b'),
                           ('native_library_directory', '/opaque/$LIB')]:
            self.proposal = copy.deepcopy(old); self.proposal[key] = value
            self.freeze_proposal(); self.freeze(); self.rejected()

    def test_expired_original_timer_is_preserved(self):
        for now in (700.0, 701.0):
            with self.subTest(now=now), self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'):
                self.verify(now=now)

    def test_io_timeouts_reject_without_input_text_or_deadline_misclassification(self):
        for module, name in [(environment.search.catalog.loader.system, 'owned_manifest'),
                             (environment.search, 'verify_declared_search_paths')]:
            with self.subTest(name=name), patch.object(module, name, side_effect=TimeoutError('opaque private input')):
                self.rejected()

    def test_deadline_timeout_text_is_sanitized(self):
        with patch.object(environment.bootstrap, 'prepare_invocation', side_effect=TimeoutError('opaque clock input')), \
                self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'):
            self.verify()

    def test_deadline_rechecked_after_final_snapshot_verification(self):
        calls = []
        original = environment.search.verify_declared_search_paths
        def checked(*args):
            calls.append(True); return original(*args)
        with patch.object(environment.search, 'verify_declared_search_paths', side_effect=checked), \
                patch.object(environment.bootstrap.time, 'monotonic', side_effect=[125.0, 700.0]), \
                self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'):
            environment.verify_proposed_environment(str(self.manifest), self.digest, batch_started_at=100.0)
        self.assertEqual(len(calls), 2)

    def test_claim_and_numeric_identities_reject(self):
        old = copy.deepcopy(self.lock)
        for key, value in [('version', 1.0), ('candidate_processes_started', False),
                           ('execution_admitted', 0), ('official_tests_opened', True), ('ptf_v1', .99)]:
            self.lock = copy.deepcopy(old); self.lock[key] = value; self.freeze(); self.rejected()

    def test_late_proposal_manifest_and_snapshot_changes_reject(self):
        original = environment.search.verify_declared_search_paths
        for path in (self.proposal_path, self.manifest, self.f.manifest, self.f.f.provider):
            blob = path.read_bytes(); changed = []
            def mutate(*args):
                result = original(*args)
                if not changed:
                    changed.append(True); path.write_bytes(b'changed after first snapshot check')
                return result
            with self.subTest(path=path.name), patch.object(environment.search, 'verify_declared_search_paths', side_effect=mutate):
                self.rejected()
            path.write_bytes(blob)


if __name__ == '__main__':
    unittest.main()
