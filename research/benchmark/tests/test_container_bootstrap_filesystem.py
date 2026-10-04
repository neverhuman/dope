"""Proposed startup-file controls with fake files and no candidate execution."""
import builtins
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import unittest
from unittest.mock import patch

from research.benchmark import container_bootstrap_filesystem as filesystem
from research.benchmark.tests import test_container_loader_environment as fixture


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FilesystemControls(unittest.TestCase):
    def setUp(self):
        self.f = fixture.EnvironmentControls('runTest'); self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.root = self.f.f.f.base / 'proposed-files'
        self.root.mkdir(mode=0o700)
        guard = patch.object(filesystem.environment.bootstrap, 'BASE', PurePosixPath(self.f.f.f.base))
        guard.start(); self.addCleanup(guard.stop)
        self.source = self.root / 'source'; self.source.mkdir(mode=0o700)
        self.job = {'fit_seed': 11, 'final': False, 'track': 'common-numeric'}
        self.job_sha = hashlib.sha256(filesystem.environment.search.catalog.loader.system.canonical(self.job).encode()).hexdigest()
        self.descriptor = {'format': 'dope-unadmitted-bootstrap-round-proposal', 'version': 1,
            'frozen_preview_job_digests': [self.job_sha], 'final_campaign_or_execution_lock': False,
            'execution_admitted': False, 'official_tests_opened': False}
        self.descriptor_path = self.root / 'round-proposal.json'
        self.descriptor_path.write_text(json.dumps(self.descriptor, sort_keys=True))
        self.cache = self.root / 'empty-cache'; self.cache.mkdir(mode=0o700)
        self.request = self.root / 'attempts' / self.job_sha / 'attempt-0001/request.json'
        self.request.parent.mkdir(mode=0o700, parents=True)
        self.req = {'round_sha256': sha(self.descriptor_path), 'job': self.job}
        self.request.write_text(json.dumps(self.req, sort_keys=True))
        self.f.proposal.update(root=str(self.root), entry=str(self.source / 'entry.py'),
            empty_bytecode_cache=str(self.cache), round_sha256=sha(self.descriptor_path),
            job_sha256=self.job_sha, request=str(self.request))
        self.f.freeze_proposal(); self.f.freeze()
        self.lock = {'format': 'dope-proposed-bootstrap-filesystem-preparation', 'version': 1,
            'environment_path': str(self.f.manifest), 'environment_sha256': self.f.digest,
            'filesystem_helper_sha256': sha(Path(filesystem.__file__)), 'source_files': {},
            'request_file': {'bytes': self.request.stat().st_size, 'sha256': sha(self.request)},
            'candidate_processes_started': 0}
        for name in filesystem.SOURCE_NAMES:
            path = self.source / name; path.write_bytes(('opaque source ' + name).encode())
            self.lock['source_files'][name] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
        for key in ('final_campaign_or_execution_lock', 'execution_admitted',
                    'full_runtime_closure_certified', 'official_tests_opened'):
            self.lock[key] = False
        for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority'):
            self.lock[key] = None
        self.manifest = self.f.f.f.base / 'filesystem.json'
        self.freeze(); self.verify()

    def freeze(self):
        self.manifest.write_text(json.dumps(self.lock, sort_keys=True)); self.digest = sha(self.manifest)

    def freeze_request(self):
        self.request.write_text(json.dumps(self.req, sort_keys=True))
        self.lock['request_file'] = {'bytes': self.request.stat().st_size, 'sha256': sha(self.request)}
        self.freeze()

    def verify(self, now=125.0, **overrides):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'numpy', 'scipy', 'sklearn', 'torch', 'catboost', 'pandas', 'sdv'}:
                self.fail('candidate dependency initialized')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded), \
                patch('subprocess.run', side_effect=AssertionError('process started')), \
                patch('subprocess.Popen', side_effect=AssertionError('process started')), \
                patch('ctypes.CDLL', side_effect=AssertionError('library loaded')), \
                patch.object(filesystem.environment.bootstrap.time, 'monotonic', return_value=now):
            return filesystem.verify_proposed_filesystem(**dict({'filesystem_path': str(self.manifest),
                'filesystem_sha256': self.digest, 'batch_started_at': 100.0}, **overrides))

    def rejected(self, **overrides):
        with self.assertRaisesRegex(ValueError, '^compressed bootstrap filesystem rejected$'):
            self.verify(**overrides)

    def test_six_owned_files_request_and_empty_cache_without_execution(self):
        result = self.verify()
        self.assertEqual(result['source_files_verified'], 6)
        self.assertEqual(result['request_job_sha256'], self.job_sha)
        self.assertEqual(result['outer_seconds_remaining'], 575.0)
        self.assertIs(result['empty_cache_verified'], True)
        self.assertIs(result['execution_admitted'], False)
        self.assertIs(result['final_campaign_or_execution_lock'], False)
        self.assertTrue(all(result[key] is None for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')))

    def test_wrong_outer_and_helper_hashes_precede_environment_guard(self):
        class Alias(str):
            pass
        with patch.object(filesystem.environment, 'verify_proposed_environment', side_effect=AssertionError('parent inspected')):
            self.rejected(filesystem_sha256='0' * 64)
            self.rejected(filesystem_sha256=Alias(self.digest))
            self.lock['filesystem_helper_sha256'] = '0' * 64; self.freeze(); self.rejected()

    def test_extra_source_entry_and_source_directory_reject(self):
        extra = self.source / 'extra.py'; extra.write_bytes(b'opaque'); self.rejected(); extra.unlink()
        entry = self.source / 'entry.py'; entry.unlink(); entry.mkdir(); self.rejected()

    def test_missing_and_changed_source_reject(self):
        path = self.source / 'entry.py'; blob = path.read_bytes()
        path.write_bytes(b'changed'); self.rejected()
        path.write_bytes(blob); self.verify(); path.unlink(); self.rejected()

    def test_source_alias_and_cache_alias_reject(self):
        entry = self.source / 'entry.py'; blob = entry.read_bytes()
        original = self.root / 'opaque-original'; original.write_bytes(blob)
        entry.unlink(); entry.symlink_to(original); self.rejected()
        entry.unlink(); entry.write_bytes(blob); self.verify()
        self.cache.rmdir(); self.cache.symlink_to(self.root); self.rejected()

    def test_nonempty_cache_and_missing_cache_reject(self):
        added = self.cache / 'opaque.pyc'; added.write_bytes(b'opaque'); self.rejected()
        added.unlink(); self.verify(); self.cache.rmdir(); self.rejected()

    def test_request_mode_and_request_alias_reject(self):
        self.request.parent.chmod(0o755); self.rejected()
        self.request.parent.chmod(0o700); self.verify()
        blob = self.request.read_bytes(); original = self.root / 'original-request'; original.write_bytes(blob)
        self.request.unlink(); self.request.symlink_to(original); self.rejected()

    def test_foreign_request_directory_owner_rejects(self):
        uid = filesystem.os.getuid()
        with patch.object(filesystem.os, 'getuid', return_value=uid + 1):
            self.rejected()

    def test_request_hash_precedes_request_decode(self):
        blob = b'not owned JSON'; self.request.write_bytes(blob)
        system = filesystem.environment.search.catalog.loader.system
        original = system.json.loads
        def guarded(data, *args, **kwargs):
            if data == blob:
                self.fail('unowned request decoded')
            return original(data, *args, **kwargs)
        with patch.object(system.json, 'loads', side_effect=guarded):
            self.rejected()

    def test_owned_request_wrong_job_numeric_identity_and_round_reject(self):
        old = copy.deepcopy(self.req)
        for key, value in [('fit_seed', 11.0), ('fit_seed', True), ('final', 0)]:
            self.req = copy.deepcopy(old); self.req['job'][key] = value
            self.freeze_request(); self.rejected()
        self.req = copy.deepcopy(old); self.req['round_sha256'] = '0' * 64
        self.freeze_request(); self.rejected()

    def test_wrong_attempt_path_reject(self):
        self.f.proposal['request'] = str(self.request.parent.parent / 'attempt-0002/request.json')
        self.f.freeze_proposal(); self.f.freeze(); self.lock['environment_sha256'] = self.f.digest
        self.freeze(); self.rejected()

    def test_descriptor_digest_before_decode_and_duplicate_jobs_reject(self):
        blob = self.descriptor_path.read_bytes(); self.descriptor_path.write_bytes(b'changed'); self.rejected()
        self.descriptor_path.write_bytes(blob); self.verify()
        self.descriptor['frozen_preview_job_digests'].append(self.job_sha)
        self.descriptor_path.write_text(json.dumps(self.descriptor, sort_keys=True))
        self.f.proposal['round_sha256'] = sha(self.descriptor_path)
        self.f.freeze_proposal(); self.f.freeze(); self.lock['environment_sha256'] = self.f.digest
        self.freeze(); self.rejected()

    def test_late_source_request_cache_and_parent_changes_reject(self):
        original = filesystem.environment.verify_proposed_environment
        for selected in ('source', 'request', 'cache', 'descriptor'):
            calls = []
            paths = [self.source / 'entry.py', self.request, self.descriptor_path]
            old = {path: path.read_bytes() for path in paths}
            def mutate(*args, **kwargs):
                result = original(*args, **kwargs); calls.append(True)
                if len(calls) == 2:
                    if selected == 'cache':
                        (self.cache / 'late.pyc').write_bytes(b'opaque')
                    else:
                        {'source': paths[0], 'request': paths[1], 'descriptor': paths[2]}[selected].write_bytes(b'late opaque')
                return result
            with self.subTest(selected=selected), patch.object(filesystem.environment, 'verify_proposed_environment', side_effect=mutate):
                self.rejected()
            for path, blob in old.items():
                path.write_bytes(blob)
            if (self.cache / 'late.pyc').exists():
                (self.cache / 'late.pyc').unlink()
            self.verify()

    def test_io_timeout_has_fixed_error_text(self):
        python = filesystem.environment.search.catalog.loader.system.python
        original = python.file_identity
        def fail(path, row):
            if path == self.request:
                raise TimeoutError('opaque private I/O timeout')
            return original(path, row)
        with patch.object(python, 'file_identity', side_effect=fail):
            self.rejected()

    def test_original_deadline_and_expiry_after_final_hashes(self):
        with self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'):
            self.verify(now=700.0)
        # Parent checks (2 each), plan check and final check share one timer.
        with patch.object(filesystem.environment.bootstrap.time, 'monotonic', side_effect=[125.0] * 5 + [700.0]), \
                self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'):
            filesystem.verify_proposed_filesystem(str(self.manifest), self.digest, batch_started_at=100.0)

    def test_claims_and_exact_inventory_identities_reject(self):
        old = copy.deepcopy(self.lock)
        for key, value in [('version', 1.0), ('candidate_processes_started', False),
                           ('execution_admitted', 0), ('final_campaign_or_execution_lock', True),
                           ('ptf_v1', .99), ('source_files', {})]:
            self.lock = copy.deepcopy(old); self.lock[key] = value; self.freeze(); self.rejected()


if __name__ == '__main__':
    unittest.main()
