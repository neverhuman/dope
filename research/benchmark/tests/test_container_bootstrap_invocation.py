"""Restricted startup planning and unchanged outer deadline; no process starts."""
from dataclasses import FrozenInstanceError
import hashlib
import json
import math
import os
import unittest
from unittest.mock import patch

from research.benchmark import container_bootstrap_invocation as bootstrap


class BootstrapInvocationControls(unittest.TestCase):
    def setUp(self):
        root = '/mnt/fast-scratch/dope-benchmark/opaque-round'
        self.proposal = {'host': 'xbabe1', 'root': root, 'entry': root + '/source/entry.py',
            'python': '/usr/bin/python3.12', 'empty_bytecode_cache': root + '/empty-cache',
            'native_library_directory': '/opaque/frozen-library', 'round_sha256': 'a' * 64,
            'job_sha256': 'b' * 64, 'attempt': 1,
            'request': root + '/attempts/' + 'b' * 64 + '/attempt-0001/request.json',
            'batch_cap_seconds': 600, 'execution_admitted': False, 'official_tests_opened': False,
            'gpu_operations_enabled': False, 'mfs_v2': None, 'ptf_v1': None,
            'release_safe': None, 'superiority': None}

    def prepare(self, start=100.0, now=125.0, digest_override=None):
        blob = json.dumps(self.proposal, sort_keys=True).encode()
        digest = hashlib.sha256(blob).hexdigest() if digest_override is None else digest_override
        with patch('subprocess.run', side_effect=AssertionError('candidate process started')), \
                patch('subprocess.Popen', side_effect=AssertionError('candidate process started')), \
                patch.object(bootstrap.time, 'monotonic', return_value=now):
            return bootstrap.prepare_invocation(blob, digest, batch_started_at=start)

    def rejected(self, **args):
        with self.assertRaisesRegex(ValueError, '^compressed bootstrap proposal rejected$'):
            self.prepare(**args)

    def test_restricted_immutable_arguments_and_environment(self):
        with patch.dict(os.environ, {'LD_PRELOAD': 'opaque', 'LD_AUDIT': 'opaque',
                                    'GLIBC_TUNABLES': 'opaque', 'PYTHONPATH': 'opaque',
                                    'PYTHONSTARTUP': 'opaque', 'CUDA_VISIBLE_DEVICES': '0'}):
            plan = self.prepare()
        self.assertEqual(plan.host, 'xbabe1')
        self.assertEqual(plan.argv, ('/usr/bin/python3.12', '-I', '-S', '-B', '-X',
            'pycache_prefix=' + self.proposal['empty_bytecode_cache'], self.proposal['entry'],
            self.proposal['root'], self.proposal['round_sha256'], self.proposal['request']))
        self.assertEqual(dict(plan.environment), {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C',
            'LD_LIBRARY_PATH': '/opaque/frozen-library', 'CUDA_VISIBLE_DEVICES': '',
            'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
        self.assertFalse(plan.execution_admitted)
        self.assertFalse(plan.filesystem_verified)
        self.assertFalse(plan.runtime_closure_verified)
        with self.assertRaises(FrozenInstanceError):
            plan.batch_started_at = 125.0

    def test_remaining_budget_decreases_without_restart(self):
        plan = self.prepare(start=100.0, now=125.0)
        with patch.object(bootstrap.time, 'monotonic', return_value=140.0):
            self.assertEqual(plan.remaining_seconds(), 560.0)
        with patch.object(bootstrap.time, 'monotonic', return_value=700.0):
            with self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'):
                plan.remaining_seconds()

    def test_expiry_during_preparation_rejected(self):
        for now in [700.0, 701.0]:
            with self.subTest(now=now), self.assertRaises(TimeoutError):
                self.prepare(start=100.0, now=now)

    def test_invalid_clocks_rejected(self):
        for start, now in [(True, 125.0), (math.nan, 125.0), (math.inf, 125.0),
                           (-1.0, 125.0), (100.0, 99.0), (100.0, math.nan),
                           (100.0, math.inf), (100.0, True), (10 ** 1000, 125.0)]:
            with self.subTest(start_type=type(start).__name__, now_type=type(now).__name__):
                self.rejected(start=start, now=now)

    def test_digest_mismatch_precedes_json_parse(self):
        with patch.object(bootstrap.json, 'loads', side_effect=AssertionError('unbound proposal parsed')):
            self.rejected(digest_override='0' * 64)

    def test_digest_subclass_rejected(self):
        class Permissive(str):
            def __eq__(self, other):
                return True
        blob = json.dumps(self.proposal, sort_keys=True).encode()
        self.rejected(digest_override=Permissive(hashlib.sha256(blob).hexdigest()))

    def test_off_limit_hosts_and_nonboolean_claims_rejected(self):
        for field, value in [('host', 'xbabe0'), ('host', 'unknown'), ('execution_admitted', True),
                             ('execution_admitted', 0), ('execution_admitted', 'false'),
                             ('official_tests_opened', True), ('gpu_operations_enabled', True),
                             ('mfs_v2', 0.99), ('batch_cap_seconds', 601), ('batch_cap_seconds', 600.0)]:
            saved = self.proposal[field]
            with self.subTest(field=field):
                self.proposal[field] = value
                self.rejected()
            self.proposal[field] = saved

    def test_retry_identity_cannot_change_by_numeric_equality(self):
        for value in [1.0, True, 0, 10000]:
            self.proposal['attempt'] = value
            self.rejected()
        self.proposal['attempt'] = 2
        self.rejected()

    def test_path_escape_and_loader_path_lists_rejected(self):
        for field, value in [('root', '/outside'), ('entry', self.proposal['root'] + '/other.py'),
                             ('request', self.proposal['root'] + '/evaluator/request.json'),
                             ('request', self.proposal['root'] + '/test.csv'),
                             ('empty_bytecode_cache', '/outside/cache'),
                             ('native_library_directory', '/opaque/library:/opaque/other'),
                             ('native_library_directory', '/opaque/../other'),
                             ('python', '/usr/bin/python3')]:
            saved = self.proposal[field]
            with self.subTest(field=field):
                self.proposal[field] = value
                self.rejected()
            self.proposal[field] = saved


if __name__ == '__main__':
    unittest.main()
