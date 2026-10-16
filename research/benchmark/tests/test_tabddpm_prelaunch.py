"""Exercise frozen entry code with canaries before optional dependency imports."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from research.benchmark.manifest import digest
from research.benchmark.score import sha256


FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures/tabddpm-native-prelaunch'
PINS = {
    'common.py': '6a4e4103797efc33554c0bd6ca393a3efe40af905eeab86115f9381e2dc68103',
    'entry.py': '061bd80a5e16af97b7363253c9a4bebbacbf6266d54434de5a8d8d31546485f5',
}


class TabDDPMPrelaunchTests(unittest.TestCase):
    def setUp(self):
        parent = Path('target/tabddpm-native-prelaunch-tests')
        parent.mkdir(parents=True, exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.case_index = 0

    def fixture(self, entry_name):
        self.case_index += 1
        root = self.root / (entry_name.replace('.py', '') + '-' + str(self.case_index))
        root.mkdir()
        source = root / 'source'
        source.mkdir()
        for name, expected in PINS.items():
            self.assertEqual(sha256(FIXTURES / name), expected)
            destination = source / (entry_name if name == 'entry.py' else name)
            destination.write_bytes((FIXTURES / name).read_bytes())
        # Fit/sample must pass their own host and GPU inventory gates so an
        # unrelated rejection cannot mask missing runtime verification.
        (source / 'inventory_hosts.py').write_text(
            "def local():\n    return {'active_gpu_processes': [], "
            "'gpus': [{'memory_free_mib': 24 * 1024}]}\n")
        environment = root / 'environment'
        environment.mkdir()
        marker = root / 'initializer-ran'
        dependency = environment / 'numpy.py'
        dependency.write_text('from pathlib import Path\nPath(' + repr(str(marker))
                              + ').write_text("ran")\n#' + ' ' * (1 << 20))
        original = root / 'original-entry.py'
        original.write_text('import numpy\ndef check_all(): raise RuntimeError("initializer reached")\n')
        command = root / 'fake-command'
        command.write_text('fixture only\n')
        worker = root / 'worker'
        worker.mkdir()
        for name in ('train.csv', 'validation.csv'):
            (worker / name).write_text('0,1\n1,0\n')
        for name in ('worker-manifest.json', 'projection.json'):
            (worker / name).write_text('{}')
        job = {'method': 'TabDDPM', 'final': False, 'worker': {
            'path': str(worker), 'files': {p.name: sha256(p) for p in worker.iterdir()}}}
        inventory = {'aliases': [], 'roots': [str(environment)],
                     'inventory_modes': {str(environment): 'all'},
                     'files': {str(p): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
                               for p in (dependency, original)}}
        inventory_path = root / 'runtime-inventory.lock.json'
        inventory_path.write_text(json.dumps(inventory))
        lock = {'source_files': {str(p): sha256(p) for p in source.iterdir()},
                'runtime_inventory_sha256': sha256(inventory_path),
                'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None,
                'ssh_binary': str(command), 'ssh_binary_sha256': sha256(command),
                'du_binary': str(command), 'du_binary_sha256': sha256(command),
                'known_hosts': str(command), 'known_hosts_sha256': sha256(command),
                'jobs': [job], 'cpu_slots': {host: sorted(os.sched_getaffinity(0))
                                           for host in ('xbabe1', 'xbabe2')},
                'environment_site': str(environment), 'original_entry': str(original)}
        round_path = root / 'round.lock.json'
        round_path.write_text(json.dumps(lock))
        expected_round = sha256(round_path)
        (root / 'execution.lock.json').write_text(json.dumps({'round_sha256': expected_round}))
        destination = root / 'attempts' / digest(job) / 'attempt-0001'
        destination.mkdir(parents=True)
        request = root / 'request.json'
        request.write_text(json.dumps({'job': job, 'receipt': str(destination / 'receipt.json')}))
        return root, dependency, marker, expected_round, sha256(source / 'common.py')

    def invoke(self, case, mode, entry_name, *, wrong_receipt_path=False,
               bypass_runtime_verification=False):
        root, _, _, expected_round, common_digest = case
        request = root / 'request.json'
        if wrong_receipt_path:
            value = json.loads(request.read_text())
            value['receipt'] = str(root / 'receipt.json')
            request.write_text(json.dumps(value))
        cache = root / 'empty-cache'
        cache.mkdir()
        bootstrap = '''import importlib.util,sys
entry,mode,request,round_digest,common_digest,bypass=sys.argv[1:]
sys.argv=[entry,mode,request,round_digest,common_digest]
spec=importlib.util.spec_from_file_location('frozen_entry',entry)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.socket.gethostname=lambda:'xbabe2' if mode=='native' else 'xbabe1'
if bypass=='true': module.common.verify_runtime=lambda lock:None
module.run()
'''
        return subprocess.run([sys.executable, '-I', '-S', '-B', '-X',
                               'pycache_prefix=' + str(cache), '-c', bootstrap,
                               str(root / 'source' / entry_name), mode, str(request),
                               expected_round, common_digest,
                               'true' if bypass_runtime_verification else 'false'],
                              capture_output=True, timeout=10, cwd=root)

    def test_dependency_tail_drift_rejected_before_any_initializer(self):
        for entry in ('entry.py', 'entry-cpu.py'):
            for mode in ('fit', 'sample', 'native'):
                with self.subTest(entry=entry, mode=mode):
                    case = self.fixture(entry)
                    with case[1].open('r+b') as stream:
                        stream.seek(-1, 2)
                        stream.write(b'X')
                    result = self.invoke(case, mode, entry)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(b'AssertionError', result.stderr)
                    self.assertFalse(case[2].exists())

    def test_added_directory_alias_rejected_before_initializer(self):
        case = self.fixture('entry.py')
        (case[0] / 'environment/alias').symlink_to(case[0] / 'environment', target_is_directory=True)
        result = self.invoke(case, 'native', 'entry.py')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'AssertionError', result.stderr)
        self.assertFalse(case[2].exists())

    def test_unchanged_dependency_reaches_initializer_in_every_mode(self):
        for entry in ('entry.py', 'entry-cpu.py'):
            for mode in ('fit', 'sample', 'native'):
                with self.subTest(entry=entry, mode=mode):
                    case = self.fixture(entry)
                    result = self.invoke(case, mode, entry)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(b'initializer reached', result.stderr)
                    self.assertTrue(case[2].exists())

    def test_bypassed_verification_reaches_drifted_canary_in_every_mode(self):
        for entry in ('entry.py', 'entry-cpu.py'):
            for mode in ('fit', 'sample', 'native'):
                with self.subTest(entry=entry, mode=mode):
                    case = self.fixture(entry)
                    with case[1].open('r+b') as stream:
                        stream.seek(-1, 2)
                        stream.write(b'X')
                    result = self.invoke(case, mode, entry, bypass_runtime_verification=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(b'initializer reached', result.stderr)
                    self.assertTrue(case[2].exists())

    def test_wrong_attempt_path_rejected_before_initializer(self):
        case = self.fixture('entry-cpu.py')
        result = self.invoke(case, 'native', 'entry-cpu.py', wrong_receipt_path=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'AssertionError', result.stderr)
        self.assertFalse(case[2].exists())


if __name__ == '__main__':
    unittest.main()
