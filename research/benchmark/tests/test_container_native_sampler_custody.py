"""Native executable custody using opaque ELF bytes and staged fake metadata."""
import builtins
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark import container_native_sampler_custody as sampler
from research.benchmark.tests.test_container_elf_inputs import opaque_object


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NativeSamplerControls(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path('target').resolve(), prefix='native-sampler-')
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        for module, name, value in [(sampler.system.python, 'BASE', self.base),
                                    (sampler.bootstrap, 'BASE', PurePosixPath(self.base))]:
            guard = patch.object(module, name, value); guard.start(); self.addCleanup(guard.stop)
        self.root = self.base / 'staged-round'; self.root.mkdir()
        self.binary = self.base / 'native.bin'; self.binary.write_bytes(bytes(opaque_object()))
        self.fit_path = self.base / 'fit-round.json'
        self.fit = {'gpu_binary': str(self.binary), 'gpu_binary_sha256': sha(self.binary),
                    'binary_product_source_sha256': {'rust/sample.rs': 'd' * 64}}
        self.job = {'fit_seed': 11, 'final': False, 'track': 'common-numeric',
            'original_generator_binary_sha256': sha(self.binary),
            'original_product_source_files_sha256': copy.deepcopy(self.fit['binary_product_source_sha256'])}
        self.proposal = {'host': 'xbabe2', 'root': str(self.root), 'entry': str(self.root / 'source/entry.py'),
            'python': '/usr/bin/python3.12', 'empty_bytecode_cache': str(self.root / 'empty-cache'),
            'native_library_directory': '/opaque/frozen-library', 'round_sha256': 'a' * 64,
            'attempt': 1, 'batch_cap_seconds': 600, 'execution_admitted': False,
            'official_tests_opened': False, 'gpu_operations_enabled': False,
            'mfs_v2': None, 'ptf_v1': None, 'release_safe': None, 'superiority': None}
        self.proposal_path = self.base / 'proposal.json'
        self.manifest = self.base / 'sampler.json'
        self.lock = {'format': 'dope-native-sampler-elf-preparation', 'version': 1,
            'proposal_path': str(self.proposal_path), 'fit_round_path': str(self.fit_path),
            'binary_path': str(self.binary), 'binary_sha256': sha(self.binary),
            'binary_bytes': self.binary.stat().st_size,
            'declarations': sampler.elf.inspect_elf_inputs(self.binary.read_bytes(), sha(self.binary)),
            'elf_helper_sha256': sha(Path(sampler.elf.__file__)),
            'bootstrap_helper_sha256': sha(Path(sampler.bootstrap.__file__)),
            'sampler_helper_sha256': sha(Path(sampler.__file__)), 'candidate_processes_started': 0}
        for key in ('loader_resolution_verified', 'full_runtime_closure_certified',
                    'execution_admitted', 'official_tests_opened'):
            self.lock[key] = False
        for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority'):
            self.lock[key] = None
        self.freeze_job(); self.freeze_fit(); self.freeze(); self.verify()

    def freeze_job(self):
        self.job_sha = hashlib.sha256(sampler.system.canonical(self.job).encode()).hexdigest()
        self.request = self.root / 'attempts' / self.job_sha / 'attempt-0001/request.json'
        self.request.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.request.write_text(json.dumps({'round_sha256': self.proposal['round_sha256'], 'job': self.job}, sort_keys=True))
        self.proposal.update(job_sha256=self.job_sha, request=str(self.request))
        self.proposal_path.write_text(json.dumps(self.proposal, sort_keys=True))
        self.lock.update(job_sha256=self.job_sha, request_sha256=sha(self.request),
                         proposal_sha256=sha(self.proposal_path))

    def freeze_fit(self):
        self.fit_path.write_text(json.dumps(self.fit, sort_keys=True))
        self.lock['fit_round_sha256'] = sha(self.fit_path)

    def freeze(self):
        self.manifest.write_text(json.dumps(self.lock, sort_keys=True)); self.digest = sha(self.manifest)

    def verify(self, now=125.0, **overrides):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'numpy', 'scipy', 'sklearn', 'torch', 'pandas', 'sdv'}:
                self.fail('candidate dependency initialized')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded), \
                patch('subprocess.run', side_effect=AssertionError('native process started')), \
                patch('subprocess.Popen', side_effect=AssertionError('native process started')), \
                patch('ctypes.CDLL', side_effect=AssertionError('native library loaded')), \
                patch.object(sampler.bootstrap.time, 'monotonic', return_value=now):
            return sampler.verify_sampler_declarations(**dict({'sampler_path': str(self.manifest),
                'sampler_sha256': self.digest, 'batch_started_at': 100.0}, **overrides))

    def rejected(self, **overrides):
        with self.assertRaisesRegex(ValueError, '^compressed native sampler custody rejected$'):
            self.verify(**overrides)

    def test_owned_native_root_stays_unresolved_and_unadmitted(self):
        result = self.verify()
        self.assertEqual(result['native_elf_roots_verified'], 1)
        self.assertEqual(result['needed_declarations'], 1)
        self.assertEqual(result['binary_sha256'], sha(self.binary))
        self.assertEqual(result['outer_seconds_remaining'], 575.0)
        for key in ('loader_resolution_verified', 'full_runtime_closure_certified',
                    'execution_admitted', 'official_tests_opened'):
            self.assertIs(result[key], False)
        self.assertTrue(all(result[key] is None for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')))

    def test_outer_digest_and_helper_before_parser(self):
        class Alias(str):
            pass
        with patch.object(sampler.elf, 'inspect_elf_inputs', side_effect=AssertionError('unowned ELF parsed')):
            self.rejected(sampler_sha256='0' * 64)
            self.rejected(sampler_sha256=Alias(self.digest))
            for key in ('elf_helper_sha256', 'bootstrap_helper_sha256', 'sampler_helper_sha256'):
                old = self.lock[key]; self.lock[key] = '0' * 64; self.freeze(); self.rejected()
                self.lock[key] = old

    def test_parent_mismatches_before_native_parser(self):
        with patch.object(sampler.elf, 'inspect_elf_inputs', side_effect=AssertionError('unowned ELF parsed')):
            for key in ('proposal_sha256', 'request_sha256', 'fit_round_sha256', 'job_sha256', 'binary_sha256'):
                old = self.lock[key]; self.lock[key] = '0' * 64; self.freeze(); self.rejected()
                self.lock[key] = old

    def test_frozen_fit_binary_and_source_identity_must_match_job(self):
        old = copy.deepcopy(self.fit)
        for key, value in [('gpu_binary', str(self.base / 'other.bin')),
                           ('gpu_binary_sha256', '0' * 64), ('binary_product_source_sha256', {})]:
            self.fit = copy.deepcopy(old); self.fit[key] = value; self.freeze_fit(); self.freeze(); self.rejected()

    def test_request_canonical_numeric_identity_and_round_mismatch_reject(self):
        req = json.loads(self.request.read_bytes())
        for key, value in [('fit_seed', 11.0), ('final', 0)]:
            modified = copy.deepcopy(req); modified['job'][key] = value
            self.request.write_text(json.dumps(modified, sort_keys=True))
            self.lock['request_sha256'] = sha(self.request); self.freeze(); self.rejected()
        req['round_sha256'] = '0' * 64; self.request.write_text(json.dumps(req))
        self.lock['request_sha256'] = sha(self.request); self.freeze(); self.rejected()

    def test_changed_binary_and_size_identity_reject(self):
        old = self.lock['binary_bytes']
        for value in (True, float(old), old - 1):
            self.lock['binary_bytes'] = value; self.freeze(); self.rejected()
        self.lock['binary_bytes'] = old; self.freeze()
        self.binary.write_bytes(b'changed native bytes'); self.rejected()

    def test_binary_alias_and_directory_reject(self):
        other = self.base / 'original.bin'; other.write_bytes(self.binary.read_bytes())
        self.binary.unlink(); self.binary.symlink_to(other); self.rejected()
        self.binary.unlink(); self.binary.mkdir(); self.rejected()

    def test_owned_declaration_replay_must_match(self):
        self.lock['declarations']['declarations']['needed'].append('unrecorded.so')
        self.freeze(); self.rejected()

    def test_regular_metadata_precedes_read_even_for_fifo(self):
        self.fit_path.unlink(); os.mkfifo(self.fit_path)
        original = Path.read_bytes
        def checked(path):
            if path == self.fit_path:
                self.fail('nonregular metadata read attempted')
            return original(path)
        with patch.object(Path, 'read_bytes', checked):
            self.rejected()

    def test_late_binary_alias_and_metadata_changes_reject(self):
        original = sampler.elf.inspect_elf_inputs
        paths = (self.binary, self.manifest, self.proposal_path, self.request, self.fit_path)
        for path in paths:
            blob = path.read_bytes()
            def changed(*args):
                result = original(*args); path.write_bytes(b'late opaque native change'); return result
            with self.subTest(path=path.name), patch.object(sampler.elf, 'inspect_elf_inputs', side_effect=changed):
                self.rejected()
            path.write_bytes(blob); self.verify()
        other = self.base / 'other.bin'; other.write_bytes(self.binary.read_bytes())
        def alias(*args):
            result = original(*args); self.binary.unlink(); self.binary.symlink_to(other); return result
        with patch.object(sampler.elf, 'inspect_elf_inputs', side_effect=alias):
            self.rejected()

    def test_original_and_final_deadline_before_receipt(self):
        with self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'):
            self.verify(now=700.0)
        with patch.object(sampler.bootstrap.time, 'monotonic', side_effect=[125.0, 700.0]), \
                self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'):
            sampler.verify_sampler_declarations(str(self.manifest), self.digest, batch_started_at=100.0)

    def test_proposal_io_timeout_is_fixed_rejection(self):
        original = Path.read_bytes; reads = []
        def timeout(path):
            if path == self.proposal_path:
                reads.append(True)
                if len(reads) == 2:
                    raise TimeoutError('opaque private native proposal')
            return original(path)
        with patch.object(Path, 'read_bytes', timeout):
            self.rejected()
        self.assertEqual(len(reads), 2)

    def test_exact_false_claims_and_counter_reject(self):
        old = copy.deepcopy(self.lock)
        for key, value in [('version', 1.0), ('candidate_processes_started', False),
                           ('execution_admitted', 0), ('loader_resolution_verified', True), ('ptf_v1', .99)]:
            self.lock = copy.deepcopy(old); self.lock[key] = value; self.freeze(); self.rejected()

    def test_final_helper_and_parent_aliases_same_bytes_reject(self):
        for module, key in ((sampler.elf, 'elf_helper_sha256'),
                            (sampler.bootstrap, 'bootstrap_helper_sha256'),
                            (sampler, 'sampler_helper_sha256')):
            for parent_alias in (False, True):
                with self.subTest(module=key, parent_alias=parent_alias):
                    root = self.base / (key + str(parent_alias)); root.mkdir()
                    helper = root / 'helper.py'; helper.write_bytes(Path(module.__file__).read_bytes())
                    other = self.base / (root.name + '-other'); other.mkdir()
                    (other / 'helper.py').write_bytes(helper.read_bytes())
                    prior = self.lock[key]; self.lock[key] = sha(helper); self.freeze()
                    with patch.object(module, '__file__', str(helper)):
                        self.verify()
                        original = sampler.elf.inspect_elf_inputs
                        def changed(*args):
                            result = original(*args)
                            if parent_alias:
                                helper.unlink(); root.rmdir(); root.symlink_to(other, target_is_directory=True)
                            else:
                                helper.unlink(); helper.symlink_to(other / 'helper.py')
                            return result
                        with patch.object(sampler.elf, 'inspect_elf_inputs', side_effect=changed): self.rejected()
                    self.lock[key] = prior; self.freeze(); self.verify()

    def test_final_metadata_fifo_rejects_before_hash(self):
        for path in (self.manifest, self.proposal_path, self.request, self.fit_path):
            with self.subTest(path=path.name):
                original = sampler.elf.inspect_elf_inputs
                original_sha = sampler.system.python.sha; blob = path.read_bytes()
                def changed(*args):
                    result = original(*args); path.unlink(); os.mkfifo(path); return result
                def guarded(p):
                    if p == path: self.fail('late nonregular metadata hash attempted')
                    return original_sha(p)
                with patch.object(sampler.elf, 'inspect_elf_inputs', side_effect=changed), \
                        patch.object(sampler.system.python, 'sha', side_effect=guarded): self.rejected()
                path.unlink(); path.write_bytes(blob); self.verify()

    def test_final_helper_fifo_rejects_before_hash(self):
        for module, key in ((sampler.elf, 'elf_helper_sha256'),
                            (sampler.bootstrap, 'bootstrap_helper_sha256'),
                            (sampler, 'sampler_helper_sha256')):
            with self.subTest(module=key):
                helper = self.base / (key + '-regular.py'); helper.write_bytes(Path(module.__file__).read_bytes())
                prior = self.lock[key]; self.lock[key] = sha(helper); self.freeze()
                with patch.object(module, '__file__', str(helper)):
                    self.verify()
                    original = sampler.elf.inspect_elf_inputs; original_sha = sampler.system.python.sha
                    def changed(*args):
                        result = original(*args); helper.unlink(); os.mkfifo(helper); return result
                    def guarded(path):
                        if path == helper and not path.is_file(): self.fail('late nonregular helper hash attempted')
                        return original_sha(path)
                    with patch.object(sampler.elf, 'inspect_elf_inputs', side_effect=changed), \
                            patch.object(sampler.system.python, 'sha', side_effect=guarded): self.rejected()
                self.lock[key] = prior; self.freeze(); self.verify()


if __name__ == '__main__':
    unittest.main()
