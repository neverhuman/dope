"""Opaque source inventories never execute their deliberately failing initializers."""
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark import container_guard_sources as guard


def sha(body):
    return hashlib.sha256(body).hexdigest()


class GuardSourceControls(unittest.TestCase):
    def setUp(self):
        Path('target').mkdir(exist_ok=True)
        directory = tempfile.TemporaryDirectory(prefix='guard-sources-', dir='target')
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name).resolve()
        self.source = self.base / 'source'
        self.source.mkdir(mode=0o700)
        self.helper = self.base / 'trusted-checker.py'
        self.helper.write_bytes(b'# opaque already trusted checker identity\n')
        self.helper.chmod(0o600)
        self.paths = {}
        for name in sorted(guard.SOURCE_NAMES):
            path = self.source / name
            path.write_bytes(b"raise AssertionError('staged initializer executed')\n# " + name.encode())
            path.chmod(0o400)
            self.paths[name] = path
        self.manifest = self.base / 'source.lock.json'
        self.lock = {'format': 'dope-startup-source-preparation', 'version': 1,
                     'source_directory': str(self.source),
                     'guard_source_helper_sha256': sha(self.helper.read_bytes()),
                     'source_files': {name: {'bytes': path.stat().st_size,
                        'sha256': sha(path.read_bytes()), 'uid': os.getuid(), 'mode': 0o400}
                        for name, path in self.paths.items()},
                     'execution_admitted': False, 'full_runtime_closure_certified': False,
                     'official_tests_opened': False, 'candidate_processes_started': 0,
                     'mfs_v2': None, 'ptf_v1': None, 'release_safe': None, 'superiority': None}
        for name, value in (('BASE', self.base), ('__file__', str(self.helper))):
            state = patch.object(guard, name, value); state.start(); self.addCleanup(state.stop)
        state = patch.object(guard.time, 'monotonic', return_value=125.0)
        state.start(); self.addCleanup(state.stop); self.clock = guard.time.monotonic
        self.freeze()
        self.verify()

    def freeze(self):
        self.manifest.write_text(json.dumps(self.lock, sort_keys=True))
        self.manifest.chmod(0o600)
        self.expected = sha(self.manifest.read_bytes())

    def verify(self):
        return guard.verify_startup_sources(str(self.manifest), self.expected, batch_started_at=100.0)

    def rejected(self):
        with self.assertRaisesRegex(ValueError, '^compressed guard source custody rejected$'):
            self.verify()

    def test_owned_bytes_and_null_nonexecution_receipt(self):
        result = self.verify()
        self.assertIs(type(result.sources), tuple)
        self.assertEqual(dict(result.sources), {name: path.read_bytes() for name, path in self.paths.items()})
        self.assertEqual(result.outer_seconds_remaining, 575.0)
        receipt = result.receipt()
        self.assertEqual(receipt['source_files_verified'], 15)
        for key in ('execution_admitted', 'sources_imported', 'full_runtime_closure_certified', 'official_tests_opened'):
            self.assertIs(receipt[key], False)
        for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority'):
            self.assertIsNone(receipt[key])
        self.assertEqual(receipt['candidate_processes_started'], 0)

    def test_declared_names_match_current_relative_import_closure(self):
        root = Path(__file__).resolve().parents[1]
        pending, reached = ['container_startup.py'], set()
        while pending:
            name = pending.pop()
            if name in reached:
                continue
            reached.add(name)
            for node in ast.walk(ast.parse((root / name).read_bytes())):
                if isinstance(node, ast.ImportFrom) and node.level == 1:
                    pending.extend([node.module + '.py'] if node.module else
                                   [alias.name + '.py' for alias in node.names])
        self.assertEqual(reached, guard.SOURCE_NAMES)

    def test_manifest_digest_drift(self):
        self.manifest.write_bytes(self.manifest.read_bytes() + b' ')
        self.rejected()

    def test_digest_subclass_rejects_before_open(self):
        class Impostor(str): pass
        with patch.object(Path, 'open', side_effect=AssertionError('unexpected open')):
            with self.assertRaisesRegex(ValueError, '^compressed guard source custody rejected$'):
                guard.verify_startup_sources(str(self.manifest), Impostor(self.expected), batch_started_at=100.0)

    def test_nonexecution_and_numeric_identities(self):
        for key, bad in (('version', True), ('candidate_processes_started', False),
                         ('execution_admitted', 0), ('mfs_v2', 0.99)):
            with self.subTest(key=key):
                old = self.lock[key]; self.lock[key] = bad; self.freeze(); self.rejected()
                self.lock[key] = old; self.freeze()

    def test_omitted_inventory_member(self):
        name = sorted(self.paths)[0]
        self.lock['source_files'].pop(name); self.freeze(); self.rejected()

    def test_added_file_or_directory(self):
        extra = self.source / 'extra'
        extra.write_bytes(b'opaque'); self.rejected(); extra.unlink()
        extra.mkdir(); self.rejected()

    def test_identical_source_alias(self):
        path = self.paths['container_startup.py']
        twin = self.base / 'same-bytes.py'; twin.write_bytes(path.read_bytes())
        path.unlink(); path.symlink_to(twin); self.rejected()

    def test_source_fifo_rejected_before_open(self):
        path = self.paths['container_startup.py']; path.unlink(); os.mkfifo(path, 0o400)
        opened = Path.open
        def check(candidate, *args, **kwargs):
            if candidate == path: raise AssertionError('FIFO source opened')
            return opened(candidate, *args, **kwargs)
        with patch.object(Path, 'open', check): self.rejected()

    def test_manifest_fifo_rejected_before_open(self):
        self.manifest.unlink(); os.mkfifo(self.manifest, 0o600)
        with patch.object(Path, 'open', side_effect=AssertionError('FIFO manifest opened')): self.rejected()

    def test_directory_alias(self):
        saved = self.base / 'original-source'; self.source.rename(saved); self.source.symlink_to(saved)
        self.rejected()

    def test_invalid_source_row_numeric_types(self):
        row = self.lock['source_files']['container_startup.py']
        for key, bad in (('bytes', True), ('uid', float(os.getuid())), ('mode', float(0o400))):
            with self.subTest(key=key):
                old = row[key]; row[key] = bad; self.freeze(); self.rejected()
                row[key] = old; self.freeze()

    def test_late_source_drift_rejected(self):
        path = self.paths['container_startup.py']; original = guard.read_regular; calls = 0
        def drift(candidate):
            nonlocal calls
            if candidate == path:
                calls += 1
                if calls == 2: path.chmod(0o600); path.write_bytes(b'changed source')
            return original(candidate)
        with patch.object(guard, 'read_regular', side_effect=drift): self.rejected()
        self.assertEqual(calls, 2)

    def test_helper_fifo_rejected_before_open(self):
        self.helper.unlink(); os.mkfifo(self.helper, 0o600)
        opened = Path.open
        def check(candidate, *args, **kwargs):
            if candidate == self.helper: raise AssertionError('FIFO helper opened')
            return opened(candidate, *args, **kwargs)
        with patch.object(Path, 'open', check): self.rejected()

    def test_deadline_before_any_read(self):
        self.clock.return_value = 700.0
        with patch.object(Path, 'open', side_effect=AssertionError('expired source read')):
            with self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'):
                self.verify()

    def test_deadline_after_final_manifest_read(self):
        original = guard.read_regular; calls = 0
        def expire(candidate):
            nonlocal calls
            result = original(candidate)
            if candidate == self.manifest:
                calls += 1
                if calls == 2: self.clock.return_value = 700.0
            return result
        with patch.object(guard, 'read_regular', side_effect=expire):
            with self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'):
                self.verify()
        self.assertEqual(calls, 2)

    def test_io_timeout_is_generic_integrity_rejection(self):
        original = guard.read_regular
        def broken(candidate):
            if candidate == self.paths['container_startup.py']: raise TimeoutError('private source detail')
            return original(candidate)
        with patch.object(guard, 'read_regular', side_effect=broken): self.rejected()

    def test_later_path_change_does_not_replace_owned_bytes(self):
        result = self.verify(); old = dict(result.sources)['container_startup.py']
        path = self.paths['container_startup.py']; path.chmod(0o600); path.write_bytes(b'new initializer')
        self.assertEqual(dict(result.sources)['container_startup.py'], old)
        self.rejected()


if __name__ == '__main__': unittest.main(verbosity=2)
