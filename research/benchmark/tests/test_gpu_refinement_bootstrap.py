"""Generated initializer controls; no native libraries, fits or real inputs."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark import gpu_refinement_bootstrap as b


class Controls(unittest.TestCase):
    def setUp(self):
        target = Path(__file__).resolve().parents[3] / 'target'
        target.mkdir(exist_ok=True)
        area = tempfile.TemporaryDirectory(dir=target, prefix='gpu-startup-controls-')
        self.addCleanup(area.cleanup)
        self.root = Path(area.name).resolve()
        self.source = self.root / 'source'
        self.source.mkdir()
        self.marker = self.root / 'initializer'
        self.common_marker = self.root / 'common-initializer'
        self.common = self.source / 'common.py'
        self.entry = self.source / 'entry.py'
        self.common.write_text('from pathlib import Path\nPath(' + repr(str(self.common_marker))
                               + ').write_text("initialized")\ndef check(expected):\n    pass\n')
        self.entry.write_text('from pathlib import Path\nimport common,sys\nPath('
                              + repr(str(self.marker)) + ').write_text(sys.argv[1])\n')
        self.lock = dict(official_tests_opened=False, full_campaign_admitted=False,
                         mfs_v2=None, ptf_v1=None, release_safe=None, superiority=None,
                         source_files={str(p): self.sha(p) for p in (self.common, self.entry)})
        self.freeze()

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def freeze(self):
        (self.root / 'round.lock.json').write_text(json.dumps(self.lock))
        self.pin = self.sha(self.root / 'round.lock.json')
        (self.root / 'execution.lock.json').write_text(json.dumps(dict(round_sha256=self.pin)))

    def invoke(self):
        with patch.dict(sys.modules), patch.object(sys, 'argv', ['control']):
            b.run(str(self.entry), self.pin, ['checked'])

    def reject(self):
        with self.assertRaises((ValueError, OSError, KeyError)):
            self.invoke()
        self.assertFalse(self.common_marker.exists())
        self.assertFalse(self.marker.exists())

    def test_positive_initializers_and_arguments(self):
        self.invoke()
        self.assertEqual(self.marker.read_text(), 'checked')
        self.assertTrue(self.common_marker.exists())

    def test_source_drift_before_both_initializers(self):
        self.entry.write_text(self.entry.read_text() + '# changed\n')
        self.reject()

    def test_common_drift_before_both_initializers(self):
        self.common.write_text(self.common.read_text() + '# changed\n')
        self.reject()

    def test_local_repin_cannot_override_external_pin(self):
        original = self.pin
        self.common.write_text(self.common.read_text() + '# changed\n')
        self.lock['source_files'][str(self.common)] = self.sha(self.common)
        self.freeze()
        self.pin = original
        self.reject()

    def test_checked_source_buffer_is_executed_after_replacement(self):
        original = Path.read_bytes
        def replace_after_read(path):
            data = original(path)
            if path == self.entry:
                self.entry.write_text('raise RuntimeError("replacement executed")\n')
            return data
        with patch.object(Path, 'read_bytes', replace_after_read):
            self.invoke()
        self.assertEqual(self.marker.read_text(), 'checked')

    def test_added_bytecode_is_rejected(self):
        (self.source / 'entry.pyc').write_bytes(b'executable')
        self.reject()

    def test_added_empty_directory_is_rejected(self):
        (self.source / 'extra').mkdir()
        self.reject()

    def test_added_directory_alias_is_rejected(self):
        (self.source / 'alias').symlink_to(self.root, target_is_directory=True)
        self.reject()

    def test_round_parent_alias_is_rejected(self):
        alias = self.root / 'round-alias'
        alias.symlink_to(self.root, target_is_directory=True)
        self.entry = alias / 'source' / 'entry.py'
        self.reject()

    def test_execution_digest_disagrees(self):
        (self.root / 'execution.lock.json').write_text(json.dumps(dict(round_sha256='a' * 64)))
        self.reject()

    def test_digest_subclass_is_rejected(self):
        class Digest(str):
            pass
        self.pin = Digest(self.pin)
        self.reject()

    def test_duplicate_round_key_is_rejected_even_when_hash_pinned(self):
        path = self.root / 'round.lock.json'
        path.write_text('{"official_tests_opened":true,' + path.read_text()[1:])
        self.pin = self.sha(path)
        (self.root / 'execution.lock.json').write_text(json.dumps(dict(round_sha256=self.pin)))
        self.reject()

    def test_publication_and_test_claims_are_rejected(self):
        for key in ('official_tests_opened', 'full_campaign_admitted', 'mfs_v2', 'ptf_v1',
                    'release_safe', 'superiority'):
            with self.subTest(key=key):
                original = self.lock[key]
                self.lock[key] = True
                self.freeze()
                self.reject()
                self.lock[key] = original

    def test_nonfinite_round_is_rejected_before_initializers(self):
        path = self.root / 'round.lock.json'
        path.write_text('{"extra":1e999,' + path.read_text()[1:])
        self.pin = self.sha(path)
        (self.root / 'execution.lock.json').write_text(json.dumps(dict(round_sha256=self.pin)))
        self.reject()

    def test_entry_syntax_error_prevents_both_initializers(self):
        self.entry.write_bytes(b'if invalid syntax\n')
        self.lock['source_files'][str(self.entry)] = self.sha(self.entry)
        self.freeze()
        with self.assertRaises(SyntaxError): self.invoke()
        self.assertFalse(self.common_marker.exists())
        self.assertFalse(self.marker.exists())

    def test_loader_rejects_a_different_module_identity(self):
        loader = b.CheckedSourceLoader('common', self.common, self.common.read_bytes())
        with self.assertRaises(ValueError): loader.get_code('different')
        self.assertFalse(self.common_marker.exists())


if __name__ == '__main__':
    unittest.main()
