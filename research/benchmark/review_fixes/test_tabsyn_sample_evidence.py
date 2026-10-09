"""Independent sample-byte evidence from a complete frozen grid, on toys."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.benchmark.review_fixes import tabsyn_sample_evidence as evidence


class SampleEvidenceTests(unittest.TestCase):
    def setUp(self):
        target = Path(__file__).resolve().parents[3] / 'target/sample-evidence-tests'
        target.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=target)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = 'ab' * 8
        self.fits = {'official_tests_opened': False, 'config_sha256': evidence.CONFIG_SHA256,
                     'cells': [{'dataset_id': self.dataset, 'status': 'fit_ok',
                                'actual_fit_exit': 0, 'fit_seed': 11}]}

    def capture(self):
        raw = json.dumps(self.fits).encode()
        with patch.object(evidence, 'FITS_SHA256', hashlib.sha256(raw).hexdigest()):
            return evidence.capture(raw, self.root)

    def test_real_file_digest_is_independent_of_any_metric_field(self):
        path = self.root / self.dataset / 'sample-101-1n.csv'
        path.parent.mkdir()
        raw = b'a,b\n0,0\n1,1\n'; path.write_bytes(raw)
        cells = self.capture()['cells']
        self.assertEqual(len(cells), 6)
        success = [c for c in cells if c['status'] == 'ok']
        self.assertEqual(len(success), 1)
        self.assertEqual(success[0]['synthetic_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(success[0]['bytes'], len(raw))
        self.assertEqual(len([c for c in cells if c['status'] == 'unavailable']), 5)

    def test_missing_lineage_has_every_planned_unavailable_disposition(self):
        cells = self.capture()['cells']
        self.assertEqual({(c['size'], c['sample_seed']) for c in cells},
                         {(s, k) for s in (1, 4) for k in (101, 211, 307)})
        self.assertTrue(all(c['status'] == 'unavailable' and c['reason'] == 'sample_csv_absent'
                            and c['synthetic_sha256'] is None for c in cells))

    def test_changed_registry_refuses_before_decode(self):
        with patch.object(evidence.json, 'loads') as decode:
            with self.assertRaisesRegex(ValueError, 'digest differs'):
                evidence.capture(b'changed registry', self.root)
            decode.assert_not_called()

    def test_foreign_dataset_and_nonliteral_closed_flag_refuse(self):
        self.fits['cells'][0]['dataset_id'] = 'worker'
        with self.assertRaisesRegex(ValueError, 'outside the frozen'):
            self.capture()
        self.fits['cells'][0]['dataset_id'] = self.dataset
        for flag in (True, 1, 0, None):
            self.fits['official_tests_opened'] = flag
            with self.assertRaisesRegex(ValueError, 'flag'):
                self.capture()

    def test_redirected_sample_or_parent_refuses(self):
        destination = self.root / 'outside.csv'; destination.write_bytes(b'opaque toy')
        parent = self.root / self.dataset; parent.mkdir()
        path = parent / 'sample-101-1n.csv'; path.symlink_to(destination)
        with self.assertRaisesRegex(ValueError, 'redirected'):
            self.capture()
        path.unlink(); parent.rmdir()
        parent.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'redirected'):
            self.capture()


if __name__ == '__main__':
    unittest.main()
