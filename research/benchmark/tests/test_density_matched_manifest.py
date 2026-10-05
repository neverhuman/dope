"""Publication pins must reject drift before artifact JSON is read."""
import unittest
from unittest.mock import patch

from research.benchmark import publish_density_matched_manifest as m


class Controls(unittest.TestCase):
    def test_figure_digest_subclass_rejected_before_file_reads(self):
        class Derived(str): pass
        with patch.object(m.Path, 'read_bytes') as read:
            with self.assertRaises(ValueError): m.build('opaque', 'opaque/figures', Derived('a' * 64))
            read.assert_not_called()

    def test_changed_figure_receipt_rejected_before_any_report_decode(self):
        with patch.object(m.Path, 'is_file', return_value=True), \
             patch.object(m.Path, 'is_symlink', return_value=False), \
             patch.object(m.Path, 'read_bytes', return_value=b'changed opaque receipt'), \
             patch.object(m.json, 'loads') as decode, \
             patch.object(m.publisher, 'load_report') as report:
            with self.assertRaises(ValueError): m.build('opaque', 'opaque/figures', 'a' * 64)
            decode.assert_not_called()
            report.assert_not_called()

    def test_figure_receipt_link_rejected_before_read(self):
        with patch.object(m.Path, 'is_file', return_value=True), \
             patch.object(m.Path, 'is_symlink', return_value=True), \
             patch.object(m.Path, 'read_bytes') as read:
            with self.assertRaises(ValueError): m.build('opaque', 'opaque/figures', 'a' * 64)
            read.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
