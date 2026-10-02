"""A complete frozen receipt set is required before publishing shared outcomes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark import publish_s3_forest as panel
from research.benchmark.score import sha256


def cells():
    return [{'dataset': 'fixture', 'method': method, 'configuration': config, 'fit_seed': 11,
             'sample_seed': seed, 'size_multiplier': size, 'status': 'ok',
             'charged_artifact_bytes': 100, 'counts_as_dope_win': False,
             'mfs_v2': None, 'ptf_v1': None, 'release_safe_l3': None,
             'utility': {auditor: {'retention': value, 'informative': True} for auditor in panel.AUDITORS}}
            for method, config in panel.CONFIGS for size in panel.SIZES
            for seed, value in zip(panel.SEEDS, (.5, .7, .9))]


class S3ForestPanelTests(unittest.TestCase):
    def workspace(self):
        target = Path('target/s3-forest-panel-tests')
        target.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=target)
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name).resolve()

    def test_complete_all_profiles_and_native_default_panel(self):
        result = panel.summarize(cells(), ['fixture'])
        self.assertEqual(len(result), 20)
        self.assertTrue(all(r['utility']['catboost']['median_retention'] == .7 for r in result))

    def test_missing_or_duplicate_sample_is_rejected(self):
        for rows in (cells()[1:], cells() + [cells()[0]]):
            with self.subTest(rows=len(rows)), self.assertRaisesRegex(ValueError, 'incomplete|duplicated'):
                panel.summarize(rows, ['fixture'])

    def test_failed_and_noninformative_samples_keep_null_medians(self):
        for mode in ('failure', 'low_signal'):
            rows = cells()
            if mode == 'failure': rows[0]['status'] = 'timeout'
            else: rows[0]['utility']['catboost']['informative'] = False
            result = panel.summarize(rows, ['fixture'])
            affected = next(r for r in result if (r['method'], r['configuration'], r['size_multiplier'])
                            == (rows[0]['method'], rows[0]['configuration'], rows[0]['size_multiplier']))
            self.assertIsNone(affected['utility']['catboost']['median_retention'])

    def test_production_score_and_win_claim_rejected(self):
        for name, value in (('mfs_v2', 99), ('ptf_v1', .99), ('release_safe_l3', True),
                            ('counts_as_dope_win', True), ('fit_seed', 23)):
            rows = cells(); rows[0][name] = value
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, 'scope changed'):
                panel.summarize(rows, ['fixture'])

    def test_artifact_charges_reconcile_across_sample_seeds(self):
        rows = cells(); rows[0]['charged_artifact_bytes'] += 1
        with self.assertRaisesRegex(ValueError, 'charges disagree'):
            panel.summarize(rows, ['fixture'])

    def test_external_anchor_rejects_receipt_rewrite_before_metric_reads(self):
        root = self.workspace()
        metric = root/'metric.json'; metric.write_text('{"private_fixture":true}')
        anchor = root/'receipt-lock-v1.json'
        anchor.write_text(json.dumps({'refs': {str(metric): sha256(metric)}}))
        expected = sha256(anchor)
        with patch.object(panel, 'BASE', root):
            metric.write_text('{"rewritten_fixture":true}')
            with patch.object(panel, 'read', wraps=panel.read) as reading:
                with self.assertRaisesRegex(ValueError, 'evidence changed'):
                    panel.anchored(root, expected, {})
                self.assertEqual([call.args[0] for call in reading.call_args_list], [anchor])

    def test_anchor_cannot_redefine_its_own_expected_digest(self):
        root = self.workspace(); anchor = root/'receipt-lock-v1.json'; anchor.write_text('{}')
        expected = sha256(anchor); anchor.write_text('{"refs":{}}')
        with patch.object(panel, 'BASE', root), patch.object(panel, 'read') as reading:
            with self.assertRaisesRegex(ValueError, 'evidence changed'):
                panel.anchored(root, expected, {})
            reading.assert_not_called()

    def test_native_runtime_rejects_executable_bytecode_and_directory_aliases(self):
        root = self.workspace(); runtime = root/'runtime'; runtime.mkdir()
        source = runtime/'source.py'; source.write_text('# opaque never-imported fixture\n')
        inventory = root/'runtime-inventory.lock.json'
        inventory.write_text(json.dumps({'roots': [str(runtime)], 'inventory_modes': {str(runtime): 'all'},
                                        'aliases': [], 'files': {str(source): {'sha256': sha256(source),
                                                                             'bytes': source.stat().st_size}}}))
        lock = {'runtime_inventory_sha256': sha256(inventory)}
        with patch.object(panel, 'BASE', root), patch.object(panel, 'NATIVE', root):
            self.assertEqual(panel.native_inventory(lock, {}), 1)
            added = runtime/'source.pyc'; added.write_bytes(b'opaque bytecode')
            with self.assertRaisesRegex(ValueError, 'inventory changed'):
                panel.native_inventory(lock, {})
            added.unlink()
            (runtime/'alias').symlink_to(runtime, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'alias added'):
                panel.native_inventory(lock, {})

    def test_failed_overbudget_or_changed_runtime_cannot_be_success(self):
        operation = {'status': 'ok', 'exit_code': 0, 'new_operation_started': True,
                     'foreign_processes_signaled': False, 'runtime_lock_files': {'runtime': 'fixed'},
                     'elapsed_seconds': 10., 'timeout_seconds': 600,
                     'peak_resident_bytes_including_coordinator': 100}
        receipt = {'official_tests_opened': False, 'new_generator_fits_started': 0,
                   'native_selection_changed': False, 'mfs_v2': None, 'ptf_v1': None,
                   'release_safe': None, 'superiority': None, 'status': 'ok', 'operation': operation}
        lock = {'runtime_lock_files': {'runtime': 'fixed'}, 'roles': {'coordinator': {'ram_bytes': 1000}}}
        panel.common_operation(receipt, lock)
        for name, value in (('status', 'timeout'), ('exit_code', 1), ('elapsed_seconds', 611),
                            ('timeout_seconds', 601), ('peak_resident_bytes_including_coordinator', 1001),
                            ('runtime_lock_files', {'runtime': 'changed'}), ('foreign_processes_signaled', True)):
            modified = copy.deepcopy(receipt); modified['operation'][name] = value
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, 'runtime or budget gate'):
                panel.common_operation(modified, lock)

    def test_manifest_rejects_rewritten_schema_tables_and_figures(self):
        from research.benchmark import publish_s3_forest_manifest as manifest
        root = self.workspace()
        report = {'official_tests_opened': False, 'production_certified': False,
                  'mfs_v2': None, 'ptf_v1': None, 'release_safe_l3': None,
                  'paired_superiority': None, 'cells': [], 'dataset_ids': [], 'summary': []}
        expected = {'.json': json.dumps(report).encode(),
                    '.schema.json': json.dumps(panel.schema(report)).encode(),
                    '.csv': b'csv', '.md': b'markdown', '.svg': b'svg', '.pdf': b'pdf'}
        for suffix, value in expected.items():
            (root/(manifest.NAME+suffix)).write_bytes(value)
        with patch.object(manifest, 'RESULTS', root), patch.object(manifest, 'build_report', return_value=report), \
                patch.object(manifest, 'tables', return_value=('csv', 'markdown')), \
                patch.object(manifest, 'render', side_effect=lambda r, kind: kind.encode()):
            for suffix in ('.schema.json', '.csv', '.md', '.svg', '.pdf'):
                path = root/(manifest.NAME+suffix)
                path.write_bytes(b'{}' if suffix == '.schema.json' else b'rewritten')
                try:
                    with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, 'differs|differ'):
                        manifest.build()
                finally:
                    path.write_bytes(expected[suffix])


if __name__ == '__main__': unittest.main()
