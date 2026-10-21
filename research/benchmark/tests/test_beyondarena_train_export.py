"""Synthetic refusal and noninterference controls; no real data or S3 requests."""
import copy
import datetime as dt
import io
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import time
import unittest

from research.benchmark import beyondarena_train_export as core
from research.benchmark import beyondarena_train_export_io as exporter


class TrainExportTests(unittest.TestCase):
    def setUp(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.table = pa.table({'target': [1., 99999., 2., 88888., 3., 4.],
                               'entity': ['a', 'b', 'c', 'd', 'a', 'c'],
                               'x': [10., 11., 12., 13., 14., 15.]})
        stream = io.BytesIO()
        pq.write_table(self.table, stream)
        self.raw = stream.getvalue()
        self.task = {'target_column_name': 'target', 'problem_type': 'regression',
                     'objective_metric_name': 'rmse', 'group_on': 'entity',
                     'time_on': None, 'group_time_on': None, 'group_labels': 'per_sample'}
        self.split = {'splits': {'0': {'0': [[0, 2, 4, 5], [1, 3]]}}}
        self.source = {'schema': 'jope.beyondarena-source/v1', 'benchmark': 'BeyondArena',
                       'name': 'fixture', 'source_family_id': 'beyondarena:fixture', 'task': 'regression',
                       'target': 'target', 'objective_metric': 'rmse', 'sampled': False,
                       'bucket': 'fixture-bucket', 'revision': 'a' * 40, 'rows': 6,
                       'columns': 3, 'license': 'CC BY', 'split_regime': 'grouped_non_iid',
                       'files': {'dataset.parquet': {'key': 'fixture.parquet', 'version_id': 'v1',
                                                    'bytes': len(self.raw), 'sha256': core.sha(self.raw)}}}
        self.calls = []
        self.rebind()

    def rebind(self):
        for name, data in [('task_metadata.predictive-ml-task-mold-v1.json', self.task),
                           ('experiment_metadata.predictive-ml-splits-mold-v1.json', self.split)]:
            self.source['files'][name] = {'sha256': core.sha(core.canonical(data))}
        self.source_raw = core.canonical(self.source)
        self.grant = {'format': 'dope-beyondarena-train-export-v1', 'role': 'trusted_data_exporter',
                      'optimizer_may_read_mixed_source': False, 'execution_authorized': True, 'exporter_uid': os.getuid(),
                      'started_unix': int(time.time()) - 1, 'expires_unix': int(time.time()) + 600,
                      'source_metadata_sha256': core.sha(self.source_raw),
                      'source_version': self.source['revision'], 'bucket': self.source['bucket'],
                      'raw_object': self.source['files']['dataset.parquet'],
                      'group_on': self.task['group_on'], 'time_on': self.task['time_on'],
                      'maximum_source_bytes': 32 * 2**20, 'producer_files': exporter.producer_files(),
                      'output_dir': str(Path(self.folder.name) / 'output'),
                      'train_membership_sha256': core.sha(core.canonical(self.split['splits']['0']['0'][0])),
                      'outer_test_membership_sha256': core.sha(core.canonical(self.split['splits']['0']['0'][1])),
                      'view': {'schema': 'jope.beyondarena-view/v1', 'benchmark': 'BeyondArena',
                               'id': 'fixture/r0/f0', 'source_family_id': self.source['source_family_id'],
                               'source_sha256': core.sha(self.source_raw), 'task': 'regression',
                               'target': 'target', 'metric': 'rmse', 'sampled': False,
                               'test_role': 'outer_test', 'repeat_id': '0', 'fold_id': '0',
                               'split_pointer': '/splits/0/0', 'train_rows': 4, 'test_rows': 2}}

    def transport(self, **kwargs):
        self.calls.append(kwargs)
        return {'VersionId': 'v1', 'ContentLength': len(self.raw), 'Body': io.BytesIO(self.raw)}

    def run_export(self, transport=None):
        raw = core.canonical(self.grant)
        return exporter.export(raw, core.sha(raw), self.source_raw,
                               core.canonical(self.task), core.canonical(self.split),
                               get_object=transport or self.transport)

    def test_legacy_array_shape_is_not_the_bound_source_schema(self):
        self.split = {"splits": [[[[0, 2, 4, 5], [1, 3]]]]}
        self.source["files"]["experiment_metadata.predictive-ml-splits-mold-v1.json"]["sha256"] = core.sha(core.canonical(self.split))
        self.source_raw = core.canonical(self.source)
        self.grant["source_metadata_sha256"] = core.sha(self.source_raw)
        self.grant["view"]["source_sha256"] = core.sha(self.source_raw)
        with self.assertRaisesRegex(core.ExportRefusal, "split_shape_invalid"):
            self.run_export()
        self.assertEqual(self.calls, [])

    def test_unsupported_group_label_modes_refuse_before_fetch(self):
        for label in [None, 'per_group', ['a', 'b'], {'a': 1}]:
            self.task['group_labels'] = label
            self.rebind()
            with self.subTest(label=label), self.assertRaisesRegex(core.ExportRefusal, 'unsupported_group_labels'):
                self.run_export()
            self.assertEqual(self.calls, [])
        self.task['group_labels'] = 'per_sample'
        self.task['group_on'] = None
        self.source['split_regime'] = 'iid'
        self.rebind()
        with self.assertRaisesRegex(core.ExportRefusal, 'unsupported_group_labels'):
            self.run_export()
        self.assertEqual(self.calls, [])

    def test_unhandled_cli_error_terminates_nonzero_without_source_diagnostics(self):
        command = [sys.executable, '-B', '-c',
                   'import sys; from research.benchmark.beyondarena_train_export_io '
                   'import terminal_error_hook; sys.excepthook = terminal_error_hook; '
                   'raise RuntimeError("PRIVATE_SOURCE_SENTINEL")']
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout),
                         {'status': 'REFUSED', 'code': 'unhandled_export_error'})
        self.assertEqual(result.stderr, '')
        self.assertNotIn('PRIVATE_SOURCE_SENTINEL', result.stdout + result.stderr)

    def test_roundtrip_only_TRAIN_target_last_and_private_files(self):
        import pyarrow.parquet as pq
        result = self.run_export()
        output = Path(self.grant['output_dir'])
        table = pq.read_table(output / 'official-train.parquet')
        self.assertEqual(table.column_names, ['x', 'target'])
        self.assertEqual(table['target'].to_pylist(), [1., 2., 3., 4.])
        membership = json.loads((output / 'membership.private.json').read_text())
        self.assertEqual(membership['official_TRAIN_original_row_indices'], [0, 2, 4, 5])
        partitions = membership['inner_partitions_in_export_order']
        keys = membership['group_or_time_keys']
        self.assertFalse({keys[i] for i in partitions['inner_fit']} &
                         {keys[i] for i in partitions['inner_validation']})
        self.assertEqual(result['license_recorded_verbatim'], 'CC BY')
        self.assertFalse(result['preprocessing_fitted'])
        self.assertFalse(result['independent_sealed_custody_proven'])
        self.assertTrue(result['mixed_parquet_physically_decoded_by_trusted_exporter'])
        self.assertEqual(self.calls, [{'Bucket': 'fixture-bucket', 'Key': 'fixture.parquet', 'VersionId': 'v1'}])
        for path in output.iterdir():
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_wrong_role_identity_deadline_version_and_group_refuse_before_fetch(self):
        original = copy.deepcopy(self.grant)
        for change in [{'execution_authorized': False}, {'role': 'optimizer'}, {'optimizer_may_read_mixed_source': True},
                       {'exporter_uid': os.getuid() + 1}, {'expires_unix': 1},
                       {'source_version': 'wrong'}, {'group_on': 'other'},
                       {'producer_files': {}}]:
            self.grant = {**copy.deepcopy(original), **change}
            with self.subTest(change=change), self.assertRaises(core.ExportRefusal):
                self.run_export()
            self.assertEqual(self.calls, [])

    def test_duplicate_or_mixed_memberships_refuse_before_fetch(self):
        for pair in [[[0, 0, 4, 5], [1, 3]], [[0, 2, 4, 5], [0, 3]],
                     [[0, 2, 4, True], [1, 3]], [[0, 2, 4, 7], [1, 3]]]:
            self.split = {'splits': {'0': {'0': pair}}}
            self.rebind()
            with self.assertRaises(core.ExportRefusal):
                self.run_export()
            self.assertEqual(self.calls, [])

    def test_unbound_split_hash_and_wrong_target_refuse(self):
        self.split['splits']['0']['0'][0].reverse()
        with self.assertRaisesRegex(core.ExportRefusal, 'metadata_hash_mismatch'):
            self.run_export()
        self.assertEqual(self.calls, [])
        self.split['splits']['0']['0'][0].reverse()
        self.grant['view']['target'] = 'x'
        with self.assertRaisesRegex(core.ExportRefusal, 'source_view_target_mismatch'):
            self.run_export()

    def test_wrong_transport_version_and_bytes_refuse(self):
        def wrong_version(**kwargs):
            result = self.transport(**kwargs)
            result['VersionId'] = 'v2'
            return result
        with self.assertRaisesRegex(core.ExportRefusal, 'transport_identity_mismatch'):
            self.run_export(wrong_version)
        self.raw = b'x' * len(self.raw)
        with self.assertRaisesRegex(core.ExportRefusal, 'raw_source_hash_mismatch'):
            self.run_export()
        self.assertFalse(Path(self.grant['output_dir']).exists())

    def test_schema_alias_and_group_overlap_are_refused(self):
        for table in [self.table.rename_columns(['target', 'x', 'x']),
                      self.table.select(['target', 'entity'])]:
            with self.assertRaises(core.ExportRefusal):
                exporter.prepare_table(table, self.source, self.task, [0, 2, 4, 5], [1, 3])
        with self.assertRaisesRegex(core.ExportRefusal, 'outer_group_overlap'):
            exporter.prepare_table(self.table, self.source, self.task, [0, 1, 2, 4], [3, 5])

    def test_outer_target_changes_do_not_change_TRAIN_rows_or_inner_partitions(self):
        import pyarrow as pa
        a, ma = exporter.prepare_table(self.table, self.source, self.task, [0, 2, 4, 5], [1, 3])
        changed = self.table.set_column(0, 'target', pa.array([1., -1e100, 2., 1e100, 3., 4.]))
        b, mb = exporter.prepare_table(changed, self.source, self.task, [0, 2, 4, 5], [1, 3])
        self.assertEqual(a, b)
        self.assertEqual(ma, mb)

    def test_temporal_blocks_preserved_and_reversed_outer_time_refused(self):
        import pyarrow as pa
        task = {**self.task, 'group_on': None, 'time_on': 'entity'}
        table = self.table.set_column(1, 'entity', pa.array([
            dt.date(2020, 1, day) for day in [1, 5, 2, 6, 3, 4]]))
        _, member = exporter.prepare_table(table, self.source, task, [0, 2, 4, 5], [1, 3])
        inner = member['inner_partitions_in_export_order']; keys = member['group_or_time_keys']
        self.assertLess(max(keys[i] for i in inner['inner_fit']), min(keys[i] for i in inner['inner_validation']))
        with self.assertRaisesRegex(core.ExportRefusal, 'outer_time_order_invalid'):
            exporter.prepare_table(table, self.source, task, [0, 1, 2, 4], [3, 5])
        with self.assertRaises(core.ExportRefusal):
            core.ordered_time('01/02/2020')

    def test_iid_feature_aliases_stay_together_without_using_target(self):
        import pyarrow as pa
        table = pa.table({'target': [1., 8., 2., 9., 100., 200.],
                          'entity': ['same', 'out', 'different', 'out', 'same', 'different'],
                          'x': [1., 4., 2., 5., 1., 2.]})
        task = {**self.task, 'group_on': None}
        _, member = exporter.prepare_table(table, self.source, task, [0, 2, 4, 5], [1, 3])
        keys = member['group_or_time_keys']
        self.assertEqual(keys[0], keys[2]); self.assertEqual(keys[1], keys[3])

    def test_existing_or_symlink_output_and_unbound_grant_refuse(self):
        output = Path(self.grant['output_dir']); output.symlink_to(self.folder.name, target_is_directory=True)
        with self.assertRaisesRegex(core.ExportRefusal, 'output_path_not_fresh_private'):
            self.run_export()
        self.assertEqual(self.calls, [])
        with self.assertRaises(core.ExportRefusal):
            core.bound_json(b'{"x":1,"x":2}', core.sha(b'{"x":1,"x":2}'))
        with self.assertRaises(core.ExportRefusal):
            core.bound_json(b'{}', '0' * 64)


if __name__ == '__main__':
    unittest.main()
