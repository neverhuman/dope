"""Opaque closure and file-custody controls; no live benchmark input is read."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark import density_publication_inputs as m


def metadata():
    flags = dict(official_tests_opened=False, native_selection_changed=False,
        global_family_selected=False, new_generator_fits_started=0,
        full_campaign_admitted=False, mfs_v2=None, ptf_v1=None, release_safe=None, superiority=None)
    jobs, cells, batches = [], [], []
    for i in range(100):
        for index, method in enumerate(('GaussianCopula', 'independent_marginals', 'Chow-Liu')):
            shared = i * 3 + index < 98
            previous = None
            for configuration in ('default', 'native_selected'):
                if configuration == 'default' or not shared:
                    job = dict(dataset=f'opaque-{i}', method=method, configuration_name=configuration,
                        fit_seed=11, final=False, track='common-numeric', sample_seeds=[101, 211, 307],
                        size_multipliers=[1, 4], artifact_bytes=7)
                    key = m.digest_identity(job)
                    jobs.append(job)
                    batches.append(dict(physical_job_sha256=key, status='ok', receipt_sha256='a'*64))
                    previous = key
                for seed in (101, 211, 307):
                    for size in (1, 4):
                        cells.append(dict(dataset=f'opaque-{i}', method=method, configuration=configuration,
                            fit_seed=11, sample_seed=seed, size_multiplier=size, artifact_bytes=7,
                            counts_as_dope_win=False, physical_job_sha256=previous))
    lock = flags | dict(jobs=jobs, logical_cells=cells)
    report = flags | dict(round_sha256=m.ROUND, complete_matrix=True, datasets=100,
        logical_sample_cells=3600, physical_batches_planned=502, physical_batches_closed=502,
        physical_status_counts={'ok': 502}, batches=batches,
        cells=[c | dict(status='ok', validation_receipt_sha256='a'*64) for c in cells])
    completion = dict(round_sha256=m.ROUND, jobs=502, logical_cells=3600, counts={'ok': 502})
    end = dict(round_sha256=m.ROUND, exit_code=0)
    return lock, report, completion, end


class MetadataControls(unittest.TestCase):
    def setUp(self):
        self.values = metadata()

    def call(self):
        return m.closed_metadata(*self.values)

    def test_complete_expected_matrix(self):
        jobs, batches = self.call()
        self.assertEqual((len(jobs), len(batches)), (502, 502))

    def test_request_identity_uses_canonical_digest_not_python_equality(self):
        job = dict(fit_seed=11, final=False)
        key = m.digest_identity(job)
        receipt = dict(job=job, round_sha256=m.ROUND, status='ok')
        request = dict(job=dict(job), round_sha256=m.ROUND)
        m.request_identity(receipt, request, key, 'ok')
        for field, value in (('fit_seed', 11.0), ('final', 0)):
            request['job'] = job | {field: value}
            self.assertEqual(request['job'], job)
            with self.assertRaises(ValueError): m.request_identity(receipt, request, key, 'ok')

    def test_partial_or_duplicate_batch_rejected(self):
        report = self.values[1]
        report['batches'].pop()
        with self.assertRaises(ValueError): self.call()
        self.values = metadata()
        report = self.values[1]
        report['batches'][-1] = dict(report['batches'][0])
        with self.assertRaises(ValueError): self.call()

    def test_partial_or_duplicate_logical_cell_rejected(self):
        for change in ('missing', 'duplicate'):
            self.values = metadata()
            rows = self.values[1]['cells']
            if change == 'missing': rows.pop()
            else: rows[-1] = dict(rows[0])
            with self.assertRaises(ValueError): self.call()

    def test_numeric_aliases_rejected(self):
        for key, value in (('fit_seed', 11.0), ('sample_seeds', [101.0, 211, 307]),
                           ('size_multipliers', [True, 4]), ('final', 0)):
            self.values = metadata()
            self.values[0]['jobs'][0][key] = value
            with self.assertRaises(ValueError): self.call()
        for key in ('sample_seed', 'size_multiplier', 'fit_seed', 'artifact_bytes'):
            self.values = metadata()
            self.values[1]['cells'][0][key] = float(self.values[1]['cells'][0][key])
            with self.assertRaises(ValueError): self.call()

    def test_closure_counts_are_exact_integers(self):
        for index, key, value in ((1, 'physical_batches_closed', 501), (1, 'datasets', 100.0),
                                  (2, 'jobs', 502.0), (3, 'exit_code', False)):
            self.values = metadata()
            self.values[index][key] = value
            with self.assertRaises(ValueError): self.call()

    def test_changed_round_status_or_receipt_rejected(self):
        for index, key, value in ((2, 'round_sha256', 'b'*64), (3, 'exit_code', 1),
                                  (1, 'physical_status_counts', {'ok': 502.0})):
            self.values = metadata()
            self.values[index][key] = value
            with self.assertRaises(ValueError): self.call()
        for key, value in (('status', 'failed'), ('validation_receipt_sha256', 'b'*64)):
            self.values = metadata()
            self.values[1]['cells'][0][key] = value
            with self.assertRaises(ValueError): self.call()

    def test_native_choice_bytes_or_identity_rewrite_rejected(self):
        for key, value in (('artifact_bytes', 8), ('configuration', 'changed'),
                           ('dataset', 'substitute'), ('physical_job_sha256', 'f'*64)):
            self.values = metadata()
            self.values[1]['cells'][0][key] = value
            with self.assertRaises((ValueError, KeyError)): self.call()

    def test_scores_selection_and_win_claim_rejected(self):
        for key, value in (('ptf_v1', .99), ('official_tests_opened', True),
                           ('native_selection_changed', True), ('new_generator_fits_started', False),
                           ('complete_matrix', 1), ('full_campaign_admitted', True)):
            self.values = metadata()
            self.values[1][key] = value
            with self.assertRaises(ValueError): self.call()
        self.values = metadata()
        self.values[1]['cells'][0]['counts_as_dope_win'] = True
        with self.assertRaises(ValueError): self.call()


class FileControls(unittest.TestCase):
    def setUp(self):
        target = Path(__file__).resolve().parents[3] / 'target'
        self.temp = tempfile.TemporaryDirectory(dir=target)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'round'
        self.root.mkdir()
        for key, value in (('BASE', self.base), ('ROOT', self.root)):
            patcher = patch.object(m, key, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def put(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def test_external_digest_types_rejected_before_reads(self):
        class Derived(str): pass
        with patch.object(m, 'owned') as read:
            for a, b in ((Derived('a'*64), 'b'*64), ('a'*64, Derived('b'*64)),
                          ('A'*64, 'b'*64), ('a'*63, 'b'*64)):
                with self.assertRaises(ValueError): m.load_closed_inputs(a, b)
            read.assert_not_called()

    def test_owned_bytes_are_the_parsed_bytes(self):
        path = self.root / 'control.json'
        h = self.put(path, b'{"owned":true}')
        original = Path.read_bytes
        def changing(p):
            data = original(p)
            p.write_bytes(b'{"owned":false}')
            return data
        with patch.object(Path, 'read_bytes', changing):
            self.assertEqual(m.bound_json(path, h), {'owned': True})
        with self.assertRaises(ValueError): m.bound_json(path, h)

    def test_duplicate_and_nonfinite_json_rejected(self):
        path = self.root / 'control.json'
        for data in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}', b'\xff'):
            with self.assertRaises(ValueError): m.bound_json(path, self.put(path, data))

    def test_added_empty_directory_file_link_and_special_entry_rejected(self):
        tree = self.root / 'tree'
        path = tree / 'declared.bin'
        h = self.put(path, b'opaque')
        m.exact_tree(tree, [path])
        m.verify_refs({str(path): h})
        extra = tree / 'extra'
        for kind in ('directory', 'file', 'link', 'fifo'):
            if kind == 'directory': extra.mkdir()
            elif kind == 'file': extra.write_bytes(b'extra')
            elif kind == 'link': extra.symlink_to(path)
            else: os.mkfifo(extra)
            with self.assertRaises(ValueError): m.exact_tree(tree, [path])
            if kind == 'directory': extra.rmdir()
            else: extra.unlink()

    def test_aliased_roots_and_sealed_test_paths_rejected(self):
        path = self.root / 'control.json'
        h = self.put(path, b'{}')
        alias = self.base / 'alias'
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError): m.bound_json(alias / path.name, h)
        for path in (self.root / 'test.csv', self.root / 'evaluator' / 'metadata.json'):
            with self.assertRaises(ValueError): m.bound_json(path, self.put(path, b'{}'))

    def test_changed_recursive_reference_rejected_before_report_parse(self):
        dependency = self.root / 'source.py'
        h = self.put(dependency, b'opaque dependency')
        report_sha = 'b'*64
        anchor = dict(round_sha256=m.ROUND, reconciliation_sha256=report_sha,
                      native_receipt_lock_sha256=m.NATIVE, official_tests_opened=False,
                      mfs_v2=None, ptf_v1=None, refs={str(dependency): h})
        receipt_sha = self.put(self.root / 'receipt-lock-v1.json', json.dumps(anchor).encode())
        dependency.write_bytes(b'changed dependency')
        with patch.object(m, 'bound_json', wraps=m.bound_json) as read:
            with self.assertRaisesRegex(ValueError, 'frozen evidence changed'):
                m.load_closed_inputs(receipt_sha, report_sha)
            self.assertEqual(read.call_count, 1)

    def test_rewritten_anchor_and_wrong_report_digest_reject_before_refs(self):
        path = self.root / 'receipt-lock-v1.json'
        h = self.put(path, b'{}')
        path.write_bytes(b'{"changed":true}')
        with patch.object(m, 'verify_refs') as read:
            with self.assertRaises(ValueError): m.load_closed_inputs(h, 'b'*64)
            read.assert_not_called()
        anchor = dict(round_sha256=m.ROUND, reconciliation_sha256='c'*64)
        h = self.put(path, json.dumps(anchor).encode())
        with patch.object(m, 'verify_refs') as read:
            with self.assertRaises(ValueError): m.load_closed_inputs(h, 'b'*64)
            read.assert_not_called()

    def test_complete_anchored_file_inputs_remain_unadmitted(self):
        lock, report, completion, end = metadata()
        refs = {}
        def put(path, data):
            h = self.put(path, data)
            refs[str(path)] = h
            return h
        def put_json(path, value):
            return put(path, json.dumps(value, sort_keys=True).encode())
        source = self.root / 'source' / 'opaque.py'
        source_sha = put(source, b'opaque; never executed')
        worker = self.base / 'worker'
        worker_sha = put(worker / 'validation.csv', b'opaque; never parsed')
        artifact = self.base / 'artifact'
        inventory = [dict(path=name, bytes=len(data), sha256=put(artifact / name, data))
                     for name, data in (('model.json', b'abc'), ('projection.json', b'defg'))]
        native_report = self.base / 'native' / 'reconciliation-v1.json'
        native_report_sha = put_json(native_report, {'opaque': True})
        native_anchor = native_report.with_name('receipt-lock-v1.json')
        native_sha = put_json(native_anchor, dict(reconciliation_sha256=native_report_sha,
                                                refs={str(native_report): native_report_sha}))
        keys, batch_by_key = {}, {b['physical_job_sha256']: b for b in report['batches']}
        for job in lock['jobs']:
            old = m.digest_identity(job)
            job.update(worker=dict(path=str(worker), files={'validation.csv': worker_sha}),
                       artifact_path=str(artifact), artifact_inventory=inventory,
                       projection_bytes_included=True)
            key = m.digest_identity(job)
            keys[old] = key
            out = self.root / 'attempts' / key / 'attempt-0001'
            request_sha = put_json(out / 'request.json', dict(round_sha256='placeholder', job=job))
            receipt = dict(round_sha256='placeholder', job=job, status='ok',
                           evidence_files={'request.json': request_sha})
            put_json(out / 'receipt.json', receipt)
            batch_by_key[old].update(physical_job_sha256=key)
        for row in lock['logical_cells'] + report['cells']:
            row['physical_job_sha256'] = keys[row['physical_job_sha256']]
        runtime = self.root / 'runtime-inventory.lock.json'
        runtime_sha = put_json(runtime, {'opaque': True})
        lock.update(source_files={str(source): source_sha}, frozen_references={str(source): source_sha},
                    runtime_lock_files={str(runtime): runtime_sha}, runtime_inventory_sha256=runtime_sha,
                    native_receipt_lock_path=str(native_anchor))
        round_sha = put_json(self.root / 'round.lock.json', lock)
        # Physical requests name the final lock digest; they are not part of that lock.
        for job in lock['jobs']:
            key = m.digest_identity(job)
            out = self.root / 'attempts' / key / 'attempt-0001'
            h = put_json(out / 'request.json', dict(round_sha256=round_sha, job=job))
            receipt_sha = put_json(out / 'receipt.json', dict(round_sha256=round_sha, job=job,
                status='ok', evidence_files={'request.json': h}, official_tests_opened=False,
                native_selection_changed=False, global_family_selected=False,
                new_generator_fits_started=0, mfs_v2=None, ptf_v1=None,
                release_safe=None, superiority=None))
            batch = next(b for b in report['batches'] if b['physical_job_sha256'] == key)
            batch['receipt_sha256'] = receipt_sha
        receipts = {b['physical_job_sha256']: b['receipt_sha256'] for b in report['batches']}
        for row in report['cells']:
            row['validation_receipt_sha256'] = receipts[row['physical_job_sha256']]
        for v in (report, completion, end): v['round_sha256'] = round_sha
        put_json(self.root / 'completion.json', completion)
        put_json(self.root / 'coordinator-exit.json', end)
        put_json(self.root / 'supervisor-launch.json', dict(pid=2147483647, supervisor_pid=2147483646))
        report.update(native_receipt_lock_sha256=native_sha, native_reconciliation_sha256=native_report_sha)
        report_sha = self.put(self.root / 'reconciliation-v1.json', json.dumps(report).encode())
        anchor = dict(round_sha256=round_sha, reconciliation_sha256=report_sha,
            native_receipt_lock_sha256=native_sha, official_tests_opened=False,
            mfs_v2=None, ptf_v1=None, refs=refs)
        receipt_sha = self.put(self.root / 'receipt-lock-v1.json', json.dumps(anchor).encode())
        with patch.object(m, 'ROUND', round_sha), patch.object(m, 'NATIVE', native_sha), \
             patch.object(m, 'NATIVE_REPORT', native_report_sha):
            result = m.load_closed_inputs(receipt_sha, report_sha)
            self.assertFalse(result['publication_admitted'])
            self.assertTrue(result['metric_native_and_cost_replay_required'])
            self.assertFalse(result['full_runtime_closure_claimed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
