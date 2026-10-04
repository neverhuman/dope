"""Opaque closure and file-custody controls; no live benchmark input is read."""
import copy
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

    def test_top_level_report_win_claim_rejected(self):
        self.values[1]['counts_as_dope_win'] = True
        with self.assertRaises(ValueError): self.call()

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

    def test_missing_auxiliary_reference_rejected_before_other_json_reads(self):
        key = 'a' * 64
        out = self.root / 'attempts' / key / 'attempt-0001'
        name = 'catboost_info/learn_error.tsv'
        h = self.put(out / name, b'opaque log')
        other = self.root / 'opaque.json'
        other_h = self.put(other, b'{"opaque":true}')
        for refs, declared_sha in (({str(other): other_h}, h),
                                   ({str(other): other_h, str(out / name): h}, 'c' * 64)):
            anchor = dict(round_sha256=m.ROUND, reconciliation_sha256='b' * 64,
                native_receipt_lock_sha256=m.NATIVE, official_tests_opened=False,
                mfs_v2=None, ptf_v1=None, refs=refs,
                attempt_auxiliary_trees={key: dict(directories=['catboost_info'],
                    files=[dict(path=name, sha256=declared_sha, bytes=10)])})
            receipt_sha = self.put(self.root / 'receipt-lock-v1.json', json.dumps(anchor).encode())
            with patch.object(m, 'bound_json', wraps=m.bound_json) as read:
                with self.assertRaisesRegex(ValueError, 'auxiliary bytes unbound'):
                    m.load_closed_inputs(receipt_sha, 'b' * 64)
                self.assertEqual(read.call_count, 1)
                self.assertEqual(read.call_args.args[0], self.root / 'receipt-lock-v1.json')

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
                status='ok', operation=dict(new_operation_started=True),
                evidence_files={'request.json': h}, official_tests_opened=False,
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

            original_receipts = dict(receipts)
            auxiliary = {}
            for job in lock['jobs']:
                key = m.digest_identity(job)
                out = self.root / 'attempts' / key / 'attempt-0001'
                for name in m.AUXILIARY_DIRECTORIES:
                    (out / name).mkdir(parents=True, exist_ok=True)
                rows = [dict(path=name, bytes=10, sha256=put(out / name, b'opaque log'))
                        for name in sorted(m.AUXILIARY_FILES)]
                auxiliary[key] = dict(files=rows, directories=sorted(m.AUXILIARY_DIRECTORIES))
            # Old flat anchors remain strict; auxiliary files/dirs require a new
            # separate anchor while every original receipt stays byte-identical.
            with self.assertRaisesRegex(ValueError, 'tree inventory'):
                m.load_closed_inputs(receipt_sha, report_sha)
            anchor['attempt_auxiliary_trees'] = auxiliary
            receipt_sha = self.put(self.root / 'receipt-lock-v1.json', json.dumps(anchor).encode())
            result = m.load_closed_inputs(receipt_sha, report_sha)
            self.assertEqual(result['auxiliary_bytes'], 502 * 40)
            self.assertFalse(result['original_receipts_changed'])
            self.assertFalse(result['auxiliary_bytes_included_in_generator_charge'])
            for key, expected in original_receipts.items():
                self.assertEqual(hashlib.sha256((self.root / 'attempts' / key / 'attempt-0001' /
                                               'receipt.json').read_bytes()).hexdigest(), expected)
            first = next(iter(auxiliary))
            declaration = auxiliary.pop(first)
            h = self.put(self.root / 'receipt-lock-v1.json', json.dumps(anchor).encode())
            with self.assertRaisesRegex(ValueError, 'physical coverage'): m.load_closed_inputs(h, report_sha)
            auxiliary[first] = declaration
            receipt_sha = self.put(self.root / 'receipt-lock-v1.json', json.dumps(anchor).encode())
            extra = self.root / 'attempts' / first / 'attempt-0001' / 'runtime-temp' / 'extra'
            extra.mkdir()
            with self.assertRaisesRegex(ValueError, 'tree inventory'):
                m.load_closed_inputs(receipt_sha, report_sha)
            extra.rmdir()
            log = self.root / 'attempts' / first / 'attempt-0001' / declaration['files'][0]['path']
            log.write_bytes(b'changed')
            with patch.object(m, 'bound_json', wraps=m.bound_json) as read:
                with self.assertRaisesRegex(ValueError, 'frozen evidence changed'):
                    m.load_closed_inputs(receipt_sha, report_sha)
                self.assertEqual(read.call_count, 1)


class AuxiliaryPathContractControls(unittest.TestCase):
    def test_immutable_names_and_counts(self):
        path = Path(m.__file__).with_name('density-auxiliary-paths.lock.json')
        dirs, files = m.auxiliary_paths(path.read_bytes())
        self.assertEqual((len(dirs), len(files)), (5, 4))
        self.assertIn('runtime-temp', dirs)
        self.assertIn('sampler.empty-cache', dirs)
        self.assertEqual((dirs, files), (m.AUXILIARY_DIRECTORIES, m.AUXILIARY_FILES))

    def test_changed_contract_rejected_before_json_decode(self):
        with patch.object(m.json, 'loads') as parse:
            for data in (b'{}', b'{"directories":[],"files":[]}', b'\xff'):
                with self.assertRaisesRegex(ValueError, 'path contract changed'):
                    m.auxiliary_paths(data)
            parse.assert_not_called()


class AuxiliaryControls(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3] / 'target');self.addCleanup(self.tmp.cleanup)
  self.base=Path(self.tmp.name).resolve();self.out=self.base/'attempt';self.out.mkdir()
  p=patch.object(m,'BASE',self.base);p.start();self.addCleanup(p.stop)
  self.refs={};self.receipt=dict(operation=dict(new_operation_started=True),evidence_files={})
  self.put('receipt.json',b'opaque receipt')
  self.receipt['evidence_files']['request.json']=self.put('request.json',b'opaque request')
  for name in sorted(m.AUXILIARY_DIRECTORIES):(self.out/name).mkdir(parents=True,exist_ok=True)
  files=[]
  for name in sorted(m.AUXILIARY_FILES):
   h=self.put(name,b'opaque log');files.append(dict(path=name,sha256=h,bytes=10))
  self.d=dict(files=files,directories=sorted(m.AUXILIARY_DIRECTORIES))
 def put(self,name,data):
  p=self.out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
  h=hashlib.sha256(data).hexdigest();self.refs[str(p)]=h;return h
 def call(self):return m.attempt_tree(self.out,self.receipt,self.refs,self.d)
 def test_explicit_complete_tree(self):
  r=self.call();self.assertEqual((len(r[0])-2,len(r[1]),r[2]),(4,5,40))
 def test_absent_auxiliary_on_unstarted(self):
  self.receipt['operation']=None
  with self.assertRaises(ValueError):self.call()
 def test_empty_unstarted_tree_is_accepted(self):
  for row in self.d['files']:(self.out/row['path']).unlink()
  for name in sorted(self.d['directories'],key=lambda x:len(Path(x).parts),reverse=True):(self.out/name).rmdir()
  self.receipt['operation']=None;self.d=dict(files=[],directories=[])
  r=self.call();self.assertEqual((len(r[0])-2,r[2]),(0,0))
 def test_unknown_directory_and_file_declarations(self):
  original=copy.deepcopy(self.d)
  for value in ('added','catboost_info/../added','/absolute'):
   self.d=copy.deepcopy(original);self.d['directories'].append(value)
   with self.subTest(value=value),self.assertRaises(ValueError):self.call()
  self.d=copy.deepcopy(original);self.d['files'][0]['path']='runtime-temp/unknown.bin'
  with self.assertRaises(ValueError):self.call()
 def test_added_empty_dirs_cache_files_and_logs(self):
  for name in ('extra/','sampler.empty-cache/extra','runtime-temp/extra','catboost_info/extra'):
   p=self.out/name
   if name.endswith('/'):p.mkdir()
   else:p.write_bytes(b'opaque extra')
   with self.subTest(name=name),self.assertRaises(ValueError):self.call()
   if p.is_dir():p.rmdir()
   else:p.unlink()
 def test_parent_missing_and_duplicate_directory(self):
  old=self.d['directories'][:]
  self.d['directories'].remove('catboost_info/learn')
  with self.assertRaises(ValueError):self.call()
  self.d['directories']=old+['catboost_info/learn']
  with self.assertRaises(ValueError):self.call()
 def test_bytes_hash_and_reference_binding(self):
  for key,value in (('bytes',True),('bytes',10.),('bytes',11),('sha256','e'*64)):
   old=self.d['files'][0][key];self.d['files'][0][key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):self.call()
   self.d['files'][0][key]=old
  p=self.out/self.d['files'][0]['path'];p.write_bytes(b'changed')
  with self.assertRaises(ValueError):self.call()
 def test_alias_and_special_entry(self):
  p=self.out/'catboost_info'/'alias';p.symlink_to(self.out,target_is_directory=True)
  with self.assertRaises(ValueError):self.call()
  p.unlink();os.mkfifo(p)
  with self.assertRaises(ValueError):self.call()
 def test_duplicate_file_and_derived_path_type(self):
  self.d['files'].append(dict(self.d['files'][0]))
  with self.assertRaises(ValueError):self.call()
  self.d['files'].pop()
  class Derived(str):pass
  self.d['files'][0]['path']=Derived(self.d['files'][0]['path'])
  with self.assertRaises(ValueError):self.call()


if __name__ == '__main__':
    unittest.main(verbosity=2)
