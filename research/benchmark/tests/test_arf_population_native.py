"""Generated native ledgers exercise publication without rows or learner imports."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark import publish_arf_population_native as m


class LedgerControls(unittest.TestCase):
    def setUp(self):
        self.lock = dict(planned_fit_jobs=800, jobs=[], native_grid=[dict(num_trees=10+i) for i in range(8)],
            native_objective=dict(implementation_role='study_evaluation_of_author_FORDE_leaf_mixture_density'),
            official_tests_opened=False)
        self.receipts = {}; self.selected = []
        for d in range(100):
            dataset = f'{d:016x}'; jobs = []; rows = []
            for t in range(8):
                config = self.lock['native_grid'][t]
                job = dict(dataset=dataset, method='ARF', track='common_numeric', dp_budget=None,
                    fit_seed=11, trial_index=t, configuration=config, configuration_sha256=m.digest(config),
                    worker=dict(files={'validation.csv': 'a'*64}))
                key = m.digest(job)
                metric = dict(objective='heldout_forde_mean_log_density', direction='maximize',
                    partition='validation', fit_seed=11, value=float(t-20), validation_sha256='a'*64,
                    not_comparable_across_methods=True)
                r = dict(job=job, job_sha256=key, round_sha256=m.ROUND, status='ok', elapsed_seconds=2.,
                    shared_outcomes_used_for_selection=False, official_tests_opened=False,
                    mfs_v2=None, ptf_v1=None, release_safe=None, superiority=None,
                    result=dict(native_kpi=metric, artifact_bytes=20000,
                        artifact_inventory={'model.json':dict(bytes=19990), 'projection.json':dict(bytes=10)}),
                    operation=dict(host='xbabe2', new_operation_started=True),
                    common_retention=1e6 if t==0 else -1e6)
                self.lock['jobs'].append(dict(job=job, job_sha256=key))
                self.receipts[key]=r; jobs.append(job); rows.append(r)
            self.selected.append(dict(dataset=dataset, fit_seed=11,
                author_default_job_sha256=m.digest(jobs[0]), **m.selection.select(jobs,rows)))
        self.report = dict(complete_native_matrix=True, planned_fit_jobs=800, closed_fit_jobs=800,
            dataset_lineages=100, actual_coordinator_exit_code=0, status_counts={'ok':800},
            default_native_cells=self.selected, operation_seconds=1600., scheduler_wall_seconds=2000.,
            official_tests_opened=False, shared_outcomes_used_for_selection=False)

    def reject(self):
        with self.assertRaises((ValueError, KeyError)):
            m.assemble(self.lock,self.report,self.receipts)

    def test_complete_and_native_only(self):
        result=m.assemble(self.lock,self.report,self.receipts)
        self.assertEqual(len(result['trials']),800); self.assertEqual(len(result['cells']),100)
        self.assertTrue(all(c['selected_native_kpi']['value']==-13 for c in result['cells']))
        self.assertEqual(result['selected_artifacts_within_l3_byte_cap'],0)
        self.assertTrue(all(result[k] is v for k,v in m.GATES.items()))
        self.assertIn('FORDE',m.markdown(result)); self.assertEqual(len(m.table(result).splitlines()),101)

    def test_missing_trial(self):
        self.receipts.pop(next(iter(self.receipts)));self.reject()

    def test_partial_matrix(self):
        self.report['closed_fit_jobs']=799;self.reject()

    def test_rewritten_selection(self):
        self.report['default_native_cells'][0]['selected_job_sha256']='f'*64;self.reject()

    def test_bad_cost(self):
        self.report['operation_seconds']+=10;self.reject()

    def test_nonboolean_gate(self):
        self.report['official_tests_opened']=0;self.reject()

    def test_production_score(self):
        self.report['ptf_v1']=.999;self.reject()

    def test_float_canonical_seed(self):
        self.lock['jobs'][0]['job']['fit_seed']=11.0;self.reject()

    def test_float_trial_index(self):
        self.lock['jobs'][0]['job']['trial_index']=0.0;self.reject()

    def test_method_copy_replaced(self):
        key=next(iter(self.receipts));self.receipts[key]['job']=copy.deepcopy(self.receipts[key]['job'])
        self.receipts[key]['job']['configuration']['num_trees']=999;self.reject()

    def test_objective_change(self):
        key=next(iter(self.receipts));self.receipts[key]['result']['native_kpi']['objective']='shared_retention';self.reject()

    def test_projection_bytes_counted(self):
        key=next(iter(self.receipts));self.receipts[key]['result']['artifact_bytes']=19990;self.reject()


class InputControls(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[3]/'target';root.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=root);self.base=Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(patch.stopall);patch.object(m,'BASE',self.base).start()
        patch.object(m,'ROOT',self.base/'native').start();m.ROOT.mkdir()

    def test_digest_subclass_rejected(self):
        class Digest(str): pass
        with self.assertRaises(ValueError):m.bound(self.base/'absent',Digest('a'*64))

    def test_sealed_input_rejected(self):
        p=self.base/'test.csv';p.write_text('never read')
        with self.assertRaises(ValueError):m.safe(p)

    def test_added_directory_or_symlink_rejected(self):
        root=self.base/'flat';root.mkdir()
        p=root/'one';p.write_text('owned');files={'one':m.sha256(p)}
        m.flat(root,files)
        (root/'extra').mkdir()
        with self.assertRaises(ValueError):m.flat(root,files)
        (root/'extra').rmdir();(root/'alias').symlink_to(p)
        with self.assertRaises(ValueError):m.flat(root,files)

    def test_rewritten_evidence_rejected_before_report_decoding(self):
        p=self.base/'metric.json';p.write_text('{}');expected=m.sha256(p)
        anchor=dict(complete_native_matrix=True,round_sha256=m.ROUND,
            reconciliation_sha256='b'*64,refs={str(p):expected})
        data=(json.dumps(anchor)+'\n').encode();a=m.ROOT/'receipt-lock-v1.json';a.write_bytes(data)
        pin=hashlib.sha256(data).hexdigest();p.write_text('{"rewritten":true}')
        with patch.object(m.guard,'decode',wraps=m.guard.decode) as decode:
            with self.assertRaisesRegex(ValueError,'frozen native evidence changed'):
                m.verify_inputs(pin,'b'*64)
            self.assertEqual(decode.call_count,1)  # Only the receipt lock is decoded.

    def test_only_frozen_empty_runtime_directory_admitted(self):
        out=self.base/'attempt';out.mkdir();(out/'receipt.json').write_text('{}')
        temp=out/'runtime-temp';temp.mkdir();m.attempt_inventory(out,{})
        (out/'extra').mkdir()
        with self.assertRaises(ValueError):m.attempt_inventory(out,{})
        (out/'extra').rmdir();(temp/'cache').write_text('unexpected')
        with self.assertRaises(ValueError):m.attempt_inventory(out,{})

    def test_special_attempt_entry_rejected(self):
        import os
        out=self.base/'attempt';out.mkdir();(out/'receipt.json').write_text('{}')
        (out/'runtime-temp').mkdir();os.mkfifo(out/'pipe')
        with self.assertRaises(ValueError):m.attempt_inventory(out,{})

    def test_path_contract_hash_precedes_decoding(self):
        with patch.object(m,'PATH_CONTRACT','a'*64), patch.object(m.guard,'decode') as decode:
            with self.assertRaisesRegex(ValueError,'runtime manifest changed'):m.directory_name()
            decode.assert_not_called()


if __name__=='__main__':unittest.main()
