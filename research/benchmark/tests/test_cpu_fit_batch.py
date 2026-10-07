"""Frozen fit-ledger integrity controls; no generator or official test is run."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from research.benchmark import publish_cpu_fit_batch as publisher


class CpuFitBatch(unittest.TestCase):
    def setUp(self):
        base=Path(__file__).resolve().parents[3]/'target/cpu-fit-ledger-controls'
        base.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.artifact=self.root/'artifact';self.artifact.mkdir()
        model=self.artifact/'model.json';model.write_text('{}\n')
        projection=self.artifact/'projection.json';projection.write_text('{}\n')
        inventory={p.name:dict(bytes=p.stat().st_size,sha256=self.sha(p)) for p in self.artifact.iterdir()}
        runtime=self.root/'runtime.lock.json';runtime.write_text('{}\n')
        source=self.root/'worker.py';source.write_text('# fixture source\n')
        self.job=dict(method='ARF',dataset='fixture-lineage',fit_seed=23,
                      configuration=dict(num_trees=30),configuration_labels=['author_default'],
                      worker=dict(files={'train.csv':'1'*64,'validation.csv':'2'*64,'projection.json':'3'*64}))
        jid=publisher.identity(self.job)
        job=self.save('job.json',self.job)
        round_lock=dict(jobs=[dict(job_sha256=jid,file_sha256=job['sha256'])],
                        official_tests_opened=False,source_files={str(source):self.sha(source)},
                        runtime_sha256=self.sha(runtime),cpu_affinity=[60,61,62,63],fit_timeout_seconds=600)
        round_ref=self.save('round.lock.json',round_lock)
        self.fit=dict(job_sha256=jid,round_sha256=round_ref['sha256'],fit_seed=23,
                      dataset='fixture-lineage',cpu_affinity=[60,61,62,63],status='ok',
                      elapsed_seconds=10.0,artifact_bytes=6,artifact_inventory=inventory,
                      native_kpi=dict(objective='heldout_forde_mean_log_density',direction='maximize',
                                      partition='validation',fit_seed=23,validation_sha256='2'*64,value=1.5),
                      official_tests_opened=False,mfs_v2=None,ptf_v1=None,release_safe=None,superiority=None)
        fit=self.save('fit.json',self.fit)
        close=self.save('close.json',dict(round_sha256=round_ref['sha256'],fit_receipt_sha256=fit['sha256'],worker_exit_code=0))
        self.manifest=dict(format='dope-cpu-fit-batch-input-v1',snapshot_utc='2026-10-07T07:00:00Z',
                           planned_new_physical_fits={'ARF':796},fits=[dict(round=round_ref,job=job,fit=fit,close=close)])

    def sha(self,path):return hashlib.sha256(path.read_bytes()).hexdigest()

    def save(self,name,value):
        p=self.root/name;p.write_text(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')
        return dict(path=str(p),sha256=self.sha(p))

    def refresh_fit(self):
        e=self.manifest['fits'][0];e['fit']=self.save('fit.json',self.fit)
        e['close']=self.save('close.json',dict(round_sha256=e['round']['sha256'],fit_receipt_sha256=e['fit']['sha256'],worker_exit_code=0))

    def test_measured_prefix_never_claims_complete_matrix_or_shared_metric(self):
        p=publisher.build(self.manifest)
        self.assertEqual(p['physical_fit_counts'],{'ARF':1})
        self.assertEqual(p['fits'][0]['native_validation_value'],1.5)
        self.assertFalse(p['five_fit_matrix_complete']);self.assertFalse(p['common_metrics_complete'])
        self.assertIsNone(p['mfs_v2']);self.assertIsNone(p['ptf_v1'])
        self.assertIn('fixture-lineage,23,ok,1.5,6,10.0,',publisher.render(p))

    def test_rewritten_receipt_rejected_before_metric_decoding(self):
        (self.root/'fit.json').write_text('{malformed changed receipt')
        with self.assertRaisesRegex(ValueError,'frozen receipt drift'):publisher.build(self.manifest)

    def test_rewritten_model_and_projection_rejected(self):
        for name in ('model.json','projection.json'):
            with self.subTest(name=name):
                p=self.artifact/name;old=p.read_bytes();p.write_bytes(old+b' ')
                with self.assertRaisesRegex(ValueError,'frozen receipt drift'):publisher.build(self.manifest)
                p.write_bytes(old)

    def test_omitted_projection_charge_rejected(self):
        del self.fit['artifact_inventory']['projection.json'];self.fit['artifact_bytes']=3;self.refresh_fit()
        with self.assertRaisesRegex(ValueError,'complete artifact accounting'):publisher.build(self.manifest)

    def test_source_changed_after_receipt_rejected(self):
        (self.root/'worker.py').write_text('# changed source')
        with self.assertRaisesRegex(ValueError,'frozen receipt drift'):publisher.build(self.manifest)

    def test_duplicate_fit_does_not_add_independent_evidence(self):
        self.manifest['fits']*=2
        with self.assertRaisesRegex(ValueError,'duplicate physical fit'):publisher.build(self.manifest)

    def test_gated_score_rejected(self):
        self.fit['mfs_v2']=.99;self.refresh_fit()
        with self.assertRaisesRegex(ValueError,'unsupported publication claim'):publisher.build(self.manifest)

    def test_added_artifact_alias_rejected(self):
        (self.artifact/'extra').symlink_to(self.artifact/'model.json')
        with self.assertRaisesRegex(ValueError,'artifact inventory differs'):publisher.build(self.manifest)

    def test_noncanonical_job_seed_rejected(self):
        self.job['fit_seed']=23.0;self.manifest['fits'][0]['job']=self.save('job.json',self.job)
        with self.assertRaisesRegex(ValueError,'job not in frozen matrix'):publisher.build(self.manifest)

    def test_wrong_validation_kpi_rejected(self):
        self.fit['native_kpi']['validation_sha256']='4'*64;self.refresh_fit()
        with self.assertRaisesRegex(ValueError,'native density evidence differs'):publisher.build(self.manifest)

    def test_committed_batch_schemas_tables_and_hashes(self):
        import jsonschema
        repo=Path(__file__).resolve().parents[3]
        root=repo/'research/benchmark/results/cpu-fivefit-progress-v1'
        manifest=json.loads((root/'manifest.json').read_text())
        jsonschema.validate(manifest,json.loads((root/'manifest.schema.json').read_text()))
        for name, row in manifest['files'].items():
            path=repo/name
            self.assertEqual(self.sha(path),row['sha256'])
            self.assertEqual(path.stat().st_size,row['bytes'])
        schema=json.loads((root/'panel.schema.json').read_text())
        for method in ('arf','forest'):
            panel=json.loads((root/method/'panel.json').read_text())
            jsonschema.validate(panel,schema)
            self.assertEqual(publisher.render(panel),(root/method/'fits.csv').read_text())
            self.assertEqual(sum(panel['physical_fit_counts'].values()),len(panel['fits']))
            self.assertFalse(panel['five_fit_matrix_complete'])

    def test_committed_complete_arf_additional_fit_matrix(self):
        import collections
        import jsonschema
        repo=Path(__file__).resolve().parents[3]
        root=repo/'research/benchmark/results/arf-complete-additional-fits-v1'
        manifest=json.loads((root/'manifest.json').read_text())
        jsonschema.validate(manifest,json.loads((root/'manifest.schema.json').read_text()))
        for name,row in manifest['files'].items():
            path=repo/name
            self.assertEqual(self.sha(path),row['sha256'])
            self.assertEqual(path.stat().st_size,row['bytes'])
        panel=json.loads((root/'panel.json').read_text())
        jsonschema.validate(panel,json.loads((root/'panel.schema.json').read_text()))
        self.assertEqual(publisher.render(panel),(root/'fits.csv').read_text())
        rows=panel['fits']
        self.assertEqual(len(rows),796)
        self.assertEqual(len({r['job_sha256'] for r in rows}),796)
        self.assertEqual({r['status'] for r in rows},{'ok'})
        self.assertEqual(len({r['dataset'] for r in rows}),100)
        self.assertEqual(collections.Counter(r['fit_seed'] for r in rows),
                         {23:199,37:199,53:199,71:199})
        logical={(r['dataset'],r['fit_seed'],label) for r in rows
                 for label in r['configuration_labels']}
        self.assertEqual(len(logical),800)
        self.assertEqual(collections.Counter((seed,label) for _,seed,label in logical),
                         {(seed,label):100 for seed in (23,37,53,71)
                          for label in ('author_default','native_selected')})
        self.assertEqual(sum(len(r['configuration_labels'])-1 for r in rows),4)
        self.assertEqual(sum(r['artifact_bytes'] for r in rows),
                         manifest['measured_costs']['sum_charged_artifact_bytes'])
        self.assertAlmostEqual(sum(r['elapsed_seconds'] for r in rows),
                               manifest['measured_costs']['summed_worker_fit_and_validation_seconds'])
        self.assertEqual({r['native_objective'] for r in rows},
                         {'heldout_forde_mean_log_density'})
        self.assertFalse(panel['common_metrics_complete'])
        self.assertFalse(panel['five_fit_matrix_complete'])
        self.assertFalse(panel['full_campaign_admitted'])
        self.assertTrue(manifest['additional_fit_matrix_complete'])
        self.assertFalse(manifest['native_kpis_ranked_across_methods'])
        self.assertFalse(manifest['official_tests_opened'])
        for gate in ('mfs_v2','ptf_v1','release_safe','superiority'):
            self.assertIsNone(manifest[gate])
            self.assertIsNone(panel[gate])



if __name__=='__main__':unittest.main()
