"""Fit custody rejects rewritten receipts, byte laundering and partial claims."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import hashlib,json,unittest

from research.benchmark import publish_dope_population_fits as pub
from research.benchmark import publish_s3_forest as paths

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def cells():
 return [{'dataset':f'{d:016x}','profile':p,'fit_seed':11,'status':'ok','new_operation_seconds':1.0,
          'counts_as_dope_win':False,'mfs_v2':None,'ptf_v1':None,'release_safe_l3':None,'paired_superiority':None}
         for d in range(100) for p in pub.PROFILES]

class PopulationFits(unittest.TestCase):
 def test_complete_matrix_requires_all_four_profiles(self):
  rows=cells();self.assertEqual(sum(x['closed'] for x in pub.summarize(rows)),400)
  with self.assertRaises(ValueError):pub.summarize(rows[:-1])
  with self.assertRaises(ValueError):pub.summarize(rows[:-1]+[rows[0]])
 def test_no_gated_claim_from_fit_counts(self):
  for key,value in (('mfs_v2',.99),('ptf_v1',.99),('release_safe_l3',True),('paired_superiority',True),('counts_as_dope_win',True),('fit_seed',23)):
   rows=cells();rows[0][key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):pub.summarize(rows)
 def test_external_receipt_digest_precedes_report_read(self):
  with TemporaryDirectory() as d:
   base=Path(d);root=base/'round';root.mkdir();receipt=root/'attempt.json';receipt.write_text('{"status":"ok"}')
   (root/'round.lock.json').write_text('{}');rh=sha(root/'round.lock.json')
   (root/'reconciliation-v1.json').write_text('{"measured":true}');report_sha=sha(root/'reconciliation-v1.json')
   anchor={'round_sha256':rh,'official_tests_opened':False,'mfs_v2':None,'ptf_v1':None,
           'refs':{str(receipt):sha(receipt)},'reconciliation_sha256':report_sha}
   p=root/'receipt-lock-v1.json';p.write_text(json.dumps(anchor));ah=sha(p)
   with patch.object(paths,'BASE',base):
    self.assertTrue(pub.anchored(root,rh,ah,report_sha)[1]['measured'])
    receipt.write_text('{"status":"ok","rewritten":true}')
    with self.assertRaises(ValueError):pub.anchored(root,rh,ah,report_sha)
    anchor['refs'][str(receipt)]=sha(receipt);p.write_text(json.dumps(anchor))
    with self.assertRaises(ValueError):pub.anchored(root,rh,ah,report_sha)
 def test_receipt_anchor_directory_alias_is_rejected(self):
  with TemporaryDirectory() as d:
   base=Path(d);real=base/'real';real.mkdir();root=base/'alias';root.symlink_to(real,target_is_directory=True)
   p=real/'round.lock.json';p.write_text('{}')
   with patch.object(paths,'BASE',base),self.assertRaises(ValueError):pub.anchored(root,sha(p),'0'*64,'0'*64)
 def test_reconciliation_rewrite_is_rejected(self):
  with TemporaryDirectory() as d:
   base=Path(d);root=base/'round';root.mkdir();(root/'round.lock.json').write_text('{}');rh=sha(root/'round.lock.json')
   p=root/'reconciliation-v1.json';p.write_text('{}');h=sha(p)
   a=root/'receipt-lock-v1.json';a.write_text(json.dumps({'round_sha256':rh,'official_tests_opened':False,'mfs_v2':None,'ptf_v1':None,'refs':{},'reconciliation_sha256':h}));ah=sha(a)
   p.write_text('{"closed_fit_cells":400}')
   with patch.object(paths,'BASE',base),self.assertRaises(ValueError):pub.anchored(root,rh,ah,h)
 def test_tables_roundtrip_without_carriage_return_or_scores(self):
  rows=cells();report={'cells':rows,'summary':pub.summarize(rows),'cost':{'new_fit_operation_seconds':400.,'coordinator_wall_seconds':1000.}}
  for r in rows:r.update(kind='new_fit',charged_artifact_bytes=100,model_bytes=90,projection_bytes=10)
  csv,md=pub.tables(report);self.assertNotIn('\r',csv);self.assertEqual(len(csv.splitlines()),401);self.assertIn('release-safe/superiority null',md)
 def fit_cell(self,status,alias=False):
  with TemporaryDirectory() as d:
   base=Path(d);root=base/'round';worker=base/'worker';worker.mkdir()
   for name in ('train.csv','validation.csv','projection.json','row-group-assignments.json','worker-manifest.json'):(worker/name).write_text('{}')
   model=base/'model.dpk';model.write_bytes(b'x'*10239)
   job={'dataset':'fixture','research_profile':pub.PROFILES[0],'seed':11,'final':False,'worker':{'path':str(worker),'files':{p.name:sha(p) for p in worker.iterdir()}}}
   key=pub.digest(job);out=root/'attempts'/key/'attempt-0001';out.mkdir(parents=True)
   receipt={'job':job,'round_sha256':pub.ROUND_SHA,'status':status,'kind':'new_fit','official_tests_opened':False,'counts_as_dope_win':False,'mfs_v2':None,'ptf_v1':None,'evidence_files':{}}
   p=out/'receipt.json';p.write_text(json.dumps(receipt))
   row={'job_sha256':key,'dataset':'fixture','profile':pub.PROFILES[0],'fit_seed':11,'receipt_sha256':sha(p),'status':status,'kind':'new_fit',
        'model_path':str(model),'model_sha256':sha(model),'charged_artifact_bytes':10241,'projection_bytes_included':True,'gpu_target_operator_verified':status=='ok',
        'new_operation_seconds':1.0,'host_operation_seconds':{'xbabe1':1.0},'historical_runtime_closure_upgraded':False}
   refs={str(p):sha(p),str(model):sha(model),**{str(p):sha(p) for p in worker.iterdir()}}
   if alias:
    link=base/'linked-model.dpk';link.symlink_to(model);row['model_path']=str(link)
   with patch.object(paths,'BASE',base),patch.object(pub,'ROOT',root):return pub.cell(job,row,refs)
 def test_projection_charge_failure_remains_a_failure(self):
  row=self.fit_cell('charged_artifact_cap');self.assertEqual(row['model_bytes'],10239);self.assertEqual(row['projection_bytes'],2)
  self.assertFalse(row['artifact_within_l3_cap']);self.assertFalse(row['counts_as_dope_win'])
  with self.assertRaises(ValueError):self.fit_cell('ok')
 def test_linked_model_is_rejected(self):
  with self.assertRaises(ValueError):self.fit_cell('charged_artifact_cap',alias=True)

if __name__=='__main__':unittest.main()
