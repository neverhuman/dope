"""Publication controls: full denominator, partial groups null, no proxy selection."""
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from research.benchmark import publish_dope_population_validation as m
from research.benchmark import publish_s3_forest as paths

class Publication(unittest.TestCase):
    def setUp(self):
        self.rows=[{'dataset':f'lineage{i:03d}','profile':p,'sample_seed':s,'size_multiplier':z,
                    'fit_seed':11,'status':'ok','charged_artifact_bytes':1024,'counts_as_dope_win':False,
                    'utility':{a:{'informative':True,'retention':-2. if i==0 else 0.9} for a in m.AUDITORS},
                    'mfs_v2':None,'ptf_v1':None,'release_safe_l3':None,'paired_superiority':None}
                   for i in range(100) for p in m.PROFILES for s in m.SEEDS for z in m.SIZES]

    def test_complete_denominators_and_negative_retention_preserved(self):
        summary,panels=m.aggregate(self.rows)
        self.assertEqual(len(summary),800);self.assertEqual(len(panels),8)
        self.assertEqual(summary[0]['utility']['catboost']['median_retention'],-2.)
        self.assertTrue(all(p['utility']['catboost']['complete_informative_lineages']==100 for p in panels))

    def test_missing_and_duplicate_logical_cell_rejected(self):
        with self.assertRaises(ValueError):m.aggregate(self.rows[:-1])
        with self.assertRaises(ValueError):m.aggregate(self.rows+[self.rows[-1]])

    def test_incomplete_or_low_signal_group_remains_null_with_denominator(self):
        for status in ('fit_unavailable','timeout','auditor_failed','low_signal'):
            rows=copy.deepcopy(self.rows)
            if status in ('fit_unavailable','timeout'):
                rows[0]['status']=status;rows[0]['utility']=None
            elif status=='auditor_failed':rows[0]['utility']['catboost']={'status':'failed','error_type':'ValueError'}
            else:rows[0]['utility']['catboost']={'informative':False,'retention':None}
            summary,panels=m.aggregate(rows)
            group=next(g for g in summary if (g['dataset'],g['profile'],g['size_multiplier'])==('lineage000',m.PROFILES[0],1))
            panel=next(p for p in panels if (p['profile'],p['size_multiplier'])==(m.PROFILES[0],1))
            with self.subTest(status=status):
                self.assertIsNone(group['utility']['catboost']['median_retention'])
                self.assertEqual(panel['utility']['catboost']['complete_informative_lineages'],99)
                self.assertEqual(panel['lineages'],100)

    def test_release_scores_or_win_claim_rejected(self):
        for k,v in [('mfs_v2',0.99),('ptf_v1',0.99),('release_safe_l3',True),('paired_superiority',True),('counts_as_dope_win',True)]:
            rows=copy.deepcopy(self.rows);rows[0][k]=v
            with self.subTest(field=k),self.assertRaises(ValueError):m.aggregate(rows)

    def test_external_anchors_required_before_evidence_read(self):
        with self.assertRaises(ValueError):m.build('','')

    def test_csv_and_markdown_regenerate_from_same_committed_values(self):
        summary,panels=m.aggregate(self.rows)
        report={'cells':self.rows,'summary':summary,'profile_panels':panels,'scope':'descriptive validation'}
        table,markdown=m.tables(report)
        self.assertEqual(len(table.splitlines()),801)
        self.assertIn('informative',markdown);self.assertIn('MFS-v2/PTF-v1/release-safe scores null',markdown)
        report['profile_panels'][0]['utility']['catboost']['median_of_lineage_sample_medians']=1.
        with self.assertRaises(ValueError):m.tables(report)

    def test_receipt_digest_verified_before_reconciliation_or_metrics(self):
        with TemporaryDirectory() as d:
            base=Path(d);root=base/'round';root.mkdir()
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            round_path=root/'round.lock.json';round_path.write_text('{}');rh=sha(round_path)
            metric=root/'metric.json';metric.write_text('{"retention":0.5}')
            report=root/'reconciliation-v1.json';report.write_text('{"closed":true}');report_hash=sha(report)
            anchor={'round_sha256':rh,'fit_receipt_lock_sha256':m.FIT_RECEIPT_SHA,
                    'official_tests_opened':False,'mfs_v2':None,'ptf_v1':None,
                    'refs':{str(metric):sha(metric)},'reconciliation_sha256':report_hash}
            receipt=root/'receipt-lock-v1.json';receipt.write_text(json.dumps(anchor));receipt_hash=sha(receipt)
            with patch.object(paths,'BASE',base),patch.object(m,'ROOT',root),patch.object(m,'ROUND_SHA',rh):
                self.assertTrue(m.anchored(receipt_hash,report_hash)[0]['closed'])
                metric.write_text('rewritten metric')
                with patch.object(m,'read',wraps=m.read) as reads,self.assertRaises(ValueError):m.anchored(receipt_hash,report_hash)
                self.assertNotIn(report,[args.args[0] for args in reads.call_args_list])
                anchor['refs'][str(metric)]=sha(metric);receipt.write_text(json.dumps(anchor))
                with self.assertRaises(ValueError):m.anchored(receipt_hash,report_hash)
                link=base/'alias';link.symlink_to(root,target_is_directory=True)
                with patch.object(m,'ROOT',link),self.assertRaises(ValueError):m.anchored(sha(receipt),report_hash)

    def test_nonfinite_and_boolean_retention_rejected(self):
        for value in (True,float('nan'),float('inf')):
            rows=copy.deepcopy(self.rows);rows[0]['utility']['catboost']['retention']=value
            with self.subTest(value=value),self.assertRaises(ValueError):m.aggregate(rows)

if __name__=='__main__':unittest.main()
