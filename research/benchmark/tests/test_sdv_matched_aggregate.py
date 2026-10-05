"""Hermetic complete/missing/duplicate/cutoff aggregation controls."""
import copy,importlib.util,unittest
from pathlib import Path
from research.benchmark import sdv_matched_aggregate as m
class Aggregation(unittest.TestCase):
 def setUp(self):
  self.cells=[dict(dataset=f'dataset{i:03d}',method=method,configuration=config,fit_seed=11,sample_seed=s,size_multiplier=z,counts_as_dope_win=False,mfs_v2=None,ptf_v1=None,release_safe_l3=None,paired_superiority=None,charged_artifact_bytes=100,artifact_within_l3_cap=True,status='ok',utility={a:dict(informative=True,retention=.75 if method=='DOPE' else .5) for a in m.AUDITORS}) for i in range(100) for method,config in m.CONFIGS for s in m.SEEDS for z in m.SIZES]
 def test_complete_paired_groups(self):
  r=m.aggregate(self.cells);self.assertEqual(r['frozen_logical_cells'],4800);self.assertEqual(len(r['paired_descriptive']),96)
  for p in r['paired_descriptive']:self.assertEqual(p['complete_paired_lineages'],100);self.assertEqual(p['median_dataset_difference'],.25);self.assertIsNone(p['paired_superiority'])
 def test_missing_scheduler_samples_never_become_losses_or_wins(self):
  for c in self.cells:
   if c['dataset']=='dataset000' and c['method']=='CTGAN':c.update(status='common_samples_unavailable',utility=None,charged_artifact_bytes=None,artifact_within_l3_cap=None)
  r=m.aggregate(self.cells)
  for p in r['paired_descriptive']:
   if p['method']=='CTGAN':self.assertEqual(p['complete_paired_lineages'],99);self.assertFalse(p['counts_as_dope_win'])
 def test_missing_logical_row_rejected(self):
  with self.assertRaises(ValueError):m.aggregate(self.cells[:-1])
 def test_duplicate_logical_row_rejected(self):
  with self.assertRaises(ValueError):m.aggregate(self.cells+[self.cells[0]])
 def test_non_boolean_seed_identity_rejected(self):
  self.cells[0]['fit_seed']=11.0
  with self.assertRaises(ValueError):m.aggregate(self.cells)
 def test_gated_score_rejected(self):
  self.cells[0]['ptf_v1']=.99
  with self.assertRaises(ValueError):m.aggregate(self.cells)
if __name__=='__main__':unittest.main()
