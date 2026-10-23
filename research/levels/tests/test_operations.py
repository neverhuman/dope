"""Monitor and allocation contracts without torch, private data or host mutations."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from research.levels.contracts import TrainConfig,ResearchError
from research.levels.dashboard import progress,HTML
from research.levels.resources import admissible


class OperationsTests(unittest.TestCase):
    def test_gpu_guard_admits_only_own_process_and_sufficient_capacity(self):
        value=dict(disk_available=20*2**30,memory_available=16*2**30,
                   gpus=[['0','GPU-a','24564','0','0']],gpu_processes=[])
        with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'0'}):
            self.assertTrue(admissible(value,cuda=True))
            self.assertFalse(admissible(dict(value,gpu_processes=[['GPU-a','987654','10']]),cuda=True))
            self.assertFalse(admissible(dict(value,disk_available=0),cuda=True))
        with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'1'}):self.assertFalse(admissible(value,cuda=True))

    def test_monitor_keeps_train_validation_and_partial_line_out_of_source_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'pilot';run.mkdir()
            (run/'recipe.json').write_text(json.dumps(dict(evidence='synthetic_development_only',phase='ordinary',model={},train={})))
            events=[dict(kind=k,step=1,levels={'low':{'total':v}}) for k,v in [('train',2.),('validation',3.)]]
            (run/'metrics.jsonl').write_text('\n'.join(json.dumps(e) for e in events)+'\n{"unfinished":')
            (run/'private.csv').write_text('PRIVATE_CANARY')
            data=progress(root)
            self.assertEqual(data['runs'][0]['metrics'],events)
            self.assertNotIn('PRIVATE_CANARY',json.dumps(data))
            self.assertIn('held-out development',HTML)

    def test_monitor_exposes_observer_coverage_without_reading_private_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'observed';run.mkdir()
            (run/'recipe.json').write_text(json.dumps(dict(evidence='synthetic_development_only',
                phase='ordinary',model={},train={})))
            event=dict(kind='validation',step=0,levels={},observer=dict(recipe='owned-balanced-unique-v2',
                scheduled_episodes=40,completed_episodes=2,unique_physical_tables=2,state='resource_pause'))
            (run/'metrics.jsonl').write_text(json.dumps(event)+'\n')
            (run/'checkpoints').mkdir();(run/'checkpoints/private.safetensors').write_bytes(b'PRIVATE_CANARY')
            data=progress(root)
            self.assertEqual(data['runs'][0]['metrics'][0]['observer'],event['observer'])
            self.assertNotIn('PRIVATE_CANARY',json.dumps(data))
            self.assertIn('distinct physical tables',HTML)
            self.assertIn('physical coverage unavailable in historical metrics',HTML)

    def test_fractional_training_bounds_refuse(self):
        with self.assertRaises(ResearchError):TrainConfig(steps=2.5).validate()
