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

    def test_fractional_training_bounds_refuse(self):
        with self.assertRaises(ResearchError):TrainConfig(steps=2.5).validate()
