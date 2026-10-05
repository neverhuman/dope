import json,os,time,unittest
from datetime import datetime,timezone
from unittest.mock import patch
from research.benchmark import arf_fit_request as w

class Controls(unittest.TestCase):
    def setUp(self):
        self.job=dict(method="ARF",track="common_numeric",dp_budget=None,fit_seed=11,
            configuration=dict(num_trees=30,min_node_size=5,max_iters=10,alpha=0.0))
        self.job["configuration_sha256"]=w.identity(self.job["configuration"])
        self.lock=dict(jobs=[dict(job=self.job,job_sha256=w.identity(self.job),adapter_applicability="supported_width")],
            deadline_utc=datetime.fromtimestamp(time.time()+1000,timezone.utc).isoformat())
        self.request=dict(job=json.loads(json.dumps(self.job)),round_sha256="a"*64,deadline_epoch=time.time()+590)
    def reject(self):
        with self.assertRaises(ValueError):w.request_job(self.request,self.lock,"a"*64)
    def test_exact_identity_passes(self):
        _,value=w.request_job(self.request,self.lock,"a"*64);self.assertEqual(value,w.identity(self.job))
    def test_incompatible_width_rejected(self):
        self.lock["jobs"][0]["adapter_applicability"]="adapter_width_incompatible";self.reject()
    def test_float_fit_seed_rejected(self):
        self.request["job"]["fit_seed"]=11.0;self.reject()
    def test_bool_fit_seed_rejected(self):
        self.request["job"]["fit_seed"]=True;self.reject()
    def test_float_tree_count_rejected(self):
        self.request["job"]["configuration"]["num_trees"]=30.0;self.reject()
    def test_invented_round_rejected(self):
        self.request["round_sha256"]="b"*64;self.reject()
    def test_duplicate_matrix_identity_rejected(self):
        self.lock["jobs"]*=2;self.reject()
    def test_too_long_deadline_rejected(self):
        self.request["deadline_epoch"]=time.time()+601;self.reject()
    def test_expired_deadline_rejected(self):
        self.request["deadline_epoch"]=time.time()-1;self.reject()
    def test_naive_round_deadline_rejected(self):
        self.lock["deadline_utc"]=datetime.fromtimestamp(time.time()+1000).isoformat();self.reject()
    def test_round_cutoff_rejected(self):
        self.lock["deadline_utc"]=datetime.fromtimestamp(time.time()+30,timezone.utc).isoformat();self.reject()
    def test_gpu_env_rejected_before_lock_or_module(self):
        with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"0"}),patch.object(w,"checked_file",side_effect=AssertionError("lock read reached")):
            with self.assertRaisesRegex(ValueError,"CPU environment required"):
                w.run(None,None,None,None,None,None)
    def test_duplicate_keys_rejected(self):
        with self.assertRaises(ValueError):w.decode(b'{"fit_seed":11,"fit_seed":23}')
    def test_nonfinite_exponent_rejected(self):
        with self.assertRaises(ValueError):w.decode(b'{"x":1e999}')

if __name__=="__main__":unittest.main()
