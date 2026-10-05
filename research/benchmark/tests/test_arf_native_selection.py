import copy,unittest
from research.benchmark import arf_native_selection as s

class Controls(unittest.TestCase):
    def setUp(self):
        self.jobs=[];self.receipts=[]
        for i,value in enumerate((-4.0,-1.0,-2.0)):
            config=dict(num_trees=10+i,min_node_size=20,max_iters=1,alpha=0.0)
            job=dict(method="ARF",track="common_numeric",dp_budget=None,dataset="a",fit_seed=11,
                configuration=config,configuration_sha256=s.identity(config),worker=dict(files={"validation.csv":"a"*64}))
            self.jobs.append(job)
            self.receipts.append(dict(job_sha256=s.identity(job),status="ok",elapsed_seconds=50.0,
                shared_retention=1000000.0 if i==0 else -1000000.0,
                result=dict(native_kpi=dict(objective="heldout_forde_mean_log_density",direction="maximize",
                    partition="validation",fit_seed=11,value=value,validation_sha256="a"*64,not_comparable_across_methods=True),
                    artifact_bytes=20,artifact_inventory={"model.json":dict(bytes=20)})))
    def reject(self):
        with self.assertRaises(ValueError):s.select(self.jobs,self.receipts)
    def test_own_density_selects_despite_common_retention(self):
        out=s.select(self.jobs,self.receipts);self.assertEqual(out["selected_job_sha256"],s.identity(self.jobs[1]));self.assertFalse(out["shared_outcomes_used_for_selection"])
    def test_bytes_break_exact_density_tie(self):
        self.receipts[2]["result"]["native_kpi"]["value"]=-1.0
        self.receipts[2]["result"]["artifact_bytes"]=10;self.receipts[2]["result"]["artifact_inventory"]["model.json"]["bytes"]=10
        self.assertEqual(s.select(self.jobs,self.receipts)["selected_job_sha256"],s.identity(self.jobs[2]))
    def test_float_job_seed_rejected(self):
        for job,row in zip(self.jobs,self.receipts):
            job["fit_seed"]=11.0;row["job_sha256"]=s.identity(job)
        self.reject()
    def test_float_native_seed_rejected(self):
        self.receipts[0]["result"]["native_kpi"]["fit_seed"]=11.0;self.reject()
    def test_different_validation_partition_rejected(self):
        self.jobs[0]["worker"]["files"]["validation.csv"]="b"*64
        self.receipts[0]["job_sha256"]=s.identity(self.jobs[0])
        self.receipts[0]["result"]["native_kpi"]["validation_sha256"]="b"*64
        self.reject()
    def test_different_runtime_source_rejected(self):
        self.jobs[0]["runtime_sha256"]="b"*64
        self.receipts[0]["job_sha256"]=s.identity(self.jobs[0]);self.reject()
    def test_missing_trial_rejected(self):
        self.receipts.pop();self.reject()
    def test_duplicate_trial_rejected(self):
        self.receipts[0]=copy.deepcopy(self.receipts[1]);self.reject()
    def test_timeouts_costed_but_ineligible(self):
        self.receipts[1].update(status="timeout",elapsed_seconds=600.0)
        out=s.select(self.jobs,self.receipts);self.assertEqual(out["selected_job_sha256"],s.identity(self.jobs[2]));self.assertEqual(out["elapsed_seconds"],700.0)
    def test_scheduling_cutoff_marked_separately(self):
        self.receipts[1].update(status="deadline_unstarted",elapsed_seconds=0.0)
        out=s.select(self.jobs,self.receipts);self.assertFalse(out["completed_search_without_scheduling_cutoff"]);self.assertEqual(out["statuses"]["deadline_unstarted"],1)
    def test_changed_objective_rejected(self):
        self.receipts[1]["result"]["native_kpi"]["objective"]="shared_retention";self.reject()
    def test_nonfinite_native_kpi_rejected(self):
        self.receipts[0]["result"]["native_kpi"]["value"]=float("inf");self.reject()
    def test_byte_charge_mismatch_rejected(self):
        self.receipts[0]["result"]["artifact_bytes"]+=1;self.reject()
    def test_cell_budget_exceeded_rejected(self):
        self.receipts[0]["elapsed_seconds"]=43201.0;self.reject()

if __name__=="__main__":unittest.main()
