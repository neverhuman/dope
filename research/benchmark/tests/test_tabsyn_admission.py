"""Scoped admission controls use generated metadata; they do not qualify CUDA."""
import ast
import copy
import fcntl
import hashlib
import importlib.util
import json
import marshal
import os
from pathlib import Path
import struct
import tempfile
import time
import unittest
from unittest.mock import patch

from research.benchmark import tabsyn_admission as admission
from research.benchmark import tabsyn_adapter as adapter
from research.benchmark import tabsyn_native_target as target

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True)+'\n')
    return sha(path)


class Admission(unittest.TestCase):
    def setUp(self):
        area=ROOT/'target/tabsyn-admission-controls'; area.mkdir(parents=True,exist_ok=True)
        self.directory=Path(tempfile.mkdtemp(dir=area))
        self.guard=self.directory/'source/tabsyn_operation_guard.py';self.guard.parent.mkdir()
        self.q=self.directory/'cuda.json';self.n=self.directory/'native.json';self.code=self.directory/'code.json'
        self.scope=dict(source_commit=admission.AUTHOR,real_data_allowed=True,
            partition='official_training_derived_validation',official_tests_opened=False,
            gpu_lease_path=admission.LEASE,maximum_trials=8,maximum_lineage_seconds=43200,
            whole_operation_seconds=600,gpu_vram_cap_bytes=16*2**30,config_sha256=['c'*64],
            epoch_budget=dict(vae_epochs=200,diffusion_epochs=1000),
            epoch_policy_sha256='1'*64,diffusion_author_patience=500,
            generated_cuda_qualification=dict(path=str(self.q),sha256=''),
            native_cuda_qualification=dict(path=str(self.n),sha256=''),
            author_mle_sha256='d'*64,TRAIN_target_helper_sha256='e'*64)
        self.qual=dict(format='dope-tabsyn-epoch-policy-generated-CUDA-two-replay-closure',
            status='declared_epoch_fit_sample_replay_verified',runtime_sha256='a'*64,
            actual_outer_process_exit_verified=True,official_tests_opened=False,real_fits_started=0,
            epoch_policy_sha256='1'*64,epoch_budget=dict(self.scope['epoch_budget']),
            diffusion_author_patience=500,config_sha256=['c'*64],timeout_checkpoint_eligible=False,
            scope='numeric_generated_CUDA_only',input_receipt_refs=[{},{}],root_actual_exit_refs=[{},{}])
        self.native=dict(format='dope-tabsyn-generated-native-CUDA-closure',status='ok',actual_exit_code=0,
            runtime_sha256='a'*64,epoch_policy_sha256='1'*64,author_mle_sha256='d'*64,TRAIN_target_helper_sha256='e'*64,
            auditor_configurations=36,postselection_refits=4,actual_outer_process_exit_verified=True,
            official_tests_opened=False)
        self.result=dict(mode='fit',config_sha256='c'*64,runtime_sha256='a'*64,operation_sha256='b'*64,
            actual_caller=admission.identity(os.getpid()),controller=admission.identity(os.getppid()),
            cpu_slot=list(range(80,96)),fresh_capacity_rpc_verified=True,
            current_candidate_bytes_and_providers_verified=True,worker_and_request_custody_verified=True,
            official_tests_opened=False,started_monotonic=time.monotonic(),deadline_monotonic=time.monotonic()+100,
            prior_fit_attempts=0,charged_prior_lineage_seconds=0,failed_fit_and_admission_wait_costs_included=True,
            epoch_budget=dict(self.scope['epoch_budget']),epoch_policy_sha256='1'*64,inherited_gpu_lease_fd=33)
        self.declaration=dict(operation_path=str(self.directory/'operation.json'),operation_sha256='b'*64)
        self.leases=[]

    def freeze(self):
        self.guard.write_text('def admit_tabsyn_operation(operation_path,pin,mode,config,runtime,code_path,code_pin,phase):\n'
            ' result='+repr(self.result)+'\n'
            ' result["mode"]=mode;result["code_sha256"]=code_pin;result["phase"]=phase\n return result\n')
        self.manifest=dict(format='dope-tabsyn-adapter-controls',version=1,runtime_sha256='a'*64,
            source_commit=admission.AUTHOR,author_files={'eval/mle/mle.py':'d'*64},
            execution_admitted=False,numeric_population_scope=self.scope,
            inventory=dict(root=str(self.directory),files={'source/tabsyn_operation_guard.py':dict(sha256=sha(self.guard)),
                'research/benchmark/tabsyn_adapter.py':dict(sha256='f'*64)}))
        binding=admission.source_binding(self.manifest)
        self.qual.update(source_binding_sha256=binding,adapter_source_sha256='f'*64)
        self.native.update(source_binding_sha256=binding,native_adapter_sha256='f'*64)
        self.scope['generated_cuda_qualification']['sha256']=write(self.q,self.qual)
        self.scope['native_cuda_qualification']['sha256']=write(self.n,self.native)
        self.code_pin=write(self.code,self.manifest)

    def verify(self,mode='fit'):
        with patch.object(admission.os,'sched_getaffinity',return_value=set(range(80,96))), \
                patch.object(admission,'held_lease',side_effect=lambda fd:self.leases.append(fd)):
            return admission.verify_operation(self.declaration,mode,'c'*64,'a'*64,self.code,self.code_pin)

    def test_full_and_scaled_epochs_are_separate_explicit_contracts(self):
        config=json.loads((ROOT/'research/benchmark/tabsyn-native-grid.proposal.json').read_bytes())['configs'][0]
        adapter.validate_config(config)
        config.update(vae_epochs=200,diffusion_epochs=1000)
        config['config_sha256']=hashlib.sha256(json.dumps({k:v for k,v in config.items() if k!='config_sha256'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        with self.assertRaises(ValueError):adapter.validate_config(config)
        adapter.validate_config(config,epoch_budget=self.scope['epoch_budget'])
        with self.assertRaises(ValueError):adapter.validate_config(config,epoch_budget=dict(vae_epochs=True,diffusion_epochs=1000))

    def test_exact_source_bound_generated_metadata_grant(self):
        self.freeze();result=self.verify();self.assertEqual(result['epoch_budget'],self.scope['epoch_budget'])
        self.assertEqual(self.leases,[33])

    def owner_one(self):
        policy=ROOT/'research/benchmark/tabsyn-one-qualification.owner-addendum.json'
        self.scope['operational_qualification_policy']=dict(path=str(policy),sha256=sha(policy))
        self.qual.update(format='dope-tabsyn-owner-one-generated-CUDA-closure-v1',
            input_receipt_refs=[{}],root_actual_exit_refs=[{}],
            operational_qualification_policy_sha256=admission.OWNER_ONE_POLICY_SHA256)

    def test_one_cuda_closure_requires_exact_owner_policy_and_source_binding(self):
        self.owner_one();self.freeze();self.assertEqual(self.verify()['mode'],'fit')
        self.assertEqual(self.leases,[33])

    def test_one_closure_without_owner_policy_rejected_before_guard_initializer(self):
        self.qual.update(input_receipt_refs=[{}],root_actual_exit_refs=[{}]);self.freeze()
        self.guard.write_text('raise AssertionError("initializer executed")\n')
        with self.assertRaises(ValueError):self.verify()
        self.assertEqual(self.leases,[])

    def test_changed_owner_policy_or_receipt_binding_rejected(self):
        for kind in ('policy_digest','policy_bytes','receipt_policy','receipt_source'):
            with self.subTest(kind=kind):
                self.owner_one();self.freeze()
                if kind=='policy_digest':
                    self.scope['operational_qualification_policy']['sha256']='0'*64
                elif kind=='policy_bytes':
                    policy=self.directory/'changed-policy.json'
                    policy.write_bytes(b'{}\n')
                    self.scope['operational_qualification_policy']['path']=str(policy)
                elif kind=='receipt_policy':
                    self.qual['operational_qualification_policy_sha256']='0'*64
                else:
                    self.qual['source_binding_sha256']='0'*64
                self.scope['generated_cuda_qualification']['sha256']=write(self.q,self.qual)
                self.code_pin=write(self.code,self.manifest)
                with self.assertRaises(ValueError):self.verify()
        self.assertEqual(self.leases,[])

    def test_owner_one_keeps_native_36_plus_4_and_actual_exit_gates(self):
        self.owner_one();self.native['auditor_configurations']=35;self.freeze()
        with self.assertRaises(ValueError):self.verify('native')
        self.assertEqual(self.leases,[])

    def test_mock_cuda_receipt_cannot_admit_real_operation(self):
        self.qual['format']='dope-tabsyn-generated-mock-controls';self.freeze()
        self.guard.write_text('raise AssertionError("initializer executed")\n')
        with self.assertRaises(ValueError):self.verify()
        self.assertEqual(self.leases,[])

    def test_missing_external_guard_rejects_without_dependency_initialization(self):
        self.freeze();self.manifest['inventory']['root']=str(self.directory/'absent')
        self.code_pin=write(self.code,self.manifest)
        with self.assertRaises((ValueError,FileNotFoundError)):self.verify()
        self.assertEqual(self.leases,[])

    def test_source_drift_rejected_before_initializer(self):
        self.freeze();self.guard.write_text('raise AssertionError("initializer executed")\n')
        with self.assertRaises(ValueError):self.verify()
        self.assertEqual(self.leases,[])

    def test_injected_guard_bytecode_is_never_executed(self):
        self.freeze();cached=Path(importlib.util.cache_from_source(str(self.guard)))
        cached.parent.mkdir(exist_ok=True)
        code=compile('raise AssertionError("cached initializer executed")',str(self.guard),'exec')
        header=importlib.util.MAGIC_NUMBER+struct.pack('<III',0,int(self.guard.stat().st_mtime),self.guard.stat().st_size)
        cached.write_bytes(header+marshal.dumps(code))
        with patch.object(admission.SourceFileLoader, 'get_data',
                          side_effect=AssertionError('loader read disk or cached bytecode')):
            self.assertEqual(self.verify()['mode'],'fit')
        self.assertEqual(self.leases,[33])

    def test_pid_reuse_and_cpu_boundary_drift_rejected(self):
        self.result['actual_caller']['start_ticks']='stale';self.freeze()
        with self.assertRaises(ValueError):self.verify()
        self.result['actual_caller']=admission.identity(os.getpid());self.result['cpu_slot']=[80];self.freeze()
        with self.assertRaises(ValueError):self.verify()
        self.assertEqual(self.leases,[])

    def test_expired_or_extended_whole_operation_rejected(self):
        for start,end in [(time.monotonic()-601,time.monotonic()),(time.monotonic(),time.monotonic()+601)]:
            self.result.update(started_monotonic=start,deadline_monotonic=end);self.freeze()
            with self.assertRaises(ValueError):self.verify()
        self.assertEqual(self.leases,[])

    def test_failed_trials_and_wait_cost_cannot_be_reset(self):
        for key,value in [('prior_fit_attempts',8),('prior_fit_attempts',True),
            ('charged_prior_lineage_seconds',43200),('failed_fit_and_admission_wait_costs_included',False)]:
            original=self.result[key];self.result[key]=value;self.freeze()
            with self.assertRaises(ValueError):self.verify()
            self.result[key]=original
        self.assertEqual(self.leases,[])

    def test_actual_held_descriptor_is_required_and_foreign_open_is_rejected(self):
        path=self.directory/'gpu.lock'
        with path.open('a') as owner,path.open('a') as unrelated:
            with self.assertRaises(ValueError):admission.held_lease(owner.fileno(),path)
            fcntl.flock(owner.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            admission.held_lease(owner.fileno(),path)
            with self.assertRaises(ValueError):admission.held_lease(unrelated.fileno(),path)
            self.assertRaises(BlockingIOError,fcntl.flock,unrelated.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_native_auditor_requires_actual_36_grid_and_exit_proof(self):
        for key,value in [('auditor_configurations',35),('actual_exit_code',1),('actual_outer_process_exit_verified',False)]:
            original=self.native[key];self.native[key]=value;self.freeze()
            with self.assertRaises(ValueError):self.verify('native')
            self.native[key]=original
        self.assertEqual(self.leases,[])

    def test_fit_and_sample_deny_missing_wrong_or_changed_grant_before_dependencies(self):
        import builtins
        import sys
        self.freeze();model=self.directory/'artifact';model.mkdir()
        original=builtins.__import__;initialized=[]
        def deny(name,*args,**kwargs):
            if name.split('.')[0] in ('numpy','torch','sklearn','safetensors','tabsyn','eval'):
                initialized.append(name);raise AssertionError('dependency initialized before rejection')
            return original(name,*args,**kwargs)
        for bad in (None,dict(self.declaration,operation_sha256='9'*64),'source_drift'):
            self.freeze()
            declaration=self.declaration if bad=='source_drift' else bad
            if bad=='source_drift':self.guard.write_text('raise AssertionError("initializer executed")\n')
            write(model/'model.json',dict(complete=True,source_commit=adapter.AUTHOR_COMMIT,
                runtime_sha256='a'*64,code_sha256=self.code_pin,generated_fixture=False,execution_admitted=True,
                config=dict(config_sha256='c'*64),declared_epoch_budget=self.scope['epoch_budget']))
            kwargs=dict(device='cuda:0',runtime_path='generated',runtime_sha256='a'*64,
                code_path=self.code,code_sha256=self.code_pin,admission=declaration)
            with patch.object(adapter,'verify_runtime',return_value={}),patch.object(adapter,'artifact_inventory',return_value={}), \
                    patch('builtins.__import__',side_effect=deny):
                with self.assertRaises(ValueError):adapter.fit(None,None,b'{}',model,dict(config_sha256='c'*64),**kwargs)
                with self.assertRaises(ValueError):adapter.sample(model,16,101,{},**kwargs)
        self.assertEqual(initialized,[])

    def test_fit_rejects_changed_projection_or_matrix_before_model_initializer(self):
        import builtins
        import numpy as np
        config=json.loads((ROOT/'research/benchmark/tabsyn-native-grid.proposal.json').read_bytes())['configs'][0]
        config.update(self.scope['epoch_budget'])
        config['config_sha256']=hashlib.sha256(json.dumps({k:v for k,v in config.items() if k!='config_sha256'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        train=np.zeros((9,2),dtype=np.float64);validation=np.zeros((2,2),dtype=np.float64)
        operation=dict(deadline_monotonic=time.monotonic()+100,epoch_budget=self.scope['epoch_budget'],
            projection_sha256=hashlib.sha256(b'{}').hexdigest(),
            joint_train_sha256=hashlib.sha256(train.tobytes(order='C')).hexdigest(),
            joint_validation_sha256=hashlib.sha256(validation.tobytes(order='C')).hexdigest())
        initialized=[];original=builtins.__import__
        def deny(name,*args,**kwargs):
            if name.split('.')[0] in ('torch','sklearn','safetensors','tabsyn','eval'):
                initialized.append(name);raise AssertionError('model initialized before input custody')
            return original(name,*args,**kwargs)
        kwargs=dict(device='cuda:0',runtime_path='generated',runtime_sha256='a'*64,
            code_path='generated',code_sha256='a'*64,admission=self.declaration)
        with patch.object(adapter,'verify_runtime',return_value={}), \
                patch.object(admission,'verify_operation',return_value=operation), \
                patch('builtins.__import__',side_effect=deny):
            # Projection differs before even the adapter NumPy import.
            with patch.dict('sys.modules',{}):
                with self.assertRaises(ValueError):adapter.fit(train,validation,b'changed',self.directory/'artifact',config,**kwargs)
            train[0,0]=1
            with self.assertRaises(ValueError):adapter.fit(train,validation,b'{}',self.directory/'artifact',config,**kwargs)
            train[0,0]=float('nan')
            with self.assertRaises(ValueError):adapter.fit(train,validation,b'{}',self.directory/'artifact',config,**kwargs)
        self.assertEqual(initialized,[])

    def test_old_adapter_cuda_qualification_cannot_admit_new_source(self):
        self.freeze();self.qual['adapter_source_sha256']='0'*64
        self.scope['generated_cuda_qualification']['sha256']=write(self.q,self.qual)
        self.code_pin=write(self.code,self.manifest)
        self.guard.write_text('raise AssertionError("initializer executed")\n')
        with self.assertRaises(ValueError):self.verify()
        self.assertEqual(self.leases,[])

    def test_generated_cuda_policy_must_match_admitted_scaled_epochs(self):
        for key,value in [('epoch_budget',dict(vae_epochs=4000,diffusion_epochs=10001)),
                ('epoch_policy_sha256','2'*64),('diffusion_author_patience',499),
                ('timeout_checkpoint_eligible',True),('config_sha256',['0'*64])]:
            with self.subTest(key=key):
                original=self.qual[key];self.qual[key]=value;self.freeze()
                with self.assertRaises(ValueError):self.verify()
                self.qual[key]=original
        self.assertEqual(self.leases,[])

    def test_TRAIN_inverse_and_uninformative_target_keep_common_units(self):
        raw=json.dumps(dict(version=1,task='regression',fit_partition='train',numeric_rule='train_minmax_clip_0_1',target_map=dict(min=2,max=500))).encode()
        state=target.target_state(raw,hashlib.sha256(raw).hexdigest())
        values=[0,.5,1];self.assertEqual(target.inverse_values(values,state),[2,251,500]);self.assertEqual(values,[0,.5,1])
        for lo,hi in [(-10,-1),(0,1),(5,5),(30000,40000)]:
            self.assertEqual(len(set(target.generated_author_labels(target.inverse_values(values,dict(min=lo,max=hi))))),1)
        receipt=target.uninformative_receipt('generated_clipped_constant')
        self.assertTrue(receipt['default_retained']);self.assertIsNone(receipt['native_winner'])
        self.assertFalse(receipt['common_projection_units_changed'])


if __name__=='__main__':unittest.main(verbosity=2)
