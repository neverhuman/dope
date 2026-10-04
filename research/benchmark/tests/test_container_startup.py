"""Compose opaque startup guards without importing ML or executing a worker."""
import builtins
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import unittest
from unittest.mock import patch

from research.benchmark.tests import test_container_bootstrap_filesystem as fs_fixture
from research.benchmark.tests import test_container_native_runtime_inventory as inventory_fixture
from research.benchmark.tests.test_container_native_sampler_custody import sha

from research.benchmark import container_startup as startup


class StartupControls(unittest.TestCase):
    def setUp(self):
        self.fs=fs_fixture.FilesystemControls();self.addCleanup(self.fs.doCleanups);self.fs.setUp()
        self.inv=inventory_fixture.NativeInventoryControls();self.addCleanup(self.inv.doCleanups);self.inv.setUp()
        self.native=self.inv.library.native;self.base=self.inv.base
        for module,key,value in ((startup.inventory.environment.library.sampler.system.python,'BASE',Path('target').resolve()),
                                 (startup.inventory.environment.library.sampler.bootstrap,'BASE',PurePosixPath(Path('target').resolve()))):
            p=patch.object(module,key,value);p.start();self.addCleanup(p.stop)
        self.fs.job=copy.deepcopy(self.native.job)
        self.fs.job_sha=hashlib.sha256(startup.inventory.environment.library.sampler.system.canonical(self.fs.job).encode()).hexdigest()
        self.fs.descriptor['frozen_preview_job_digests']=[self.fs.job_sha]
        self.fs.descriptor_path.write_text(json.dumps(self.fs.descriptor,sort_keys=True))
        self.native.root=self.fs.root
        self.native.proposal=copy.deepcopy(self.fs.f.proposal)
        self.native.proposal.update(host='xbabe2',round_sha256=sha(self.fs.descriptor_path))
        self.native.freeze_job();self.native.freeze();self.native.verify()
        self.fs.request=self.native.request
        self.fs.req=json.loads(self.native.request.read_bytes())
        self.fs.f.proposal=copy.deepcopy(self.native.proposal)
        self.fs.f.proposal_path=self.native.proposal_path
        self.fs.f.lock['proposal_path']=str(self.native.proposal_path)
        self.fs.f.freeze_proposal();self.fs.f.freeze()
        self.fs.lock['environment_sha256']=self.fs.f.digest
        self.fs.freeze_request();self.fs.verify()
        self.inv.library.lock['sampler_sha256']=self.native.digest;self.inv.library.freeze()
        self.inv.f.lock['library_sha256']=self.inv.library.digest
        self.inv.f.proposal=copy.deepcopy(self.native.proposal)
        self.inv.f.proposal['native_library_directory']=str(self.inv.library.directory)
        self.inv.f.freeze_parents();self.inv.f.freeze();self.inv.f.verify()
        self.inv.lock['environment_sha256']=self.inv.f.digest;self.inv.freeze();self.inv.verify()
        # These orchestration tests stub only the independently tested Python/root stage.
        # Proposed FS, native executable, child environment and full recorded files run normally.
        loader=self.fs.f.f.f;projection=self.fs.f.f
        self.root_runtime=self.base/'opaque-root-runtime.json';self.root_runtime.write_bytes(b'{}')
        self.root_auxiliary=self.base/'opaque-root-auxiliary.json';self.root_auxiliary.write_bytes(b'{}')
        declaration=json.loads(loader.declarations.read_bytes())
        declaration.update(runtime_path=str(self.root_runtime),runtime_sha256=sha(self.root_runtime),
                           auxiliary_path=str(self.root_auxiliary),auxiliary_sha256=sha(self.root_auxiliary))
        loader.declarations.write_text(json.dumps(declaration,sort_keys=True))
        unresolved=json.loads(loader.unresolved.read_bytes());unresolved['declarations_sha256']=sha(loader.declarations)
        loader.unresolved.write_text(json.dumps(unresolved,sort_keys=True))
        loader.lock.update(declarations_sha256=sha(loader.declarations),unresolved_sha256=sha(loader.unresolved));loader.freeze()
        projection.search['loader_sha256']=loader.digest;projection.freeze_search()
        projection.lock.update(loader_sha256=loader.digest,declarations_sha256=sha(loader.declarations));projection.freeze()
        self.fs.f.lock['projection_sha256']=projection.digest;self.fs.f.freeze()
        self.fs.lock['environment_sha256']=self.fs.f.digest;self.fs.freeze();self.fs.verify()
        system=startup.inventory.environment.library.sampler.system
        def roots(declarations_path,declarations_sha,runtime_path,runtime_sha,auxiliary_path,auxiliary_sha):
            self.assertEqual(declarations_path,str(loader.declarations));self.assertEqual(declarations_sha,sha(loader.declarations))
            self.assertEqual((runtime_path,runtime_sha),(str(self.root_runtime),sha(self.root_runtime)))
            self.assertEqual((auxiliary_path,auxiliary_sha),(str(self.root_auxiliary),sha(self.root_auxiliary)))
            return {'declared_elf_files_verified':4}
        p=patch.object(system,'verify_declared_elf_inputs',side_effect=roots);p.start();self.addCleanup(p.stop)
        self.manifest=self.base/'startup.json'
        self.lock={'format':'dope-proposed-startup-preparation','version':1,
            'filesystem_path':str(self.fs.manifest),'filesystem_sha256':self.fs.digest,
            'sampler_path':str(self.native.manifest),'sampler_sha256':self.native.digest,
            'inventory_path':str(self.inv.manifest),'inventory_sha256':self.inv.digest,
            'filesystem_helper_sha256':sha(Path(startup.filesystem.__file__)),
            'sampler_helper_sha256':sha(Path(startup.inventory.environment.library.sampler.__file__)),
            'inventory_helper_sha256':sha(Path(startup.inventory.__file__)),
            'system_helper_sha256':sha(Path(startup.inventory.environment.library.sampler.system.__file__)),
            'startup_helper_sha256':sha(Path(startup.__file__)),'candidate_processes_started':0}
        for key in ('actual_loader_selection_verified','full_runtime_closure_certified','execution_admitted','official_tests_opened'):self.lock[key]=False
        for key in ('mfs_v2','ptf_v1','release_safe','superiority'):self.lock[key]=None
        self.freeze();self.verify()

    def freeze(self):
        self.manifest.write_text(json.dumps(self.lock,sort_keys=True));self.digest=sha(self.manifest)

    def verify(self,**overrides):
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.split('.')[0] in {'numpy','torch','scipy','sklearn','pandas','sdv'}:self.fail('ML initialized')
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=guarded), \
                patch('subprocess.run',side_effect=AssertionError('process started')), \
                patch('subprocess.Popen',side_effect=AssertionError('process started')), \
                patch('ctypes.CDLL',side_effect=AssertionError('native loaded')), \
                patch.object(startup.inventory.environment.library.sampler.bootstrap.time,'monotonic',return_value=125.0):
            return startup.verify_proposed_startup(**dict({'startup_path':str(self.manifest),
                'startup_sha256':self.digest,'batch_started_at':100.0},**overrides))

    def rejected(self,**overrides):
        with self.assertRaisesRegex(ValueError,'^compressed startup custody rejected$'):self.verify(**overrides)

    def test_combined_positive_chain_and_null_scores(self):
        result=self.verify();self.assertEqual(result['staged_source_files_verified'],6)
        self.assertEqual(result['native_elf_roots_verified'],1);self.assertEqual(result['recorded_native_files_verified'],4)
        self.assertEqual(result['outer_seconds_remaining'],575.0)
        for key in ('actual_loader_selection_verified','full_runtime_closure_certified','execution_admitted','official_tests_opened'):self.assertIs(result[key],False)
        for key in ('mfs_v2','ptf_v1','release_safe','superiority'):self.assertIsNone(result[key])


    def test_external_manifest_identity_before_any_stage(self):
        class Alias(str):pass
        system=startup.inventory.environment.library.sampler.system
        with patch.object(system,'verify_declared_elf_inputs',side_effect=AssertionError('unowned stage called')):
            self.rejected(startup_sha256='0'*64);self.rejected(startup_sha256=Alias(self.digest))

    def test_initial_helpers_regular_before_open(self):
        helper=self.base/'startup-helper.py';helper.write_bytes(Path(startup.__file__).read_bytes())
        self.lock['startup_helper_sha256']=sha(helper);self.freeze()
        with patch.object(startup,'__file__',str(helper)):
            self.verify();helper.unlink();os.mkfifo(helper)
            original=Path.open
            def guarded(path,*args,**kwargs):
                if path==helper:self.fail('initial FIFO helper opened')
                return original(path,*args,**kwargs)
            with patch.object(Path,'open',guarded):self.rejected()

    def test_external_parent_digests_precede_root_stage(self):
        original=copy.deepcopy(self.lock);system=startup.inventory.environment.library.sampler.system
        for key in ('filesystem_sha256','sampler_sha256','inventory_sha256'):
            self.lock=copy.deepcopy(original);self.lock[key]='0'*64;self.freeze()
            with patch.object(system,'verify_declared_elf_inputs',side_effect=AssertionError('unowned stage called')):self.rejected()

    def test_canonical_proposal_parent_binding_precedes_root_stage(self):
        other=self.base/'identical-original-proposal.json';other.write_bytes(self.native.proposal_path.read_bytes())
        self.fs.f.lock['proposal_path']=str(other);self.fs.f.freeze()
        self.fs.lock['environment_sha256']=self.fs.f.digest;self.fs.freeze()
        self.lock['filesystem_sha256']=self.fs.digest;self.freeze()
        with patch.object(startup.inventory.environment.library.sampler.system,'verify_declared_elf_inputs',side_effect=AssertionError('unbound stage called')):self.rejected()

    def test_each_actual_stage_failure_prevents_success(self):
        sampler=startup.inventory.environment.library.sampler
        for module,name in ((sampler.system,'verify_declared_elf_inputs'),(startup.filesystem,'verify_proposed_filesystem'),
                            (sampler,'verify_sampler_declarations'),(startup.inventory,'verify_native_runtime_inventory')):
            with self.subTest(stage=name):
                self.verify()
                with patch.object(module,name,side_effect=ValueError('opaque stage detail')):self.rejected()

    def test_owned_metadata_io_timeout_is_sanitized(self):
        original=Path.read_bytes
        def timeout(path):
            if path==self.inv.manifest:raise TimeoutError('opaque private metadata')
            return original(path)
        with patch.object(Path,'read_bytes',timeout):self.rejected()

    def test_proposal_read_io_timeout_is_integrity_rejection(self):
        original=Path.read_bytes;calls=[]
        def timed_out(path):
            if path==self.inv.f.proposal_path:
                calls.append(True)
                if len(calls)==2:raise TimeoutError('opaque proposal I/O timeout')
            return original(path)
        self.verify()
        with patch.object(Path,'read_bytes',timed_out):self.rejected()
        self.assertEqual(calls,[True,True])

    def test_final_parent_fifo_rejects_before_open(self):
        original=startup.inventory.verify_native_runtime_inventory
        parents=(self.manifest,self.fs.manifest,self.native.manifest,self.inv.manifest,self.inv.f.manifest,
                 self.inv.library.manifest,self.fs.f.manifest,self.fs.f.f.manifest,self.fs.f.f.f.declarations,
                 self.root_runtime,self.root_auxiliary,self.inv.f.proposal_path)
        original_open=Path.open
        for parent in parents:
            with self.subTest(parent=parent.name):
                self.verify();blob=parent.read_bytes();changed=[]
                def mutate(*args,**kwargs):
                    result=original(*args,**kwargs);parent.unlink();os.mkfifo(parent);changed.append(True);return result
                def guarded(path,*args,**kwargs):
                    if path==parent and not path.is_file():self.fail('final FIFO metadata opened')
                    return original_open(path,*args,**kwargs)
                with patch.object(startup.inventory,'verify_native_runtime_inventory',side_effect=mutate), \
                        patch.object(Path,'open',guarded):self.rejected()
                self.assertEqual(changed,[True]);parent.unlink();parent.write_bytes(blob)

    def test_final_startup_helper_alias_rejects_before_open(self):
        helper=self.base/'final-startup-helper.py';helper.write_bytes(Path(startup.__file__).read_bytes())
        target=self.base/'same-startup-helper.py';target.write_bytes(helper.read_bytes())
        self.lock['startup_helper_sha256']=sha(helper);self.freeze()
        original=startup.inventory.verify_native_runtime_inventory;original_open=Path.open;changed=[]
        with patch.object(startup,'__file__',str(helper)):
            self.verify()
            def mutate(*args,**kwargs):
                result=original(*args,**kwargs);helper.unlink();helper.symlink_to(target);changed.append(True);return result
            def guarded(path,*args,**kwargs):
                if path==helper and path.is_symlink():self.fail('final helper alias opened')
                return original_open(path,*args,**kwargs)
            with patch.object(startup.inventory,'verify_native_runtime_inventory',side_effect=mutate), \
                    patch.object(Path,'open',guarded):self.rejected()
        self.assertEqual(changed,[True])

    def test_deadline_after_root_stage_uses_original_timer(self):
        sampler=startup.inventory.environment.library.sampler;expired=[False]
        original=sampler.system.verify_declared_elf_inputs
        def roots(*args):
            result=original(*args);expired[0]=True;return result
        with patch.object(sampler.system,'verify_declared_elf_inputs',side_effect=roots), \
                patch.object(sampler.bootstrap.time,'monotonic',side_effect=lambda:700.0 if expired[0] else 125.0), \
                self.assertRaisesRegex(TimeoutError,'^compressed bootstrap batch deadline exhausted$'):
            startup.verify_proposed_startup(str(self.manifest),self.digest,batch_started_at=100.0)
        self.assertIs(expired[0],True)

    def test_deadline_after_final_integrity_hash(self):
        sampler=startup.inventory.environment.library.sampler;original=sampler.system.python.sha;calls=[];expired=[False]
        def checked(path):
            result=original(path)
            if path==Path(startup.__file__):
                calls.append(True)
                if len(calls)==2:expired[0]=True
            return result
        with patch.object(sampler.system.python,'sha',side_effect=checked), \
                patch.object(sampler.bootstrap.time,'monotonic',side_effect=lambda:700.0 if expired[0] else 125.0), \
                self.assertRaisesRegex(TimeoutError,'^compressed bootstrap batch deadline exhausted$'):
            startup.verify_proposed_startup(str(self.manifest),self.digest,batch_started_at=100.0)
        self.assertEqual(calls,[True,True])

    def test_exact_false_null_version_and_counter(self):
        original=copy.deepcopy(self.lock)
        for key,value in [('version',1.0),('candidate_processes_started',False),('execution_admitted',0),
                          ('official_tests_opened',1),('mfs_v2',.99)]:
            self.lock=copy.deepcopy(original);self.lock[key]=value;self.freeze();self.rejected()

if __name__=='__main__':unittest.main()
