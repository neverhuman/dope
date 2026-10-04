"""Proposed native child environment controls; no child is executed."""
import copy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from research.benchmark.tests import test_container_native_library_custody as fixture
from research.benchmark.tests.test_container_native_sampler_custody import sha

from research.benchmark import container_native_child_environment as environment


class NativeChildControls(unittest.TestCase):
    def setUp(self):
        self.f = fixture.NativeLibraryControls(); self.addCleanup(self.f.doCleanups); self.f.setUp()
        self.base = self.f.base
        self.preload = self.base / 'missing-preload'
        guard = patch.object(environment, 'PRELOAD', str(self.preload));guard.start();self.addCleanup(guard.stop)
        self.loader_path = self.base / 'loader.json'
        self.loader = {'absent': {str(self.preload): environment.library.loader.path_identity(str(self.preload))}}
        self.proposal_path = self.base / 'child-proposal.json'
        self.proposal = json.loads(self.f.native.proposal_path.read_bytes())
        self.proposal['native_library_directory'] = str(self.f.directory)
        self.manifest = self.base / 'child-environment.json'
        self.lock = {'format': 'dope-native-child-environment-preparation', 'version': 1,
            'library_path': str(self.f.manifest), 'library_sha256': self.f.digest,
            'proposal_path': str(self.proposal_path), 'loader_path': str(self.loader_path),
            'library_helper_sha256': sha(Path(environment.library.__file__)),
            'bootstrap_helper_sha256': sha(Path(environment.library.sampler.bootstrap.__file__)),
            'loader_helper_sha256': sha(Path(environment.library.loader.__file__)),
            'environment_helper_sha256': sha(Path(environment.__file__)), 'candidate_processes_started': 0}
        for key in ('actual_child_environment_verified', 'actual_loader_selection_verified',
                    'full_runtime_closure_certified', 'execution_admitted', 'official_tests_opened'):
            self.lock[key] = False
        for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority'): self.lock[key] = None
        self.freeze_parents(); self.freeze(); self.verify()

    def freeze_parents(self):
        self.proposal_path.write_text(json.dumps(self.proposal,sort_keys=True))
        self.lock['proposal_sha256'] = sha(self.proposal_path)
        self.loader_path.write_text(json.dumps(self.loader,sort_keys=True))
        self.lock['loader_sha256'] = sha(self.loader_path)
        with patch.object(environment.library.sampler.bootstrap.time, 'monotonic', return_value=125.0):
            plan = environment.library.sampler.bootstrap.prepare_invocation(self.proposal_path.read_bytes(),
                self.lock['proposal_sha256'],batch_started_at=100.0)
        self.lock['environment'] = [list(row) for row in plan.environment]

    def freeze(self):
        self.manifest.write_text(json.dumps(self.lock,sort_keys=True));self.digest=sha(self.manifest)

    def verify(self, **overrides):
        import builtins
        original = builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.split('.')[0] in {'numpy','torch','scipy','sklearn','pandas','sdv'}:
                self.fail('candidate dependency initialized')
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=guarded), \
                patch('subprocess.run',side_effect=AssertionError('child process started')), \
                patch('subprocess.Popen',side_effect=AssertionError('child process started')), \
                patch('ctypes.CDLL',side_effect=AssertionError('child library loaded')), \
                patch.object(environment.library.sampler.bootstrap.time,'monotonic',return_value=125.0):
            return environment.verify_native_child_environment(**dict({'environment_path':str(self.manifest),
                'environment_sha256':self.digest,'batch_started_at':100.0},**overrides))

    def rejected(self, **overrides):
        with self.assertRaisesRegex(ValueError,'^compressed native child environment rejected$'):
            self.verify(**overrides)

    def test_proposal_binding_grants_no_execution(self):
        r=self.verify();self.assertEqual(r['proposed_environment_entries_verified'],7)
        self.assertEqual(r['selected_library_files_verified'],2);self.assertEqual(r['outer_seconds_remaining'],575.0)
        for key in ('actual_child_environment_verified','actual_loader_selection_verified',
                    'full_runtime_closure_certified','execution_admitted','official_tests_opened'):
            self.assertIs(r[key],False)
        for key in ('mfs_v2','ptf_v1','release_safe','superiority'):self.assertIsNone(r[key])

    def test_external_digest_rejects_before_selected_library(self):
        class Alias(str):pass
        with patch.object(environment.library,'verify_native_library_directory',side_effect=AssertionError('unowned read')):
            self.rejected(environment_sha256='0'*64);self.rejected(environment_sha256=Alias(self.digest))

    def test_non_library_proposal_changes_reject(self):
        old=copy.deepcopy(self.proposal)
        for key,value in [('host','xbabe3'),('round_sha256','b'*64),('attempt',2),('job_sha256','c'*64)]:
            self.proposal=copy.deepcopy(old);self.proposal[key]=value
            self.proposal_path.write_text(json.dumps(self.proposal));self.lock['proposal_sha256']=sha(self.proposal_path)
            self.freeze();self.rejected()

    def test_library_directory_must_equal_selected_original(self):
        self.proposal['native_library_directory']=str(self.base);self.freeze_parents();self.freeze();self.rejected()

    def test_environment_cannot_add_or_change_inherited_values(self):
        old=copy.deepcopy(self.lock['environment'])
        for value in (old+[['TOKEN','opaque']],list(reversed(old)),[['CUDA_VISIBLE_DEVICES','0']]+old[1:],old[:-1]):
            self.lock['environment']=value;self.freeze();self.rejected()

    def test_missing_preload_snapshot_or_new_file_or_alias_reject(self):
        original=copy.deepcopy(self.loader)
        self.loader['absent']={};self.freeze_parents();self.freeze();self.rejected()
        self.loader=original;self.freeze_parents();self.freeze();self.verify()
        self.preload.write_bytes(b'opaque new preload');self.rejected();self.preload.unlink();self.verify()
        self.preload.symlink_to('missing-elsewhere');self.rejected()

    def test_initial_helper_special_files_reject_before_open(self):
        for module,key in ((environment.library,'library_helper_sha256'),
                           (environment.library.sampler.bootstrap,'bootstrap_helper_sha256'),
                           (environment.library.loader,'loader_helper_sha256'),
                           (environment,'environment_helper_sha256')):
            for kind in ('fifo','directory'):
                with self.subTest(helper=key,kind=kind):
                    helper=self.base/(key+kind+'.py');helper.write_bytes(Path(module.__file__).read_bytes())
                    old=self.lock[key];self.lock[key]=sha(helper);self.freeze()
                    with patch.object(module,'__file__',str(helper)):
                        self.verify();helper.unlink()
                        if kind=='fifo':os.mkfifo(helper)
                        else:helper.mkdir()
                        original=Path.open
                        def guarded(path,*args,**kwargs):
                            if path==helper:self.fail('initial nonregular helper opened')
                            return original(path,*args,**kwargs)
                        with patch.object(Path,'open',guarded):self.rejected()
                    if kind=='fifo':helper.unlink()
                    else:helper.rmdir()
                    self.lock[key]=old;self.freeze();self.verify()

    def test_final_parent_special_files_reject_before_open(self):
        parents=(self.manifest,self.f.manifest,self.f.native.manifest,self.f.native.proposal_path,
                 self.proposal_path,self.f.native.request,self.loader_path)
        original_verify=environment.library.verify_native_library_directory
        for path in parents:
            with self.subTest(parent=path.name):
                blob=path.read_bytes()
                def changed(*args,**kwargs):
                    result=original_verify(*args,**kwargs);path.unlink();os.mkfifo(path);return result
                original_open=Path.open
                def guarded(p,*args,**kwargs):
                    if p==path and not p.is_file():self.fail('late nonregular parent opened')
                    return original_open(p,*args,**kwargs)
                with patch.object(environment.library,'verify_native_library_directory',side_effect=changed), \
                        patch.object(Path,'open',guarded):self.rejected()
                path.unlink();path.write_bytes(blob);self.verify()

    def test_owned_request_numeric_identity_rejects(self):
        request=json.loads(self.f.native.request.read_bytes())
        old=copy.deepcopy(request)
        for key,value in [('fit_seed',11.0),('fit_seed',True),('final',0)]:
            request=copy.deepcopy(old);request['job'][key]=value
            self.f.native.request.write_text(json.dumps(request,sort_keys=True))
            self.f.native.lock['request_sha256']=sha(self.f.native.request)
            self.f.native.freeze();self.f.lock['sampler_sha256']=self.f.native.digest;self.f.freeze()
            self.lock['library_sha256']=self.f.digest;self.freeze();self.rejected()

    def test_final_original_deadline_after_last_helper_hash(self):
        original=environment.library.sampler.system.python.sha;calls=[];expired=[False]
        helper=Path(environment.__file__)
        def checked(path):
            result=original(path)
            if path==helper:
                calls.append(True)
                if len(calls)==2:expired[0]=True
            return result
        def now():return 700.0 if expired[0] else 125.0
        with patch.object(environment.library.sampler.system.python,'sha',side_effect=checked), \
                patch.object(environment.library.sampler.bootstrap.time,'monotonic',side_effect=now), \
                self.assertRaisesRegex(TimeoutError,'^compressed bootstrap batch deadline exhausted$'):
            environment.verify_native_child_environment(str(self.manifest),self.digest,batch_started_at=100.0)
        self.assertEqual(calls,[True,True])

    def test_final_preload_change_after_metadata_hash_rejects(self):
        original=environment.library.sampler.system.python.sha;calls=[]
        def changed(path):
            result=original(path)
            if path==self.manifest:
                calls.append(True)
                if len(calls)==1:self.preload.write_bytes(b'late opaque preload')
            return result
        with patch.object(environment.library.sampler.system.python,'sha',side_effect=changed):self.rejected()
        self.assertEqual(calls,[True])

    def test_parent_io_error_is_sanitized(self):
        original=Path.read_bytes
        def timeout(path):
            if path==self.loader_path:raise TimeoutError('opaque private environment')
            return original(path)
        with patch.object(Path,'read_bytes',timeout):self.rejected()

    def test_original_timer_is_not_reset(self):
        with patch.object(environment.library.sampler.bootstrap.time,'monotonic',return_value=700.0), \
                self.assertRaisesRegex(TimeoutError,'^compressed bootstrap batch deadline exhausted$'):
            environment.verify_native_child_environment(str(self.manifest),self.digest,batch_started_at=100.0)

    def test_exact_flags_version_counters_and_scores(self):
        original=copy.deepcopy(self.lock)
        for key,value in [('version',1.0),('candidate_processes_started',False),
                          ('execution_admitted',0),('actual_child_environment_verified',True),('ptf_v1',.99)]:
            self.lock=copy.deepcopy(original);self.lock[key]=value;self.freeze();self.rejected()


if __name__=='__main__':unittest.main()
