"""Opaque full recorded native inventory controls; no runtime is loaded."""
import copy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from research.benchmark.tests import test_container_native_child_environment as fixture
from research.benchmark.tests.test_container_native_sampler_custody import sha

from research.benchmark import container_native_runtime_inventory as inventory


class NativeInventoryControls(unittest.TestCase):
    def setUp(self):
        self.f=fixture.NativeChildControls();self.addCleanup(self.f.doCleanups);self.f.setUp()
        self.base=self.f.base;self.library=self.f.f
        self.extra=self.base/'outside-directory.so';self.extra.write_bytes(b'opaque outside root native file')
        self.alias=self.base/'outside-alias.so';self.alias.symlink_to('./outside-directory.so')
        for p in (self.extra,self.alias):
            self.library.runtime['files'][str(p)]={'resolved_path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)}
        self.library.freeze_parents();self.library.freeze()
        self.f.lock['library_sha256']=self.library.digest;self.f.freeze();self.f.verify()
        self.manifest=self.base/'native-inventory.json'
        self.lock={'format':'dope-native-runtime-inventory-preparation','version':1,
            'environment_path':str(self.f.manifest),'environment_sha256':self.f.digest,
            'files':{name:{'identity':inventory.environment.library.loader.path_identity(name),
                          'bytes':row['bytes'],'sha256':row['sha256']}
                     for name,row in self.library.runtime['files'].items()},
            'environment_helper_sha256':sha(Path(inventory.environment.__file__)),
            'loader_helper_sha256':sha(Path(inventory.environment.library.loader.__file__)),
            'inventory_helper_sha256':sha(Path(inventory.__file__)),'candidate_processes_started':0}
        for key in ('actual_linker_transcript_replayed','actual_loader_selection_verified',
                    'full_runtime_closure_certified','execution_admitted','official_tests_opened'):
            self.lock[key]=False
        for key in ('mfs_v2','ptf_v1','release_safe','superiority'):self.lock[key]=None
        self.freeze();self.verify()

    def freeze(self):
        self.manifest.write_text(json.dumps(self.lock,sort_keys=True));self.digest=sha(self.manifest)

    def verify(self, **overrides):
        import builtins
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.split('.')[0] in {'numpy','torch','scipy','sklearn','pandas','sdv'}:self.fail('ML initialized')
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=guarded), \
                patch('subprocess.run',side_effect=AssertionError('native process started')), \
                patch('subprocess.Popen',side_effect=AssertionError('native process started')), \
                patch('ctypes.CDLL',side_effect=AssertionError('native library loaded')), \
                patch.object(inventory.environment.library.sampler.bootstrap.time,'monotonic',return_value=125.0):
            return inventory.verify_native_runtime_inventory(**dict({'inventory_path':str(self.manifest),
                'inventory_sha256':self.digest,'batch_started_at':100.0},**overrides))

    def rejected(self, **overrides):
        with self.assertRaisesRegex(ValueError,'^compressed native runtime inventory rejected$'):self.verify(**overrides)

    def test_all_recorded_paths_include_outside_selected_root(self):
        r=self.verify();self.assertEqual(r['recorded_native_files_verified'],4)
        self.assertEqual(r['selected_library_files_verified'],2);self.assertEqual(r['outer_seconds_remaining'],575.0)
        for key in ('actual_linker_transcript_replayed','actual_loader_selection_verified',
                    'full_runtime_closure_certified','execution_admitted','official_tests_opened'):self.assertIs(r[key],False)
        for key in ('mfs_v2','ptf_v1','release_safe','superiority'):self.assertIsNone(r[key])

    def test_external_digest_before_child_guard(self):
        class Alias(str):pass
        with patch.object(inventory.environment,'verify_native_child_environment',side_effect=AssertionError('unowned guard')):
            self.rejected(inventory_sha256='0'*64);self.rejected(inventory_sha256=Alias(self.digest))

    def test_exact_original_inventory_membership(self):
        original=copy.deepcopy(self.lock['files'])
        self.lock['files'].pop(str(self.extra));self.freeze();self.rejected()
        self.lock['files']=copy.deepcopy(original)
        self.lock['files'][str(self.base/'added.so')]=copy.deepcopy(original[str(self.extra)])
        self.freeze();self.rejected()

    def test_outside_file_digest_and_literal_alias_changes(self):
        self.extra.write_bytes(b'changed outside file');self.rejected()
        self.extra.write_bytes(b'opaque outside root native file');self.verify()
        self.alias.unlink();self.alias.symlink_to('outside-directory.so');self.rejected()

    def test_exact_bytes_and_historical_resolved_path(self):
        key=str(self.extra);original=copy.deepcopy(self.lock['files'][key])
        for value in (True,float(original['bytes']),original['bytes']+1):
            self.lock['files'][key]=copy.deepcopy(original);self.lock['files'][key]['bytes']=value
            self.freeze();self.rejected()
        self.lock['files'][key]=original;self.freeze()
        self.library.runtime['files'][key]['resolved_path']=str(self.base/'other.so')
        self.library.freeze_parents();self.library.freeze();self.f.lock['library_sha256']=self.library.digest;self.f.freeze()
        self.lock['environment_sha256']=self.f.digest;self.freeze();self.rejected()

    def test_outside_fifo_rejects_before_hash(self):
        self.extra.unlink();os.mkfifo(self.extra)
        original=Path.open
        def guarded(path,*args,**kwargs):
            if path==self.extra:self.fail('outside FIFO opened')
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',guarded):self.rejected()

    def test_late_outside_file_change_rejects(self):
        original=inventory.environment.library.sampler.system.python.sha;calls=[]
        def changed(path):
            result=original(path)
            if path==self.extra:
                calls.append(True)
                if len(calls)==1:self.extra.write_bytes(b'late outside change')
            return result
        with patch.object(inventory.environment.library.sampler.system.python,'sha',side_effect=changed):self.rejected()

    def test_initial_helper_special_file_before_open(self):
        helper=self.base/'inventory-helper.py';helper.write_bytes(Path(inventory.__file__).read_bytes())
        self.lock['inventory_helper_sha256']=sha(helper);self.freeze()
        with patch.object(inventory,'__file__',str(helper)):
            self.verify();helper.unlink();os.mkfifo(helper)
            original=Path.open
            def guarded(path,*args,**kwargs):
                if path==helper:self.fail('inventory FIFO helper opened')
                return original(path,*args,**kwargs)
            with patch.object(Path,'open',guarded):self.rejected()

    def test_final_parent_nonregular_replacements_before_open(self):
        parents=(self.manifest,self.f.manifest,self.library.manifest,self.library.native.manifest,
                 self.f.proposal_path,self.library.validation_path,self.library.runtime_path)
        original_sha=inventory.environment.library.sampler.system.python.sha
        original_open=Path.open
        for parent in parents:
            with self.subTest(parent=parent.name):
                self.verify();blob=parent.read_bytes();calls=[]
                def changed(path):
                    result=original_sha(path)
                    if path==self.extra:
                        calls.append(True)
                        if len(calls)==4:parent.unlink();os.mkfifo(parent)
                    return result
                def guarded(path,*args,**kwargs):
                    if path==parent and not path.is_file():self.fail('final FIFO parent opened')
                    return original_open(path,*args,**kwargs)
                with patch.object(inventory.environment.library.sampler.system.python,'sha',side_effect=changed), \
                        patch.object(Path,'open',guarded):self.rejected()
                self.assertEqual(calls,[True]*4)
                parent.unlink();parent.write_bytes(blob);self.verify()

    def test_final_inventory_helper_alias_rejects_before_open(self):
        helper=self.base/'final-inventory-helper.py';helper.write_bytes(Path(inventory.__file__).read_bytes())
        self.lock['inventory_helper_sha256']=sha(helper);self.freeze()
        original_sha=inventory.environment.library.sampler.system.python.sha
        original_open=Path.open;calls=[]
        with patch.object(inventory,'__file__',str(helper)):
            self.verify()
            target=self.base/'same-helper-bytes.py';target.write_bytes(helper.read_bytes())
            def changed(path):
                result=original_sha(path)
                if path==self.extra:
                    calls.append(True)
                    if len(calls)==4:helper.unlink();helper.symlink_to(target)
                return result
            def guarded(path,*args,**kwargs):
                if path==helper and path.is_symlink():self.fail('final aliased helper opened')
                return original_open(path,*args,**kwargs)
            with patch.object(inventory.environment.library.sampler.system.python,'sha',side_effect=changed), \
                    patch.object(Path,'open',guarded):self.rejected()
        self.assertEqual(calls,[True]*4)

    def test_deadline_after_final_integrity_hash(self):
        original=inventory.environment.library.sampler.system.python.sha;calls=[];expired=[False]
        helper=Path(inventory.__file__)
        def checked(path):
            result=original(path)
            if path==helper:
                calls.append(True)
                if len(calls)==2:expired[0]=True
            return result
        def now():return 700.0 if expired[0] else 125.0
        with patch.object(inventory.environment.library.sampler.system.python,'sha',side_effect=checked), \
                patch.object(inventory.environment.library.sampler.bootstrap.time,'monotonic',side_effect=now), \
                self.assertRaisesRegex(TimeoutError,'^compressed bootstrap batch deadline exhausted$'):
            inventory.verify_native_runtime_inventory(str(self.manifest),self.digest,batch_started_at=100.0)
        self.assertEqual(calls,[True,True])

    def test_generic_inventory_io_timeout_is_sanitized(self):
        original=Path.open
        def timeout(path,*args,**kwargs):
            if path==self.extra:raise TimeoutError('opaque outside path')
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',timeout):self.rejected()

    def test_original_timer_is_not_reset(self):
        with patch.object(inventory.environment.library.sampler.bootstrap.time,'monotonic',return_value=700.0), \
                self.assertRaisesRegex(TimeoutError,'^compressed bootstrap batch deadline exhausted$'):
            inventory.verify_native_runtime_inventory(str(self.manifest),self.digest,batch_started_at=100.0)

    def test_exact_null_and_false_claims(self):
        original=copy.deepcopy(self.lock)
        for key,value in [('version',1.0),('candidate_processes_started',False),
                          ('actual_linker_transcript_replayed',True),('execution_admitted',0),('ptf_v1',.99)]:
            self.lock=copy.deepcopy(original);self.lock[key]=value;self.freeze();self.rejected()


if __name__=='__main__':unittest.main()
