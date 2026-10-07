"""Reject provider drift before any dependency initializer can execute."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research.benchmark import arf_runtime_guard as guard
from research.benchmark import arf_adapter as adapter


class Guard(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[3] / 'target/arf-guard-controls'
        base.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=base); self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        prefix = self.base / 'python312'; python = prefix / 'bin/python3.12'
        python.parent.mkdir(parents=True); python.write_bytes(b'owned interpreter fixture')
        library = prefix / 'lib/python3.12'; library.mkdir(parents=True)
        dynload = library / 'lib-dynload'; dynload.mkdir()
        (library / 'os.py').write_bytes(b'# owned stdlib fixture')
        source = self.base / 'source'; source.mkdir()
        for name in ('guard.py','adapter.py','native.py'):(source/name).write_bytes(b'# owned guard fixture')
        site = library / 'site-packages'; (site / 'numpy').mkdir(parents=True)
        self.numpy = site / 'numpy/__init__.py'; self.numpy.write_bytes(b'raise AssertionError("initializer executed")')
        self.manifest = self.base / 'inventory.json'
        rows = {str(p): dict(bytes=p.stat().st_size, sha256=guard.hash_file(p))
                for p in prefix.rglob('*') if p.is_file()}
        self.manifest.write_text(json.dumps(dict(roots=[str(prefix)], files=rows, aliases=[])))
        cache = self.base / 'empty-cache'; cache.mkdir()
        self.lock = self.base / 'runtime.lock.json'
        self.lock.write_text(json.dumps(dict(format='dope-arf-pandas-compatible-runtime-preparation',version=2,
            author_commit='8b63c1b3999981125b4af2828ff52cba8e29169d',
            directory_inventories={str(prefix):[str(p) for p in prefix.rglob('*') if p.is_dir()],str(source):[]},
            new_generator_fits_started=0, official_tests_opened=False,
            execution_admitted=False, inventory_locks=[dict(path=str(self.manifest),sha256=guard.hash_file(self.manifest))],
            source_root=str(source), source_files={str(p):dict(bytes=p.stat().st_size,sha256=guard.hash_file(p)) for p in source.iterdir()},
            python=str(python), python_sha256=guard.hash_file(python), stdlib_root=str(library),
            stdlib_dynload_root=str(dynload), absent_zip=str(prefix/'lib/python312.zip'),
            empty_cache=str(cache), package_sites=[str(site)])))
        self.sha = guard.hash_file(self.lock)
        self.original_path = list(sys.path); self.addCleanup(lambda: sys.path.__setitem__(slice(None), self.original_path))
        for context in (patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':''}),
                        patch.object(sys,'executable',str(python)),
                        patch.object(sys,'pycache_prefix',str(cache)),
                        patch.object(sys,'flags',SimpleNamespace(isolated=1,no_site=1)),
                        patch.object(guard,'__file__',str(source/'guard.py')),
                        patch.object(adapter,'__file__',str(source/'adapter.py')),
                        patch.object(adapter.native,'__file__',str(source/'native.py')),
                        patch.object(sys,'dont_write_bytecode',True)):
            context.start();self.addCleanup(context.stop)

    def rejected_before_import(self, action='probe', arguments=None):
        original = __import__
        initialized = []
        def deny(name, *args, **kwargs):
            if name.split('.')[0] in ('numpy','pandas','scipy','sklearn','arfpy'):
                initialized.append(name);raise AssertionError('dependency initializer reached')
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=deny):
            with self.assertRaises(ValueError):
                if action=='probe':guard.probe(self.lock,self.sha,self.base)
                else:
                    kw=dict(runtime_path=self.lock,runtime_sha256=self.sha,base=self.base,deadline_epoch=__import__('time').time()+590)
                    kw.update(arguments or {})
                    getattr(adapter,action)(**kw)
        self.assertEqual(initialized,[])

    def test_valid_owned_runtime_declaration(self):
        self.assertEqual(guard.verify(self.lock,self.sha,self.base)['package_sites'],[str(self.numpy.parents[1])])

    def test_dependency_initializer_changed_before_import(self):
        self.numpy.write_bytes(b'raise RuntimeError("different initializer")');self.rejected_before_import()

    def test_added_bytecode_rejected_before_import(self):
        cache=self.numpy.parent/'__pycache__';cache.mkdir();(cache/'__init__.cpython-312.pyc').write_bytes(b'unbound code');self.rejected_before_import()

    def test_added_directory_alias_rejected_before_import(self):
        (self.numpy.parent/'alias').symlink_to(self.numpy.parent,target_is_directory=True);self.rejected_before_import()

    def test_interpreter_zip_appearance_rejected_before_import(self):
        Path(json.loads(self.lock.read_bytes())['absent_zip']).write_bytes(b'unbound archive');self.rejected_before_import()

    def test_wrong_anchor_rejected_before_import(self):
        self.sha='0'*64;self.rejected_before_import()

    def test_non_builtin_digest_rejected_before_import(self):
        class Digest(str):pass
        self.sha=Digest(self.sha);self.rejected_before_import()

    def test_cuda_environment_checked_before_manifest_read(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}),patch.object(guard,'bound',side_effect=AssertionError('manifest read too early')):
            with self.assertRaises(ValueError):guard.verify(self.lock,self.sha,self.base)

    def fit_request(self):
        worker=self.base/'worker';worker.mkdir()
        files={}
        for name,data in [('train.csv',b'0.1,0.2\n0.2,0.3\n'),('validation.csv',b'0.15,0.25\n0.25,0.35\n'),('projection.json',b'{}')]:
            p=worker/name;p.write_bytes(data);files[name]=p
        return dict(train=files['train.csv'],train_sha256=guard.hash_file(files['train.csv']),
            validation=files['validation.csv'],validation_sha256=guard.hash_file(files['validation.csv']),
            projection=files['projection.json'],projection_sha256=guard.hash_file(files['projection.json']),
            artifact=self.base/'artifact',configuration=dict(num_trees=10,min_node_size=20,max_iters=1,alpha=0.0),fit_seed=11)

    def test_fit_and_sample_reject_dependency_drift_before_initializer(self):
        kwargs=self.fit_request();self.numpy.write_bytes(b'raise RuntimeError("changed provider")')
        self.rejected_before_import('fit',kwargs)
        self.rejected_before_import('sample',dict(artifact=self.base/'artifact',expected_inventory={},sample_seed=101,rows=8,output=self.base/'sample.csv'))

    def test_fit_rejects_changed_training_before_initializer(self):
        kwargs=self.fit_request();kwargs['train'].write_bytes(b'private_malformed_marker,0.2\n0.2,0.3\n')
        self.rejected_before_import('fit',kwargs)

    def test_fit_rejects_sealed_worker_before_initializer(self):
        kwargs=self.fit_request();(kwargs['train'].parent/'test.csv').write_bytes(b'never open this')
        self.rejected_before_import('fit',kwargs)

    def test_fit_rejects_overlong_deadline_before_initializer(self):
        kwargs=self.fit_request();kwargs['deadline_epoch']=__import__('time').time()+1200
        self.rejected_before_import('fit',kwargs)

    def test_fit_rejects_noninteger_seed_before_initializer(self):
        kwargs=self.fit_request();kwargs['fit_seed']=11.0
        self.rejected_before_import('fit',kwargs)

    def test_fit_rejects_boolean_configuration_before_initializer(self):
        kwargs=self.fit_request();kwargs['configuration']['num_trees']=True
        self.rejected_before_import('fit',kwargs)

    def test_cpu_allocation_rejected_before_runtime_or_dependency_initialization(self):
        kwargs=self.fit_request()
        with patch.object(adapter, 'check_runtime', side_effect=AssertionError('runtime reached')):
            for threads in (True, 0, 17, 4.0, '4'):
                with self.subTest(threads=threads):
                    self.rejected_before_import('fit', kwargs | {'cpu_threads': threads})
            with patch.object(adapter.os, 'sched_getaffinity', return_value={60, 61}):
                self.rejected_before_import('fit', kwargs | {'cpu_threads': 4})

    def test_four_core_fit_passes_allocation_to_original_author_constructor(self):
        kwargs=self.fit_request()
        constructor=__import__('unittest.mock', fromlist=['Mock']).Mock(side_effect=RuntimeError('stop at fit'))
        numpy=SimpleNamespace(random=SimpleNamespace(seed=lambda _: None), asarray=lambda x: x)
        author=SimpleNamespace(arf=constructor)
        with patch.object(adapter, 'check_runtime'), \
             patch.object(adapter.native, 'prepare_frames', return_value=('train', 'validation')), \
             patch.object(adapter.os, 'sched_getaffinity', return_value={60,61,62,63}), \
             patch.dict(sys.modules, {'numpy': numpy, 'arfpy': SimpleNamespace(), 'arfpy.arf': author}):
            with self.assertRaisesRegex(RuntimeError, 'stop at fit'):
                adapter.fit(runtime_path=self.lock, runtime_sha256=self.sha, base=self.base,
                            deadline_epoch=__import__('time').time()+590, cpu_threads=4, **kwargs)
        self.assertEqual(constructor.call_args.kwargs['n_jobs'], 4)
        self.assertEqual(constructor.call_args.kwargs['random_state'], 11)
        self.assertEqual(constructor.call_args.args, ('train',))

    def test_extra_empty_runtime_directory_rejected_before_import(self):
        (self.numpy.parent/'unbound_namespace').mkdir();self.rejected_before_import()

    def test_duplicate_bound_json_key_rejected(self):
        self.lock.write_bytes(b'{"version":2,"version":2}')
        self.sha=guard.hash_file(self.lock);self.rejected_before_import()

    def test_malformed_numeric_error_never_echoes_value(self):
        import traceback
        marker='private_numeric_fixture_marker'
        try:adapter.numeric((marker+',0.2\n0.3,0.4\n').encode())
        except ValueError as error:
            self.assertEqual(str(error),'invalid common-numeric input')
            self.assertNotIn(marker,''.join(traceback.format_exception(error)))
        else:self.fail('malformed numeric input accepted')

    def sample_request(self):
        artifact=self.base/'artifact';artifact.mkdir()
        blobs={n:b'opaque factor fixture\n' for n in ('bounds.csv','continuous.csv','categories.csv')}
        metadata=dict(format='dope-arf-forge-factors-v1',source_rows_required=False,
            restricted_research_artifact=True,names=['c0','c1'],num_trees=10,dist='truncnorm',
            factor_columns=[False,False],object_columns=[False,False],levels={},
            files={n:hashlib.sha256(data).hexdigest() for n,data in blobs.items()})
        blobs.update({'model.json':json.dumps(metadata).encode(),'projection.json':b'{}',
            'adapter.json':json.dumps(dict(format='dope-arf-research-adapter-v1',
                runtime_sha256=self.sha,source_rows_required=False,support='clip_to_unit_interval')).encode()})
        for n,data in blobs.items():(artifact/n).write_bytes(data)
        return dict(artifact=artifact,expected_inventory=adapter.inventory(artifact),sample_seed=101,
                    rows=8,output=self.base/'sample.csv')

    def test_sample_rejects_changed_factor_before_initializer(self):
        kwargs=self.sample_request();(kwargs['artifact']/'continuous.csv').write_bytes(b'changed')
        self.rejected_before_import('sample',kwargs)

    def test_sample_rejects_added_artifact_alias_before_initializer(self):
        kwargs=self.sample_request();(kwargs['artifact']/'alias').symlink_to(kwargs['artifact'],target_is_directory=True)
        self.rejected_before_import('sample',kwargs)

    def test_sample_rejects_float_seed_before_initializer(self):
        kwargs=self.sample_request();kwargs['sample_seed']=101.0
        self.rejected_before_import('sample',kwargs)

    def test_bound_json_rejects_float_overflow(self):
        with self.assertRaises(ValueError):guard.decode(b'{"value":1e999}')

    def test_different_executing_guard_rejected_before_initializer(self):
        with patch.object(guard,'__file__',str(self.base/'inventory.json')):self.rejected_before_import()

    def test_different_executing_adapter_rejected_before_initializer(self):
        kwargs=self.fit_request()
        with patch.object(adapter,'__file__',str(self.base/'inventory.json')):self.rejected_before_import('fit',kwargs)


class FixturePathTests(unittest.TestCase):
    def test_fixture_root_does_not_depend_on_working_directory(self):
        target=Path(__file__).resolve().parents[3]/'target';target.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target,prefix='arf-cwd-control-') as directory:
            cwd=Path.cwd()
            try:
                os.chdir(directory)
                outcome=Guard('test_valid_owned_runtime_declaration').run()
                self.assertTrue(outcome.wasSuccessful(),outcome.errors+outcome.failures)
                self.assertFalse((Path(directory)/'target').exists())
            finally:os.chdir(cwd)


if __name__=='__main__':unittest.main()
