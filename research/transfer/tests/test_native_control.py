"""Mandatory Linux native parity/denial tests; Root builds the real ELF first.

No mocked child can establish parity or isolation. These fixtures are authored,
not executed by the source preparation turn. Other platforms are unsupported.
"""
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import tempfile
import unittest
from unittest import mock

from research.transfer.tdl import control_boundary as b
from research.transfer.tdl import control_native as n
from research.transfer.tdl.generators import GaussianCopulaGenerator, CompressedRowsGenerator, _pack_control
from research.transfer.tdl.literal_worker import _control_document, _control_sample, _literal_payload

ROOT=Path(__file__).resolve().parents[3]
SUPPORTED=platform.system()=='Linux' and platform.machine()=='x86_64'
SEEDS=(0,1,11,(1<<53)-1,1<<53,(1<<63),(1<<64)-2,(1<<64)-1)


@unittest.skipUnless(SUPPORTED,'native-control-v2 explicitly Linux x86_64 only')
class NativeControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        override = os.environ.get('DOPE_NATIVE_CONTROL_BUILD_RECEIPT')
        if override is None:
            cls.runtime = json.loads((ROOT/'target/native-control-runtime.json').read_bytes())['runtime']
            cls.binary = Path(cls.runtime['executable']).resolve(strict=True)
        else:
            cls.binary=Path(os.environ.get('DOPE_NATIVE_CONTROL_BINARY',str(ROOT/'target/x86_64-unknown-linux-gnu/release/dope-native-control-runtime'))).resolve(strict=True)
            package=ROOT/'ops/native_control'
            pins={name:b.digest(b.regular_bytes(package/('src/'+name if name.endswith('.rs') else name))) for name in n.SOURCE_NAMES}
            binary=b.regular_bytes(cls.binary)
            cls.runtime={'scope':n.SCOPE,'abi':2,'executable':str(cls.binary),'executable_sha256':b.digest(binary),
                'source_dir':str(package),'source_pins':pins,'closure_pins':{str(cls.binary):b.digest(binary)},
                'build_receipt':str(Path(override).resolve(strict=True))}
            cls.runtime['build_receipt_sha256']=b.digest(b.regular_bytes(cls.runtime['build_receipt']))
        n.verify_runtime(cls.runtime)

    def setUp(self):
        parent=ROOT/'target/native-control-tests';parent.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=parent);self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name);self.counter=0
        source=self.directory/'input.csv';schema=self.directory/'schema.json'
        # Target first deliberately exercises source-order to feature-order binding.
        source.write_text('private_target,private_left,private_right\n'+''.join(
            f'{(i%7)-3.0},{(i-32)/7.0},{((i*13)%29-14)/3.0}\n' for i in range(64)))
        schema.write_bytes(b.canonical({'format':'dope-table-schema','version':1,'columns':[
            {'id':'c1','source_name':'private_left','role':'feature','dtype':'continuous'},
            {'id':'c2','source_name':'private_right','role':'feature','dtype':'continuous'},
            {'id':'c3','source_name':'private_target','role':'target','dtype':'continuous'}]}))
        # Raw target-first sources enter through the one explicit intake normalizer.
        # Every compiler call below receives only canonical target-last files.
        from research.transfer.target_last import normalize_csv
        normalized=normalize_csv(source.read_bytes(),schema.read_bytes(),target='private_target')
        source.write_bytes(normalized.csv);schema.write_bytes(normalized.schema)
        self.artifacts={}
        for method,kind in (('compressed_real_rows',CompressedRowsGenerator),('gaussian_copula',GaussianCopulaGenerator)):
            generator=kind(byte_budget=32768,artifact_abi=2);generator.fit(source,schema,(1<<64)-1)
            path=self.directory/(method+'.tdl');generator.export(path);self.artifacts[method]=path

    def sample(self,method,seed=11,rows=17,**options):
        self.counter+=1;artifact=options.pop('artifact',self.artifacts[method]);runtime=options.pop('runtime',self.runtime)
        cap=options.pop('byte_budget',1<<30) # explicit fixture cap, never a pilot cap waiver
        return b.sample_control(artifact,b.digest(artifact.read_bytes()),method,rows,seed,
            trusted_runtime=runtime,workdir=self.directory/str(self.counter),wall_seconds=30,
            byte_budget=cap,**options)

    def oracle(self,method,seed,rows):
        document=_control_document(_literal_payload(self.artifacts[method].read_bytes()),method)
        return b''.join(struct.pack('<d',v) for row in _control_sample(document,rows,seed) for v in row)

    def test_both_actual_algorithms_exact_python_bytes_all_uint64_edges(self):
        # Intentionally exact, including copula. A libm-vs-host-libm rounding
        # discrepancy is a real qualification failure, never tolerance-relaxed.
        for method in self.artifacts:
            for seed in SEEDS:
                with self.subTest(method=method,seed=seed):
                    raw,receipt=self.sample(method,seed)
                    self.assertEqual(raw,self.oracle(method,seed,17))
                    self.assertEqual(receipt['seed'],seed)
                    self.assertEqual(receipt['columns'],3)
                    self.assertFalse(receipt['scientific_admission'])

    def test_native_replay_n_4n_and_zero_maps_to_one(self):
        for method in self.artifacts:
            first,_=self.sample(method,0,16);again,_=self.sample(method,0,16)
            one,_=self.sample(method,1,16);four,_=self.sample(method,0,64)
            self.assertEqual(first,again);self.assertEqual(first,one);self.assertEqual(first,four[:len(first)])
            a,_=self.sample(method,(1<<64)-2,16);z,_=self.sample(method,(1<<64)-1,16)
            self.assertNotEqual(a,z)

    def test_cap_underflow_refuses_before_child_or_workdir(self):
        method='compressed_real_rows';path=self.artifacts[method]
        files=((self.runtime['executable'],self.binary.read_bytes()),)
        req=n.request(self.runtime,b.digest(path.read_bytes()),path.stat().st_size,method,16,11,files)
        exact=req['artifact_bytes']+req['runtime_bytes']+n.invocation_bytes(req)
        nextdir=self.directory/str(self.counter+1)
        with mock.patch.object(b,'bounded_child') as child:
            with self.assertRaises(b.BoundaryError):self.sample(method,11,16,byte_budget=exact-1)
        child.assert_not_called();self.assertFalse(nextdir.exists())
        raw,receipt=self.sample(method,11,16,byte_budget=exact)
        self.assertEqual(receipt['sampling_closure_bytes'],exact)
        self.assertEqual(receipt['invocation_bytes'],len(b.canonical(req))+104)
        self.assertEqual(len(raw),16*3*8)

    def test_real_open_socket_exec_and_inherited_read_denials(self):
        starts=[]
        raw,receipt=self.sample('gaussian_copula',capture_started=starts.append)
        self.assertEqual(len(starts),1);self.assertGreater(starts[0],0)
        fields=n.ATTEST.unpack(bytes.fromhex(receipt['native_attestation_hex']))
        self.assertEqual(fields[6],len(raw));self.assertEqual(fields[5],0x3f)
        retained=self.directory/str(self.counter)/'runtime/native.stderr.bin'
        self.assertEqual(retained.read_bytes(),bytes.fromhex(receipt['native_attestation_hex']))
        self.assertEqual(receipt['isolation'],n.ISOLATION)
        self.assertFalse(receipt['pipeline_complete'])

    def direct(self,artifact, *, method=1,rows=5,seed=11,request_mutator=None,extra_fd=None):
        self.counter+=1;root=self.directory/('direct-'+str(self.counter));root.mkdir(mode=0o700)
        (root/'control.tdl').write_bytes(artifact)
        request=n.WIRE.pack(b'DNC2REQ\0',method,0,rows,seed,len(artifact),hashlib.sha256(artifact).digest(),bytes(32))
        if request_mutator:request=request_mutator(request)
        (root/'request.bin').write_bytes(request)
        return subprocess.run([str(self.binary)],cwd=root,stdin=subprocess.DEVNULL,capture_output=True,
            pass_fds=() if extra_fd is None else (extra_fd,),timeout=30)

    def test_inherited_source_descriptor_is_actually_closed(self):
        source=self.directory/'outside-secret';source.write_bytes(b'NEVER_READ_PRIVATE_VALUE')
        with source.open('rb') as stream:
            result=self.direct(self.artifacts['compressed_real_rows'].read_bytes(),extra_fd=stream.fileno())
        self.assertEqual(result.returncode,0)
        self.assertEqual(n.ATTEST.unpack(result.stderr)[5],0x3f)
        self.assertNotIn(b'NEVER_READ_PRIVATE_VALUE',result.stdout+result.stderr)

    def test_corrupt_frame_crc_count_type_and_unknown_fields_refuse(self):
        good=self.artifacts['compressed_real_rows'].read_bytes()
        document=_control_document(_literal_payload(good),'compressed_real_rows')
        cases=[good[:-1],b'BAD!'+good[4:],good+b'extra',good[:-8]+bytes([good[-8]^1])+good[-7:],good[:-4]+bytes([good[-4]^1])+good[-3:]]
        for key,value in (('columns',True),('columns',0),('columns',65),('fit_seed',-1),
                          ('fit_seed',1<<64),('fit_seed',1.0),('rows',[]),('unknown',1)):
            changed=copy.deepcopy(document);changed[key]=value;cases.append(_pack_control(changed)[1])
        changed=copy.deepcopy(document);changed['rows'][0][0]=1<<53;cases.append(_pack_control(changed)[1])
        for raw in cases:
            with self.subTest(sha=b.digest(raw)):
                result=self.direct(raw)
                self.assertNotEqual(result.returncode,0);self.assertEqual(result.stdout,b'')
                self.assertEqual(result.stderr,b'native_control_refused\n')

    def test_duplicate_json_keys_nonfinite_and_trailing_gzip_refuse(self):
        raw=_literal_payload(self.artifacts['compressed_real_rows'].read_bytes())
        document=b.canonical(_control_document(raw,'compressed_real_rows'))
        self.assertIn(b'"columns":3',document)
        corrupt=[b'{"columns":3,'+document[1:],document.replace(b'"columns":3',b'"columns":NaN'),document+b'{}']
        from research.transfer.tdl.charge import canonical_container
        for encoded in corrupt:
            artifact=canonical_container([{'name':'literal_payload','hex':gzip.compress(encoded,mtime=0).hex()}])
            result=self.direct(artifact);self.assertNotEqual(result.returncode,0);self.assertEqual(result.stdout,b'')
        artifact=canonical_container([{'name':'literal_payload','hex':(raw+gzip.compress(b'{}')).hex()}])
        self.assertNotEqual(self.direct(artifact).returncode,0)

    def test_binary_payload_bounds_nonfinite_and_deflate_modes(self):
        from research.transfer.tdl.charge import canonical_container
        import zlib
        raw=gzip.decompress(_literal_payload(self.artifacts['compressed_real_rows'].read_bytes()))
        def artifact(payload,level=9,strategy=zlib.Z_DEFAULT_STRATEGY):
            compressor=zlib.compressobj(level,zlib.DEFLATED,31,9,strategy)
            encoded=compressor.compress(payload)+compressor.flush()
            return canonical_container([{'name':'literal_payload','hex':encoded.hex()}])
        expected=self.oracle('compressed_real_rows',11,5)
        for level,strategy in ((0,zlib.Z_DEFAULT_STRATEGY),(9,zlib.Z_FIXED),(9,zlib.Z_DEFAULT_STRATEGY)):
            result=self.direct(artifact(raw,level,strategy))
            self.assertEqual(result.returncode,0);self.assertEqual(result.stdout,expected)
        invalid=[raw[:-1],raw+b'X',b'BADMAGIC'+raw[8:]]
        for offset,wire in ((8,struct.pack('<I',3)),(12,struct.pack('<I',0)),
                            (16,struct.pack('<Q',4097)),(32,struct.pack('<I',3)),
                            (36,struct.pack('<I',0)),(40,struct.pack('<d',float('nan')))):
            invalid.append(raw[:offset]+wire+raw[offset+len(wire):])
        for payload in invalid:
            result=self.direct(artifact(payload))
            self.assertNotEqual(result.returncode,0);self.assertEqual(result.stdout,b'')
            self.assertEqual(result.stderr,b'native_control_refused\n')

    def test_invalid_request_refuses_before_artifact_access(self):
        # Wire validation and digest binding reject unsupported requests;
        # neither route prints a path or private value.
        good=self.artifacts['compressed_real_rows'].read_bytes()
        for field in ('method','rows','flags','magic','artifact_hash','request_size'):
            def mutate(raw):
                if field=='method':return raw[:8]+struct.pack('<I',3)+raw[12:]
                if field=='rows':return raw[:16]+struct.pack('<Q',0)+raw[24:]
                if field=='flags':return raw[:12]+struct.pack('<I',1)+raw[16:]
                if field=='magic':return b'BADMAGIC'+raw[8:]
                if field=='artifact_hash':return raw[:40]+bytes(32)+raw[72:]
                return raw+b'X'
            result=self.direct(good,request_mutator=mutate)
            self.assertNotEqual(result.returncode,0);self.assertEqual(result.stdout,b'')

    def test_elf_dynamic_or_wrong_abi_and_pin_refuse_without_child(self):
        for field,value in (('abi',1),('executable_sha256','f'*64)):
            rt=copy.deepcopy(self.runtime);rt[field]=value
            with mock.patch.object(b,'bounded_child') as child:
                with self.assertRaises(b.BoundaryError):self.sample('compressed_real_rows',runtime=rt)
            child.assert_not_called()
        raw=bytearray(self.binary.read_bytes());at=struct.unpack_from('<Q',raw,32)[0]
        struct.pack_into('<I',raw,at,3)
        with self.assertRaises(b.BoundaryError):n.static_elf(bytes(raw))

    def test_every_quantile_count_and_degenerate_marginals_match_oracle(self):
        for size in (3,5,9,17,33,65,129,257):
            doc={'format':'tdl-bounded-control-v1','method':'gaussian_copula',
                'fit_rows':64,'fit_seed':(1<<64)-1,'columns':2,'column_order':'features_then_target',
                'quantile_count':size,'quantiles':[[-7.0]*size,[float(i-size//2) for i in range(size)]],
                'factor':[[1.0],[0.0,1.0]],'shrinkage':1e-6}
            artifact=self.directory/('quantiles-'+str(size)+'.tdl');artifact.write_bytes(_pack_control(doc,artifact_abi=2)[1])
            raw,_=self.sample('gaussian_copula',(1<<64)-1,rows=19,artifact=artifact)
            expected=b''.join(struct.pack('<d',v) for row in _control_sample(doc,19,(1<<64)-1) for v in row)
            self.assertEqual(raw,expected)
        for field,value in (('quantile_count',4),('shrinkage',0),('fit_rows',2),
                            ('factor',[[1.0],[0.0,-1.0]]),('factor',[[1.0],[0.0,0.9]])):
            bad=copy.deepcopy(doc);bad[field]=value
            result=self.direct(_pack_control(bad)[1],method=2)
            self.assertNotEqual(result.returncode,0);self.assertEqual(result.stdout,b'')

    def test_finite_decimal_and_signed_zero_literal_bytes(self):
        doc={'format':'tdl-bounded-control-v1','method':'compressed_real_rows','fit_rows':2,
            'fit_seed':0,'columns':5,'column_order':'features_then_target',
            'rows':[[0.0,-0.0,1e-300,-1e200,9007199254740991],
                    [-0.0,0.0,1.7976931348623157e308,2.2250738585072014e-308,-9007199254740991]]}
        artifact=self.directory/'edge-floats.tdl';artifact.write_bytes(_pack_control(doc,artifact_abi=2)[1])
        raw,_=self.sample('compressed_real_rows',(1<<64)-1,rows=32,artifact=artifact)
        expected=b''.join(struct.pack('<d',v) for row in _control_sample(doc,32,(1<<64)-1) for v in row)
        self.assertEqual(raw,expected)

    def test_invalid_python_count_and_seed_types_never_spawn(self):
        for seed in (-1,1<<64,1.0,True):
            with mock.patch.object(b,'bounded_child') as child, mock.patch.object(n,'verify_runtime') as access:
                with self.assertRaises(b.BoundaryError):self.sample('compressed_real_rows',seed)
            access.assert_not_called()
            child.assert_not_called()
        for rows in (0,1_000_001,1.0,True):
            with mock.patch.object(b,'bounded_child') as child:
                with self.assertRaises(b.BoundaryError):self.sample('compressed_real_rows',rows=rows)
            child.assert_not_called()

    def test_native_attestation_tamper_refuses(self):
        raw,receipt=self.sample('compressed_real_rows')
        req=json.loads((self.directory/str(self.counter)/'runtime/request.json').read_bytes())
        for index in (0,8,16,24,32,36,40,48,80,112):
            altered=bytearray.fromhex(receipt['native_attestation_hex']);altered[index]^=1
            with self.assertRaises(b.BoundaryError):n.attest(raw,bytes(altered),req)


@unittest.skipUnless(SUPPORTED,'native-control-v2 explicitly Linux x86_64 only')
class NativeTypedParentTests(unittest.TestCase):
    """Actual native child through original live eight-stage fixture journal."""
    @classmethod
    def setUpClass(cls):
        NativeControlTests.setUpClass()
        cls.runtime=copy.deepcopy(NativeControlTests.runtime)

    def setUp(self):
        from research.transfer.tests import test_pilot_worker_runtime as fixtures
        fixtures.BoundWorkerTests.setUp(self)

    def prepare_native(self,method,budget):
        from research.transfer.tests import test_pilot_worker_runtime as fixtures
        from research.transfer.tests import test_pilot_control_runtime as control_fixtures
        from research.transfer.pilot_control_runtime import PilotControlRuntime,COMPILER_FILES
        from research.transfer.pilot_native_seed import ORIGINAL_STREAM_HEX
        # This is explicitly a fixture-only executor, never a Root admission.
        class NativeFixtureExecutor(control_fixtures.FixtureExecutor):
            def __call__(self,operation,claims,evidence):
                if operation=='draw_handoff':
                    if (evidence!=control_fixtures.GRANT or claims['cell_plan_sha256']!=self.cell_sha or
                        claims['library_sha256']!=self.runtime['executable_sha256']):
                        raise ValueError('native_fixture_handoff_mismatch')
                    self.events.append(operation)
                    return None
                return super().__call__(operation,claims,evidence)
        executor=NativeFixtureExecutor(self.runtime)
        source=Path(b.__file__).resolve().parent
        control=PilotControlRuntime(trusted_runtime=self.runtime,
            compiler_source_pins={name:b.digest(b.regular_bytes(source/name)) for name in COMPILER_FILES},
            admission_manifest=control_fixtures.GRANT,verify_operation=executor)
        prepared=fixtures.BoundWorkerTests.prepare(self,comparator=method,control_runtime=control,
            byte_budget=budget,learner_seed=ORIGINAL_STREAM_HEX)
        executor.cell_sha=prepared[3].sha256
        return prepared,control

    def test_actual_native_parent_into_fixture_learners_and_retained_reload(self):
        from research.transfer.pilot_runner import execute_pilot_cell
        from research.transfer.pilot_materialized_draws import load_pair
        # Separately declared fixture cap; actual pilot caps stay unchanged.
        budget=Path(self.runtime['executable']).stat().st_size+(1<<20)
        prepared,control=self.prepare_native('compressed_real_rows',budget)
        spec,pilot,contract,cell,root,auth,worker=prepared
        receipt=execute_pilot_cell(spec,pilot,contract,cell,root,worker_runtime=worker,**auth)
        self.assertEqual(receipt['status'],'ok',receipt)
        self.assertEqual(len(receipt['stages']),8);self.assertEqual(control.processes,2)
        self.assertFalse(receipt['scientific_admission'])
        pair=load_pair(root,receipt['materialized_draw_manifest_sha256'],artifact=root/'model.raw',
            comparator='compressed_real_rows',control_runtime=control)
        self.assertEqual([draw.spec.rows for draw in pair],[16,64])
        for draw in pair:
            request=json.loads(draw.request);report=json.loads(draw.receipt)
            self.assertEqual(request['abi'],2)
            self.assertEqual(report['isolation'],n.ISOLATION)
            self.assertEqual(report['sampling_closure_bytes'],request['artifact_bytes']+
                request['runtime_bytes']+len(b.canonical(request))+104)
            for features,target in zip(draw.pool().batch.features.rows,draw.pool().batch.targets):
                self.assertAlmostEqual(target,10*features[0],places=9)

    def test_32768_reserves_complete_native_sampling_closure_for_both_controls(self):
        from research.transfer.pilot_runner import execute_pilot_cell
        for method in ('compressed_real_rows', 'gaussian_copula'):
            with self.subTest(method=method):
                original = self.directory
                self.directory = original/method
                self.directory.mkdir(mode=0o700)
                try:
                    prepared, control = self.prepare_native(method, 32768)
                    spec,pilot,contract,cell,root,auth,worker = prepared
                    receipt = execute_pilot_cell(spec,pilot,contract,cell,root,
                                                 worker_runtime=worker,**auth)
                    self.assertEqual(receipt['status'], 'ok', receipt)
                    self.assertEqual(control.processes, 2)
                    charge = json.loads((root/'control-sampling-charge.json').read_bytes())
                    witness = json.loads((root/'control-compile-witness.json').read_bytes())
                    self.assertLessEqual(max(charge['per_draw_bytes']), 32768)
                    self.assertGreater(witness['sampling_reservation']['runtime_and_max_invocation_bytes'], 0)
                    self.assertLess(witness['artifact_ceiling'], 32768)
                finally:
                    self.directory = original

    def test_original_caps_are_never_raised_and_charge_is_terminal_when_too_small(self):
        from research.transfer.pilot_runner import execute_pilot_cell
        from research.transfer.tests import test_pilot_learning_execution as fixtures
        # Each is a new independent synthetic fixture; no empirical retry.
        for cap in (4096,32768):
            with self.subTest(cap=cap):
                # The fixture factory uses exclusive files, so give each its own directory.
                original=self.directory
                self.directory=original/str(cap);self.directory.mkdir(mode=0o700)
                try:
                    prepared,control=self.prepare_native('compressed_real_rows',cap)
                    spec,pilot,contract,cell,root,auth,worker=prepared
                    receipt=execute_pilot_cell(spec,pilot,contract,cell,root,worker_runtime=worker,**auth)
                    self.assertEqual(cell.document()['byte_budget'],cap)
                    charge=json.loads((root/'control-sampling-charge.json').read_bytes())
                    if max(charge['per_draw_bytes'])>cap:
                        self.assertEqual(receipt['stages'][-1]['status'],'over_budget')
                        self.assertEqual(control.processes,0)
                    else:
                        self.assertEqual(receipt['status'],'ok',receipt)
                    self.assertFalse(receipt['scientific_admission'])
                finally:
                    self.directory=original


if __name__=='__main__':unittest.main()
