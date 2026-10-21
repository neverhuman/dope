"""Authored synthetic bindings only; no real compile/confinement/admission claim."""
from dataclasses import replace
import gzip
import hashlib
import json
import struct
import unittest

from research.transfer.campaign_runner import CampaignRunnerError
from research.transfer.pilot_materialized_draws import DrawSpec
from research.transfer.pilot_control_materialized_draws import (
    TrustedCompileBinding, TypedControlDraw, control_pair,
)
from research.transfer.tdl import control_boundary as boundary


def h(raw):
    return hashlib.sha256(raw).hexdigest()


def fixture(method='compressed_real_rows', scale='primary_n', rows=2,
            sample_seed='0xffffffffffffffff', verifier=None):
    schema = boundary.canonical({'format': 'dope-table-schema', 'version': 1,
        'columns': [{'id': 'target', 'source_name': 'y', 'role': 'target', 'dtype': 'continuous'},
                    {'id': 'feature', 'source_name': 'x', 'role': 'feature', 'dtype': 'continuous'}]})
    source = b'x,y\n1,2\n3,4\n5,6\n'
    from research.transfer.target_last import normalize_csv
    normalized = normalize_csv(source,schema,target='y')
    source,schema = normalized.csv,normalized.schema

    from research.transfer.target_last import normalize_csv
    normalized = normalize_csv(source,schema,target='y')
    source,schema = normalized.csv,normalized.schema

    document = {'format': 'tdl-bounded-control-v1', 'method': method, 'fit_rows': 3,
                'fit_seed': (1 << 64) - 1, 'columns': 2, 'column_order': 'features_then_target'}
    if method == 'compressed_real_rows':
        document['rows'] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    else:
        document.update(quantile_count=3, quantiles=[[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]],
                        factor=[[1.0], [0.0, 1.0]], shrinkage=1e-6)
    compressed = gzip.compress(boundary.canonical(document), mtime=0)
    name = b'literal_payload'
    artifact = b'TDL1' + struct.pack('<II', 1, len(name)) + name + struct.pack('<Q', len(compressed)) + compressed
    compiler, compile_proof, admission = b'fixture compiler', b'fixture compile witness', b'fixture admission'
    binding = TrustedCompileBinding(method, h(artifact), h(b"aggregate-cell-input"), h(source), h(schema), h(compiler),
                                    h(compile_proof), h(admission), '0xffffffffffffffff', 3,
                                    ('feature', 'target'), 1 << 20)
    files = tuple(('/fixture/' + name, ('fixture ' + name).encode()) for name in boundary.SOURCES)
    files += (('/fixture/boundary.so', b'fixture library'), ('/fixture/python', b'fixture python'))
    source_pins = {name: h(dict(files)['/fixture/' + name]) for name in boundary.SOURCES}
    closure = [{'sha256': h(raw), 'bytes': len(raw)} for path, raw in sorted(files)]
    request = {'method': method, 'rows': rows, 'seed': int(sample_seed, 16), 'abi': boundary.ABI,
               'artifact_sha256': h(artifact), 'artifact_bytes': len(artifact),
               'library_sha256': h(b'fixture library'), 'source_sha256': h(boundary.canonical(source_pins)),
               'source_pins': source_pins, 'runtime_closure_sha256': h(boundary.canonical(closure)),
               'runtime_bytes': sum(x['bytes'] for x in closure),
               'closure_pins': {path: h(raw) for path, raw in files}}
    request_raw = boundary.canonical(request)
    raw = b''.join(struct.pack('<dd', 1.0, 2.0) for _ in range(rows))
    receipt = {k: request[k] for k in ('method', 'rows', 'seed', 'abi', 'artifact_sha256',
               'artifact_bytes', 'library_sha256', 'source_sha256', 'runtime_closure_sha256', 'runtime_bytes')}
    receipt.update(state='sampled', format='f64le', layout='row_major', columns=2,
                   column_order='features_then_target', sample_bytes=len(raw), sample_sha256=h(raw),
                   isolation='existing_rust_landlock_seccomp_tsync', invocation_bytes=len(request_raw),
                   sampling_closure_bytes=len(artifact) + request['runtime_bytes'] + len(request_raw),
                   pipeline_complete=False, scientific_admission=False, citable=False)
    spec = DrawSpec(scale, method, rows, 2, sample_seed, h(artifact), h(schema), h(b"aggregate-cell-input"), 1 << 20)
    # This is explicitly a synthetic trusted verifier fixture, not real authority.
    if verifier is None:
        verifier = lambda b, c, proof, grant: b == binding and proof == compile_proof and grant == admission
    return TypedControlDraw(spec, binding, raw, boundary.canonical(receipt), request_raw,
                            artifact, schema, source, compiler, compile_proof, admission,
                            files, 0, verifier)


class TypedControlBindingTests(unittest.TestCase):
    def test_real_worker_receipt_shape_has_no_echoed_schema(self):
        for method in ('compressed_real_rows', 'gaussian_copula'):
            draw = fixture(method)
            self.assertNotIn('schema', json.loads(draw.receipt))
            pool = draw.pool()
            self.assertEqual(pool.batch.features.rows, ((1.0,), (1.0,)))
            self.assertEqual(pool.batch.targets, (2.0, 2.0))
            self.assertEqual(pool.draw_seed, '0xffffffffffffffff')

    def test_hashes_without_trusted_compile_and_admission_refuse(self):
        with self.assertRaises(CampaignRunnerError):
            fixture(verifier=lambda *args: False)
        with self.assertRaises(CampaignRunnerError):
            fixture(verifier=lambda *args: 1)  # exact True, not a truthy declaration

    def test_same_width_schema_swap_cannot_authorize_compile(self):
        draw = fixture()
        schema = json.loads(draw.schema_document)
        schema['columns'][0]['id'] = 'different_target'
        changed = boundary.canonical(schema)
        # Even self-consistent replacement pins must pass the trusted compile witness.
        binding = replace(draw.binding, schema_sha256=h(changed), ordered_column_ids=('feature', 'different_target'))
        with self.assertRaises(CampaignRunnerError):
            replace(draw, schema_document=changed, binding=binding,
                    spec=replace(draw.spec, schema_sha256=h(changed)))

    def test_input_artifact_runtime_and_proof_tamper_refuse(self):
        draw = fixture()
        changes = [{'input_document': b'x,y\n9,9\n'}, {'artifact': draw.artifact + b'x'},
                   {'runtime_files': draw.runtime_files[:-1]}, {'compile_receipt': b'forged'},
                   {'admission_manifest': b'forged'}, {'compiler_source': b'foreign'}]
        for change in changes:
            with self.subTest(fields=tuple(change)):
                with self.assertRaises(CampaignRunnerError):
                    replace(draw, **change)

    def test_failed_exit_nonfinite_and_seed_rounding_refuse(self):
        draw = fixture()
        with self.assertRaises(CampaignRunnerError):
            replace(draw, returncode=1)
        with self.assertRaises(CampaignRunnerError):
            replace(draw, returncode=False)
        request = json.loads(draw.request)
        request['seed'] = float(request['seed'])
        with self.assertRaises(CampaignRunnerError):
            replace(draw, request=boundary.canonical(request))
        with self.assertRaises(CampaignRunnerError):
            replace(draw, raw=struct.pack('<dddd', float('nan'), 1.0, 2.0, 3.0))

    def test_n4n_distinct_streams_and_row_identity(self):
        left, right = fixture(), fixture(scale='sensitivity_4n', rows=8, sample_seed='0x0000000000000000')
        pools = control_pair(left, right)
        self.assertEqual(tuple(p.produced_rows for p in pools), (2, 8))
        self.assertFalse(set(left.row_ids) & set(right.row_ids))
        with self.assertRaises(CampaignRunnerError):
            control_pair(left, fixture(scale='sensitivity_4n', rows=8))
        with self.assertRaises(CampaignRunnerError):
            left.pool(forbidden_row_ids=left.row_ids)

    def test_pool_revalidates_deliberately_mutated_instance(self):
        draw = fixture()
        object.__setattr__(draw, 'raw', draw.raw[:-8])
        with self.assertRaises(CampaignRunnerError):
            draw.pool()

    def test_full_sampling_closure_not_artifact_only_budget(self):
        draw = fixture()
        budget = len(draw.artifact)
        binding = replace(draw.binding, byte_budget=budget)
        with self.assertRaises(CampaignRunnerError):
            replace(draw, binding=binding, spec=replace(draw.spec, byte_budget=budget),
                    trusted_verifier=lambda *args: True)


    def test_aggregate_input_is_not_the_csv_byte_identity(self):
        draw = fixture()
        self.assertNotEqual(draw.binding.input_sha256, draw.binding.fit_csv_sha256)
        self.assertEqual(draw.spec.input_sha256, draw.binding.input_sha256)
        self.assertEqual(draw.binding.fit_csv_sha256, h(draw.input_document))
        with self.assertRaises(CampaignRunnerError):
            replace(draw, binding=replace(draw.binding, fit_csv_sha256=draw.binding.input_sha256))

    def test_method_and_exact_draw_seed_changes_refuse(self):
        draw = fixture()
        for spec in (replace(draw.spec, method='gaussian_copula'),
                     replace(draw.spec, seed_hex='0xfffffffffffffffe')):
            with self.subTest(spec=spec):
                with self.assertRaises(CampaignRunnerError):
                    replace(draw, spec=spec)

    def test_width_only_header_match_is_not_a_schema_witness(self):
        draw = fixture()
        changed = draw.input_document.replace(b'x,y', b'other,y')
        with self.assertRaises(CampaignRunnerError):
            replace(draw, input_document=changed,
                    binding=replace(draw.binding, fit_csv_sha256=h(changed)),
                    trusted_verifier=lambda *args: True)

    def test_compiler_seed_change_with_fresh_binding_still_refuses_artifact(self):
        draw = fixture()
        with self.assertRaises(CampaignRunnerError):
            replace(draw, binding=replace(draw.binding, fit_seed_hex='0xfffffffffffffffe'),
                    trusted_verifier=lambda *args: True)
