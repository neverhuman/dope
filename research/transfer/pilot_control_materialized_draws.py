"""Typed numeric control handoff; trusted compile/admission binding is mandatory.

No compile, sampling, admission manufacture or v1 materializer override occurs
here. The existing confined worker does not emit a PublicColumn schema. Its
receipt is joined to separately authenticated compile provenance, never an
invented schema echo. Root must supply a live trusted verifier of that provenance.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import csv
import hashlib
import io
import math
import struct
import zlib
from typing import Callable

from .campaign_admission import seed_uint64
from .campaign_artifacts import MAX_INPUT
from .campaign_runner import CampaignRunnerError
from .evidence import digest
from .pilot_materialized_draws import DrawSpec, _native_json
from .tdl import control_boundary as boundary
from .tdl.literal_worker import _control_document, _literal_payload, MAX_CONTROL_ROWS

METHODS = ('gaussian_copula', 'compressed_real_rows')


def require(ok):
    if not ok:
        raise CampaignRunnerError('typed_control_binding_refused')


def sha(raw):
    require(type(raw) is bytes and len(raw) <= boundary.MAX_FILE)
    return hashlib.sha256(raw).hexdigest()


def pin(value):
    require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value))


@dataclass(frozen=True, repr=False)
class TrustedCompileBinding:
    """Owner-authenticated compile accession, not a producer schema receipt.

    The verifier must establish actual compile input/schema/seed/artifact/source
    provenance AND current rights/runtime/owner/resource scope from retained
    evidence. Hash matches or a caller-supplied true flag do not grant admission.
    """
    method: str
    artifact_sha256: str
    input_sha256: str  # aggregate cell input identity
    fit_csv_sha256: str  # exact captured fit CSV bytes
    schema_sha256: str
    compiler_source_sha256: str
    compile_receipt_sha256: str
    admission_manifest_sha256: str
    fit_seed_hex: str
    fit_rows: int
    ordered_column_ids: tuple[str, ...]
    byte_budget: int

    def __post_init__(self):
        require(self.method in METHODS)
        for value in (self.artifact_sha256, self.input_sha256, self.fit_csv_sha256, self.schema_sha256,
                      self.compiler_source_sha256, self.compile_receipt_sha256,
                      self.admission_manifest_sha256):
            pin(value)
        seed_uint64(self.fit_seed_hex)
        require(type(self.fit_rows) is int and 0 < self.fit_rows <= MAX_CONTROL_ROWS)
        require(type(self.byte_budget) is int and 0 < self.byte_budget <= 1 << 30)
        require(type(self.ordered_column_ids) is tuple and 2 <= len(self.ordered_column_ids) <= 64
                and all(type(x) is str and x for x in self.ordered_column_ids)
                and len(set(self.ordered_column_ids)) == len(self.ordered_column_ids))

    @property
    def sha256(self):
        return digest({**asdict(self), 'ordered_column_ids': list(self.ordered_column_ids)})


def _compile_schema(schema_raw, input_raw, binding):
    require(sha(schema_raw) == binding.schema_sha256 and sha(input_raw) == binding.fit_csv_sha256
            and len(input_raw) <= MAX_INPUT)
    from .tdl.target_last import require_target_last
    require_target_last(input_raw, schema_raw)
    schema = _native_json(schema_raw)
    require(type(schema) is dict and schema.get('format') == 'dope-table-schema'
            and type(schema.get('version')) is int and schema['version'] == 1)
    columns = schema['columns']
    require(type(columns) is list and len(columns) == len(binding.ordered_column_ids))
    require(all(type(c) is dict and c.get('dtype') == 'continuous'
                and c.get('role') in ('feature', 'target') for c in columns))
    require(sum(c['role'] == 'target' for c in columns) == 1)
    headers = [c['source_name'] for c in columns]
    ids = [c['id'] for c in columns]
    require(all(type(v) is str and v for v in headers + ids)
            and len(set(headers)) == len(headers) and len(set(ids)) == len(ids))
    ordered = sorted(columns, key=lambda c: c['role'] == 'target')
    require(tuple(c['id'] for c in ordered) == binding.ordered_column_ids)
    table = list(csv.reader(io.StringIO(input_raw.decode('utf-8'))))
    require(table and len(table[0]) == len(headers) and len(set(table[0])) == len(headers))
    require(set(table[0]) == set(headers) and len(table) - 1 == binding.fit_rows)
    require(all(len(r) == len(headers) for r in table[1:]))
    require(all(math.isfinite(float(v)) for r in table[1:] for v in r))
    return tuple(c['id'] for c in ordered)


def _runtime(request, files):
    require(type(files) is tuple and all(type(x) is tuple and len(x) == 2 for x in files))
    require(all(type(path) is str and type(raw) is bytes for path, raw in files))
    require(len({path for path, _ in files}) == len(files))
    captured = dict(files)
    require(type(request['closure_pins']) is dict and set(captured) == set(request['closure_pins']))
    closure = []
    for path in sorted(captured):
        expected = request['closure_pins'][path]
        pin(expected)
        require(sha(captured[path]) == expected)
        closure.append({'sha256': expected, 'bytes': len(captured[path])})
    require(type(request['runtime_bytes']) is int and request['runtime_bytes'] == sum(x['bytes'] for x in closure))
    require(request['runtime_closure_sha256'] == boundary.digest(boundary.canonical(closure)))
    require(type(request['abi']) is int and request['abi'] in (boundary.ABI, 2))
    if request['abi'] == 2:
        from .tdl.control_native import SOURCE_NAMES, static_elf
        require(len(files) == 1 and request['library_sha256'] == sha(files[0][1]))
        static_elf(files[0][1])
        require(set(request['source_pins']) == set(SOURCE_NAMES))
        for expected in request['source_pins'].values():
            pin(expected)
        require(request['source_sha256'] == boundary.digest(boundary.canonical(request['source_pins'])))
        return  # compiled source provenance is authenticated by the trusted parent

    require(request['library_sha256'] in request['closure_pins'].values())
    require(type(request['source_pins']) is dict and set(request['source_pins']) == set(boundary.SOURCES))
    for name, expected in request['source_pins'].items():
        pin(expected)
        matches = [raw for path, raw in files if path.endswith('/' + name)]
        require(any(sha(raw) == expected for raw in matches))
    require(request['source_sha256'] == boundary.digest(boundary.canonical(request['source_pins'])))


@dataclass(frozen=True, repr=False)
class TypedControlDraw:
    """Private retained bytes; direct construction and each Pool use reverify."""
    spec: DrawSpec
    binding: TrustedCompileBinding
    raw: bytes
    receipt: bytes
    request: bytes
    artifact: bytes
    schema_document: bytes
    input_document: bytes
    compiler_source: bytes
    compile_receipt: bytes
    admission_manifest: bytes
    runtime_files: tuple[tuple[str, bytes], ...]
    returncode: int
    trusted_verifier: Callable

    def __post_init__(self):
        self.validate()

    def validate(self):
        try:
            require(type(self.spec) is DrawSpec and type(self.binding) is TrustedCompileBinding)
            require(callable(self.trusted_verifier))
            spec, binding = self.spec, self.binding
            # Reconstruct immutable contracts to catch deliberate frozen-object mutation.
            DrawSpec(**asdict(spec))
            TrustedCompileBinding(**asdict(binding))
            require(spec.method == binding.method and spec.method in METHODS)
            require((spec.artifact_sha256, spec.schema_sha256, spec.input_sha256, spec.byte_budget) ==
                    (binding.artifact_sha256, binding.schema_sha256, binding.input_sha256, binding.byte_budget))
            require(sha(self.artifact) == binding.artifact_sha256 and len(self.artifact) <= binding.byte_budget)
            require(sha(self.compiler_source) == binding.compiler_source_sha256)
            require(sha(self.compile_receipt) == binding.compile_receipt_sha256
                    and sha(self.admission_manifest) == binding.admission_manifest_sha256)
            column_ids = _compile_schema(self.schema_document, self.input_document, binding)
            document = _control_document(_literal_payload(self.artifact), spec.method)
            require(document['fit_seed'] == seed_uint64(binding.fit_seed_hex)
                    and document['fit_rows'] == binding.fit_rows
                    and document['columns'] == len(column_ids) == spec.columns)
            request, receipt = _native_json(self.request), _native_json(self.receipt)
            require(set(receipt) == {'method', 'rows', 'seed', 'abi', 'artifact_sha256',
                    'artifact_bytes', 'library_sha256', 'source_sha256', 'runtime_closure_sha256',
                    'runtime_bytes', 'state', 'format', 'layout', 'columns', 'column_order',
                    'sample_bytes', 'sample_sha256', 'isolation', 'invocation_bytes',
                    'sampling_closure_bytes', 'pipeline_complete', 'scientific_admission', 'citable'} |
                    ({'native_attestation_hex'} if request.get('abi') == 2 else set()))
            require(set(request) == {'method', 'rows', 'seed', 'abi', 'artifact_sha256',
                    'artifact_bytes', 'library_sha256', 'source_sha256', 'source_pins',
                    'runtime_closure_sha256', 'runtime_bytes', 'closure_pins'})
            require(self.request == boundary.canonical(request))
            require(type(self.returncode) is int and self.returncode == 0)
            require(type(request['rows']) is int and request['rows'] == spec.rows
                    and type(request['seed']) is int and request['seed'] == seed_uint64(spec.seed_hex)
                    and request['method'] == spec.method
                    and request['artifact_sha256'] == binding.artifact_sha256
                    and type(request['artifact_bytes']) is int and request['artifact_bytes'] == len(self.artifact))
            _runtime(request, self.runtime_files)
            require(type(self.raw) is bytes and len(self.raw) <= MAX_INPUT)
            boundary.validate_output(self.returncode, self.raw, self.receipt, request)
            require(receipt['columns'] == spec.columns)
            require(type(receipt['sampling_closure_bytes']) is int
                    and receipt['sampling_closure_bytes'] <= spec.byte_budget)
            # A separate trusted parent authenticates provenance, live scope,
            # confinement/exit and runtime identity. Neither JSON nor hashes do.
            claims = {'binding_sha256': binding.sha256, 'spec': asdict(spec),
                      'request_sha256': sha(self.request), 'receipt_sha256': sha(self.receipt),
                      'sample_sha256': sha(self.raw), 'runtime_closure_sha256': request['runtime_closure_sha256'],
                      'source_sha256': request['source_sha256'], 'library_sha256': request['library_sha256'],
                      'sampling_closure_bytes': receipt['sampling_closure_bytes']}
            require(self.trusted_verifier(binding, claims, self.compile_receipt, self.admission_manifest) is True)
        except (ValueError, TypeError, LookupError, ArithmeticError, RuntimeError, OSError, csv.Error, EOFError, zlib.error):
            raise CampaignRunnerError('typed_control_binding_refused') from None

    @property
    def row_ids(self):
        identity = {'format': 'typed-control-draw-v1', 'binding': self.binding.sha256,
                    'spec': asdict(self.spec), 'sample_sha256': sha(self.raw)}
        return tuple(digest({**identity, 'row': i}) for i in range(self.spec.rows))

    def pool(self, *, forbidden_row_ids=()):
        from .adapters.base import FeatureBatch, TrainingBatch
        from .pilot_learning_contract import Pool
        self.validate()
        require(not set(self.row_ids) & set(forbidden_row_ids))
        values = tuple(v for (v,) in struct.iter_unpack('<d', self.raw))
        rows = tuple(values[i:i + self.spec.columns] for i in range(0, len(values), self.spec.columns))
        training = TrainingBatch(FeatureBatch(self.row_ids, tuple(r[:-1] for r in rows),
                                              self.spec.schema_sha256),
                                 tuple(r[-1] for r in rows), 'synthetic_train')
        return Pool(training, 'synthetic_draw', self.spec.input_sha256, self.spec.artifact_sha256,
                    self.spec.rows, self.spec.rows, self.spec.seed_hex)


def control_pair(primary, sensitivity, *, forbidden_row_ids=()):
    """Validate independent n/4n draws without modifying the DOPE/v1 loader."""
    require(type(primary) is TypedControlDraw and type(sensitivity) is TypedControlDraw)
    primary.validate()
    sensitivity.validate()
    require(primary.binding == sensitivity.binding)
    require(primary.spec.scale == 'primary_n' and sensitivity.spec.scale == 'sensitivity_4n'
            and sensitivity.spec.rows == primary.spec.rows * 4
            and primary.spec.seed_hex != sensitivity.spec.seed_hex)
    left = primary.pool(forbidden_row_ids=forbidden_row_ids)
    right = sensitivity.pool(forbidden_row_ids=tuple(forbidden_row_ids) + primary.row_ids)
    return left, right
