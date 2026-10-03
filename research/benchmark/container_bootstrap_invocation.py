"""Plan restricted interpreter inputs; never launch or admit a research job."""
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import PurePosixPath
import re
import time

BASE = PurePosixPath('/mnt/fast-scratch/dope-benchmark')
BATCH_SECONDS = 600


def require(condition):
    if not condition:
        raise ValueError('compressed bootstrap proposal rejected')


def digest(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None)
    return value


def path(value, scratch=False):
    require(type(value) is str and not any(c in value for c in ('\0', '\n', '\r', ':')))
    result = PurePosixPath(value)
    require(result.is_absolute() and result.as_posix() == value and '..' not in result.parts)
    require('evaluator' not in result.parts and 'test.csv' not in result.parts)
    if scratch:
        require(result.is_relative_to(BASE) and result != BASE)
    return result


@dataclass(frozen=True)
class InvocationPlan:
    host: str
    argv: tuple
    environment: tuple
    batch_started_at: float
    execution_admitted: bool = field(default=False, init=False)
    runtime_closure_verified: bool = field(default=False, init=False)
    filesystem_verified: bool = field(default=False, init=False)

    def remaining_seconds(self):
        now = time.monotonic()
        require(type(now) in (int, float) and math.isfinite(now) and now >= self.batch_started_at)
        remaining = BATCH_SECONDS - (now - self.batch_started_at)
        if remaining <= 0:
            raise TimeoutError('compressed bootstrap batch deadline exhausted')
        return remaining


def prepare_invocation(proposal_bytes, proposal_sha256, *, batch_started_at):
    """Consume externally bound owned bytes; the outer timer already includes guards.

    Returned arguments are planning inputs, never evidence that files, ELF roots,
    transport, predecessor closure or resource capacity permit their execution.
    """
    require(type(proposal_bytes) is bytes)
    require(hashlib.sha256(proposal_bytes).hexdigest() == digest(proposal_sha256))
    try:
        proposal = json.loads(proposal_bytes)
        require(type(proposal) is dict and proposal['batch_cap_seconds'] == BATCH_SECONDS
                and type(proposal['batch_cap_seconds']) is int)
        require(proposal['execution_admitted'] is False and proposal['official_tests_opened'] is False)
        require(proposal['gpu_operations_enabled'] is False)
        require(type(proposal['host']) is str and proposal['host'] in ('xbabe1', 'xbabe2', 'xbabe3'))
        require(all(proposal[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')))
        require(type(batch_started_at) in (int, float))
        start = float(batch_started_at)
        require(math.isfinite(start) and start >= 0)
        require(proposal['python'] == '/usr/bin/python3.12')
        root = path(proposal['root'], scratch=True)
        entry = path(proposal['entry'], scratch=True)
        require(entry == root / 'source/entry.py')
        cache = path(proposal['empty_bytecode_cache'], scratch=True)
        library = path(proposal['native_library_directory'])
        require(not any(c in str(library) for c in (';', '$')))
        expected_round = digest(proposal['round_sha256'])
        job = digest(proposal['job_sha256'])
        attempt = proposal['attempt']
        require(type(attempt) is int and 1 <= attempt <= 9999)
        request = path(proposal['request'], scratch=True)
        require(request == root / 'attempts' / job / f'attempt-{attempt:04d}' / 'request.json')
        argv = ('/usr/bin/python3.12', '-I', '-S', '-B', '-X', f'pycache_prefix={cache}',
                str(entry), str(root), expected_round, str(request))
        env = (('CUDA_VISIBLE_DEVICES', ''), ('LC_ALL', 'C'), ('LD_LIBRARY_PATH', str(library)),
               ('MKL_NUM_THREADS', '1'), ('OMP_NUM_THREADS', '1'), ('OPENBLAS_NUM_THREADS', '1'),
               ('PATH', '/usr/bin:/bin'))
        plan = InvocationPlan(proposal['host'], argv, env, start)
        plan.remaining_seconds()
        return plan
    except (KeyError, TypeError, ValueError, OverflowError):
        raise ValueError('compressed bootstrap proposal rejected') from None
