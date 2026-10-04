"""Compile owned startup guard bytes without initializing or dispatching them."""
from dataclasses import dataclass
import hashlib
from pathlib import Path

from . import container_guard_sources as sources


def require(condition):
    if not condition:
        raise ValueError('compressed guard compilation rejected')


@dataclass(frozen=True)
class PreparedStartupCode:
    """Code objects remain in this trusted interpreter; no serialized cache input."""
    modules: tuple
    source_sha256: tuple
    manifest_sha256: str
    compiler_sha256: str
    outer_seconds_remaining: float

    def receipt(self):
        return {'format': 'dope-compressed-startup-code-preparation', 'version': 1,
                'manifest_sha256': self.manifest_sha256, 'compiler_sha256': self.compiler_sha256,
                'source_files_compiled': len(self.modules),
                'outer_seconds_remaining': self.outer_seconds_remaining,
                'source_initializers_run': False, 'candidate_processes_started': 0,
                'full_runtime_closure_certified': False, 'execution_admitted': False,
                'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None,
                'release_safe': None, 'superiority': None}


def prepare_startup_code(manifest_path, manifest_sha256, compiler_sha256, *, batch_started_at):
    """Caller already owns both seed helpers and complete interpreter/import closure.

    Compile verified owned bytes only. Initializing these modules, supplying their
    __file__ bindings and admitting a process remain separate bootstrap operations.
    """
    try:
        sources.remaining(batch_started_at)
        expected_compiler = sources.digest(compiler_sha256)
        compiler = sources.unaliased(Path(__file__))
        require(hashlib.sha256(sources.read_regular(compiler)[0]).hexdigest() == expected_compiler)
        try:
            owned = sources.verify_startup_sources(manifest_path, manifest_sha256,
                                                  batch_started_at=batch_started_at)
        except TimeoutError:
            # This trusted helper sanitizes ordinary I/O into ValueError; its
            # public TimeoutError is exclusively its genuine original deadline.
            raise sources._DeadlineExhausted() from None
        modules, hashes = [], []
        for name, body in owned.sources:
            sources.remaining(batch_started_at)
            code = compile(body, '<owned startup guard: ' + name + '>', 'exec',
                           flags=0, dont_inherit=True, optimize=0)
            modules.append((name, code))
            hashes.append((name, hashlib.sha256(body).hexdigest()))
            sources.remaining(batch_started_at)
        sources.remaining(batch_started_at)
        require(hashlib.sha256(sources.read_regular(compiler)[0]).hexdigest() == expected_compiler)
        sources.remaining(batch_started_at)
        manifest = sources.scratch(manifest_path)
        require(hashlib.sha256(sources.read_regular(manifest)[0]).hexdigest() == owned.manifest_sha256)
        remaining = sources.remaining(batch_started_at)
        return PreparedStartupCode(tuple(modules), tuple(hashes), owned.manifest_sha256,
                                   expected_compiler, remaining)
    except sources._DeadlineExhausted:
        raise TimeoutError('compressed bootstrap batch deadline exhausted') from None
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, OverflowError, SyntaxError):
        raise ValueError('compressed guard compilation rejected') from None
