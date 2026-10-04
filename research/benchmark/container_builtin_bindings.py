"""Prepare restricted builtin reference selections without initializing guards."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType, ModuleType

from . import container_import_hook as imports

BUILTIN_NAMES = frozenset({
    'Exception', 'FileNotFoundError', 'KeyError', 'OSError', 'OverflowError',
    'RuntimeError', 'TimeoutError', 'TypeError', 'UnicodeError', 'ValueError',
    '__build_class__', 'all', 'any', 'bool', 'bytes', 'dict', 'enumerate', 'float',
    'int', 'iter', 'len', 'list', 'max', 'min', 'next', 'range', 'set', 'sorted',
    'str', 'sum', 'tuple', 'type',
})


def require(condition):
    if not condition:
        raise ValueError('compressed guard builtin bindings rejected')


@dataclass(frozen=True)
class PreparedBuiltinBindings:
    import_hook: object
    builtin_references: object
    namespace_bindings: tuple
    helper_sha256: str

    def receipt(self):
        return {'format': 'dope-compressed-startup-builtin-bindings', 'version': 1,
                'manifest_sha256': self.import_hook.plan.manifest_sha256,
                'helper_sha256': self.helper_sha256,
                'builtin_references_selected': len(self.builtin_references),
                'namespace_bindings_planned': len(self.namespace_bindings),
                'builtins_installed': False, 'source_initializers_run': False,
                'modules_installed': False, 'actual_import_resolution_verified': False,
                'candidate_processes_started': 0, 'execution_admitted': False,
                'full_runtime_closure_certified': False, 'official_tests_opened': False,
                'mfs_v2': None, 'ptf_v1': None, 'release_safe': None, 'superiority': None}


def prepare_builtin_bindings(manifest_path, manifest_sha256, compiler_sha256,
                             binding_helper_sha256, declaration_helper_sha256,
                             import_helper_sha256, helper_sha256, stdlib_modules,
                             builtin_module, *, batch_started_at):
    """Caller already owns six seed helpers and all interpreter/module references.

    Metadata and self-file hashes do not prove loaded builtin identity. Select
    fixed references, replace the normal importer with the prepared hook and plan
    later bindings. No guard namespace, sys.modules or supplied module is changed.
    """
    binding = imports.declarations.binding
    compiler = binding.compiler
    sources = compiler.sources
    try:
        sources.remaining(batch_started_at)
        expected_helper = sources.digest(helper_sha256)
        helper = sources.unaliased(Path(__file__))
        require(hashlib.sha256(sources.read_regular(helper)[0]).hexdigest() == expected_helper)
        require(type(builtin_module) is ModuleType)
        values = builtin_module.__dict__
        require(type(values.get('__name__')) is str and values['__name__'] == 'builtins')
        require(BUILTIN_NAMES <= values.keys())
        selected = {name: values[name] for name in sorted(BUILTIN_NAMES)}
        try:
            hook = imports.prepare_import_hook(manifest_path, manifest_sha256, compiler_sha256,
                binding_helper_sha256, declaration_helper_sha256, import_helper_sha256,
                stdlib_modules, batch_started_at=batch_started_at)
        except TimeoutError:
            raise sources._DeadlineExhausted() from None
        selected['__import__'] = hook
        references = MappingProxyType(selected)
        planned = tuple((name, references) for name in hook.plan.initialization_order)
        require(len(planned) == len(sources.SOURCE_NAMES)
                and set(name for name, _ in planned) == set(hook.guard_modules))
        for path, expected in ((Path(binding.__file__), sources.digest(binding_helper_sha256)),
                               (Path(compiler.__file__), sources.digest(compiler_sha256)),
                               (Path(imports.declarations.__file__), sources.digest(declaration_helper_sha256)),
                               (Path(imports.__file__), sources.digest(import_helper_sha256)),
                               (helper, expected_helper)):
            sources.remaining(batch_started_at)
            require(hashlib.sha256(sources.read_regular(path)[0]).hexdigest() == expected)
        sources.remaining(batch_started_at)
        manifest = sources.scratch(manifest_path)
        require(hashlib.sha256(sources.read_regular(manifest)[0]).hexdigest() == hook.plan.manifest_sha256)
        sources.remaining(batch_started_at)
        return PreparedBuiltinBindings(hook, references, planned, expected_helper)
    except sources._DeadlineExhausted:
        raise TimeoutError('compressed bootstrap batch deadline exhausted') from None
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, OverflowError):
        raise ValueError('compressed guard builtin bindings rejected') from None
