"""Bind prepared builtin references in private namespaces without running guards."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType, ModuleType

from . import container_builtin_bindings as selector

EMPTY_KEYS = frozenset({'__name__', '__doc__', '__package__', '__loader__',
                        '__spec__', '__file__', '__cached__'})


def require(condition):
    if not condition:
        raise ValueError('compressed guard namespace bindings rejected')


@dataclass(frozen=True)
class PreparedGuardNamespaces:
    builtin_bindings: object
    helper_sha256: str

    def receipt(self):
        hook = self.builtin_bindings.import_hook
        return {'format': 'dope-compressed-startup-namespace-bindings', 'version': 1,
                'manifest_sha256': hook.plan.manifest_sha256, 'helper_sha256': self.helper_sha256,
                'private_namespaces_bound': len(hook.guard_modules),
                'builtins_bound_in_private_namespaces': True,
                'source_initializers_run': False, 'modules_installed': False,
                'actual_import_resolution_verified': False, 'candidate_processes_started': 0,
                'execution_admitted': False, 'full_runtime_closure_certified': False,
                'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None,
                'release_safe': None, 'superiority': None}


def prepare_guard_namespaces(manifest_path, manifest_sha256, compiler_sha256,
                             binding_helper_sha256, declaration_helper_sha256,
                             import_helper_sha256, builtin_helper_sha256, helper_sha256,
                             stdlib_modules, builtin_module, *, batch_started_at):
    """Caller owns seven seed helpers and complete interpreter/module references.

    Bind only newly prepared private dictionaries. This does not exclude names
    from a live registry, install modules or execute owned code. No atomic lease
    or loaded-object attestation follows from metadata or self-file hashes.
    """
    imports = selector.imports
    binding = imports.declarations.binding
    compiler = binding.compiler
    sources = compiler.sources
    bound = []
    references = None
    accepted = False
    try:
        sources.remaining(batch_started_at)
        expected_helper = sources.digest(helper_sha256)
        helper = sources.unaliased(Path(__file__))
        require(hashlib.sha256(sources.read_regular(helper)[0]).hexdigest() == expected_helper)
        try:
            prepared = selector.prepare_builtin_bindings(manifest_path, manifest_sha256,
                compiler_sha256, binding_helper_sha256, declaration_helper_sha256,
                import_helper_sha256, builtin_helper_sha256, stdlib_modules, builtin_module,
                batch_started_at=batch_started_at)
        except TimeoutError:
            raise sources._DeadlineExhausted() from None
        require(type(prepared) is selector.PreparedBuiltinBindings)
        hook = prepared.import_hook
        references = prepared.builtin_references
        require(type(references) is MappingProxyType
                and set(references) == selector.BUILTIN_NAMES | {'__import__'}
                and references['__import__'] is hook)
        require(tuple(name for name, _ in prepared.namespace_bindings) == hook.plan.initialization_order)
        descriptors = {module.name: module for module in hook.plan.modules}
        for name, selected in prepared.namespace_bindings:
            sources.remaining(batch_started_at)
            require(selected is references)
            module = hook.guard_modules[name]
            require(type(module) is ModuleType and set(module.__dict__) == EMPTY_KEYS)
            namespace = module.__dict__
            descriptor = descriptors[name]
            for key, expected in (('__name__', name), ('__package__', descriptor.package),
                                  ('__file__', descriptor.file)):
                require(type(namespace[key]) is str and namespace[key] == expected)
            require(all(namespace[key] is None for key in ('__doc__', '__loader__', '__spec__', '__cached__')))
            require(hook.package.__dict__.get(name.rsplit('.', 1)[1]) is module)
            namespace['__builtins__'] = references
            bound.append(namespace)
        require(len(bound) == len(sources.SOURCE_NAMES))
        try:
            owned = sources.verify_startup_sources(manifest_path, manifest_sha256,
                batch_started_at=batch_started_at)
        except TimeoutError:
            raise sources._DeadlineExhausted() from None
        require(owned.manifest_sha256 == hook.plan.manifest_sha256
                and {filename[:-3]: hashlib.sha256(body).hexdigest() for filename, body in owned.sources}
                == {module.name.rsplit('.', 1)[1]: module.source_sha256 for module in hook.plan.modules})
        for path, expected in ((Path(binding.__file__), sources.digest(binding_helper_sha256)),
                               (Path(compiler.__file__), sources.digest(compiler_sha256)),
                               (Path(imports.declarations.__file__), sources.digest(declaration_helper_sha256)),
                               (Path(imports.__file__), sources.digest(import_helper_sha256)),
                               (Path(selector.__file__), sources.digest(builtin_helper_sha256)),
                               (helper, expected_helper)):
            sources.remaining(batch_started_at)
            require(hashlib.sha256(sources.read_regular(path)[0]).hexdigest() == expected)
        sources.remaining(batch_started_at)
        manifest = sources.scratch(manifest_path)
        require(hashlib.sha256(sources.read_regular(manifest)[0]).hexdigest() == hook.plan.manifest_sha256)
        require(all(namespace.get('__builtins__') is references for namespace in bound))
        for descriptor in hook.plan.modules:
            module = hook.guard_modules[descriptor.name]
            require(type(module) is ModuleType)
            namespace = module.__dict__
            require(set(namespace) == EMPTY_KEYS | {'__builtins__'})
            for key, expected in (('__name__', descriptor.name), ('__package__', descriptor.package),
                                  ('__file__', descriptor.file)):
                require(type(namespace[key]) is str and namespace[key] == expected)
            require(all(namespace[key] is None for key in ('__doc__', '__loader__', '__spec__', '__cached__')))
            require(namespace['__builtins__'] is references
                    and hook.package.__dict__.get(descriptor.name.rsplit('.', 1)[1]) is module)
        sources.remaining(batch_started_at)
        accepted = True
        return PreparedGuardNamespaces(prepared, expected_helper)
    except sources._DeadlineExhausted:
        raise TimeoutError('compressed bootstrap batch deadline exhausted') from None
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, OverflowError):
        raise ValueError('compressed guard namespace bindings rejected') from None
    finally:
        # Failed preparation removes only its own bindings from its new dictionaries.
        # The successful return retains them; no live registry is touched.
        if not accepted:
            for namespace in bound:
                if namespace.get('__builtins__') is references: namespace.pop('__builtins__')
