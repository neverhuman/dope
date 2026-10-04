"""Bind owned guard initialization instructions without executing their code."""
import ast
from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
from types import CodeType, MappingProxyType, ModuleType

from . import container_private_registry as private


def require(condition):
    if not condition:
        raise ValueError('compressed guard initialization preparation rejected')


def code_signature(value):
    """Compare owned executable objects, including nested code and signed zeros."""
    if type(value) is CodeType:
        return tuple((name, code_signature(getattr(value, name))) for name in dir(value)
                     if name.startswith('co_') and not callable(getattr(value, name)))
    if type(value) is tuple:
        return ('tuple', tuple(code_signature(item) for item in value))
    if type(value) is frozenset:
        return ('frozenset', frozenset(code_signature(item) for item in value))
    if type(value) is float:
        return ('float', struct.pack('>d', value))
    if type(value) is complex:
        return ('complex', struct.pack('>dd', value.real, value.imag))
    require(type(value) in (type(None), bool, int, str, bytes, type(Ellipsis)))
    return (type(value).__name__, value)


@dataclass(frozen=True)
class GuardInitializationStep:
    name: str
    file: str
    source_sha256: str
    code: object
    module: object
    namespace: object
    dependencies: tuple


@dataclass(frozen=True)
class PreparedGuardInitialization:
    private_registry: object
    steps: tuple
    helper_sha256: str
    outer_seconds_remaining: float

    def receipt(self):
        return self.private_registry.receipt() | {
            'format': 'dope-compressed-guard-initialization-preparation',
            'helper_sha256': self.helper_sha256,
            'initialization_steps_bound': len(self.steps),
            'owned_code_recompiled_and_compared': True,
            'outer_seconds_remaining': self.outer_seconds_remaining,
            'initializer_execution_prepared': True,
            'live_registry_activation_required': True,
            'source_initializers_run': False, 'execution_admitted': False}


def prepare_guard_initialization(manifest_path, manifest_sha256, compiler_sha256,
                                 binding_helper_sha256, declaration_helper_sha256,
                                 import_helper_sha256, builtin_helper_sha256,
                                 namespace_helper_sha256, registry_helper_sha256,
                                 private_helper_sha256, helper_sha256,
                                 stdlib_modules, builtin_module, registry, *, batch_started_at):
    """Caller owns ten seed helpers, interpreter closure and supplied references.

    Instructions retain exact code, namespace and private-registry references.
    They are not an executor or an atomic lease. Standard-library decorators may
    consult the live module registry; the private overlay alone cannot admit them.
    """
    namespaces = private.registry_plan.namespaces
    selector = namespaces.selector
    imports = selector.imports
    binding = imports.declarations.binding
    compiler = binding.compiler
    sources = compiler.sources
    try:
        sources.remaining(batch_started_at)
        expected_helper = sources.digest(helper_sha256)
        helper = sources.unaliased(Path(__file__))
        require(hashlib.sha256(sources.read_regular(helper)[0]).hexdigest() == expected_helper)
        private.registry_plan.require_vacant(registry, binding.PACKAGE)
        registry_snapshot = dict(registry)
        helper_hashes = tuple((Path(module.__file__), sources.digest(expected)) for module, expected in (
            (sources, hashlib.sha256(sources.read_regular(Path(sources.__file__))[0]).hexdigest()),
            (compiler, compiler_sha256), (binding, binding_helper_sha256),
            (imports.declarations, declaration_helper_sha256), (imports, import_helper_sha256),
            (selector, builtin_helper_sha256), (namespaces, namespace_helper_sha256),
            (private.registry_plan, registry_helper_sha256), (private, private_helper_sha256)))
        try:
            prepared = private.prepare_private_registry(manifest_path, manifest_sha256,
                compiler_sha256, binding_helper_sha256, declaration_helper_sha256,
                import_helper_sha256, builtin_helper_sha256, namespace_helper_sha256,
                registry_helper_sha256, private_helper_sha256, stdlib_modules,
                builtin_module, registry, batch_started_at=batch_started_at)
            owned = sources.verify_startup_sources(manifest_path, manifest_sha256,
                batch_started_at=batch_started_at)
        except TimeoutError:
            raise sources._DeadlineExhausted() from None
        require(type(prepared) is private.PreparedPrivateRegistry)
        hook = prepared.bindings.namespaces.builtin_bindings.import_hook
        plan = hook.plan
        require(owned.manifest_sha256 == plan.manifest_sha256)
        descriptors = {item.name: item for item in plan.modules}
        require(len(descriptors) == len(sources.SOURCE_NAMES)
                and type(plan.initialization_order) is tuple
                and all(type(name) is str for name in plan.initialization_order)
                and len(plan.initialization_order) == len(descriptors)
                and set(plan.initialization_order) == set(descriptors))
        bodies = {binding.PACKAGE + '.' + name[:-3]: (name, body) for name, body in owned.sources}
        require(set(bodies) == set(descriptors))
        rows = dict(plan.imports)
        require(len(plan.imports) == len(descriptors) and set(rows) == set(descriptors))
        plan_snapshot = (plan.modules, plan.imports, plan.initialization_order)
        selected = prepared.bindings.namespaces.builtin_bindings.builtin_references
        steps, seen = [], set()
        for name in plan.initialization_order:
            sources.remaining(batch_started_at)
            descriptor = descriptors[name]
            filename, body = bodies[name]
            require(type(descriptor) is binding.ModuleBinding and descriptor.name == name
                    and descriptor.package == binding.PACKAGE
                    and type(descriptor.code) is CodeType
                    and hashlib.sha256(body).hexdigest() == descriptor.source_sha256)
            compiled = compile(body, '<owned startup guard: ' + filename + '>', 'exec',
                               flags=0, dont_inherit=True, optimize=0)
            require(code_signature(descriptor.code) == code_signature(compiled))
            declared = []
            for node in ast.walk(ast.parse(body, filename='<owned guard initialization>')):
                if isinstance(node, ast.Import):
                    declared.extend(('absolute', item.name, None, item.asname) for item in node.names)
                elif isinstance(node, ast.ImportFrom):
                    declared.extend(('relative', binding.PACKAGE + '.' + item.name, None, item.asname)
                                    if node.level == 1 else
                                    ('absolute', node.module, item.name, item.asname) for item in node.names)
            require(rows[name] == tuple(declared))
            dependencies = tuple(sorted({row[1] for row in rows[name] if row[0] == 'relative'}))
            require(set(dependencies) <= seen)
            module = hook.guard_modules[name]
            require(type(module) is ModuleType and prepared.registry_references[name] is module)
            steps.append(GuardInitializationStep(name, descriptor.file, descriptor.source_sha256,
                descriptor.code, module,
                MappingProxyType(module.__dict__), dependencies))
            seen.add(name)
        for path, expected in (*helper_hashes, (helper, expected_helper)):
            sources.remaining(batch_started_at)
            require(hashlib.sha256(sources.read_regular(path)[0]).hexdigest() == expected)
        try:
            require(sources.verify_startup_sources(manifest_path, manifest_sha256,
                batch_started_at=batch_started_at).sources == owned.sources)
        except TimeoutError:
            raise sources._DeadlineExhausted() from None
        sources.remaining(batch_started_at)
        manifest = sources.scratch(manifest_path)
        require(hashlib.sha256(sources.read_regular(manifest)[0]).hexdigest() == plan.manifest_sha256)
        require(hook.plan is plan and plan.modules is plan_snapshot[0]
                and plan.imports is plan_snapshot[1] and plan.initialization_order is plan_snapshot[2])
        for step in steps:
            descriptor = descriptors[step.name]
            values = step.module.__dict__
            require(descriptor.name == step.name and descriptor.package == binding.PACKAGE
                    and descriptor.file == step.file and descriptor.source_sha256 == step.source_sha256
                    and descriptor.code is step.code and hook.guard_modules[step.name] is step.module
                    and prepared.registry_references[step.name] is step.module
                    and hook.package.__dict__.get(step.name.rsplit('.', 1)[1]) is step.module
                    and set(values) == namespaces.EMPTY_KEYS | {'__builtins__'}
                    and values['__builtins__'] is selected)
            for key, expected in (('__name__', step.name), ('__package__', binding.PACKAGE),
                                  ('__file__', step.file)):
                require(type(values[key]) is str and values[key] == expected)
            require(all(values[key] is None for key in ('__doc__', '__loader__', '__spec__', '__cached__')))
        require(prepared.registry_references[binding.PACKAGE] is hook.package
                and type(hook.package.__dict__.get('__name__')) is str
                and hook.package.__dict__['__name__'] == binding.PACKAGE
                and '__path__' not in hook.package.__dict__)
        private.registry_plan.require_vacant(registry, binding.PACKAGE)
        require(set(registry) == set(registry_snapshot)
                and all(registry[name] is value for name, value in registry_snapshot.items()))
        remaining = sources.remaining(batch_started_at)
        return PreparedGuardInitialization(prepared, tuple(steps), expected_helper, remaining)
    except sources._DeadlineExhausted:
        raise TimeoutError('compressed bootstrap batch deadline exhausted') from None
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, OverflowError, SyntaxError):
        raise ValueError('compressed guard initialization preparation rejected') from None
