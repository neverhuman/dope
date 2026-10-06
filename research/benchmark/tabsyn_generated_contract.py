"""Generated numeric rows only; no benchmark data, GPU, or official tests."""
import ast
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import CodeType, FunctionType


class CheckedSourceLoader(importlib.machinery.SourceFileLoader):
    """Execute only verified bytes; never read a cached or replacement module."""
    def __init__(self, name, path, body):
        super().__init__(name, str(path))
        self.code = compile(body, str(path), 'exec', dont_inherit=True, optimize=0)

    def get_code(self, fullname):
        if fullname != self.name:
            raise ValueError('guard module identity changed')
        return self.code


def load_guard(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('guard entry digest changed')
    loader = CheckedSourceLoader('tabsyn_pinned_guard', path, raw)
    specification = importlib.util.spec_from_file_location(loader.name, path, loader=loader)
    module = importlib.util.module_from_spec(specification)
    loader.exec_module(module)
    return module


def load_original_loss(source, scope):
    tree = ast.parse(source.read_bytes())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_loss')
    code = compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec')
    functions = [c for c in code.co_consts if isinstance(c, CodeType)]
    if len(functions) != 1 or functions[0].co_name != 'compute_loss':
        raise ValueError('pinned numeric loss bytecode differs')
    return FunctionType(functions[0], scope, 'compute_loss')


def main():
    base, runtime_path, expected, code_path, code_expected, guard_expected = sys.argv[1:]
    base = Path(base)
    guard = base/'control/research/benchmark/tabsyn_runtime_guard.py'
    module = load_guard(guard, guard_expected)
    namespace = vars(module)
    lock = module.verify(runtime_path, expected, code_path, code_expected)
    custody = dict(runtime_path=runtime_path, runtime_sha256=expected,
                   code_path=code_path, code_sha256=code_expected)
    if set(os.sched_getaffinity(0)) != set(range(80, 96)) or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('fixture CPU admission differs')
    available = int(next(r.split()[1] for r in Path('/proc/meminfo').read_text().splitlines()
                         if r.startswith('MemAvailable:')))*1024
    if available < 8*2**30 or os.getloadavg()[0]+16 > 160:
        raise ValueError('fixture capacity admission failed')
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    import torch
    import xgboost
    from torch import nn
    from research.benchmark.tabsyn_adapter import fit, sample, artifact_inventory, numeric_loss, select_native, validate_config, check_deadline
    versions = dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
        sklearn=sklearn.__version__, torch=torch.__version__, xgboost=xgboost.__version__)
    if versions != dict(numpy='1.24.4', pandas='1.5.3', scipy='1.10.1', sklearn='1.2.2',
                        torch='2.0.1+cu117', xgboost='1.7.6'):
        raise ValueError('fixture dependency API differs')
    source = Path(lock['source'])/'tabsyn/vae/main.py'
    original = load_original_loss(source, {'torch': torch, 'nn': nn})
    torch.manual_seed(11)
    x = torch.randn(16, 4)
    recon = torch.randn(16, 4, requires_grad=True)
    mu = torch.randn(16, 4, requires_grad=True)
    logvar = torch.randn(16, 4, requires_grad=True)
    empty = torch.empty((16, 0), dtype=torch.long)
    try:
        original(x, empty, recon, [], mu, logvar)
    except UnboundLocalError:
        pass
    else:
        raise AssertionError('original numeric failure not reproduced')
    numeric = numeric_loss(x, recon, mu, logvar)
    assert numeric[1].item() == numeric[3].item() == 0
    cats = torch.stack([torch.randint(0, 2, (16,)), torch.randint(0, 3, (16,))], dim=1)
    crecon = [torch.randn(16, 2, requires_grad=True), torch.randn(16, 3, requires_grad=True)]
    mixed = original(x, cats, recon, crecon, mu, logvar)
    assert torch.equal(numeric[0], mixed[0]) and torch.equal(numeric[2], mixed[2])
    gradient_numeric = torch.autograd.grad(numeric[0]+numeric[2], [recon, mu, logvar], retain_graph=True)
    gradient_mixed = torch.autograd.grad(mixed[0]+mixed[2], [recon, mu, logvar])
    assert all(torch.equal(a, b) for a, b in zip(gradient_numeric, gradient_mixed))
    # Closed-inventory rejection uses opaque owned toy files, never mutates the
    # installed runtime or source root.
    toy = base/'opaque-inventory-control'
    toy.mkdir(exist_ok=False)
    provider = toy/'opaque.py'
    provider.write_bytes(b'opaque')
    inv = dict(root=str(toy), files={'opaque.py': dict(bytes=6,
        sha256=hashlib.sha256(b'opaque').hexdigest())}, aliases={}, directories=[])
    namespace['verify_inventory'](inv)
    provider.write_bytes(b'change')
    try:
        namespace['verify_inventory'](inv)
    except ValueError:
        pass
    else:
        raise AssertionError('runtime mutation accepted')
    provider.write_bytes(b'opaque')
    (toy/'extra.py').write_bytes(b'opaque')
    try:
        namespace['verify_inventory'](inv)
    except ValueError:
        pass
    else:
        raise AssertionError('additional provider accepted')
    proposal = json.loads((base/'tabsyn-native-grid.proposal.json').read_text())
    config = proposal['configs'][0].copy()
    config['vae_epochs'] = config['diffusion_epochs'] = 2
    raw = json.dumps({k: v for k, v in config.items() if k != 'config_sha256'},
                     separators=(',', ':'), sort_keys=True).encode()
    config['config_sha256'] = hashlib.sha256(raw).hexdigest()
    validate_config(config, fixture=True)
    rng = np.random.default_rng(11)
    train = rng.normal(size=(16, 4))
    validation = rng.normal(size=(8, 4))
    artifact = base/'generated-fixture-artifact'
    fit_receipt = fit(train, validation, b'{"purpose":"generated-opaque-fixture"}\n',
                      artifact, config, generated_fixture=True, timeout_seconds=120, **custody)
    inventory = artifact_inventory(artifact)
    records = []
    started = time.monotonic()
    seed_outputs = {}
    for multiplier in [1, 2, 4, 8]:
        for seed in [101, 211, 307]:
            result = sample(artifact, 16*multiplier, seed, inventory, **custody)
            assert result.shape == (16*multiplier, 4) and np.isfinite(result).all()
            records.append(dict(multiplier=multiplier, seed=seed, rows=result.shape[0],
                                digest=hashlib.sha256(result.tobytes()).hexdigest()))
            if multiplier == 1:
                seed_outputs[seed] = result
    repeated = sample(artifact, 16, 101, inventory, **custody)
    assert np.array_equal(repeated, seed_outputs[101])
    assert not np.array_equal(seed_outputs[101], seed_outputs[211])
    try:
        sample(artifact, 17, 101, inventory, **custody)
    except ValueError:
        pass
    else:
        raise AssertionError('noncanonical sample rows accepted')
    altered = json.loads(json.dumps(inventory))
    altered['artifact_bytes'] += 1
    try:
        sample(artifact, 16, 101, altered, **custody)
    except ValueError:
        pass
    else:
        raise AssertionError('artifact charge drift accepted')
    try:
        check_deadline(time.monotonic()-1)
    except TimeoutError:
        pass
    else:
        raise AssertionError('expired whole-fit deadline accepted')
    common = dict(status='ok', native_value=.5, elapsed_seconds=1.)
    rows = [dict(common, artifact_bytes=20, config_sha256='a'*64),
            dict(common, artifact_bytes=10, config_sha256='b'*64),
            dict(common, artifact_bytes=10, config_sha256='a'*64),
            dict(status='partial_failure', elapsed_seconds=600, native_value=99,
                 artifact_bytes=1, config_sha256='0'*64)]
    assert select_native(rows) == rows[2]
    assert select_native([rows[3]]) is None
    receipt = dict(format='dope-tabsyn-generated-fit-sample-contract', version=1,
        runtime_lock_sha256=expected, code_lock_sha256=code_expected, source_commit=proposal['author_commit'],
        source_audit_sha256=proposal['author_source_audit_sha256'], versions=versions,
        cpu_affinity=sorted(os.sched_getaffinity(0)), nice=os.getpriority(os.PRIO_PROCESS, 0),
        generated_rows_only=True, real_training_rows_decoded=False, gpu_used=False,
        original_numeric_failure_reproduced=True, numeric_mse_kl_and_gradients='exact',
        changed_or_additional_runtime_provider_rejected=True, artifact_charge_drift_rejected=True,
        whole_fit_deadline_rejected=True, partial_native_trial_ineligible=True,
        native_tie_breaks=['artifact_bytes_ascending', 'config_sha256_ascending'],
        fit=fit_receipt, artifact_inventory=inventory, samples=records,
        repeat_same_seed_exact=True, distinct_seeds_differ=True,
        sample_elapsed_seconds=time.monotonic()-started, generated_fixture_fits_started=1,
        new_real_generator_fits_started=0, official_tests_opened=False, execution_admitted=False,
        source_note='Numeric all-feature wrapper; source files unchanged. Zero worker DataLoader preserves isolated-worker contract.',
        author_rng_initialization_order_preserved=True,
        remaining_gates=['Full immutable worker fit/validation request admission',
            'Fresh root-coordinated GPU/RAM/VRAM admission and 600-second process watchdog',
            'Native author GPU evaluator measured generated contract',
            'Full author epoch attempt on training-derived rows only after launch order'],
        mfs_v2=None, ptf_v1=None, release_safe_l3=None, superiority=None)
    (base/'fixture-controls.receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(status='generated_contract_pass', fit_seconds=fit_receipt['whole_fit_elapsed_seconds'],
        sample_seconds=receipt['sample_elapsed_seconds'], samples=len(records),
        artifact_bytes=inventory['artifact_bytes'], new_real_fits=0, official_tests_opened=False)))


if __name__ == '__main__':
    main()
