"""Publish the frozen author TabDDPM probe without opening official test rows."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import tempfile
import tomllib

from .publish_s3_matched import artifact_inventory, require, schema
from .score import sha256

BASE = Path('/mnt/fast-scratch/dope-benchmark')
ROOT = BASE / 'tabddpm-contract-v3'
AUTHOR = BASE / 'method-source-audits/tabddpm-v1/source'
ENV = BASE / 'envs/tabddpm-py39'
INTERPRETER_LIB = BASE / 'envs/tabddpm-python39/cpython-3.9.25-linux-x86_64-gnu/lib'
CUSTODY = BASE / 'tabddpm-publication-custody-v1/receipt-source.lock.json'
CUSTODY_SHA256 = '6ecc1b3dcbec64baee27f0cc83ec1e24ed935cabbc74f95fa299d79e57b1478f'
EXECUTABLE_CUSTODY = BASE / 'tabddpm-publication-custody-v1/executable-closure-v3.lock.json'
EXECUTABLE_CUSTODY_SHA256 = '26dca41a25f3143eefa4d68aeda4e6a8a957d06338a127cf3edc72eddca196e1'
COMPLETION_SHA256 = '0e9bc146d708b545b3741b2ad15a76b6e3f17d7b7ce35964087fd46a3edad5f1'
NAME = 'tabddpm-author-contract'


def read(path):
    return json.loads(Path(path).read_text())


def verify_custody(path, expected_sha256, base):
    """Reject changes to original receipts and executable closure before reads."""
    require(not path.is_symlink() and sha256(path) == expected_sha256,
            'original TabDDPM custody lock changed')
    lock = read(path)
    require(lock['official_tests_opened'] is False and lock['new_fits_started'] == 0,
            'TabDDPM publication scope changed')
    files = {}
    interpreter = base / 'envs/tabddpm-py39/bin/python3.9'
    for row in lock['files']:
        item = Path(row['path'])
        require(item.is_absolute() and item.is_relative_to(base)
                and 'evaluator' not in item.parts and item.name != 'test.csv',
                'TabDDPM evidence outside sealed scope')
        resolved = item.resolve(strict=True)
        if item == interpreter:
            require(item.is_symlink() and str(item.readlink()) == row['symlink_target']
                    and str(resolved) == row['resolved_path'], 'TabDDPM interpreter alias changed')
        else:
            require(resolved == item and not item.is_symlink(), 'TabDDPM evidence contains link')
        require(str(item) not in files and item.stat().st_size == row['bytes']
                and sha256(item) == row['sha256'], 'original TabDDPM evidence changed')
        files[str(item)] = row['sha256']
    return files


def native_value(row):
    components = row['components']
    require(len(components) == 5 and {r['sample_seed'] for r in components} == set(range(5))
            and row['name'] == 'author_five_synthetic_seed_validation_catboost_r2_mean'
            and row['direction'] == 'maximize'
            and row['partition'] == 'official_training_derived_validation'
            and row['official_tests_opened'] is False
            and row['mfs_v2'] is None and row['ptf_v1'] is None
            and all(type(r['sample_seed']) is int and type(r['r2']) in (int, float)
                    and math.isfinite(r['r2']) for r in components)
            and type(row['value']) in (int, float) and math.isfinite(row['value'])
            and abs(row['value'] - sum(r['r2'] for r in components) / 5) <= 1e-14,
            'TabDDPM native objective or seed coverage changed')
    return row['value']


def verify_aliases(roots, aliases):
    seen_aliases = set()
    for root in roots:
        require(root.is_dir() and not root.is_symlink() and root.resolve() == root,
                'TabDDPM source root contains link')
        for path in root.rglob('*'):
            if path.is_symlink():
                row = aliases.get(str(path))
                require(row is not None and str(path.readlink()) == row['symlink_target']
                        and str(path.resolve(strict=True)) == row['resolved_path'],
                        'TabDDPM source inventory contains unpinned link')
                seen_aliases.add(str(path))
    require(seen_aliases == set(aliases), 'TabDDPM pinned alias inventory changed')


def verify_source_inventory(files, environment, author, aliases=None):
    verify_aliases((environment, author), aliases or {})
    def runtime_file(path):
        return path.suffix in ('.py', '.pyc', '.pyo', '.so') or '.so.' in path.name
    expected = {p for p in files if Path(p).is_relative_to(environment) and runtime_file(Path(p))}
    actual = {str(p) for p in environment.rglob('*') if p.is_file() and runtime_file(p)}
    require(actual == expected, 'TabDDPM runtime file inventory changed')
    expected = {p for p in files if Path(p).is_relative_to(author)}
    actual = {str(p) for p in author.rglob('*') if p.is_file()}
    require(actual == expected, 'TabDDPM author file inventory changed')


def verify_executable_closure(files):
    """Supplement the unchanged original custody with current cache evidence."""
    require(not EXECUTABLE_CUSTODY.is_symlink()
            and sha256(EXECUTABLE_CUSTODY) == EXECUTABLE_CUSTODY_SHA256,
            'TabDDPM executable custody lock changed')
    lock = read(EXECUTABLE_CUSTODY)
    require(lock['original_custody'] == {'path': str(CUSTODY), 'sha256': CUSTODY_SHA256}
            and lock['historical_fit_executable_closure_verified'] is False,
            'TabDDPM historical executable attestation changed')
    aliases = {r['path']: r for r in lock['aliases']}
    require(len(aliases) == len(lock['aliases']), 'duplicate TabDDPM source alias')
    augmented = dict(files)
    library = lock['interpreter_library']
    require(library['root'] == str(INTERPRETER_LIB), 'TabDDPM interpreter library root changed')
    library_aliases = {r['path']: r for r in library['aliases']}
    require(len(library_aliases) == len(library['aliases']), 'duplicate TabDDPM library alias')
    verify_aliases((INTERPRETER_LIB,), library_aliases)
    expected_library = {r['path'] for r in library['files']}
    require(len(expected_library) == len(library['files'])
            and expected_library == {str(p) for p in INTERPRETER_LIB.rglob('*') if p.is_file()},
            'TabDDPM interpreter library inventory changed')
    for row in library['files']:
        path = Path(row['path'])
        require(path.is_relative_to(INTERPRETER_LIB) and str(path) not in augmented
                and path.stat().st_size == row['bytes'] and sha256(path) == row['sha256'],
                'TabDDPM interpreter library changed')
        augmented[str(path)] = row['sha256']
    # lib/python39.zip is included if present; adding it changes the inventory.
    cache_paths = set()
    for row in lock['caches']:
        path = Path(row['path'])
        require(path.is_relative_to(ENV) or path.is_relative_to(AUTHOR)
                or path.is_relative_to(INTERPRETER_LIB),
                'TabDDPM cache outside source scope')
        require(path.resolve(strict=True) == path and not path.is_symlink()
                and str(path) not in cache_paths and path.stat().st_size == row['bytes']
                and sha256(path) == row['sha256']
                and augmented.get(str(path), row['sha256']) == row['sha256']
                and augmented.get(row['source_path']) == row['source_sha256'],
                'TabDDPM executable cache or source changed')
        cache_paths.add(str(path))
        augmented[str(path)] = row['sha256']
    require(cache_paths == {p for p in augmented if Path(p).suffix == '.pyc'
                            and any(Path(p).is_relative_to(root)
                                    for root in (ENV, AUTHOR, INTERPRETER_LIB))},
            'TabDDPM executable cache inventory changed')
    verify_source_inventory(augmented, ENV, AUTHOR, aliases)
    helper = Path(__file__).with_name('tabddpm_bytecode.py')
    cache_parent = Path('target/tabddpm-cache-verification')
    cache_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_parent) as empty_cache:
        # A fresh empty prefix plus -B prevents bootstrap from reading old caches
        # before the helper has established their source equivalence.
        completed = subprocess.run([str(ENV / 'bin/python3.9'), '-I', '-S', '-B', '-X',
                                    'pycache_prefix=' + str(Path(empty_cache).resolve()), str(helper)],
                                   input=json.dumps(lock['caches']), text=True,
                                   capture_output=True, timeout=30, check=False)
    require(completed.returncode == 0, 'TabDDPM executable cache verification failed')
    result = json.loads(completed.stdout)
    expected = [{k: r[k] for k in ('path', 'sha256', 'source_path', 'source_sha256')}
                | {'code_equivalent': True} for r in lock['caches']]
    require(result == {'status': 'ok', 'caches': expected},
            'TabDDPM executable cache differs from pinned source')
    return augmented, {'custody': {'path': str(EXECUTABLE_CUSTODY), 'sha256': EXECUTABLE_CUSTODY_SHA256},
                       'cache_verifier_source_sha256': sha256(helper),
                       'cache_file_count': len(expected),
                       'interpreter_library_file_count': len(expected_library),
                       'verification_startup': 'pinned interpreter libraries; -I -S -B and fresh empty pycache prefix',
                       'scope': 'author, environment and resolved interpreter-library inventories; system shared libraries and drivers not inventoried',
                       'current_declared_inventories_verified': True,
                       'current_executable_closure_verified': False,
                       'historical_fit_executable_closure_verified': False,
                       'aliases': lock['aliases']}


def physical_operations(paths, aliases):
    """Charge declared inherited operation aliases once, preserving failed fits."""
    alias_map = {r['alias_path']: r['physical_operation_path'] for r in aliases}
    require(len(alias_map) == len(aliases), 'duplicate TabDDPM operation alias')
    available = {str(p) for p in paths}
    require(set(alias_map) <= available and set(alias_map.values()) <= available
            and not set(alias_map) & set(alias_map.values()), 'TabDDPM alias target missing')
    result = {}
    for path in paths:
        name = str(path)
        canonical = alias_map.get(name, name)
        require(canonical == name or sha256(path) == sha256(Path(canonical)),
                'TabDDPM inherited operation differs')
        row = read(path)
        require(type(row['elapsed_seconds']) in (int, float)
                and math.isfinite(row['elapsed_seconds']) and row['elapsed_seconds'] >= 0,
                'invalid TabDDPM operation cost')
        if canonical in result:
            require(result[canonical] == row, 'TabDDPM physical operation differs')
        else:
            result[canonical] = row
    return result


def build():
    files = verify_custody(CUSTODY, CUSTODY_SHA256, BASE)
    original_file_count = len(files)
    files, executable_closure = verify_executable_closure(files)

    def frozen(path, expected=None):
        path = Path(path)
        require(str(path) in files and (expected is None or files[str(path)] == expected),
                'TabDDPM publication reference was not frozen')
        return {'path': str(path), 'sha256': files[str(path)]}

    completion_path = ROOT / 'completion-verification-v2.json'
    frozen(completion_path, COMPLETION_SHA256)
    completion = read(completion_path)
    require(completion['all_declared_contract_cells_complete'] is True
            and completion['official_tests_opened'] is False
            and completion['mfs_v2'] is None and completion['ptf_v1'] is None
            and completion['release_safe_l3'] is None and completion['counts_as_dope_win'] is False,
            'TabDDPM completion claim changed')
    round_path, runtime_path = ROOT / 'round.lock.json', ROOT / 'runtime.lock.json'
    frozen(round_path, completion['round_sha256'])
    frozen(runtime_path, completion['runtime_sha256'])
    round_lock, runtime = read(round_path), read(runtime_path)
    resume_path = ROOT / 'resume-v1.lock.json'
    frozen(resume_path, completion['resume_sha256'])
    require(round_lock['fit_seeds'] == [11, 23] and round_lock['sample_seeds'] == list(range(5))
            and round_lock['official_tests_opened'] is False, 'TabDDPM probe schedule changed')
    for name, expected in round_lock['source_files'].items():
        frozen(ROOT / 'source' / name, expected)
    for root, members in ((ENV, runtime['environment_files']), (AUTHOR, runtime['author_files'])):
        for name, expected in members.items():
            frozen(root / name, expected)
    worker = round_lock['worker']
    require(not (Path(worker['path']) / 'test.csv').exists(), 'TabDDPM worker contains tests')
    for name, expected in worker['files'].items():
        require(name in ('train.csv', 'validation.csv', 'projection.json', 'worker-manifest.json'),
                'TabDDPM worker partition not allowed')
        frozen(Path(worker['path']) / name, expected)
    config = round_lock['config']
    author_config = tomllib.loads((AUTHOR / 'exp/california/config.toml').read_text())
    require(author_config['train']['main'] == {k: config[k] for k in ('steps', 'lr', 'weight_decay', 'batch_size')}
            and author_config['model_params']['rtdl_params']['d_layers'] == config['d_layers']
            and author_config['diffusion_params']['num_timesteps'] == config['num_timesteps'],
            'TabDDPM declared author base settings differ')
    catboost_path = AUTHOR / 'tuned_models/catboost/california_cv.json'
    frozen(catboost_path, round_lock['native_catboost_config_sha256'])
    fits = []
    for seed in round_lock['fit_seeds']:
        path = ROOT / 'resume-v1' / f'fit-seed-{seed}' / 'receipt.json'
        reference = frozen(path)
        receipt = read(path)
        require(receipt['status'] == 'ok' and receipt['job']['fit_seed'] == seed
                and receipt['job']['worker'] == worker and receipt['job']['config'] == config
                and receipt['round_sha256'] == completion['round_sha256']
                and receipt['resume_lock_sha256'] == completion['resume_sha256']
                and receipt['official_tests_opened'] is False
                and receipt['mfs_v2'] is None and receipt['ptf_v1'] is None
                and receipt['artifact_only_sampler'] is True and receipt['sample_replay_exact'] is True,
                'TabDDPM completed fit identity changed')
        fit_reference = receipt['fit_reference']
        fit_path = Path(fit_reference['path'])
        frozen(fit_path, fit_reference['sha256'])
        require(read(fit_path) == receipt['fit'] and receipt['fit']['trained_gpu'] is True,
                'TabDDPM fit reference changed')
        inventory = artifact_inventory(fit_path.parent / 'artifact')
        charged = sum(r['bytes'] for r in inventory)
        require(inventory == receipt['fit']['artifact_inventory']
                and charged == receipt['fit']['artifact_bytes'], 'TabDDPM artifact accounting differs')
        fit_operation = read(fit_path.parent / 'fit.operation.json')
        fit_admission = read(fit_path.parent / 'fit.admission.json')
        require(fit_operation['status'] == 'ok' and fit_operation['elapsed_seconds'] <= 250
                and fit_admission['capacity']['admitted'] is True
                and fit_admission['global_scratch_admitted'] is True
                and fit_admission['host']['host'] in ('xbabe1', 'xbabe3')
                and fit_admission['host']['active_gpu_processes'] == []
                and receipt['fit']['peak_gpu_allocated_bytes'] <= 16 * 1024**3,
                'TabDDPM GPU fit admission or cap changed')
        for name, expected in receipt['evidence_files'].items():
            frozen(path.parent / name, expected)
        for sample_seed in range(5):
            sample = read(path.parent / f'sample-{sample_seed}.json')
            require(sample['status'] == 'ok' and sample['rows'] == worker['cohort']['train_rows']
                    and sample['artifact_only_sampling'] is True
                    and sample['sample_sha256'] == files[str(path.parent / f'sample-{sample_seed}.csv')],
                    'TabDDPM sample contract changed')
        require(files[str(path.parent / 'repeat.csv')] == files[str(path.parent / 'sample-0.csv')],
                'TabDDPM sample replay differs')
        native = receipt['native_kpi']
        native_value(native)
        require(read(path.parent / 'native-kpi.json') == native, 'TabDDPM native receipt differs')
        fits.append({'fit_seed': seed, 'receipt': reference, 'fit_reference': fit_reference,
                     'artifact_inventory': inventory, 'charged_artifact_bytes': charged,
                     'within_l3_bytes': charged <= 10240,
                     'gpu_fit_seconds': receipt['fit']['fit_seconds'],
                     'fit_entrypoint_seconds': fit_operation['elapsed_seconds'],
                     'host': fit_admission['host']['host'],
                     'gpu': fit_admission['host']['gpus'][0]['name'],
                     'peak_torch_allocated_bytes': receipt['fit']['peak_gpu_allocated_bytes'],
                     'sample_replay_exact': True, 'native_validation_kpi': native})
    paths = [Path(p) for p in files if p.endswith('.operation.json')]
    operations = physical_operations(paths, completion['sample_operation_aliases'])
    costs = {name: sum(r['elapsed_seconds'] for p, r in operations.items()
                      if Path(p).name == name) for name in ('fit.operation.json', 'native.operation.json')}
    sample_seconds = sum(r['elapsed_seconds'] for p, r in operations.items()
                         if Path(p).name.startswith(('sample-', 'repeat.')))
    require(len(operations) == 20 and sum(r['status'] == 'ok' for r in operations.values()) == 16
            and abs(costs['fit.operation.json'] - completion['all_fit_entrypoint_seconds_including_prior_failures']) < 1e-9
            and costs['fit.operation.json'] <= 1200
            and abs(sample_seconds - completion['sample_repeat_operation_seconds']) < 1e-9
            and abs(costs['native.operation.json'] - completion['native_operation_seconds']) < 1e-9,
            'TabDDPM cost reconciliation differs')
    attempts = []
    for p in sorted(Path(p) for p in files if Path(p).name == 'receipt.json'):
        row = read(p)
        operation = p.parent / 'fit.operation.json'
        physical = operations.get(str(operation))
        attempts.append({'receipt': frozen(p), 'fit_seed': row['job']['fit_seed'], 'status': row['status'],
                         'fit_operation': frozen(operation) if physical else None,
                         'fit_entrypoint_seconds': physical['elapsed_seconds'] if physical else 0,
                         'fit_entrypoint_status': physical['status'] if physical else 'none_or_reused',
                         'compute_charging': 'physical operation costs below; inherited successful fit counted once'})
    return {'format': 'dope-tabddpm-author-contract-publication', 'version': 1,
            'scope': completion['scope'], 'method': 'TabDDPM', 'track': 'common_numeric',
            'bibliography_entry': 'kotelnikov2023tabddpm', 'source_commit': round_lock['author_commit'],
            'source_license': 'MIT', 'source_archive_sha256': round_lock['author_archive_sha256'],
            'license_evidence': 'https://github.com/yandex-research/tab-ddpm/blob/' + round_lock['author_commit'] + '/LICENSE.md',
            'source_license_sha256': files[str(AUTHOR / 'LICENSE.md')],
            'publication_custody': {'path': str(CUSTODY), 'sha256': CUSTODY_SHA256,
                                    'verified_file_count': original_file_count},
            'executable_closure': executable_closure,
            'completion': frozen(completion_path), 'round': frozen(round_path), 'runtime': frozen(runtime_path),
            'execution': frozen(ROOT / 'execution.lock.json'), 'resume': frozen(resume_path),
            'publisher_source_sha256': sha256(Path(__file__)), 'runtime_versions': runtime['versions'],
            'runtime_python': runtime['python_version'], 'runtime_python_variation': runtime['python_variation'],
            'adapter': frozen(ROOT / 'source/adapter.py'), 'entry': frozen(ROOT / 'source/entry.py'),
            'native_implementation': frozen(ROOT / 'coordinator-retry-v1.py'),
            'author_objective_source': frozen(AUTHOR / 'scripts/tune_ddpm.py'),
            'author_native_catboost_config': frozen(catboost_path),
            'worker': worker, 'configuration': config, 'fit_seeds': [11, 23], 'sample_seeds': list(range(5)),
            'author_default_source': frozen(AUTHOR / 'exp/california/config.toml'),
            'adaptations': ['One S3 train-derived 72-row regression cohort, five projected features; not California benchmark rows.',
                            'Only fit rows enter preprocessing/training; official tests never staged.',
                            'Fit seeds11/23; author numeric base settings, feature width follows projected data.',
                            'Sample n rows with min(n,2000) batch, versus author52800 rows/batch8192.',
                            'Tensor-only weights and numeric quantile/discrete preprocessing arrays plus projection; no Python-object deserialization.'],
            'fits': fits, 'attempt_receipts': attempts,
            'costs': {'physical_operations_including_four_failed_fit_entrypoints': len(operations),
                      'successful_physical_operations': 16, 'physical_sample_repeat_operations': 12,
                      'inherited_sample_operation_aliases': completion['sample_operation_aliases'],
                      'fit_entrypoint_seconds_including_failures': costs['fit.operation.json'],
                      'failed_fit_entrypoint_seconds': completion['prior_failed_fit_entrypoint_seconds'],
                      'gpu_training_seconds': sum(r['gpu_fit_seconds'] for r in fits),
                      'sample_repeat_entrypoint_seconds': sample_seconds,
                      'native_evaluation_seconds': costs['native.operation.json'], 'energy_joules': None},
            'limits': {'neural_fit_seconds': 600, 'repair_fit_seconds': 250, 'total_fit_entrypoint_seconds': 1200,
                       'gpu_vram_bytes': 16 * 1024**3, 'original_pilot_deadline_unchanged': True},
            'limitations': ['Author-core contract probe, not two reported-experiment reproductions, native tuning or a matched common-outcome benchmark.',
                            'Historical runtime lock omitted executable bytecode and directory aliases: fit-time executable closure is unverified. Current caches match pinned source; this cannot establish their historical bytes.',
                            'Publication verifies the resolved interpreter library tree before invoking its helper with an empty cache prefix. System shared libraries and drivers are not inventoried; no complete current runtime-closure claim.',
                            'The original lib64-to-lib ABI alias and interpreter aliases are explicitly pinned; every additional source/runtime symlink is rejected.',
                            'Regression only; classification/categorical and generic final-runner integration remain pending.',
                            'Learned quantiles/discrete state are charged; no claim that preprocessing is free of source observations.',
                            'Both artifacts exceed L3; copy, leakage, attacks, real-vs-real and production profile gates are not complete.',
                            'Native values are not ranked across methods; no tuned configuration or winning fit seed selected.'],
            'official_tests_opened': False, 'new_fits_started': 0, 'native_tuning_complete': False,
            'final_campaign_admitted': False, 'production_certified': False, 'formal_dp_claim': False,
            'counts_as_dope_win': False, 'mfs_v2': None, 'ptf_v1': None, 'release_safe_l3': None}


def source_audit(report, report_sha256):
    return {'format': 'dope-tabddpm-author-source-audit', 'version': 1, 'method': 'TabDDPM',
            'status': 'pilot_locked', 'upstream_url': 'https://github.com/yandex-research/tab-ddpm',
            'upstream_commit': report['source_commit'], 'source_license': 'MIT',
            'license_evidence': report['license_evidence'], 'license_sha256': report['source_license_sha256'],
            'source_archive': str(AUTHOR.parent / 'source.tar.gz'),
            'source_archive_sha256': report['source_archive_sha256'],
            'contract_publication_sha256': report_sha256,
            'contract_publication': 'research/benchmark/results/' + NAME + '.json',
            'default_config': report['configuration'], 'default_source': report['author_default_source'],
            'adapter': report['adapter'], 'entry': report['entry'], 'runtime': report['runtime'],
            'executable_closure': report['executable_closure'],
            'runtime_python': report['runtime_python'], 'runtime_versions': report['runtime_versions'],
            'worker_startup': 'historical -S worker checked source/native-library hashes before dependency imports; executable caches and directory aliases were omitted. Current publication verifies cache/source equivalence, not historical fit-time closure.',
            'fit_command': str(ENV / 'bin/python3.9') + ' -S ' + report['entry']['path'] + ' fit VERIFIED_REQUEST',
            'sampling_command': str(ENV / 'bin/python3.9') + ' -S ' + report['entry']['path'] + ' sample VERIFIED_REQUEST',
            'native_objective': {'status': 'locked', 'name': 'author_five_synthetic_seed_validation_catboost_r2_mean',
                                 'direction': 'maximize', 'partition': 'validation', 'track': 'common_numeric',
                                 'implementation_sha256': report['native_implementation']['sha256'],
                                 'source': 'https://github.com/yandex-research/tab-ddpm/blob/' + report['source_commit'] + '/scripts/tune_ddpm.py',
                                 'source_sha256': report['author_objective_source']['sha256'],
                                 'metric_seed': 0, 'sample_seeds': list(range(5)),
                                 'tie_breaks': ['artifact_bytes_ascending', 'config_sha256_ascending'],
                                 'tie_break_origin': 'study-prespecified protocol; not an author tie-break claim',
                                 'shared_kpi_used_for_selection': False, 'native_values_cross_method_ranking': False},
            'author_tuning_search_space': {'lr_loguniform': [1e-5, .003], 'even_hidden_layers': [2, 4, 6, 8],
                                          'hidden_width_powers_of_two': [128, 256, 512, 1024],
                                          'hidden_width_assignment': 'separate first/middle/last choices; same middle width repeated',
                                          'batch_size': [256, 4096], 'steps': [5000, 20000, 30000],
                                          'weight_decay': [0.0], 'num_timesteps': [100, 1000],
                                          'native_synthetic_rows_train_size_power_of_two': [-2, -1, 0, 1]},
            'bounded_trial_grid_status': 'pending_before_tuning; author space recorded, no tuned configuration selected',
            'supported_task': 'regression', 'track': 'common_numeric', 'formal_dp_claim': False,
            'independent_core_implementation': False,
            'two_reported_experiment_reproduction': False,
            'contract_fit_seeds': report['fit_seeds'], 'fit_timeout_seconds': 600,
            'gpu_vram_cap_bytes': 16 * 1024**3, 'tuning_trials_per_method_dataset': 8,
            'tuning_wall_time_hours_per_method_dataset': 12,
            'adaptations': report['adaptations'], 'limitations': report['limitations'],
            'publication_custody': report['publication_custody'],
            'native_tuning_complete': False, 'final_campaign_admitted': False,
            'official_tests_opened': False, 'production_certified': False,
            'mfs_v2': None, 'ptf_v1': None, 'release_safe_l3': None, 'counts_as_dope_win': False}


def table(report):
    rows = ['# Author TabDDPM contract probe', '',
            'Two author-core GPU fits on the same 72-row S3 training-derived regression cohort (210_cloud). '
            'This is a fit/sample/native-objective contract probe. It is not a matched common-outcome benchmark or native tuning.', '',
            '| Fit seed | Native five-sample validation R² mean | Charged bytes | GPU training seconds | Fit entrypoint seconds | Host |',
            '|---:|---:|---:|---:|---:|---|']
    for fit in report['fits']:
        rows.append(f"| {fit['fit_seed']} | {fit['native_validation_kpi']['value']:.10f} | "
                    f"{fit['charged_artifact_bytes']} | {fit['gpu_fit_seconds']:.6f} | "
                    f"{fit['fit_entrypoint_seconds']:.6f} | {fit['host']} |")
    costs = report['costs']
    rows += ['', 'Both artifacts exceed the 10,240-byte L3 cap. Sample-seed 0 replay is byte-exact for both fits. '
             'All five native synthetic seeds are accounted for each fit; native values are never ranked across methods.', '',
             f"Physical operation costs include all four failed fit entrypoints: {costs['fit_entrypoint_seconds_including_failures']:.6f}s "
             f"fit total ({costs['failed_fit_entrypoint_seconds']:.6f}s failed preflight), "
             f"{costs['sample_repeat_entrypoint_seconds']:.6f}s sample/repeat and {costs['native_evaluation_seconds']:.6f}s native evaluation. "
             'Three inherited sample-operation aliases are verified and charged once. Energy is unavailable.', '',
             '## Scope and limitations', '']
    rows += ['- ' + item for item in report['adaptations'] + report['limitations']]
    rows += ['', 'Official tests remain sealed. MFS-v2, PTF-v1 and release-safe scores are null; no DOPE win, privacy or certification claim.', '',
             'Author source and objective: [TabDDPM](https://github.com/yandex-research/tab-ddpm), bibliography key `kotelnikov2023tabddpm`.', '']
    return '\n'.join(rows)


def main():
    report = build()
    root = Path(__file__).with_name('results')
    (root / (NAME + '.json')).write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    (root / (NAME + '.schema.json')).write_text(json.dumps(schema(report), sort_keys=True, indent=2) + '\n')
    (root / (NAME + '.md')).write_text(table(report))
    audit = source_audit(report, sha256(root / (NAME + '.json')))
    for suffix, document in (('.json', audit), ('.schema.json', schema(audit))):
        (root.parent / ('tabddpm-source.lock' + suffix)).write_text(json.dumps(document, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'fit_seeds': report['fit_seeds'], 'verified_files': report['publication_custody']['verified_file_count'],
                      'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None}))


if __name__ == '__main__':
    main()
