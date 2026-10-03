"""CPU-only native/scoring reuse core for complete SDV native-v2 fits.

This library has no launcher or admission interface. Its trusted caller must
verify immutable closed predecessor custody, fit/native receipt digests, full
runtime BEFORE adapter import, and fresh CPU/owner-registry admission. An
external monitor must charge verification/imports and enforce 600s for the
whole process. This core does not change the live frozen research rounds.
"""
from pathlib import Path
import hashlib
import json
import os
import time

from . import sdv_native_accounting as accounting
from .manifest import digest
from .reconcile_sdv_population import ROOT, ROUND_SHA
from .sdv_native_accounting import native_metric

BASE = ROOT.parent


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def verify_artifact(artifact, fit, worker):
    accounting.artifact(artifact, fit['artifact_inventory'], fit['artifact_bytes'],
                        worker, BASE, {})


def cpu_reuse(adapter, job, fit, artifact, parent, action,
              inherited_native_metric=None):
    """Call only in a freshly admitted CPU process with an external 600s monitor.

    native: resample seed101 and run the unchanged author efficacy.
    sample: require valid frozen native evidence; generate common n/4n and
    exact seed replay, independent of its shared utility outcomes.
    No fit is called and the model files must stay unchanged.
    """
    start = time.monotonic()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and os.environ.get('LOKY_MAX_CPU_COUNT') == '1',
            'CPU reuse requires CUDA hidden before dependency initialization')
    require(action in ('native', 'sample') and job['final'] is False
            and job['method'] in ('CTGAN', 'TVAE') and job['fit_seed'] == 11
            and job['dp_budget'] is None and job['track'] == 'common-numeric',
            'reuse left the frozen research identity')
    expected_adapter = job['native_objective']['implementation_sha256']
    require(sha(adapter.__file__) == expected_adapter
            and job['native_objective']['name'] == 'sdmetrics_mean_regression_r2'
            and job['native_objective']['direction'] == 'maximize',
            'native source or objective changed')
    artifact, parent = Path(artifact), Path(parent)
    worker = Path(job['worker']['path'])
    key = digest(job)
    accounting.claims(fit)
    require(fit['status'] == 'ok' and fit['job_sha256'] == key
            and fit['round_sha256'] == ROUND_SHA
            and fit['source_adapter_unchanged'] is True
            and fit['gpu_fit_required'] is True and fit['projection_bytes_included'] is True
            and artifact == ROOT / 'attempts' / key / 'attempt-0001' / 'artifact',
            'fitted artifact source, original job or predecessor round changed')
    accounting.safe(parent, BASE)
    require(not parent.resolve().is_relative_to(ROOT)
            and not parent.resolve().is_relative_to(worker.resolve())
            and type(job['worker']['train_rows']) is int and job['worker']['train_rows'] >= 2,
            'reuse output would modify predecessor or worker custody')
    accounting.worker_files(job['worker'], BASE, {})
    require(parent.is_dir() and not list(parent.iterdir()), 'reuse output already exists')
    verify_artifact(artifact, fit, job['worker'])
    contract = {'adapter_sha256': expected_adapter, 'native_metric_seed': 1729}
    if action == 'sample':
        require(inherited_native_metric is not None
                and inherited_native_metric['job_sha256'] == key,
                'common sampling requires native evidence for the same fitted job')
        accounting.claims(inherited_native_metric)
        native_metric(inherited_native_metric, job['worker'], job['native_objective'], contract)
    flags = {'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None,
             'counts_as_dope_win': False, 'admission_performed_by_core': False,
             'full_campaign_admitted': False, 'job_sha256': key,
             'shared_kpi_used_for_selection': False,
             'predecessor_round_sha256': ROUND_SHA,
             'component_source_sha256': sha(__file__),
             'new_fit_started': False, 'requires_gpu': False}
    completed = []
    try:
        if action == 'native':
            output = parent / 'native-sample.csv'
            sample = adapter.sample(artifact, job['worker']['train_rows'], 101, output)
            require(sample['sha256'] == sha(output)
                    and sample['rows'] == job['worker']['train_rows'], 'native sample identity changed')
            verify_artifact(artifact, fit, job['worker'])
            completed.append('native_sample')
            result = adapter.efficacy(output, worker / 'validation.csv', 1729)
            native_metric(result, job['worker'], job['native_objective'], contract)
            require(result['synthetic_sha256'] == sha(output), 'native efficacy sample changed')
            verify_artifact(artifact, fit, job['worker'])
            completed.append('native_metric')
            pending_native = result | flags
        else:
            result = []
            schedule = [(1, seed) for seed in (101, 211, 307)] + [(4, seed) for seed in (101, 211, 307)] + [(1, 101)]
            for index, (multiplier, seed) in enumerate(schedule):
                name = ('n' if multiplier == 1 else '4n') + f'-seed{seed}' if index < 6 else 'repeat'
                verify_artifact(artifact, fit, job['worker'])
                output = parent / (name + '.csv')
                sample = adapter.sample(artifact, job['worker']['train_rows'] * multiplier, seed, output)
                require(sample['sha256'] == sha(output) and sample['rows'] == job['worker']['train_rows'] * multiplier,
                        'common sample identity changed')
                verify_artifact(artifact, fit, job['worker'])
                if index == 0:
                    require(sha(output) == inherited_native_metric['synthetic_sha256'],
                            'common sample differs from inherited native sample')
                once(parent / (name + '.json'), sample | {'multiplier': multiplier, 'seed': seed} | flags)
                result.append(sample | {'multiplier': multiplier, 'seed': seed})
                completed.append(name)
            require(sha(parent / 'repeat.csv') == sha(parent / 'n-seed101.csv')
                    == inherited_native_metric['synthetic_sha256'], 'common/native seed replay changed')
            pending_samples = {'samples': result, 'sample_replay_exact': True} | flags
        verify_artifact(artifact, fit, job['worker'])
        accounting.worker_files(job['worker'], BASE, {})
        if action == 'native':
            once(parent / 'native.json', pending_native)
        else:
            once(parent / 'sample.json', pending_samples)
        receipt = {'action': action, 'status': 'ok', 'completed_steps': completed,
            'elapsed_core_seconds': time.monotonic()-start,
            'artifact_bytes': fit['artifact_bytes'],
            'projection_bytes_included': True, **flags}
    except Exception as error:
        once(parent / 'failure.json', {'action': action, 'status': 'failed',
            'error_type': type(error).__name__, 'elapsed_core_seconds': time.monotonic()-start,
            'completed_steps': completed, **flags})
        raise
    once(parent / 'receipt.json', receipt)
    return receipt
