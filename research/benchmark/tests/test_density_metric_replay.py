"""Opaque saved-metric controls; no dataset, learner or live round is opened."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark import density_metric_replay as m
from research.benchmark.tests.test_density_publication_inputs import metadata


def metric(size=1):
    return dict(format='dope-benchmark-validation-pilot-metrics', version=1,
        implementation_sha256=m.METRIC_SHA256, dependencies={'opaque': '1'},
        task='regression', gate_profile_complete=False, mfs_v2=None,
        rows=dict(train=10, validation=5, synthetic=10*size), marginal_ks_mean=.2,
        pair_correlation_fidelity=.8, c2st_auc=.7, metric_seconds=.1, null_loss=10.,
        copy_counts=dict(exact=0, near=1), real_vs_real_control_counts=dict(exact=0, near=0),
        utility={a: dict(trtr_loss=2., tstr_loss=4., informative=True, retention=.75,
                        low_signal_noninferior=None) for a in ('catboost', 'linear', 'mlp')})


class ArithmeticControls(unittest.TestCase):
    def setUp(self):
        self.job = dict(worker=dict(train_rows=10))
        self.runtime = dict(metric_sha256=m.METRIC_SHA256, expected_versions={'opaque': '1'})

    def call(self, row):
        return m.verify_metric(row, self.job, 1, self.runtime)

    def test_unclipped_negative_and_above_one_retention(self):
        for tstr, expected in ((18., -1.), (0., 1.25)):
            row = metric()
            row['utility']['linear'].update(tstr_loss=tstr, retention=expected)
            self.assertEqual(self.call(row)['utility']['linear']['retention'], expected)

    def test_frozen_low_signal_and_explicit_auditor_failure(self):
        row = metric()
        row['utility']['linear'].update(trtr_loss=9.95, tstr_loss=10., informative=False,
                                       retention=None, low_signal_noninferior=True)
        row['utility']['mlp'] = dict(status='failed', error_type='RuntimeError')
        self.call(row)
        row['utility']['linear']['low_signal_noninferior'] = 1
        with self.assertRaises(ValueError): self.call(row)

    def test_changed_arithmetic_signal_and_zero_gap_rejected(self):
        for changes in (dict(retention=1.), dict(retention=True), dict(informative=1),
                        dict(informative=False), dict(low_signal_noninferior=False),
                        dict(trtr_loss=-1), dict(tstr_loss=float('nan'))):
            row = metric()
            row['utility']['catboost'].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError): self.call(row)
        row = metric()
        row['null_loss'] = 0.
        row['utility']['catboost'].update(trtr_loss=0., tstr_loss=0.)
        with self.assertRaises(ValueError): self.call(row)

    def test_auditors_runtime_row_counts_controls_and_claims_rejected(self):
        mutations = [lambda r: r.update(version=1.), lambda r: r.update(dependencies={'opaque': '2'}),
            lambda r: r.update(implementation_sha256='b'*64), lambda r: r['rows'].update(train=10.),
            lambda r: r['rows'].update(synthetic=40), lambda r: r['rows'].update(validation=True),
            lambda r: r['copy_counts'].update(near=11), lambda r: r['copy_counts'].update(exact=2),
            lambda r: r['real_vs_real_control_counts'].update(near=6),
            lambda r: r['real_vs_real_control_counts'].update(exact=False),
            lambda r: r['utility'].pop('mlp'), lambda r: r.update(c2st_auc=float('inf')),
            lambda r: r.update(marginal_ks_mean=True), lambda r: r.update(metric_seconds=-.1),
            lambda r: r.update(null_loss=10**400), lambda r: r.update(gate_profile_complete=0),
            lambda r: r.update(mfs_v2=.99), lambda r: r.update(counts_as_dope_win=True)]
        for index, mutate in enumerate(mutations):
            row = metric()
            mutate(row)
            with self.subTest(index=index), self.assertRaises(ValueError): self.call(row)


class SavedBatchControls(unittest.TestCase):
    def setUp(self):
        target = Path(__file__).resolve().parents[3] / 'target'
        self.temp = tempfile.TemporaryDirectory(dir=target)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'round'
        for key, value in (('BASE', self.base), ('ROOT', self.root)):
            p = patch.object(m.inputs, key, value)
            p.start()
            self.addCleanup(p.stop)
        self.job = dict(worker=dict(train_rows=10), fit_seed=11, final=False,
                        metric_replay_required=True)
        self.key = m.inputs.digest_identity(self.job)
        self.out = self.root / 'attempts' / self.key / 'attempt-0001'
        self.out.mkdir(parents=True)
        self.flags = dict(official_tests_opened=False, native_selection_changed=False,
            global_family_selected=False, new_generator_fits_started=0,
            mfs_v2=None, ptf_v1=None, release_safe=None, superiority=None)
        self.runtime = dict(metric_sha256=m.METRIC_SHA256, expected_versions={'opaque': '1'})
        self.lock = dict(metric_source_sha256=m.METRIC_SHA256)
        self.refs, self.evidence = {}, {}
        self.put('request.json', dict(job=self.job, round_sha256=m.inputs.ROUND))
        samples = []
        for size in (1, 4):
            for seed in (101, 211, 307):
                name = f'n{size}-seed{seed}'
                sample_sha = self.put(name+'.csv', b'opaque; not parsed or executed')
                metric_sha = self.put(name+'.metric.json', metric(size))
                if (size, seed) == (1, 101): self.put(name+'.repeat.csv', b'opaque; not parsed or executed')
                if (size, seed) == (4, 101): self.put(name+'.metric-replay.json', metric(size) | dict(metric_seconds=.2))
                samples.append(dict(size_multiplier=size, sample_seed=seed, rows=10*size,
                    sample_file=name+'.csv', metric_file=name+'.metric.json', sample_sha256=sample_sha,
                    metric_sha256=metric_sha, sample_replay='exact' if (size, seed) == (1, 101) else 'not_repeated',
                    metric_replay='exact' if (size, seed) == (4, 101) else 'not_repeated'))
        self.batch = self.flags | dict(job_sha256=self.key, metric_source_sha256=m.METRIC_SHA256,
                                     sample_replays_exact=True, gate_profile_complete=False, samples=samples)
        self.save()

    def put(self, name, value):
        data = value if type(value) is bytes else json.dumps(value).encode()
        path = self.out / name
        path.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        self.refs[str(path)] = digest
        if name != 'receipt.json': self.evidence[name] = digest
        return digest

    def save(self, status='ok'):
        self.put('batch.json', self.batch)
        self.put('receipt.json', self.flags | dict(job=self.job, round_sha256=m.inputs.ROUND,
                 status=status, evidence_files=self.evidence))

    def call(self):
        return m.replay_batch_metrics(self.job, self.lock, self.runtime, self.refs)

    def test_six_owned_metrics_and_repetitions_without_dependency_initializer(self):
        import builtins
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('numpy', 'torch', 'sklearn', 'catboost'):
                self.fail('dependency initializer invoked')
            return original(name, *args, **kwargs)
        with patch.object(builtins, '__import__', guarded):
            result = self.call()
        self.assertEqual(len(result['samples']), 6)
        self.assertEqual(result['samples'][0]['metric']['utility']['linear']['retention'], .75)

    def test_unstarted_timeout_and_infra_keep_raw_outcome_and_no_metrics(self):
        for status in ('deadline_unstarted', 'timeout', 'prelaunch_or_metric_failure'):
            self.save(status)
            result = self.call()
            self.assertEqual((result['status'], result['samples']), (status, []))

    def test_schedule_missing_duplicate_numeric_alias_rejected(self):
        original = copy.deepcopy(self.batch)
        for kind in ('missing', 'duplicate', 'alias'):
            self.batch = copy.deepcopy(original)
            if kind == 'missing': self.batch['samples'].pop()
            elif kind == 'duplicate': self.batch['samples'][-1] = dict(self.batch['samples'][0])
            else: self.batch['samples'][0]['sample_seed'] = 101.
            self.save()
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.call()

    def test_frozen_metric_rewrite_or_unbound_digest_rejected(self):
        path = self.out / 'n1-seed101.metric.json'
        path.write_bytes(json.dumps(metric() | dict(null_loss=20.)).encode())
        with self.assertRaises(ValueError): self.call()
        self.put(path.name, metric())
        self.refs[str(path)] = 'b'*64
        with self.assertRaises(ValueError): self.call()

    def test_required_sample_and_metric_repetition_missing_or_changed(self):
        for name in ('n1-seed101.repeat.csv', 'n4-seed101.metric-replay.json'):
            old = self.evidence.pop(name)
            self.save()
            with self.subTest(name=name), self.assertRaises(ValueError): self.call()
            self.evidence[name] = old
        self.put('n1-seed101.repeat.csv', b'changed')
        self.save()
        with self.assertRaises(ValueError): self.call()
        self.put('n1-seed101.repeat.csv', b'opaque; not parsed or executed')
        replay = metric(4)
        replay['utility']['linear'].update(tstr_loss=2., retention=1.)
        self.put('n4-seed101.metric-replay.json', replay)
        self.save()
        with self.assertRaises(ValueError): self.call()

    def test_replay_json_numeric_alias_and_changed_validation_declaration_rejected(self):
        self.put('n4-seed101.metric-replay.json', metric(4) | dict(null_loss=10))
        self.save()
        with self.assertRaises(ValueError): self.call()
        self.put('n4-seed101.metric-replay.json', metric(4))
        changed = metric(4)
        changed['rows']['validation'] = 6
        name = 'n4-seed307.metric.json'
        self.batch['samples'][-1]['metric_sha256'] = self.put(name, changed)
        self.save()
        with self.assertRaises(ValueError): self.call()

    def test_request_numeric_alias_and_batch_claim_rejected(self):
        self.put('request.json', dict(job=self.job | dict(fit_seed=11.), round_sha256=m.inputs.ROUND))
        self.save()
        with self.assertRaises(ValueError): self.call()
        self.put('request.json', dict(job=self.job, round_sha256=m.inputs.ROUND))
        self.batch['counts_as_dope_win'] = True
        self.save()
        with self.assertRaises(ValueError): self.call()

    def test_closed_entry_requires_anchors_and_complete_custody_before_metric_reads(self):
        with patch.object(m, 'replay_batch_metrics') as replay:
            with self.assertRaises(ValueError): m.replay_closed_metrics('A'*64, 'b'*64)
            replay.assert_not_called()
            with self.assertRaises(FileNotFoundError): m.replay_closed_metrics('a'*64, 'b'*64)
            replay.assert_not_called()


class LogicalBindingControls(unittest.TestCase):
    def setUp(self):
        lock, self.report, _, _ = metadata()
        samples = lambda: [dict(evidence=dict(size_multiplier=z, sample_seed=s, metric_sha256='a'*64),
                               metric=metric(z)) for z in (1, 4) for s in (101, 211, 307)]
        self.batches = [dict(physical_job_sha256=m.inputs.digest_identity(j), status='ok', samples=samples())
                        for j in lock['jobs']]
        self.by_key = {b['physical_job_sha256']: b for b in self.batches}
        for cell in self.report['cells']:
            b = self.by_key[cell['physical_job_sha256']]
            cell['sample_evidence'] = next(s['evidence'] for s in b['samples']
                if (s['evidence']['size_multiplier'], s['evidence']['sample_seed'])
                == (cell['size_multiplier'], cell['sample_seed']))

    def call(self):
        return m.bind_logical_metrics(self.report, self.batches)

    def test_complete_logical_matrix_shares_owned_physical_metrics(self):
        result = self.call()
        self.assertEqual(len(result), 3600)
        self.assertEqual(sum(r['metric'] is not None for r in result), 3600)

    def test_partial_closure_rewritten_pointer_and_raw_status_rejected(self):
        original = copy.deepcopy(self.report['cells'][0])
        self.report['cells'][0]['sample_evidence'] = original['sample_evidence'] | dict(sample_seed=101.)
        with self.assertRaises(ValueError): self.call()
        self.report['cells'][0] = original
        self.report['cells'][0]['status'] = 'failed'
        with self.assertRaises(ValueError): self.call()
        self.report['cells'][0] = original
        self.batches.pop()
        with self.assertRaises(ValueError): self.call()

    def test_failed_physical_batch_never_acquires_logical_metrics(self):
        key = self.batches[0]['physical_job_sha256']
        self.batches[0].update(status='deadline_unstarted', samples=[])
        count = 0
        for cell in self.report['cells']:
            if cell['physical_job_sha256'] == key:
                cell.update(status='deadline_unstarted', sample_evidence=None)
                count += 1
        self.assertEqual(sum(r['metric'] is None for r in self.call()), count)


if __name__ == '__main__':
    unittest.main(verbosity=2)
