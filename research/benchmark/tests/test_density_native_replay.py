"""Complete opaque native metadata fixture; no models, rows or runtimes execute."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark import density_native_replay as m


class NativeControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3] / 'target')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'native'
        self.refs, self.report = {}, {}
        self.rounds, self.selections, self.cells, self.trials = [], [], [], []
        source = {}
        for name in ('adapters.py', 'native_objective.py', 'tune_density.py', 'tune_copula.py'):
            source[str(self.root/'source'/name)] = self.put(self.root/'source'/name,
                b'raise AssertionError("must never initialize frozen source")')
        impl = source[str(self.root/'source'/'native_objective.py')]
        objective = dict(name='mean_log_density', direction='maximize', implementation_sha256=impl)
        methods = {name: dict(default_config=dict(bins=1 if name == 'GaussianCopula' else 2),
            tuning_search_space=dict(bins=list(range(1, 8)) if name == 'GaussianCopula' else [1, 2]),
            native_objective=objective, adapter_sha256=impl)
            for name in ('GaussianCopula', 'independent_marginals', 'Chow-Liu')}
        method_path = self.base/'original-methods.json'
        method_sha = self.put_json(method_path, dict(methods=methods))
        inventory = [dict(path='model.json', bytes=3, sha256='a'*64),
                     dict(path='projection.json', bytes=4, sha256='b'*64)]
        for round_name, method_key, names in (
            ('compact-native-all-v1', 'method_lock_sha256', ('independent_marginals', 'Chow-Liu')),
            ('copula-native-all-v1', 'methods_sha256', ('GaussianCopula',))):
            jobs = [dict(dataset=f'opaque-{i}', method=name, fit_seed=11, budget_seconds=43200,
                        train_sha256='c'*64, validation_sha256='d'*64,
                        configurations=[dict(bins=n) for n in methods[name]['tuning_search_space']['bins']])
                    for i in range(100) for name in names]
            lock = {method_key: method_sha, 'validation_only': True, 'jobs': jobs, 'adapter_sha256': impl}
            round_path = self.base/round_name/'round.lock.json'
            lock_sha = self.put_json(round_path, lock)
            # The production round pins are deliberately replaced only in this opaque fixture.
            self.rounds.append((round_path, lock_sha))
            for job in jobs:
                dataset, name = job['dataset'], job['method']
                rows = []
                for index, config in enumerate(job['configurations']):
                    path = self.base/round_name/dataset/name/f'trial-{index}'/'attempt.json'
                    metric = dict(format='dope-benchmark-native-validation-kpi', version=1,
                        partition='validation', method=name, objective='mean_log_density', direction='maximize',
                        implementation_sha256=impl, value=float(index-10), artifact_sha256='a'*64,
                        validation_sha256='d'*64)
                    metric_path = path.with_name('metric.json')
                    raw = dict(status='ok', config=config, native_kpi=metric['value'], artifact_bytes=7,
                        artifact_inventory=inventory, artifact_sha256='a'*64, wall_seconds=1.,
                        metric_receipt_path=str(metric_path), metric_receipt_sha256=self.put_json(metric_path, metric),
                        identity=dict(dataset=dataset, method=name, fit_seed=11, round_sha256=lock_sha,
                                      config=config, native_objective_sha256=impl, adapter_sha256=impl,
                                      train_sha256='c'*64, validation_sha256='d'*64))
                    if name == 'GaussianCopula':
                        del raw['identity']['native_objective_sha256']
                        del raw['identity']['adapter_sha256']
                    h = self.put_json(path, raw)
                    self.trials.append(dict(dataset=dataset, method=name, trial_index=index, status='ok',
                        native_validation_kpi=raw['native_kpi'], charged_artifact_bytes=7, seconds=1.,
                        attempt_receipt_path=str(path), attempt_receipt_sha256=h))
                    rows.append(raw)
                winner = len(rows)-1
                selected = dict(format='dope-benchmark-validation-selection', version=1,
                    dataset=dataset, method=name, partition='validation', test_opened=False,
                    round_sha256=lock_sha, objective=objective, selected_trial_index=winner,
                    selected_config=rows[winner]['config'],
                    total_wall_seconds=float(len(rows)), trials=rows)
                path = self.base/round_name/dataset/name/'selection.json'
                h = self.put_json(path, selected)
                self.selections.append((path, selected))
                default = 0 if name == 'GaussianCopula' else 1
                for kind, index in (('default', default), ('native_selected', winner)):
                    trial = next(t for t in self.trials if (t['dataset'], t['method'], t['trial_index'])
                                 == (dataset, name, index))
                    self.cells.append(dict(dataset=dataset, method=name, configuration=kind, fit_seed=11,
                        selected_trial_index=index, config=rows[index]['config'],
                        native_validation_kpi=rows[index]['native_kpi'], native_objective=objective,
                        artifact_bytes=7, artifact_inventory=inventory, model_sha256='a'*64,
                        selection_receipt_path=str(path), selection_receipt_sha256=h,
                        fit_attempt_receipt_path=trial['attempt_receipt_path'],
                        fit_attempt_receipt_sha256=trial['attempt_receipt_sha256'],
                        shared_kpi_used_for_selection=False, native_values_cross_ranked=False,
                        counts_as_dope_win=False, projection_bytes_included=True,
                        historical_full_runtime_closure_upgraded=False,
                        implementation_kind='author_library' if name == 'GaussianCopula' else 'study_reference'))
        self.report = dict(complete_native_matrix=True, datasets=100, native_cells=300, trial_count=1100,
            native_objective='mean_log_density', direction='maximize', official_tests_opened=False,
            shared_kpi_used_for_selection=False, counts_as_dope_win=False, native_kpis_never_ranked_across_methods=True,
            source_files=source, default_native_cells=self.cells, trials=self.trials,
            native_trial_operation_seconds=1100., new_generator_fits_started=0, new_tuning_trials=0,
            new_samples_started=0, historical_full_runtime_closure_upgraded=False,
            reported_experiment_reproduction_claim=False, mfs_v2=None, ptf_v1=None, release_safe=None)
        for module, key, value in ((m, 'BASE', self.base), (m, 'ROOT', self.root),
            (m, 'METHOD_PATH', method_path), (m, 'METHOD_SHA', method_sha), (m.owned, 'BASE', self.base)):
            p = patch.object(module, key, value)
            p.start()
            self.addCleanup(p.stop)
        self.patch_rounds = patch.object(m, 'ROUNDS', tuple((p.parent.name, h, k) for (p, h), k in
            zip(self.rounds, ('method_lock_sha256', 'methods_sha256'))))
        self.patch_rounds.start()
        self.addCleanup(self.patch_rounds.stop)
        self.save()

    def put(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        h = hashlib.sha256(data).hexdigest()
        self.refs[str(path)] = h
        return h

    def put_json(self, path, value):
        return self.put(path, json.dumps(value, sort_keys=True, allow_nan=False).encode())

    def save(self):
        report_sha = self.put_json(self.root/'reconciliation-v1.json', self.report)
        anchor_sha = self.put_json(self.root/'receipt-lock-v1.json',
                                   dict(reconciliation_sha256=report_sha, refs=dict(self.refs)))
        for key, value in (('NATIVE', anchor_sha), ('NATIVE_REPORT', report_sha)):
            p = patch.object(m.owned, key, value)
            p.start()
            self.addCleanup(p.stop)

    def call(self):
        return m.replay_native_metadata()

    def test_complete_native_grid_defaults_and_cost_remain_unadmitted(self):
        result = self.call()
        self.assertEqual((result['native_lineages'], result['native_trial_receipts']), (300, 1100))
        self.assertTrue(result['unchanged_native_choices'])
        self.assertFalse(result['publication_admitted'])
        self.assertTrue(result['runtime_worker_artifact_replay_required'])
        self.assertEqual(result['legacy_selections_without_validation_hash'], 300)
        self.assertEqual(result['legacy_trials_without_source_identity'], 700)

    def test_historical_optional_hashes_reject_conflicts(self):
        path, selection = self.selections[0]
        selection['validation_sha256'] = 'd'*64
        h = self.put_json(path, selection)
        for cell in self.cells:
            if cell['selection_receipt_path'] == str(path): cell['selection_receipt_sha256'] = h
        self.save()
        self.assertEqual(self.call()['legacy_selections_without_validation_hash'], 299)
        selection['validation_sha256'] = 'e'*64
        h = self.put_json(path, selection)
        for cell in self.cells:
            if cell['selection_receipt_path'] == str(path): cell['selection_receipt_sha256'] = h
        self.save()
        with self.assertRaises(ValueError): self.call()

    def test_conflicting_or_missing_trial_source_identities_rejected(self):
        summary = self.trials[0]
        path = Path(summary['attempt_receipt_path'])
        raw = json.loads(path.read_bytes())
        for field in ('native_objective_sha256', 'adapter_sha256'):
            old = raw['identity'].pop(field)
            summary['attempt_receipt_sha256'] = self.put_json(path, raw)
            self.save()
            with self.subTest(field=field), self.assertRaises(ValueError): self.call()
            raw['identity'][field] = old
        summary = next(t for t in self.trials if t['method'] == 'GaussianCopula')
        path = Path(summary['attempt_receipt_path'])
        raw = json.loads(path.read_bytes())
        raw['identity']['adapter_sha256'] = 'e'*64
        summary['attempt_receipt_sha256'] = self.put_json(path, raw)
        self.save()
        with self.assertRaises(ValueError): self.call()

    def test_wrong_default_winner_and_numeric_alias_rejected(self):
        cell = self.cells[0]
        for changes in (dict(selected_trial_index=0), dict(selected_trial_index=1.),
                        dict(config=dict(bins=2.)), dict(native_objective=dict(name='shared_utility'))):
            original = dict(cell)
            cell.update(changes)
            self.save()
            with self.subTest(changes=changes), self.assertRaises(ValueError): self.call()
            cell.clear(); cell.update(original)

    def test_trial_digest_drift_is_rejected_before_metric_use(self):
        p = Path(self.trials[0]['attempt_receipt_path'])
        p.write_bytes(b'{"changed":true}')
        with self.assertRaises(ValueError): self.call()

    def test_forged_selection_and_cell_cost_rejected(self):
        path, selection = self.selections[0]
        for key, value in (('selected_trial_index', 0), ('selected_trial_index', True),
                           ('selected_config', dict(bins=2.)),
                           ('total_wall_seconds', 43201.), ('total_wall_seconds', 1.)):
            old = selection[key]
            selection[key] = value
            h = self.put_json(path, selection)
            for cell in self.cells:
                if cell['selection_receipt_path'] == str(path): cell['selection_receipt_sha256'] = h
            self.save()
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): self.call()
            selection[key] = old

    def test_numeric_values_and_external_method_snapshot(self):
        for value in (True, float('nan'), float('inf'), 10**400):
            self.assertFalse(m.number(value))
        m.METHOD_PATH.write_bytes(b'{"methods":{}}')
        with self.assertRaises(ValueError): self.call()

    def test_native_summary_numeric_aliases_rejected(self):
        summary = self.trials[0]
        for key, value in (('seconds', 1), ('charged_artifact_bytes', 7.),
                           ('native_validation_kpi', -10)):
            old = summary[key]
            summary[key] = value
            self.save()
            with self.subTest(key=key), self.assertRaises(ValueError): self.call()
            summary[key] = old

    def test_coverage_claim_and_cost_rewrites_rejected(self):
        for key, value in (('datasets', 100.), ('mfs_v2', .99), ('new_generator_fits_started', False),
                           ('native_trial_operation_seconds', 1099.), ('shared_kpi_used_for_selection', True)):
            old = self.report[key]
            self.report[key] = value
            self.save()
            with self.subTest(key=key), self.assertRaises(ValueError): self.call()
            self.report[key] = old

    def test_source_alias_and_added_empty_directory_rejected(self):
        directory = self.root/'source'/'added'
        directory.mkdir()
        with self.assertRaises(ValueError): self.call()
        directory.rmdir()
        (self.root/'source'/'alias').symlink_to(self.root/'source', target_is_directory=True)
        with self.assertRaises(ValueError): self.call()


if __name__ == '__main__':
    unittest.main(verbosity=2)
