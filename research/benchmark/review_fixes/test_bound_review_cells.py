"""Scalar-only privacy/TabSyn admission; no rows, models, or sealed registry."""
import copy
import json
import unittest
from unittest.mock import patch

from research.benchmark.review_fixes import bind_review_cells as bind


def privacy_cell():
    return {
        'method': 'DOPE', 'configuration': 'features12_steps2048', 'dataset': 'public-fixture',
        'sample_seed': 101, 'size': 1, 'status': 'ok', 'reason': None,
        'official_tests_opened': False, 'formal_dp': False, 'hipaa_deidentification': False,
        'synthetic_sha256': 'ab' * 32, 'rows': {'fit': 4, 'synthetic': 4, 'validation': 2},
        'c2st_catboost_auc': 0.5, 'c2st_rows_per_class': 2, 'c2st_status': 'ok',
        'dcr_fit_median': 0.1, 'dcr_validation_median': 0.2, 'nndr_fit_median': 0.9,
        'distance_mia_auc': 0.5, 'distance_mia_status': 'ok',
    }


def tabsyn_cell():
    return {
        'dataset': 'public-fixture', 'sample_seed': 101, 'size': 1, 'status': 'ok',
        'official_tests_opened': False, 'formal_dp': False, 'synthetic_sha256': 'ab' * 32,
        'null_loss': 0.2, 'rows': {'train': 4, 'synthetic': 4, 'validation': 2},
        'utility': {name: {'informative': True, 'low_signal_noninferior': None,
                          'retention': 0.5, 'trtr_loss': 0.1, 'tstr_loss': 0.15}
                    for name in ('catboost', 'linear', 'mlp')},
    }


class BoundCellTests(unittest.TestCase):
    def setUp(self):
        self.privacy_sources = {'research/benchmark/results/fixture.json': json.dumps({'cells': [{
            'method': 'DOPE', 'configuration': 'features12_steps2048', 'dataset': 'public-fixture',
            'size_multiplier': 1, 'sample_seed': 101, 'fit_seed': 11,
            'metric_receipt': {'sample_sha256': 'ab' * 32},
        }]}).encode()}
        self.sources = {
            bind.PREDECLARE: json.dumps({'tabsyn': {'config_sha256': 'cd' * 32}}).encode(),
            bind.TABSYN_FITS: json.dumps({'official_tests_opened': False, 'config_sha256': 'cd' * 32,
                'cells': [{'dataset_id': 'public-fixture', 'status': 'fit_ok', 'actual_fit_exit': 0,
                           'fit_complete': {'sha256': 'ef' * 32}, 'fit_seed': 11,
                           'config_sha256': 'cd' * 32}]}).encode(),
            bind.TABSYN_REFERENCE: json.dumps({'official_tests_opened': False, 'cells': [{
                'method': 'TabSyn', 'dataset': '02a45777900441ce', 'sample_seed': 101,
                'row_multiplier': 4, 'metrics': {'utility': {'auditors': {
                    'catboost': {'retention': 0.5}}}},
            }]}).encode(),
        }
        self.binding = {
            'parser_check': {'dataset': '02a45777900441ce', 'sample_seed': 101,
                'row_multiplier': 4, 'official_tests_opened': False, 'match': True,
                'expected_retention': 0.5, 'scored_retention': 0.5},
            'operations': [{'dataset': 'public-fixture', 'seed': 101, 'size': 1,
                            'status': 'ok', 'official_tests_opened': False, 'rows': 4}],
            'unavailable': [], 'rejected_operations': [],
        }


        sample_cells = []
        for size in (1, 4):
            for seed in (101, 211, 307):
                sample = {'dataset': 'public-fixture', 'size': size, 'sample_seed': seed,
                          'official_tests_opened': False, 'formal_dp': False,
                          'status': 'unavailable', 'reason': 'sample_csv_absent',
                          'bytes': None, 'synthetic_sha256': None}
                if (size, seed) == (1, 101):
                    sample.update(status='ok', reason=None, bytes=100, synthetic_sha256='ab' * 32)
                else:
                    self.binding['unavailable'].append(copy.deepcopy(sample))
                sample_cells.append(sample)
        import hashlib
        self.sources[bind.TABSYN_SAMPLES] = json.dumps({
            'official_tests_opened': False, 'formal_dp': False,
            'fit_registry_sha256': hashlib.sha256(self.sources[bind.TABSYN_FITS]).hexdigest(),
            'cells': sample_cells}).encode()

        self.sources[bind.TABSYN_OPERATIONS] = json.dumps({'parser_check': self.binding['parser_check'], 'operations': self.binding['operations']}).encode()

    def replay(self, kind, cells, binding=None):
        sources = self.privacy_sources if kind == 'privacy' else self.sources
        raw = ''.join(json.dumps(c) + '\n' for c in cells).encode()
        with patch('research.benchmark.review_fixes.privacy_fidelity.PUBLISHED',
                   [('fixture.json', 'DOPE', 'features12_steps2048')]):
            return bind.replay(kind, raw, sources, binding)

    def refuse(self, kind, cells, reason, binding=None):
        with patch(f'docs.whitepaper.scripts.review_{kind}.payload_from_cells') as reducer:
            with self.assertRaisesRegex(ValueError, reason):
                self.replay(kind, cells, binding)
            reducer.assert_not_called()

    def test_closed_privacy_and_tabsyn_admit_the_existing_reducers(self):
        for kind, cell, binding in [('privacy', privacy_cell(), None),
                                    ('tabsyn', tabsyn_cell(), self.binding)]:
            with self.subTest(kind=kind), patch(
                    f'docs.whitepaper.scripts.review_{kind}.payload_from_cells', return_value={'ok': True}) as reduce:
                self.assertEqual(self.replay(kind, [cell], binding), {'ok': True})
                reduce.assert_called_once()

    def test_official_or_formal_privacy_flags_refuse_reduction(self):
        for kind, make in [('privacy', privacy_cell), ('tabsyn', tabsyn_cell)]:
            for field in ('official_tests_opened', 'formal_dp'):
                for value in (True, 1, 0, None):
                    with self.subTest(kind=kind, field=field, value=value):
                        cell = make(); cell[field] = value
                        self.refuse(kind, [cell], 'flag refused', self.binding)

    def test_changed_privacy_sample_pin_refuses_reduction(self):
        cell = privacy_cell(); cell['synthetic_sha256'] = 'cd' * 32
        self.refuse('privacy', [cell], 'synthetic hash differs')

    def test_missing_planned_privacy_cell_refuses_reduction(self):
        cells = json.loads(next(iter(self.privacy_sources.values())))['cells']
        extra = copy.deepcopy(cells[0]); extra['sample_seed'] = 211; cells.append(extra)
        self.privacy_sources['research/benchmark/results/fixture.json'] = json.dumps({'cells': cells}).encode()
        self.refuse('privacy', [privacy_cell()], 'frozen published cohort')

    def test_duplicate_privacy_or_tabsyn_identity_refuses_reduction(self):
        for kind, make in [('privacy', privacy_cell), ('tabsyn', tabsyn_cell)]:
            with self.subTest(kind=kind):
                self.refuse(kind, [make(), make()], 'identity is duplicated', self.binding)

    def test_private_row_keys_refuse_reduction(self):
        for kind, make in [('privacy', privacy_cell), ('tabsyn', tabsyn_cell)]:
            with self.subTest(kind=kind):
                cell = make(); cell['source_rows'] = ['public toy value']
                self.refuse(kind, [cell], 'scalar', self.binding)

    def test_unavailable_privacy_is_retained_without_becoming_a_win(self):
        cell = {k: v for k, v in privacy_cell().items() if k in bind.PRIVACY_BASE}
        cell.update(status='unavailable', reason='sample_csv_unresolved')
        with patch('docs.whitepaper.scripts.review_privacy.payload_from_cells', return_value={'ok': True}) as reduce:
            self.replay('privacy', [cell])
            reduce.assert_called_once_with([cell], sources=self.privacy_sources)

    def test_tabsyn_parser_drift_and_open_flag_refuse_reduction(self):
        for field, value in [('scored_retention', 0.6), ('expected_retention', 0.6),
                             ('official_tests_opened', True), ('match', False)]:
            with self.subTest(field=field):
                binding = copy.deepcopy(self.binding); binding['parser_check'][field] = value
                self.refuse('tabsyn', [tabsyn_cell()], 'parser', binding)

    def test_tabsyn_frozen_fit_config_drift_refuses_reduction(self):
        fits = json.loads(self.sources[bind.TABSYN_FITS]); fits['config_sha256'] = 'ef' * 32
        self.sources[bind.TABSYN_FITS] = json.dumps(fits).encode()
        self.refuse('tabsyn', [tabsyn_cell()], 'fit configuration', self.binding)

    def test_tabsyn_without_successful_sampling_receipt_refuses_reduction(self):
        binding = copy.deepcopy(self.binding); binding['operations'] = []
        self.refuse('tabsyn', [tabsyn_cell()], 'frozen grid', binding)

    def test_tabsyn_open_sampling_flag_refuses_reduction(self):
        binding = copy.deepcopy(self.binding); binding['operations'][0]['official_tests_opened'] = True
        self.refuse('tabsyn', [tabsyn_cell()], 'sampling official test flag', binding)

    def test_tabsyn_size_or_auditor_drift_refuses_reduction(self):
        cell = tabsyn_cell(); cell['rows']['synthetic'] = 16
        self.refuse('tabsyn', [cell], 'synthetic size', self.binding)
        cell = tabsyn_cell(); del cell['utility']['mlp']
        self.refuse('tabsyn', [cell], 'auditor set', self.binding)


    def test_tabsyn_sample_digest_cannot_be_rewritten_in_the_metric(self):
        cell = tabsyn_cell(); cell['synthetic_sha256'] = 'ff' * 32
        self.refuse('tabsyn', [cell], 'independent CSV evidence', self.binding)

    def test_tabsyn_parser_identity_is_the_predeclared_reference(self):
        for field, value in [('dataset', 'other-fixture'), ('sample_seed', 211), ('row_multiplier', 1)]:
            binding = copy.deepcopy(self.binding); binding['parser_check'][field] = value
            self.refuse('tabsyn', [tabsyn_cell()], 'parser identity', binding)

    def test_tabsyn_dispositions_cannot_drop_duplicate_or_overlap_identities(self):
        mutations = [lambda b: b['unavailable'].pop(),
                     lambda b: b['unavailable'].append(copy.deepcopy(b['unavailable'][0])),
                     lambda b: b['unavailable'].append({**b['unavailable'][0], 'size': 1, 'sample_seed': 101})]
        for mutate in mutations:
            binding = copy.deepcopy(self.binding); mutate(binding)
            self.refuse('tabsyn', [tabsyn_cell()], 'grid|duplicated|overlaps', binding)

    def test_tabsyn_foreign_sampling_identity_never_supplies_a_cell(self):
        binding = copy.deepcopy(self.binding); binding['operations'][0]['dataset'] = 'worker'
        self.refuse('tabsyn', [tabsyn_cell()], 'outside the frozen grid', binding)

    def test_tabsyn_unavailable_flag_must_be_literal_false(self):
        for value in (True, 1, 0, None):
            binding = copy.deepcopy(self.binding); binding['unavailable'][0]['official_tests_opened'] = value
            self.refuse('tabsyn', [tabsyn_cell()], 'unavailable official', binding)

    def test_tabsyn_operation_rows_must_match_the_scored_sample(self):
        binding = copy.deepcopy(self.binding); binding['operations'][0]['rows'] = 3
        self.refuse('tabsyn', [tabsyn_cell()], 'sampling rows', binding)

    def test_privacy_scalar_domains_cannot_change(self):
        for field, value in [('distance_mia_auc', 1.1), ('c2st_catboost_auc', -0.1),
                             ('dcr_fit_median', -1), ('nndr_fit_median', 1.1)]:
            cell = privacy_cell(); cell[field] = value
            self.refuse('privacy', [cell], 'outside its domain')

    def test_tabsyn_published_utility_equation_and_types_are_enforced(self):
        for field, value in [('retention', 99), ('informative', False),
                             ('low_signal_noninferior', 1), ('trtr_loss', -1)]:
            cell = tabsyn_cell(); cell['utility']['catboost'][field] = value
            self.refuse('tabsyn', [cell], 'retention|signal|fields|domain', self.binding)

    def test_tabsyn_missing_metric_or_extra_metric_refuses_reduction(self):
        # A second independently hashed successful sample requires its metric too.
        binding = copy.deepcopy(self.binding)
        binding['operations'].append({**binding['operations'][0], 'seed': 211})
        binding['unavailable'] = [c for c in binding['unavailable'] if (c['size'], c['sample_seed']) != (1, 211)]
        samples = json.loads(self.sources[bind.TABSYN_SAMPLES])
        for sample in samples['cells']:
            if (sample['size'], sample['sample_seed']) == (1, 211):
                sample.update(status='ok', reason=None, bytes=100, synthetic_sha256='ab' * 32)
        self.sources[bind.TABSYN_SAMPLES] = json.dumps(samples).encode()
        self.refuse('tabsyn', [tabsyn_cell()], 'metric cohort differs', binding)


if __name__ == '__main__':
    unittest.main()
