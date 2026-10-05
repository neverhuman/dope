"""Opaque controls for complete coverage and dataset-paired arithmetic."""
import copy
import unittest
from research.benchmark import density_matched_aggregate as m


def fixture():
    return [dict(dataset=f'opaque-{d:03}', method=method, configuration=config,
                 fit_seed=11, sample_seed=seed, size_multiplier=size, status='ok',
                 charged_artifact_bytes=100, artifact_within_l3_cap=True,
                 utility={a: dict(informative=True, retention=float(d + 1)) for a in m.AUDITORS},
                 mfs_v2=None, ptf_v1=None, release_safe_l3=None,
                 paired_superiority=None, counts_as_dope_win=False)
            for d in range(100) for method, config in m.CONFIGS
            for seed in m.SEEDS for size in m.SIZES]


class Controls(unittest.TestCase):
    def setUp(self):
        self.cells = fixture()

    def test_complete_panel_preserves_all_configs_and_null_claims(self):
        result = m.aggregate(self.cells)
        self.assertEqual(len(result['summary']), 2000)
        self.assertEqual(len(result['configuration_panels']), 20)
        self.assertEqual(len(result['paired_descriptive']), 144)
        self.assertFalse(result['global_family_selected'])
        self.assertIsNone(result['paired_superiority'])

    def test_missing_and_duplicate_and_unknown_config_rejected(self):
        for mutation in ('missing', 'duplicate', 'unknown'):
            cells = copy.deepcopy(self.cells)
            if mutation == 'missing': cells.pop()
            elif mutation == 'duplicate': cells[-1] = copy.deepcopy(cells[0])
            else: cells[0]['configuration'] = 'undeclared'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                m.aggregate(cells)

    def test_numeric_identity_is_not_python_equality(self):
        for field, value in [('fit_seed', 11.0), ('sample_seed', 101.0),
                             ('size_multiplier', True)]:
            cells = copy.deepcopy(self.cells)
            cells[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                m.aggregate(cells)

    def test_pairing_uses_median_differences_on_common_complete_groups(self):
        x, y = (0., 100., 101.), (1., 2., 200.)
        for c in self.cells:
            d = int(c['dataset'].split('-')[-1])
            for u in c['utility'].values():
                u.update(informative=d < 3, retention=(x[d] if c['method'] == 'DOPE'
                                                       else y[d]) if d < 3 else None)
        pair = m.aggregate(self.cells)['paired_descriptive'][0]
        self.assertEqual(pair['complete_paired_lineages'], 3)
        self.assertEqual(pair['dope_matched_median'], 100.)
        self.assertEqual(pair['baseline_matched_median'], 2.)
        self.assertEqual(pair['median_dataset_difference'], -1.)
        self.assertFalse(pair['counts_as_dope_win'])

    def test_missing_sample_and_failed_auditor_reduce_coverage_only(self):
        self.cells[0]['status'] = 'fit_unavailable'
        self.cells[0]['utility'] = None
        other = next(c for c in self.cells if c['method'] == 'GaussianCopula'
                     and c['configuration'] == 'default' and c['size_multiplier'] == 1)
        other['utility']['catboost'] = dict(status='failed', error_type='ValueError')
        result = m.aggregate(self.cells)
        pair = result['paired_descriptive'][0]
        self.assertEqual(pair['complete_paired_lineages'], 99)
        dope_group = next(g for g in result['summary']
                          if g['dataset'] == 'opaque-000' and g['method'] == 'DOPE'
                          and g['configuration'] == m.PROFILES[0] and g['size_multiplier'] == 1)
        self.assertIsNone(dope_group['utility']['catboost']['median_retention'])
        self.assertEqual(sum(sum(g['statuses'].values()) for g in result['summary']), 6000)

    def test_charge_overrun_remains_visible_with_no_release_claim(self):
        for c in self.cells:
            if c['method'] == 'GaussianCopula':
                c.update(charged_artifact_bytes=26000, artifact_within_l3_cap=False)
        result = m.aggregate(self.cells)
        self.assertIsNone(result['release_safe_l3'])
        self.assertTrue(any(g['charged_artifact_bytes'] == 26000 for g in result['summary']))
        self.cells[0]['charged_artifact_bytes'] = 10241
        with self.assertRaises(ValueError): m.aggregate(self.cells)

    def test_no_clipping_and_no_bool_nonfinite_claim_values(self):
        for c in self.cells:
            for u in c['utility'].values(): u['retention'] = -2.
        self.assertEqual(m.aggregate(self.cells)['configuration_panels'][0]
                         ['utility']['catboost']['median_of_lineage_sample_medians'], -2.)
        for value in (True, float('nan'), float('inf')):
            self.cells[0]['utility']['catboost']['retention'] = value
            with self.subTest(value=value), self.assertRaises(ValueError): m.aggregate(self.cells)
        self.cells[0]['utility']['catboost']['retention'] = -2.
        self.cells[0]['ptf_v1'] = .99
        with self.assertRaises(ValueError): m.aggregate(self.cells)


if __name__ == '__main__': unittest.main(verbosity=2)
