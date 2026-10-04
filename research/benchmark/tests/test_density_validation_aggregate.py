"""Opaque completion controls; no live receipt or dataset is read."""
import unittest

from research.benchmark import density_validation_aggregate as helper


class DensityClosureControls(unittest.TestCase):
    def setUp(self):
        self.ids = tuple(f'opaque-{d}' for d in range(100))
        self.cells = [dict(dataset=f'opaque-{d}', method=m, configuration=k, fit_seed=11,
            sample_seed=s, size_multiplier=z, status='ok', artifact_bytes=12000,
            counts_as_dope_win=False, mfs_v2=None, ptf_v1=None, release_safe_l3=None,
            paired_superiority=None, utility={a:dict(informative=True, retention=-0.25)
                                            for a in helper.AUDITORS})
            for d in range(100) for m in helper.METHODS for k in helper.CONFIGS
            for s in helper.SEEDS for z in helper.SIZES]

    def test_complete_unclipped_negative_panel_denominators_and_reference_labels(self):
        lineages, panels = helper.aggregate(self.cells, self.ids)
        self.assertEqual((len(lineages), len(panels)), (1200, 12))
        self.assertTrue(all(p['lineages'] == 100 and p['counts_as_dope_win'] is False for p in panels))
        self.assertTrue(all(p['utility']['catboost']['median_of_lineage_sample_medians'] == -0.25 for p in panels))
        self.assertTrue(all(p['lineages_within_l3_byte_cap'] == 0 for p in panels))

    def test_one_missing_cell_rejected(self):
        with self.assertRaises(ValueError): helper.aggregate(self.cells[:-1], self.ids)

    def test_one_missing_dataset_rejected(self):
        with self.assertRaises(ValueError): helper.aggregate([c for c in self.cells if c['dataset'] != 'opaque-99'], self.ids)

    def test_duplicate_rejected(self):
        self.cells[-1] = dict(self.cells[0])
        with self.assertRaises(ValueError): helper.aggregate(self.cells, self.ids)

    def test_numeric_identity_aliases_rejected(self):
        for key, value in (('fit_seed', 11.0), ('sample_seed', 101.0), ('size_multiplier', True)):
            old = self.cells[0][key]; self.cells[0][key] = value
            with self.assertRaises(ValueError): helper.aggregate(self.cells, self.ids)
            self.cells[0][key] = old

    def test_replaced_complete_lineage_rejected(self):
        for c in self.cells:
            if c['dataset']=='opaque-99': c['dataset']='different-lineage'
        with self.assertRaises(ValueError): helper.aggregate(self.cells, self.ids)

    def test_bad_expected_lineage_set_rejected(self):
        for ids in (self.ids[:-1], self.ids[:-1]+(self.ids[0],), list(self.ids)):
            with self.assertRaises(ValueError): helper.aggregate(self.cells, ids)

    def test_unknown_method_or_configuration_rejected(self):
        for key in ('method','configuration'):
            old = self.cells[0][key]; self.cells[0][key] = 'unknown'
            with self.assertRaises(ValueError): helper.aggregate(self.cells, self.ids)
            self.cells[0][key] = old

    def test_inconsistent_projection_inclusive_bytes_rejected(self):
        self.cells[0]['artifact_bytes'] += 1
        with self.assertRaises(ValueError): helper.aggregate(self.cells, self.ids)

    def test_bad_bytes_rejected(self):
        for value in (-1, True, 1.0):
            self.cells[0]['artifact_bytes'] = value
            with self.assertRaises(ValueError): helper.aggregate(self.cells, self.ids)

    def test_gated_or_win_claim_rejected(self):
        self.cells[0]['ptf_v1'] = .99
        with self.assertRaises(ValueError): helper.aggregate(self.cells, self.ids)
        self.cells[0]['ptf_v1'] = None; self.cells[0]['counts_as_dope_win'] = True
        with self.assertRaises(ValueError): helper.aggregate(self.cells, self.ids)

    def test_missing_sample_metric_has_null_group_and_keeps_population_denominator(self):
        self.cells[0]['utility']['catboost']['informative'] = False
        groups, panels = helper.aggregate(self.cells, self.ids)
        group = next(g for g in groups if (g['dataset'],g['method'],g['configuration'],g['size_multiplier'])
                     == ('opaque-0',helper.METHODS[0],helper.CONFIGS[0],1))
        self.assertIsNone(group['utility']['catboost']['median_retention'])
        panel = next(p for p in panels if (p['method'],p['configuration'],p['size_multiplier'])
                     == (helper.METHODS[0],helper.CONFIGS[0],1))
        self.assertEqual(panel['lineages'],100)
        self.assertEqual(panel['utility']['catboost']['complete_informative_lineages'],99)

    def test_nonfinite_or_boolean_retention_rejected(self):
        for value in (float('nan'), float('inf'), True):
            self.cells[0]['utility']['catboost']['retention'] = value
            with self.assertRaises(ValueError): helper.aggregate(self.cells, self.ids)


if __name__ == '__main__':
    unittest.main(verbosity=2)
