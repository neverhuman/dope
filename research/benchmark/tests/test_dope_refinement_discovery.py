"""Coverage, missingness and descriptive aggregation controls without campaign IO."""
import copy
import unittest

from research.benchmark import publish_dope_refinement_discovery as publish


def cell(dataset, profile, size, seed, value=0.75):
    return dict(dataset=dataset, method='DOPE', configuration=profile, size_multiplier=size,
        sample_seed=seed, fit_seed=11, status='ok', unavailable_reason=None,
        charged_artifact_bytes=1000, projection_bytes_included=True,
        utility={a: dict(informative=True, retention=value) for a in publish.AUDITORS},
        mfs_v2=None, ptf_v1=None, release_safe=None, superiority=None)


class DiscoveryPanel(unittest.TestCase):
    def setUp(self):
        self.datasets = [f'opaque{i}' for i in range(6)]
        self.rows = [cell(d, p, z, s) for d in self.datasets for p in publish.PROFILES
                     for z in publish.SIZES for s in publish.SEEDS]
        for row in self.rows:
            if row['dataset'] == 'opaque5' and row['configuration'] == publish.PROFILES[1]:
                row.update(status='fit_unavailable', unavailable_reason='charged_artifact_cap',
                           charged_artifact_bytes=10381, utility=None)

    def test_complete_failure_remains_in_denominator(self):
        publish.validate_matrix(self.rows, self.datasets)
        _, panels = publish.summaries(self.rows)
        failed = [r for r in panels if r['configuration'] == publish.PROFILES[1]]
        self.assertTrue(all(r['planned_lineages'] == 6 and r['measured_lineages'] == 5 for r in failed))
        self.assertTrue(all(r['charged_bytes_max'] == 10381 for r in failed))

    def test_missing_cell_rejected(self):
        with self.assertRaises(ValueError): publish.validate_matrix(self.rows[:-1], self.datasets)

    def test_duplicate_replacing_cell_rejected(self):
        with self.assertRaises(ValueError): publish.validate_matrix(self.rows[:-1] + [self.rows[0]], self.datasets)

    def test_hidden_artifact_failure_rejected(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row['status'] != 'ok': row['unavailable_reason'] = 'source_unavailable'
        with self.assertRaises(ValueError): publish.validate_matrix(rows, self.datasets)

    def test_projection_charge_required(self):
        self.rows[0]['projection_bytes_included'] = False
        with self.assertRaises(ValueError): publish.validate_matrix(self.rows, self.datasets)

    def test_gated_score_rejected(self):
        self.rows[0]['ptf_v1'] = 0.99
        with self.assertRaises(ValueError): publish.validate_matrix(self.rows, self.datasets)

    def test_seed_float_rejected(self):
        self.rows[0]['sample_seed'] = 101.0
        with self.assertRaises(ValueError): publish.summaries(self.rows)

    def test_negative_retention_preserved(self):
        rows = [cell('opaque', 'profile', 4, s, -2.5) for s in publish.SEEDS]
        _, panels = publish.summaries(rows)
        self.assertEqual(panels[0]['utility']['catboost']['median_retention'], -2.5)

    def test_auditor_failure_makes_lineage_group_null(self):
        rows = [cell('opaque', 'profile', 4, s) for s in publish.SEEDS]
        rows[0]['utility']['mlp'] = dict(status='failed', error_type='RuntimeError')
        _, panels = publish.summaries(rows)
        self.assertEqual(panels[0]['utility']['mlp'], dict(informative_complete_lineages=0, median_retention=None))

    def test_low_signal_not_counted_as_informative(self):
        rows = [cell('opaque', 'profile', 4, s) for s in publish.SEEDS]
        rows[0]['utility']['linear'] = dict(informative=False, retention=None)
        _, panels = publish.summaries(rows)
        self.assertEqual(panels[0]['utility']['linear']['informative_complete_lineages'], 0)

    def test_nonfinite_retention_rejected(self):
        self.rows[0]['utility']['linear']['retention'] = float('nan')
        with self.assertRaises(ValueError): publish.summaries(self.rows)

    def test_dataset_median_follows_within_lineage_seed_median(self):
        rows = [cell('first', 'profile', 4, s, v) for s, v in zip(publish.SEEDS, [0., 0., 100.])]
        rows += [cell('second', 'profile', 4, s, v) for s, v in zip(publish.SEEDS, [10., 20., 20.])]
        _, panels = publish.summaries(rows)
        self.assertEqual(panels[0]['utility']['catboost']['median_retention'], 10.)

    def test_incomplete_seed_group_rejected(self):
        with self.assertRaises(ValueError): publish.summaries(self.rows[:-1])

    def test_pairs_keep_cap_failure_out_of_complete_pairs(self):
        rows = copy.deepcopy(self.rows)
        for d in self.datasets:
            for method, config in [('DOPE', 'features12_steps2048'), ('GaussianCopula', 'native_selected'),
                                   ('Chow-Liu', 'native_selected'), ('independent_marginals', 'native_selected')]:
                for z in publish.SIZES:
                    for s in publish.SEEDS:
                        row = cell(d, config, z, s, 0.25); row['method'] = method; rows.append(row)
        groups, _ = publish.summaries(rows)
        pairs = publish.paired(groups)
        self.assertEqual(len(pairs), 96)
        failed = [r for r in pairs if r['configuration'] == publish.PROFILES[1]]
        self.assertTrue(all(r['planned_lineages'] == 6 and r['paired_complete_informative_lineages'] == 5 for r in failed))
        self.assertTrue(all(r['median_paired_difference'] == 0.5 and r['superiority'] is None for r in pairs))


if __name__ == '__main__': unittest.main()
