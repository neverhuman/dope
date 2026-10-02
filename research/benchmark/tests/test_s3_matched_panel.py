"""Missing/duplicate cells and native selection cannot acquire published scores."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark.publish_s3_matched import (
    CONFIGS, SEEDS, SIZES, native_winner, schema, summarize, verify_cell,
)


def cells():
    return [dict(dataset='synthetic', method=m, configuration=c, fit_seed=11,
                 sample_seed=s, size_multiplier=z, status='ok', charged_artifact_bytes=100,
                 utility={a: {'retention': v, 'informative': True}
                          for a in ('catboost', 'linear', 'mlp')})
            for m, c in CONFIGS for z in SIZES for s, v in zip(SEEDS, (0.5, 0.7, 0.9))]


def attempts():
    return [dict(job={'trial': t, 'dataset': 'synthetic', 'method': 'TVAE', 'fit_seed': 11,
                      'train_sha256': 'a'*64, 'validation_sha256': 'b'*64,
                      'projection_sha256': 'c'*64, 'config': {'trial': t}},
                 native_kpi={'value': v}, artifact_bytes=100, wall_seconds=10,
                 common_retention=100 if t == 0 else -100)
            for t, v in enumerate((0.1, 0.8, 0.5, 0.2))]


class S3MatchedPanelTests(unittest.TestCase):
    def test_incomplete_and_duplicate_matrices_fail(self):
        rows = cells()
        for broken in (rows[:-1], rows + [rows[0]], rows[1:] + [rows[1]]):
            with self.assertRaisesRegex(ValueError, 'incomplete or duplicated'):
                summarize(broken, ['synthetic'])
        result = summarize(rows, ['synthetic'])
        self.assertEqual(len(result), 16)
        self.assertEqual(result[0]['utility']['catboost']['median_retention'], 0.7)

    def test_missing_or_low_signal_sample_makes_median_null(self):
        rows = cells()
        rows[0]['status'] = 'fit_unavailable'
        rows[6]['utility']['linear']['informative'] = False
        rows[12]['utility']['mlp']['retention'] = None
        rows[18]['utility']['catboost'] = {'status': 'failed', 'error_type': 'ValueError'}
        result = summarize(rows, ['synthetic'])
        for auditor in ('catboost', 'linear', 'mlp'):
            self.assertTrue(any(r['utility'][auditor]['median_retention'] is None for r in result))

    def test_native_winner_ignores_common_utility(self):
        rows = attempts()
        self.assertEqual(native_winner(rows)['job']['trial'], 1)
        rows[2]['native_kpi']['value'] = .8
        rows[2]['artifact_bytes'] = 99
        self.assertEqual(native_winner(rows)['job']['trial'], 2)
        for mutation in ('lineage', 'time', 'duplicate'):
            bad = copy.deepcopy(rows)
            if mutation == 'lineage':
                bad[0]['job']['validation_sha256'] = 'd'*64
            elif mutation == 'time':
                bad[0]['wall_seconds'] = 43200
            else:
                bad[0]['job']['trial'] = 1
            with self.assertRaises(ValueError):
                native_winner(bad)

    def test_missing_receipt_rejected_before_metrics(self):
        target = Path('target/s3-matched-tests')
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as root, patch(
                'research.benchmark.publish_s3_matched.read') as read:
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                verify_cell(Path(root), {'synthetic': True}, 'a'*64)
            read.assert_not_called()

    def test_committed_schema_cannot_promote_or_unseal(self):
        import jsonschema
        document = {'format': 'synthetic', 'official_tests_opened': False,
                    'production_certified': False, 'mfs_v2': None, 'ptf_v1': None}
        contract = schema(document)
        jsonschema.validate(document, contract)
        for key, value in [('official_tests_opened', True), ('production_certified', True),
                           ('mfs_v2', .99), ('ptf_v1', .99)]:
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(document | {key: value}, contract)


if __name__ == '__main__':
    unittest.main()
