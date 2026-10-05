"""Generated metadata controls only; no benchmark metrics or learners."""
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from research.benchmark import publish_dope_refinement_population as pub


def cells():
    return [dict(dataset=f'{d:016x}', method='DOPE', configuration=p, fit_seed=11,
        sample_seed=s, size_multiplier=n, status='ok', unavailable_reason=None, charged_artifact_bytes=100,
        projection_bytes_included=True,
        counts_as_dope_win=False, mfs_v2=None, ptf_v1=None, release_safe=None, superiority=None,
        utility={a:dict(informative=True, retention=s / 1000) for a in ('catboost', 'linear', 'mlp')})
        for d in range(100) for p in pub.fits.PROFILES for n in pub.SIZES for s in pub.SEEDS]


class PopulationValidation(unittest.TestCase):
    def metric(self, tstr=2.0, trtr=.2):
        row=dict(informative=True, retention=(1-tstr)/(1-trtr), tstr_loss=tstr, trtr_loss=trtr)
        return dict(implementation_sha256='0'*64, task='regression', gate_profile_complete=False, mfs_v2=None,
            rows=dict(train=10, synthetic=40, validation=3), null_loss=1.,
            utility={a:dict(row) for a in ('catboost', 'linear', 'mlp')},
            copy_counts=dict(exact=0, near=1), real_vs_real_control_counts=dict(exact=0, near=0))

    def test_negative_and_above_one_retention_are_not_clipped(self):
        for tstr in (2., 0.):
            value=self.metric(tstr=tstr)
            pub.metric_check(value, {'train_rows':10}, 4, '0'*64)
            self.assertEqual(value['utility']['catboost']['retention'], (1-tstr)/.8)
        value=self.metric(); value['utility']['catboost']['retention']=0.
        with self.assertRaises(ValueError): pub.metric_check(value, {'train_rows':10}, 4, '0'*64)

    def test_low_signal_and_auditor_failures_remain_null(self):
        value=self.metric(trtr=.999)
        for row in value['utility'].values(): row.update(informative=False, retention=None)
        value['utility']['mlp']={'status':'failed','error_type':'ValueError'}
        pub.metric_check(value, {'train_rows':10}, 4, '0'*64)
        value['utility']['catboost']['retention']=.99
        with self.assertRaises(ValueError): pub.metric_check(value, {'train_rows':10}, 4, '0'*64)

    def test_copy_controls_and_boolean_losses_are_rejected(self):
        for mutation in ('copies', 'loss', 'rows'):
            value=self.metric()
            if mutation=='copies': value['copy_counts']={'exact':2,'near':1}
            if mutation=='loss': value['null_loss']=True
            if mutation=='rows': value['rows']['synthetic']=10
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                pub.metric_check(value, {'train_rows':10}, 4, '0'*64)

    def test_receipt_hash_precedes_any_metric_or_report_decoding(self):
        with TemporaryDirectory() as tmp:
            base=Path(tmp); root=base/'validation'; root.mkdir()
            metric=root/'metric.json'; metric.write_text('not a metric')
            pin=hashlib.sha256(metric.read_bytes()).hexdigest()
            anchor=root/'receipt-lock-v1.json'; report_pin='2'*64
            value=dict(round_sha256=pub.ROUND, fit_receipt_lock_sha256=pub.fits.RECEIPTS,
                reconciliation_sha256=report_pin, refs={str(metric):pin})
            anchor.write_text(json.dumps(value)); anchor_pin=hashlib.sha256(anchor.read_bytes()).hexdigest()
            metric.write_text('rewritten invalid metric')
            with patch.object(pub, 'ROOT', root), patch.object(pub.fits, 'BASE', base):
                with self.assertRaisesRegex(ValueError, 'immutable evidence changed'): pub.build(anchor_pin, report_pin)
                value['refs'][str(metric)]=hashlib.sha256(metric.read_bytes()).hexdigest(); anchor.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, 'runtime manifest changed'): pub.build(anchor_pin, report_pin)

    def test_complete_matrix_and_typed_seed(self):
        rows = cells(); self.assertEqual(len(pub.complete_matrix(rows)), 100)
        for bad in (rows[:-1], rows[:-1] + [rows[0]]):
            with self.assertRaises(ValueError): pub.complete_matrix(bad)
        rows[0]['sample_seed'] = 101.0
        with self.assertRaises(ValueError): pub.complete_matrix(rows)

    def test_failed_samples_retain_null_utility_and_reason(self):
        rows = cells(); rows[0].update(status='fit_unavailable', unavailable_reason='transport_or_prelaunch_failure',
                                       utility=None, charged_artifact_bytes=None)
        pub.complete_matrix(rows)
        for changes in ({'unavailable_reason': None}, {'utility': {'catboost': {'retention': 1}}}):
            bad = copy.deepcopy(rows); bad[0].update(changes)
            with self.assertRaises(ValueError): pub.complete_matrix(bad)

    def test_sample_seeds_reduce_before_dataset_groups(self):
        rows = cells(); groups, panels = pub.summaries(rows)
        self.assertEqual(len(groups), 400)
        self.assertTrue(all(g['utility']['catboost']['median_retention'] == .211 for g in groups))
        rows[0].update(status='fit_unavailable', unavailable_reason='charged_artifact_cap', utility=None, charged_artifact_bytes=12000)
        # Other seeds of the same fit must retain its identical artifact charge.
        for r in rows:
            if r['dataset'] == rows[0]['dataset'] and r['configuration'] == rows[0]['configuration']:
                r['charged_artifact_bytes'] = 12000
                r.update(status='fit_unavailable', unavailable_reason='charged_artifact_cap', utility=None)
        groups, panels = pub.summaries(rows)
        missing = [g for g in groups if g['dataset'] == rows[0]['dataset'] and g['configuration'] == rows[0]['configuration']]
        self.assertEqual(len(missing), 2)
        self.assertTrue(all(g['utility']['catboost']['complete_informative'] is False for g in missing))

    def test_no_gated_score_from_normalized_retention(self):
        for key, value in [('ptf_v1', .99), ('mfs_v2', .99), ('counts_as_dope_win', True)]:
            rows = cells(); rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): pub.complete_matrix(rows)

    def test_fixed_pins_are_valid_digests(self):
        for pin in (pub.REFERENCE, pub.FIT_PUBLICATION, pub.ROUND, pub.AUXILIARY): pub.guard.digest(pin)

    def test_added_batch_aliases_fail_before_metric_decode(self):
        with TemporaryDirectory() as tmp:
            base=Path(tmp); root=base/'batch'; root.mkdir()
            metric=root/'n4-seed101.metric.json'; metric.write_text('not decoded by inventory check')
            metric_pin=hashlib.sha256(metric.read_bytes()).hexdigest()
            receipt=root/'receipt.json'; receipt.write_text(json.dumps(dict(evidence_files={metric.name:metric_pin})))
            pin=hashlib.sha256(receipt.read_bytes()).hexdigest()
            refs={str(receipt):pin, str(metric):metric_pin}
            report=dict(cells=[dict(sample_evidence=dict(metric_path=str(metric)),validation_receipt_sha256=pin)])
            auxiliary=dict(batches={str(root):dict(directories=[],files={str(p):dict(bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (metric,receipt)})})
            with patch.object(pub.fits, 'BASE', base), patch.object(pub, 'BASE', base):
                pub.sampling_inventories(report, refs, auxiliary)
                (root/'alias').symlink_to(root, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, 'runtime alias or special entry'):
                    pub.sampling_inventories(report, refs, auxiliary)

    def test_offline_aggregates_are_replayed_and_gates_stay_null(self):
        rows=cells(); references=[]
        for method, config in [('DOPE','features12_steps2048'), ('GaussianCopula','native_selected'),
                               ('Chow-Liu','native_selected'), ('independent_marginals','native_selected')]:
            references += [dict(r, method=method, configuration=config) for r in rows
                           if r['configuration']==pub.fits.PROFILES[0]]
        groups, panels=pub.summaries(rows+references)
        report=dict(cells=rows, datasets=sorted({r['dataset'] for r in rows}),
            source_sha256=pub.fits.sha256(Path(pub.__file__)), s3_data_lock_sha256=pub.fits.DATA,
            source_locks=dict(round=pub.ROUND, receipts='1'*64, reconciliation='2'*64,
                              density_reference=pub.REFERENCE, batch_auxiliary=pub.AUXILIARY),
            fit_ledger_reference=dict(sha256=pub.FIT_PUBLICATION), matched_reference=dict(sha256=pub.REFERENCE),
            logical_validation_cells=1200, physical_validation_batches=112, immutable_validation_cells_reused=144,
            global_configuration_tuned_or_selected=False, final_five_fit_coverage_complete=False,
            privacy_attack_coverage_complete=False, system_dynamic_library_closure_certified=False,
            logical_status_counts={'ok':1200}, lineage_groups=groups, summary=panels,
            paired_descriptive=pub.paired(groups, pub.fits.PROFILES), **pub.fits.GATES)
        with patch.object(pub, 'reference_cells', return_value=dict(cells=references)):
            pub.validate_report(report)
            for field in ('lineage_groups', 'summary', 'paired_descriptive', 'logical_status_counts',
                          'privacy_attack_coverage_complete', 'ptf_v1'):
                changed=copy.deepcopy(report)
                if field in ('lineage_groups', 'summary', 'paired_descriptive'): changed[field]=[]
                elif field=='logical_status_counts': changed[field]={'ok':1199}
                else: changed[field]=True
                with self.subTest(field=field), self.assertRaises(ValueError): pub.validate_report(changed)


if __name__ == '__main__': unittest.main()
