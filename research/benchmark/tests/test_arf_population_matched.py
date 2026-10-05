"""Generated closure and paired-analysis controls, with no benchmark learners."""
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from research.benchmark import publish_arf_population_matched as pub
from research.benchmark.tests.arf_matched_fixture import closed_fixture


def rows():
    return [dict(dataset=f'{d:016x}', method='ARF', configuration=c, fit_seed=11,
        sample_seed=s, size_multiplier=n, status='ok', unavailable_reason=None,
        charged_artifact_bytes=15000, projection_bytes_included=True, l3_artifact_cap_satisfied=False,
        counts_as_dope_win=False, mfs_v2=None, ptf_v1=None, release_safe=None, superiority=None,
        utility={a:dict(informative=True, retention=s/1000) for a in ('catboost','linear','mlp')})
        for d in range(100) for c in pub.CONFIGS for n in pub.dope.SIZES for s in pub.dope.SEEDS]


def references(cells):
    return [dict(r, method='DOPE', configuration=p,
        utility={a:dict(informative=True, retention=1+r['sample_seed']/1000) for a in ('catboost','linear','mlp')})
        for r in cells if r['configuration']==pub.CONFIGS[0] for p in pub.PROFILES]


class MatchedARFControls(unittest.TestCase):
    def test_complete_grid_rejects_missing_duplicate_and_float_alias_cells(self):
        r=rows(); self.assertEqual(len(pub.complete_matrix(r)),100)
        for value in (r[:-1],r[:-1]+[r[0]]):
            with self.assertRaises(ValueError): pub.complete_matrix(value)
        r[0]['fit_seed']=11.0
        with self.assertRaises(ValueError): pub.complete_matrix(r)

    def test_large_original_artifacts_remain_visible_with_false_byte_gate(self):
        r=rows(); pub.complete_matrix(r)
        r[0]['l3_artifact_cap_satisfied']=True
        with self.assertRaises(ValueError): pub.complete_matrix(r)
        r[0].update(charged_artifact_bytes=100,l3_artifact_cap_satisfied=True)
        pub.complete_matrix(r)

    def test_unavailable_cells_do_not_acquire_utility_or_win(self):
        r=rows(); r[0].update(status='sampling_unavailable',unavailable_reason='timeout',utility=None)
        pub.complete_matrix(r)
        for change in (dict(unavailable_reason=None),dict(utility={'catboost':{'retention':1}}),dict(counts_as_dope_win=True)):
            bad=copy.deepcopy(r);bad[0].update(change)
            with self.assertRaises(ValueError):pub.complete_matrix(bad)

    def test_native_and_common_objectives_never_create_production_claims(self):
        for key in ('mfs_v2','ptf_v1','release_safe','superiority'):
            r=rows();r[0][key]=.99
            with self.subTest(key=key),self.assertRaises(ValueError):pub.complete_matrix(r)

    def test_sample_seed_reduction_precedes_each_dataset_pair(self):
        r=rows(); groups,panels=pub.summaries(r+references(r));pairs=pub.paired(groups)
        self.assertEqual(len(pairs),36)
        self.assertTrue(all(p['paired_complete_informative_lineages']==100 for p in pairs))
        self.assertTrue(all(abs(p['median_paired_difference']-1)<1e-12 for p in pairs))
        self.assertTrue(all(p['superiority'] is None for p in pairs))
        self.assertTrue(all(g['utility']['catboost']['median_retention']==.211 for g in groups if g['method']=='ARF'))

    def test_partial_lineage_is_excluded_only_from_its_affected_pairs(self):
        r=rows(); ref=references(r)
        for c in r:
            if c['dataset']==r[0]['dataset'] and c['configuration']=='native_selected':
                c.update(status='failed',utility=None,unavailable_reason='timeout')
        groups,_=pub.summaries(r+ref);pairs=pub.paired(groups)
        self.assertTrue(all(p['paired_complete_informative_lineages']==(99 if p['reference_configuration']=='native_selected' else 100) for p in pairs))

    def test_no_partial_live_round_is_read_for_publication(self):
        with TemporaryDirectory() as tmp,patch.object(pub,'ROOT',Path(tmp)):
            with self.assertRaisesRegex(ValueError,'complete common matrix not yet closed'):
                pub.build('0'*64,'1'*64,'2'*64)

    def test_external_receipt_digest_precedes_any_metric_decode(self):
        with TemporaryDirectory() as tmp:
            base=Path(tmp);root=base/'round';root.mkdir();metric=root/'metric.json';metric.write_text('invalid fixture metric')
            pin=hashlib.sha256(metric.read_bytes()).hexdigest()
            anchor=root/'receipt-lock-v1.json';value=dict(complete_matrix=True,reconciliation_sha256='1'*64,refs={str(metric):pin})
            anchor.write_text(json.dumps(value));receipt_pin=hashlib.sha256(anchor.read_bytes()).hexdigest();metric.write_text('rewritten invalid metric')
            with patch.object(pub,'ROOT',root),patch.object(pub.native,'BASE',base),patch.object(pub.dope.fits,'BASE',base):
                with self.assertRaisesRegex(ValueError,'immutable evidence changed'):pub.build(receipt_pin,'1'*64,'2'*64)
                value['refs'][str(metric)]=hashlib.sha256(metric.read_bytes()).hexdigest();anchor.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError,'runtime manifest changed'):pub.build(receipt_pin,'1'*64,'2'*64)

    def test_offline_descriptive_replay_rejects_edited_aggregates_and_gates(self):
        r=rows();ref=references(r);groups,panels=pub.summaries(r+ref)
        report=dict(cells=r,datasets=sorted({c['dataset'] for c in r}),logical_validation_cells=1200,
            physical_sampling_batches=199,physical_metric_batches=199,source_sha256=pub.dope.fits.sha256(Path(pub.__file__)),
            s3_data_lock_sha256=pub.dope.fits.DATA,native_selection_reference=dict(sha256=pub.NATIVE_PUBLICATION),
            dope_reference=dict(sha256=pub.DOPE_PUBLICATION,earlier_density_sha256=pub.dope.REFERENCE),
            release_safe_l3_comparison_complete=False,logical_status_counts={'ok':1200},
            lineage_groups=groups,summary=panels,paired_descriptive=pub.paired(groups),
            source_locks=dict(sampling_round=pub.SAMPLE_ROUND,native_receipts=pub.NATIVE_RECEIPTS,native_reconciliation=pub.NATIVE_REPORT),**pub.GATES)
        with patch.object(pub,'reference_cells',return_value=ref):
            pub.validate_report(report)
            for key in ['summary','paired_descriptive','lineage_groups']:
                bad=copy.deepcopy(report);bad[key]=[]
                with self.assertRaises(ValueError):pub.validate_report(bad)
            for key in ['ptf_v1','global_family_selected','native_values_ranked_across_methods']:
                bad=copy.deepcopy(report);bad[key]=.99
                with self.assertRaises(ValueError):pub.validate_report(bad)
            bad=copy.deepcopy(report);bad['source_locks']['native_receipts']='0'*64
            with self.assertRaises(ValueError):pub.validate_report(bad)

    def test_complete_builder_refuses_uncovered_transitive_sample_before_metric_decode(self):
        directory=Path.cwd()/'target';directory.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=directory) as tmp,closed_fixture(Path(tmp),rows(),references(rows())) as (run,leaf,decoded):
            result=run()
            group=next(g for g in result['lineage_groups'] if g['dataset']==result['datasets'][0]
                       and g['method']=='ARF' and g['configuration']=='author_default' and g['size_multiplier']==1)
            self.assertEqual(len(result['cells']),1200)
            self.assertAlmostEqual(group['utility']['catboost']['median_retention'],.6)
            with self.assertRaisesRegex(ValueError,'transitive parent evidence'):
                run(omitted_leaf=leaf,mutate_leaf=True)
            self.assertEqual(decoded,[])

    def test_complete_builder_binds_metric_receipt_sample_and_replay_to_cell(self):
        directory=Path.cwd()/'target';directory.mkdir(exist_ok=True)
        changes=[lambda r:r['cells'][0].update(sample_evidence=copy.deepcopy(r['cells'][2]['sample_evidence'])),
            lambda r:r['cells'][0]['sample_evidence'].update(sample_seed=307),
            lambda r:r['cells'][0].update(validation_receipt_sha256='0'*64),
            lambda r:r['cells'][0]['sample_evidence'].update(sample_sha256='0'*64),
            lambda r:r['cells'][3]['sample_evidence'].update(metric_replay='not_repeated')]
        with TemporaryDirectory(dir=directory) as tmp,closed_fixture(Path(tmp),rows(),references(rows())) as (run,leaf,decoded):
            for index,change in enumerate(changes):
                with self.subTest(case=index),self.assertRaises(ValueError):run(change)
                # The last control reaches earlier valid cells before its n4 replay cell.
                if index<4:self.assertEqual(decoded,[])

    def test_complete_builder_does_not_replace_noncanonical_input_fit_seed(self):
        directory=Path.cwd()/'target';directory.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=directory) as tmp,closed_fixture(Path(tmp),rows(),references(rows())) as (run,leaf,decoded):
            for seed in [11.0,12,True]:
                with self.subTest(seed=seed),self.assertRaisesRegex(ValueError,'input common seed'):
                    run(lambda r:r['cells'][0].update(fit_seed=seed))
                self.assertEqual(decoded,[])

    def test_complete_builder_cross_checks_original_sampling_fit_seed(self):
        directory=Path.cwd()/'target';directory.mkdir(exist_ok=True)
        for seed in [11.0,12]:
            with self.subTest(seed=seed),TemporaryDirectory(dir=directory) as tmp:
                with closed_fixture(Path(tmp),rows(),references(rows()),sampling_fit_seed=seed) as (run,leaf,decoded):
                    with self.assertRaisesRegex(ValueError,'physical common lineage'):run()
                    self.assertEqual(decoded,[])

    def test_frozen_reference_digests_are_exact_builtin_valid_identities(self):
        for p in [pub.SAMPLE_ROUND,pub.NATIVE_RECEIPTS,pub.NATIVE_REPORT,pub.NATIVE_PUBLICATION,pub.DOPE_PUBLICATION]:pub.guard.digest(p)


if __name__=='__main__':unittest.main(verbosity=2)
