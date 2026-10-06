"""Generated closure and paired-analysis controls, with no benchmark learners."""
import copy
import hashlib
import json
import os
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
    def test_publisher_replay_accepts_only_the_exact_frozen_historical_report(self):
        report = json.loads(Path(pub.__file__).with_name('results').joinpath(pub.NAME + '.json').read_bytes())
        pub.validate_publisher_identity(report)
        self.assertEqual(report['source_sha256'], pub.HISTORICAL_PUBLISHER_SHA)
        for change in (lambda r: r.update(source_sha256='0' * 64),
                       lambda r: r.update(logical_validation_cells=1199),
                       lambda r: r['cells'][0].update(charged_artifact_bytes=1)):
            altered = copy.deepcopy(report); change(altered)
            with self.assertRaisesRegex(ValueError, 'publication producing source differs'):
                pub.validate_publisher_identity(altered)

    def test_sampling_directory_comes_from_verified_source_declaration(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); out = root / 'attempt'; out.mkdir()
            (out / 'cache-area').mkdir(); (out / 'unexpected-area').mkdir()
            record = out / 'opaque.json'; record.write_text('{}')
            inventory = {str(record): dict(bytes=2, sha256=hashlib.sha256(record.read_bytes()).hexdigest())}
            source = root / 'source' / 'coordinator.py'; source.parent.mkdir()
            source.write_text("cache_directory = out / 'cache-area'\nenv = {'TMPDIR': str(cache_directory)}\n")
            refs = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
            with patch.object(pub, 'SAMPLING', root), patch.object(pub, 'BASE', root):
                self.assertEqual(pub.sampling_directories(out, refs), [str(out / 'cache-area')])
                with self.assertRaises(ValueError):
                    pub.guard.inventory(out, inventory, root, pub.sampling_directories(out, refs))
                (out / 'unexpected-area').rmdir()
                pub.guard.inventory(out, inventory, root, pub.sampling_directories(out, refs))
                refs[str(source)] = '0' * 64
                with self.assertRaisesRegex(ValueError, 'source anchor'):
                    pub.sampling_directories(out, refs)

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
        accounting = [dict(job_sha256=pub.digest({'job': i}), receipt_sha256=pub.digest({'receipt': i}),
            sampling_job_sha256=pub.digest({'job': i}), status='ok', operation_seconds=0,
            new_operation_started=True) for i in range(199)]
        for i, row in enumerate(r):
            row['validation_receipt_sha256'] = accounting[i % 199]['receipt_sha256']
            row['sampling_job_sha256'] = accounting[i % 199]['job_sha256']
        report=dict(cells=r,datasets=sorted({c['dataset'] for c in r}),logical_validation_cells=1200,
            physical_sampling_batches=199,physical_metric_batches=199,source_sha256=pub.dope.fits.sha256(Path(pub.__file__)),
            s3_data_lock_sha256=pub.dope.fits.DATA,native_selection_reference=dict(sha256=pub.NATIVE_PUBLICATION),
            dope_reference=dict(sha256=pub.DOPE_PUBLICATION,earlier_density_sha256=pub.dope.REFERENCE),
            release_safe_l3_comparison_complete=False,logical_status_counts={'ok':1200},
            lineage_groups=groups,summary=panels,paired_descriptive=pub.paired(groups),
            physical_metric_receipts=accounting, physical_sampling_receipts=copy.deepcopy(accounting),
            physical_metric_status_counts={'ok': 199}, sample_status_counts={'ok': 199},
            scheduler_wall_seconds_evidence=dict(sample=0, metric=0),
            cost=dict(native_fit=copy.deepcopy(pub.frozen_native_cost()),
                sample_operation_seconds=0, metric_operation_seconds=0,
                sample_scheduler_wall_seconds=0, metric_scheduler_wall_seconds=0),
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
            for change in [lambda v:v.update(physical_metric_status_counts={'timeout':199}),
                lambda v:v.update(logical_status_counts={'ok':1200.0}),
                lambda v:v.update(physical_metric_status_counts={'ok':199.0}),
                lambda v:v.update(sample_status_counts={'ok':199.0}),
                lambda v:v.update(logical_validation_cells=1200.0),
                lambda v:v.update(physical_sampling_batches=199.0),
                lambda v:v.update(physical_metric_batches=199.0),
                lambda v:v['cost'].update(metric_operation_seconds=-1),
                lambda v:v['cost'].update(metric_scheduler_wall_seconds=1),
                lambda v:v['physical_metric_receipts'][0].update(operation_seconds=True),
                lambda v:v['physical_metric_receipts'][0].update(status='timeout')]:
                bad=copy.deepcopy(report);change(bad)
                with self.assertRaises(ValueError):pub.validate_report(bad)
            for change in [dict(operation_seconds=-17, scheduler_wall_seconds=-19),
                dict(operation_seconds=True, scheduler_wall_seconds=True),
                dict(operation_seconds=12345, scheduler_wall_seconds=67890),
                dict(operation_seconds=float('nan'), scheduler_wall_seconds=0),
                dict(operation_seconds=0, scheduler_wall_seconds=float('inf')),
                dict(operation_seconds=0), dict(report['cost']['native_fit'], extra_seconds=0)]:
                bad=copy.deepcopy(report);bad['cost']['native_fit']=change
                with self.subTest(native_cost=change),self.assertRaises(ValueError):pub.validate_report(bad)
            bad=copy.deepcopy(report);bad['cells'][0]['validation_receipt_sha256']=None
            with self.assertRaisesRegex(ValueError,'metric outcome lacks'):pub.validate_report(bad)

            failed=copy.deepcopy(report)
            pin=failed['physical_metric_receipts'][-1]['receipt_sha256']
            failed['physical_metric_receipts'][-1]['status']='timeout'
            failed['physical_metric_status_counts']={'ok':198,'timeout':1}
            for cell in failed['cells']:
                if cell['validation_receipt_sha256']==pin:
                    cell.update(status='timeout',utility=None,unavailable_reason='timeout')
            failed['logical_status_counts']=dict(pub.Counter(c['status'] for c in failed['cells']))
            groups,panels=pub.summaries(failed['cells']+ref)
            failed.update(lineage_groups=groups,summary=panels,paired_descriptive=pub.paired(groups))
            pub.validate_report(failed)
            promoted=copy.deepcopy(failed)
            cell=next(c for c in promoted['cells'] if c['status']=='timeout')
            cell.update(status='ok',unavailable_reason=None,validation_receipt_sha256=None,
                utility=copy.deepcopy(report['cells'][0]['utility']))
            promoted['logical_status_counts']=dict(pub.Counter(c['status'] for c in promoted['cells']))
            groups,panels=pub.summaries(promoted['cells']+ref)
            promoted.update(lineage_groups=groups,summary=panels,paired_descriptive=pub.paired(groups))
            with self.assertRaisesRegex(ValueError,'metric outcome lacks'):pub.validate_report(promoted)
            forged=copy.deepcopy(failed)
            cell=next(c for c in forged['cells'] if c['status']=='timeout')
            cell.update(status='sampling_unavailable',validation_receipt_sha256=None,
                sample_evidence=None,null_loss=None,copy_counts=None,real_vs_real_control_counts=None,
                marginal_ks_mean=None,pair_correlation_fidelity=None,c2st_auc=None)
            forged['logical_status_counts']=dict(pub.Counter(c['status'] for c in forged['cells']))
            with self.assertRaisesRegex(ValueError,'metric outcome lacks'):pub.validate_report(forged)

            unavailable=copy.deepcopy(report)
            unavailable['physical_metric_receipts'].pop()
            unavailable['physical_metric_batches']=198
            unavailable['physical_metric_status_counts']={'ok':198}
            unavailable['physical_sampling_receipts'][-1]['status']='timeout'
            unavailable['sample_status_counts']={'ok':198,'timeout':1}
            for cell in unavailable['cells']:
                if cell['validation_receipt_sha256']==pin:
                    cell.update(status='sampling_unavailable',unavailable_reason='timeout',
                        validation_receipt_sha256=None,sample_evidence=None,utility=None,null_loss=None,
                        copy_counts=None,real_vs_real_control_counts=None,marginal_ks_mean=None,
                        pair_correlation_fidelity=None,c2st_auc=None)
            unavailable['logical_status_counts']=dict(pub.Counter(c['status'] for c in unavailable['cells']))
            groups,panels=pub.summaries(unavailable['cells']+ref)
            unavailable.update(lineage_groups=groups,summary=panels,paired_descriptive=pub.paired(groups))
            pub.validate_report(unavailable)

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
                self.assertEqual(decoded,[])

    def test_every_common_receipt_leaf_needs_both_authenticated_inventories(self):
        directory=Path.cwd()/'target';directory.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=directory) as tmp:
            with closed_fixture(Path(tmp),rows(),references(rows()),
                common_mutation=lambda r:r['evidence_files'].update({'operation.json':'f'*64})) as (run,leaf,decoded):
                with self.assertRaisesRegex(ValueError,'common receipt evidence anchor'):run()
                self.assertEqual(decoded,[])

    def test_null_common_receipt_leaf_never_aliases_absent_inventory_entries(self):
        directory=Path.cwd()/'target';directory.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=directory) as tmp:
            with closed_fixture(Path(tmp),rows(),references(rows()),
                common_mutation=lambda r:r['evidence_files'].update({'operation.json':None})) as (run,leaf,decoded):
                with self.assertRaisesRegex(ValueError,'frozen runtime digest'):run()
                self.assertEqual(decoded,[])

    def test_status_counts_and_coverage_reject_numeric_aliases(self):
        self.assertEqual(pub.status_counts({'ok':198,'timeout':1}),{'ok':198,'timeout':1})
        self.assertEqual(pub.count(199),199)
        for value in [True,False,1.0,-1,'1',None]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError,'canonical status counts'):
                    pub.status_counts({'ok':198,'timeout':value})
                with self.assertRaisesRegex(ValueError,'canonical nonnegative count'):pub.count(value)
        for value in [{'':1},{1:1},[],None]:
            with self.subTest(value=value),self.assertRaises(ValueError):pub.status_counts(value)

    def test_both_closure_count_maps_reject_a_boolean_failed_denominator(self):
        root=Path('/generated-closure');refs={str(root/'coordinator-exit.json'):'a'*64,
            str(root/'completion.json'):'b'*64}
        records=[dict(status='ok',operation_seconds=1),dict(status='timeout',operation_seconds=17)]
        valid={'ok':1,'timeout':1}
        for done_counts,report_counts in [(valid,{'ok':1,'timeout':True}),({'ok':1,'timeout':True},valid)]:
            with self.subTest(done=done_counts,report=report_counts):
                with patch.object(pub.native,'bound',side_effect=[dict(elapsed_seconds=20),dict(counts=done_counts)]):
                    with self.assertRaisesRegex(ValueError,'canonical status counts'):
                        pub.closure_accounting(root,refs,dict(physical_status_counts=report_counts,
                            new_metric_operation_seconds=18,coordinator_wall_seconds=20),records,'new_metric_operation_seconds')

    def test_last_common_metric_and_required_replay_are_preflighted_before_any_metric_decode(self):
        directory=Path.cwd()/'target';directory.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=directory) as tmp,closed_fixture(Path(tmp),rows(),references(rows())) as (run,leaf,decoded):
            run(); last_metric=decoded[-1]
            first_replay=next(Path(tmp).glob('common/attempts/*/attempt-0001/*.metric-replay.json'))
            for omitted in (last_metric,str(first_replay)):
                with self.subTest(omitted=Path(omitted).name),self.assertRaises(ValueError):run(omitted_leaf=omitted)
                self.assertEqual(decoded,[])

    def test_common_receipt_scope_cannot_be_replaced_with_public_null_or_false_gates(self):
        directory=Path.cwd()/'target';directory.mkdir(exist_ok=True)
        changes=[dict(official_tests_opened=True),dict(native_selection_changed=True),
            dict(new_generator_fits_started=1),dict(new_generator_fits_started=False),dict(global_family_selected=True),
            dict(mfs_v2=.9,ptf_v1=.9,release_safe=True,superiority=.9)]
        for change in changes:
            with self.subTest(change=change),TemporaryDirectory(dir=directory) as tmp:
                with closed_fixture(Path(tmp),rows(),references(rows()),common_mutation=lambda r:r.update(change)) as (run,leaf,decoded):
                    with self.assertRaises(ValueError):run()
                    self.assertEqual(decoded,[])

    def test_common_status_and_cost_denominators_derive_from_unique_receipts(self):
        directory=Path.cwd()/'target';directory.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=directory) as tmp,closed_fixture(Path(tmp),rows(),references(rows())) as (run,leaf,decoded):
            for change in [dict(physical_status_counts={'ok':0,'timeout':199}),dict(new_metric_operation_seconds=-1),
                dict(new_metric_operation_seconds=1),dict(coordinator_wall_seconds=-1),dict(coordinator_wall_seconds=1)]:
                with self.subTest(change=change),self.assertRaises(ValueError):run(lambda r:r.update(change))
                self.assertEqual(decoded,[])

    def test_failed_common_receipt_scope_is_checked_before_any_metric_decode(self):
        directory=Path.cwd()/'target';directory.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=directory) as tmp:
            with closed_fixture(Path(tmp),rows(),references(rows()),
                common_mutation=lambda r:r.update(status='prelaunch_or_metric_failure',official_tests_opened=True)) as (run,leaf,decoded):
                with self.assertRaisesRegex(ValueError,'receipt scope'):run()
                self.assertEqual(decoded,[])

    def test_failed_operation_cost_is_retained_and_unstarted_or_nonfinite_cost_rejects(self):
        receipt=dict(status='timeout',official_tests_opened=False,new_generator_fits_started=0,
            native_selection_changed=False,mfs_v2=None,ptf_v1=None,release_safe=None,superiority=None,
            operation=dict(new_operation_started=True,elapsed_seconds=17,official_tests_opened=False,
                foreign_processes_signaled=False))
        self.assertEqual(pub.operation_accounting(receipt,'a'*64,'b'*64)['operation_seconds'],17)
        for amount in [-1,True,float('nan'),float('inf')]:
            bad=copy.deepcopy(receipt);bad['operation']['elapsed_seconds']=amount
            with self.assertRaises(ValueError):pub.operation_accounting(bad,'a'*64,'b'*64)
        receipt['operation']['new_operation_started']=False
        with self.assertRaises(ValueError):pub.operation_accounting(receipt,'a'*64,'b'*64)

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
                    with self.assertRaisesRegex(ValueError,'original sampling receipt anchor'):run()
                    self.assertEqual(decoded,[])

    def assert_original_sampling_rejected(self, mutation):
        directory=Path.cwd()/'target';directory.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=directory) as tmp:
            with closed_fixture(Path(tmp),rows(),references(rows()),sampling_mutation=mutation) as (run,leaf,decoded):
                with self.assertRaises(ValueError):run()
                self.assertEqual(decoded,[])

    def test_original_sample_omitted_from_both_inventories_rejects_before_metrics(self):
        self.assert_original_sampling_rejected('omitted_both_sample')

    def test_null_original_sample_cannot_alias_absent_inventory_members(self):
        self.assert_original_sampling_rejected('null_omitted_sample')

    def test_false_sampling_unavailability_rejects_before_any_metric_decode(self):
        directory=Path.cwd()/'target';directory.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=directory) as tmp:
            with closed_fixture(Path(tmp),rows(),references(rows()),
                                sampling_unavailable='false_unavailable') as (run,leaf,decoded):
                with self.assertRaises(ValueError):run()
                self.assertEqual(decoded,[])

    def test_genuine_sampling_timeout_retains_failed_cost_and_available_metrics(self):
        directory=Path.cwd()/'target';directory.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=directory) as tmp:
            with closed_fixture(Path(tmp),rows(),references(rows()),
                                sampling_unavailable='genuine_timeout') as (run,leaf,decoded):
                report=run()
                self.assertEqual(report['sample_status_counts'],{'ok':198,'timeout':1})
                self.assertEqual(report['physical_metric_batches'],198)
                self.assertEqual(report['logical_status_counts'],{'sampling_unavailable':12,'ok':1188})
                self.assertEqual(report['cost']['sample_operation_seconds'],17)
                self.assertEqual(len(decoded),1188)
                unavailable=[c for c in report['cells'] if c['status']=='sampling_unavailable']
                self.assertEqual(len(unavailable),12)
                self.assertTrue(all(c['utility'] is None and c['validation_receipt_sha256'] is None
                                    and c['counts_as_dope_win'] is False for c in unavailable))

    def test_original_sample_omitted_from_original_inventory_rejects_before_metrics(self):
        self.assert_original_sampling_rejected('original_anchor_sample_omission')

    def test_original_sample_digest_must_match_authenticated_batch(self):
        self.assert_original_sampling_rejected('sample_digest')

    def test_original_receipt_digest_must_match_authenticated_file(self):
        self.assert_original_sampling_rejected('receipt_digest')

    def test_frozen_reference_digests_are_exact_builtin_valid_identities(self):
        for p in [pub.SAMPLE_ROUND,pub.NATIVE_RECEIPTS,pub.NATIVE_REPORT,pub.NATIVE_PUBLICATION,pub.DOPE_PUBLICATION]:pub.guard.digest(p)



    def test_authenticated_historical_source_projection_preserves_origin_and_exact_coverage(self):
        directory=Path.cwd()/'target';directory.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=directory) as tmp:
            root=Path(tmp);round_path=root/'historical/round.lock.json'
            origin=round_path.parent/'package/research/benchmark';origin.mkdir(parents=True)
            for name in pub.HISTORICAL_FILES:
                (origin/name).write_text('raise AssertionError("generated source must never execute")\n')
            cache=origin/'__pycache__';cache.mkdir();(cache/'fixture.pyc').write_bytes(b'generated cache never read')
            files={name:hashlib.sha256((origin/name).read_bytes()).hexdigest() for name in pub.HISTORICAL_FILES}
            round_path.write_text(json.dumps(dict(source_files=files)))
            round_pin=hashlib.sha256(round_path.read_bytes()).hexdigest()
            refs={str(round_path):round_pin,**{str(origin/name):pin for name,pin in files.items()}}
            projection=root/'projection'
            with patch.multiple(pub,HISTORICAL_ROUND=round_path,HISTORICAL_ROUND_SHA=round_pin,
                HISTORICAL_SOURCE=origin,HISTORICAL_PROJECTION=projection,HISTORICAL_FILES=files),patch.object(pub.native,'BASE',root):
                missing=dict(refs);missing.pop(str(origin/'manifest.py'))
                with self.assertRaisesRegex(ValueError,'authenticated historical source origin'):
                    pub.project_historical_sources(round_path,round_pin,files,missing)
                self.assertFalse(projection.exists())
                with self.assertRaisesRegex(ValueError,'authenticated historical source origin'):
                    pub.project_historical_sources(round_path,'0'*64,files,refs)
                self.assertFalse(projection.exists())
                binding=pub.flat_round_sources(round_path,files,round_pin,refs)
                self.assertEqual(set(p.name for p in projection.iterdir()),set(files))
                self.assertFalse(binding['source_files_executed']);self.assertFalse(binding['source_cache_bodies_read'])
                self.assertEqual((cache/'fixture.pyc').read_bytes(),b'generated cache never read')
                self.assertEqual(binding,pub.project_historical_sources(round_path,round_pin,files,refs))
                pub.verify_historical_projection(binding)
                alias=root/'outside-projection-alias.py'
                os.link(projection/'manifest.py',alias)
                with self.assertRaisesRegex(ValueError,'historical source body type or size'):
                    pub.verify_historical_projection(binding)
                alias.unlink()
                pub.verify_historical_projection(binding)
                (projection/'extra.py').write_bytes(b'generated extra')
                with self.assertRaisesRegex(ValueError,'extra member'):
                    pub.project_historical_sources(round_path,round_pin,files,refs)
                with self.assertRaisesRegex(ValueError,'flat inventory'):
                    pub.verify_historical_projection(binding)
                (projection/'extra.py').unlink()
                for path in [origin/'manifest.py',projection/'manifest.py']:
                    raw=path.read_bytes();path.write_bytes(b'changed generated body')
                    with self.assertRaises(ValueError):pub.verify_historical_projection(binding)
                    path.write_bytes(raw)
                path=projection/'manifest.py';raw=path.read_bytes();path.unlink();path.symlink_to(origin/'manifest.py')
                with self.assertRaisesRegex(ValueError,'runtime path not owned'):
                    pub.project_historical_sources(round_path,round_pin,files,refs)
                path.unlink();path.write_bytes(raw)
                with self.assertRaisesRegex(ValueError,'authenticated historical source origin'):
                    pub.flat_round_sources(root/'other/round.lock.json',files,round_pin,refs)
                pub.verify_historical_projection(binding)

    def test_source_key_formats_reject_mixed_nested_and_nonbuiltin_aliases(self):
        directory=Path.cwd()/'target';directory.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=directory) as tmp:
            root=Path(tmp);source=root/'source';source.mkdir()
            path=source/'fixture.py';path.write_bytes(b'# generated source\n')
            pin=hashlib.sha256(path.read_bytes()).hexdigest();files={str(path):pin}
            with patch.object(pub.native,'BASE',root):
                self.assertIsNone(pub.flat_round_sources(root/'round.lock.json',files,'a'*64,{}))
                class StringAlias(str):pass
                for bad in [{**files,'flat.py':pin},{str(source/'nested/fixture.py'):pin},
                            {str(source)+'/./fixture.py':pin},{str(source)+'//fixture.py':pin},
                            {'nested\\fixture.py':pin},{StringAlias(str(path)):pin},{path:pin}]:
                    with self.subTest(keys=list(bad)),self.assertRaises(ValueError):
                        pub.flat_round_sources(root/'round.lock.json',bad,'a'*64,{})

if __name__=='__main__':unittest.main(verbosity=2)
