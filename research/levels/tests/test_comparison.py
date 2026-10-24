"""Readout determinism, honest denominators and a real bounded benchmark journey."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from research.levels.benchmark import run
from research.levels.comparison import schedule, ridge_loss, retention, summarize
from research.levels.contracts import ResearchError,digest,canonical,LANDMARK_OBSERVED_GRAPH
from research.levels.corpus import Corpus, file_sha
from research.levels.packages import export_package
from research.levels.tests.test_models import tiny
from research.levels.tests.test_training import config
from research.levels.train import train


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_schedule_never_opens_final_and_rejects_duplicate_seeds(self):
        cases = schedule(['low', 'ultra'], [1, 2], [3, 4])
        self.assertEqual(len(cases), 32)
        self.assertEqual({r['family'] for r in cases}, {7, 8})
        with self.assertRaises(ResearchError):
            schedule(['low'], [1, 1], [3])

    def test_ridge_is_seed_independent_and_uninformative_retention_stays_null(self):
        values = np.asarray([[x, 2*x+1] for x in range(-8, 9)], dtype=np.float64)
        observed = np.ones(values.shape, dtype=bool)
        query = SimpleNamespace(values=values[::2], observed=observed[::2])
        first = ridge_loss(values, observed, query, False)
        np.random.seed(12); np.random.normal(size=1000)
        self.assertEqual(first, ridge_loss(values, observed, query, False))
        self.assertLess(first, .001)
        self.assertIsNone(retention(.1, .2, .01))
        self.assertIsNone(retention(0., 0., 0.))
        with self.assertRaises(ResearchError):
            retention(1., .2, float('nan'))

    def test_summary_preserves_failed_cells_and_refuses_missing_or_duplicate_rows(self):
        cases = schedule(['low'], [1], [3])
        rows = [dict(case, state='complete', synthetic_loss=.2, real_loss=.1, null_loss=.5,
                     retention=.75) for case in cases]
        rows[0] = dict(cases[0], state='failed', failure_type='RuntimeError')
        result = summarize(cases, rows)
        self.assertEqual(result['scheduled'], 4)
        self.assertEqual(result['complete'], 3)
        self.assertEqual(result['failures'], 1)
        self.assertFalse(result['all_cells_complete'])
        self.assertFalse(result['superiority_qualified'])
        with self.assertRaises(ResearchError):
            summarize(cases, rows[:-1])
        with self.assertRaises(ResearchError):
            summarize(cases, rows+[rows[0]])

    def test_benchmark_binds_protocol_and_retains_a_native_sample_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = replace(config(), steps=1, validate_every=1, checkpoint_every=1,
                          transfer_weight=0, driver_weight=0)
            _, result = train(tiny(), cfg, Corpus(prior_only=True), root/'train', source_sha256='a'*64)
            export_package(result['last_checkpoint'], root/'model')
            binary = root/'native fixture'
            binary.write_text('''#!/usr/bin/env python3
import csv, pathlib, shutil, sys
args=sys.argv[1:];args=args[args.index('--threads')+2:]
command=args[0];destination=pathlib.Path(args[args.index('-o')+1])
if command=='fit':
 source=pathlib.Path(args[1]);assert (source.parent.parent/'protocol.json').is_file()
 destination.write_bytes(b'fixture');shutil.copyfile(source,str(destination)+'.csv')
else:
 if args[args.index('--seed')+1]=='211':destination.write_text('')
 else:shutil.copyfile(str(pathlib.Path(args[1]))+'.csv',destination)
''')
            binary.chmod(0o700)
            admission = dict(disk_available=20*2**30, memory_available=16*2**30,
                             cpu_available=16, load=(0, 0, 0))
            with patch('research.levels.resources.snapshot', return_value=admission), patch('os.nice'):
                summary = run(root/'model', binary, file_sha(binary), root/'comparison', selected=['low'],
                              table_seeds=[20261101], sample_seeds=[101, 211], resource_check=lambda: True)
            self.assertEqual(summary['scheduled'], 8)
            self.assertEqual(summary['complete'], 6)
            self.assertEqual(summary['failures'], 2)
            self.assertFalse(summary['superiority_qualified'])
            terminal = json.loads((root/'comparison/terminal.json').read_bytes())
            self.assertEqual(terminal['state'], 'complete_with_failures')
            protocol = json.loads((root/'comparison/protocol.json').read_bytes())
            self.assertFalse(protocol['official_test_access'])
            self.assertFalse(protocol['final_family_access'])
            self.assertEqual(protocol['baseline_sha256'], file_sha(binary))
            with self.assertRaisesRegex(ResearchError, 'auditor_pin_required'):
                run(root/'model', binary, file_sha(binary), root/'unpinned',
                    auditor_engine=binary)
            self.assertFalse((root/'unpinned').exists())
            # A successful generator is not evidence that the separate auditor ran.
            # This fixture has no eval command; every auxiliary-audit failure stays counted.
            with patch('research.levels.resources.snapshot', return_value=admission), patch('os.nice'):
                audited = run(root/'model', binary, file_sha(binary), root/'audited', selected=['low'],
                    table_seeds=[20261101], sample_seeds=[101], resource_check=lambda: True,
                    auditor_engine=binary, expected_auditor_sha256=file_sha(binary))
            self.assertEqual(audited['scheduled'], 4)
            self.assertEqual(audited['failures'], 4)
            self.assertEqual(audited['complete'], 0)
            protocol = json.loads((root/'audited/protocol.json').read_bytes())
            self.assertEqual(protocol['format'], 'dope-owned-comparison-protocol-v3')
            self.assertEqual(protocol['native_auditor']['engine_sha256'], file_sha(binary))
            self.assertFalse(protocol['native_auditor']['driver_qualified'])

    def test_landmark_benchmark_binds_cache_passes_inputs_and_keeps_missing_table_cases(self):
        from research.levels.landmark_features import FEATURE_RECIPE,OBSERVED_RECIPE,derive_observed
        from research.levels.tests.test_landmark_inputs import write_cache
        from research.levels.tests.test_v2 import graph_config
        from research.levels.priors import table
        from research.levels.packages import encode_table
        class FixtureCorpus(Corpus):
            def __init__(self):
                super().__init__(prior_only=True)
                class Cache:
                    feature_recipe=OBSERVED_RECIPE
                    def attach(self,data):return data
                self.input_cache=Cache()
            def get(self,*args):
                d=super().get(*args);d['landmark_input']=([-.25]+[0.]*22,[True]*23);return d
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);cfg=replace(config(),steps=1,validate_every=1,checkpoint_every=1,
                                           transfer_weight=0,driver_weight=0)
            _,trained=train(graph_config(LANDMARK_OBSERVED_GRAPH),cfg,FixtureCorpus(),root/'train',source_sha256='a'*64)
            export_package(trained['last_checkpoint'],root/'model')
            pins=[];seed=20291201
            for family in (7,8):
                data=table(family,seed);data['source_provenance']=dict(kind='owned_prior',physical_table_id=digest(dict(family=family,seed=seed)))
                folder=root/f'features-{family}';folder.mkdir();pin=write_cache(folder,data)
                report=json.loads((folder/'report.json').read_bytes());report['cells'][0].update(state='uninformative',gain=-.25)
                manifest=json.loads((folder/'manifest.json').read_bytes());manifest['report_sha256']=digest(report)
                (folder/'report.json').write_bytes(canonical(report));(folder/'manifest.json').write_bytes(canonical(manifest))
                pins.append(derive_observed(folder,digest(manifest),'a'*64,root/f'observed-{family}'))
            index=dict(format='dope-fit-cv-landmark-cache-v1',feature_recipe=OBSERVED_RECIPE,entries=pins)
            binary=root/'inert-baseline';binary.write_text('''#!/usr/bin/env python3
import pathlib,shutil,sys
a=sys.argv[1:];a=a[a.index('--threads')+2:];dst=pathlib.Path(a[a.index('-o')+1])
if a[0]=='fit':dst.write_bytes(b'fixture');shutil.copyfile(a[1],str(dst)+'.csv')
else:shutil.copyfile(str(pathlib.Path(a[1]))+'.csv',dst)
''');binary.chmod(0o700)
            admission=dict(disk_available=20*2**30,memory_available=16*2**30,cpu_available=16,load=(0,0,0))
            with patch('research.levels.resources.snapshot',return_value=admission),patch('os.nice'):
                with patch('research.levels.packages.encode_table',wraps=encode_table)as encoded:
                    result=run(root/'model',binary,file_sha(binary),root/'observed',selected=['low'],
                        table_seeds=[seed],sample_seeds=[101],resource_check=lambda:True,landmark_input_cache=index)
                    self.assertEqual(encoded.call_count,2)
                    for call in encoded.call_args_list:
                        self.assertEqual(call.kwargs['landmark_features'][0][0],-.25)
                        self.assertTrue(call.kwargs['landmark_features'][1][0])
                self.assertEqual(result['complete'],4);self.assertEqual(result['failures'],0)
                protocol=json.loads((root/'observed/protocol.json').read_bytes())
                self.assertEqual(protocol['format'],'dope-owned-comparison-protocol-v4')
                self.assertEqual(protocol['landmark_inputs']['cache_sha256'],digest(index))
                self.assertEqual(protocol['landmark_inputs']['feature_recipe'],OBSERVED_RECIPE)
                records=[json.loads(line)for line in(root/'observed/records.jsonl').read_text().splitlines()]
                self.assertTrue(all(r['landmark_input_manifest_sha256']in[p['manifest_sha256']for p in pins]for r in records))
                partial=dict(index,entries=pins[:1])
                missing=run(root/'model',binary,file_sha(binary),root/'missing',selected=['low'],
                    table_seeds=[seed],sample_seeds=[101],resource_check=lambda:True,landmark_input_cache=partial)
                self.assertEqual(missing['scheduled'],4);self.assertEqual(missing['complete'],2);self.assertEqual(missing['failures'],2)
                failed=[json.loads(line)for line in(root/'missing/records.jsonl').read_text().splitlines()if json.loads(line)['state']=='failed']
                self.assertEqual({r['family']for r in failed},{8});self.assertEqual({r['method']for r in failed},{'shared','baseline'})
                with self.assertRaisesRegex(ResearchError,'cache_required_for_graph'):
                    run(root/'model',binary,file_sha(binary),root/'no-cache',resource_check=lambda:True)
                with self.assertRaisesRegex(ResearchError,'cache_recipe_mismatch'):
                    run(root/'model',binary,file_sha(binary),root/'wrong-cache',resource_check=lambda:True,
                        landmark_input_cache=dict(index,feature_recipe=FEATURE_RECIPE))
