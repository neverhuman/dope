"""Fixed observer replay, physical roles, zero updates and bounded interruption."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
import torch
from research.levels.contracts import ResearchError, digest
from research.levels.corpus import Corpus
from research.levels.data import episode
from research.levels.losses import objective
from research.levels.model import SharedModel
from research.levels.train import prepared, train
from research.levels.validation import observe
from research.levels.tests.test_models import tiny
from research.levels.tests.test_training import config


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_legacy_draws_objectives_and_recipe_bytes_replay(self):
        cfg=config(); corpus=Corpus(prior_only=True); model=SharedModel(tiny())
        records=[]
        with torch.random.fork_rng():
            torch.manual_seed(cfg.seed+719)
            with torch.no_grad():
                for cursor in range(6):
                    context,query,_,profile=prepared(corpus,'development',cursor,model.config)
                    _,raw=objective(model,context,query,profile,cfg,2,differentiable=False)
                    records.append(raw)
        expected={level:{key:sum(r[level][key] for r in records)/6 for key in
            ('total','generation','numeric','categorical','missing','transfer_proxy',
             'driver_proxy','profile','consistency')} for level in records[0]}
        measured,panel=observe(model,corpus,cfg,2)
        self.assertEqual(expected,measured)
        self.assertEqual(panel['unique_physical_tables'],3)
        self.assertEqual(panel['task_counts'],dict(classification=4,regression=2))
        historical=asdict(cfg);historical.pop('validation_recipe');historical.pop('validation_episodes')
        self.assertEqual(cfg.manifest(),historical)

    def test_unique_panel_replays_balanced_physical_roles_without_changing_training(self):
        cfg=replace(config(),validation_recipe='owned-balanced-unique-v2',validation_episodes=40)
        corpus=Corpus(prior_only=True,prior_recipe='owned-mechanisms-unique-v2')
        training=[corpus.get('train',i) for i in range(40)]
        train_ids={r for source in training for r in source['row_ids']}
        for cursor in range(40):
            source=corpus.development(cursor,cfg.validation_recipe)
            context,query,_=episode(
                **{k:source[k] for k in ('rows','kinds','slots','row_ids','task','roles')},
                config=tiny(),cursor=cursor,seed=corpus.seed)
            self.assertFalse(train_ids.intersection(source['row_ids']))
            forbidden={source['row_ids'][i] for role in ('calibration','confirmation') for i in source['roles'][role]}
            self.assertFalse(forbidden.intersection(context.row_ids+query.row_ids))
            self.assertFalse(set(context.row_ids).intersection(query.row_ids))
            self.assertEqual(source,corpus.development(cursor,cfg.validation_recipe))
        model=SharedModel(tiny());rng=torch.get_rng_state().clone()
        weights={n:t.clone() for n,t in model.state_dict().items()}
        a,panel=observe(model,corpus,cfg,2); b,again=observe(model,corpus,cfg,2)
        self.assertEqual(a,b);self.assertEqual(panel['panel_sha256'],again['panel_sha256'])
        self.assertEqual(panel['unique_physical_tables'],40)
        self.assertEqual(panel['task_counts'],dict(classification=20,regression=20))
        self.assertEqual(sorted(panel['family_counts'].values()),[20,20])
        self.assertEqual(panel['optimizer_updates'],0)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(model.training)
        for n,t in model.state_dict().items():self.assertTrue(torch.equal(weights[n],t),n)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual([corpus.get('train',i) for i in range(40)],training)

    def test_panel_interruptions_publish_no_partial_means_and_restore_model_mode(self):
        cfg=replace(config(),validation_recipe='owned-balanced-unique-v2')
        model=SharedModel(tiny()).eval();corpus=Corpus(prior_only=True)
        for keyword,state in [('time_remaining','overbudget'),('resource_check','resource_pause')]:
            calls=[]
            def admit():calls.append(True);return len(calls)<=2
            levels,panel=observe(model,corpus,cfg,2,**{keyword:admit})
            self.assertIsNone(levels);self.assertEqual(panel['state'],state)
            self.assertEqual(panel['completed_episodes'],2)
            self.assertFalse(model.training)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);calls=[]
            def admit():calls.append(True);return len(calls)<=2
            _,terminal=train(tiny(),cfg,corpus,root/'paused',source_sha256='a'*64,resource_check=admit)
            self.assertEqual(terminal['state'],'resource_pause');self.assertEqual(terminal['optimizer_updates'],0)
            events=[json.loads(line) for line in (root/'paused/metrics.jsonl').read_text().splitlines()]
            self.assertEqual(events[0]['levels'],{});self.assertEqual(events[0]['observer']['completed_episodes'],2)
            self.assertTrue(Path(terminal['last_checkpoint']).is_dir())

    def test_observer_recipe_is_bound_to_exact_resume_and_retains_full_state(self):
        cfg=replace(config(),validation_recipe='owned-balanced-unique-v2',validation_episodes=4,
                    transfer_weight=0,driver_weight=0)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);corpus=Corpus(prior_only=True)
            whole,_=train(tiny(),cfg,corpus,root/'whole',source_sha256='a'*64)
            _,part=train(tiny(),cfg,corpus,root/'part',source_sha256='a'*64,stop_after=2)
            resumed,result=train(tiny(),cfg,corpus,root/'resumed',source_sha256='a'*64,resume=part['last_checkpoint'])
            from safetensors.torch import load_file
            def tensors(folder):
                checkpoint=json.loads((root/folder/'terminal.json').read_bytes())['last_checkpoint']
                return load_file(str(Path(checkpoint)/'state.safetensors'))
            a,b=tensors('whole'),tensors('resumed')
            self.assertEqual(set(a),set(b))
            for n in a:self.assertTrue(torch.equal(a[n],b[n]),n)
            def metadata(folder):
                checkpoint=json.loads((root/folder/'terminal.json').read_bytes())['last_checkpoint']
                return json.loads((Path(checkpoint)/'manifest.json').read_bytes())
            ma,mb=metadata('whole'),metadata('resumed')
            for key in ('python_rng','numpy_rng','optimizer_entries','optimizer_groups','recipe'):
                self.assertEqual(ma[key],mb[key],key)
            for key in ('step','cursor','episodes','processed_cells','optimizer_updates'):
                self.assertEqual(ma['state'][key],mb['state'][key],key)
            self.assertEqual(result['optimizer_updates'],4)
            for name in ('validation_episodes','validation_recipe'):
                changed=replace(cfg,**({name:6} if name=='validation_episodes' else
                    dict(validation_recipe='fixed-six-v1',validation_episodes=6)))
                with self.assertRaisesRegex(ResearchError,'resume_recipe_or_runtime_mismatch'):
                    train(tiny(),changed,corpus,root/name,source_sha256='a'*64,resume=part['last_checkpoint'])
            self.assertNotEqual(digest(cfg.manifest()),digest(replace(cfg,validation_episodes=6).manifest()))

    def test_invalid_panel_allocations_and_real_corpus_are_refused(self):
        for changes in [dict(validation_episodes=x) for x in (True,4,0,65,1.5)]+[
            dict(validation_recipe='unknown'),dict(validation_recipe='owned-balanced-unique-v2',validation_episodes=3)]:
            with self.assertRaises(ResearchError):replace(config(),**changes).validate()
        corpus=Corpus(prior_only=True);corpus.prior_only=False
        with self.assertRaisesRegex(ResearchError,'unique_validation_requires_owned_priors'):
            corpus.development(0,'owned-balanced-unique-v2')
        for cursor in (-1,True,1.5):
            with self.assertRaisesRegex(ResearchError,'invalid_corpus_cursor'):
                corpus.development(cursor,'owned-balanced-unique-v2')

    def test_p0_truth_repairs_rww2_masked_probe_and_methods_registry(self) -> None:
        import numpy as np
        from research.levels.descriptors import LANDMARK_NAMES, FINE_ADDED_INDICES
        from research.levels.rapid_validate import extract_masked_ultra_probe_vector
        from research.levels.scaling import FitOnlyTableTransform
        from research.levels.v3_canonical_roster_eval import (
            pack_rows_complete_wire,
            unpack_rows_complete_wire,
        )
        from research.levels.v3_canonical_summary import _compute_elo_and_paired_block
        from research.levels.world_class_eval import INTERNAL_STANDIN_ALIASES, _baseline_generator

        X_ext = np.column_stack([np.linspace(-6.0, 6.0, 50), np.linspace(5.0, -5.0, 50)])
        y_ext = np.linspace(-2.0, 2.0, 50)
        sc = FitOnlyTableTransform.fit(X_ext, y_ext, task="regression")
        pkt = pack_rows_complete_wire(X_ext, y_ext, sc, "ultra", seed=17)
        self.assertEqual(pkt[:4], b"RWW2")
        X_dec, y_dec, _ = unpack_rows_complete_wire(pkt, 400, seed=17)
        self.assertGreater(float(np.max(X_dec[:, 0])), 5.5)
        self.assertLess(float(np.min(X_dec[:, 0])), -5.5)
        self.assertEqual(y_dec.shape, (400,))

        base_report = {
            "task": "regression",
            "rows": 500,
            "features": 4,
            "target_stats": [0.2, 0.1, 0.0],
            "cells": [
                {"id": i + 1, "name": name, "state": "complete", "gain": 0.25 + 0.01 * i, "se": 0.02, "stable": True, "lambda_indices": [2]}
                for i, name in enumerate(LANDMARK_NAMES)
            ],
            "spectrum": [{"slot": s, "values": [0.1 * (k + 1) for k in range(9)]} for s in range(4)],
        }
        mut_report = json.loads(json.dumps(base_report))
        for idx in FINE_ADDED_INDICES:
            mut_report["cells"][idx]["gain"] = 0.99
            mut_report["cells"][idx]["se"] = 0.50
        v_base = extract_masked_ultra_probe_vector(base_report)
        v_mut = extract_masked_ultra_probe_vector(mut_report)
        self.assertGreater(float(v_base[69]), 0.2)
        self.assertGreater(float(np.abs(v_base[91:107]).sum()), 0.5)
        np.testing.assert_array_equal(v_base, v_mut)

        reg_path = Path(__file__).resolve().parents[3] / "bench" / "methods-registry.json"
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
        for alias, canon in INTERNAL_STANDIN_ALIASES.items():
            entry = reg["methods"][canon]
            self.assertEqual(entry["authenticity"], "internal_standin")
            self.assertFalse(entry["allowed_in_headline_claims"])
            Xs_a, _ = _baseline_generator(alias, X_ext, y_ext, "high", "regression", seed=9)
            Xs_c, _ = _baseline_generator(canon, X_ext, y_ext, "high", "regression", seed=9)
            np.testing.assert_array_equal(Xs_a, Xs_c)

        rows = [
            {
                "dataset_id": f"ds_{fam_i}_{view_i}",
                "family_id": f"fam_{fam_i}",
                "level": "high",
                "task": "regression",
                "domains": {"sender_scaled_test_domain": {"per_auditor": {
                    "catboost": {"clipped_R": 0.90}, "lightgbm": {"clipped_R": 0.90}, "mlp": {"clipped_R": 0.90},
                }}},
                "comparators": {"rows_wire": {"sender_scaled": {"per_auditor": {
                    "catboost": {"clipped_R": 0.85}, "lightgbm": {"clipped_R": 0.85}, "mlp": {"clipped_R": 0.85},
                }}}},
            }
            for fam_i in range(4) for view_i in range(2)
        ]
        _, paired = _compute_elo_and_paired_block(rows, n_boot=20)
        blk = paired["dope_mlc_v3_minus_rows_wire"]
        self.assertEqual(blk["paired_cluster_count"], 4)
        self.assertEqual(blk["bootstrap_unit"], "physical_source_family_or_dataset_cluster")
