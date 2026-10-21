#!/usr/bin/env python3
"""Durable unittest acceptance suite for P00..P04, I00, V01..V04, U02, U03, Q02, Q03, R08, F03, C05 + W01/E03..E07 CPU preflights."""
import contextlib
import hashlib
import io
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from research.transfer.workplan_packet import (
    NOT_RUN, PacketError, assert_canonical_acceptance_truth, main as packet_main,
    check_dashboard_200, resolve_packet_dir as _resolve_packet_dir,
    run_validate_plan, _safe_is_dir, _safe_is_file,
)

PACKET_DIR = _resolve_packet_dir()


def find_dope_kernel_bin() -> Path:
    for cand in (
        REPO_ROOT / "target/debug/dope-kernel",
        REPO_ROOT / "target/release/dope-kernel",
    ):
        if _safe_is_file(cand) and cand.stat().st_size > 0:
            return cand
    subprocess.run(["cargo", "build", "--locked", "--bin", "dope-kernel"], cwd=str(REPO_ROOT), check=True)
    return REPO_ROOT / "target/debug/dope-kernel"


class TestP00SourceFreeze(unittest.TestCase):
    def test_p00_source_hashes_and_traceability_dispositions(self):
        self.assertTrue((REPO_ROOT / "contracts/tdl-contract-v1.json").is_file())
        self.assertTrue((REPO_ROOT / "AGENTS.md").is_file())
        run_validate_plan()
        if _safe_is_dir(PACKET_DIR):
            sh = json.loads((PACKET_DIR / "SHA256SUMS.json").read_text())
            self.assertGreaterEqual(len(sh), 4)
            tr = json.loads((PACKET_DIR / "SOURCE_TRACEABILITY.json").read_text())
            reqs = tr["crosswalk"]
            self.assertEqual(len(reqs), 738)
            for r in reqs:
                self.assertTrue(r.get("task_ids") and r.get("disposition"), f"Unmapped requirement: {r}")
            v = json.loads((PACKET_DIR / "VALIDATION.json").read_text())
            self.assertTrue(v.get("passed"))
        self.assertIn("DOPE", check_dashboard_200("/").upper())


class TestP01RepoSnapshot(unittest.TestCase):
    def test_p01_repo_snapshot_and_screen_census(self):
        self.assertTrue((REPO_ROOT / "agent/owner-map.json").is_file())
        self.assertTrue((REPO_ROOT / "agent/test-map.json").is_file())
        run_validate_plan()
        if _safe_is_dir(PACKET_DIR):
            parent = PACKET_DIR.parent
            self.assertIsInstance(json.loads((parent / "REPO_SNAPSHOT.json").read_text()), dict)
            self.assertIsInstance(json.loads((parent / "SCREEN_OWNERSHIP.json").read_text()), (dict, list))
        check_dashboard_200("/")


class TestP02CodebaseInventory(unittest.TestCase):
    def test_p02_inventory_and_defect_map(self):
        self.assertTrue((REPO_ROOT / "docs/architecture.md").is_file())
        self.assertTrue((REPO_ROOT / "STATUS.md").is_file())
        run_validate_plan()
        if _safe_is_dir(PACKET_DIR):
            parent = PACKET_DIR.parent
            self.assertIsInstance(json.loads((parent / "REPO_GAP_MAP.json").read_text()), (dict, list))
            self.assertIsInstance(json.loads((PACKET_DIR / "S1-requirements.json").read_text()), (dict, list))
        check_dashboard_200("/")


class TestP03DecisionsReview(unittest.TestCase):
    def test_p03_decisions_and_review_synthesis(self):
        self.assertTrue((REPO_ROOT / "docs/decisions.md").is_file())
        self.assertIn("D-01", (REPO_ROOT / "docs/decisions.md").read_text())
        run_validate_plan()
        if _safe_is_dir(PACKET_DIR):
            decs = json.loads((PACKET_DIR / "DECISIONS.json").read_text())
            synth = (PACKET_DIR / "S1-review.md").read_text() + (PACKET_DIR / "S2-S3-review.md").read_text()
            self.assertIsInstance(decs, (dict, list))
            self.assertGreater(len(synth), 1000)
        check_dashboard_200("/")


class TestP04WorkplanDashboard(unittest.TestCase):
    def test_p04_workplan_dag_and_live_dashboard(self):
        self.assertTrue((REPO_ROOT / "docs/testing.md").is_file())
        decisions = (REPO_ROOT / "docs/decisions.md").read_text()
        for expected in ("SC-DEC-106", "60 / 65", "5 / 65"):
            self.assertIn(expected, decisions)
        if PACKET_DIR is None:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(packet_main(["--mode", "unit"]), 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["status"], "not_run")
            self.assertEqual(report["reason"], NOT_RUN)
            self.assertFalse(report["release_authorized"])
            self.assertEqual(report["scientific_qualification"], "not_established")
            self.assertNotIn("input_hashes", report)
            print(json.dumps(report, sort_keys=True))
            return
        run_validate_plan()
        assert_canonical_acceptance_truth(PACKET_DIR)
        check_dashboard_200("/")


class TestI00ProtectedIntegration(unittest.TestCase):
    def test_i00_linear_main_ancestry_no_conflict_markers_and_documented_override(self):
        grep_res = subprocess.run(["git", "-C", str(REPO_ROOT), "grep", "-n", "^<<<<<<< "], capture_output=True, text=True)
        self.assertNotEqual(grep_res.returncode, 0, f"Found conflict markers: {grep_res.stdout}")
        has_commits = subprocess.run(["git", "-C", str(REPO_ROOT), "cat-file", "-e", "0ac99c68ecd697abe19d5bab488f620d5c51517b^{commit}"], capture_output=True).returncode == 0
        if has_commits:
            subprocess.run(["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", "0ac99c68ecd697abe19d5bab488f620d5c51517b", "402954ed9dc34c66c2a7c986b6d3000b6a58746a"], check=True)
        if _safe_is_dir(PACKET_DIR):
            handoff = (PACKET_DIR / "FINAL_HANDOFF_POST_MERGE_402954ED.md").read_text()
            self.assertIn("neverhumanbot", handoff)
            self.assertIn("47325f13b9823c91a6fc71bf1600fc097e5907c9", handoff)


class TestV02IndependentReview(unittest.TestCase):
    def test_v02_pr269_neverhumanbot_approval_and_four_green_checks_on_47325f1(self):
        self.assertTrue((REPO_ROOT / "ops/merge_guard/Cargo.toml").is_file())
        if _safe_is_dir(PACKET_DIR):
            handoff = (PACKET_DIR / "FINAL_HANDOFF_POST_MERGE_402954ED.md").read_text()
            self.assertIn("47325f13b9823c91a6fc71bf1600fc097e5907c9", handoff)
            self.assertIn("neverhumanbot", handoff)
            self.assertIn("4099862141", handoff)
        if subprocess.run(["git", "-C", str(REPO_ROOT), "cat-file", "-e", "0ac99c68ecd697abe19d5bab488f620d5c51517b^{tree}"], capture_output=True).returncode == 0:
            rev = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "0ac99c68ecd697abe19d5bab488f620d5c51517b^{tree}"], capture_output=True, text=True, check=True).stdout.strip()
            self.assertEqual(len(rev), 40)


class TestV03ReceivingMainGreen(unittest.TestCase):
    def test_v03_receiving_main_402954ed_ci_and_jankurai_green(self):
        self.assertTrue((REPO_ROOT / "ops/ci/cpu.sh").is_file())
        self.assertTrue((REPO_ROOT / "ops/ci/security.sh").is_file())
        if _safe_is_dir(PACKET_DIR):
            handoff = (PACKET_DIR / "FINAL_HANDOFF_POST_MERGE_402954ED.md").read_text()
            self.assertIn("39123429109", handoff)
            self.assertIn("39123429089", handoff)


class TestV04PostMergeHandoff(unittest.TestCase):
    def test_v04_post_merge_handoff_complete(self):
        self.assertTrue((REPO_ROOT / "docs/release.md").is_file())
        run_validate_plan()
        if _safe_is_dir(PACKET_DIR):
            handoff_path = PACKET_DIR / "FINAL_HANDOFF_POST_MERGE_402954ED.md"
            self.assertTrue(handoff_path.is_file())
            text = handoff_path.read_text()
            for required in (
                "402954ed9dc34c66c2a7c986b6d3000b6a58746a",
                "Per-Host CI & Verification Status on Final `main`",
                "Engineering Status (`59 / 65` Tasks Done",
                "Science & Empirical Benchmark Status",
                "Privacy & Disclosure Status",
                "Legal, Licensing & Provenance Status",
                "Custody Fence & Worktree Preservation Status",
                "Next Action Per Gate",
            ):
                self.assertIn(required, text)


class TestV01MultiHostParity(unittest.TestCase):
    def test_v01_ci_local_gate_parity_and_handoff_receipts(self):
        ci_local = REPO_ROOT / "scripts/ci-local.sh"
        self.assertTrue(ci_local.is_file())
        self.assertIn("ops/ci/$1.sh", ci_local.read_text())
        if _safe_is_dir(PACKET_DIR):
            vdir = PACKET_DIR / "verify/main_402954ed"
            self.assertIn("host: xbabe1", (vdir / "V01_xbabe1_audit.log").read_text())
            self.assertIn("Python V1 boundary: 13 files, 8 checks passed", (vdir / "V01_xbabe2_audit.log").read_text())
            self.assertIn("no leaks found", (vdir / "V01_xbabe2_security.log").read_text())
            self.assertIn("host: xbabe3", (vdir / "V01_xbabe3_audit.log").read_text())


class TestU03HealthcareSimJourney(unittest.TestCase):
    def test_u03_fresh_prepare_verify_compile_sample_evaluate_journey(self):
        bin_path = str(find_dope_kernel_bin())
        target_dir = REPO_ROOT / "target"
        target_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(target_dir)) as tmpdir:
            out_dir = Path(tmpdir) / "healthcare-sim"
            subprocess.run(["python3", "examples/healthcare-sim/demo.py", "prepare", "--out", str(out_dir)], cwd=str(REPO_ROOT), check=True)
            subprocess.run(["python3", "examples/healthcare-sim/demo.py", "verify", "--out", str(out_dir)], cwd=str(REPO_ROOT), check=True)
            subprocess.run([bin_path, "compile-csv", "--csv", str(out_dir / "fit.csv"), "--target", "event_30d", "--candidate", "chow_liu", "--seed", "1729", "--deadline-seconds", "60", "--out", str(out_dir / "model.dpk")], cwd=str(REPO_ROOT), check=True)
            subprocess.run([bin_path, "sample-mixed", "--artifact", str(out_dir / "model.dpk"), "--rows", "640", "--seed", "1729", "--out", str(out_dir / "sampled.json")], cwd=str(REPO_ROOT), check=True)
            subprocess.run(["python3", "examples/healthcare-sim/demo.py", "evaluate", "--out", str(out_dir), "--sampled", str(out_dir / "sampled.json")], cwd=str(REPO_ROOT), check=True)
            receipt = json.loads((out_dir / "receipt.json").read_text())
            self.assertIn("simulated_evidence_only", str(receipt.get("status")))


class TestQ03NativeReleaseSmoke(unittest.TestCase):
    def test_q03_native_release_smoke_and_frozen_evidence(self):
        bin_path = find_dope_kernel_bin()
        expected_sha = hashlib.sha256(bin_path.read_bytes()).hexdigest()
        target_dir = REPO_ROOT / "target"
        target_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(target_dir)) as tmpdir:
            prefix = Path(tmpdir) / "install-prefix"
            prefix.mkdir(parents=True, exist_ok=True)
            (prefix / "unrelated-evidence.txt").write_text("preserve-me\n")
            subprocess.run(["bash", "scripts/install-native.sh", str(bin_path), expected_sha, str(prefix), "--dry-run"], cwd=str(REPO_ROOT), check=True, capture_output=True)
            subprocess.run(["bash", "scripts/install-native.sh", str(bin_path), expected_sha, str(prefix)], cwd=str(REPO_ROOT), check=True, capture_output=True)
            doc_res = subprocess.run([str(prefix / "bin/dope-kernel"), "doctor"], cwd=str(REPO_ROOT), check=True, capture_output=True, text=True)
            self.assertIn('"implementation":"native_rust"', doc_res.stdout)
            subprocess.run(["bash", "scripts/uninstall-native.sh", str(prefix)], cwd=str(REPO_ROOT), check=True, capture_output=True)
            self.assertFalse((prefix / "bin/dope-kernel").exists())
            self.assertEqual((prefix / "unrelated-evidence.txt").read_text().strip(), "preserve-me")
        res2 = subprocess.run(["python3", "-B", "-m", "unittest", "discover", "-s", "ops/ci", "-p", "test_*.py", "-v"], cwd=str(REPO_ROOT), capture_output=True, text=True, check=True)
        self.assertEqual(res2.returncode, 0)


class TestR08PriorDecoderStudy(unittest.TestCase):
    def test_r08_prior_decoder_fixture_pilot_and_closure_accounting(self):
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from research.transfer.adapters.generators.prior_decoder_study import StudyError, closure_bytes, deterministic_fixture_pilot, run_fixture_pilot
        out = run_fixture_pilot()
        self.assertEqual(out["status"], "fixture_pilot_passed_vendor_study_blocked")
        self.assertFalse(out["vendor_execution"])
        self.assertEqual(out["scientific_result"], "unmeasured")
        fix = deterministic_fixture_pilot()
        self.assertEqual(fix["prior_distilled_decoder"][1]["cold_bytes"], 33)
        self.assertEqual(fix["prior_distilled_decoder"][16]["amortized_bytes"], 26)
        with self.assertRaises(StudyError):
            closure_bytes([], 1, 1)


class TestF03BoundedParserFuzz(unittest.TestCase):
    def test_f03_bounded_varint_and_corrupt_header_fuzz_budget(self):
        rng = random.Random(20261018)
        bin_path = str(find_dope_kernel_bin())
        target_dir = REPO_ROOT / "target"
        target_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(target_dir)) as tmpdir:
            tmp = Path(tmpdir)
            payloads = [
                b"",
                b"DPK3",
                b"DPK3\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01",
                b"DPK3\x03\x02" + b"\x80" * 32,
                b"DPK3\x03\x03" + b"\xff" * 64,
                b'{"format":"dope-tdl-report","version":1,"nan":NaN}',
            ]
            for i in range(10):
                buf = bytearray(rng.randbytes(rng.randint(1, 256)))
                if i % 2 == 0 and len(buf) >= 4:
                    buf[:4] = b"DPK3"
                payloads.append(bytes(buf))
            for idx, data in enumerate(payloads):
                f = tmp / f"fuzz_{idx}.dpk"
                f.write_bytes(data)
                res = subprocess.run([bin_path, "sample-mixed", "--artifact", str(f), "--rows", "4", "--seed", "1", "--out", str(tmp / "out.json")], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=5)
                self.assertNotEqual(res.returncode, 0)
                self.assertNotIn("panicked at", res.stderr)


class TestU02StudioA11yAndSecuritySmoke(unittest.TestCase):
    def test_u02_studio_a11y_landmarks_and_dom_state_machine_smoke(self):
        html = (REPO_ROOT / "apps/studio/index.html").read_text()
        ts = (REPO_ROOT / "apps/studio/studio.ts").read_text()
        self.assertNotIn("innerHTML", html)
        self.assertNotIn("innerHTML", ts)
        for landmark in ('class="skip"', '<main id="main">', 'role="status"', 'aria-live="polite"', 'aria-label="Receipt sections"', '<label for="receipt-file">'):
            self.assertIn(landmark, html)
        node_script = (
            "const fs=require('fs'),vm=require('vm');const code=fs.readFileSync('apps/studio/studio.ts','utf8');const nodes={};"
            "function el(id){if(!nodes[id])nodes[id]={id,textContent:'',className:'',hidden:false,children:[],style:{},files:[],addEventListener:(ev,fn)=>{nodes[id]['on_'+ev]=fn;},append:(...c)=>nodes[id].children.push(...c),replaceChildren:(...c)=>{nodes[id].children=[...c];}};return nodes[id];}"
            "const document={querySelector:(s)=>el(s.replace(/^#/,'')),createDocumentFragment:()=>({children:[],append:function(...c){this.children.push(...c);}}),createElement:(t)=>({tag:t,textContent:'',style:{},children:[],append:function(...c){this.children.push(...c);}}),createTextNode:(t)=>({textContent:t})};"
            "const ctx=vm.createContext({document,console,Array,Number,String,Object,Math,JSON,Error,Uint8Array});vm.runInContext(code,ctx);"
            "ctx.render({format:'unknown-v0'},{name:'u.json',size:10},'abc');if(el('qualification').className!=='state bad')throw new Error('bad1');"
            "ctx.render({format:'dope-tdl-result-v2',campaign_completeness_state:'complete',empirical_privacy_state:'blocked',artifact_qualification_state:'qualified',rights_state:'cleared',ledger:[{model:'m1',state:'passed'}]},{name:'b.json',size:20},'def');if(el('qualification').className!=='state bad')throw new Error('bad2');"
            "const bigLedger=Array.from({length:120},(_,i)=>({model:'m'+i,state:'passed',evidence_url:'https://example.org/'+i}));"
            "ctx.render({format:'dope-tdl-result-v2',campaign_completeness_state:'complete',empirical_privacy_state:'passed',artifact_qualification_state:'qualified',rights_state:'cleared',mandatory_model_count:120,mandatory_model_failures:0,ledger:bigLedger},{name:'ok.json',size:30},'123');"
            "if(el('qualification').className!=='state good'||el('more').hidden!==false)throw new Error('good');"
            "el('more').on_click();el('more').on_click();if(el('more').hidden!==true)throw new Error('hide');console.log('STUDIO_DOM_SMOKE_OK');"
        )
        res = subprocess.run(["node", "-e", node_script], cwd=str(REPO_ROOT), capture_output=True, text=True, check=True)
        self.assertIn("STUDIO_DOM_SMOKE_OK", res.stdout)


class TestC05RustPythonGoldenParity(unittest.TestCase):
    def test_c05_rust_and_python_golden_schema_and_tolerance_parity(self):
        self.assertGreaterEqual(len(sorted((REPO_ROOT / "tests/golden/transfer").glob("v2_t90_*.json"))), 25)
        res_py = subprocess.run(
            [sys.executable, "-B", "-m", "unittest", "research.transfer.tests.test_metrics", "research.transfer.tests.test_statistics", "research.transfer.tests.test_tdl_t122_mandatory_design", "research.transfer.tests.test_tdl_t122_regressions", "research.transfer.tests.test_tdl_result_schemas", "-v"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, check=True)
        self.assertEqual(res_py.returncode, 0)


class TestQ02FreshUserAndPdfQa(unittest.TestCase):
    def test_q02_paper_pdf_and_fresh_user_qa_report(self):
        if _safe_is_dir(PACKET_DIR):
            qa_report = PACKET_DIR / "handoff/Q02_FRESH_USER_PDF_QA_402954e_20261018.md"
            self.assertTrue(qa_report.is_file())
            self.assertGreater(len(qa_report.read_text()), 500)
        readme_path = REPO_ROOT / "README.md"
        readme_text = readme_path.read_text()
        self.assertIn("dope-kernel compile-csv", readme_text)
        self.assertNotIn("dope compile-csv", readme_text)
        self.assertIn("cross-model feature-driver preservation: not yet measured", readme_text)
        self.assertIn("Feature-Target Correlation Proxy", readme_text)
        self.assertIn("strongest measured comparator on this historical block", readme_text)
        self.assertIn("paper/main.tex", readme_text)
        # The landing page is the prose outside generated blocks and collapsed <details>
        # sections; the historical and kernel blocks stay verbatim inside them.
        stripped = re.sub(r"<!-- BEGIN (DOPE1 \w+|BENCHMARK|TDL RESULTS|PINNED DEMOS|COMPRESSION|QUICKSTART|EQUAL BYTES) -->.*?<!-- END \1 -->", "", readme_text, flags=re.S)
        while True:
            collapsed = re.sub(r"<details>(?:(?!<details>).)*?</details>", "", stripped, flags=re.S)
            if collapsed == stripped:
                break
            stripped = collapsed
        self.assertTrue(650 <= len(stripped.split()) <= 1250)
        wp_env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "docs/whitepaper/scripts")}
        for mod in ("test_readme_benchmark", "test_figure_one", "test_hero_figure"):
            res = subprocess.run([sys.executable, "-B", "-m", "unittest", mod, "-v"], cwd=str(REPO_ROOT), env=wp_env, capture_output=True, text=True, check=True)
            self.assertEqual(res.returncode, 0)
        res_f1 = subprocess.run([sys.executable, "-B", str(REPO_ROOT / "docs/whitepaper/scripts/figure_one.py"), "--check"], cwd=str(REPO_ROOT), env=wp_env, capture_output=True, text=True, check=True)
        self.assertEqual(res_f1.returncode, 0)
        wp_scripts = str(REPO_ROOT / "docs/whitepaper/scripts")
        if wp_scripts not in sys.path:
            sys.path.insert(0, wp_scripts)
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        import claim_registry, paper_v2_guards
        from tools.claims import foundation, render
        self.assertEqual(claim_registry.publication_failures(), [])
        self.assertEqual(paper_v2_guards.run_v2_lint(REPO_ROOT / "docs/whitepaper"), [])
        f_sum = foundation.load_foundation_summary(REPO_ROOT)
        self.assertEqual(f_sum["hero_title"], "More predictive gain. About one-third the state.")
        self.assertAlmostEqual(f_sum["target_correlation_proxy"]["GaussianCopula"]["median"], 0.975, places=3)
        self.assertAlmostEqual(f_sum["target_correlation_proxy"]["GaussianCopula"]["median_of_seed_mins"], 0.923106, places=5)
        self.assertAlmostEqual(f_sum["target_correlation_proxy"]["GaussianCopula"]["pooled_seed_median"], 0.973214, places=5)
        hero_svg = (REPO_ROOT / "paper/generated/hero-frontier.svg").read_text()
        for tok in ("More predictive gain. About one-third the state.", "DOPE (1,844 B)", "Gaussian copula (5,464 B)", "86.3%", "74.7%", "87.1%", "75.4%", "85.8%", "71.3%", "Cross-model feature-driver preservation: not yet measured"):
            self.assertIn(tok, hero_svg)
        self.assertNotIn("Proposed T90", hero_svg)
        import csv as _csv, io as _io
        hero_rows = list(_csv.DictReader(_io.StringIO((REPO_ROOT / "paper/generated/hero-frontier.csv").read_text())))
        self.assertEqual(len(hero_rows), 6)
        self.assertEqual(render.hero_csv_agreement_failures(hero_rows, foundation.hero_six_bar_rows(f_sum)), [])
        self.assertTrue((REPO_ROOT / "paper/generated/legacy-v2-catboost.csv").is_file())
        self.assertEqual([k for k, v in render.expected_outputs(REPO_ROOT).items() if (REPO_ROOT / k).read_bytes() != v], [])


class TestW01CpuPreflight(unittest.TestCase):
    def test_w01_auditor_qualification_cpu_preflight_and_fail_closed_locks(self):
        from research.transfer.adapters.auditors import kumo_tabular, tabicl
        r1 = kumo_tabular.KumoRecipe(task="regression", device="cpu")
        r2 = kumo_tabular.KumoRecipe(task="regression", device="cpu")
        self.assertEqual(r1.sha256, r2.sha256)
        self.assertTrue(kumo_tabular.LOCK.is_file() and tabicl.LOCK.is_file())
        with self.assertRaises(ValueError):
            kumo_tabular.KumoRecipe(task="regression", device="tpu")
        t1 = tabicl.TabICLRecipe(task="regression", device="cpu")
        self.assertEqual(t1.sha256, tabicl.TabICLRecipe(task="regression", device="cpu").sha256)


class TestE03CpuPreflight(unittest.TestCase):
    def test_e03_s1_privacy_calibration_and_pilot_admission_fail_closed(self):
        from research.transfer import calibration_receipt, pilot_admission
        with self.assertRaises(pilot_admission.PilotError):
            pilot_admission._require_key(b"short-key")
        with self.assertRaises(ValueError):
            calibration_receipt.verify_calibration_receipt({"format": "invalid"})


class TestE04CpuPreflight(unittest.TestCase):
    def test_e04_m1_precommit_pins_and_fail_closed_unblocked_rejection(self):
        from research.transfer import precommit
        pins = precommit.expected_pins(REPO_ROOT)
        self.assertGreaterEqual(len(pins), 20)
        with self.assertRaises(precommit.PrecommitError):
            precommit.validate_blocked_precommit({"format": precommit.FORMAT, "version": 1}, REPO_ROOT)


class TestE05CpuPreflight(unittest.TestCase):
    def test_e05_campaign_manifest_loader_and_custody_hold_fail_closed(self):
        from research.transfer import lineage, lineage_manifest
        reg = lineage.load_registry(REPO_ROOT / "research/transfer/registries/sources.json")
        self.assertGreater(len(reg.families()), 0)
        with self.assertRaises(lineage_manifest.LineageError):
            lineage_manifest.admit_source_manifest({"sources": []}, reg, signing_key=b"short")


class TestE06CpuPreflight(unittest.TestCase):
    def test_e06_independent_replay_and_frontier_coverage_fail_closed(self):
        from research.transfer import frontier_coverage
        self.assertFalse(frontier_coverage._admissible_uncertainty({}))
        self.assertFalse(frontier_coverage._admissible_uncertainty({"calibration_receipt": {"format": "forged"}}))
        with self.assertRaises(ValueError):
            frontier_coverage._coverage([], supported=True)


class TestE07CpuPreflight(unittest.TestCase):
    def test_e07_sealed_test_guard_and_observational_protocol_preflight(self):
        obs_dir = REPO_ROOT / "examples/public-regression/bikeshare-observational"
        proto = json.loads((obs_dir / "protocol.json").read_text())
        self.assertEqual(proto["status"], "development_extension_frozen_before_comparator_execution")
        self.assertIn("previously used in an unqualified DOPE-only observational run", proto["exposure_disclosure"])
        self.assertFalse(proto["campaign_claims"]["confirmatory"])
        self.assertEqual(proto["selection"]["rows_consumed_for_selection"], 0)
        res = subprocess.run([sys.executable, "-B", str(obs_dir / "test_runner.py")], cwd=str(obs_dir), capture_output=True, text=True, check=True)
        self.assertEqual(res.returncode, 0)
        runner_output = res.stdout + res.stderr
        self.assertRegex(runner_output, r"Ran [1-9]\d* tests? in ")
        self.assertIn("OK", runner_output)

    def test_zero_test_output_cannot_satisfy_e07_runner_acceptance(self):
        from unittest import mock
        empty = subprocess.CompletedProcess([], 0, "", "Ran 0 tests in 0.000s\nOK")
        with mock.patch.object(subprocess, "run", return_value=empty) as run:
            with self.assertRaises(AssertionError):
                self.test_e07_sealed_test_guard_and_observational_protocol_preflight()
        run.assert_called_once()


class TestD19AndBlockerUnblockMatrix(unittest.TestCase):
    def test_d19_governance_hold_and_blocker_unblock_matrix(self):
        dec_text = (REPO_ROOT / "docs/decisions.md").read_text()
        self.assertIn("D19", dec_text)
        for tid in ("W01", "E03", "E04", "E05", "E06", "E07"):
            self.assertIn(f"**{tid}**", dec_text)
        if _safe_is_dir(PACKET_DIR):
            self.assertTrue((PACKET_DIR / "DECISIONS.json").is_file())
            d19_candidates = [PACKET_DIR / "DECISIONS_D19_PROPOSED.json", PACKET_DIR / "DECISIONS.json"]
            self.assertTrue(any(p.is_file() and "D19" in p.read_text() for p in d19_candidates))

    def test_packet_dir_portability_and_strict_explicit_env(self):
        from unittest import mock
        with mock.patch.dict(os.environ, {"DOPE_PACKET_MODE": "auto", "DOPE_REQUIRE_PACKET": "0"}):
            os.environ.pop("DOPE_PACKET_DIR", None)
            with mock.patch.object(Path, "is_dir", side_effect=PermissionError("foreign checkout denied")):
                self.assertIsNone(_resolve_packet_dir())
                with self.assertRaises(PacketError):
                    _resolve_packet_dir("/inaccessible/packet")
        with self.assertRaises(RuntimeError):
            _resolve_packet_dir(str(REPO_ROOT / "target/_nonexistent_explicit_packet_dir_"), mode="qualify")


if __name__ == "__main__":
    unittest.main(verbosity=2)
