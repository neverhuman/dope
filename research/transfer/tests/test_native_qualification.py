"""Fail-closed W01 admission and evidence-accounting tests."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from research.transfer import native_qualification as nq


def pin(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def payload():
    cases = []
    for model, spec in nq.MODEL_FILES.items():
        code_repo, code_revision, checkpoint_repo, checkpoint_revision = nq.MODEL_SOURCES[model]
        for task in nq.TASKS:
            if model == "tabicl_v2":
                checkpoint = "tabicl-classifier-v2-20260212.ckpt" if task == "classification" else "tabicl-regressor-v2-20260212.ckpt"
            else:
                checkpoint = "large/classifier.pt" if task == "classification" else "large/regressor.pt"
            cases.append({"model": model, "task": task, "code_repo": code_repo,
                          "code_revision": code_revision, "checkpoint_repo": checkpoint_repo,
                          "checkpoint_revision": checkpoint_revision, "checkpoint": checkpoint,
                          "dependency_closure_sha256": pin(model + "closure"),
                          "code_license_sha256": pin(model + "code-license"),
                          "weights_license_sha256": pin(model + "weights-license"),
                          "role": "selection",
                          "class_order": ["negative", "positive"] if task == "classification" else [],
                          "recipe_sha256": pin(model + task + "recipe"),
                          "context_sha256": pin(model + task + "context"),
                          "query_sha256": pin(model + task + "query"),
                          "resource_caps": {"wall_seconds": 3600, "memory_bytes": 8_000_000_000,
                                            "gpu_bytes": 0, "threads": 1}})
    return {"schema": nq.SCHEMA, "status": "admitted", "lease_id": "test-lease",
            "source_commit": "1" * 40,
            "admission": {"rights": "approved", "resource": "approved", "roles": "approved"},
            "cases": cases}


def sign_manifest(directory: Path, value: dict) -> tuple[Path, Path, str]:
    private, public = directory / "private.pem", directory / "public.pem"
    subprocess.run(["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(private)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["openssl", "pkey", "-in", str(private), "-pubout", "-out", str(public)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    source, signature = directory / "payload.json", directory / "payload.sig"
    source.write_bytes(body)
    subprocess.run(["openssl", "pkeyutl", "-sign", "-inkey", str(private), "-rawin",
                    "-in", str(source), "-out", str(signature)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    envelope = directory / "manifest.json"
    envelope.write_text(json.dumps({"payload": value,
                                    "signature_ed25519_base64": base64.b64encode(signature.read_bytes()).decode()}))
    return envelope, public, hashlib.sha256(public.read_bytes()).hexdigest()


def q2_parity(case):
    order = case["class_order"]
    digest = pin(case["model"] + case["task"] + "parity")
    return {"native_checkpoint_loaded": True, "adapter_output_parity": "established",
            "adapter_vs_sdk_max_abs_delta": 0.0, "tolerance": 1e-6,
            "adapter_output_sha256": digest, "sdk_output_sha256": digest,
            "adapter_output_shape": [4, len(order) or 1],
            "sdk_output_shape": [4, len(order) or 1],
            "adapter_class_order": order, "sdk_class_order": order,
            "class_order": order, "recipe_sha256": case["recipe_sha256"],
            "preprocessing_sha256": pin(case["model"] + "preprocessing"),
            "output_aggregation_sha256": pin(case["model"] + "output"),
            "seed": 42, "precision": "float32"}


class NativeQualificationTests(unittest.TestCase):
    def test_admission_requires_exact_four_model_task_cases_and_explicit_approvals(self):
        value = payload()
        nq.validate_admission_payload(value)
        for mutate in (
            lambda p: p.update(status="draft"),
            lambda p: p["admission"].update(roles="pending"),
            lambda p: p["cases"].pop(),
            lambda p: p["cases"][0].update(task="unknown"),
            lambda p: p["cases"][0].update(code_revision="0" * 40),
            lambda p: p["cases"][1].update(class_order=["same", "same"]),
            lambda p: p["cases"][0].update(checkpoint="large/regressor.pt"),
            lambda p: p["cases"][0]["resource_caps"].update(threads=0),
            lambda p: p["cases"][0].update(context_sha256="not-a-pin"),
        ):
            broken = payload()
            mutate(broken)
            with self.assertRaises(nq.QualificationError):
                nq.validate_admission_payload(broken)

    def test_signed_manifest_requires_out_of_band_key_pin_and_valid_signature(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, public, key_hash = sign_manifest(root, payload())
            verified = nq.verify_signed_manifest(manifest, public, key_hash)
            self.assertEqual(verified["lease_id"], "test-lease")
            with self.assertRaisesRegex(nq.QualificationError, "trusted_key_pin_mismatch"):
                nq.verify_signed_manifest(manifest, public, pin("wrong key"))
            envelope = json.loads(manifest.read_text())
            envelope["payload"]["status"] = "draft"
            manifest.write_text(json.dumps(envelope))
            with self.assertRaisesRegex(nq.QualificationError, "signature_invalid"):
                nq.verify_signed_manifest(manifest, public, key_hash)

    def test_checkpoint_verification_checks_exact_bytes_and_contains_paths(self):
        value = payload()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected = {}
            for model in nq.MODEL_FILES:
                for case in value["cases"]:
                    if case["model"] != model:
                        continue
                    path = root / model / case["checkpoint"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    data = (model + case["task"]).encode()
                    path.write_bytes(data)
                    expected[case["checkpoint"]] = (len(data), hashlib.sha256(data).hexdigest())
            patched = {model: {"revision": spec["revision"], "files": dict(expected)}
                       for model, spec in nq.MODEL_FILES.items()}
            # The fixtures use distinct names, so retain only each model's two entries.
            for model in nq.MODEL_FILES:
                patched[model]["files"] = {
                    case["checkpoint"]: expected[case["checkpoint"]]
                    for case in value["cases"] if case["model"] == model
                }
            with patch.object(nq, "MODEL_FILES", patched):
                verified = nq.verify_checkpoints(value, root)
                self.assertEqual(len(verified), 4)
                self.assertTrue(all(item["verified"] for item in verified.values()))
                case = value["cases"][0]
                path = root / case["model"] / case["checkpoint"]
                path.write_bytes(b"tampered")
                with self.assertRaisesRegex(nq.QualificationError, "checkpoint_pin_mismatch"):
                    nq.verify_checkpoints(value, root)

    def test_partial_evidence_never_counts_as_qualification(self):
        result = nq.new_result(payload())
        self.assertEqual(nq.assess_result(result), "incomplete")
        for case in result["cases"]:
            for gate in nq.GATES:
                case["gates"][gate] = {"status": "passed", "evidence_sha256": pin(case["model"] + gate)}
            case["gates"]["Q2"]["observation"] = {"native_checkpoint_loaded": True}
            case["gates"]["Q9"]["independent_review"] = {
                "reviewer_id": "eligible-reviewer", "disposition": "accepted",
                "receipt_sha256": pin(case["model"] + "independent-review"),
            }
        self.assertEqual(nq.assess_result(result), "technical_smoke_reviewed")
        for case in result["cases"]:
            case["gates"]["Q2"]["observation"] = q2_parity(case)
        self.assertEqual(nq.assess_result(result), "qualified")
        result["cases"][0]["gates"]["Q2"]["observation"]["sdk_output_sha256"] = pin("foreign reference")
        self.assertEqual(nq.assess_result(result), "incomplete")
        result["cases"][0]["gates"]["Q2"]["observation"] = q2_parity(result["cases"][0])
        result["cases"][0]["gates"]["Q9"]["evidence_sha256"] = None
        self.assertEqual(nq.assess_result(result), "incomplete")

    def test_q2_requires_source_bound_output_parity_and_matching_review(self):
        result = nq.new_result(payload())
        for case in result["cases"]:
            for gate in nq.GATES:
                case["gates"][gate] = {"status": "passed", "evidence_sha256": pin(case["model"] + gate)}
            case["gates"]["Q2"]["observation"] = q2_parity(case)
            case["gates"]["Q9"]["independent_review"] = {
                "reviewer_id": "eligible-reviewer", "disposition": "accepted",
                "receipt_sha256": pin(case["model"] + "review"),
            }
        self.assertEqual(nq.assess_result(result), "qualified")
        result["cases"][0]["gates"]["Q2"]["observation"]["adapter_vs_sdk_max_abs_delta"] = 1e-3
        self.assertEqual(nq.assess_result(result), "incomplete")

    def test_bind_independent_review_copies_only_exact_accepted_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = nq.new_result(payload())
            source["environment"] = {"dependency_closure_sha256": pin("closure")}
            for case in source["cases"]:
                for gate in nq.GATES[:-1]:
                    case["gates"][gate] = {"status": "passed", "evidence_sha256": pin(case["model"] + case["task"] + gate)}
                case["gates"]["Q2"]["observation"] = {"native_checkpoint_loaded": True}
                case["gates"]["Q9"] = {"status": "pending_independent_review", "evidence_sha256": None}
            result_path = root / "capped.json"
            result_path.write_text(json.dumps(source, sort_keys=True))
            result_sha = nq.sha256_file(result_path)
            with self.assertRaisesRegex(nq.QualificationError, "file_unavailable"):
                nq.bind_independent_review(
                    result_path, root / "missing-review.json", root / "missing-qualified.json",
                    expected_result_sha256=result_sha, expected_review_sha256=pin("missing"),
                    expected_reviewer_id="reviewer-1",
                )
            review = {
                "schema": "dope.independent_cross_family_review.f65_01_w01_q9_signoff.v1",
                "disposition": "accepted",
                "bound_execution_artifacts": {
                    "capped_result_sha256": result_sha,
                    "lease_id": source["lease_id"],
                    "source_commit": source["source_commit"],
                    "dependency_closure_sha256": source["environment"]["dependency_closure_sha256"],
                },
                "reviewer": {"reviewer_id": "reviewer-1", "author_separation_verified": True,
                             "authored_by_reviewer": False,
                             "role": "independent_cross_family_reviewer"},
            }
            review_path = root / "review.json"
            review_path.write_text(json.dumps(review, sort_keys=True))
            review_sha = nq.sha256_file(review_path)
            output_path = root / "qualified.json"
            qualified = nq.bind_independent_review(
                result_path, review_path, output_path,
                expected_result_sha256=result_sha, expected_review_sha256=review_sha,
                expected_reviewer_id="reviewer-1",
            )
            self.assertEqual(qualified["status"], "technical_smoke_reviewed")
            self.assertEqual(nq.assess_result(qualified), "technical_smoke_reviewed")
            self.assertEqual(qualified["release_authorization"], "not_authorized")
            self.assertTrue(all(case["gates"]["Q9"]["evidence_sha256"] == review_sha
                                for case in qualified["cases"]))
            self.assertEqual(nq.sha256_file(result_path), result_sha)
            with self.assertRaisesRegex(nq.QualificationError, "qualified_result_already_exists"):
                nq.bind_independent_review(
                    result_path, review_path, output_path,
                    expected_result_sha256=result_sha, expected_review_sha256=review_sha,
                    expected_reviewer_id="reviewer-1",
                )

    def test_bind_independent_review_fails_closed_on_unbound_or_unaccepted_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = nq.new_result(payload())
            source["environment"] = {"dependency_closure_sha256": pin("closure")}
            for case in source["cases"]:
                for gate in nq.GATES[:-1]:
                    case["gates"][gate] = {"status": "passed", "evidence_sha256": pin(case["model"] + gate)}
            result_path = root / "capped.json"
            result_path.write_text(json.dumps(source, sort_keys=True))
            result_sha = nq.sha256_file(result_path)
            review = {
                "schema": "dope.independent_cross_family_review.f65_01_w01_q9_signoff.v1",
                "disposition": "accepted",
                "bound_execution_artifacts": {
                    "capped_result_sha256": "0" * 64,
                    "lease_id": source["lease_id"],
                    "source_commit": source["source_commit"],
                    "dependency_closure_sha256": source["environment"]["dependency_closure_sha256"],
                },
                "reviewer": {"reviewer_id": "reviewer-1", "author_separation_verified": True,
                             "authored_by_reviewer": False,
                             "role": "independent_cross_family_reviewer"},
            }
            review_path = root / "review.json"
            review_path.write_text(json.dumps(review, sort_keys=True))
            review_sha = nq.sha256_file(review_path)
            with self.assertRaisesRegex(nq.QualificationError, "review_execution_binding_mismatch"):
                nq.bind_independent_review(
                    result_path, review_path, root / "qualified.json",
                    expected_result_sha256=result_sha, expected_review_sha256=review_sha,
                    expected_reviewer_id="reviewer-1",
                )
            review["bound_execution_artifacts"]["capped_result_sha256"] = result_sha
            review["disposition"] = "rejected"
            review_path.write_text(json.dumps(review, sort_keys=True))
            with self.assertRaisesRegex(nq.QualificationError, "independent_review_not_accepted"):
                nq.bind_independent_review(
                    result_path, review_path, root / "qualified.json",
                    expected_result_sha256=result_sha,
                    expected_review_sha256=nq.sha256_file(review_path),
                    expected_reviewer_id="reviewer-1",
                )
            review["disposition"] = "accepted"
            review["bound_execution_artifacts"]["source_commit"] = "0" * 40
            review_path.write_text(json.dumps(review, sort_keys=True))
            with self.assertRaisesRegex(nq.QualificationError, "review_execution_binding_mismatch"):
                nq.bind_independent_review(
                    result_path, review_path, root / "qualified.json",
                    expected_result_sha256=result_sha,
                    expected_review_sha256=nq.sha256_file(review_path),
                    expected_reviewer_id="reviewer-1",
                )

    def test_native_runner_runtime_closure_and_case_input_pin_verification(self):
        from research.transfer import native_qualification_runtime as nqr
        from research.transfer import run_native_qualification as rnq

        self.assertEqual(rnq.RUNNER_FILE, Path(rnq.__file__).resolve())
        self.assertEqual(nqr.RUNTIME_FILE, Path(nqr.__file__).resolve())
        self.assertNotEqual(rnq.RUNNER_FILE, nqr.RUNTIME_FILE)
        for mod_path in (
            Path(nq.__file__).resolve(),
            nqr.RUNTIME_FILE,
            rnq.RUNNER_FILE,
            Path(__file__).resolve(),
        ):
            self.assertLess(len(mod_path.read_text(encoding="utf-8").splitlines()), 500)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs_dir = root / "inputs"
            inputs_dir.mkdir(parents=True, exist_ok=True)
            ctx_doc = {"feature_columns": ["f0"], "target_column": "y_reg", "rows": [{"f0": 1.0, "y_reg": 2.0}]}
            qry_doc = {"feature_columns": ["f0"], "rows": [{"f0": 3.0}]}
            rec_doc = {"allow_auto_download": False, "device": "cpu", "threads": 1, "n_estimators": 2, "seed": 42}
            base = inputs_dir / "tabicl_v2_regression"
            for suffix, doc in (("_context.json", ctx_doc), ("_query.json", qry_doc), ("_recipe.json", rec_doc)):
                Path(str(base) + suffix).write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
            case = {
                "model": "tabicl_v2",
                "task": "regression",
                "context_sha256": nq.sha256_file(Path(str(base) + "_context.json")),
                "query_sha256": nq.sha256_file(Path(str(base) + "_query.json")),
                "recipe_sha256": nq.sha256_file(Path(str(base) + "_recipe.json")),
            }
            loaded_ctx, loaded_qry, loaded_rec = rnq._case_inputs(case, root)
            self.assertEqual((loaded_ctx, loaded_qry, loaded_rec), (ctx_doc, qry_doc, rec_doc))
            Path(str(base) + "_query.json").write_text(json.dumps({"rows": []}, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(nq.QualificationError, "case_input_pin_mismatch"):
                rnq._case_inputs(case, root)
            worker_out = root / "worker-failed.json"
            self.assertEqual(rnq._worker(case, root, worker_out), 1)
            worker_receipt = json.loads(worker_out.read_text(encoding="utf-8"))
            self.assertEqual(worker_receipt["status"], "failed")
            self.assertEqual(worker_receipt["error_type"], "QualificationError")
            self.assertEqual(worker_receipt["error"], "case_input_pin_mismatch")




class NativeRuntimeMetadataPinTests(unittest.TestCase):
    def test_vendored_record_cannot_replace_the_installed_distribution_record(self):
        from pathlib import PurePosixPath
        from types import SimpleNamespace
        from research.transfer import native_qualification_runtime as runtime
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            installed = PurePosixPath("installed-1.dist-info/RECORD")
            vendor = PurePosixPath("pkg/_vendor/foreign-2.dist-info/RECORD")
            for name, raw in ((installed, b"installed-record"), (vendor, b"vendor-record")):
                path = root / str(name)
                path.parent.mkdir(parents=True)
                path.write_bytes(raw)
            dist = SimpleNamespace(version="1", files=[vendor, installed],
                                   locate_file=lambda item: root / str(item))
            self.assertEqual(runtime._installed_record(dist, "1", pin("installed-record")),
                             root / str(installed))
            with self.assertRaises(nq.QualificationError):
                runtime._installed_record(dist, "1", pin("vendor-record"))
            with self.assertRaises(nq.QualificationError):
                runtime._installed_record(dist, "2", pin("installed-record"))

    def test_missing_or_ambiguous_installed_records_refuse_before_model_execution(self):
        from pathlib import PurePosixPath
        from types import SimpleNamespace
        from research.transfer import native_qualification_runtime as runtime
        for files in (None, [], [PurePosixPath("pkg/vendor/other.dist-info/RECORD")],
                      [PurePosixPath("a.dist-info/RECORD"), PurePosixPath("b.dist-info/RECORD")]):
            dist = SimpleNamespace(version="1", files=files,
                                   locate_file=lambda _: self.fail("must refuse ambiguous inventory first"))
            with self.assertRaises(nq.QualificationError):
                runtime._installed_record(dist, "1", pin("unused"))

    def test_component_license_labels_bind_the_reviewed_source_and_keep_weights_distinct(self):
        from research.transfer import native_qualification_runtime as runtime
        for model, expected in (("kumo_tabular", ("apache-2.0", "openmdw-1.1")),
                                ("tabicl_v2", ("bsd-3-clause", "bsd-3-clause"))):
            case = dict(zip(("code_repo", "code_revision", "checkpoint_repo", "checkpoint_revision"),
                            nq.MODEL_SOURCES[model]), model=model)
            self.assertEqual(runtime._component_license_ids(case), expected)
            for key in ("code_revision", "checkpoint_revision"):
                with self.assertRaises(nq.QualificationError):
                    runtime._component_license_ids({**case, key: "0" * 40})
        with self.assertRaises(nq.QualificationError):
            runtime._component_license_ids({"model": "unknown"})


if __name__ == "__main__":
    unittest.main()

