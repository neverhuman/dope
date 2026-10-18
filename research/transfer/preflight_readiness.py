"""End-to-end CPU surrogate campaign readiness dry-run harness (0% GPU).

Exercises deterministic CPU preflight contracts for blocked campaign tasks:
- W01: schema/contract verification across all 3 declared modern auditor slots
  (kumo-tabular, tabicl, mitra) + classical baselines (CatBoost, Linear, MLP),
  verifying fail-closed behavior when external GPU weights are absent.
- E03 & E04: synthetic 1-table x 2-seed CPU calibration and M1 development
  comparison dry-run verifying that empirical attack calibration and M1
  retention confidence intervals compute deterministically on CPU with 0% GPU.
- E05, E06, E07: 100-table manifest hash verification, independent replay digest
  verification on fixture artifacts, and sealed-test split lock verification
  (official_tests_opened == False).

No external checkpoints, restricted sources, or sealed test labels are opened.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile

from research.transfer import (
    calibration_receipt,
    evidence,
    frontier_coverage,
    lineage,
    lineage_manifest,
    losses,
    metrics,
    pilot_admission,
    precommit,
    statistics,
)
from research.transfer.adapters.auditors import (
    conventional,
    kumo_tabular,
    mitra,
    tabicl,
)
from research.transfer.adapters.base import (
    AdapterUnavailable,
    ModelIdentity,
    ResourceLimits,
)
from research.transfer.privacy import attacks

MODERN_AUDITOR_SLOTS = ("kumo-tabular", "tabicl", "mitra")
CLASSICAL_BASELINE_SLOTS = ("CatBoost", "Linear", "MLP")


def _pin(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def verify_w01_auditor_contracts(repo_root: Path) -> dict:
    """Verify W01 modern + classical auditor contracts and fail-closed absent weights."""
    repo_root = Path(repo_root).resolve()
    auditors_doc = json.loads((repo_root / "research/transfer/registries/auditors.json").read_text())
    models_doc = json.loads((repo_root / "research/transfer/registries/models.json").read_text())
    kumo_lock = json.loads((repo_root / "research/transfer/locks/kumo-tabular.json").read_text())
    other_lock = json.loads((repo_root / "research/transfer/locks/other-auditors.json").read_text())

    auditor_ids = {entry["id"]: entry for entry in auditors_doc["entries"]}
    model_names = {entry["name"]: entry for entry in models_doc["entries"]}

    modern_map = {
        "kumo-tabular": ("kumo_tabular", kumo_lock["status"], kumo_tabular.KumoRecipe("regression")),
        "tabicl": ("tabicl_v2", other_lock["status"], tabicl.TabICLRecipe("regression")),
        "mitra": (
            "mitra_v2",
            other_lock["status"],
            mitra.MitraRecipe("classification", config_sha256=_pin("mitra-cfg"), class_order=("0", "1")),
        ),
    }
    classical_map = {
        "CatBoost": (
            "catboost",
            conventional.ConventionalLearnerRecipe(
                "catboost", "regression", (("depth", 4),), selection_sha256=_pin("cb-sel"), precision="float32"
            ),
        ),
        "Linear": (
            "linear",
            conventional.ConventionalLearnerRecipe("linear", "regression", (("fit_intercept", True),)),
        ),
        "MLP": (
            "independent_neural",
            conventional.ConventionalLearnerRecipe(
                "mlp", "regression", (("max_iter", 20),), selection_sha256=_pin("mlp-sel")
            ),
        ),
    }

    modern_verified = {}
    for slot, (reg_id, lock_status, recipe) in modern_map.items():
        if reg_id not in auditor_ids or reg_id not in model_names:
            raise ValueError(f"missing_modern_auditor_slot:{slot}")
        if lock_status != "unqualified":
            raise ValueError(f"unexpected_lock_qualification:{slot}")
        modern_verified[slot] = {
            "registry_id": reg_id,
            "lock_status": lock_status,
            "recipe_sha256": recipe.sha256,
            "preprocessing_sha256": recipe.preprocessing_sha256,
            "output_sha256": recipe.output_sha256,
        }

    classical_verified = {}
    for slot, (model_name, recipe) in classical_map.items():
        if model_name not in model_names:
            raise ValueError(f"missing_classical_baseline_slot:{slot}")
        classical_verified[slot] = {
            "model_name": model_name,
            "family": recipe.family,
            "recipe_sha256": recipe.sha256,
            "preprocessing_sha256": recipe.preprocessing_sha256,
            "output_sha256": recipe.output_sha256,
        }

    # Verify fail-closed behavior when external GPU weights/checkpoints are absent on disk.
    tmp_base = repo_root / "target/vnext/preflight-w01"
    tmp_base.mkdir(parents=True, exist_ok=True)
    fail_closed_slots = []
    with tempfile.TemporaryDirectory(dir=tmp_base) as tmp:
        root = Path(tmp)
        code = root / "code.bin"
        dep = root / "dep.bin"
        lic_code = root / "lic_code.bin"
        lic_weights = root / "lic_weights.bin"
        code.write_bytes(b"synthetic-code")
        dep.write_bytes(b"synthetic-dep")
        lic_code.write_bytes(b"synthetic-lic-code")
        lic_weights.write_bytes(b"synthetic-lic-weights")
        missing_weights = root / "missing_vendor_checkpoint.safetensors"

        for slot, (reg_id, _, recipe) in modern_map.items():
            mode = "frozen_context"
            rev = (
                kumo_tabular.CODE_REVISION
                if slot == "kumo-tabular"
                else (tabicl.CODE_REVISION if slot == "tabicl" else mitra.AUTOGLUON_REVISION)
            )
            ident = ModelIdentity(
                name=reg_id,
                mode=mode,
                code_revision=rev,
                code_sha256=conventional.file_sha256(code),
                weights_sha256=_pin(f"absent-weights-{slot}"),
                dependencies=(("pkg", "1.0", conventional.file_sha256(dep)),),
                licenses=(
                    ("code", "Apache-2.0", conventional.file_sha256(lic_code)),
                    ("weights", "Apache-2.0", conventional.file_sha256(lic_weights)),
                ),
                preprocessing_sha256=recipe.preprocessing_sha256,
                recipe_sha256=recipe.sha256,
                output_aggregation_sha256=recipe.output_sha256,
                environment_sha256=_pin(f"env-{slot}"),
                precision=recipe.precision,
                context_cap=1024,
                context_policy="all",
                query_policy="batch",
                query_batch_size=256,
                order_sensitive=True,
                determinism="deterministic",
                resources=ResourceLimits(30.0, 1024 * 1024, 0),
            )
            artifacts = conventional.OfflineArtifacts(
                code=code,
                dependencies=(dep,),
                licenses=(lic_code, lic_weights),
                weights=missing_weights,
            )
            try:
                artifacts.verify(ident, version_reader=lambda _: "1.0")
            except AdapterUnavailable:
                fail_closed_slots.append(slot)
            else:
                raise AssertionError(f"Expected fail-closed AdapterUnavailable for {slot}")

    return {
        "task_id": "W01",
        "modern_auditors": tuple(modern_verified),
        "classical_baselines": tuple(classical_verified),
        "modern_contracts": modern_verified,
        "classical_contracts": classical_verified,
        "fail_closed_absent_weights_slots": tuple(fail_closed_slots),
        "fail_closed_without_external_weights": tuple(fail_closed_slots) == MODERN_AUDITOR_SLOTS,
        "gpu_utilization_percent": 0,
    }


def _single_calibration_and_m1_pass(repo_root: Path) -> dict:
    repo_root = Path(repo_root).resolve()
    e03_manifest = json.loads((repo_root / "research/transfer/manifests/e03-pilot.json").read_text())
    if e03_manifest["execution_authorized"] is not False or e03_manifest["panel_pass"] is not False:
        raise ValueError("e03_pilot_must_remain_fail_closed")

    # E03 synthetic 1-table x 2-seed empirical attack calibration on CPU (0% GPU)
    recipe_pin = _pin("e03-synthetic-attack-recipe-v1")
    dev_view = attacks.ScoreView(
        members=(0.82, 0.76, 0.91, 0.68),
        nonmembers=(0.12, 0.24, 0.31, 0.19),
        member_ids=tuple(_pin(f"dev-m-{i}") for i in range(4)),
        nonmember_ids=tuple(_pin(f"dev-n-{i}") for i in range(4)),
        units=tuple(_pin(f"dev-u-{i}") for i in range(8)),
        role="attack_development",
        recipe_sha256=recipe_pin,
        input_sha256=_pin("dev-input-table-01"),
    )
    eval_view = attacks.ScoreView(
        members=(0.79, 0.85, 0.73, 0.88),
        nonmembers=(0.15, 0.28, 0.22, 0.18),
        member_ids=tuple(_pin(f"eval-m-{i}") for i in range(4)),
        nonmember_ids=tuple(_pin(f"eval-n-{i}") for i in range(4)),
        units=tuple(_pin(f"eval-u-{i}") for i in range(8)),
        role="attack_evaluation",
        recipe_sha256=recipe_pin,
        input_sha256=_pin("eval-input-table-01"),
    )
    policy = attacks.DisclosurePolicy(target_fpr=0.25, alpha=0.05, comparisons=2, attacker_prior=0.5)
    cal = attacks.AttackCalibration.freeze(dev_view, policy)
    threat = attacks.DisclosureThreatModel(
        access="row_only",
        protected_unit="row",
        artifact_inventory_sha256=_pin("inv"),
        decoder_sha256=_pin("dec"),
        ledger_sha256=_pin("led"),
        auxiliary_sha256=_pin("aux"),
        sample_budget=8,
        reference_rows=8,
    )
    ev = cal.evaluate(eval_view, threat)

    proto = calibration_receipt.development_protocol()
    cal_body = {
        "format": calibration_receipt.FORMAT,
        "version": calibration_receipt.VERSION,
        "protocol_id": proto["protocol_id"],
        "protocol_digest": evidence.digest(proto),
        "repetitions": 2,
        "bootstrap_draws": 200,
        "frozen_for_confirmation": False,
        "manifest_digest": _pin("e03-dev-manifest"),
        "tolerances_frozen_before_outcomes": True,
        "mcse_reported": True,
        "wilson_reported": True,
    }
    cal_record = {**cal_body, "binding_digest": evidence.digest(cal_body)}
    cal_decision = calibration_receipt.verify_calibration_receipt(cal_record)

    # E04 synthetic 1-table x 2-seed M1 development comparison dry-run on CPU (0% GPU)
    scale = losses.TrainingScale(0.5)
    margin = metrics.Margin(relative=0.10, absolute_scaled=0.02)
    q_ids = ("q0", "q1", "q2", "q3")
    c_ids = ("c0", "c0", "c1", "c1")
    runs = []
    for seed_idx, (dope_synth, copula_synth) in enumerate(
        (((0.21, 0.25, 0.19, 0.23), (0.34, 0.39, 0.31, 0.36)),
         ((0.22, 0.24, 0.20, 0.22), (0.35, 0.38, 0.33, 0.37)))
    ):
        real_losses = (0.20, 0.24, 0.18, 0.22)
        null_losses = (0.80, 0.85, 0.78, 0.82)
        for fam_id, task_id in (("fam-a", "table-01"), ("fam-b", "table-02")):
            for method_id, synth_losses in (("dope", dope_synth), ("gaussian_copula_control", copula_synth)):
                pairs = losses.PairedLosses(
                    query_ids=q_ids,
                    cluster_ids=c_ids,
                    real=real_losses,
                    synthetic=synth_losses,
                    null=null_losses,
                    loss_space=losses.LossSpace.ORIGINAL_MSE,
                )
                runs.append(
                    statistics.EvaluationRun(
                        family_id=fam_id,
                        task_id=task_id,
                        method_id=method_id,
                        fit_id=f"fit-seed-{seed_idx}",
                        sample_id="sample-0",
                        learner_seed_id="learner-0",
                        pairs=pairs,
                        scale=scale,
                        margin=margin,
                        global_seed_namespace="campaign-dev-seed",
                        global_seed_id=f"seed-{seed_idx}",
                    )
                )
    design = statistics.CrossedDesign(
        runs=tuple(runs),
        required_methods=("dope", "gaussian_copula_control"),
        scheduled_keys=tuple(r.key for r in runs),
    )
    point = statistics.paired_family_effect(design, endpoint="z")
    boot = statistics.bootstrap_effects(design, endpoint="z", estimand="method_new_source", draws=200, seed=1729)
    ucb = statistics.simultaneous_ucb(boot, alpha=0.05)
    pins = precommit.expected_pins(repo_root)

    return {
        "tasks": ("E03", "E04"),
        "gpu_utilization_percent": 0,
        "e03_calibration_sha256": cal.sha256,
        "e03_evidence_sha256": ev.sha256,
        "e03_oriented_auc": ev.value["oriented_auc"],
        "e03_auc_bounds": tuple(ev.value["auc_bounds"]),
        "e03_release_authorized": ev.value["release_authorized"],
        "e03_receipt_authenticates": bool(cal_decision["authenticates"]),
        "e03_receipt_qualifies": bool(cal_decision["qualifies"]),
        "e04_point_estimates": point["estimates"],
        "e04_ucb_bounds": {
            k: (v["lower"], v["estimate"], v["upper"]) for k, v in ucb["bounds"].items()
        },
        "e04_precommit_pin_count": len(pins),
    }


def run_e03_e04_cpu_calibration_and_m1_dry_run(repo_root: Path) -> dict:
    """Run two passes of E03/E04 CPU calibration + M1 CIs and assert exact determinism."""
    first = _single_calibration_and_m1_pass(repo_root)
    second = _single_calibration_and_m1_pass(repo_root)
    if first != second:
        raise AssertionError("non_deterministic_e03_e04_cpu_dry_run")
    return {**first, "deterministic_repeat_verified": True}


def verify_e05_e06_e07_manifest_replay_and_sealed_locks(repo_root: Path) -> dict:
    """Verify E05 100-table manifest hashes, E06 fixture replay digest, and E07 sealed test locks."""
    repo_root = Path(repo_root).resolve()

    # E05: 100-table classical panel + vNext datasets manifest hash and custody verification
    datasets_path = repo_root / "research/transfer/manifests/datasets.json"
    datasets_sha = hashlib.sha256(datasets_path.read_bytes()).hexdigest()
    expected_datasets_sha = precommit.expected_pins(repo_root)["research/transfer/manifests/datasets.json"]
    if datasets_sha != expected_datasets_sha:
        raise ValueError("datasets_manifest_hash_mismatch")
    datasets_doc = json.loads(datasets_path.read_text())
    entries = datasets_doc["entries"]
    if len(entries) < 100 or datasets_doc["protocol_frozen"] is not False:
        raise ValueError("datasets_manifest_invalid")
    if any(e.get("sealed_test_opened") is not False or e.get("headline_eligible") is not False for e in entries):
        raise ValueError("sealed_test_or_headline_prematurely_opened")

    panel_path = repo_root / "research/benchmark/results/retained-classical-validation/panel.json"
    panel_doc = json.loads(panel_path.read_text())
    if panel_doc["complete_classical_lineages"] != 100:
        raise ValueError("classical_100_table_lineage_count_mismatch")
    exec_records = panel_doc["execution_records"]
    if any(rec.get("official_tests_opened") is not False for rec in exec_records.values()):
        raise ValueError("official_tests_opened_in_panel")

    reg = lineage.load_registry(repo_root / "research/transfer/registries/sources.json")
    custody_hold_enforced = False
    try:
        lineage_manifest.admit_source_manifest({"sources": []}, reg, signing_key=b"k" * 32)
    except lineage_manifest.LineageError:
        custody_hold_enforced = True
    if not custody_hold_enforced:
        raise AssertionError("Expected D18 custody/schema hold on empty source manifest")

    # E06: Independent replay digest verification on fixture artifacts
    from research.transfer.tests.test_evidence import ReceiptFixture

    fixture = ReceiptFixture(confirmation=False)
    verified_receipt = fixture.verify()
    replay_digest = verified_receipt.receipt_hash
    tampered = replace_receipt_loss(fixture.receipt, 999.0)
    tamper_rejected = False
    try:
        evidence.verify_receipt(evidence.UnverifiedReceipt(evidence.canonical_json(tampered)), fixture.root, fixture.policy())
    except evidence.EvidenceError:
        tamper_rejected = True
    if not tamper_rejected:
        raise AssertionError("Expected tampered replay receipt to fail verification")
    if frontier_coverage._admissible_uncertainty({}) is not False:
        raise AssertionError("Expected empty uncertainty record to be inadmissible")

    # E07: Sealed-test split lock verification (official_tests_opened == False)
    obs_proto_path = repo_root / "examples/public-regression/bikeshare-observational/protocol.json"
    obs_proto = json.loads(obs_proto_path.read_text())
    official_tests_opened = (
        any(e.get("sealed_test_opened") is not False for e in entries)
        or any(rec.get("official_tests_opened") is not False for rec in exec_records.values())
        or obs_proto["selection"]["rows_consumed_for_selection"] != 0
    )
    if official_tests_opened is not False:
        raise ValueError("official_tests_opened_lock_violated")

    return {
        "tasks": ("E05", "E06", "E07"),
        "gpu_utilization_percent": 0,
        "e05_datasets_manifest_sha256": datasets_sha,
        "e05_datasets_entry_count": len(entries),
        "e05_classical_100_table_lineages": panel_doc["complete_classical_lineages"],
        "e05_custody_hold_enforced": custody_hold_enforced,
        "e06_independent_replay_receipt_sha256": replay_digest,
        "e06_tamper_rejected": tamper_rejected,
        "e07_official_tests_opened": official_tests_opened,
        "e07_observational_status": obs_proto["status"],
    }


def replace_receipt_loss(receipt_doc: dict, bad_real_loss: float) -> dict:
    clone = json.loads(json.dumps(receipt_doc))
    clone["losses"]["real"] = bad_real_loss
    return clone
