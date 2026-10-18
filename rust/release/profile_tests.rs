use super::*;
use serde_json::{Value, json};

fn accounting() -> ReleaseAccounting {
    ReleaseAccounting {
        artifact_inventory_sha256: "1".repeat(64),
        release_history_sha256: "2".repeat(64),
        source_derived_lengths_sha256: "3".repeat(64),
        source_derived_statuses_sha256: "4".repeat(64),
        source_derived_metrics_sha256: "5".repeat(64),
        release_count: 2,
    }
}

fn evidence(axis: Axis) -> EvidenceIdentity {
    let scope = accounting().identity().unwrap();
    let mut value = json!({"axis":axis,"receipt_sha256":"6".repeat(64),"accounting_sha256":scope});
    for key in match axis {
        Axis::Utility => vec![],
        Axis::EmpiricalPrivacy => vec!["threat_model_sha256", "attack_power_sha256"],
        Axis::FormalDp => vec!["mechanism_sha256", "accountant_sha256", "proof_sha256"],
        Axis::LegalRelease => vec!["rights_owner_sha256"],
    } {
        value[key] = json!("7".repeat(64));
    }
    if axis == Axis::FormalDp {
        value["epsilon"] = json!(1.);
        value["delta"] = json!(0.00001);
    }
    serde_json::from_value(value).unwrap()
}

// Fixture-only allowlist of exact synthetic identities, never a product verifier.
struct FixtureVerifier;
impl ReleaseEvidenceVerifier for FixtureVerifier {
    fn verify(&self, receipt: &EvidenceIdentity) -> bool {
        *receipt == evidence(receipt.axis())
    }
}

fn assessed(axes: &[Axis]) -> ReleaseProfile {
    let mut profile = ReleaseProfile::with_accounting(accounting()).unwrap();
    for axis in axes {
        profile = profile
            .begin(*axis)
            .complete(evidence(*axis), &FixtureVerifier)
            .unwrap();
    }
    profile
}

#[test]
fn defaults_pending_and_small_artifacts_cannot_authorize_any_surface() {
    let profile = ReleaseProfile::default();
    for surface in [
        ReleaseSurface::Cli,
        ReleaseSurface::Service,
        ReleaseSurface::Ui,
        ReleaseSurface::Report,
    ] {
        let view = profile.view(surface, false, &NoReleaseEvidence).unwrap();
        assert!(!view.release_authorized);
        assert_eq!(view.label, "local_research");
    }
    assert!(
        profile
            .view(ReleaseSurface::Export, false, &NoReleaseEvidence)
            .is_err()
    );
    for axis in [
        Axis::Utility,
        Axis::EmpiricalPrivacy,
        Axis::FormalDp,
        Axis::LegalRelease,
    ] {
        assert!(
            profile
                .begin(axis)
                .view(ReleaseSurface::Export, false, &NoReleaseEvidence)
                .is_err()
        );
        assert!(profile.complete(evidence(axis), &FixtureVerifier).is_err());
    }
    let utility = assessed(&[Axis::Utility]);
    assert_eq!(
        utility.empirical_privacy_state,
        EmpiricalPrivacyState::Unassessed
    );
    assert_eq!(utility.formal_dp_state, FormalDpState::Unassessed);
    assert_eq!(utility.legal_release_state, LegalReleaseState::Unassessed);
    assert!(
        utility
            .view(ReleaseSurface::Export, false, &FixtureVerifier)
            .is_err()
    );
    let mut bytes = serde_json::to_value(utility).unwrap();
    bytes["artifact_bytes"] = json!(1);
    assert!(
        ReleaseProfile::decode(&serde_json::to_vec(&bytes).unwrap(), &FixtureVerifier).is_err()
    );
}

#[test]
fn all_axis_subsets_remain_independent_and_all_surfaces_agree() {
    let axes = [
        Axis::Utility,
        Axis::EmpiricalPrivacy,
        Axis::FormalDp,
        Axis::LegalRelease,
    ];
    for mask in 0..16 {
        let selected = axes
            .iter()
            .enumerate()
            .filter_map(|(i, axis)| (mask & (1 << i) != 0).then_some(*axis))
            .collect::<Vec<_>>();
        let profile = assessed(&selected);
        for require_dp in [false, true] {
            let authorized = mask & 11 == 11 && (!require_dp || mask & 4 != 0);
            for surface in [
                ReleaseSurface::Cli,
                ReleaseSurface::Service,
                ReleaseSurface::Ui,
                ReleaseSurface::Report,
            ] {
                assert_eq!(
                    profile
                        .view(surface, require_dp, &FixtureVerifier)
                        .unwrap()
                        .release_authorized,
                    authorized
                );
            }
            assert_eq!(
                profile
                    .view(ReleaseSurface::Export, require_dp, &FixtureVerifier)
                    .is_ok(),
                authorized
            );
        }
    }
}

#[test]
fn evidence_is_not_a_json_flag_and_must_be_reverified() {
    let pending = ReleaseProfile::with_accounting(accounting())
        .unwrap()
        .begin(Axis::FormalDp);
    assert!(
        pending
            .complete(evidence(Axis::FormalDp), &NoReleaseEvidence)
            .is_err()
    );
    let positive = assessed(&[Axis::FormalDp]);
    let encoded = serde_json::to_vec(&positive).unwrap();
    assert!(ReleaseProfile::decode(&encoded, &FixtureVerifier).is_ok());
    assert!(ReleaseProfile::decode(&encoded, &NoReleaseEvidence).is_err());
    assert!(
        positive
            .view(ReleaseSurface::Report, false, &NoReleaseEvidence)
            .is_err()
    );
    assert!(
        positive
            .check_claim("formal_dp_verified", Some(1.), &FixtureVerifier)
            .is_ok()
    );
    assert!(
        positive
            .check_claim("formal_dp_verified", Some(2.), &FixtureVerifier)
            .is_err()
    );
    let value = serde_json::to_value(positive).unwrap();
    for key in [
        "mechanism_sha256",
        "accountant_sha256",
        "proof_sha256",
        "accounting_sha256",
    ] {
        let mut changed = value.clone();
        changed["evidence"][0].as_object_mut().unwrap().remove(key);
        assert!(
            ReleaseProfile::decode(&serde_json::to_vec(&changed).unwrap(), &FixtureVerifier)
                .is_err()
        );
    }
}

#[test]
fn revocation_failures_inconclusive_and_new_scope_block_export() {
    let all = assessed(&[
        Axis::Utility,
        Axis::EmpiricalPrivacy,
        Axis::FormalDp,
        Axis::LegalRelease,
    ]);
    for axis in [
        Axis::Utility,
        Axis::EmpiricalPrivacy,
        Axis::FormalDp,
        Axis::LegalRelease,
    ] {
        let pending = all.begin(axis);
        assert!(
            pending
                .view(ReleaseSurface::Export, false, &FixtureVerifier)
                .is_err()
        );
        let failed = pending.reject(axis).unwrap();
        assert!(
            failed
                .view(ReleaseSurface::Export, false, &FixtureVerifier)
                .is_err()
        );
        assert!(failed.complete(evidence(axis), &FixtureVerifier).is_err());
    }
    let inconclusive = all
        .begin(Axis::EmpiricalPrivacy)
        .empirical_inconclusive()
        .unwrap();
    assert!(
        inconclusive
            .view(ReleaseSurface::Export, false, &FixtureVerifier)
            .is_err()
    );
    let mut changed = accounting();
    changed.release_count += 1;
    let reset = ReleaseProfile::with_accounting(changed).unwrap();
    assert!(reset.evidence.is_empty());
    assert!(
        reset
            .begin(Axis::Utility)
            .complete(evidence(Axis::Utility), &FixtureVerifier)
            .is_err()
    );
    let mut forged = serde_json::to_value(all).unwrap();
    forged["accounting"]["source_derived_metrics_sha256"] = json!("8".repeat(64));
    assert!(
        ReleaseProfile::decode(&serde_json::to_vec(&forged).unwrap(), &FixtureVerifier).is_err()
    );
}

#[test]
fn labels_epsilon_and_errors_never_echo_source_canaries() {
    let profile = assessed(&[
        Axis::Utility,
        Axis::EmpiricalPrivacy,
        Axis::FormalDp,
        Axis::LegalRelease,
    ]);
    for label in [
        "HIPAA",
        "anonymous",
        "de-identified",
        "safe-share",
        "PRIVATE_HEADER_97531",
        "/private/source.csv",
        "-987654.25",
    ] {
        let error = profile
            .check_claim(label, None, &FixtureVerifier)
            .unwrap_err();
        assert!(!error.to_string().contains(label));
        let bytes = format!("{{\"{label}\":true}}");
        let error = ReleaseProfile::decode(bytes.as_bytes(), &FixtureVerifier).unwrap_err();
        assert!(!error.to_string().contains(label));
        let mut accounting = accounting();
        accounting.source_derived_lengths_sha256 = label.into();
        let error = ReleaseProfile::with_accounting(accounting).unwrap_err();
        assert!(!error.to_string().contains(label));
        assert!(
            !serde_json::to_string(
                &profile
                    .view(ReleaseSurface::Report, false, &FixtureVerifier)
                    .unwrap()
            )
            .unwrap()
            .contains(label)
        );
    }
    assert!(
        ReleaseProfile::default()
            .check_claim("local_research", Some(1.), &NoReleaseEvidence)
            .is_err()
    );
    for epsilon in [f64::NAN, f64::INFINITY, 0., -1.] {
        assert!(
            profile
                .check_claim("formal_dp_verified", Some(epsilon), &FixtureVerifier)
                .is_err()
        );
    }
}

#[test]
fn strict_decoding_rejects_missing_wrong_duplicate_and_extra_fields() {
    let value = serde_json::to_value(ReleaseProfile::default()).unwrap();
    for field in value.as_object().unwrap().keys() {
        let mut missing = value.clone();
        missing.as_object_mut().unwrap().remove(field);
        assert!(
            ReleaseProfile::decode(&serde_json::to_vec(&missing).unwrap(), &NoReleaseEvidence)
                .is_err()
        );
    }
    for invalid in [json!(true), json!("pass"), json!(null), json!(1)] {
        for field in [
            "utility_qualification_state",
            "empirical_privacy_state",
            "formal_dp_state",
            "legal_release_state",
        ] {
            let mut changed = value.clone();
            changed[field] = invalid.clone();
            assert!(
                ReleaseProfile::decode(&serde_json::to_vec(&changed).unwrap(), &NoReleaseEvidence)
                    .is_err()
            );
        }
    }
    let mut changed = value;
    changed["utility_qualification_state"] = json!("passed");
    assert!(
        ReleaseProfile::decode(&serde_json::to_vec(&changed).unwrap(), &NoReleaseEvidence).is_err()
    );
    let bytes = serde_json::to_string(&ReleaseProfile::default())
        .unwrap()
        .replacen("\"version\":1", "\"version\":1,\"version\":1", 1);
    assert!(ReleaseProfile::decode(bytes.as_bytes(), &NoReleaseEvidence).is_err());
    assert!(ReleaseProfile::decode(&vec![b' '; 65_537], &NoReleaseEvidence).is_err());
    let mut duplicate = serde_json::to_value(assessed(&[Axis::Utility])).unwrap();
    let receipt = duplicate["evidence"][0].clone();
    duplicate["evidence"].as_array_mut().unwrap().push(receipt);
    assert!(
        ReleaseProfile::decode(&serde_json::to_vec(&duplicate).unwrap(), &FixtureVerifier).is_err()
    );
}

#[test]
fn current_cli_export_blocks_before_source_access_or_output_creation() {
    let private = std::path::Path::new("/private/PRIVATE_HEADER_97531.csv");
    let result = crate::release::convert_release(
        private,
        private,
        crate::model::Task::Regression,
        private,
        &crate::release::ConversionOptions::default(),
    );
    assert_eq!(
        result.unwrap_err().to_string(),
        "invalid data: release_authorization_required"
    );
}

#[test]
fn rust_and_python_accounting_use_the_same_canonical_sha256() {
    // Independently calculated by Python hashlib over sorted compact JSON.
    assert_eq!(
        accounting().identity().unwrap(),
        "f425715568d69b6a31bf1e120be0e08ae2c9233fc5149191dfbc25ea0a1ce3e1"
    );
    let schema: Value =
        serde_json::from_str(include_str!("../../schemas/release-profile-v1.schema.json")).unwrap();
    assert_eq!(
        schema["properties"]["utility_qualification_state"]["default"],
        "unassessed"
    );
}
