/// Multi-target execution is held until a qualified single-task artifact from
/// the frozen supported panel is available. This guard deliberately does not
/// create a campaign or fall back to an unqualified panel.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg(test)]
pub(crate) struct SingleTaskQualificationEvidence {
    pub receipt_sha256: String,
    pub supported_panel_sha256: String,
    pub artifact_qualified: bool,
}

#[cfg(test)]
pub(crate) fn multi_target_admission_blocker(
    evidence: Option<&SingleTaskQualificationEvidence>,
    expected_panel_sha256: &str,
) -> Option<&'static str> {
    let Some(evidence) = evidence else {
        return Some("single_task_qualified_artifact_receipt_missing");
    };
    if !evidence.artifact_qualified {
        return Some("single_task_artifact_not_qualified");
    }
    if !is_sha256_hex(&evidence.receipt_sha256) {
        return Some("single_task_receipt_identity_invalid");
    }
    if !is_sha256_hex(&evidence.supported_panel_sha256)
        || evidence.supported_panel_sha256 != expected_panel_sha256
    {
        return Some("single_task_supported_panel_mismatch");
    }
    None
}

#[cfg(test)]
fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod multi_target_admission_tests {
    use super::{multi_target_admission_blocker, SingleTaskQualificationEvidence};

    const PANEL: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    #[test]
    fn multi_target_requires_qualified_single_task_receipt_on_frozen_panel() {
        assert_eq!(
            multi_target_admission_blocker(None, PANEL),
            Some("single_task_qualified_artifact_receipt_missing")
        );
        let unqualified = SingleTaskQualificationEvidence {
            receipt_sha256: "b".repeat(64),
            supported_panel_sha256: PANEL.to_owned(),
            artifact_qualified: false,
        };
        assert_eq!(
            multi_target_admission_blocker(Some(&unqualified), PANEL),
            Some("single_task_artifact_not_qualified")
        );
        let qualified = SingleTaskQualificationEvidence {
            receipt_sha256: "b".repeat(64),
            supported_panel_sha256: PANEL.to_owned(),
            artifact_qualified: true,
        };
        assert_eq!(multi_target_admission_blocker(Some(&qualified), PANEL), None);
        assert_eq!(
            multi_target_admission_blocker(Some(&qualified), &"c".repeat(64)),
            Some("single_task_supported_panel_mismatch")
        );
    }
}
