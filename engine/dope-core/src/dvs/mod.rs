//! Downstream Value Suite: source-free paired evidence, never release authority.
//! Common coordinates are prepared upstream from real FIT only. This module
//! consumes frozen learner evidence; it never fits preprocessing or selects on
//! real TEST. All scheduled cells remain represented, including missing cells.
pub mod drivers;
pub mod permutation;
pub mod receipt;
pub mod roles;
pub mod selection;

use receipt::{CellReceipt, DriverPolicyBinding, DvsReceipt, FamilyReceipt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

pub const PROTOCOL: &str = "dope-dvs-v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DvsError(pub &'static str);
impl std::fmt::Display for DvsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for DvsError {}
pub(crate) fn ensure(ok: bool, reason: &'static str) -> Result<(), DvsError> {
    if ok {
        Ok(())
    } else {
        Err(DvsError(reason))
    }
}
pub(crate) fn finite_nonnegative(value: f64) -> Result<(), DvsError> {
    ensure(value.is_finite() && value >= 0.0, "invalid_dvs_number")
}
fn pin(value: &str) -> Result<(), DvsError> {
    ensure(
        value.len() == 64
            && value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "invalid_dvs_pin",
    )
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DataPins {
    pub fit_scope_sha256: String,
    pub real_test_sha256: String,
    pub prepared_view_sha256: String,
    pub groups_sha256: String,
    pub roster_sha256: String,
}
impl DataPins {
    fn validate(&self) -> Result<(), DvsError> {
        for p in [
            &self.fit_scope_sha256,
            &self.real_test_sha256,
            &self.prepared_view_sha256,
            &self.groups_sha256,
            &self.roster_sha256,
        ] {
            pin(p)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduledCell {
    pub cell_sha256: String,
    pub family_id: u32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureReason {
    NativeFailure,
    InvalidPrediction,
    GuardStop,
    UnsupportedTask,
    MissingEvidence,
}
impl FailureReason {
    fn code(&self) -> &'static str {
        match self {
            Self::NativeFailure => "native_failure",
            Self::InvalidPrediction => "invalid_prediction",
            Self::GuardStop => "guard_stop",
            Self::UnsupportedTask => "unsupported_task",
            Self::MissingEvidence => "missing_evidence",
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum CellOutcome {
    Complete {
        null_losses: Vec<f64>,
        real_losses: Vec<f64>,
        synthetic_losses: Vec<f64>,
        real_gain_lcb: f64,
        absolute_gain_floor: f64,
        relative_gain_floor: f64,
        driver_evidence: Option<Value>,
        wall_seconds: Option<f64>,
        fit_count: Option<u64>,
    },
    Failed {
        reason: FailureReason,
        wall_seconds: Option<f64>,
        fit_count: Option<u64>,
    },
    TimedOut {
        wall_seconds: Option<f64>,
        fit_count: Option<u64>,
    },
    Unsupported,
    Missing,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuditorCell {
    pub cell_sha256: String,
    pub family_id: u32,
    pub pins: DataPins,
    pub outcome: CellOutcome,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactInputs {
    pub packet: Vec<u8>,
    pub public_schema: Vec<u8>,
    pub mandatory_sidecars: Vec<Vec<u8>>,
    pub shared_decoder_parts: Vec<Vec<u8>>,
    pub owner_private_parts: Vec<Vec<u8>>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrivacyInputs {
    pub pins: DataPins,
    pub real_fit: Vec<Vec<f64>>,
    pub synthetic: Vec<Vec<f64>>,
    pub real_holdout: Vec<Vec<f64>>,
    pub max_distance_pairs: u64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DvsRequest {
    pub protocol: String,
    pub preparation_recipe: String,
    pub pins: DataPins,
    pub scheduled: Vec<ScheduledCell>,
    pub required_families: Vec<u32>,
    pub cells: Vec<AuditorCell>,
    /// Required for new receipt issuance; absence fails closed before cells or selection.
    pub declared_roles: Option<roles::DeclaredEvidenceRoles>,
    pub selection: Option<selection::SelectionEvidence>,
    pub artifacts: ArtifactInputs,
    pub privacy: Option<PrivacyInputs>,
}

fn mean(values: &[f64]) -> Result<f64, DvsError> {
    ensure(!values.is_empty(), "empty_dvs_losses")?;
    let mut out = 0.0;
    for &v in values {
        finite_nonnegative(v)?;
        out += v / values.len() as f64;
    }
    finite_nonnegative(out)?;
    Ok(out)
}

mod accounting;
pub use accounting::{account_bytes, privacy_diagnostics};

fn cell_receipt(
    cell: Option<&AuditorCell>,
    schedule: &ScheduledCell,
    pins: &DataPins,
) -> Result<CellReceipt, DvsError> {
    let mut out = CellReceipt {
        cell_sha256: schedule.cell_sha256.clone(),
        family_id: schedule.family_id,
        status: "missing",
        reason: Some("missing_cell"),
        query_rows: 0,
        null_loss: None,
        real_loss: None,
        synthetic_loss: None,
        raw_retention: None,
        driver: json!({"status": "not_run"}),
        wall_seconds: None,
        fit_count: None,
    };
    let Some(cell) = cell else {
        return Ok(out);
    };
    ensure(
        cell.family_id == schedule.family_id && &cell.pins == pins,
        "dvs_cell_identity_mismatch",
    )?;
    match &cell.outcome {
        CellOutcome::Complete {
            null_losses,
            real_losses,
            synthetic_losses,
            real_gain_lcb,
            absolute_gain_floor,
            relative_gain_floor,
            driver_evidence,
            wall_seconds,
            fit_count,
        } => {
            ensure(
                !null_losses.is_empty()
                    && null_losses.len() == real_losses.len()
                    && null_losses.len() == synthetic_losses.len(),
                "paired_loss_shape_mismatch",
            )?;
            finite_nonnegative(*absolute_gain_floor)?;
            finite_nonnegative(*relative_gain_floor)?;
            ensure(
                *absolute_gain_floor > 0.0 && *relative_gain_floor > 0.0,
                "invalid_real_gain_floor",
            )?;
            ensure(real_gain_lcb.is_finite(), "invalid_dvs_number")?;
            let (n, r, s) = (
                mean(null_losses)?,
                mean(real_losses)?,
                mean(synthetic_losses)?,
            );
            out.query_rows = null_losses.len();
            out.null_loss = Some(n);
            out.real_loss = Some(r);
            out.synthetic_loss = Some(s);
            out.wall_seconds = *wall_seconds;
            out.fit_count = *fit_count;
            let gain = n - r;
            if gain <= 0.0
                || gain < absolute_gain_floor.max(relative_gain_floor * n.abs())
                || *real_gain_lcb <= 0.0
            {
                out.status = "uninformative";
                out.reason = Some("uninformative_real_gain");
            } else {
                let retention = (n - s) / gain;
                ensure(retention.is_finite(), "nonfinite_retention")?;
                out.raw_retention = Some(retention);
                out.status = "complete";
                out.reason = None;
            }
            if let Some(e) = driver_evidence {
                for (key, expected) in [
                    ("query_sha256", &pins.real_test_sha256),
                    ("groups_sha256", &pins.groups_sha256),
                    ("prepared_view_sha256", &pins.prepared_view_sha256),
                ] {
                    ensure(
                        e[key].as_str() == Some(expected.as_str()),
                        "driver_identity_mismatch",
                    )?;
                }
                out.driver = drivers::evaluate(e);
                if out.status == "complete" && out.driver["status"] != "complete" {
                    out.status = "incomplete";
                    out.reason = Some("incomplete_driver_evidence");
                }
            } else if out.status == "complete" {
                out.status = "incomplete";
                out.reason = Some("missing_driver_evidence");
            }
        }
        CellOutcome::Failed {
            reason,
            wall_seconds,
            fit_count,
        } => {
            out.status = "failed";
            out.reason = Some(reason.code());
            out.wall_seconds = *wall_seconds;
            out.fit_count = *fit_count;
        }
        CellOutcome::TimedOut {
            wall_seconds,
            fit_count,
        } => {
            out.status = "timed_out";
            out.reason = Some("timed_out");
            out.wall_seconds = *wall_seconds;
            out.fit_count = *fit_count;
        }
        CellOutcome::Unsupported => {
            out.status = "unsupported";
            out.reason = Some("unsupported_task");
        }
        CellOutcome::Missing => {}
    }
    if let Some(w) = out.wall_seconds {
        finite_nonnegative(w)?;
    }
    Ok(out)
}

pub fn evaluate(input: &DvsRequest) -> Result<DvsReceipt, DvsError> {
    ensure(input.protocol == PROTOCOL, "unsupported_dvs_protocol")?;
    ensure(
        matches!(
            input.preparation_recipe.as_str(),
            "nc1-minmax-onehot-v1" | "legacy-real-fit-affine-target-v1"
        ),
        "unsupported_preparation_recipe",
    )?;
    input.pins.validate()?;
    let role_binding = input
        .declared_roles
        .as_ref()
        .ok_or(DvsError("missing_dvs_role_binding"))?
        .validate(&input.pins)?;
    ensure(
        !input.scheduled.is_empty() && !input.required_families.is_empty(),
        "empty_dvs_schedule",
    )?;
    let families: BTreeSet<_> = input.required_families.iter().copied().collect();
    ensure(
        families.len() == input.required_families.len(),
        "duplicate_dvs_family",
    )?;
    let mut scheduled = BTreeMap::new();
    for row in &input.scheduled {
        pin(&row.cell_sha256)?;
        ensure(
            families.contains(&row.family_id) && scheduled.insert(&row.cell_sha256, row).is_none(),
            "invalid_dvs_schedule",
        )?;
    }
    let mut actual = BTreeMap::new();
    for cell in &input.cells {
        pin(&cell.cell_sha256)?;
        cell.pins.validate()?;
        ensure(
            scheduled.contains_key(&cell.cell_sha256)
                && actual.insert(&cell.cell_sha256, cell).is_none(),
            "unexpected_or_duplicate_cell",
        )?;
    }
    let mut cells = Vec::new();
    for row in &input.scheduled {
        cells.push(cell_receipt(
            actual.get(&row.cell_sha256).copied(),
            row,
            &input.pins,
        )?);
    }
    let complete = cells.iter().filter(|c| c.status == "complete").count();
    let uninformative = cells.iter().filter(|c| c.status == "uninformative").count();
    let failed = cells.len() - complete - uninformative;
    let mut family_receipts = Vec::new();
    for id in families {
        let scheduled_cells = cells.iter().filter(|c| c.family_id == id).count();
        let values: Vec<f64> = cells
            .iter()
            .filter(|c| c.family_id == id)
            .filter_map(|c| c.raw_retention)
            .collect();
        let descriptive = if values.is_empty() {
            None
        } else {
            let v = values.iter().map(|v| v / values.len() as f64).sum::<f64>();
            ensure(v.is_finite(), "nonfinite_retention")?;
            Some(v)
        };
        family_receipts.push(FamilyReceipt {
            family_id: id,
            scheduled_cells,
            informative_cells: values.len(),
            descriptive_mean_retention: descriptive,
        });
    }
    let robust = if failed == 0 && family_receipts.iter().all(|f| f.informative_cells > 0) {
        family_receipts
            .iter()
            .filter_map(|f| f.descriptive_mean_retention)
            .reduce(f64::min)
    } else {
        None
    };
    let selection = match &input.selection {
        Some(e) => serde_json::to_value(selection::evaluate(e)?)
            .map_err(|_| DvsError("receipt_serialization_failed"))?,
        None => selection::not_run(),
    };
    let privacy = match &input.privacy {
        Some(p) => {
            ensure(p.pins == input.pins, "privacy_identity_mismatch")?;
            privacy_diagnostics(p)?
        }
        None => {
            json!({"status":"not_run", "reason":"privacy_not_supplied", "exact_copies":null, "dcr_ratio":null})
        }
    };
    Ok(DvsReceipt {
        protocol: PROTOCOL,
        preparation_recipe: input.preparation_recipe.clone(),
        status: if failed > 0
            || uninformative > 0
            || robust.is_none()
            || selection["status"] != "complete"
            || privacy["status"] != "complete"
        {
            "incomplete"
        } else {
            "complete"
        },
        scheduled_cells: cells.len(),
        complete_cells: complete,
        uninformative_cells: uninformative,
        failed_or_missing_cells: failed,
        cells,
        families: family_receipts,
        robust_retention: robust,
        selection,
        bytes: account_bytes(&input.artifacts)?,
        privacy,
        pins: input.pins.clone(),
        driver_policy_binding: DriverPolicyBinding::required(input.pins.clone()),
        role_binding,
        qualification_state: "not_run",
        formal_dp_state: "not_run",
        release_authorization_state: "not_run",
    })
}
