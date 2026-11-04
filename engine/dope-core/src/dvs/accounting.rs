//! Private byte accounting and empirical privacy diagnostics.
use super::receipt::{ByteReceipt, PartReceipt};
use super::{ensure, ArtifactInputs, DvsError, PrivacyInputs};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn account_bytes(input: &ArtifactInputs) -> Result<ByteReceipt, DvsError> {
    ensure(
        !input.packet.is_empty() && !input.public_schema.is_empty(),
        "missing_artifact_part",
    )?;
    let (mut payload, mut shared, mut private) = (0u64, 0u64, 0u64);
    let mut parts = Vec::new();
    for (scope, blobs) in [
        ("packet", vec![input.packet.as_slice()]),
        ("public_schema", vec![input.public_schema.as_slice()]),
        (
            "mandatory_sidecar",
            input.mandatory_sidecars.iter().map(Vec::as_slice).collect(),
        ),
        (
            "shared_decoder",
            input
                .shared_decoder_parts
                .iter()
                .map(Vec::as_slice)
                .collect(),
        ),
        (
            "owner_private",
            input
                .owner_private_parts
                .iter()
                .map(Vec::as_slice)
                .collect(),
        ),
    ] {
        for bytes in blobs {
            ensure(!bytes.is_empty(), "empty_artifact_part")?;
            let n = u64::try_from(bytes.len()).map_err(|_| DvsError("byte_count_overflow"))?;
            let target = if scope == "shared_decoder" {
                &mut shared
            } else if scope == "owner_private" {
                &mut private
            } else {
                &mut payload
            };
            *target = target
                .checked_add(n)
                .ok_or(DvsError("byte_count_overflow"))?;
            parts.push(PartReceipt {
                scope,
                bytes: n,
                sha256: hash(bytes),
            });
        }
    }
    let closure = payload
        .checked_add(shared)
        .ok_or(DvsError("byte_count_overflow"))?;
    Ok(ByteReceipt {
        payload_bytes: payload,
        shared_bytes: shared,
        sampling_closure_bytes: closure,
        owner_private_dependency_bytes: private,
        parts,
        clean_room_sampling_state: "not_run",
    })
}

pub fn privacy_diagnostics(p: &PrivacyInputs) -> Result<Value, DvsError> {
    p.pins.validate()?;
    ensure(
        !p.real_fit.is_empty() && !p.synthetic.is_empty() && !p.real_holdout.is_empty(),
        "empty_privacy_input",
    )?;
    let width = p.real_fit[0].len();
    ensure(width > 0, "invalid_privacy_shape")?;
    for rows in [&p.real_fit, &p.synthetic, &p.real_holdout] {
        for row in rows {
            ensure(
                row.len() == width && row.iter().all(|v| v.is_finite()),
                "invalid_privacy_shape",
            )?;
        }
    }
    let comparisons = p
        .synthetic
        .len()
        .checked_add(p.real_holdout.len())
        .and_then(|n| n.checked_mul(p.real_fit.len()))
        .ok_or(DvsError("privacy_budget_exceeded"))?;
    ensure(
        p.max_distance_pairs > 0 && (comparisons as u64) <= p.max_distance_pairs,
        "privacy_budget_exceeded",
    )?;
    let nearest = |row: &[f64]| -> Result<f64, DvsError> {
        let mut best = f64::INFINITY;
        for reference in &p.real_fit {
            let mut d = 0.0_f64;
            for (&a, &b) in row.iter().zip(reference) {
                let delta = a - b;
                ensure(delta.is_finite(), "nonfinite_privacy_distance")?;
                d = d.hypot(delta);
            }
            ensure(d.is_finite(), "nonfinite_privacy_distance")?;
            best = best.min(d);
        }
        Ok(best)
    };
    let syn: Vec<_> = p
        .synthetic
        .iter()
        .map(|row| nearest(row))
        .collect::<Result<_, _>>()?;
    let real: Vec<_> = p
        .real_holdout
        .iter()
        .map(|row| nearest(row))
        .collect::<Result<_, _>>()?;
    let median = |v: &[f64]| {
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        let m = s.len() / 2;
        if s.len().is_multiple_of(2) {
            s[m - 1] / 2.0 + s[m] / 2.0
        } else {
            s[m]
        }
    };
    let (a, b) = (median(&syn), median(&real));
    let ratio = if b > 0.0 {
        let r = a / b;
        ensure(r.is_finite(), "nonfinite_privacy_distance")?;
        Some(r)
    } else {
        None
    };
    Ok(
        json!({"status": if ratio.is_some() { "complete" } else { "uninformative" },
        "reason": if ratio.is_some() { None } else { Some("zero_real_reference_distance") },
        "method": "euclidean-prepared-nearest-v1", "exact_copies": syn.iter().filter(|&&d| d == 0.0).count(),
        "synthetic_rows": p.synthetic.len(), "real_reference_rows": p.real_fit.len(), "heldout_rows": p.real_holdout.len(),
        "distance_pairs": comparisons, "dcr_ratio": ratio, "formal_dp_state": "not_run"}),
    )
}
