use super::exit_codes::{Failure, Result};
use super::io;
use super::registry::Limits;

/// Reject oversized decoded state before calling historical allocating readers.
pub fn budget(bytes: &[u8], limits: &Limits) -> Result<()> {
    if bytes.starts_with(b"DPK3") || bytes.starts_with(b"DPK2") {
        if bytes.len() < 56 {
            return Err(Failure::usage("artifact_invalid"));
        }
        let features = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        if features as usize >= dope_kernel::schema::MAX_COLUMNS {
            return Err(Failure::budget());
        }
        // Both binary generations share section framing; accounting takes DPK3 magic.
        let mut framed = bytes.to_vec();
        framed[..4].copy_from_slice(b"DPK3");
        let accounting = dope_kernel::codec::account_artifact(&framed)?;
        let decoded = accounting
            .sections
            .iter()
            .try_fold(0u64, |sum, section| {
                sum.checked_add(section.raw_bytes as u64)
            })
            .ok_or_else(Failure::budget)?;
        if decoded > limits.max_input_bytes {
            return Err(Failure::budget());
        }
    }
    if bytes.starts_with(b"DOPERAW1")
        || bytes.starts_with(b"DOPERAW2")
        || bytes.starts_with(b"DOPMIX01")
    {
        if bytes.len() < 56 {
            return Err(Failure::usage("artifact_invalid"));
        }
        let state_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let offset = state_len
            .checked_add(56)
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(Failure::budget)?;
        let payload = bytes
            .get(offset..)
            .ok_or_else(|| Failure::usage("artifact_invalid"))?;
        if !payload.starts_with(b"DPK3") {
            return Err(Failure::usage("artifact_invalid"));
        }
        budget(payload, limits)?;
    }
    if bytes.first() == Some(&b'{') || bytes.first().is_some_and(u8::is_ascii_whitespace) {
        let value = io::json(bytes)?;
        if value["format"] == "dope-kernel" && value["version"] == 1 {
            let width = value["shell"]["p"]
                .as_u64()
                .ok_or_else(|| Failure::usage("v1_shape_invalid"))?;
            if width == 0 || width >= dope_kernel::schema::MAX_COLUMNS as u64 {
                return Err(Failure::budget());
            }
        }
    }
    Ok(())
}

/// Legacy v1 predates strict native decoding. Validate panic-prone structures
/// before sampling, and refuse its unbounded tree traversal in this facade.
pub fn sampling(bytes: &[u8], limits: &Limits) -> Result<()> {
    if bytes.starts_with(b"DPK3") || bytes.starts_with(b"DPK2") || bytes.starts_with(b"DOPERAW") {
        return Ok(());
    }
    let value = io::json(bytes)?;
    if value["format"] != "dope-kernel" || value["version"] != 1 {
        return Err(Failure::unsupported());
    }
    let width = value["shell"]["p"]
        .as_u64()
        .ok_or_else(|| Failure::usage("v1_shape_invalid"))?;
    if value["dependence"]["kind"] == "gauss_copula" && width.saturating_pow(3) > limits.max_cells {
        return Err(Failure::budget());
    }
    if matches!(
        value["target"]["kind"].as_str(),
        Some("linear_sparse" | "lift_logit")
    ) {
        let terms = value["target"]["params"]["terms"]
            .as_array()
            .ok_or_else(|| Failure::usage("v1_target_invalid"))?;
        for term in terms {
            let pair = term
                .as_array()
                .ok_or_else(|| Failure::usage("v1_target_invalid"))?;
            if pair.len() != 2
                || pair[0].as_u64().is_none_or(|index| index >= width)
                || !pair[1].is_number()
            {
                return Err(Failure::usage("v1_target_invalid"));
            }
        }
    }
    if value["target"]["kind"] == "tree_piecewise"
        && value["target"]["params"].get("constant_prob").is_none()
    {
        return Err(Failure::unsupported());
    }
    if let Some(marginals) = value["marginals"].as_array() {
        for marginal in marginals {
            if marginal["kind"] == "empq" {
                let qs = marginal["params"]["qs"]
                    .as_array()
                    .ok_or_else(|| Failure::usage("v1_quantile_invalid"))?;
                let values = marginal["params"]["values"]
                    .as_array()
                    .ok_or_else(|| Failure::usage("v1_quantile_invalid"))?;
                if qs.len() < 2
                    || qs.len() != values.len()
                    || qs.iter().chain(values).any(|v| v.as_f64().is_none())
                {
                    return Err(Failure::usage("v1_quantile_invalid"));
                }
                if qs
                    .windows(2)
                    .any(|pair| pair[0].as_f64().unwrap() >= pair[1].as_f64().unwrap())
                {
                    return Err(Failure::usage("v1_quantile_invalid"));
                }
            }
        }
    }
    Ok(())
}
