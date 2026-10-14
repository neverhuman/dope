/// Content commitment for G1's separate dope-frozen-geometry-state v1 body.
/// Recompute the shared finite-JSON digest; a caller label or digest alone is
/// insufficient. Coordinate identity qualifies neither transfer nor privacy.
pub fn frozen_geometry_state_v1(bytes: &[u8]) -> Result<Value> {
    let value = successor_record(bytes)?;
    successor_fields(
        &value,
        &["format", "version", "encoder_id", "schema_id", "columns"],
        "invalid_frozen_geometry",
    )?;
    require(
        value["format"] == "dope-frozen-geometry-state" && value["version"].as_u64() == Some(1),
        "invalid_frozen_geometry",
    )?;
    for name in ["encoder_id", "schema_id"] {
        require(
            value[name].as_str().is_some_and(|id| !id.is_empty()),
            "invalid_frozen_geometry",
        )?;
    }
    let columns = value["columns"]
        .as_array()
        .ok_or_else(|| error("invalid_frozen_geometry"))?;
    require(!columns.is_empty(), "invalid_frozen_geometry")?;
    for column in columns {
        match column["kind"].as_str() {
            Some("continuous") => {
                successor_fields(
                    column,
                    &["kind", "minimum", "maximum"],
                    "invalid_frozen_geometry",
                )?;
                require(
                    column["minimum"].is_f64() && column["maximum"].is_f64(),
                    "invalid_frozen_geometry",
                )?;
                let lo = column["minimum"]
                    .as_f64()
                    .ok_or_else(|| error("invalid_frozen_geometry"))?;
                let hi = column["maximum"]
                    .as_f64()
                    .ok_or_else(|| error("invalid_frozen_geometry"))?;
                require(
                    lo.is_finite() && hi.is_finite() && lo < hi,
                    "invalid_frozen_geometry",
                )?;
            }
            Some("nominal") => {
                successor_fields(column, &["kind", "levels"], "invalid_frozen_geometry")?;
                let levels =
                    successor_families(&column["levels"], false, "invalid_frozen_geometry")?;
                require(levels.len() >= 2, "invalid_frozen_geometry")?;
            }
            _ => return Err(error("invalid_frozen_geometry")),
        }
    }
    Ok(json!({"format":"dope-frozen-geometry-state","version":1,
        "state_digest":digest(&value)?,"label_verified":false,
        "privacy_result":false,"qualifies_transfer":false}))
}

/// Both original private bodies are validated and recomputed. Different
/// ordered columns, levels, scales, encoders or schemas never share a basis.
pub fn compare_frozen_geometry_v1(left: &[u8], right: &[u8]) -> Result<Value> {
    let left = frozen_geometry_state_v1(left)?;
    let right = frozen_geometry_state_v1(right)?;
    let same = left["state_digest"] == right["state_digest"];
    Ok(
        json!({"shared_geometry":same,"reason":if same {"common_frozen_state"} else {"different_frozen_state"},
        "label_verified":false,"privacy_result":false,"qualifies_transfer":false}),
    )
}
