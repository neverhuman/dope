// Separate successor entries. Historical behavior-contract-v1 validation and
// receipt decoding stay unchanged; structural authentication grants no campaign.

fn successor_record(bytes: &[u8]) -> Result<Value> {
    require(
        bytes.len() <= MAX_MANIFEST,
        "successor_receipt_bytes_exceeded",
    )?;
    parse_json(bytes)
}

fn successor_version(value: &Value, expected: u64) -> bool {
    matches!(value, Value::Number(number) if number.as_f64() == Some(expected as f64))
}

fn successor_fields(value: &Value, fields: &[&str], code: &'static str) -> Result<()> {
    let object = value.as_object().ok_or_else(|| error(code))?;
    require(
        object.len() == fields.len() && fields.iter().all(|name| object.contains_key(*name)),
        code,
    )
}

fn successor_families(
    value: &Value,
    empty_allowed: bool,
    code: &'static str,
) -> Result<Vec<String>> {
    let array = value.as_array().ok_or_else(|| error(code))?;
    require(empty_allowed || !array.is_empty(), code)?;
    let values = array
        .iter()
        .map(|item| {
            item.as_str()
                .filter(|text| !text.is_empty())
                .map(str::to_owned)
                .ok_or_else(|| error(code))
        })
        .collect::<Result<Vec<_>>>()?;
    require(
        values.iter().collect::<BTreeSet<_>>().len() == values.len(),
        code,
    )?;
    Ok(values)
}

/// urn:dope:learner-roles:v2. A valid role assignment is development metadata;
/// it does not freeze a protocol, authorize confirmation or certify an artifact.
pub fn validate_learner_roles_v2(bytes: &[u8]) -> Result<Value> {
    let value = successor_record(bytes)?;
    require(value.is_object(), "invalid_learner_roles")?;
    require(
        value.get("required_families").is_none() && !successor_version(&value["version"], 1),
        "legacy_role_identity",
    )?;
    successor_fields(
        &value,
        &[
            "format",
            "version",
            "selection_families",
            "evaluation_required_families",
            "held_out_families",
        ],
        "invalid_learner_roles",
    )?;
    require(
        value["format"] == "dope-learner-roles" && successor_version(&value["version"], 2),
        "invalid_learner_roles",
    )?;
    let selection = successor_families(
        &value["selection_families"],
        false,
        "invalid_selection_families",
    )?;
    let evaluation = successor_families(
        &value["evaluation_required_families"],
        false,
        "invalid_evaluation_families",
    )?;
    let held = successor_families(
        &value["held_out_families"],
        true,
        "invalid_held_out_families",
    )?;
    require(
        !selection.iter().any(|family| held.contains(family)),
        "held_out_selected",
    )?;
    require(
        held.iter().all(|family| evaluation.contains(family)),
        "held_out_not_evaluated",
    )?;
    let mut not_evaluated = selection
        .iter()
        .filter(|family| !evaluation.contains(family))
        .cloned()
        .collect::<Vec<_>>();
    not_evaluated.sort();
    Ok(json!({"format":"dope-learner-roles", "version":2,
        "selection_families":selection, "evaluation_required_families":evaluation,
        "held_out_families":held, "selection_not_evaluated":not_evaluated,
        "protocol_frozen":false, "confirmation_allowed":false, "campaign":"unmeasured",
        "native_certification":false, "qualification":false}))
}

/// Held-out families are skipped before reading their loss. The returned winner
/// remains a development choice, including when its role metadata authenticates.
pub fn select_development_learner_v2(roles: &[u8], candidates: &[u8]) -> Result<Value> {
    let mut assigned = validate_learner_roles_v2(roles)?;
    let candidates = successor_record(candidates)?;
    let rows = candidates
        .as_array()
        .ok_or_else(|| error("invalid_learner_candidate"))?;
    let mut best: Option<(f64, String)> = None;
    for candidate in rows {
        let family = candidate["family_id"]
            .as_str()
            .filter(|id| !id.is_empty())
            .ok_or_else(|| error("invalid_learner_candidate"))?;
        let learner = candidate["learner_id"]
            .as_str()
            .filter(|id| !id.is_empty())
            .ok_or_else(|| error("invalid_learner_candidate"))?;
        let selected = assigned["selection_families"]
            .as_array()
            .expect("validated roles")
            .iter()
            .any(|value| value == family);
        let held = assigned["held_out_families"]
            .as_array()
            .expect("validated roles")
            .iter()
            .any(|value| value == family);
        if held || !selected {
            continue;
        }
        let loss = candidate["development_loss"]
            .as_f64()
            .filter(|loss| loss.is_finite())
            .ok_or_else(|| error("invalid_development_loss"))?;
        if best.as_ref().is_none_or(|(previous, id)| {
            loss < *previous || (loss == *previous && learner < id.as_str())
        }) {
            best = Some((loss, learner.to_owned()));
        }
    }
    assigned["selected_learner"] = best.map(|(_, id)| json!(id)).unwrap_or(Value::Null);
    assigned["held_out_excluded_from_selection"] = json!(true);
    Ok(assigned)
}

fn calibration_decision(authenticates: bool, reason: &'static str) -> Value {
    json!({"format":"dope-calibration-decision", "version":1, "authenticates":authenticates,
        "qualifies":false, "calibration":"unmeasured", "reason":reason,
        "campaign":"unmeasured", "n13b":false, "release":false})
}

/// urn:dope:calibration-receipt:v1. Match the captured development protocol and
/// canonical binding, while refusing any measured N13b/coverage/release inference.
pub fn verify_calibration_receipt_v1(bytes: &[u8]) -> Result<Value> {
    let mut value = successor_record(bytes)?;
    let assertions = [
        "qualifies",
        "calibration",
        "n13b",
        "authenticates",
        "campaign",
        "release",
    ];
    require(
        !assertions.iter().any(|name| value.get(*name).is_some()),
        "self_asserted_qualification",
    )?;
    if value["version"].is_u64() && !successor_version(&value["version"], 1) {
        return Err(error("unknown_calibration_receipt_version"));
    }
    let fields = [
        "format",
        "version",
        "protocol_id",
        "protocol_digest",
        "repetitions",
        "bootstrap_draws",
        "frozen_for_confirmation",
        "manifest_digest",
        "tolerances_frozen_before_outcomes",
        "mcse_reported",
        "wilson_reported",
        "binding_digest",
    ];
    successor_fields(&value, &fields, "invalid_calibration_receipt")?;
    require(
        value["format"] == "dope-calibration-receipt" && successor_version(&value["version"], 1),
        "invalid_calibration_receipt",
    )?;
    require(
        value["protocol_id"]
            .as_str()
            .is_some_and(|id| !id.is_empty()),
        "invalid_calibration_receipt",
    )?;
    for field in ["protocol_digest", "binding_digest"] {
        require(
            value[field].as_str().is_some_and(is_digest),
            "invalid_calibration_receipt",
        )?;
    }
    require(
        value["manifest_digest"].is_null()
            || value["manifest_digest"].as_str().is_some_and(is_digest),
        "invalid_calibration_receipt",
    )?;
    for field in ["repetitions", "bootstrap_draws"] {
        require(
            value[field]
                .as_u64()
                .is_some_and(|count| count > 0 && count <= MAX_INTEGER),
            "invalid_calibration_receipt",
        )?;
    }
    for field in [
        "frozen_for_confirmation",
        "tolerances_frozen_before_outcomes",
        "mcse_reported",
        "wilson_reported",
    ] {
        require(value[field].is_boolean(), "invalid_calibration_receipt")?;
    }
    let binding = value
        .as_object_mut()
        .expect("validated receipt")
        .remove("binding_digest")
        .expect("validated field");
    if binding != digest(&value)? {
        return Ok(calibration_decision(false, "tampered_calibration_receipt"));
    }
    let protocol = parse_json(include_bytes!(
        "../../research/transfer/protocols/statistics-v1.json"
    ))?;
    if value["protocol_id"] != protocol["protocol_id"] {
        return Ok(calibration_decision(false, "unknown_protocol"));
    }
    if value["protocol_digest"] != digest(&protocol)? {
        return Ok(calibration_decision(false, "protocol_digest_mismatch"));
    }
    Ok(calibration_decision(
        true,
        "development_protocol_is_not_n13b",
    ))
}
