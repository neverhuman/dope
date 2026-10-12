// Closed Draft 2020-12 subset used by the three tracked schemas. Both runtimes
// validate against these sources. Unsupported shape changes need verifier changes.
fn validate(value: &Value, schema: &Value) -> Result<()> {
    const KNOWN: &[&str] = &[
        "$schema",
        "$id",
        "$defs",
        "title",
        "type",
        "anyOf",
        "allOf",
        "if",
        "then",
        "else",
        "const",
        "enum",
        "required",
        "properties",
        "additionalProperties",
        "items",
        "minItems",
        "uniqueItems",
        "minLength",
        "pattern",
        "minimum",
        "maximum",
    ];
    if let Some(m) = schema.as_object() {
        require(
            m.keys().all(|k| KNOWN.contains(&k.as_str())),
            "unsupported_schema_keyword",
        )?;
    }
    if let Some(alternatives) = schema["anyOf"].as_array() {
        require(
            alternatives.iter().any(|s| validate(value, s).is_ok()),
            "schema_mismatch",
        )?;
    }
    if let Some(t) = schema["type"].as_str() {
        let valid = match t {
            "object" => value.is_object(),
            "array" => value.is_array(),
            "string" => value.is_string(),
            "integer" => value.as_number().is_some_and(|n| !n.is_f64()),
            "number" => value.is_number(),
            "boolean" => value.is_boolean(),
            "null" => value.is_null(),
            _ => false,
        };
        require(valid, "schema_mismatch")?;
    }
    if let Some(c) = schema.get("const") {
        require(
            canonical_json(value)? == canonical_json(c)?,
            "schema_mismatch",
        )?;
    }
    if let Some(a) = schema["enum"].as_array() {
        require(a.iter().any(|v| v == value), "schema_mismatch")?;
    }
    if let Some(m) = value.as_object() {
        if let Some(required) = schema["required"].as_array() {
            require(
                required
                    .iter()
                    .all(|k| k.as_str().is_some_and(|k| m.contains_key(k))),
                "schema_mismatch",
            )?;
        }
        let properties = schema["properties"].as_object();
        if schema["additionalProperties"] == false {
            require(
                m.keys()
                    .all(|k| properties.is_some_and(|p| p.contains_key(k))),
                "schema_mismatch",
            )?;
        }
        if let Some(properties) = properties {
            for (k, s) in properties {
                if let Some(v) = m.get(k) {
                    validate(v, s)?;
                }
            }
        }
    }
    if let Some(a) = value.as_array() {
        require(
            a.len() as u64 >= schema["minItems"].as_u64().unwrap_or(0),
            "schema_mismatch",
        )?;
        if schema["uniqueItems"] == true {
            let mut unique = BTreeSet::new();
            for v in a {
                require(unique.insert(canonical_json(v)?), "schema_mismatch")?;
            }
        }
        if let Some(s) = schema.get("items") {
            for v in a {
                validate(v, s)?;
            }
        }
    }
    if let Some(s) = value.as_str() {
        require(
            s.chars().count() as u64 >= schema["minLength"].as_u64().unwrap_or(0),
            "schema_mismatch",
        )?;
        if schema.get("pattern").is_some() {
            require(
                schema["pattern"] == "^[0-9a-f]{64}$" && is_digest(s),
                "schema_mismatch",
            )?;
        }
    }
    if let Some(n) = value.as_f64() {
        require(
            n >= schema["minimum"].as_f64().unwrap_or(f64::NEG_INFINITY)
                && n <= schema["maximum"].as_f64().unwrap_or(f64::INFINITY),
            "schema_mismatch",
        )?;
    }
    if let Some(a) = schema["allOf"].as_array() {
        for s in a {
            validate(value, s)?;
        }
    }
    if let Some(condition) = schema.get("if") {
        let branch = if validate(value, condition).is_ok() {
            "then"
        } else {
            "else"
        };
        if let Some(s) = schema.get(branch) {
            validate(value, s)?;
        }
    }
    Ok(())
}
fn schema(text: &str) -> Result<Value> {
    parse_json(text.as_bytes())
}
pub fn validate_behavior_contract(value: &Value) -> Result<()> {
    canonical_json(value)?;
    validate(value, &schema(BEHAVIOR_SCHEMA)?)?;
    let required = value["learners"]["required_families"]
        .as_array()
        .ok_or_else(|| error("schema_mismatch"))?;
    let held_out = value["learners"]["held_out_families"]
        .as_array()
        .ok_or_else(|| error("schema_mismatch"))?;
    require(
        !required.iter().any(|v| held_out.contains(v)),
        "learner_holdout_overlap",
    )
}
