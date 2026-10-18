// Closed schema subset for the pinned ABI. No dynamic schemas or remote refs.
use super::*;

pub(super) fn validate(value: &Value, schema: &Value) -> Result<()> {
    const KNOWN: &[&str] = &[
        "type",
        "const",
        "enum",
        "anyOf",
        "additionalProperties",
        "required",
        "properties",
        "items",
        "pattern",
        "minLength",
        "minimum",
        "maximum",
    ];
    require(
        schema
            .as_object()
            .is_some_and(|s| s.keys().all(|k| KNOWN.contains(&k.as_str()))),
    )?;
    if let Some(kind) = schema["type"].as_str() {
        require(match kind {
            "object" => value.is_object(),
            "array" => value.is_array(),
            "string" => value.is_string(),
            "integer" => value.as_number().is_some_and(|n| !n.is_f64()),
            "boolean" => value.is_boolean(),
            "null" => value.is_null(),
            _ => false,
        })?;
    }
    if let Some(alternatives) = schema["anyOf"].as_array() {
        require(alternatives.iter().any(|s| validate(value, s).is_ok()))?;
    }
    if let Some(constant) = schema.get("const") {
        require(canonical_json(value).ok() == canonical_json(constant).ok())?;
    }
    if let Some(options) = schema["enum"].as_array() {
        require(options.contains(value))?;
    }
    if let (Some(object), Some(properties)) = (value.as_object(), schema["properties"].as_object())
    {
        require(object.keys().all(|k| properties.contains_key(k)))?;
        for key in schema["required"].as_array().ok_or_else(invalid)? {
            require(key.as_str().is_some_and(|k| object.contains_key(k)))?;
        }
        for (key, item) in object {
            validate(item, &properties[key])?;
        }
    }
    if let Some(array) = value.as_array() {
        for item in array {
            validate(item, &schema["items"])?;
        }
    }
    if let Some(text) = value.as_str() {
        require(text.len() as u64 >= n(&schema["minLength"]).unwrap_or(0))?;
        if schema.get("pattern").is_some() {
            require(schema["pattern"] == "^[0-9a-f]{64}$" && super::super::digest_valid(text))?;
        }
    }
    if let Some(number) = value.as_f64() {
        require(
            number >= schema["minimum"].as_f64().unwrap_or(f64::NEG_INFINITY)
                && number <= schema["maximum"].as_f64().unwrap_or(f64::INFINITY),
        )?;
    }
    Ok(())
}

pub(super) fn shape(value: &Value, index: usize) -> Result<()> {
    let schema = parse_json(include_bytes!(
        "../../../schemas/model-worker-v1.schema.json"
    ))
    .map_err(|_| invalid())?;
    validate(value, &schema["anyOf"][index])
}
