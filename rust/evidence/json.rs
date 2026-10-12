// serde_json::Value alone silently accepts duplicate object keys.
struct StrictValue(Value);
struct StrictSeed<'a>(&'a mut std::vec::IntoIter<Number>);
impl<'de> DeserializeSeed<'de> for StrictSeed<'_> {
    type Value = StrictValue;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> std::result::Result<Self::Value, D::Error> {
        struct StrictVisitor<'a>(&'a mut std::vec::IntoIter<Number>);
        impl<'de> Visitor<'de> for StrictVisitor<'_> {
            type Value = StrictValue;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("finite JSON")
            }
            fn visit_bool<E: de::Error>(self, v: bool) -> std::result::Result<Self::Value, E> {
                Ok(StrictValue(v.into()))
            }
            fn visit_u64<E: de::Error>(
                self,
                _placeholder: u64,
            ) -> std::result::Result<Self::Value, E> {
                self.0
                    .next()
                    .map(|n| StrictValue(Value::Number(n)))
                    .ok_or_else(|| E::custom("invalid_number_stream"))
            }
            fn visit_i64<E: de::Error>(
                self,
                placeholder: i64,
            ) -> std::result::Result<Self::Value, E> {
                self.visit_u64(placeholder as u64)
            }
            fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Self::Value, E> {
                Ok(StrictValue(v.into()))
            }
            fn visit_none<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
                Ok(StrictValue(Value::Null))
            }
            fn visit_unit<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
                Ok(StrictValue(Value::Null))
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut a: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(v) = a.next_element_seed(StrictSeed(&mut *self.0))? {
                    values.push(v.0);
                }
                Ok(StrictValue(Value::Array(values)))
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut a: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut values = Map::new();
                while let Some(k) = a.next_key::<String>()? {
                    if values.contains_key(&k) {
                        return Err(de::Error::custom("duplicate_json_key"));
                    }
                    values.insert(k, a.next_value_seed(StrictSeed(&mut *self.0))?.0);
                }
                Ok(StrictValue(Value::Object(values)))
            }
        }
        d.deserialize_any(StrictVisitor(self.0))
    }
}

fn number_token(token: &str) -> Result<Number> {
    // JSON number grammar; replacing tokens below must not admit e.g. 01 or 1.
    let bytes = token.as_bytes();
    let mut i = usize::from(bytes.first() == Some(&b'-'));
    if bytes.get(i) == Some(&b'0') {
        i += 1;
    } else {
        require(
            bytes.get(i).is_some_and(|b| (b'1'..=b'9').contains(b)),
            "invalid_finite_json",
        )?;
        while bytes.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
    }
    if bytes.get(i) == Some(&b'.') {
        i += 1;
        let start = i;
        while bytes.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        require(i > start, "invalid_finite_json")?;
    }
    if bytes.get(i).is_some_and(|b| matches!(b, b'e' | b'E')) {
        i += 1;
        if bytes.get(i).is_some_and(|b| matches!(b, b'+' | b'-')) {
            i += 1;
        }
        let start = i;
        while bytes.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        require(i > start, "invalid_finite_json")?;
    }
    require(i == bytes.len(), "invalid_finite_json")?;
    if token.contains(['.', 'e', 'E']) {
        Number::from_f64(
            token
                .parse::<f64>()
                .map_err(|_| error("invalid_finite_json"))?,
        )
        .ok_or_else(|| error("nonfinite_json_number"))
    } else {
        let n = token
            .parse::<i64>()
            .map_err(|_| error("integer_out_of_range"))?;
        require(n.unsigned_abs() <= MAX_INTEGER, "integer_out_of_range")?;
        Ok(n.into())
    }
}

pub fn parse_json(data: &[u8]) -> Result<Value> {
    // Parse numeric tokens with std's correctly rounded f64 parser, and let
    // serde handle JSON structure, escaping and duplicate-key detection. The
    // number stream restores original scalar types in encounter order. This
    // confines numeric behavior to vNext without changing historical parsers.
    let mut offset = 0;
    let mut quoted = false;
    let mut copied = 0;
    let mut structure = Vec::with_capacity(data.len());
    let mut numbers = Vec::new();
    while offset < data.len() {
        match data[offset] {
            b'\\' if quoted => offset += 2,
            b'"' => {
                quoted = !quoted;
                offset += 1;
            }
            b'-' | b'0'..=b'9' if !quoted => {
                let start = offset;
                offset += 1;
                while offset < data.len()
                    && matches!(data[offset], b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-')
                {
                    offset += 1;
                }
                let token = std::str::from_utf8(&data[start..offset])
                    .map_err(|_| error("invalid_finite_json"))?;
                numbers.push(number_token(token)?);
                structure.extend_from_slice(&data[copied..start]);
                structure.push(b'0');
                copied = offset;
            }
            _ => offset += 1,
        }
    }
    structure.extend_from_slice(&data[copied..]);
    let mut numbers = numbers.into_iter();
    let mut deserializer = serde_json::Deserializer::from_slice(&structure);
    let value = StrictSeed(&mut numbers)
        .deserialize(&mut deserializer)
        .map_err(|_| error("invalid_finite_json"))?
        .0;
    deserializer
        .end()
        .map_err(|_| error("invalid_finite_json"))?;
    require(numbers.next().is_none(), "invalid_number_stream")?;
    Ok(value)
}

/// dope-finite-json-v1, not RFC 8785. Float/int representations remain distinct.
pub fn canonical_json(value: &Value) -> Result<Vec<u8>> {
    fn encode(value: &Value, out: &mut String) -> Result<()> {
        match value {
            Value::Object(m) => {
                out.push('{');
                for (i, (k, v)) in m.iter().collect::<BTreeMap<_, _>>().iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(
                        &serde_json::to_string(k).map_err(|_| error("invalid_json_encoding"))?,
                    );
                    out.push(':');
                    encode(v, out)?;
                }
                out.push('}');
            }
            Value::Array(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    encode(v, out)?;
                }
                out.push(']');
            }
            Value::Number(n) if n.is_f64() => {
                let v = n.as_f64().ok_or_else(|| error("nonfinite_json_number"))?;
                require(v.is_finite(), "nonfinite_json_number")?;
                out.push_str(&format!("{v:.17e}"));
            }
            Value::Number(n) => {
                let magnitude = n
                    .as_i64()
                    .map(i64::unsigned_abs)
                    .or_else(|| n.as_u64())
                    .ok_or_else(|| error("integer_out_of_range"))?;
                require(magnitude <= MAX_INTEGER, "integer_out_of_range")?;
                out.push_str(&n.to_string());
            }
            _ => out.push_str(
                &serde_json::to_string(value).map_err(|_| error("invalid_json_encoding"))?,
            ),
        }
        Ok(())
    }
    let mut out = String::new();
    encode(value, &mut out)?;
    Ok(out.into_bytes())
}
fn digest(value: &Value) -> Result<String> {
    Ok(hash(&canonical_json(value)?))
}
