#[derive(Clone, Debug, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Int(i64),
    String(String),
    Array(Vec<Json>),
    Object(BTreeMap<String, Json>),
}

struct Parser<'a> {
    data: &'a [u8],
    index: usize,
    depth: usize,
}

fn parse_json_subset(data: &[u8]) -> Result<Json, BundleError> {
    let mut parser = Parser {
        data,
        index: 0,
        depth: 0,
    };
    parser.skip();
    let value = parser.value()?;
    parser.skip();
    if parser.index != parser.data.len() {
        return Err(err("invalid_manifest"));
    }
    Ok(value)
}

fn canonical_subset(value: &Json) -> Result<Vec<u8>, BundleError> {
    let mut out = Vec::new();
    write_json(value, &mut out)?;
    Ok(out)
}

fn write_json(value: &Json, out: &mut Vec<u8>) -> Result<(), BundleError> {
    match value {
        Json::Null => out.extend_from_slice(b"null"),
        Json::Bool(true) => out.extend_from_slice(b"true"),
        Json::Bool(false) => out.extend_from_slice(b"false"),
        Json::Int(value) => out.extend_from_slice(itoa(*value).as_bytes()),
        Json::String(value) => write_string(value, out)?,
        Json::Array(values) => {
            out.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    out.push(b',');
                }
                write_json(value, out)?;
            }
            out.push(b']');
        }
        Json::Object(values) => {
            out.push(b'{');
            for (index, (key, value)) in values.iter().enumerate() {
                if index > 0 {
                    out.push(b',');
                }
                write_string(key, out)?;
                out.push(b':');
                write_json(value, out)?;
            }
            out.push(b'}');
        }
    }
    Ok(())
}

fn itoa(value: i64) -> String {
    value.to_string()
}

fn write_string(value: &str, out: &mut Vec<u8>) -> Result<(), BundleError> {
    out.push(b'"');
    for char in value.chars() {
        match char {
            '"' => out.extend_from_slice(b"\\\""),
            '\\' => out.extend_from_slice(b"\\\\"),
            '\u{0008}' => out.extend_from_slice(b"\\b"),
            '\u{000c}' => out.extend_from_slice(b"\\f"),
            '\n' => out.extend_from_slice(b"\\n"),
            '\r' => out.extend_from_slice(b"\\r"),
            '\t' => out.extend_from_slice(b"\\t"),
            ch if (ch as u32) < 0x20 => {
                let code = u32::from(ch);
                out.extend_from_slice(format!("\\u{code:04x}").as_bytes());
            }
            ch if ch.is_ascii() => out.push(ch as u8),
            _ => return Err(err("invalid_manifest")),
        }
    }
    out.push(b'"');
    Ok(())
}

impl<'a> Parser<'a> {
    fn skip(&mut self) {
        while self
            .data
            .get(self.index)
            .is_some_and(|byte| byte.is_ascii_whitespace())
        {
            self.index += 1;
        }
    }

    fn value(&mut self) -> Result<Json, BundleError> {
        self.skip();
        if self.depth > 8 {
            return Err(err("invalid_manifest"));
        }
        match self.data.get(self.index).copied() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::String(self.string()?)),
            Some(b't') => self.literal(b"true", Json::Bool(true)),
            Some(b'f') => self.literal(b"false", Json::Bool(false)),
            Some(b'n') => self.literal(b"null", Json::Null),
            Some(b'N') | Some(b'I') => Err(err("nonfinite_json_number")),
            Some(b'-') if self.data[self.index..].starts_with(b"-Infinity") => {
                Err(err("nonfinite_json_number"))
            }
            Some(b'-') | Some(b'0'..=b'9') => self.number(),
            _ => Err(err("invalid_manifest")),
        }
    }

    fn literal(&mut self, text: &[u8], value: Json) -> Result<Json, BundleError> {
        if self.data[self.index..].starts_with(text) {
            self.index += text.len();
            Ok(value)
        } else {
            Err(err("invalid_manifest"))
        }
    }

    fn number(&mut self) -> Result<Json, BundleError> {
        let start = self.index;
        if self.data.get(self.index) == Some(&b'-') {
            self.index += 1;
        }
        if self.data.get(self.index) == Some(&b'0') {
            self.index += 1;
            if self
                .data
                .get(self.index)
                .is_some_and(|byte| byte.is_ascii_digit())
            {
                return Err(err("invalid_manifest"));
            }
        } else if self
            .data
            .get(self.index)
            .is_some_and(|byte| byte.is_ascii_digit())
        {
            while self
                .data
                .get(self.index)
                .is_some_and(|byte| byte.is_ascii_digit())
            {
                self.index += 1;
            }
        } else {
            return Err(err("invalid_manifest"));
        }
        if self
            .data
            .get(self.index)
            .is_some_and(|byte| matches!(byte, b'.' | b'e' | b'E'))
        {
            return Err(err("invalid_manifest"));
        }
        let text = std::str::from_utf8(&self.data[start..self.index])
            .map_err(|_| err("invalid_manifest"))?;
        if text == "-0" || text == "-" {
            return Err(err("invalid_manifest"));
        }
        let value = text
            .parse::<i64>()
            .map_err(|_| err("integer_out_of_range"))?;
        if value.unsigned_abs() > JSON_INT_MAX {
            return Err(err("integer_out_of_range"));
        }
        Ok(Json::Int(value))
    }

    fn string(&mut self) -> Result<String, BundleError> {
        if self.data.get(self.index) != Some(&b'"') {
            return Err(err("invalid_manifest"));
        }
        self.index += 1;
        let mut out = String::new();
        loop {
            let byte = *self
                .data
                .get(self.index)
                .ok_or_else(|| err("invalid_manifest"))?;
            self.index += 1;
            match byte {
                b'"' => return Ok(out),
                b'\\' => {
                    let escaped = *self
                        .data
                        .get(self.index)
                        .ok_or_else(|| err("invalid_manifest"))?;
                    self.index += 1;
                    match escaped {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => out.push(self.hex_char()?),
                        _ => return Err(err("invalid_manifest")),
                    }
                }
                0x20..=0x7e => out.push(byte as char),
                _ => return Err(err("invalid_manifest")),
            }
            if out.len() > 4096 {
                return Err(err("invalid_manifest"));
            }
        }
    }

    fn hex_char(&mut self) -> Result<char, BundleError> {
        if self.index + 4 > self.data.len() {
            return Err(err("invalid_manifest"));
        }
        let text = std::str::from_utf8(&self.data[self.index..self.index + 4])
            .map_err(|_| err("invalid_manifest"))?;
        self.index += 4;
        let code = u32::from_str_radix(text, 16).map_err(|_| err("invalid_manifest"))?;
        char::from_u32(code)
            .filter(|char| char.is_ascii())
            .ok_or_else(|| err("invalid_manifest"))
    }

    fn array(&mut self) -> Result<Json, BundleError> {
        self.index += 1;
        self.depth += 1;
        let mut values = Vec::new();
        self.skip();
        if self.data.get(self.index) == Some(&b']') {
            self.index += 1;
            self.depth -= 1;
            return Ok(Json::Array(values));
        }
        loop {
            values.push(self.value()?);
            self.skip();
            match self.data.get(self.index).copied() {
                Some(b',') => {
                    self.index += 1;
                    self.skip();
                }
                Some(b']') => {
                    self.index += 1;
                    self.depth -= 1;
                    return Ok(Json::Array(values));
                }
                _ => return Err(err("invalid_manifest")),
            }
        }
    }

    fn object(&mut self) -> Result<Json, BundleError> {
        self.index += 1;
        self.depth += 1;
        let mut values = BTreeMap::new();
        self.skip();
        if self.data.get(self.index) == Some(&b'}') {
            self.index += 1;
            self.depth -= 1;
            return Ok(Json::Object(values));
        }
        loop {
            self.skip();
            let key = self.string()?;
            if values.contains_key(&key) {
                return Err(err("duplicate_json_key"));
            }
            self.skip();
            if self.data.get(self.index) != Some(&b':') {
                return Err(err("invalid_manifest"));
            }
            self.index += 1;
            values.insert(key, self.value()?);
            self.skip();
            match self.data.get(self.index).copied() {
                Some(b',') => self.index += 1,
                Some(b'}') => {
                    self.index += 1;
                    self.depth -= 1;
                    return Ok(Json::Object(values));
                }
                _ => return Err(err("invalid_manifest")),
            }
        }
    }
}

fn object(value: &Json) -> Result<&BTreeMap<String, Json>, BundleError> {
    match value {
        Json::Object(value) => Ok(value),
        _ => Err(err("schema_mismatch")),
    }
}

fn exact_keys(value: &BTreeMap<String, Json>, keys: &[&str]) -> Result<(), BundleError> {
    let found: BTreeSet<&str> = value.keys().map(String::as_str).collect();
    let expected: BTreeSet<&str> = keys.iter().copied().collect();
    if found.difference(&expected).next().is_some() {
        return Err(err("unknown_field"));
    }
    if expected.difference(&found).next().is_some() {
        return Err(err("schema_mismatch"));
    }
    Ok(())
}

fn json_str(value: &Json) -> Result<&str, BundleError> {
    match value {
        Json::String(value) => Ok(value),
        _ => Err(err("schema_mismatch")),
    }
}

fn json_bool(value: &Json) -> Result<bool, BundleError> {
    match value {
        Json::Bool(value) => Ok(*value),
        _ => Err(err("schema_mismatch")),
    }
}

fn json_u64(value: &Json, code: &'static str) -> Result<u64, BundleError> {
    match value {
        Json::Int(value) if *value >= 0 => Ok(*value as u64),
        _ => Err(err(code)),
    }
}

fn identifier(value: &str) -> Result<(), BundleError> {
    let mut chars = value.chars();
    match chars.next() {
        Some(char) if char.is_ascii_lowercase() => {}
        _ => return Err(err("schema_mismatch")),
    }
    if value.len() > 64
        || chars.any(|char| !char.is_ascii_lowercase() && !char.is_ascii_digit() && char != '-')
    {
        return Err(err("schema_mismatch"));
    }
    Ok(())
}

fn sha256_hex(value: &str) -> Result<(), BundleError> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        Ok(())
    } else {
        Err(err("schema_mismatch"))
    }
}

fn component_path(value: &str) -> Result<(), BundleError> {
    if value.len() > 512 || value.bytes().any(|byte| matches!(byte, b'\\' | 0 | b':')) {
        return Err(err("unsafe_path"));
    }
    let pieces: Vec<&str> = value.split('/').collect();
    if pieces.is_empty() || pieces.len() > 8 {
        return Err(err("unsafe_path"));
    }
    for piece in &pieces {
        let mut chars = piece.chars();
        match chars.next() {
            Some(char) if char.is_ascii_alphanumeric() => {}
            _ => return Err(err("unsafe_path")),
        }
        if piece.len() > 64
            || chars.any(|char| !char.is_ascii_alphanumeric() && !matches!(char, '.' | '_' | '-'))
        {
            return Err(err("unsafe_path"));
        }
    }
    if let Some((_, suffix)) = pieces.last().unwrap_or(&"").rsplit_once('.') {
        let lower = suffix.to_ascii_lowercase();
        if matches!(
            lower.as_str(),
            "dll" | "dylib" | "exe" | "jar" | "node" | "pkl" | "py" | "pyc" | "sh" | "so" | "wasm"
        ) {
            return Err(err("executable_rejected"));
        }
    }
    if value == MANIFEST_NAME {
        return Err(err("unsafe_path"));
    }
    Ok(())
}
