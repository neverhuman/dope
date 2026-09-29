
pub fn decode_kernel(artifact: &[u8]) -> Result<Kernel> {
    if artifact.len() < HEADER_BYTES || artifact.len() > MAX_ARTIFACT_BYTES {
        return Err(DopeError::Codec("artifact violates the size bounds".into()));
    }
    let v3 = &artifact[..4] == MAGIC;
    let v2 = &artifact[..4] == V2_MAGIC;
    if !v3 && !v2 {
        return Err(DopeError::Codec("not a V2 or V3 .dpk artifact".into()));
    }
    let supported_version = (v3 && artifact[4] == VERSION_MAJOR && artifact[5] <= VERSION_MINOR)
        || (v2 && artifact[4] == V2_VERSION_MAJOR && artifact[5] <= V2_VERSION_MINOR);
    if !supported_version || artifact[6] != DECODER_ID {
        return Err(DopeError::Codec(
            "unsupported bytecode or decoder version".into(),
        ));
    }
    let task = match artifact[7] {
        0 => Task::Regression,
        1 => Task::Binary,
        _ => return Err(DopeError::Codec("unknown task code".into())),
    };
    let quantization_bits = artifact[8];
    let seed_policy = artifact[9];
    let flags = u16::from_le_bytes(artifact[10..12].try_into().unwrap());
    if flags & !1 != 0 {
        return Err(DopeError::Codec("unknown header flags".into()));
    }
    let features = u32::from_le_bytes(artifact[12..16].try_into().unwrap());
    let rows_fitted = u64::from_le_bytes(artifact[16..24].try_into().unwrap());
    let seed = u64::from_le_bytes(artifact[24..32].try_into().unwrap());
    let payload_length = u64::from_le_bytes(artifact[32..40].try_into().unwrap()) as usize;
    if payload_length != artifact.len() - HEADER_BYTES {
        return Err(DopeError::Codec("payload length mismatch".into()));
    }
    let payload = &artifact[HEADER_BYTES..];
    if blake3::hash(payload).as_bytes()[..16] != artifact[40..56] {
        return Err(DopeError::Codec("payload checksum mismatch".into()));
    }

    let mut payload_reader = Reader::new(payload);
    let mut sections: [Option<Vec<u8>>; 6] = Default::default();
    let mut prior = 0u8;
    while payload_reader.remaining() > 0 {
        let encoded_tag = payload_reader.byte()?;
        let tag = encoded_tag & 0x7f;
        let maximum_tag = if v3 && artifact[5] >= 2 { 6 } else { 5 };
        if !(1..=maximum_tag).contains(&tag) || tag <= prior {
            return Err(DopeError::Codec(
                "sections are unknown, duplicated, or non-canonical".into(),
            ));
        }
        if tag == 6 && artifact.len() > MAX_NEURAL_ARTIFACT_BYTES {
            return Err(DopeError::Codec(
                "neural pilot artifact exceeds the 16 MiB cap".into(),
            ));
        }
        let encoded_length = payload_reader.usize(payload_reader.remaining())?;
        let raw_length = if encoded_tag & 0x80 != 0 {
            Some(payload_reader.usize(if tag == 6 {
                MAX_NEURAL_ARTIFACT_BYTES
            } else {
                MAX_ARTIFACT_BYTES
            })?)
        } else {
            None
        };
        let encoded = payload_reader.take(encoded_length)?;
        sections[tag as usize - 1] = Some(decode_section(
            encoded_tag,
            encoded,
            raw_length,
            v2 && artifact[5] == 0,
        )?);
        prior = tag;
    }
    let [schema, marginals, dependence, target, noise, neural] = sections;
    let schema = schema.ok_or_else(|| DopeError::Codec("missing schema section".into()))?;
    let marginals = marginals.ok_or_else(|| DopeError::Codec("missing marginal section".into()))?;
    let symbolic_count = [&dependence, &target, &noise]
        .into_iter()
        .filter(|section| section.is_some())
        .count();
    let program = match (symbolic_count, neural) {
        (3, None) => KernelProgram::Symbolic {
            dependence: decode_dependence(&dependence.unwrap(), features as usize)?,
            target: decode_target(&target.unwrap(), features as usize)?,
            noise: decode_noise(&noise.unwrap())?,
        },
        (0, Some(encoded)) if v3 && artifact[5] >= 2 => {
            let generator: crate::model::JointGenerator = serde_json::from_slice(&encoded)
                .map_err(|error| DopeError::Codec(format!("invalid neural section: {error}")))?;
            if serde_json::to_vec(&generator)? != encoded {
                return Err(DopeError::Codec(
                    "neural section is non-canonical or has unknown fields".into(),
                ));
            }
            let historical_shape = generator.profile.is_full()
                && generator.architecture != crate::model::NeuralArchitecture::TabDdpm;
            if (artifact[5] == 2) != historical_shape {
                return Err(DopeError::Codec(
                    "neural profile requires its canonical DPK3 minor version".into(),
                ));
            }
            KernelProgram::NeuralJoint(generator)
        }
        (0, None) => return Err(DopeError::Codec("missing kernel program section".into())),
        _ => {
            return Err(DopeError::Codec(
                "ambiguous or incomplete symbolic/neural payload".into(),
            ));
        }
    };
    let width = features as usize;
    let kernel = Kernel {
        task,
        rows_fitted,
        features,
        seed,
        seed_policy,
        quantization_bits,
        compliant: flags & 1 != 0,
        schema: decode_schema(&schema, width, v3 || artifact[5] >= 1)?,
        marginals: decode_marginals(&marginals, width)?,
        program,
    };
    if matches!(kernel.program, KernelProgram::NeuralJoint(_))
        && artifact.len() > MAX_NEURAL_ARTIFACT_BYTES
    {
        return Err(DopeError::Codec(
            "neural pilot artifact exceeds the 16 MiB cap".into(),
        ));
    }
    kernel.validate().map_err(DopeError::Codec)?;
    Ok(kernel)
}

pub fn load_kernel(path: &Path) -> Result<LoadedKernel> {
    let bytes = fs::read(path).map_err(|error| io_error(path, error))?;
    if bytes.starts_with(MAGIC) {
        Ok(LoadedKernel::V3(Box::new(decode_kernel(&bytes)?)))
    } else if bytes.starts_with(V2_MAGIC) {
        Ok(LoadedKernel::V2(Box::new(decode_kernel(&bytes)?)))
    } else {
        let value: Value = serde_json::from_slice(&bytes)?;
        if value.get("format").and_then(Value::as_str) != Some("dope-kernel")
            || value.get("version").and_then(Value::as_u64) != Some(1)
        {
            return Err(DopeError::Codec(
                "not a supported V1 JSON, V2 .dpk, or V3 .dpk artifact".into(),
            ));
        }
        Ok(LoadedKernel::V1(value))
    }
}