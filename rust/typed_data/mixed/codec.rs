impl State {
    fn width(&self) -> usize {
        self.features.iter().map(Feature::width).sum()
    }

    fn validate(&self) -> Result<()> {
        self.affine.validate().map_err(schema_error)?;
        if self.format != "dope-mixed-state"
            || self.version != 1
            || self.features.is_empty()
            || self.width() > MAX_LATENT_FEATURES
            || self.features.len() + 1 != self.affine.columns.len()
            || self
                .affine
                .columns
                .last()
                .is_none_or(|c| c.role != ColumnRole::Target || c.dtype != ColumnType::Continuous)
        {
            return Err(rejected("mixed_state_invalid"));
        }
        let mut start = 0;
        for (column, feature) in self.affine.columns.iter().zip(&self.features) {
            if column.role != ColumnRole::Feature
                || column.id != feature.id
                || feature.start != start
                || (!column.dtype.numeric() && column.dtype != ColumnType::Nominal)
                || (column.dtype == ColumnType::Nominal) == feature.categories.is_empty()
                || feature.categories.len() > MAX_CATEGORIES
                || feature.categories.windows(2).any(|pair| pair[0] >= pair[1])
                || feature
                    .categories
                    .iter()
                    .any(|value| value.is_empty() || value.len() > self.affine.max_field_bytes)
                || !feature.completion.is_finite()
            {
                return Err(rejected("mixed_state_invalid"));
            }
            if let Some(crate::schema::ColumnDomain::Categorical { values }) = &column.domain {
                if feature
                    .categories
                    .iter()
                    .any(|value| !values.contains(value))
                {
                    return Err(rejected("mixed_dictionary_domain_invalid"));
                }
            }
            if feature.categories.is_empty() {
                if !(0.0..=1.0).contains(&feature.completion) {
                    return Err(rejected("mixed_completion_invalid"));
                }
            } else if feature.completion.fract() != 0.0
                || feature.completion as usize >= feature.categories.len()
            {
                return Err(rejected("mixed_completion_invalid"));
            }
            let validate_probs = |probs: &[f64]| -> Result<()> {
                if probs.len() != feature.observed_classes() + MASKS.len()
                    || probs
                        .iter()
                        .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
                    || (probs.iter().sum::<f64>() - 1.0).abs() > 1e-12
                    || probs[..feature.observed_classes()].iter().sum::<f64>() == 0.0
                        && feature.split.is_none()
                    || MASKS.iter().enumerate().any(|(at, mask)| {
                        probs[feature.observed_classes() + at] > 0.0
                            && !column.mask_states.contains(mask)
                    })
                {
                    return Err(rejected("mixed_probabilities_invalid"));
                }
                Ok(())
            };
            validate_probs(&feature.probabilities)?;
            if let Some(split) = &feature.split {
                let anchor = self
                    .features
                    .get(split.anchor)
                    .ok_or_else(|| rejected("mixed_anchor_invalid"))?;
                if self.profile != MixedProfile::NumericSplitV1
                    || !split.threshold.is_finite()
                    || !(0.0..1.0).contains(&split.threshold)
                    || anchor.split.is_some()
                    || !anchor.categories.is_empty()
                    || anchor.probabilities.first() != Some(&1.0)
                {
                    return Err(rejected("mixed_anchor_invalid"));
                }
                validate_probs(&split.left)?;
                validate_probs(&split.right)?;
            }
            start += feature.width();
        }
        Ok(())
    }
}

fn pack(state: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut content = state.to_vec();
    content.extend_from_slice(payload);
    let mut output = MAGIC.to_vec();
    output.extend_from_slice(&(state.len() as u64).to_le_bytes());
    output.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    output.extend_from_slice(blake3::hash(&content).as_bytes());
    output.extend_from_slice(&content);
    output
}

fn unpack(artifact: &[u8]) -> Result<(State, &[u8])> {
    if artifact.len() < HEADER || artifact.len() > MAX_BYTES || &artifact[..8] != MAGIC {
        return Err(rejected("mixed_artifact_invalid"));
    }
    let length = |offset| {
        usize::try_from(u64::from_le_bytes(
            artifact[offset..offset + 8].try_into().expect("header"),
        ))
        .map_err(|_| rejected("mixed_artifact_size_overflow"))
    };
    let state_len = length(8)?;
    let payload_len = length(16)?;
    if state_len > crate::schema::MAX_SCHEMA_BYTES
        || HEADER
            .checked_add(state_len)
            .and_then(|n| n.checked_add(payload_len))
            != Some(artifact.len())
        || blake3::hash(&artifact[HEADER..]).as_bytes() != &artifact[24..56]
    {
        return Err(rejected("mixed_artifact_integrity_failed"));
    }
    let state: State = serde_json::from_slice(&artifact[HEADER..HEADER + state_len])
        .map_err(|_| rejected("mixed_state_invalid"))?;
    state.validate()?;
    let payload = &artifact[HEADER + state_len..];
    crate::bundle::preflight_dpk(payload, &crate::bundle::BundleLimits::default())
        .map_err(|_| rejected("mixed_payload_limits_invalid"))?;
    let kernel = crate::codec::decode_kernel(payload)?;
    if kernel.features as usize != state.width()
        || kernel.task != Task::Regression
        || !matches!(kernel.program, KernelProgram::Symbolic { .. })
        || kernel.schema.iter().any(|c| c.missing_probability != 0.0)
    {
        return Err(rejected("mixed_sampler_closure_mismatch"));
    }
    Ok((state, payload))
}

/// Bound regular carrier files before allocation, just like the historical raw reader.
pub fn read_mixed_artifact(path: &std::path::Path) -> Result<Vec<u8>> {
    use std::io::Read;
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(|_| rejected("mixed_artifact_read_failed"))?;
    let metadata = file
        .metadata()
        .map_err(|_| rejected("mixed_artifact_read_failed"))?;
    if !metadata.is_file() || metadata.nlink() != 1 || metadata.len() > MAX_BYTES as u64 {
        return Err(rejected("mixed_artifact_carrier_invalid"));
    }
    let mut bytes = Vec::new();
    file.take(MAX_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| rejected("mixed_artifact_read_failed"))?;
    unpack(&bytes)?;
    Ok(bytes)
}

pub fn inspect_mixed(artifact: &[u8]) -> Result<MixedInspection> {
    let (state, payload) = unpack(artifact)?;
    Ok(MixedInspection {
        format: "dope-mixed-sampling-artifact",
        version: 1,
        profile: state.profile,
        capability: "bounded_numeric_nominal_features_five_masks_continuous_regression",
        support: "fit_range_numeric_one_hot_nominal_fit_only_empirical_classes_optional_one_numeric_split",
        privacy: "unmeasured_private_affine_dictionary_completion_probabilities_retained",
        certification: "unqualified_development_artifact",
        schema: state.affine.public_projection().map_err(schema_error)?,
        latent_features: state.width(),
        conditional_columns: state.features.iter().filter(|f| f.split.is_some()).count(),
        accounting: super::workflow::RawAccounting {
            byte_profile_id: "native-mixed-complete-v1",
            framing_bytes: HEADER,
            transform_schema_bytes: artifact.len() - HEADER - payload.len(),
            native_payload_bytes: payload.len(),
            complete_sampling_bytes: artifact.len(),
            transport_archive_bytes: artifact.len(),
            generic_runtime_bytes: None,
            cold_sampling_closure_bytes: None,
        },
        native_sections: crate::codec::account_artifact(payload)?,
        mathematical_parameter_count: None,
    })
}
