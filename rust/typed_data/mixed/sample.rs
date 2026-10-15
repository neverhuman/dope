#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum SampleCell {
    Number(f64),
    Category(String),
    Mask(MaskState),
}

#[derive(Serialize)]
pub struct MixedSamples {
    pub values: Vec<SampleCell>,
    pub integer_rounding_repairs: usize,
    pub boolean_rounding_repairs: usize,
    pub masks_emitted: usize,
    pub physical_domain_rejections: usize,
}

// Stable, separate draw stream for this explicit development profile. It is
// reproducibility machinery, not a security or privacy randomness guarantee.
fn class_uniform(seed: u64, row: usize, column: usize) -> f64 {
    let mut value = seed
        ^ (row as u64).wrapping_mul(0x9e3779b97f4a7c15)
        ^ (column as u64).wrapping_mul(0xd1b54a32d192ed03)
        ^ 0x6d697865645f7631;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((value ^ (value >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}

fn draw_class(state: &State, feature: &Feature, row: &[f32], uniform: f64) -> usize {
    let probabilities = feature
        .split
        .as_ref()
        .map_or(&feature.probabilities, |split| {
            if row[state.features[split.anchor].start] > split.threshold {
                &split.right
            } else {
                &split.left
            }
        });
    let mut cumulative = 0.0;
    for (index, p) in probabilities.iter().enumerate() {
        cumulative += p;
        if uniform < cumulative {
            return index;
        }
    }
    // A rounding remainder belongs to the last nonzero class, never an
    // undeclared trailing mask with probability zero.
    probabilities
        .iter()
        .rposition(|p| *p > 0.0)
        .expect("validated probabilities")
}

fn restore(column: &super::RetainedColumn, normalized: f32) -> Result<(f64, usize, usize)> {
    let original = f64::from(normalized) * column.scale + column.offset;
    let (value, integer, boolean) = match column.dtype {
        ColumnType::Integer | ColumnType::Ordinal => (
            original.round(),
            usize::from(original.round() != original),
            0,
        ),
        ColumnType::Boolean => {
            let value = f64::from(u8::from(original >= 0.5));
            (value, 0, usize::from(value != original))
        }
        _ => (original, 0, 0),
    };
    super::validate_numeric_cell(value, column.dtype, &column.domain, |_| {
        crate::schema::SchemaError::new("mixed_sample_physical_domain_violation", None, None)
    })
    .map_err(schema_error)?;
    Ok((value, integer, boolean))
}

/// Bytes alone determine typed cells. Class completion is applied before
/// target/noise evaluation; masked public cells keep their distinct mask state.
pub fn sample_mixed(artifact: &[u8], rows: usize, seed: Option<u64>) -> Result<MixedSamples> {
    let (state, payload) = unpack(artifact)?;
    if rows == 0
        || rows > 100_000
        || rows
            .checked_mul(state.affine.columns.len())
            .is_none_or(|n| n > 1_000_000)
        || rows
            .checked_mul(state.width() + 1)
            .is_none_or(|n| n > 1_000_000)
    {
        return Err(rejected("mixed_sample_cap_exceeded"));
    }
    let kernel = crate::codec::decode_kernel(payload)?;
    let seed = seed.unwrap_or(kernel.seed);
    let normalized = crate::sample::sample_symbolic_with_features(
        &kernel,
        crate::sample::SampleOptions {
            rows,
            seed: Some(seed),
        },
        false,
        |row_index, values| {
            for (index, feature) in state.features.iter().enumerate() {
                let class = draw_class(
                    &state,
                    feature,
                    values,
                    class_uniform(seed, row_index, index),
                );
                if feature.categories.is_empty() {
                    if class != 0 {
                        values[feature.start] = feature.completion;
                    }
                } else {
                    let selected = if class < feature.categories.len() {
                        class
                    } else {
                        feature.completion as usize
                    };
                    for at in 0..feature.width() {
                        values[feature.start + at] = u8::from(at == selected) as f32;
                    }
                }
            }
        },
    )?;
    let mut samples = MixedSamples {
        values: Vec::with_capacity(rows * state.affine.columns.len()),
        integer_rounding_repairs: 0,
        boolean_rounding_repairs: 0,
        masks_emitted: 0,
        physical_domain_rejections: 0,
    };
    for (row_index, row) in normalized.chunks_exact(state.width() + 1).enumerate() {
        for (index, (column, feature)) in
            state.affine.columns.iter().zip(&state.features).enumerate()
        {
            let class = draw_class(&state, feature, row, class_uniform(seed, row_index, index));
            if class >= feature.observed_classes() {
                samples
                    .values
                    .push(SampleCell::Mask(MASKS[class - feature.observed_classes()]));
                samples.masks_emitted += 1;
            } else if feature.categories.is_empty() {
                let (value, integer, boolean) = restore(column, row[feature.start])?;
                samples.integer_rounding_repairs += integer;
                samples.boolean_rounding_repairs += boolean;
                samples.values.push(SampleCell::Number(value));
            } else {
                samples
                    .values
                    .push(SampleCell::Category(feature.categories[class].clone()));
            }
        }
        let target = state.affine.columns.last().expect("validated target");
        samples.values.push(SampleCell::Number(
            restore(
                target,
                if state.constant_target {
                    0.0
                } else {
                    row[state.width()]
                },
            )?
            .0,
        ));
    }
    Ok(samples)
}
