use super::*;

fn fixture(held: &str, unit: f64) -> TypedTable {
    let masks = serde_json::json!([
        {"token":"M","state":"missing"},{"token":"U","state":"unknown_category"},
        {"token":"S","state":"structural_absence"},{"token":"P","state":"padding"},
        {"token":"C","state":"censored"}
    ]);
    let schema = crate::schema::TableSchemaV1::from_private_json(&serde_json::to_vec(&serde_json::json!({
        "format":"dope-table-schema","version":1,"columns":[
            {"id":"c0001","source_name":"secret_anchor","role":"feature","dtype":"continuous"},
            {"id":"c0002","source_name":"secret_number","role":"feature","dtype":"integer","mask_tokens":masks},
            {"id":"c0003","source_name":"secret_cat","role":"feature","dtype":"nominal","mask_tokens":masks},
            {"id":"c0004","source_name":"secret_other","role":"feature","dtype":"nominal"},
            {"id":"c0005","source_name":"secret_target","role":"target","dtype":"continuous"},
            {"id":"c0006","source_name":"secret_group","role":"group_split_key","dtype":"nominal"}
        ]
    })).unwrap()).unwrap();
    let mut csv = String::from(
        "secret_anchor,secret_number,secret_cat,secret_other,secret_target,secret_group\n",
    );
    for row in 0..120 {
        let mask = ["M", "U", "S", "P", "C"][(row / 2) % 5];
        let numeric = if row >= 60 && row % 2 == 0 {
            mask.to_owned()
        } else {
            (10 + row % 5).to_string()
        };
        let category = if row >= 60 && row % 2 == 0 {
            mask
        } else if row < 60 {
            "Aa"
        } else {
            "BB"
        };
        csv.push_str(&format!(
            "{},{},{},{},{},private_group\n",
            row as f64 * unit,
            numeric,
            category,
            if row % 2 == 0 { "BB" } else { "Aa" },
            (100.0 + row as f64) * unit
        ));
    }
    csv.push_str(&format!("99999,12,{held},Aa,99999,held_group\n"));
    TypedTable::read_csv(csv.as_bytes(), schema, super::super::InputLimits::default()).unwrap()
}

fn options() -> CompileOptions {
    CompileOptions {
        seed: Some(1729),
        backend_id: Some("independent_quantile".into()),
        quantization_profiles: vec![8],
        ..CompileOptions::default()
    }
}

#[test]
fn masks_nominal_dictionaries_and_conditional_classes_are_fit_only_and_charged() {
    let rows = (0..120).collect::<Vec<_>>();
    let a = compile_mixed(
        &fixture("held_private_A", 1.0),
        &rows,
        &options(),
        32768,
        MixedProfile::NumericSplitV1,
    )
    .unwrap();
    let b = compile_mixed(
        &fixture("held_private_B", 1.0),
        &rows,
        &options(),
        32768,
        MixedProfile::NumericSplitV1,
    )
    .unwrap();
    assert_eq!(a.artifact, b.artifact);
    assert!(super::super::workflow::inspect_raw(&a.artifact).is_err());
    let (state, _) = unpack(&a.artifact).unwrap();
    assert_eq!(state.features[2].categories, ["Aa", "BB"]);
    assert_eq!(state.features[3].categories, ["Aa", "BB"]);
    assert_ne!(state.features[2].start, state.features[3].start);
    assert_eq!(state.features[1].probabilities.len(), 6);
    assert!(
        state.features[1].probabilities[1..]
            .iter()
            .all(|p| *p > 0.0)
    );
    assert!(
        !state
            .affine
            .columns
            .iter()
            .any(|c| c.role == ColumnRole::GroupSplitKey)
    );
    let account = &a.inspection.accounting;
    assert_eq!(
        account.complete_sampling_bytes,
        HEADER + account.transform_schema_bytes + account.native_payload_bytes
    );
    assert_eq!(account.complete_sampling_bytes, a.artifact.len());
    assert_eq!(
        compile_mixed(
            &fixture("x", 1.0),
            &rows,
            &options(),
            a.artifact.len(),
            MixedProfile::NumericSplitV1
        )
        .unwrap()
        .artifact,
        a.artifact
    );
    assert!(
        compile_mixed(
            &fixture("x", 1.0),
            &rows,
            &options(),
            a.artifact.len() - 1,
            MixedProfile::NumericSplitV1
        )
        .is_err()
    );
    let public = serde_json::to_string(&a.inspection).unwrap();
    for private in [
        "secret_anchor",
        "held_private_A",
        "private_group",
        "\"categories\"",
        "\"probabilities\"",
        "\"offset\"",
        "Aa",
        "BB",
    ] {
        assert!(!public.contains(private));
    }
}

#[test]
fn sampler_consumes_conditional_masks_and_nominal_values_without_mask_collapse() {
    let table = fixture("held", 1.0);
    let fit = (0..120).collect::<Vec<_>>();
    let conditional = compile_mixed(
        &table,
        &fit,
        &options(),
        32768,
        MixedProfile::NumericSplitV1,
    )
    .unwrap();
    let independent =
        compile_mixed(&table, &fit, &options(), 32768, MixedProfile::IndependentV1).unwrap();
    let a = sample_mixed(&conditional.artifact, 2000, Some(7)).unwrap();
    assert_eq!(
        a.values,
        sample_mixed(&conditional.artifact, 2000, Some(7))
            .unwrap()
            .values
    );
    let b = sample_mixed(&independent.artifact, 2000, Some(7)).unwrap();
    let mask_low = |values: &[SampleCell]| {
        values
            .chunks_exact(5)
            .filter(|row| {
                matches!(row[0], SampleCell::Number(v) if v < 59.0)
                    && matches!(row[1], SampleCell::Mask(_))
            })
            .count()
    };
    assert_eq!(mask_low(&a.values), 0);
    assert!(mask_low(&b.values) > 100);
    for mask in MASKS {
        assert!(a.values.contains(&SampleCell::Mask(mask)));
    }
    assert!(a.values.chunks_exact(5).all(|row| {
        matches!(row[1], SampleCell::Number(v) if v.fract() == 0.0)
            || matches!(row[1], SampleCell::Mask(_))
    }));
    assert!(a.values.iter().all(|cell| match cell {
        SampleCell::Number(v) => v.is_finite(),
        SampleCell::Category(v) => v == "Aa" || v == "BB",
        _ => true,
    }));
}

#[test]
fn completion_and_one_hot_category_are_consumed_before_target_evaluation() {
    let compiled = compile_mixed(
        &fixture("held", 1.0),
        &(0..120).collect::<Vec<_>>(),
        &options(),
        32768,
        MixedProfile::NumericSplitV1,
    )
    .unwrap();
    let (state, payload) = unpack(&compiled.artifact).unwrap();
    let mut kernel = crate::codec::decode_kernel(payload).unwrap();
    let KernelProgram::Symbolic { target, noise, .. } = &mut kernel.program else {
        panic!("symbolic");
    };
    *target = crate::model::Target::SparseLinear {
        intercept: 0.0,
        terms: vec![crate::model::LinearTerm {
            feature: state.features[1].start as u32,
            coefficient: 1.0,
        }],
    };
    *noise = crate::model::Noise::Homoscedastic { sigma: 0.0 };
    let artifact = pack(
        &serde_json::to_vec(&state).unwrap(),
        &crate::codec::encode_kernel(&kernel).unwrap(),
    );
    let sampled = sample_mixed(&artifact, 1000, Some(7)).unwrap();
    let target = state.affine.columns.last().unwrap();
    let completion = f64::from(state.features[1].completion) * target.scale + target.offset;
    let masked = sampled
        .values
        .chunks_exact(5)
        .filter(|row| matches!(row[1], SampleCell::Mask(_)))
        .collect::<Vec<_>>();
    assert!(!masked.is_empty());
    assert!(
        masked
            .iter()
            .all(|row| matches!(row[4], SampleCell::Number(v) if (v - completion).abs() < 0.0001))
    );
    let KernelProgram::Symbolic { target, .. } = &mut kernel.program else {
        unreachable!()
    };
    *target = crate::model::Target::SparseLinear {
        intercept: 0.0,
        terms: vec![crate::model::LinearTerm {
            feature: state.features[2].start as u32,
            coefficient: 1.0,
        }],
    };
    let artifact = pack(
        &serde_json::to_vec(&state).unwrap(),
        &crate::codec::encode_kernel(&kernel).unwrap(),
    );
    let samples = sample_mixed(&artifact, 1000, Some(7)).unwrap();
    assert!(samples.values.chunks_exact(5).all(|row| match &row[2] {
        SampleCell::Category(v) if v == "Aa" => row[4] == SampleCell::Number(219.0),
        SampleCell::Category(v) if v == "BB" => row[4] == SampleCell::Number(100.0),
        _ => true,
    }));
}

#[test]
fn affine_units_and_constant_target_survive_mixed_reload() {
    let rows = (0..120).collect::<Vec<_>>();
    let a = compile_mixed(
        &fixture("held", 1.0),
        &rows,
        &options(),
        32768,
        MixedProfile::NumericSplitV1,
    )
    .unwrap();
    let b = compile_mixed(
        &fixture("held", 10.0),
        &rows,
        &options(),
        32768,
        MixedProfile::NumericSplitV1,
    )
    .unwrap();
    assert_eq!(a.report.selected_candidate, b.report.selected_candidate);
    let a = sample_mixed(&a.artifact, 128, Some(23)).unwrap();
    let b = sample_mixed(&b.artifact, 128, Some(23)).unwrap();
    for (a, b) in a.values.chunks_exact(5).zip(b.values.chunks_exact(5)) {
        for index in [0, 4] {
            let (SampleCell::Number(a), SampleCell::Number(b)) = (&a[index], &b[index]) else {
                panic!("numeric")
            };
            assert!((a - b / 10.0).abs() < 1e-4);
        }
        assert_eq!(a[1..4], b[1..4]);
    }
    let mut constant = fixture("held", 1.0);
    assert_eq!(constant.schema.columns.last().unwrap().id, "c0005");
    constant
        .columns
        .last_mut()
        .unwrap()
        .fill(Cell::Number(300.0));
    let result = compile_mixed(
        &constant,
        &rows,
        &options(),
        32768,
        MixedProfile::NumericSplitV1,
    )
    .unwrap();
    assert!(
        sample_mixed(&result.artifact, 128, Some(7))
            .unwrap()
            .values
            .chunks_exact(5)
            .all(|row| row[4] == SampleCell::Number(300.0))
    );
}

#[test]
fn malformed_state_unknown_categories_caps_and_unavailable_are_static() {
    let mut table = fixture("held_secret", 1.0);
    let rows = (0..120).collect::<Vec<_>>();
    let original = compile_mixed(
        &table,
        &rows,
        &options(),
        32768,
        MixedProfile::NumericSplitV1,
    )
    .unwrap();
    let (state, payload) = unpack(&original.artifact).unwrap();
    for at in [8, 16, 24, 56, original.artifact.len() - 1] {
        let mut bad = original.artifact.clone();
        bad[at] ^= 1;
        assert!(inspect_mixed(&bad).is_err());
    }
    let mut bad = state.clone();
    bad.features[1].split.as_mut().unwrap().anchor = 1;
    assert!(inspect_mixed(&pack(&serde_json::to_vec(&bad).unwrap(), payload)).is_err());
    let mut bad = state.clone();
    bad.features[1].probabilities[1] = -0.1;
    assert!(inspect_mixed(&pack(&serde_json::to_vec(&bad).unwrap(), payload)).is_err());
    let mut bad = state.clone();
    bad.features[2].categories[1] = bad.features[2].categories[0].clone();
    assert!(inspect_mixed(&pack(&serde_json::to_vec(&bad).unwrap(), payload)).is_err());
    let mut options = options();
    options.deadline = Some(Duration::ZERO);
    assert!(
        compile_mixed(&table, &rows, &options, 32768, MixedProfile::NumericSplitV1)
            .err()
            .unwrap()
            .to_string()
            .contains("mixed_preparation_deadline_exhausted")
    );
    assert!(sample_mixed(&original.artifact, 100001, Some(1)).is_err());
    table.columns[0].fill(Cell::Number(1.0));
    let error = compile_mixed(
        &table,
        &rows,
        &self::options(),
        32768,
        MixedProfile::NumericSplitV1,
    )
    .err()
    .unwrap()
    .to_string();
    assert!(
        error.contains("mixed_conditional_anchor_unavailable") && !error.contains("held_secret")
    );
    assert!(
        compile_mixed(
            &table,
            &rows,
            &self::options(),
            32768,
            MixedProfile::IndependentV1
        )
        .is_ok()
    );
    assert_eq!(table.schema.columns.last().unwrap().id, "c0005");
    table.columns.last_mut().unwrap()[0] = Cell::Mask(MaskState::Missing);
    assert!(
        compile_mixed(
            &table,
            &rows,
            &self::options(),
            32768,
            MixedProfile::IndependentV1
        )
        .is_err()
    );
}
