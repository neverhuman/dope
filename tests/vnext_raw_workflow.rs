use dope_kernel::compiler::CompileOptions;
use dope_kernel::schema::TableSchemaV1;
use dope_kernel::typed_data::workflow::{compile_raw, inspect_raw, predict_raw, sample_raw};
use dope_kernel::typed_data::{InputLimits, TypedTable};
use serde_json::json;

#[path = "support/raw_residual_profiles.rs"]
mod raw_residual_profiles;

#[path = "support/raw_mixed_cli.rs"]
mod raw_mixed_cli;

fn table(x_scale: f64, y_scale: f64, query: f64) -> TypedTable {
    let schema = TableSchemaV1::from_private_json(
        &serde_json::to_vec(&json!({
            "format":"dope-table-schema", "version":1,
            "columns":[
                {"id":"c0001","source_name":"secret_x","role":"feature","dtype":"continuous"},
                {"id":"c0002","source_name":"secret_y","role":"target","dtype":"continuous"}
            ]
        }))
        .unwrap(),
    )
    .unwrap();
    let mut csv = String::from("secret_y,secret_x\n");
    for row in 0..64 {
        let x = row as f64 / 63.0;
        csv.push_str(&format!(
            "{},{}\n",
            (100.0 + 200.0 * x) * y_scale,
            (10.0 + 20.0 * x) * x_scale
        ));
    }
    csv.push_str(&format!("{},{}\n", query * y_scale, query * x_scale));
    TypedTable::read_csv(csv.as_bytes(), schema, InputLimits::default()).unwrap()
}

fn options() -> CompileOptions {
    CompileOptions {
        seed: Some(1729),
        backend_id: Some("independent_quantile".into()),
        ..CompileOptions::default()
    }
}

#[test]
fn original_unit_workflow_reloads_and_query_labels_do_not_change_artifact() {
    let a = compile_raw(
        &table(1.0, 1.0, 99999.0),
        &(0..64).collect::<Vec<_>>(),
        &options(),
        32768,
    )
    .unwrap();
    let b = compile_raw(
        &table(1.0, 1.0, -99999.0),
        &(0..64).collect::<Vec<_>>(),
        &options(),
        32768,
    )
    .unwrap();
    assert_eq!(a.artifact, b.artifact);
    let predictions = predict_raw(&a.artifact, &[10.0, 20.0, 30.0]).unwrap();
    // The fixed bounded compiler is regularized; its prediction is not the
    // ground-truth teacher. Freeze the decoded control's actual golden values.
    assert_eq!(predictions, [115.625, 200.0, 284.375]);
    let mse = predictions
        .iter()
        .zip([100.0, 200.0, 300.0])
        .map(|(p, y)| (p - y).powi(2))
        .sum::<f64>()
        / 3.0;
    assert!((mse - 162.76041666666666).abs() < 1e-9);
    let samples = sample_raw(&a.artifact, 128, Some(91)).unwrap();
    assert_eq!(samples, sample_raw(&a.artifact, 128, Some(91)).unwrap());
    assert!(
        samples
            .chunks_exact(2)
            .all(|row| row[0] >= 10.0 && row[0] <= 30.0 && row[1].is_finite())
    );
    let public = serde_json::to_string(&a.inspection).unwrap();
    for secret in ["secret_x", "secret_y", "99999", "\"offset\":", "\"scale\":"] {
        assert!(!public.contains(secret));
    }
}

#[test]
fn affine_units_preserve_decisions_and_original_unit_predictions() {
    let rows = (0..64).collect::<Vec<_>>();
    let a = compile_raw(&table(1.0, 1.0, 1000.0), &rows, &options(), 32768).unwrap();
    let b = compile_raw(&table(100.0, 0.001, 1000.0), &rows, &options(), 32768).unwrap();
    assert_eq!(a.report.selected_candidate, b.report.selected_candidate);
    let pa = predict_raw(&a.artifact, &[10.0, 20.0, 30.0]).unwrap();
    let pb = predict_raw(&b.artifact, &[1000.0, 2000.0, 3000.0]).unwrap();
    for (a, b) in pa.iter().zip(pb) {
        assert!((a - b * 1000.0).abs() < 0.001);
    }
}

#[test]
fn exact_complete_cap_and_integrity_are_enforced() {
    let table = table(1.0, 1.0, 9000.0);
    let rows = (0..64).collect::<Vec<_>>();
    let a = compile_raw(&table, &rows, &options(), 32768).unwrap();
    let account = &a.inspection.accounting;
    assert_eq!(
        account.framing_bytes + account.transform_schema_bytes + account.native_payload_bytes,
        a.artifact.len()
    );
    assert_eq!(account.complete_sampling_bytes, a.artifact.len());
    assert!(compile_raw(&table, &rows, &options(), a.artifact.len()).is_ok());
    assert!(compile_raw(&table, &rows, &options(), a.artifact.len() - 1).is_err());
    for position in [8, 16, 24, 56, a.artifact.len() - 1] {
        let mut corrupt = a.artifact.clone();
        corrupt[position] ^= 1;
        assert!(inspect_raw(&corrupt).is_err());
    }
    let mut extra = a.artifact;
    extra.push(0);
    assert!(inspect_raw(&extra).is_err());
}

#[test]
fn successor_target_keeps_unclipped_noise_while_historical_dpk_stays_clipped() {
    let a = compile_raw(
        &table(1.0, 1.0, 999.0),
        &(0..64).collect::<Vec<_>>(),
        &options(),
        32768,
    )
    .unwrap();
    let state_len = u64::from_le_bytes(a.artifact[8..16].try_into().unwrap()) as usize;
    let state = &a.artifact[56..56 + state_len];
    let mut kernel = dope_kernel::codec::decode_kernel(&a.artifact[56 + state_len..]).unwrap();
    let dope_kernel::model::KernelProgram::Symbolic { noise, .. } = &mut kernel.program else {
        panic!("symbolic");
    };
    *noise = dope_kernel::model::Noise::Homoscedastic { sigma: 0.8 };
    let payload = dope_kernel::codec::encode_kernel(&kernel).unwrap();
    let mut content = state.to_vec();
    content.extend_from_slice(&payload);
    let mut artifact = b"DOPERAW1".to_vec();
    artifact.extend_from_slice(&(state.len() as u64).to_le_bytes());
    artifact.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    artifact.extend_from_slice(blake3::hash(&content).as_bytes());
    artifact.extend_from_slice(&content);
    let raw = sample_raw(&artifact, 512, Some(71)).unwrap();
    assert!(raw.chunks_exact(2).any(|row| row[1] < 100.0));
    assert!(raw.chunks_exact(2).any(|row| row[1] > 300.0));
    let old = dope_kernel::sample::sample_kernel(
        &dope_kernel::codec::LoadedKernel::V3(Box::new(kernel)),
        dope_kernel::sample::SampleOptions {
            rows: 512,
            seed: Some(71),
        },
    )
    .unwrap();
    assert!(old.chunks_exact(2).all(|row| (0.0..=1.0).contains(&row[1])));
}

#[test]
fn constant_target_is_explicit_and_mixed_capability_refuses() {
    let declaration = json!({"format":"dope-table-schema","version":1,"columns":[
        {"id":"c0001","source_name":"private_x","role":"feature","dtype":"continuous"},
        {"id":"c0002","source_name":"private_y","role":"target","dtype":"continuous"}
    ]});
    let parse = |d: &serde_json::Value, csv: &str| {
        TypedTable::read_csv(
            csv.as_bytes(),
            TableSchemaV1::from_private_json(&serde_json::to_vec(d).unwrap()).unwrap(),
            InputLimits::default(),
        )
        .unwrap()
    };
    let constant = parse(
        &declaration,
        "private_x,private_y\n10,300\n20,300\n30,300\n",
    );
    let result = compile_raw(&constant, &[0, 1, 2], &options(), 32768).unwrap();
    assert_eq!(
        result.inspection.constant_target_rule,
        "unit_scale_with_fitted_offset"
    );
    assert!(
        predict_raw(&result.artifact, &[10.0, 30.0])
            .unwrap()
            .iter()
            .all(|v| (*v - 300.0).abs() < 0.01)
    );
    let mut nominal = declaration;
    nominal["columns"][0]["dtype"] = json!("nominal");
    let mixed = parse(
        &nominal,
        "private_x,private_y\nsecret_cat,1\nsecret_other,2\n",
    );
    let error = compile_raw(&mixed, &[0, 1], &options(), 32768)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("raw_profile_requires_numeric_columns"));
    assert!(!error.contains("secret_cat") && !error.contains("private_x"));
}

#[test]
fn numeric_predictor_types_survive_reload_with_counted_repairs() {
    let declaration = json!({"format":"dope-table-schema","version":1,"columns":[
        {"id":"c0001","source_name":"frequency","role":"feature","dtype":"integer","domain":{"kind":"numeric","lower":0}},
        {"id":"c0002","source_name":"flag","role":"feature","dtype":"boolean"},
        {"id":"c0003","source_name":"target","role":"target","dtype":"continuous"}
    ]});
    let csv = "frequency,flag,target\n10,true,10\n11,false,20\n13,true,30\n19,false,40\n";
    let table = TypedTable::read_csv(
        csv.as_bytes(),
        TableSchemaV1::from_private_json(&serde_json::to_vec(&declaration).unwrap()).unwrap(),
        InputLimits::default(),
    )
    .unwrap();
    let result = compile_raw(&table, &[0, 1, 2, 3], &options(), 32768).unwrap();
    assert_eq!(result.inspection.profile, "raw-numeric-symbolic-v1");
    let detailed =
        dope_kernel::typed_data::workflow::sample_raw_detailed(&result.artifact, 128, Some(87))
            .unwrap();
    assert!(
        detailed
            .values
            .chunks_exact(3)
            .all(|row| row[0].fract() == 0.0 && row[0] >= 0.0 && (row[1] == 0.0 || row[1] == 1.0))
    );
    assert_eq!(detailed.physical_domain_rejections, 0);
    assert_eq!(
        detailed.values,
        sample_raw(&result.artifact, 128, Some(87)).unwrap()
    );
}
