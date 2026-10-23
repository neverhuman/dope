//! Tiny public synthetic CPU graph fixtures; no quality or privacy qualification.
use candle_core::{Device, Tensor};
use dope_core::{
    learned::{self, manifest::Schema, LocalEncoding},
    levels::Level,
};
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/learned-v1")
}
fn fixture() -> (serde_json::Value, std::collections::HashMap<String, Tensor>) {
    let raw = std::fs::read(root().join("fixture.json")).unwrap();
    let meta = serde_json::from_slice(&raw).unwrap();
    let weights =
        candle_core::safetensors::load(root().join("parity.safetensors"), &Device::Cpu).unwrap();
    (meta, weights)
}
fn close(got: &Tensor, expected: &Tensor, tolerance: f32) {
    assert_eq!(got.dims(), expected.dims());
    let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let maximum = got
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        maximum <= tolerance,
        "maximum absolute error {maximum}, tolerance {tolerance}"
    );
}

#[test]
fn encoder_codes_and_selected_prefixes_match_pytorch_fixture() {
    let (meta, tensors) = fixture();
    let schema: Schema = serde_json::from_value(meta["schema"].clone()).unwrap();
    let (encoder, manifest) = learned::load_encoder(&root().join("model")).unwrap();
    let high = encoder
        .encode(
            &tensors["values"],
            &tensors["observed"],
            &schema,
            true,
            4096,
        )
        .unwrap();
    close(&high, &tensors["code_high"], 0.002);
    let target_free = encoder
        .encode(
            &tensors["values"],
            &tensors["observed"],
            &schema,
            false,
            4096,
        )
        .unwrap();
    close(&target_free, &tensors["code_target_free"], 0.002);
    let lower = LocalEncoding::encode(
        &encoder,
        manifest.clone(),
        &tensors["values"],
        &tensors["observed"],
        schema.clone(),
        true,
        Level::Low,
    )
    .unwrap()
    .export(&[Level::Low], 65536)
    .unwrap();
    let from_high = LocalEncoding::encode(
        &encoder,
        manifest.clone(),
        &tensors["values"],
        &tensors["observed"],
        schema,
        true,
        Level::High,
    )
    .unwrap()
    .export(&[Level::Low], 65536)
    .unwrap();
    assert_eq!(lower.bytes(), from_high.bytes());
    let (code, inventory) = learned::read_code(lower.bytes(), &manifest).unwrap();
    assert_eq!(code.dims(), [256]);
    assert_eq!(inventory.exported_levels, ["low"]);
}

#[test]
fn decoder_operators_match_pytorch_with_identical_code_and_noisy_inputs() {
    let (meta, tensors) = fixture();
    let schema: Schema = serde_json::from_value(meta["schema"].clone()).unwrap();
    let (decoder, _) = learned::load_decoder(&root().join("model")).unwrap();
    let code = tensors["code_high"].narrow(0, 0, 256).unwrap();
    let (numeric, categorical, missing) = decoder
        .forward(
            &code,
            &schema,
            &tensors["noisy"],
            &tensors["noisy_observed"],
            0.37,
        )
        .unwrap();
    close(&numeric, &tensors["numeric"], 2e-4);
    close(&categorical, &tensors["categorical"], 2e-4);
    close(&missing, &tensors["missing"], 2e-4);
}

#[test]
fn recipient_needs_only_code_model_manifest_and_decoder_weights() {
    let staging = std::env::temp_dir().join(format!("dope-code-only-{}", std::process::id()));
    std::fs::create_dir(&staging).unwrap();
    for name in ["model.json", "decoder.safetensors"] {
        std::fs::copy(root().join("model").join(name), staging.join(name)).unwrap();
    }
    let (decoder, manifest) = learned::load_decoder(&staging).unwrap();
    let raw = std::fs::read(root().join("low.code")).unwrap();
    let (code, inventory) = learned::read_code(&raw, &manifest).unwrap();
    let rows = learned::sample(&decoder, &code, &inventory.schema, 16, 42, 8).unwrap();
    let again = learned::sample(&decoder, &code, &inventory.schema, 16, 42, 8).unwrap();
    assert_eq!(rows.rows, again.rows);
    for row in &rows.rows {
        assert_eq!(row.len(), inventory.schema.columns.len());
        assert!(row.last().unwrap().is_some());
        for (value, column) in row.iter().zip(&inventory.schema.columns) {
            if let Some(value) = value {
                assert!(value.is_finite());
                if column.kind == "categorical" {
                    assert!(
                        *value >= 2. && *value < column.categories as f64 && value.fract() == 0.
                    );
                }
            }
        }
    }
    assert!(!staging.join("encoder.safetensors").exists());
    assert!(learned::load_encoder(&staging).is_err());
    std::fs::remove_dir_all(staging).unwrap();
}

#[test]
fn unknown_graph_space_extra_tensors_and_truncated_codes_refuse() {
    let manifest = learned::read_manifest(&root().join("model")).unwrap();
    let raw = std::fs::read(root().join("low.code")).unwrap();
    for end in [0, 8, raw.len() - 1] {
        assert!(learned::read_code(&raw[..end], &manifest).is_err());
    }
    let header_len = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let mut header: serde_json::Value = serde_json::from_slice(&raw[8..8 + header_len]).unwrap();
    header["extra"] = serde_json::json!({"dtype":"F16","shape":[0],"data_offsets":[512,512]});
    let text = serde_json::to_vec(&header).unwrap();
    let mut mutated = (text.len() as u64).to_le_bytes().to_vec();
    mutated.extend(text);
    mutated.extend(&raw[8 + header_len..]);
    assert!(learned::read_code(&mutated, &manifest).is_err());
    let mut other = manifest.clone();
    other.coordinate_space_id = "0".repeat(64);
    assert!(learned::read_code(&raw, &other).is_err());
    let mut graph = manifest.graph.clone();
    graph.width = usize::MAX;
    assert!(graph.validate().is_err());
    graph = manifest.graph.clone();
    graph.graph = "untrusted-operator".into();
    assert!(graph.validate().is_err());
}

#[test]
fn physical_context_selection_is_target_independent_and_exports_no_labels() {
    let staging = std::env::temp_dir().join(format!("dope-learned-intake-{}", std::process::id()));
    std::fs::create_dir(&staging).unwrap();
    let path = staging.join("context.csv");
    let (encoder, model) = learned::load_encoder(&root().join("model")).unwrap();
    let kinds = vec!["numeric".into(), "categorical".into(), "numeric".into()];
    let csv = |flip: bool| {
        let mut text = String::from("private_feature,private_category,private_target\n");
        for i in 0..80 {
            text += &format!(
                "{},{},{}\n",
                i,
                if i % 2 == 0 {
                    "PRIVATE_CANARY_A"
                } else {
                    "PRIVATE_CANARY_B"
                },
                if flip { -i } else { i }
            );
        }
        text
    };
    std::fs::write(&path, csv(false)).unwrap();
    let first = learned::intake::read_context(
        &path,
        "private_target",
        &kinds,
        "regression",
        &model.graph,
        17,
    )
    .unwrap();
    std::fs::write(&path, csv(true)).unwrap();
    let second = learned::intake::read_context(
        &path,
        "private_target",
        &kinds,
        "regression",
        &model.graph,
        17,
    )
    .unwrap();
    assert_eq!(first.physical_rows, second.physical_rows);
    assert_eq!(
        first.schema.columns[0].center,
        second.schema.columns[0].center
    );
    let export = |context: learned::intake::Context| {
        learned::LocalEncoding::encode(
            &encoder,
            model.clone(),
            &context.values,
            &context.observed,
            context.schema,
            false,
            Level::Low,
        )
        .unwrap()
        .export(&[Level::Low], 65536)
        .unwrap()
    };
    let a = export(first);
    let b = export(second);
    assert_eq!(a.bytes(), b.bytes());
    let text = String::from_utf8_lossy(a.bytes());
    assert!(!text.contains("PRIVATE_CANARY"));
    assert!(!text.contains("private_feature"));
    std::fs::remove_dir_all(staging).unwrap();
}
