//! Public tiny two-update fixtures; these qualify operators, not model utility.
use candle_core::{Device, Tensor};
use dope_core::{
    learned::{self, manifest::Schema, LocalEncoding},
    levels::Level,
};
use std::path::PathBuf;

fn root(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/learned-v2")
        .join(name)
}
fn close(a: &Tensor, b: &Tensor, tolerance: f32) {
    assert_eq!(a.dims(), b.dims());
    let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let error = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(
        error <= tolerance,
        "maximum error {error}, tolerance {tolerance}"
    );
}

#[test]
fn all_v2_graphs_match_codes_and_decoder_operators_for_every_tier() {
    for name in [
        "slot",
        "matched",
        "pooled",
        "slot-bounded",
        "matched-bounded",
        "pooled-bounded",
    ] {
        let path = root(name);
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("fixture.json")).unwrap()).unwrap();
        let schema: Schema = serde_json::from_value(meta["schema"].clone()).unwrap();
        let tensors =
            candle_core::safetensors::load(path.join("parity.safetensors"), &Device::Cpu).unwrap();
        let (encoder, model) = learned::load_encoder(&path.join("model")).unwrap();
        let (decoder, _) = learned::load_decoder(&path.join("model")).unwrap();
        let full = encoder
            .encode(
                &tensors["values"],
                &tensors["observed"],
                &schema,
                true,
                10240,
            )
            .unwrap();
        for (level, width) in Level::ALL.iter().zip([1024, 4096, 8192, 10240]) {
            let code = encoder
                .encode(
                    &tensors["values"],
                    &tensors["observed"],
                    &schema,
                    true,
                    width,
                )
                .unwrap();
            close(&code, &tensors[&format!("code_{}", level.name())], 0.002);
            if model.graph.bounded_code() {
                assert!(code.to_vec1::<f32>().unwrap().iter().all(|v| v.abs() <= 1.));
            }
            close(&code, &full.narrow(0, 0, width).unwrap(), 0.);
            let expected = &tensors[&format!("code_{}", level.name())];
            let (n, c, m) = decoder
                .forward(
                    expected,
                    &schema,
                    &tensors["noisy"],
                    &tensors["noisy_observed"],
                    0.37,
                )
                .unwrap();
            close(&n, &tensors[&format!("numeric_{}", level.name())], 2e-4);
            close(&c, &tensors[&format!("categorical_{}", level.name())], 2e-4);
            close(&m, &tensors[&format!("missing_{}", level.name())], 2e-4);
            let raw = LocalEncoding::encode(
                &encoder,
                model.clone(),
                &tensors["values"],
                &tensors["observed"],
                schema.clone(),
                true,
                *level,
            )
            .unwrap()
            .export(&[*level], 65536)
            .unwrap();
            let from_full = LocalEncoding::encode(
                &encoder,
                model.clone(),
                &tensors["values"],
                &tensors["observed"],
                schema.clone(),
                true,
                Level::Ultra,
            )
            .unwrap()
            .export(&[*level], 65536)
            .unwrap();
            assert_eq!(raw.bytes(), from_full.bytes());
            let (read, inventory) = learned::read_code(raw.bytes(), &model).unwrap();
            assert_eq!(read.dims(), [width]);
            assert_eq!(inventory.views[0].length, width);
        }
        let free = encoder
            .encode(
                &tensors["values"],
                &tensors["observed"],
                &schema,
                false,
                8192,
            )
            .unwrap();
        close(&free, &tensors["code_target_free"], 0.002);
    }
}

#[test]
fn bounded_graph_refuses_out_of_range_carrier_instead_of_clamping() {
    let path = root("slot-bounded");
    let model = learned::read_manifest(&path.join("model")).unwrap();
    let mut raw = std::fs::read(path.join("low.code")).unwrap();
    let offset = 8 + u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    raw[offset..offset + 2].copy_from_slice(&0x3e00u16.to_le_bytes());
    assert!(learned::read_code(&raw, &model).is_err());
}

#[test]
fn recipient_without_encoder_samples_all_v2_tiers_and_rejects_v1_mismatch() {
    let path = root("slot");
    let staging = std::env::temp_dir().join(format!(
        "dope-v2-recipient-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&staging).unwrap();
    for name in ["model.json", "decoder.safetensors"] {
        std::fs::copy(path.join("model").join(name), staging.join(name)).unwrap();
    }
    let (decoder, model) = learned::load_decoder(&staging).unwrap();
    assert!(!staging.join("encoder.safetensors").exists());
    for level in Level::ALL {
        let raw = std::fs::read(path.join(format!("{}.code", level.name()))).unwrap();
        let (code, inventory) = learned::read_code(&raw, &model).unwrap();
        let a = learned::sample(&decoder, &code, &inventory.schema, 4, 111, 8).unwrap();
        let b = learned::sample(&decoder, &code, &inventory.schema, 4, 111, 8).unwrap();
        assert_eq!(a.rows, b.rows);
        assert!(a.rows.iter().all(|r| r.last().unwrap().is_some()));
        let mut wrong = model.clone();
        wrong.layout = learned::manifest::LAYOUT.into();
        assert!(learned::read_code(&raw, &wrong).is_err());
    }
    let mut graph = model.graph.clone();
    graph.layout = learned::manifest::LAYOUT.into();
    assert!(graph.validate().is_err());
    let mut graph = model.graph.clone();
    graph.summary_slots = usize::MAX;
    assert!(graph.validate().is_err());
    std::fs::remove_dir_all(staging).unwrap();
}
