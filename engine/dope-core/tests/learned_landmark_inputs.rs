//! Tiny public graph/input conformance; no model-quality or privacy claim.
use candle_core::{Device, Tensor};
use dope_core::{
    learned::{self, manifest::Schema, EncoderInputs, LocalEncoding},
    levels::Level,
};
use std::path::PathBuf;
fn root(version: &str, name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("tests/fixtures/learned-{version}"))
        .join(name)
}
fn close(a: &Tensor, b: &Tensor, tolerance: f32) {
    assert_eq!(a.dims(), b.dims());
    let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let error = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(error <= tolerance, "error {error} exceeds {tolerance}");
}
#[test]
fn new_landmark_input_graphs_match_all_tiers_and_preserve_target_free_isolation() {
    for (version, name) in [
        ("v3", "input"),
        ("v3", "masked"),
        ("v4", "input"),
        ("v4", "masked"),
    ] {
        let r = root(version, name);
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(r.join("fixture.json")).unwrap()).unwrap();
        let schema: Schema = serde_json::from_value(meta["schema"].clone()).unwrap();
        let t = candle_core::safetensors::load(r.join("parity.safetensors"), &Device::Cpu).unwrap();
        let features: [f32; 46] = t["landmark_inputs"]
            .to_vec1::<f32>()
            .unwrap()
            .try_into()
            .unwrap();
        let (encoder, model) = learned::load_encoder(&r.join("model")).unwrap();
        let (decoder, _) = learned::load_decoder(&r.join("model")).unwrap();
        assert!(encoder
            .encode(&t["values"], &t["observed"], &schema, true, 1024)
            .is_err());
        let full = encoder
            .encode_with_landmarks(
                &t["values"],
                &t["observed"],
                &schema,
                true,
                10240,
                Some(&features),
            )
            .unwrap();
        for (level, width) in Level::ALL.iter().zip([1024, 4096, 8192, 10240]) {
            let code = encoder
                .encode_with_landmarks(
                    &t["values"],
                    &t["observed"],
                    &schema,
                    true,
                    width,
                    Some(&features),
                )
                .unwrap();
            close(&code, &t[&format!("code_{}", level.name())], 0.002);
            close(&code, &full.narrow(0, 0, width).unwrap(), 0.);
            let (n, c, m) = decoder
                .forward(
                    &t[&format!("code_{}", level.name())],
                    &schema,
                    &t["noisy"],
                    &t["noisy_observed"],
                    0.37,
                )
                .unwrap();
            close(&n, &t[&format!("numeric_{}", level.name())], 2e-4);
            close(&c, &t[&format!("categorical_{}", level.name())], 2e-4);
            close(&m, &t[&format!("missing_{}", level.name())], 2e-4);
            let local = LocalEncoding::encode_inputs(
                &encoder,
                model.clone(),
                &t["values"],
                &t["observed"],
                schema.clone(),
                EncoderInputs {
                    target_included: true,
                    through: *level,
                    landmarks: Some(&features),
                },
            )
            .unwrap();
            let raw = local.export(&[*level], 65536).unwrap();
            let (read, inventory) = learned::read_code(raw.bytes(), &model).unwrap();
            close(&read, &code, 0.);
            assert_eq!(inventory.views[0].length, width);
        }
        let free = encoder
            .encode(&t["values"], &t["observed"], &schema, false, 8192)
            .unwrap();
        close(&free, &t["code_target_free"], 0.002);
        let mut other = features;
        other[0] += 0.2;
        let other_code = encoder
            .encode_with_landmarks(
                &t["values"],
                &t["observed"],
                &schema,
                true,
                10240,
                Some(&other),
            )
            .unwrap();
        let mut start = 0;
        for width in [1024, 3072, 4096, 2048] {
            close(
                &full.narrow(0, start, width / 2).unwrap(),
                &other_code.narrow(0, start, width / 2).unwrap(),
                0.,
            );
            start += width;
        }
        if name == "masked" {
            close(&full, &other_code, 0.);
        }
        let target_free = encoder
            .encode_with_landmarks(
                &t["values"],
                &t["observed"],
                &schema,
                false,
                8192,
                Some(&other),
            )
            .unwrap();
        close(&free, &target_free, 0.);
        let mut invalid = features;
        invalid[0] = f32::NAN;
        assert!(encoder
            .encode_with_landmarks(
                &t["values"],
                &t["observed"],
                &schema,
                true,
                1024,
                Some(&invalid)
            )
            .is_err());
        // Sampling depends only on the code/schema and decoder. No landmark file enters this call.
        for (level, width) in Level::ALL.iter().zip([1024, 4096, 8192, 10240]) {
            let (code, meta) = learned::read_code(
                &std::fs::read(r.join(format!("{}.code", level.name()))).unwrap(),
                &model,
            )
            .unwrap();
            assert_eq!(code.dims(), [width]);
            assert!(learned::sample(&decoder, &code, &meta.schema, 4, 101, 8)
                .unwrap()
                .rows
                .iter()
                .all(|r| r.last().unwrap().is_some()));
        }
    }
}
