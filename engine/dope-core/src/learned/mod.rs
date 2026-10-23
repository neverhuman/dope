//! Experimental frozen table codes and code-only diffusion, using a closed CPU graph.
//! Model weights are explicit local dependencies. These APIs grant no certification.
mod blocks;
mod carrier;
pub mod decoder;
pub mod intake;
pub mod manifest;
mod sampling;
pub mod table_encoder;
mod tokenizer;
mod weights;

use crate::{
    levels::{digest, Level},
    Result,
};
pub use carrier::{read_code, ExportedCode, LocalEncoding};
use manifest::{bad, check, ModelManifest, PREPROCESSING};
pub use sampling::{sample, sample_with_schedule, CategoricalSchedule, LearnedSamples};
use std::path::Path;

pub fn read_manifest(root: &Path) -> Result<ModelManifest> {
    let path = root.join("model.json");
    check(
        std::fs::symlink_metadata(&path)?.is_file() && std::fs::metadata(&path)?.len() <= 262144,
        "learned model manifest limit",
    )?;
    let raw = std::fs::read(path)?;
    let mut value: serde_json::Value =
        serde_json::from_slice(&raw).map_err(|_| bad("learned model manifest invalid"))?;
    let identity = value
        .as_object_mut()
        .and_then(|v| v.remove("coordinate_space_id"))
        .and_then(|v| v.as_str().map(str::to_string))
        .ok_or_else(|| bad("missing learned coordinate identity"))?;
    check(
        digest(&serde_json::to_vec(&value)?) == identity,
        "learned coordinate identity mismatch",
    )?;
    let model: ModelManifest =
        serde_json::from_slice(&raw).map_err(|_| bad("learned model manifest invalid"))?;
    model.graph.validate()?;
    for identity in [&model.source_sha256, &model.training_manifest_sha256] {
        check(
            identity.len() == 64
                && identity
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "learned provenance identity invalid",
        )?;
    }
    check(
        model.format == "dope-shared-model-v1"
            && model.layout == model.graph.layout
            && model.preprocessing == PREPROCESSING
            && model.quantization == "fixed-fp16-v1"
            && model.dtype == "F32"
            && !model.certified
            && model.output_schema == "positional_numeric_and_category_ids"
            && model.supported_backends == ["cpu_reference"]
            && matches!(
                model.evidence.as_str(),
                "synthetic_development_only" | "unqualified_development"
            )
            && model.weights.len() == 2
            && model.weights.contains_key("encoder")
            && model.weights.contains_key("decoder"),
        "unsupported learned model manifest",
    )?;
    let levels = Level::parse_list(&model.trained_levels.join(","))?;
    check(
        levels == [Level::Low, Level::Medium, Level::High] || levels == Level::ALL,
        "unsupported trained learned levels",
    )?;
    check(
        levels.iter().map(|l| l.name()).collect::<Vec<_>>() == model.trained_levels,
        "noncanonical learned levels",
    )?;
    for (component, record) in &model.weights {
        check(
            record.file == format!("{component}.safetensors")
                && record.bytes <= 536870912
                && record.parameters <= 100000000
                && record.sha256.len() == 64
                && record
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "learned weight inventory invalid",
        )?;
    }
    check(
        model.weights.values().map(|v| v.parameters).sum::<usize>() <= 100000000,
        "learned aggregate parameters exceeded",
    )?;
    Ok(model)
}

/// Encoding does not open decoder weights.
pub fn load_encoder(root: &Path) -> Result<(table_encoder::TableEncoder, ModelManifest)> {
    let manifest = read_manifest(root)?;
    let mut weights = weights::Weights::load(root, "encoder", &manifest.weights["encoder"])?;
    let encoder = table_encoder::TableEncoder::load(&mut weights, &manifest.graph)?;
    weights.finish()?;
    Ok((encoder, manifest))
}

/// Source-free sampling does not open encoder weights, raw contexts or the network.
pub fn load_decoder(root: &Path) -> Result<(decoder::Decoder, ModelManifest)> {
    let manifest = read_manifest(root)?;
    let mut weights = weights::Weights::load(root, "decoder", &manifest.weights["decoder"])?;
    let decoder = decoder::Decoder::load(&mut weights, &manifest.graph)?;
    weights.finish()?;
    Ok((decoder, manifest))
}
