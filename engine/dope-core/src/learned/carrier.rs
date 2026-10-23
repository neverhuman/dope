//! Exact selected-code carrier. Private higher states are not serialized or hashed.
use super::{
    manifest::{bad, check, ModelManifest, Schema, PREPROCESSING},
    table_encoder::TableEncoder,
};
use crate::{levels::Level, Result};
use candle_core::{Device, Tensor};
use half::f16;
use safetensors::{Dtype, SafeTensors, View};
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    collections::{BTreeSet, HashMap},
};

#[derive(Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CodeView {
    pub level: String,
    pub offset: usize,
    pub length: usize,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodeManifest {
    pub format: String,
    pub representation_kind: String,
    pub layout: String,
    pub preprocessing: String,
    pub coordinate_space_id: String,
    pub exported_levels: Vec<String>,
    pub target_included: bool,
    pub dtype: String,
    pub schema: Schema,
    pub views: Vec<CodeView>,
    pub normalization: String,
    pub evidence: String,
    pub certified: bool,
}
pub struct LocalEncoding {
    code: Tensor,
    schema: Schema,
    model: ModelManifest,
    target_included: bool,
}
pub struct ExportedCode {
    bytes: Vec<u8>,
    pub manifest: CodeManifest,
}
impl ExportedCode {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}
struct Stored {
    bytes: Vec<u8>,
    shape: [usize; 1],
}
impl View for &Stored {
    fn dtype(&self) -> Dtype {
        Dtype::F16
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.bytes)
    }
    fn data_len(&self) -> usize {
        self.bytes.len()
    }
}
impl LocalEncoding {
    pub fn encode(
        encoder: &TableEncoder,
        model: ModelManifest,
        values: &Tensor,
        observed: &Tensor,
        mut schema: Schema,
        target_included: bool,
        through: Level,
    ) -> Result<Self> {
        check(
            model.graph == encoder.graph && model.layout == encoder.graph.layout,
            "learned encoder model binding mismatch",
        )?;
        check(
            model.trained_levels.iter().any(|l| l == through.name()),
            "requested learned level is untrained",
        )?;
        let code = encoder.encode(
            values,
            observed,
            &schema,
            target_included,
            model.graph.widths()[through as usize],
        )?;
        if !target_included {
            schema.columns.pop();
            schema.target_index = None;
            schema.task = None;
        }
        schema.validate(&model.graph)?;
        Ok(Self {
            code,
            schema,
            model,
            target_included,
        })
    }
    pub fn export(&self, selected: &[Level], max_bytes: usize) -> Result<ExportedCode> {
        let set: BTreeSet<_> = selected.iter().copied().collect();
        check(
            !selected.is_empty() && set.len() == selected.len(),
            "invalid learned export levels",
        )?;
        let selected: Vec<_> = set.into_iter().collect();
        let size = self.model.graph.widths()[*selected.last().unwrap() as usize];
        check(
            size <= self.code.dim(0)?
                && selected.iter().all(|l| {
                    self.model
                        .trained_levels
                        .iter()
                        .any(|name| name == l.name())
                }),
            "learned export view unavailable",
        )?;
        let values = self.code.narrow(0, 0, size)?.to_vec1::<f32>()?;
        check(
            values
                .iter()
                .all(|v| v.is_finite() && f16::from_f32(*v).is_finite()),
            "learned code nonfinite",
        )?;
        let stored = Stored {
            bytes: values
                .iter()
                .flat_map(|v| f16::from_f32(*v).to_le_bytes())
                .collect(),
            shape: [size],
        };
        let manifest = CodeManifest {
            format: "dope-table-code-v1".into(),
            representation_kind: "table_code".into(),
            layout: self.model.layout.clone(),
            preprocessing: PREPROCESSING.into(),
            coordinate_space_id: self.model.coordinate_space_id.clone(),
            exported_levels: selected.iter().map(|l| l.name().into()).collect(),
            target_included: self.target_included,
            dtype: "F16".into(),
            schema: self.schema.clone(),
            views: selected
                .iter()
                .map(|l| CodeView {
                    level: l.name().into(),
                    offset: 0,
                    length: self.model.graph.widths()[*l as usize],
                })
                .collect(),
            normalization: "raw_generation_code".into(),
            evidence: self.model.evidence.clone(),
            certified: false,
        };
        let metadata = HashMap::from([("manifest".into(), serde_json::to_string(&manifest)?)]);
        let bytes = safetensors::serialize([("code", &stored)], Some(metadata))
            .map_err(|_| bad("learned code serialization"))?;
        check(
            bytes.len() <= max_bytes && bytes.len() <= 1048576,
            "learned code payload cap exceeded",
        )?;
        Ok(ExportedCode { bytes, manifest })
    }
}

pub fn read_code(bytes: &[u8], model: &ModelManifest) -> Result<(Tensor, CodeManifest)> {
    model.graph.validate()?;
    check(
        model.layout == model.graph.layout,
        "learned layout binding mismatch",
    )?;
    check(
        (8..=1048576).contains(&bytes.len()),
        "learned code size limit",
    )?;
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    check(header_len <= 262144, "learned code header limit")?;
    let (n, header) = SafeTensors::read_metadata(bytes).map_err(|_| bad("learned code header"))?;
    let metadata = header
        .metadata()
        .as_ref()
        .ok_or_else(|| bad("learned code metadata missing"))?;
    check(metadata.len() == 1, "unexpected learned code metadata")?;
    let text = metadata
        .get("manifest")
        .ok_or_else(|| bad("learned code manifest missing"))?;
    let manifest: CodeManifest =
        serde_json::from_str(text).map_err(|_| bad("learned code manifest invalid"))?;
    check(
        manifest.format == "dope-table-code-v1"
            && manifest.representation_kind == "table_code"
            && manifest.layout == model.layout
            && manifest.preprocessing == PREPROCESSING
            && manifest.coordinate_space_id == model.coordinate_space_id
            && manifest.dtype == "F16"
            && manifest.normalization == "raw_generation_code"
            && !manifest.certified
            && manifest.evidence == model.evidence,
        "learned code space or manifest mismatch",
    )?;
    manifest.schema.validate(&model.graph)?;
    check(
        manifest.target_included == manifest.schema.target_index.is_some(),
        "learned target disclosure mismatch",
    )?;
    let levels = Level::parse_list(&manifest.exported_levels.join(","))?;
    check(
        levels.iter().map(|l| l.name()).collect::<Vec<_>>() == manifest.exported_levels
            && levels
                .iter()
                .all(|l| model.trained_levels.iter().any(|n| n == l.name()))
            && manifest.views
                == levels
                    .iter()
                    .map(|l| CodeView {
                        level: l.name().into(),
                        offset: 0,
                        length: model.graph.widths()[*l as usize],
                    })
                    .collect::<Vec<_>>(),
        "learned code views invalid",
    )?;
    let size = model.graph.widths()[*levels.last().unwrap() as usize];
    let st = SafeTensors::deserialize(bytes).map_err(|_| bad("learned code tensor table"))?;
    check(st.len() == 1, "unexpected learned code tensor")?;
    let tensor = st
        .tensor("code")
        .map_err(|_| bad("learned code tensor missing"))?;
    check(
        tensor.dtype() == Dtype::F16
            && tensor.shape() == [size]
            && n.checked_add(8).and_then(|n| n.checked_add(size * 2)) == Some(bytes.len()),
        "learned code inventory mismatch",
    )?;
    let values: Vec<f32> = tensor
        .data()
        .chunks_exact(2)
        .map(|v| f16::from_le_bytes(v.try_into().unwrap()).to_f32())
        .collect();
    check(
        values.iter().all(|v| v.is_finite()),
        "learned code nonfinite",
    )?;
    Ok((Tensor::from_vec(values, size, &Device::Cpu)?, manifest))
}
