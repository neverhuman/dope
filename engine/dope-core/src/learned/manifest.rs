//! Closed, bounded model/schema contracts. No custom operators or remote resolution.
use crate::{DopeError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const LAYOUT: &str = "nested-256-1024-4096-plus-1024-v1";
pub const LAYOUT_V2: &str = "nested-1024-4096-8192-plus-2048-v2";
pub const PREPROCESSING: &str = "fit-only-positional-mixed-v1";
pub const BLOCKS: [usize; 4] = [256, 768, 3072, 1024];
pub const WIDTHS: [usize; 4] = [256, 1024, 4096, 5120];
pub fn bad(code: &'static str) -> DopeError {
    DopeError::Artifact(code.into())
}
pub fn check(value: bool, code: &'static str) -> Result<()> {
    if value {
        Ok(())
    } else {
        Err(bad(code))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Graph {
    pub width: usize,
    pub heads: usize,
    pub feature_blocks: usize,
    pub inducing_blocks: usize,
    pub inducing_slots: usize,
    pub summary_slots: usize,
    pub target_blocks: usize,
    pub decoder_blocks: usize,
    pub condition_slots: usize,
    pub max_features: usize,
    pub max_categories: usize,
    pub max_context: usize,
    pub max_query: usize,
    pub cell_cap: usize,
    pub diffusion_steps: usize,
    pub layout: String,
    pub graph: String,
}
impl Graph {
    pub fn slot_readout(&self) -> bool {
        matches!(
            self.graph.as_str(),
            "slot-code-diffusion-v2"
                | "pooled-control-code-diffusion-v2"
                | "slot-bounded-code-diffusion-v2"
                | "pooled-control-bounded-code-diffusion-v2"
        )
    }
    pub fn bounded_code(&self) -> bool {
        matches!(
            self.graph.as_str(),
            "slot-bounded-code-diffusion-v2"
                | "pooled-control-bounded-code-diffusion-v2"
                | "pooled-bounded-code-diffusion-v2"
        )
    }
    pub fn pooled_control(&self) -> bool {
        matches!(
            self.graph.as_str(),
            "pooled-control-code-diffusion-v2" | "pooled-control-bounded-code-diffusion-v2"
        )
    }
    pub fn readout_slots(&self) -> usize {
        if self.slot_readout() {
            self.summary_slots / 2
        } else {
            1
        }
    }
    pub fn widths(&self) -> [usize; 4] {
        if self.layout == LAYOUT_V2 {
            [1024, 4096, 8192, 10240]
        } else {
            WIDTHS
        }
    }
    pub fn blocks(&self) -> [usize; 4] {
        if self.layout == LAYOUT_V2 {
            [1024, 3072, 4096, 2048]
        } else {
            BLOCKS
        }
    }
    pub fn validate(&self) -> Result<()> {
        check(
            (self.layout == LAYOUT && self.graph == "mixed-code-diffusion-v1")
                || (self.layout == LAYOUT_V2
                    && matches!(
                        self.graph.as_str(),
                        "slot-code-diffusion-v2"
                            | "pooled-control-code-diffusion-v2"
                            | "pooled-code-diffusion-v2"
                            | "slot-bounded-code-diffusion-v2"
                            | "pooled-control-bounded-code-diffusion-v2"
                            | "pooled-bounded-code-diffusion-v2"
                    )),
            "unknown learned graph",
        )?;
        check(
            (1..=512).contains(&self.width)
                && (1..=16).contains(&self.heads)
                && self.width % self.heads == 0
                && (1..=8).contains(&self.feature_blocks)
                && (1..=4).contains(&self.inducing_blocks)
                && (1..=128).contains(&self.inducing_slots)
                && (1..=64).contains(&self.summary_slots)
                && (1..=4).contains(&self.target_blocks)
                && (1..=12).contains(&self.decoder_blocks)
                && (1..=16).contains(&self.condition_slots)
                && (1..=256).contains(&self.max_features)
                && (2..=256).contains(&self.max_categories)
                && (1..=16384).contains(&self.max_context)
                && (1..=1024).contains(&self.max_query)
                && (1..=262144).contains(&self.cell_cap)
                && (1..=64).contains(&self.diffusion_steps)
                && (!self.slot_readout()
                    || ((2..=32).contains(&self.summary_slots) && self.summary_slots % 2 == 0)),
            "learned graph envelope exceeded",
        )
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WeightFile {
    pub file: String,
    pub bytes: usize,
    pub sha256: String,
    pub parameters: usize,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelManifest {
    pub format: String,
    pub graph: Graph,
    pub preprocessing: String,
    pub layout: String,
    pub weights: BTreeMap<String, WeightFile>,
    pub trained_levels: Vec<String>,
    pub dtype: String,
    pub source_sha256: String,
    pub training_manifest_sha256: String,
    pub evidence: String,
    pub certified: bool,
    pub output_schema: String,
    pub supported_backends: Vec<String>,
    pub quantization: String,
    pub coordinate_space_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Column {
    pub slot: u32,
    pub kind: String,
    pub center: f64,
    pub scale: f64,
    pub categories: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Schema {
    pub preprocessing: String,
    pub task: Option<String>,
    pub target_index: Option<usize>,
    pub output: String,
    pub columns: Vec<Column>,
}
impl Schema {
    pub fn validate(&self, graph: &Graph) -> Result<()> {
        check(
            self.preprocessing == PREPROCESSING
                && self.output == "positional_numeric_and_category_ids"
                && !self.columns.is_empty()
                && self.columns.len() <= graph.max_features + 1,
            "unsupported learned schema",
        )?;
        let mut seen = std::collections::BTreeSet::new();
        for (i, c) in self.columns.iter().enumerate() {
            check(
                seen.insert(c.slot) && c.center.is_finite() && c.scale.is_finite() && c.scale > 0.,
                "invalid learned feature binding or transform",
            )?;
            check(
                (c.kind == "numeric" && c.categories == 0)
                    || (c.kind == "categorical"
                        && (3..=graph.max_categories).contains(&c.categories)
                        && c.center == 0.
                        && c.scale == 1.),
                "unsupported learned feature kind",
            )?;
            check(
                (self.target_index == Some(i)) == (c.slot == 0),
                "learned target slot mismatch",
            )?;
        }
        match self.target_index {
            Some(index) => {
                check(
                    index + 1 == self.columns.len(),
                    "learned target must be last",
                )?;
                let kind = self.columns[index].kind.as_str();
                check(
                    matches!(
                        (self.task.as_deref(), kind),
                        (Some("regression"), "numeric") | (Some("classification"), "categorical")
                    ),
                    "learned target task mismatch",
                )
            }
            None => check(
                self.task.is_none() && self.columns.len() <= graph.max_features,
                "target-free schema mismatch",
            ),
        }
    }
}
