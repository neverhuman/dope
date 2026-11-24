//! Shared `DPA1` V8 wrapper-byte budgeting and reserved-vs-serving packet section comparison.
//!
//! Both serving (`fit_dpa1_v8`) and training (`dope-ffi`) derive the exact `DPA1` header +
//! `V7Schema` section byte cost through [`prepare_dpa_v8_budget`] so layout and section
//! budgets match across canonical and arbitrary named/categorical schemas.

use super::container::{parse_dpa1, Encoder, DPA1_HEADER_BYTES};
use super::v7_schema::{decode_section, encode_section, V7Column, V7Kind, V7Schema};
use crate::error::{DopeError, Result};
use crate::single_encoding::v8::packet::{TAG_CODE, TAG_WEIGHTS_FP};
use crate::single_encoding::v8::planner::{DSE8_CRC_BYTES, DSE8_HEADER_BYTES, TLV_HDR_BYTES};
use crate::single_encoding::v8::unpack_dse8_packet;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Exact `DPA1` V8 wrapper section and remaining inner `DSE8` packet byte budget.
#[derive(Clone, Debug)]
pub struct PreparedDpaV8Budget {
    pub schema_section: Vec<u8>,
    pub wrapper_bytes: usize,
    pub packet_budget: usize,
}

/// Compute the exact `DPA1` header + `V7Schema` section bytes and remaining `DSE8` packet room.
pub(crate) fn prepare_dpa_v8_budget(
    schema: &V7Schema,
    total_budget: usize,
    anonymize: bool,
) -> Result<PreparedDpaV8Budget> {
    if total_budget <= DPA1_HEADER_BYTES {
        return Err(DopeError::Budget(format!(
            "{total_budget} bytes cannot hold the {DPA1_HEADER_BYTES}-byte DPA1 header and a payload"
        )));
    }
    let schema_section = encode_section(schema, anonymize)?;
    let wrapper_bytes = DPA1_HEADER_BYTES + schema_section.len();
    let packet_budget = total_budget.saturating_sub(wrapper_bytes);
    Ok(PreparedDpaV8Budget {
        schema_section,
        wrapper_bytes,
        packet_budget,
    })
}

/// JSON-serializable feature column kind specification for exact `V7Schema` wrapper sizing.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DpaV8KindSpec {
    Real,
    Integer,
    Categorical(Vec<String>),
}

impl From<&DpaV8KindSpec> for V7Kind {
    fn from(spec: &DpaV8KindSpec) -> Self {
        match spec {
            DpaV8KindSpec::Real => V7Kind::Real,
            DpaV8KindSpec::Integer => V7Kind::Integer,
            DpaV8KindSpec::Categorical(labels) => V7Kind::Categorical(labels.clone()),
        }
    }
}

/// Schema specification carrying real feature names, source types, and target vocabulary.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DpaV8SchemaSpec {
    pub feature_names: Vec<String>,
    #[serde(default)]
    pub feature_kinds: Option<Vec<DpaV8KindSpec>>,
    #[serde(default)]
    pub target_name: Option<String>,
    #[serde(default)]
    pub target_labels: Option<Vec<String>>,
}

impl DpaV8SchemaSpec {
    pub(crate) fn to_v7_schema(&self) -> Result<V7Schema> {
        if self.feature_names.is_empty() {
            return Err(DopeError::Input("v8_schema_empty_features".into()));
        }
        if let Some(kinds) = &self.feature_kinds {
            if kinds.len() != self.feature_names.len() {
                return Err(DopeError::Input("v8_schema_kind_len_mismatch".into()));
            }
        }
        let features = self
            .feature_names
            .iter()
            .enumerate()
            .map(|(i, name)| V7Column {
                name: name.clone(),
                kind: self
                    .feature_kinds
                    .as_ref()
                    .map_or(V7Kind::Real, |ks| V7Kind::from(&ks[i])),
            })
            .collect();
        let target = V7Column {
            name: self
                .target_name
                .clone()
                .unwrap_or_else(|| "target".to_string()),
            kind: match &self.target_labels {
                Some(labels) if labels.len() >= 2 => V7Kind::Categorical(labels.clone()),
                Some(_) => return Err(DopeError::Input("v8_schema_single_target_label".into())),
                None => V7Kind::Real,
            },
        };
        Ok(V7Schema { features, target })
    }
}

/// Exact `DPA1` header + `V7Schema` section byte count for a real JSON schema specification.
pub fn dpa_v8_wrapper_bytes_from_json(schema_json: &str, anonymize: bool) -> Result<usize> {
    let spec: DpaV8SchemaSpec =
        serde_json::from_str(schema_json).map_err(|_| DopeError::Input("v8_schema_json".into()))?;
    let schema = spec.to_v7_schema()?;
    Ok(prepare_dpa_v8_budget(&schema, usize::MAX, anonymize)?.wrapper_bytes)
}

/// Build the canonical numeric `V7Schema` (`c0..c{p-1}, target`) from FIT arrays and task id.
pub(crate) fn canonical_fit_v7_schema(x_raw: &[Vec<f64>], y_raw: &[f64], task_id: u8) -> V7Schema {
    let features = x_raw
        .iter()
        .enumerate()
        .map(|(i, col)| {
            let mut has_fin = false;
            let mut all_int = true;
            for &v in col {
                if v.is_finite() {
                    has_fin = true;
                    if v.fract() != 0.0 {
                        all_int = false;
                        break;
                    }
                }
            }
            V7Column {
                name: format!("c{i}"),
                kind: if has_fin && all_int {
                    V7Kind::Integer
                } else {
                    V7Kind::Real
                },
            }
        })
        .collect();
    let target = V7Column {
        name: "target".to_string(),
        kind: if task_id == 0 {
            V7Kind::Real
        } else {
            let mut classes = BTreeSet::<i32>::new();
            for &v in y_raw {
                if v.is_finite() {
                    classes.insert(v.round() as i32);
                }
            }
            let labels: Vec<String> = if classes.len() >= 2 {
                classes.into_iter().map(|c| c.to_string()).collect()
            } else {
                vec!["0".to_string(), "1".to_string()]
            };
            V7Kind::Categorical(labels)
        },
    };
    V7Schema { features, target }
}

/// Exact `DPA1` header + `V7Schema` section byte overhead for unnamed numeric FIT arrays.
pub fn estimate_dpa_v8_wrapper_bytes(x_raw: &[Vec<f64>], y_raw: &[f64], task_id: u8) -> usize {
    let schema = canonical_fit_v7_schema(x_raw, y_raw, task_id);
    prepare_dpa_v8_budget(&schema, usize::MAX, false).map_or(DPA1_HEADER_BYTES, |p| p.wrapper_bytes)
}

/// Per-TLV-section byte comparison between a training reserved pack and a serving `DPA1` pack.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Dse8SectionDiff {
    pub tag: u8,
    pub reserved_len: usize,
    pub serving_len: usize,
    pub bytes_equal: bool,
}

/// Direct byte-and-section comparison between a training reserved `DSE8` pack and serving `DPA1`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Dse8PackComparison {
    pub wrapper_bytes: usize,
    pub reserved_inner_bytes: usize,
    pub serving_inner_bytes: usize,
    pub serving_total_bytes: usize,
    pub p_eff: u8,
    pub rank_use: u8,
    pub knot_level: u8,
    pub code_tier: u8,
    pub header_bytes_equal: bool,
    pub section_tags_equal: bool,
    pub section_lengths_equal: bool,
    pub non_neural_sections_byte_equal: bool,
    pub sections: Vec<Dse8SectionDiff>,
}

fn parse_raw_tlvs(packet: &[u8]) -> Result<Vec<(u8, &[u8])>> {
    if packet.len() < DSE8_HEADER_BYTES + DSE8_CRC_BYTES {
        return Err(DopeError::Artifact("dse8_truncated".into()));
    }
    let body_end = packet.len() - DSE8_CRC_BYTES;
    let mut at = DSE8_HEADER_BYTES;
    let mut out = Vec::new();
    while at < body_end {
        if at + TLV_HDR_BYTES > body_end {
            return Err(DopeError::Artifact("dse8_tlv_truncated".into()));
        }
        let tag = packet[at];
        let len = u16::from_le_bytes([packet[at + 1], packet[at + 2]]) as usize;
        at += TLV_HDR_BYTES;
        if at + len > body_end {
            return Err(DopeError::Artifact("dse8_tlv_overflow".into()));
        }
        out.push((tag, &packet[at..at + len]));
        at += len;
    }
    Ok(out)
}

/// Compare a training reserved `DSE8` inner packet directly against a serving `DPA1` V8 artifact.
pub fn compare_reserved_and_serving_dpa1(
    reserved_inner: &[u8],
    serving_dpa1: &[u8],
) -> Result<Dse8PackComparison> {
    let dpa = parse_dpa1(serving_dpa1)?;
    if dpa.encoder != Encoder::V8 {
        return Err(DopeError::Artifact("dpa1_not_v8".into()));
    }
    let (_, schema_used) = decode_section(dpa.payload)?;
    let wrapper_bytes = DPA1_HEADER_BYTES + schema_used;
    let serving_inner = &dpa.payload[schema_used..];
    let res_pkt = unpack_dse8_packet(reserved_inner)?;
    let srv_pkt = unpack_dse8_packet(serving_inner)?;
    let res_tlvs = parse_raw_tlvs(reserved_inner)?;
    let srv_tlvs = parse_raw_tlvs(serving_inner)?;

    let header_bytes_equal =
        reserved_inner[..DSE8_HEADER_BYTES] == serving_inner[..DSE8_HEADER_BYTES];
    let section_tags_equal = res_tlvs.len() == srv_tlvs.len()
        && res_tlvs
            .iter()
            .zip(&srv_tlvs)
            .all(|((ta, _), (tb, _))| ta == tb);
    let mut sections = Vec::with_capacity(res_tlvs.len().max(srv_tlvs.len()));
    let mut section_lengths_equal = section_tags_equal;
    let mut non_neural_sections_byte_equal = section_tags_equal;
    for i in 0..res_tlvs.len().max(srv_tlvs.len()) {
        let a = res_tlvs.get(i);
        let b = srv_tlvs.get(i);
        let (tag, r_len, s_len, eq) = match (a, b) {
            (Some(&(ta, pa)), Some(&(tb, pb))) if ta == tb => (ta, pa.len(), pb.len(), pa == pb),
            (Some(&(ta, pa)), Some(&(_, pb))) => (ta, pa.len(), pb.len(), false),
            (Some(&(ta, pa)), None) => (ta, pa.len(), 0, false),
            (None, Some(&(tb, pb))) => (tb, 0, pb.len(), false),
            (None, None) => unreachable!(),
        };
        if r_len != s_len {
            section_lengths_equal = false;
        }
        if tag != TAG_CODE && tag != TAG_WEIGHTS_FP && !eq {
            non_neural_sections_byte_equal = false;
        }
        sections.push(Dse8SectionDiff {
            tag,
            reserved_len: r_len,
            serving_len: s_len,
            bytes_equal: eq,
        });
    }
    let _ = res_pkt;
    Ok(Dse8PackComparison {
        wrapper_bytes,
        reserved_inner_bytes: reserved_inner.len(),
        serving_inner_bytes: serving_inner.len(),
        serving_total_bytes: serving_dpa1.len(),
        p_eff: srv_pkt.header.p_eff,
        rank_use: srv_pkt.header.rank_use,
        knot_level: srv_pkt.header.knot_level,
        code_tier: srv_pkt.header.code_tier,
        header_bytes_equal,
        section_tags_equal,
        section_lengths_equal,
        non_neural_sections_byte_equal,
        sections,
    })
}
