use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{DopeError, Result, io_error};
use crate::model::{
    AdditiveTerm, ColumnSchema, CopulaEdge, Dependence, FactorLoading, InteractionTerm, Kernel,
    KernelProgram, LinearTerm, Marginal, MarsFactor, MarsTerm, Noise, SchemaKind, Target, Task,
    Transform, TriangularTerm, VineEdge,
};

pub const MAGIC: &[u8; 4] = b"DPK3";
pub const V2_MAGIC: &[u8; 4] = b"DPK2";
pub const HEADER_BYTES: usize = 56;

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct SectionAccounting {
    pub tag: u8,
    pub framing_bytes: usize,
    pub raw_bytes: usize,
    pub encoded_bytes: usize,
    pub compressed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct ArtifactAccounting {
    pub header_bytes: usize,
    pub sections: Vec<SectionAccounting>,
    pub total_bytes: usize,
}

pub fn account_artifact(artifact: &[u8]) -> Result<ArtifactAccounting> {
    if artifact.len() < HEADER_BYTES || !artifact.starts_with(MAGIC) {
        return Err(DopeError::Codec(
            "accounting requires a DPK3 artifact".into(),
        ));
    }
    let payload_length = u64::from_le_bytes(artifact[32..40].try_into().unwrap()) as usize;
    if payload_length != artifact.len() - HEADER_BYTES {
        return Err(DopeError::Codec("payload length mismatch".into()));
    }
    let mut reader = Reader::new(&artifact[HEADER_BYTES..]);
    let mut sections = Vec::new();
    while reader.remaining() > 0 {
        let start = reader.offset;
        let encoded_tag = reader.byte()?;
        let encoded_bytes = reader.usize(reader.remaining())?;
        let compressed = encoded_tag & 0x80 != 0;
        let raw_bytes = if compressed {
            reader.usize(MAX_ARTIFACT_BYTES)?
        } else {
            encoded_bytes
        };
        let framing_bytes = reader.offset - start;
        reader.take(encoded_bytes)?;
        sections.push(SectionAccounting {
            tag: encoded_tag & 0x7f,
            framing_bytes,
            raw_bytes,
            encoded_bytes,
            compressed,
        });
    }
    let total_bytes = HEADER_BYTES
        + sections
            .iter()
            .map(|section| section.framing_bytes + section.encoded_bytes)
            .sum::<usize>();
    if total_bytes != artifact.len() {
        return Err(DopeError::Codec(
            "section totals do not match artifact length".into(),
        ));
    }
    Ok(ArtifactAccounting {
        header_bytes: HEADER_BYTES,
        sections,
        total_bytes,
    })
}
/// V3.1 raises the empirical champion ceiling while retaining all earlier
/// decoder limits on individual operators and collections.
pub const MAX_ARTIFACT_BYTES: usize = 2 * 1024 * 1024 * 1024;
pub const MAX_NEURAL_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;
const VERSION_MAJOR: u8 = 3;
const VERSION_MINOR: u8 = 3;
const V2_VERSION_MAJOR: u8 = 2;
const V2_VERSION_MINOR: u8 = 1;
const DECODER_ID: u8 = 1;
const FIXED_SCALE: f64 = 1_000_000.0;

#[derive(Clone, Debug)]
pub enum LoadedKernel {
    V3(Box<Kernel>),
    V2(Box<Kernel>),
    V1(Value),
}

#[derive(Default)]
struct Writer(Vec<u8>);

impl Writer {
    fn byte(&mut self, value: u8) {
        self.0.push(value);
    }

    fn u16(&mut self, value: u16) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn varint(&mut self, mut value: u64) {
        while value >= 0x80 {
            self.byte((value as u8 & 0x7f) | 0x80);
            value >>= 7;
        }
        self.byte(value as u8);
    }

    fn signed(&mut self, value: i64) {
        let zigzag = if value >= 0 {
            (value as u64) << 1
        } else {
            ((-value) as u64) * 2 - 1
        };
        self.varint(zigzag);
    }

    fn fixed(&mut self, value: f32) {
        self.signed((value as f64 * FIXED_SCALE).round() as i64);
    }

    fn unit(&mut self, value: f32) {
        self.u16((value.clamp(0.0, 1.0) * u16::MAX as f32).round() as u16);
    }
}

struct Reader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.offset)
    }

    fn byte(&mut self) -> Result<u8> {
        let value = *self
            .data
            .get(self.offset)
            .ok_or_else(|| DopeError::Codec("truncated bytecode".into()))?;
        self.offset += 1;
        Ok(value)
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| DopeError::Codec("section length overflow".into()))?;
        let value = self
            .data
            .get(self.offset..end)
            .ok_or_else(|| DopeError::Codec("truncated bytecode section".into()))?;
        self.offset = end;
        Ok(value)
    }

    fn u16(&mut self) -> Result<u16> {
        let bytes: [u8; 2] = self.take(2)?.try_into().unwrap();
        Ok(u16::from_le_bytes(bytes))
    }

    fn varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        for shift in (0..70).step_by(7) {
            let byte = self.byte()?;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(DopeError::Codec("varint exceeds 64 bits".into()))
    }

    fn usize(&mut self, bound: usize) -> Result<usize> {
        let value = usize::try_from(self.varint()?)
            .map_err(|_| DopeError::Codec("index exceeds host size".into()))?;
        if value > bound {
            return Err(DopeError::Codec(
                "encoded collection exceeds its bound".into(),
            ));
        }
        Ok(value)
    }

    fn signed(&mut self) -> Result<i64> {
        let value = self.varint()?;
        Ok(if value & 1 == 0 {
            (value >> 1) as i64
        } else {
            -((value >> 1) as i64) - 1
        })
    }

    fn fixed(&mut self) -> Result<f32> {
        Ok((self.signed()? as f64 / FIXED_SCALE) as f32)
    }

    fn unit(&mut self) -> Result<f32> {
        Ok(self.u16()? as f32 / u16::MAX as f32)
    }

    fn finished(&self) -> Result<()> {
        if self.remaining() == 0 {
            Ok(())
        } else {
            Err(DopeError::Codec(
                "trailing bytes in bytecode section".into(),
            ))
        }
    }
}

fn schema_kind_code(kind: SchemaKind) -> u8 {
    match kind {
        SchemaKind::Constant => 0,
        SchemaKind::Binary => 1,
        SchemaKind::Ordinal => 2,
        SchemaKind::CategoricalGrid => 3,
        SchemaKind::CountLike => 4,
        SchemaKind::Continuous => 5,
        SchemaKind::Inflated => 6,
    }
}

fn encode_schema(kernel: &Kernel) -> Vec<u8> {
    let mut writer = Writer::default();
    writer.varint(kernel.schema.len() as u64);
    for schema in &kernel.schema {
        writer.byte(schema_kind_code(schema.kind));
        writer.unit(schema.missing_probability);
        writer.unit(schema.impute);
        match schema.transform {
            Transform::Identity => writer.byte(0),
            Transform::Log1p { scale } => {
                writer.byte(1);
                writer.fixed(scale);
            }
            Transform::Power { exponent } => {
                writer.byte(2);
                writer.fixed(exponent);
            }
            Transform::Logit { epsilon } => {
                writer.byte(3);
                writer.unit(epsilon);
            }
            Transform::Winsorize { lower, upper } => {
                writer.byte(4);
                writer.unit(lower);
                writer.unit(upper);
            }
        }
    }
    writer.0
}

fn encode_marginal(writer: &mut Writer, marginal: &Marginal) {
    match marginal {
        Marginal::Constant { value } => {
            writer.byte(0);
            writer.unit(*value);
        }
        Marginal::Bernoulli { probability } => {
            writer.byte(1);
            writer.unit(*probability);
        }
        Marginal::Grid {
            values,
            probabilities,
        } => {
            writer.byte(2);
            writer.varint(values.len() as u64);
            for value in values {
                writer.unit(*value);
            }
            for probability in probabilities {
                writer.unit(*probability);
            }
        }
        Marginal::QuantileSpline { values } => {
            writer.byte(3);
            writer.varint(values.len() as u64);
            for value in values {
                writer.unit(*value);
            }
        }
        Marginal::Beta { alpha, beta } => {
            writer.byte(4);
            writer.fixed(*alpha);
            writer.fixed(*beta);
        }
        Marginal::Gaussian { mean, sigma } => {
            writer.byte(5);
            writer.unit(*mean);
            writer.fixed(*sigma);
        }
        Marginal::Histogram {
            edges,
            probabilities,
        } => {
            writer.byte(6);
            writer.varint(edges.len() as u64);
            for edge in edges {
                writer.unit(*edge);
            }
            for probability in probabilities {
                writer.unit(*probability);
            }
        }
        Marginal::ZeroInflated {
            point,
            probability,
            base,
        } => {
            writer.byte(7);
            writer.unit(*point);
            writer.unit(*probability);
            encode_marginal(writer, base);
        }
    }
}

fn encode_marginals(kernel: &Kernel) -> Vec<u8> {
    let mut writer = Writer::default();
    writer.varint(kernel.marginals.len() as u64);
    for marginal in &kernel.marginals {
        encode_marginal(&mut writer, marginal);
    }
    writer.0
}

fn encode_edges(writer: &mut Writer, edges: &[CopulaEdge]) {
    writer.varint(edges.len() as u64);
    for edge in edges {
        writer.varint(u64::from(edge.parent));
        writer.varint(u64::from(edge.child));
        writer.u16(
            ((edge.correlation.clamp(-0.995, 0.995) + 1.0) * 0.5 * u16::MAX as f32).round() as u16,
        );
    }
}

fn encode_dependence_value(writer: &mut Writer, dependence: &Dependence) {
    match dependence {
        Dependence::Independent => writer.byte(0),
        Dependence::ChowLiu { root, edges } => {
            writer.byte(1);
            writer.varint(u64::from(*root));
            encode_edges(writer, edges);
        }
        Dependence::GaussianCopula { cholesky } => {
            writer.byte(2);
            writer.varint(cholesky.len() as u64);
            for value in cholesky {
                writer.fixed(*value);
            }
        }
        Dependence::SparseGraph { edges } => {
            writer.byte(3);
            encode_edges(writer, edges);
        }
        Dependence::Poet {
            rank,
            loadings,
            residual_edges,
        } => {
            writer.byte(4);
            writer.byte(*rank);
            writer.varint(loadings.len() as u64);
            for loading in loadings {
                writer.varint(u64::from(loading.feature));
                writer.byte(loading.factor);
                writer.fixed(loading.loading);
            }
            encode_edges(writer, residual_edges);
        }
        Dependence::Vine { edges } => {
            writer.byte(5);
            writer.varint(edges.len() as u64);
            for edge in edges {
                writer.varint(u64::from(edge.left));
                writer.varint(u64::from(edge.right));
                writer.byte(edge.conditioning_depth);
                writer.fixed(edge.parameter);
            }
        }
        Dependence::Mixture {
            weights,
            components,
        } => {
            writer.byte(6);
            writer.varint(components.len() as u64);
            for weight in weights {
                writer.unit(*weight);
            }
            for component in components {
                encode_dependence_value(writer, component);
            }
        }
        Dependence::Triangular { terms } => {
            writer.byte(7);
            writer.varint(terms.len() as u64);
            for term in terms {
                writer.varint(u64::from(term.parent));
                writer.varint(u64::from(term.child));
                writer.fixed(term.coefficient);
            }
        }
    }
}

fn encode_dependence(dependence: &Dependence) -> Vec<u8> {
    let mut writer = Writer::default();
    encode_dependence_value(&mut writer, dependence);
    writer.0
}

fn encode_additive(writer: &mut Writer, term: &AdditiveTerm) {
    writer.varint(u64::from(term.feature));
    writer.varint(term.knots.len() as u64);
    for knot in &term.knots {
        writer.unit(*knot);
    }
    for value in &term.values {
        writer.fixed(*value);
    }
}

fn encode_target(target: &Target) -> Vec<u8> {
    let mut writer = Writer::default();
    match target {
        Target::SparseLinear { intercept, terms } | Target::SparseLogistic { intercept, terms } => {
            writer.byte(matches!(target, Target::SparseLogistic { .. }) as u8);
            writer.fixed(*intercept);
            writer.varint(terms.len() as u64);
            let mut ordered = terms.clone();
            ordered.sort_by_key(|term| term.feature);
            let mut prior = 0u32;
            for (index, term) in ordered.iter().enumerate() {
                let delta = if index == 0 {
                    term.feature
                } else {
                    term.feature - prior
                };
                writer.varint(u64::from(delta));
                writer.fixed(term.coefficient);
                prior = term.feature;
            }
        }
        Target::SparseGam {
            intercept,
            logistic,
            terms,
        } => {
            writer.byte(2);
            writer.fixed(*intercept);
            writer.byte(*logistic as u8);
            writer.varint(terms.len() as u64);
            for term in terms {
                encode_additive(&mut writer, term);
            }
        }
        Target::Ga2m {
            intercept,
            logistic,
            main_terms,
            interactions,
        } => {
            writer.byte(3);
            writer.fixed(*intercept);
            writer.byte(*logistic as u8);
            writer.varint(main_terms.len() as u64);
            for term in main_terms {
                encode_additive(&mut writer, term);
            }
            writer.varint(interactions.len() as u64);
            for term in interactions {
                writer.varint(u64::from(term.left));
                writer.varint(u64::from(term.right));
                writer.varint(term.left_knots.len() as u64);
                writer.varint(term.right_knots.len() as u64);
                for knot in &term.left_knots {
                    writer.unit(*knot);
                }
                for knot in &term.right_knots {
                    writer.unit(*knot);
                }
                for value in &term.values {
                    writer.fixed(*value);
                }
            }
        }
        Target::Mars {
            intercept,
            logistic,
            terms,
        } => {
            writer.byte(4);
            writer.fixed(*intercept);
            writer.byte(*logistic as u8);
            writer.varint(terms.len() as u64);
            for term in terms {
                writer.fixed(term.coefficient);
                writer.varint(term.factors.len() as u64);
                for factor in &term.factors {
                    writer.varint(u64::from(factor.feature));
                    writer.unit(factor.knot);
                    writer.byte((factor.direction > 0) as u8);
                }
            }
        }
        Target::ObliviousTree {
            logistic,
            features,
            thresholds,
            leaves,
        } => {
            writer.byte(5);
            writer.fixed(0.0);
            writer.byte(*logistic as u8);
            writer.varint(features.len() as u64);
            for (feature, threshold) in features.iter().zip(thresholds) {
                writer.varint(u64::from(*feature));
                writer.unit(*threshold);
            }
            for leaf in leaves {
                writer.fixed(*leaf);
            }
        }
        Target::CompactNeuralResidual {
            intercept,
            logistic,
            linear_terms,
            hidden_features,
            hidden_width,
            input_weights,
            hidden_biases,
            output_weights,
        } => {
            writer.byte(6);
            writer.fixed(*intercept);
            writer.byte(*logistic as u8);
            writer.varint(linear_terms.len() as u64);
            for term in linear_terms {
                writer.varint(u64::from(term.feature));
                writer.fixed(term.coefficient);
            }
            writer.varint(hidden_features.len() as u64);
            writer.byte(*hidden_width);
            for feature in hidden_features {
                writer.varint(u64::from(*feature));
            }
            for value in input_weights {
                writer.fixed(*value);
            }
            for value in hidden_biases {
                writer.fixed(*value);
            }
            for value in output_weights {
                writer.fixed(*value);
            }
        }
    }
    writer.0
}

fn encode_noise(noise: &Noise) -> Vec<u8> {
    let mut writer = Writer::default();
    match noise {
        Noise::Homoscedastic { sigma } => {
            writer.byte(0);
            writer.unit(*sigma);
        }
        Noise::BernoulliCalibration { base_rate } => {
            writer.byte(1);
            writer.unit(*base_rate);
        }
        Noise::Heteroscedastic {
            intercept,
            terms,
            minimum_sigma,
            maximum_sigma,
        } => {
            writer.byte(2);
            writer.fixed(*intercept);
            writer.unit(*minimum_sigma);
            writer.unit(*maximum_sigma);
            writer.varint(terms.len() as u64);
            for term in terms {
                writer.varint(u64::from(term.feature));
                writer.fixed(term.coefficient);
            }
        }
        Noise::IsotonicCalibration {
            knots,
            probabilities,
        } => {
            writer.byte(3);
            writer.varint(knots.len() as u64);
            for knot in knots {
                writer.unit(*knot);
            }
            for probability in probabilities {
                writer.unit(*probability);
            }
        }
    }
    writer.0
}

const RANS_SCALE_BITS: u32 = 12;
const RANS_TOTAL: u32 = 1 << RANS_SCALE_BITS;
const RANS_LOWER_BOUND: u64 = 1 << 23;

fn normalized_frequencies(raw: &[u8]) -> [u16; 256] {
    let mut counts = [0usize; 256];
    for &byte in raw {
        counts[byte as usize] += 1;
    }
    let mut frequencies = [0u16; 256];
    for symbol in 0..256 {
        if counts[symbol] > 0 {
            frequencies[symbol] =
                ((counts[symbol] * RANS_TOTAL as usize / raw.len()).max(1)) as u16;
        }
    }
    let mut total = frequencies
        .iter()
        .map(|value| u32::from(*value))
        .sum::<u32>();
    while total > RANS_TOTAL {
        let symbol = (0..256)
            .filter(|symbol| frequencies[*symbol] > 1)
            .max_by_key(|symbol| (frequencies[*symbol], counts[*symbol], 255 - *symbol))
            .expect("normalization has a decrementable frequency");
        frequencies[symbol] -= 1;
        total -= 1;
    }
    while total < RANS_TOTAL {
        let symbol = (0..256)
            .filter(|symbol| counts[*symbol] > 0)
            .max_by_key(|symbol| (counts[*symbol], 255 - *symbol))
            .expect("non-empty input has a symbol");
        frequencies[symbol] += 1;
        total += 1;
    }
    frequencies
}

fn rans_encode(raw: &[u8]) -> Vec<u8> {
    let frequencies = normalized_frequencies(raw);
    let mut cumulative = [0u32; 256];
    let mut sum = 0u32;
    for symbol in 0..256 {
        cumulative[symbol] = sum;
        sum += u32::from(frequencies[symbol]);
    }
    let mut state = RANS_LOWER_BOUND;
    let mut renormalized = Vec::new();
    for &byte in raw.iter().rev() {
        let frequency = u64::from(frequencies[byte as usize]);
        let threshold = ((RANS_LOWER_BOUND >> RANS_SCALE_BITS) << 8) * frequency;
        while state >= threshold {
            renormalized.push(state as u8);
            state >>= 8;
        }
        state = ((state / frequency) << RANS_SCALE_BITS)
            + state % frequency
            + u64::from(cumulative[byte as usize]);
    }
    let mut encoded = Vec::new();
    encoded.push(1); // canonical static byte-rANS
    let nonzero = frequencies.iter().filter(|value| **value > 0).count() as u16;
    encoded.extend_from_slice(&nonzero.to_le_bytes());
    for (symbol, frequency) in frequencies.iter().enumerate() {
        if *frequency > 0 {
            encoded.push(symbol as u8);
            encoded.extend_from_slice(&frequency.to_le_bytes());
        }
    }
    encoded.extend_from_slice(&(state as u32).to_le_bytes());
    renormalized.reverse();
    encoded.extend_from_slice(&renormalized);
    encoded
}

fn rans_decode(encoded: &[u8], raw_length: usize) -> Result<Vec<u8>> {
    let mut reader = Reader::new(encoded);
    if reader.byte()? != 1 {
        return Err(DopeError::Codec("unknown range-coder identifier".into()));
    }
    let count = reader.u16()? as usize;
    if count == 0 || count > 256 {
        return Err(DopeError::Codec("invalid rANS frequency table".into()));
    }
    let mut frequencies = [0u16; 256];
    for _ in 0..count {
        let symbol = reader.byte()? as usize;
        let frequency = reader.u16()?;
        if frequency == 0 || frequencies[symbol] != 0 {
            return Err(DopeError::Codec(
                "non-canonical rANS frequency table".into(),
            ));
        }
        frequencies[symbol] = frequency;
    }
    if frequencies
        .iter()
        .map(|value| u32::from(*value))
        .sum::<u32>()
        != RANS_TOTAL
    {
        return Err(DopeError::Codec(
            "rANS frequencies do not sum to scale".into(),
        ));
    }
    let mut cumulative = [0u32; 256];
    let mut lookup = [0u8; RANS_TOTAL as usize];
    let mut sum = 0u32;
    for symbol in 0..256 {
        cumulative[symbol] = sum;
        for slot in sum..sum + u32::from(frequencies[symbol]) {
            lookup[slot as usize] = symbol as u8;
        }
        sum += u32::from(frequencies[symbol]);
    }
    let state_bytes: [u8; 4] = reader.take(4)?.try_into().expect("four bytes requested");
    let mut state = u64::from(u32::from_le_bytes(state_bytes));
    let mut output = Vec::with_capacity(raw_length);
    for _ in 0..raw_length {
        let residue = (state & u64::from(RANS_TOTAL - 1)) as u32;
        let symbol = lookup[residue as usize];
        output.push(symbol);
        let frequency = u64::from(frequencies[symbol as usize]);
        state = frequency * (state >> RANS_SCALE_BITS)
            + u64::from(residue - cumulative[symbol as usize]);
        while state < RANS_LOWER_BOUND {
            state = (state << 8) | u64::from(reader.byte()?);
        }
    }
    if reader.remaining() != 0 {
        return Err(DopeError::Codec("trailing bytes in rANS stream".into()));
    }
    Ok(output)
}

include!("codec/section_encoding.rs");
include!("codec/decoding.rs");
