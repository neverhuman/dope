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
const VERSION_MINOR: u8 = 2;
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

fn append_section(payload: &mut Writer, tag: u8, raw: &[u8]) -> Result<()> {
    let compressed = if raw.len() >= 64 {
        let candidate = rans_encode(raw);
        (candidate.len() + 4 < raw.len()).then_some(candidate)
    } else {
        None
    };
    if let Some(compressed) = compressed {
        payload.byte(tag | 0x80);
        payload.varint(compressed.len() as u64);
        payload.varint(raw.len() as u64);
        payload.0.extend_from_slice(&compressed);
    } else {
        payload.byte(tag);
        payload.varint(raw.len() as u64);
        payload.0.extend_from_slice(raw);
    }
    Ok(())
}

pub fn encode_kernel(kernel: &Kernel) -> Result<Vec<u8>> {
    kernel.validate().map_err(DopeError::Codec)?;
    let mut payload = Writer::default();
    append_section(&mut payload, 1, &encode_schema(kernel))?;
    append_section(&mut payload, 2, &encode_marginals(kernel))?;
    let limit = match &kernel.program {
        KernelProgram::Symbolic {
            dependence,
            target,
            noise,
        } => {
            append_section(&mut payload, 3, &encode_dependence(dependence))?;
            append_section(&mut payload, 4, &encode_target(target))?;
            append_section(&mut payload, 5, &encode_noise(noise))?;
            MAX_ARTIFACT_BYTES
        }
        KernelProgram::NeuralJoint(generator) => {
            let neural = serde_json::to_vec(generator)?;
            append_section(&mut payload, 6, &neural)?;
            MAX_NEURAL_ARTIFACT_BYTES
        }
    };
    if HEADER_BYTES + payload.0.len() > limit {
        return Err(DopeError::Codec(if limit == MAX_NEURAL_ARTIFACT_BYTES {
            "neural pilot artifact exceeds the 16 MiB cap".into()
        } else {
            "artifact exceeds the 2 GiB cap".into()
        }));
    }
    let checksum = blake3::hash(&payload.0);
    let mut artifact = Vec::with_capacity(HEADER_BYTES + payload.0.len());
    artifact.extend_from_slice(MAGIC);
    artifact.extend_from_slice(&[
        VERSION_MAJOR,
        VERSION_MINOR,
        DECODER_ID,
        kernel.task.code(),
        kernel.quantization_bits,
        kernel.seed_policy,
    ]);
    artifact.extend_from_slice(&(kernel.compliant as u16).to_le_bytes());
    artifact.extend_from_slice(&kernel.features.to_le_bytes());
    artifact.extend_from_slice(&kernel.rows_fitted.to_le_bytes());
    artifact.extend_from_slice(&kernel.seed.to_le_bytes());
    artifact.extend_from_slice(&(payload.0.len() as u64).to_le_bytes());
    artifact.extend_from_slice(&checksum.as_bytes()[..16]);
    debug_assert_eq!(artifact.len(), HEADER_BYTES);
    artifact.extend_from_slice(&payload.0);
    Ok(artifact)
}

fn decode_section(
    tag: u8,
    encoded: &[u8],
    raw_length: Option<usize>,
    legacy_zstd: bool,
) -> Result<Vec<u8>> {
    if tag & 0x80 == 0 {
        return Ok(encoded.to_vec());
    }
    let raw_length = raw_length
        .ok_or_else(|| DopeError::Codec("compressed section has no decoded length".into()))?;
    if legacy_zstd {
        zstd::bulk::decompress(encoded, raw_length)
            .map_err(|error| DopeError::Codec(format!("zstd decode failure: {error}")))
    } else {
        rans_decode(encoded, raw_length)
    }
}

fn decode_schema(data: &[u8], features: usize, has_transforms: bool) -> Result<Vec<ColumnSchema>> {
    let mut reader = Reader::new(data);
    let count = reader.usize(features)?;
    if count != features {
        return Err(DopeError::Codec("schema width differs from header".into()));
    }
    let mut schema = Vec::with_capacity(count);
    for _ in 0..count {
        let kind = match reader.byte()? {
            0 => SchemaKind::Constant,
            1 => SchemaKind::Binary,
            2 => SchemaKind::Ordinal,
            3 => SchemaKind::CategoricalGrid,
            4 => SchemaKind::CountLike,
            5 => SchemaKind::Continuous,
            6 => SchemaKind::Inflated,
            tag => return Err(DopeError::Codec(format!("unknown schema opcode {tag}"))),
        };
        let missing_probability = reader.unit()?;
        let impute = reader.unit()?;
        let transform = if !has_transforms {
            Transform::Identity
        } else {
            match reader.byte()? {
                0 => Transform::Identity,
                1 => Transform::Log1p {
                    scale: reader.fixed()?,
                },
                2 => Transform::Power {
                    exponent: reader.fixed()?,
                },
                3 => Transform::Logit {
                    epsilon: reader.unit()?,
                },
                4 => Transform::Winsorize {
                    lower: reader.unit()?,
                    upper: reader.unit()?,
                },
                tag => return Err(DopeError::Codec(format!("unknown transform opcode {tag}"))),
            }
        };
        schema.push(ColumnSchema {
            kind,
            missing_probability,
            impute,
            transform,
        });
    }
    reader.finished()?;
    Ok(schema)
}

fn decode_marginal(reader: &mut Reader<'_>, depth: usize) -> Result<Marginal> {
    if depth > 2 {
        return Err(DopeError::Codec(
            "marginal nesting exceeds its bound".into(),
        ));
    }
    Ok(match reader.byte()? {
        0 => Marginal::Constant {
            value: reader.unit()?,
        },
        1 => Marginal::Bernoulli {
            probability: reader.unit()?,
        },
        2 => {
            let length = reader.usize(65_536)?;
            if length == 0 {
                return Err(DopeError::Codec("empty categorical grid".into()));
            }
            let values = (0..length)
                .map(|_| reader.unit())
                .collect::<Result<Vec<_>>>()?;
            let probabilities = (0..length)
                .map(|_| reader.unit())
                .collect::<Result<Vec<_>>>()?;
            Marginal::Grid {
                values,
                probabilities,
            }
        }
        3 => {
            let length = reader.usize(65)?;
            if !matches!(length, 5 | 9 | 17 | 33 | 65) {
                return Err(DopeError::Codec(
                    "quantile spline violates the knot ladder".into(),
                ));
            }
            Marginal::QuantileSpline {
                values: (0..length)
                    .map(|_| reader.unit())
                    .collect::<Result<Vec<_>>>()?,
            }
        }
        4 => Marginal::Beta {
            alpha: reader.fixed()?,
            beta: reader.fixed()?,
        },
        5 => Marginal::Gaussian {
            mean: reader.unit()?,
            sigma: reader.fixed()?,
        },
        6 => {
            let length = reader.usize(257)?;
            if length < 2 {
                return Err(DopeError::Codec("histogram requires two edges".into()));
            }
            Marginal::Histogram {
                edges: (0..length)
                    .map(|_| reader.unit())
                    .collect::<Result<Vec<_>>>()?,
                probabilities: (0..length - 1)
                    .map(|_| reader.unit())
                    .collect::<Result<Vec<_>>>()?,
            }
        }
        7 => Marginal::ZeroInflated {
            point: reader.unit()?,
            probability: reader.unit()?,
            base: Box::new(decode_marginal(reader, depth + 1)?),
        },
        tag => return Err(DopeError::Codec(format!("unknown marginal opcode {tag}"))),
    })
}

fn decode_marginals(data: &[u8], features: usize) -> Result<Vec<Marginal>> {
    let mut reader = Reader::new(data);
    let count = reader.usize(features)?;
    if count != features {
        return Err(DopeError::Codec(
            "marginal width differs from header".into(),
        ));
    }
    let mut marginals = Vec::with_capacity(count);
    for _ in 0..count {
        marginals.push(decode_marginal(&mut reader, 0)?);
    }
    reader.finished()?;
    Ok(marginals)
}

fn decode_edges(reader: &mut Reader<'_>, features: usize, bound: usize) -> Result<Vec<CopulaEdge>> {
    let count = reader.usize(bound)?;
    (0..count)
        .map(|_| {
            Ok(CopulaEdge {
                parent: reader.usize(features.saturating_sub(1))? as u32,
                child: reader.usize(features.saturating_sub(1))? as u32,
                correlation: reader.u16()? as f32 / u16::MAX as f32 * 2.0 - 1.0,
            })
        })
        .collect()
}

fn decode_dependence_value(
    reader: &mut Reader<'_>,
    features: usize,
    depth: usize,
) -> Result<Dependence> {
    if depth > 2 {
        return Err(DopeError::Codec(
            "dependence mixture nesting exceeds its bound".into(),
        ));
    }
    Ok(match reader.byte()? {
        0 => Dependence::Independent,
        1 => {
            let root = reader.usize(features.saturating_sub(1))? as u32;
            let edges = decode_edges(reader, features, features.saturating_sub(1))?;
            Dependence::ChowLiu { root, edges }
        }
        2 => {
            let length = reader.usize(features.saturating_mul(features))?;
            Dependence::GaussianCopula {
                cholesky: (0..length)
                    .map(|_| reader.fixed())
                    .collect::<Result<Vec<_>>>()?,
            }
        }
        3 => Dependence::SparseGraph {
            edges: decode_edges(reader, features, features.saturating_mul(8))?,
        },
        4 => {
            let rank = reader.byte()?;
            let count = reader.usize(features.saturating_mul(16))?;
            let loadings = (0..count)
                .map(|_| {
                    Ok(FactorLoading {
                        feature: reader.usize(features.saturating_sub(1))? as u32,
                        factor: reader.byte()?,
                        loading: reader.fixed()?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Dependence::Poet {
                rank,
                loadings,
                residual_edges: decode_edges(reader, features, features.saturating_mul(4))?,
            }
        }
        5 => {
            let count = reader.usize(features.saturating_mul(8))?;
            Dependence::Vine {
                edges: (0..count)
                    .map(|_| {
                        Ok(VineEdge {
                            left: reader.usize(features.saturating_sub(1))? as u32,
                            right: reader.usize(features.saturating_sub(1))? as u32,
                            conditioning_depth: reader.byte()?,
                            parameter: reader.fixed()?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            }
        }
        6 => {
            let count = reader.usize(8)?;
            if count == 0 {
                return Err(DopeError::Codec("empty dependence mixture".into()));
            }
            let weights = (0..count)
                .map(|_| reader.unit())
                .collect::<Result<Vec<_>>>()?;
            let components = (0..count)
                .map(|_| decode_dependence_value(reader, features, depth + 1))
                .collect::<Result<Vec<_>>>()?;
            Dependence::Mixture {
                weights,
                components,
            }
        }
        7 => {
            let count = reader.usize(features.saturating_mul(8))?;
            Dependence::Triangular {
                terms: (0..count)
                    .map(|_| {
                        Ok(TriangularTerm {
                            parent: reader.usize(features.saturating_sub(1))? as u32,
                            child: reader.usize(features.saturating_sub(1))? as u32,
                            coefficient: reader.fixed()?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            }
        }
        tag => return Err(DopeError::Codec(format!("unknown dependence opcode {tag}"))),
    })
}

fn decode_dependence(data: &[u8], features: usize) -> Result<Dependence> {
    let mut reader = Reader::new(data);
    let dependence = decode_dependence_value(&mut reader, features, 0)?;
    reader.finished()?;
    Ok(dependence)
}

fn decode_additive(reader: &mut Reader<'_>, features: usize) -> Result<AdditiveTerm> {
    let feature = reader.usize(features.saturating_sub(1))? as u32;
    let length = reader.usize(17)?;
    Ok(AdditiveTerm {
        feature,
        knots: (0..length)
            .map(|_| reader.unit())
            .collect::<Result<Vec<_>>>()?,
        values: (0..length)
            .map(|_| reader.fixed())
            .collect::<Result<Vec<_>>>()?,
    })
}

fn decode_target(data: &[u8], features: usize) -> Result<Target> {
    let mut reader = Reader::new(data);
    let tag = reader.byte()?;
    let intercept = reader.fixed()?;
    let target = match tag {
        0 | 1 => {
            let count = reader.usize(features.min(64))?;
            let mut terms = Vec::with_capacity(count);
            let mut prior = 0u32;
            for index in 0..count {
                let delta = reader.usize(features.saturating_sub(1))? as u32;
                let feature = if index == 0 {
                    delta
                } else {
                    prior
                        .checked_add(delta)
                        .ok_or_else(|| DopeError::Codec("target feature gap overflow".into()))?
                };
                terms.push(LinearTerm {
                    feature,
                    coefficient: reader.fixed()?,
                });
                prior = feature;
            }
            if tag == 0 {
                Target::SparseLinear { intercept, terms }
            } else {
                Target::SparseLogistic { intercept, terms }
            }
        }
        2 => {
            let logistic = reader.byte()? != 0;
            let count = reader.usize(64)?;
            Target::SparseGam {
                intercept,
                logistic,
                terms: (0..count)
                    .map(|_| decode_additive(&mut reader, features))
                    .collect::<Result<Vec<_>>>()?,
            }
        }
        3 => {
            let logistic = reader.byte()? != 0;
            let main_count = reader.usize(64)?;
            let main_terms = (0..main_count)
                .map(|_| decode_additive(&mut reader, features))
                .collect::<Result<Vec<_>>>()?;
            let interaction_count = reader.usize(64)?;
            let mut interactions = Vec::with_capacity(interaction_count);
            for _ in 0..interaction_count {
                let left = reader.usize(features.saturating_sub(1))? as u32;
                let right = reader.usize(features.saturating_sub(1))? as u32;
                let left_length = reader.usize(9)?;
                let right_length = reader.usize(9)?;
                interactions.push(InteractionTerm {
                    left,
                    right,
                    left_knots: (0..left_length)
                        .map(|_| reader.unit())
                        .collect::<Result<Vec<_>>>()?,
                    right_knots: (0..right_length)
                        .map(|_| reader.unit())
                        .collect::<Result<Vec<_>>>()?,
                    values: (0..left_length * right_length)
                        .map(|_| reader.fixed())
                        .collect::<Result<Vec<_>>>()?,
                });
            }
            Target::Ga2m {
                intercept,
                logistic,
                main_terms,
                interactions,
            }
        }
        4 => {
            let logistic = reader.byte()? != 0;
            let count = reader.usize(128)?;
            let mut terms = Vec::with_capacity(count);
            for _ in 0..count {
                let coefficient = reader.fixed()?;
                let factor_count = reader.usize(3)?;
                let factors = (0..factor_count)
                    .map(|_| {
                        Ok(MarsFactor {
                            feature: reader.usize(features.saturating_sub(1))? as u32,
                            knot: reader.unit()?,
                            direction: if reader.byte()? == 0 { -1 } else { 1 },
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                terms.push(MarsTerm {
                    coefficient,
                    factors,
                });
            }
            Target::Mars {
                intercept,
                logistic,
                terms,
            }
        }
        5 => {
            let logistic = reader.byte()? != 0;
            let depth = reader.usize(8)?;
            let mut features_out = Vec::with_capacity(depth);
            let mut thresholds = Vec::with_capacity(depth);
            for _ in 0..depth {
                features_out.push(reader.usize(features.saturating_sub(1))? as u32);
                thresholds.push(reader.unit()?);
            }
            Target::ObliviousTree {
                logistic,
                features: features_out,
                thresholds,
                leaves: (0..1usize << depth)
                    .map(|_| reader.fixed())
                    .collect::<Result<Vec<_>>>()?,
            }
        }
        6 => {
            let logistic = reader.byte()? != 0;
            let linear_count = reader.usize(features.min(64))?;
            let linear_terms = (0..linear_count)
                .map(|_| {
                    Ok(LinearTerm {
                        feature: reader.usize(features.saturating_sub(1))? as u32,
                        coefficient: reader.fixed()?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let input_count = reader.usize(features.min(24))?;
            let hidden_width = reader.byte()?;
            if !(1..=16).contains(&hidden_width) {
                return Err(DopeError::Codec(
                    "compact neural hidden width is outside 1..=16".into(),
                ));
            }
            let hidden_features = (0..input_count)
                .map(|_| {
                    reader
                        .usize(features.saturating_sub(1))
                        .map(|value| value as u32)
                })
                .collect::<Result<Vec<_>>>()?;
            let parameter_count = input_count * usize::from(hidden_width);
            Target::CompactNeuralResidual {
                intercept,
                logistic,
                linear_terms,
                hidden_features,
                hidden_width,
                input_weights: (0..parameter_count)
                    .map(|_| reader.fixed())
                    .collect::<Result<Vec<_>>>()?,
                hidden_biases: (0..hidden_width)
                    .map(|_| reader.fixed())
                    .collect::<Result<Vec<_>>>()?,
                output_weights: (0..hidden_width)
                    .map(|_| reader.fixed())
                    .collect::<Result<Vec<_>>>()?,
            }
        }
        _ => return Err(DopeError::Codec(format!("unknown target opcode {tag}"))),
    };
    reader.finished()?;
    Ok(target)
}

fn decode_noise(data: &[u8]) -> Result<Noise> {
    let mut reader = Reader::new(data);
    let noise = match reader.byte()? {
        0 => Noise::Homoscedastic {
            sigma: reader.unit()?,
        },
        1 => Noise::BernoulliCalibration {
            base_rate: reader.unit()?,
        },
        2 => {
            let intercept = reader.fixed()?;
            let minimum_sigma = reader.unit()?;
            let maximum_sigma = reader.unit()?;
            let count = reader.usize(64)?;
            Noise::Heteroscedastic {
                intercept,
                minimum_sigma,
                maximum_sigma,
                terms: (0..count)
                    .map(|_| {
                        Ok(LinearTerm {
                            feature: reader.varint()? as u32,
                            coefficient: reader.fixed()?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            }
        }
        3 => {
            let count = reader.usize(33)?;
            Noise::IsotonicCalibration {
                knots: (0..count)
                    .map(|_| reader.unit())
                    .collect::<Result<Vec<_>>>()?,
                probabilities: (0..count)
                    .map(|_| reader.unit())
                    .collect::<Result<Vec<_>>>()?,
            }
        }
        tag => return Err(DopeError::Codec(format!("unknown noise opcode {tag}"))),
    };
    reader.finished()?;
    Ok(noise)
}

pub fn decode_kernel(artifact: &[u8]) -> Result<Kernel> {
    if artifact.len() < HEADER_BYTES || artifact.len() > MAX_ARTIFACT_BYTES {
        return Err(DopeError::Codec("artifact violates the size bounds".into()));
    }
    let v3 = &artifact[..4] == MAGIC;
    let v2 = &artifact[..4] == V2_MAGIC;
    if !v3 && !v2 {
        return Err(DopeError::Codec("not a V2 or V3 .dpk artifact".into()));
    }
    let supported_version = (v3 && artifact[4] == VERSION_MAJOR && artifact[5] <= VERSION_MINOR)
        || (v2 && artifact[4] == V2_VERSION_MAJOR && artifact[5] <= V2_VERSION_MINOR);
    if !supported_version || artifact[6] != DECODER_ID {
        return Err(DopeError::Codec(
            "unsupported bytecode or decoder version".into(),
        ));
    }
    let task = match artifact[7] {
        0 => Task::Regression,
        1 => Task::Binary,
        _ => return Err(DopeError::Codec("unknown task code".into())),
    };
    let quantization_bits = artifact[8];
    let seed_policy = artifact[9];
    let flags = u16::from_le_bytes(artifact[10..12].try_into().unwrap());
    if flags & !1 != 0 {
        return Err(DopeError::Codec("unknown header flags".into()));
    }
    let features = u32::from_le_bytes(artifact[12..16].try_into().unwrap());
    let rows_fitted = u64::from_le_bytes(artifact[16..24].try_into().unwrap());
    let seed = u64::from_le_bytes(artifact[24..32].try_into().unwrap());
    let payload_length = u64::from_le_bytes(artifact[32..40].try_into().unwrap()) as usize;
    if payload_length != artifact.len() - HEADER_BYTES {
        return Err(DopeError::Codec("payload length mismatch".into()));
    }
    let payload = &artifact[HEADER_BYTES..];
    if blake3::hash(payload).as_bytes()[..16] != artifact[40..56] {
        return Err(DopeError::Codec("payload checksum mismatch".into()));
    }

    let mut payload_reader = Reader::new(payload);
    let mut sections: [Option<Vec<u8>>; 6] = Default::default();
    let mut prior = 0u8;
    while payload_reader.remaining() > 0 {
        let encoded_tag = payload_reader.byte()?;
        let tag = encoded_tag & 0x7f;
        let maximum_tag = if v3 && artifact[5] >= 2 { 6 } else { 5 };
        if !(1..=maximum_tag).contains(&tag) || tag <= prior {
            return Err(DopeError::Codec(
                "sections are unknown, duplicated, or non-canonical".into(),
            ));
        }
        if tag == 6 && artifact.len() > MAX_NEURAL_ARTIFACT_BYTES {
            return Err(DopeError::Codec(
                "neural pilot artifact exceeds the 16 MiB cap".into(),
            ));
        }
        let encoded_length = payload_reader.usize(payload_reader.remaining())?;
        let raw_length = if encoded_tag & 0x80 != 0 {
            Some(payload_reader.usize(if tag == 6 {
                MAX_NEURAL_ARTIFACT_BYTES
            } else {
                MAX_ARTIFACT_BYTES
            })?)
        } else {
            None
        };
        let encoded = payload_reader.take(encoded_length)?;
        sections[tag as usize - 1] = Some(decode_section(
            encoded_tag,
            encoded,
            raw_length,
            v2 && artifact[5] == 0,
        )?);
        prior = tag;
    }
    let [schema, marginals, dependence, target, noise, neural] = sections;
    let schema = schema.ok_or_else(|| DopeError::Codec("missing schema section".into()))?;
    let marginals = marginals.ok_or_else(|| DopeError::Codec("missing marginal section".into()))?;
    let symbolic_count = [&dependence, &target, &noise]
        .into_iter()
        .filter(|section| section.is_some())
        .count();
    let program = match (symbolic_count, neural) {
        (3, None) => KernelProgram::Symbolic {
            dependence: decode_dependence(&dependence.unwrap(), features as usize)?,
            target: decode_target(&target.unwrap(), features as usize)?,
            noise: decode_noise(&noise.unwrap())?,
        },
        (0, Some(encoded)) if v3 && artifact[5] >= 2 => {
            let generator: crate::model::JointGenerator = serde_json::from_slice(&encoded)
                .map_err(|error| DopeError::Codec(format!("invalid neural section: {error}")))?;
            if serde_json::to_vec(&generator)? != encoded {
                return Err(DopeError::Codec(
                    "neural section is non-canonical or has unknown fields".into(),
                ));
            }
            KernelProgram::NeuralJoint(generator)
        }
        (0, None) => return Err(DopeError::Codec("missing kernel program section".into())),
        _ => {
            return Err(DopeError::Codec(
                "ambiguous or incomplete symbolic/neural payload".into(),
            ));
        }
    };
    let width = features as usize;
    let kernel = Kernel {
        task,
        rows_fitted,
        features,
        seed,
        seed_policy,
        quantization_bits,
        compliant: flags & 1 != 0,
        schema: decode_schema(&schema, width, v3 || artifact[5] >= 1)?,
        marginals: decode_marginals(&marginals, width)?,
        program,
    };
    if matches!(kernel.program, KernelProgram::NeuralJoint(_))
        && artifact.len() > MAX_NEURAL_ARTIFACT_BYTES
    {
        return Err(DopeError::Codec(
            "neural pilot artifact exceeds the 16 MiB cap".into(),
        ));
    }
    kernel.validate().map_err(DopeError::Codec)?;
    Ok(kernel)
}

pub fn load_kernel(path: &Path) -> Result<LoadedKernel> {
    let bytes = fs::read(path).map_err(|error| io_error(path, error))?;
    if bytes.starts_with(MAGIC) {
        Ok(LoadedKernel::V3(Box::new(decode_kernel(&bytes)?)))
    } else if bytes.starts_with(V2_MAGIC) {
        Ok(LoadedKernel::V2(Box::new(decode_kernel(&bytes)?)))
    } else {
        let value: Value = serde_json::from_slice(&bytes)?;
        if value.get("format").and_then(Value::as_str) != Some("dope-kernel")
            || value.get("version").and_then(Value::as_u64) != Some(1)
        {
            return Err(DopeError::Codec(
                "not a supported V1 JSON, V2 .dpk, or V3 .dpk artifact".into(),
            ));
        }
        Ok(LoadedKernel::V1(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kernel() -> Kernel {
        Kernel {
            task: Task::Regression,
            rows_fitted: 100,
            features: 2,
            seed: 9,
            seed_policy: 0,
            quantization_bits: 8,
            compliant: false,
            schema: vec![
                ColumnSchema {
                    kind: SchemaKind::Continuous,
                    missing_probability: 0.0,
                    impute: 0.5,
                    transform: Transform::Identity,
                },
                ColumnSchema {
                    kind: SchemaKind::Binary,
                    missing_probability: 0.1,
                    impute: 0.0,
                    transform: Transform::Identity,
                },
            ],
            marginals: vec![
                Marginal::QuantileSpline {
                    values: vec![0.0, 0.2, 0.5, 0.8, 1.0],
                },
                Marginal::Bernoulli { probability: 0.25 },
            ],
            program: KernelProgram::Symbolic {
                dependence: Dependence::ChowLiu {
                    root: 0,
                    edges: vec![CopulaEdge {
                        parent: 0,
                        child: 1,
                        correlation: 0.4,
                    }],
                },
                target: Target::SparseLinear {
                    intercept: 0.1,
                    terms: vec![LinearTerm {
                        feature: 0,
                        coefficient: 0.7,
                    }],
                },
                noise: Noise::Homoscedastic { sigma: 0.05 },
            },
        }
    }

    #[test]
    fn canonical_round_trip_and_corruption_detection() {
        let encoded = encode_kernel(&kernel()).unwrap();
        let decoded = decode_kernel(&encoded).unwrap();
        assert_eq!(encode_kernel(&decoded).unwrap(), encoded);
        let mut corrupt = encoded;
        *corrupt.last_mut().unwrap() ^= 1;
        assert!(
            decode_kernel(&corrupt)
                .unwrap_err()
                .to_string()
                .contains("checksum")
        );
        let mut unsupported = encode_kernel(&kernel()).unwrap();
        unsupported[4] = 4;
        assert!(
            decode_kernel(&unsupported)
                .unwrap_err()
                .to_string()
                .contains("unsupported")
        );
    }

    #[test]
    fn section_accounting_reconciles_exact_encoded_length() {
        let encoded = encode_kernel(&kernel()).unwrap();
        let accounting = account_artifact(&encoded).unwrap();
        assert_eq!(accounting.header_bytes, HEADER_BYTES);
        assert_eq!(accounting.total_bytes, encoded.len());
        assert_eq!(
            accounting
                .sections
                .iter()
                .map(|section| section.framing_bytes + section.encoded_bytes)
                .sum::<usize>()
                + HEADER_BYTES,
            encoded.len()
        );
        assert!(
            accounting
                .sections
                .iter()
                .all(|section| section.raw_bytes >= section.encoded_bytes || !section.compressed)
        );
    }

    #[test]
    fn reads_v2_1_payloads_and_emits_v3() {
        let encoded_v3 = encode_kernel(&kernel()).unwrap();
        let expected = decode_kernel(&encoded_v3).unwrap();
        let mut encoded_v2 = encoded_v3;
        encoded_v2[..4].copy_from_slice(V2_MAGIC);
        encoded_v2[4] = V2_VERSION_MAJOR;
        encoded_v2[5] = V2_VERSION_MINOR;
        assert_eq!(decode_kernel(&encoded_v2).unwrap(), expected);
        assert!(encode_kernel(&expected).unwrap().starts_with(MAGIC));
    }

    #[test]
    fn reads_v3_0_and_v3_1_symbolic_payloads() {
        let encoded = encode_kernel(&kernel()).unwrap();
        let expected = decode_kernel(&encoded).unwrap();
        for minor in [0, 1] {
            let mut legacy = encoded.clone();
            legacy[5] = minor;
            assert_eq!(decode_kernel(&legacy).unwrap(), expected);
        }
    }

    fn assert_canonical(mut value: Kernel) {
        value.compliant = false;
        let encoded = encode_kernel(&value).unwrap();
        let decoded = decode_kernel(&encoded).unwrap();
        assert_eq!(encode_kernel(&decoded).unwrap(), encoded);
    }

    #[test]
    fn every_bounded_v3_operator_has_a_canonical_codec() {
        for transform in [
            Transform::Identity,
            Transform::Log1p { scale: 3.0 },
            Transform::Power { exponent: 2.0 },
            Transform::Logit { epsilon: 0.01 },
            Transform::Winsorize {
                lower: 0.1,
                upper: 0.9,
            },
        ] {
            let mut value = kernel();
            value.schema[0].transform = transform;
            assert_canonical(value);
        }

        for marginal in [
            Marginal::Beta {
                alpha: 2.0,
                beta: 3.0,
            },
            Marginal::Gaussian {
                mean: 0.5,
                sigma: 0.1,
            },
            Marginal::Histogram {
                edges: vec![0.0, 0.25, 1.0],
                probabilities: vec![0.3, 0.7],
            },
            Marginal::ZeroInflated {
                point: 0.0,
                probability: 0.2,
                base: Box::new(Marginal::Beta {
                    alpha: 2.0,
                    beta: 2.0,
                }),
            },
        ] {
            let mut value = kernel();
            value.marginals[0] = marginal;
            assert_canonical(value);
        }

        for dependence in [
            Dependence::GaussianCopula {
                cholesky: vec![1.0, 0.0, 0.3, 0.95],
            },
            Dependence::SparseGraph {
                edges: vec![CopulaEdge {
                    parent: 0,
                    child: 1,
                    correlation: 0.3,
                }],
            },
            Dependence::Poet {
                rank: 1,
                loadings: vec![FactorLoading {
                    feature: 1,
                    factor: 0,
                    loading: 0.4,
                }],
                residual_edges: vec![],
            },
            Dependence::Vine {
                edges: vec![VineEdge {
                    left: 0,
                    right: 1,
                    conditioning_depth: 0,
                    parameter: 0.4,
                }],
            },
            Dependence::Mixture {
                weights: vec![0.4, 0.6],
                components: vec![
                    Dependence::Independent,
                    Dependence::Triangular {
                        terms: vec![TriangularTerm {
                            parent: 0,
                            child: 1,
                            coefficient: 0.2,
                        }],
                    },
                ],
            },
            Dependence::Triangular {
                terms: vec![TriangularTerm {
                    parent: 0,
                    child: 1,
                    coefficient: 0.2,
                }],
            },
        ] {
            let mut value = kernel();
            value.symbolic_mut().unwrap().0.clone_from(&dependence);
            assert_canonical(value);
        }

        let additive = AdditiveTerm {
            feature: 0,
            knots: vec![0.0, 0.5, 1.0],
            values: vec![-0.2, 0.0, 0.2],
        };
        for target in [
            Target::SparseGam {
                intercept: 0.2,
                logistic: false,
                terms: vec![additive.clone()],
            },
            Target::Ga2m {
                intercept: 0.2,
                logistic: false,
                main_terms: vec![additive],
                interactions: vec![InteractionTerm {
                    left: 0,
                    right: 1,
                    left_knots: vec![0.0, 0.5, 1.0],
                    right_knots: vec![0.0, 0.5, 1.0],
                    values: vec![0.0; 9],
                }],
            },
            Target::Mars {
                intercept: 0.2,
                logistic: false,
                terms: vec![MarsTerm {
                    coefficient: 0.3,
                    factors: vec![MarsFactor {
                        feature: 0,
                        knot: 0.5,
                        direction: 1,
                    }],
                }],
            },
            Target::ObliviousTree {
                logistic: false,
                features: vec![0, 1],
                thresholds: vec![0.5, 0.5],
                leaves: vec![0.1, 0.3, 0.7, 0.9],
            },
            Target::CompactNeuralResidual {
                intercept: 0.2,
                logistic: false,
                linear_terms: vec![LinearTerm {
                    feature: 0,
                    coefficient: 0.3,
                }],
                hidden_features: vec![0, 1],
                hidden_width: 2,
                input_weights: vec![0.4, -0.2, -0.1, 0.5],
                hidden_biases: vec![0.1, -0.1],
                output_weights: vec![0.25, -0.3],
            },
        ] {
            let mut value = kernel();
            value.symbolic_mut().unwrap().1.clone_from(&target);
            assert_canonical(value);
        }

        for noise in [
            Noise::Heteroscedastic {
                intercept: -3.0,
                terms: vec![LinearTerm {
                    feature: 0,
                    coefficient: 0.5,
                }],
                minimum_sigma: 0.01,
                maximum_sigma: 0.2,
            },
            Noise::IsotonicCalibration {
                knots: vec![0.0, 0.5, 1.0],
                probabilities: vec![0.0, 0.4, 1.0],
            },
        ] {
            let mut value = kernel();
            value.symbolic_mut().unwrap().2.clone_from(&noise);
            assert_canonical(value);
        }
    }

    #[test]
    fn canonical_rans_round_trip() {
        let raw: Vec<u8> = (0..4096).map(|index| (index % 7) as u8).collect();
        let encoded = rans_encode(&raw);
        assert!(encoded.len() < raw.len());
        assert_eq!(rans_decode(&encoded, raw.len()).unwrap(), raw);
        assert_eq!(rans_encode(&raw), encoded);
    }
}
