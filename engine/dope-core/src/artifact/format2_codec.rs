//! End-to-end Format 2 binary program codec and stateless emitter (`G-01`, `G-02`, `G-05`).
//!
//! Encodes & decodes a complete runnable `Format2Program` (`Schema`, `n_fit`, `ColumnMarginalV2`,
//! `CatfModel`, `RuleSet`, `DistilledStudent` with bit-packed `Forest`, `TeacherResidualModel`,
//! and 32-dim `TAG_EMBED_SUMMARY`) inside `.dope` Format 2 with zero JSON in the byte cap.

use super::format2::{
    decode_format2, encode_format2, Format2Packet, Format2Section, TierPrefixEntry, CODEC_RAW,
    TAG_CATF, TAG_EMBED_SUMMARY, TAG_MARGINALS_V2, TAG_NOISE_TEMP, TAG_RULES, TAG_STUDENT_FOREST,
    TAG_STUDENT_MAIN, TAG_STUDENT_PAIRS, TAG_UNIT_SCHEMA,
};
use super::format2_bin::{
    decode_marginals_section, decode_noise_temp, decode_schema_section, encode_marginals_section,
    encode_noise_temp, encode_schema_section, BinReader, BinWriter,
};
use super::format2_rules_catf::{
    decode_catf_section, decode_rules_section, encode_catf_section, encode_rules_section,
};
use crate::catf::sample::sample_post_rule_targets;
use crate::catf::CatfModel;
use crate::distil::main_effects::{MainEffect, MainEffectSet};
use crate::distil::pairs::PairEffect;
use crate::distil::DistilledStudent;
use crate::error::{DopeError, Result};
use crate::forest::Forest;
use crate::sample::{EmissionMode, EmitOptions, EmittedSamples, Samples};
use crate::schema::{Kind, Schema};
use crate::target::Head;
use std::io::Write;

pub const FORMAT2_EMBED_SUMMARY_DIM: usize = 32;

#[derive(Clone, Debug)]
pub struct Format2Program {
    pub schema: Schema,
    pub col_f32_heritage: Vec<bool>,
    pub n_fit: u64,
    pub budget_bytes: u32,
    pub catf: CatfModel,
    pub student: DistilledStudent,
    pub embed_summary: [f64; FORMAT2_EMBED_SUMMARY_DIM],
}

impl Format2Program {
    pub fn features(&self) -> usize {
        self.schema.features.len()
    }

    /// True `n_fit`-driven `--rows auto` recommendation (`faithful` 4x, `distill` 8x, `matched` 1x).
    pub fn auto_rows(&self, mode: EmissionMode) -> usize {
        let nf = (self.n_fit as usize).max(1);
        match mode {
            EmissionMode::Faithful => (nf.saturating_mul(4)).clamp(2_000, 200_000),
            EmissionMode::Distill => (nf.saturating_mul(8)).clamp(10_000, 2_000_000),
            EmissionMode::Matched | EmissionMode::Legacy => nf,
        }
    }

    /// Stateless Format-2 emission supporting `faithful`, `distill` (antithetic pairs + weights
    /// sidecar), and `matched` modes directly from the decoded Format-2 program.
    pub fn sample_with_mode(&self, opts: &EmitOptions) -> Result<EmittedSamples> {
        let rows = opts.rows.max(1);
        match opts.mode {
            EmissionMode::Distill => {
                // Rao-Blackwellized conditional emission (EM-12): weights sum to 1.0 per pair_id.
                let d = self.features();
                let (n_base, is_reg) = match self.student.head {
                    crate::target::Head::Regression => (rows, true),
                    crate::target::Head::Binary => (rows.div_ceil(2), false),
                    crate::target::Head::Multiclass(k) => ((rows / k.max(2)).max(1), false),
                };
                let xs = self.catf.sample_x(n_base, opts.seed, opts.table_id);
                let pred = self.student.predict_expectations(&xs);
                if is_reg {
                    return Ok(EmittedSamples {
                        samples: Samples { x: xs, y: pred },
                        weights: Some(vec![1.0; n_base]),
                        pair_ids: Some((0..n_base as u64).collect()),
                        mode: EmissionMode::Distill,
                        table_id: opts.table_id,
                    });
                }
                let mut x = vec![Vec::with_capacity(rows); d];
                let (mut y, mut weights, mut pair_ids) = (Vec::new(), Vec::new(), Vec::new());
                for r in 0..n_base {
                    let pairs: Vec<(f64, f64)> = match self.student.head {
                        crate::target::Head::Binary => {
                            let p1 = pred[r].clamp(1e-5, 1.0 - 1e-5);
                            vec![(0.0, 1.0 - p1), (1.0, p1)]
                        }
                        crate::target::Head::Multiclass(k) => {
                            let sl = &pred[r * k..(r + 1) * k];
                            let mut ks: Vec<(f64, f64)> = sl
                                .iter()
                                .enumerate()
                                .filter(|&(_, &p)| p >= 1e-4)
                                .map(|(c, &p)| (c as f64, p))
                                .collect();
                            let sp: f64 = ks.iter().map(|&(_, p)| p).sum::<f64>().max(1e-12);
                            for kv in &mut ks {
                                kv.1 /= sp;
                            }
                            ks
                        }
                        crate::target::Head::Regression => unreachable!(),
                    };
                    for (lbl, w) in pairs {
                        for j in 0..d {
                            x[j].push(xs[j][r]);
                        }
                        y.push(lbl);
                        weights.push(w);
                        pair_ids.push(r as u64);
                    }
                }
                Ok(EmittedSamples {
                    samples: Samples { x, y },
                    weights: Some(weights),
                    pair_ids: Some(pair_ids),
                    mode: EmissionMode::Distill,
                    table_id: opts.table_id,
                })
            }
            EmissionMode::Faithful | EmissionMode::Matched | EmissionMode::Legacy => {
                let samples =
                    self.catf
                        .sample_with_student(&self.student, rows, opts.seed, opts.table_id);
                Ok(EmittedSamples {
                    samples,
                    weights: None,
                    pair_ids: None,
                    mode: opts.mode,
                    table_id: opts.table_id,
                })
            }
        }
    }

    /// Diagnostic `--given X` conditional sampler: draws `Y ~ Q(Y|X)` at supplied covariates `x`.
    pub fn sample_given(&self, x_model: &[Vec<f64>], seed: u64) -> Result<Samples> {
        if x_model.len() != self.features() {
            return Err(DopeError::Input(
                "given feature count does not match Format 2 program".into(),
            ));
        }
        Ok(sample_post_rule_targets(&self.student, x_model, seed, 0))
    }

    /// Write emitted samples in raw CSV units with exact shortest float precision (`T27`).
    pub fn write_raw<W: Write>(&self, s: &Samples, mut w: W) -> Result<()> {
        let mut header: Vec<&str> = self
            .schema
            .features
            .iter()
            .map(|c| c.name.as_str())
            .collect();
        header.push(self.schema.target.name.as_str());
        writeln!(w, "{}", header.join(","))?;
        let n = s.y.len();
        let d = self.features();
        let t_f32 = self.col_f32_heritage.get(d).copied().unwrap_or(false);
        let mut row = Vec::with_capacity(d + 1);
        for r in 0..n {
            row.clear();
            for (j, col) in self.schema.features.iter().enumerate() {
                let f32h = self.col_f32_heritage.get(j).copied().unwrap_or(false);
                row.push(format_cell(col, s.x[j][r], f32h));
            }
            let ty = match &self.schema.target.kind {
                Kind::Numeric(spec) => format_exact(spec.to_raw(s.y[r]), t_f32),
                Kind::Categorical(spec) => {
                    let idx = (s.y[r].round() as usize).min(spec.levels.len().saturating_sub(1));
                    spec.levels.get(idx).cloned().unwrap_or_default()
                }
            };
            row.push(ty);
            writeln!(w, "{}", row.join(","))?;
        }
        Ok(())
    }
}

fn format_exact(x: f64, f32_heritage: bool) -> String {
    if !x.is_finite() {
        return String::new();
    }
    let v = if x == 0.0 {
        0.0
    } else if f32_heritage {
        f64::from(x as f32)
    } else {
        x
    };
    if v == v.trunc() && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

fn format_cell(col: &crate::schema::ColumnSpec, v: f64, f32_heritage: bool) -> String {
    if !v.is_finite() {
        return String::new();
    }
    match &col.kind {
        Kind::Numeric(spec) => format_exact(spec.to_raw(v), f32_heritage),
        Kind::Categorical(spec) => {
            let idx = spec.level_from_code(v);
            spec.levels.get(idx).cloned().unwrap_or_default()
        }
    }
}

fn encode_head(w: &mut BinWriter, head: Head) {
    match head {
        Head::Regression => {
            w.u8(0);
            w.u16(1);
        }
        Head::Binary => {
            w.u8(1);
            w.u16(2);
        }
        Head::Multiclass(k) => {
            w.u8(2);
            w.u16(k as u16);
        }
    }
}

fn decode_head(r: &mut BinReader<'_>) -> Result<Head> {
    let tag = r.u8()?;
    let k = r.u16()? as usize;
    match tag {
        0 => Ok(Head::Regression),
        1 => Ok(Head::Binary),
        2 if k >= 2 => Ok(Head::Multiclass(k)),
        _ => Err(DopeError::Artifact("format2 invalid head".into())),
    }
}

fn encode_student_main(student: &DistilledStudent) -> Vec<u8> {
    let mut w = BinWriter::new();
    encode_head(&mut w, student.head);
    w.u32(student.d as u32);
    w.f32_slice(&student.impute);
    w.u32(student.main_effects.outputs as u32);
    w.f64_slice(&student.main_effects.intercept);
    w.u32(student.main_effects.effects.len() as u32);
    for e in &student.main_effects.effects {
        w.u32(e.feature as u32);
        w.u32(e.outputs as u32);
        w.f32_slice(&e.edges);
        w.f32_slice(&e.values);
    }
    w.0
}

fn decode_student_main(bytes: &[u8]) -> Result<(Head, usize, Vec<f64>, MainEffectSet)> {
    let mut r = BinReader::new(bytes);
    let head = decode_head(&mut r)?;
    let d = r.u32()? as usize;
    let impute = r.f32_vec()?;
    let outputs = r.u32()? as usize;
    let intercept = r.f64_vec()?;
    let ne = r.len()?;
    let mut effects = Vec::with_capacity(ne);
    for _ in 0..ne {
        effects.push(MainEffect {
            feature: r.u32()? as usize,
            outputs: r.u32()? as usize,
            edges: r.f32_vec()?,
            values: r.f32_vec()?,
        });
    }
    Ok((
        head,
        d,
        impute,
        MainEffectSet {
            intercept,
            effects,
            outputs,
        },
    ))
}

fn encode_student_pairs(pairs: &[PairEffect]) -> Vec<u8> {
    let mut w = BinWriter::new();
    w.u32(pairs.len() as u32);
    for p in pairs {
        w.u32(p.feat_a as u32);
        w.u32(p.feat_b as u32);
        w.u32(p.outputs as u32);
        w.f32(p.gain);
        w.f32_slice(&p.edges_a);
        w.f32_slice(&p.edges_b);
        w.f32_slice(&p.grid);
    }
    w.0
}

fn decode_student_pairs(bytes: &[u8]) -> Result<Vec<PairEffect>> {
    let mut r = BinReader::new(bytes);
    let np = r.len()?;
    let mut out = Vec::with_capacity(np);
    for _ in 0..np {
        out.push(PairEffect {
            feat_a: r.u32()? as usize,
            feat_b: r.u32()? as usize,
            outputs: r.u32()? as usize,
            gain: r.f32()?,
            edges_a: r.f32_vec()?,
            edges_b: r.f32_vec()?,
            grid: r.f32_vec()?,
        });
    }
    Ok(out)
}

fn encode_student_forest(student: &DistilledStudent) -> Vec<u8> {
    let mut w = BinWriter::new();
    let f = &student.forest;
    w.u32(f.trees() as u32);
    w.u16(f.depth as u16);
    w.u16(f.outputs as u16);
    w.u8(f.bits);
    let blob = f.to_bytes(student.d);
    w.u32(blob.len() as u32);
    w.0.extend(blob);
    w.0
}

fn decode_student_forest(bytes: &[u8], d: usize) -> Result<Forest> {
    let mut r = BinReader::new(bytes);
    let trees = r.u32()? as usize;
    let depth = r.u16()? as usize;
    let outputs = r.u16()? as usize;
    let bits = r.u8()?;
    let blen = r.len()?;
    let blob = r.take(blen)?;
    Forest::from_bytes(blob, d, trees, depth, outputs, bits)
        .ok_or_else(|| DopeError::Artifact("format2 invalid quantized forest".into()))
}

fn encode_embed_summary(summary: &[f64; FORMAT2_EMBED_SUMMARY_DIM]) -> Vec<u8> {
    let mut w = BinWriter::new();
    for &v in summary {
        w.f32(v);
    }
    w.0
}

pub(crate) fn decode_embed_summary(bytes: &[u8]) -> Result<[f64; FORMAT2_EMBED_SUMMARY_DIM]> {
    let mut r = BinReader::new(bytes);
    let mut out = [0.0; FORMAT2_EMBED_SUMMARY_DIM];
    for slot in &mut out {
        *slot = r.f32()?;
    }
    Ok(out)
}

/// Serialize a complete `Format2Program` into a compressed, CRC32-verified `.dope` Format 2 file.
pub fn encode_format2_program(
    prog: &Format2Program,
    tiers: Vec<TierPrefixEntry>,
) -> Result<Vec<u8>> {
    let pkt = Format2Packet {
        flags: 1,
        tiers,
        sections: [
            (
                TAG_UNIT_SCHEMA,
                encode_schema_section(
                    &prog.schema,
                    &prog.col_f32_heritage,
                    prog.n_fit,
                    prog.budget_bytes,
                ),
            ),
            (
                TAG_MARGINALS_V2,
                encode_marginals_section(&prog.catf.marginals),
            ),
            (TAG_RULES, encode_rules_section(&prog.catf.rules)),
            (TAG_CATF, encode_catf_section(&prog.catf)),
            (TAG_STUDENT_MAIN, encode_student_main(&prog.student)),
            (
                TAG_STUDENT_PAIRS,
                encode_student_pairs(&prog.student.pair_effects),
            ),
            (TAG_STUDENT_FOREST, encode_student_forest(&prog.student)),
            (
                TAG_NOISE_TEMP,
                encode_noise_temp(prog.student.temperature, &prog.student.residual_model),
            ),
            (TAG_EMBED_SUMMARY, encode_embed_summary(&prog.embed_summary)),
        ]
        .into_iter()
        .map(|(tag, data)| Format2Section {
            tag,
            codec: CODEC_RAW,
            data,
        })
        .collect(),
    };
    encode_format2(&pkt)
}

/// Decode and validate a complete runnable `Format2Program` from `.dope` Format 2 bytes.
pub fn decode_format2_program(bytes: &[u8]) -> Result<Format2Program> {
    let pkt = decode_format2(bytes)?;
    let find = |tag: u16| -> Result<&[u8]> {
        pkt.sections
            .iter()
            .find(|s| s.tag == tag)
            .map(|s| s.data.as_slice())
            .ok_or_else(|| DopeError::Artifact(format!("format2 missing required section {tag}")))
    };
    let (schema, col_f32_heritage, n_fit, budget_bytes) =
        decode_schema_section(find(TAG_UNIT_SCHEMA)?)?;
    let marginals = decode_marginals_section(find(TAG_MARGINALS_V2)?)?;
    let rules = decode_rules_section(find(TAG_RULES)?)?;
    let catf = decode_catf_section(find(TAG_CATF)?, marginals, rules)?;
    let (head, d, impute, main_effects) = decode_student_main(find(TAG_STUDENT_MAIN)?)?;
    let pair_effects = decode_student_pairs(find(TAG_STUDENT_PAIRS)?)?;
    let forest = decode_student_forest(find(TAG_STUDENT_FOREST)?, d)?;
    let (temperature, residual_model) = decode_noise_temp(find(TAG_NOISE_TEMP)?)?;
    let embed_summary = match find(TAG_EMBED_SUMMARY) {
        Ok(b) if b.len() == FORMAT2_EMBED_SUMMARY_DIM * 4 => decode_embed_summary(b)?,
        _ => [0.0; FORMAT2_EMBED_SUMMARY_DIM],
    };
    let forest_trees = forest.trees();
    let forest_bytes = forest.to_bytes(d).len();
    let student = DistilledStudent {
        head,
        d,
        impute,
        main_effects,
        pair_effects,
        forest,
        forest_trees,
        forest_bytes,
        temperature,
        residual_model,
    };
    Ok(Format2Program {
        schema,
        col_f32_heritage,
        n_fit,
        budget_bytes,
        catf,
        student,
        embed_summary,
    })
}
