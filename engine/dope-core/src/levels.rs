//! Portable multilevel sessions on format-1 programs. This is the measured
//! baseline for the shared learned encoder, not a pretrained neural model.
//! Each level gets its own RNG and search budget; higher levels cannot affect
//! lower candidates, selection or emitted bytes.
use crate::{
    eval::schema_program,
    landmark::{signature, LandmarkFrame, LandmarkOptions, TransferReference},
    rng::hash3,
    schema::{infer, IntakeOptions},
    DopeError, FitOptions, FitReport, Program, RawTable, Result, SampleOptions, Samples,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    Low,
    Medium,
    High,
    Ultra,
}
impl Level {
    pub const ALL: [Level; 4] = [Self::Low, Self::Medium, Self::High, Self::Ultra];
    pub fn name(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Ultra => "ultra",
        }
    }
    pub fn cap(self) -> usize {
        match self {
            Self::Low => 2048,
            Self::Medium => 4096,
            Self::High => 16384,
            Self::Ultra => 65536,
        }
    }
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "low" | "small" => Some(Self::Low),
            "medium" | "med" => Some(Self::Medium),
            "high" | "large" => Some(Self::High),
            "ultra" => Some(Self::Ultra),
            _ => None,
        }
    }
    pub fn parse_list(value: &str) -> Result<Vec<Self>> {
        let values = if value == "all" {
            vec![Self::Low, Self::Medium, Self::High]
        } else {
            value
                .split(',')
                .map(|s| {
                    Self::parse(s.trim())
                        .ok_or_else(|| DopeError::Input("invalid level list".into()))
                })
                .collect::<Result<Vec<_>>>()?
        };
        let set: BTreeSet<Self> = values.iter().copied().collect();
        if values.is_empty() || set.len() != values.len() {
            return Err(DopeError::Input(
                "level list must be nonempty and unique".into(),
            ));
        }
        Ok(set.into_iter().collect())
    }
}

pub struct LevelOutput {
    pub level: Level,
    pub bytes: Vec<u8>,
    pub program: Program,
    pub descriptor: Option<Vec<u8>>,
}
/// No Serialize: an export must enumerate only authorized level objects.
pub struct LevelSession {
    pub outputs: Vec<LevelOutput>,
    private: serde_json::Value,
}
impl LevelSession {
    pub fn private_report(&self) -> &serde_json::Value {
        &self.private
    }
}

#[derive(Clone, Debug)]
pub struct LevelOptions {
    pub compute_through: Level,
    pub export_levels: Vec<Level>,
    pub descriptors: bool,
    pub landmarks: LandmarkOptions,
    pub fit: FitOptions,
}
impl Default for LevelOptions {
    fn default() -> Self {
        Self {
            compute_through: Level::High,
            export_levels: vec![Level::Low, Level::Medium, Level::High],
            descriptors: false,
            landmarks: LandmarkOptions::default(),
            fit: FitOptions::default(),
        }
    }
}

pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Conservative duplicate guard across fit/query. It hashes canonical ordered
/// cells, not ambiguous delimiter joins. Values and hashes never enter errors.
pub fn ensure_disjoint(train: &RawTable, query: &RawTable) -> Result<()> {
    if train.headers.len() != query.headers.len()
        || train.headers.iter().any(|h| !query.headers.contains(h))
    {
        return Err(DopeError::Input(
            "transfer tables have different headers".into(),
        ));
    }
    let indices: Vec<usize> = train
        .headers
        .iter()
        .map(|h| query.headers.iter().position(|q| q == h).unwrap())
        .collect();
    let key = |table: &RawTable, r: usize, indices: &[usize]| -> Vec<u8> {
        let mut h = Sha256::new();
        for &j in indices {
            let s = table.columns[j][r].trim();
            let value = if crate::table::is_missing(s) {
                String::new()
            } else {
                crate::table::parse_number(s).map_or_else(
                    || s.to_string(),
                    |v| if v == 0.0 { "0".into() } else { v.to_string() },
                )
            };
            h.update((value.len() as u64).to_le_bytes());
            h.update(value.as_bytes());
        }
        h.finalize().to_vec()
    };
    let direct: Vec<usize> = (0..train.width()).collect();
    let hashes: BTreeSet<Vec<u8>> = (0..train.rows()).map(|r| key(train, r, &direct)).collect();
    if (0..query.rows()).any(|r| hashes.contains(&key(query, r, &indices))) {
        return Err(DopeError::Input(
            "transfer tables contain overlapping rows".into(),
        ));
    }
    Ok(())
}

/// Convert an artifact's samples into a frozen reference schema, including
/// class-label remapping and raw numeric units. No query data is read.
pub fn remap_samples(source: &Program, samples: &Samples, reference: &Program) -> Result<Samples> {
    if source.schema.features.len() != reference.schema.features.len()
        || source
            .schema
            .features
            .iter()
            .any(|f| !reference.schema.features.iter().any(|r| r.name == f.name))
        || source.schema.target.name != reference.schema.target.name
    {
        return Err(DopeError::Input(
            "artifact and reference schemas differ".into(),
        ));
    }
    let bad = || DopeError::Input("incompatible synthetic reference mapping".into());
    if samples.x.len() != source.features() || samples.x.iter().any(|c| c.len() != samples.y.len())
    {
        return Err(bad());
    }
    let map = |source: &crate::schema::Kind,
               target: &crate::schema::Kind,
               value: f64,
               is_target: bool|
     -> Result<f64> {
        if !value.is_finite() {
            return if is_target { Err(bad()) } else { Ok(f64::NAN) };
        }
        match (source, target) {
            (crate::schema::Kind::Numeric(a), crate::schema::Kind::Numeric(b)) => {
                let v = b.to_model(a.to_raw(value));
                if v.is_finite() {
                    Ok(v)
                } else {
                    Err(bad())
                }
            }
            (crate::schema::Kind::Categorical(a), crate::schema::Kind::Categorical(b)) => {
                let index = if is_target {
                    if value < 0.0 || value.fract() != 0.0 || value >= a.levels.len() as f64 {
                        return Err(bad());
                    }
                    value as usize
                } else {
                    a.level_from_code(value)
                };
                let label = &a.levels[index];
                let k = if is_target {
                    b.levels.iter().position(|v| v == label)
                } else {
                    b.level_of(label)
                };
                match k {
                    Some(k) => Ok(if is_target { k as f64 } else { b.code(k) }),
                    None if !is_target => Ok(f64::NAN),
                    None => Err(bad()),
                }
            }
            _ => Err(bad()),
        }
    };
    let x = reference
        .schema
        .features
        .iter()
        .map(|f| {
            let j = source
                .schema
                .features
                .iter()
                .position(|s| s.name == f.name)
                .ok_or_else(bad)?;
            samples.x[j]
                .iter()
                .map(|&v| map(&source.schema.features[j].kind, &f.kind, v, false))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let y = samples
        .y
        .iter()
        .map(|&v| {
            map(
                &source.schema.target.kind,
                &reference.schema.target.kind,
                v,
                true,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Samples { x, y })
}

/// A bounded portable baseline: one private fit/tune split, independent
/// proposals, and tune-selected carry/replace. External test data is not an
/// argument. Confirmation and learned models remain separate gated work.
pub fn fit_levels(raw: &RawTable, target: usize, opts: &LevelOptions) -> Result<LevelSession> {
    if target >= raw.width()
        || opts.export_levels.is_empty()
        || opts.export_levels.iter().any(|l| *l > opts.compute_through)
        || opts
            .export_levels
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != opts.export_levels.len()
        || raw.rows() < 100
        || opts.fit.budget.is_some()
        || opts.fit.anonymize
    {
        return Err(DopeError::Input(
            "invalid level session options or insufficient rows".into(),
        ));
    }
    opts.landmarks.validate()?;
    let mut rows: Vec<usize> = (0..raw.rows()).collect();
    rows.sort_by_key(|&r| hash3(opts.fit.seed, 0x524f4c45, r as u64));
    let nq = (raw.rows() / 5).clamp(20, 5000);
    let (query, train) = (raw.select_rows(&rows[..nq]), raw.select_rows(&rows[nq..]));
    ensure_disjoint(&train, &query)?;
    let schema = infer(
        &train,
        target,
        &IntakeOptions {
            task: opts.fit.task,
            ..Default::default()
        },
    )?;
    let reference_program = schema_program(&schema, &train)?;
    let mut combined = train.clone();
    for j in 0..raw.width() {
        combined.columns[j].extend(query.columns[j].clone());
    }
    let data = schema.to_model(&combined)?;
    let reference = TransferReference::new(
        &reference_program,
        &data,
        &(0..train.rows()).collect::<Vec<_>>(),
        &(train.rows()..combined.rows()).collect::<Vec<_>>(),
        &opts.landmarks,
        false,
    )?;
    let mut carry: Option<(Vec<u8>, Program, Option<f64>)> = None;
    let mut outputs = Vec::new();
    let mut reports = Vec::new();
    for level in Level::ALL
        .into_iter()
        .take_while(|l| *l <= opts.compute_through)
    {
        let mut fit_opts = opts.fit.clone();
        fit_opts.budget = Some(level.cap());
        fit_opts.seed = hash3(opts.fit.seed, 0x4c4556454c, level as u64);
        // Anonymization is applied only after candidate scoring so class and
        // feature names can be mapped into the fixed reference representation.
        fit_opts.anonymize = false;
        let fitted = crate::fit(&train, target, &fit_opts)?;
        if fitted.bytes.len() > level.cap() {
            return Err(DopeError::Budget(
                "level proposal exceeded standalone cap".into(),
            ));
        }
        let samples = crate::sample(
            &fitted.program,
            &SampleOptions {
                rows: train.rows().min(opts.landmarks.small_rows),
                seed: hash3(opts.fit.seed, 0x4155444954, level as u64),
                ..Default::default()
            },
        )?;
        let t = reference.score(&remap_samples(
            &fitted.program,
            &samples,
            &reference_program,
        )?)?;
        let proposal = t.mirror_retention;
        let keep_carry = carry.as_ref().is_some_and(|(b, _, score)| {
            b.len() <= level.cap()
                && match (*score, proposal) {
                    (Some(a), Some(b)) => b <= a + 1e-3,
                    (Some(_), None) => true,
                    _ => false,
                }
        });
        let fit_report: FitReport = fitted.report;
        if !keep_carry {
            carry = Some((fitted.bytes, fitted.program, proposal));
        }
        let (bytes, program, score) = carry.as_ref().unwrap();
        reports.push(serde_json::json!({"level":level,"cap":level.cap(),"carried":keep_carry,"selection_retention":score,"proposal_transfer":t.private_report(),"proposal_fit":fit_report}));
        if opts.export_levels.contains(&level) {
            let descriptor = if opts.descriptors {
                let s = crate::sample(
                    program,
                    &SampleOptions {
                        rows: 4096,
                        seed: 0x4445534352495054,
                        ..Default::default()
                    },
                )?;
                let mut lo = opts.landmarks.clone();
                lo.seed = 0x4445534352495054;
                lo.spectrum_features = 0;
                Some(serde_json::to_vec(
                    &signature(&LandmarkFrame::synthetic(program, &s)?, &lo)?.global_coordinates(),
                )?)
            } else {
                None
            };
            let (bytes, program) = (bytes.clone(), program.clone());
            if bytes.len() > level.cap() {
                return Err(DopeError::Budget(
                    "released level exceeded standalone cap".into(),
                ));
            }
            outputs.push(LevelOutput {
                level,
                bytes,
                program,
                descriptor,
            });
        }
    }
    Ok(LevelSession {
        outputs,
        private: serde_json::json!({"protocol":"portable-levels-pilot-v1","privacy":"private","confirmation":"not_run","external_transfer":"not_measured","levels":reports}),
    })
}
