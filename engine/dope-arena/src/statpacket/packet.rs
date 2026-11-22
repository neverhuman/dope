//! In-memory form of a `dope.single-encoding/v1` statistics packet (kind 0).
//!
//! Every field is an integer code exactly as it travels on the wire, so two
//! packets are equal if and only if their bytes are equal. Floating-point
//! meaning is attached by the dequantisers in `marginals`, `dependence` and
//! `decode`. The byte layout lives in `sections` and `codec`; the normative
//! text is `docs/plans/v2-gap-closure/single-encoding-v1.md`.

pub use super::binary16::{f16_to_f64, f64_to_f16};
use super::StatError;

pub const MAGIC: [u8; 4] = *b"DSE1";
pub const VERSION: u8 = 1;
pub const KIND_STAT: u8 = 0;
pub const BIT_PROFILE_V1: u8 = 0;
pub const HEADER_LEN: usize = 24;
pub const CRC_LEN: usize = 4;
/// Factors stored at level E; levels B and C/D use the first 4 and 8.
pub const MAX_RANK: usize = 12;
pub const RANK_BLOCK: usize = 4;
/// Entries of the pair section S10 (always written in full).
pub const PAIRS: usize = 16;
/// Bins of the main-effect section S8.
pub const MAIN_BINS: usize = 4;
/// Support floor: a discrete level is stored only if at least this many FIT rows carry it.
pub const SUPPORT_K: usize = 10;
pub const MAX_ATOMS: usize = 16;
/// Largest 6-bit code.
pub const CODE6_MAX: u8 = 63;
/// Largest 12-bit half-span code.
pub const HALF_SPAN_CODE_MAX: u16 = 4095;
/// Header flag: S3 starts with a column bitmap and stores weights for the set columns only.
pub const FLAG_SPARSE_TARGET: u8 = 1;
/// Smallest and largest accepted `floor(log2 n_fit)`.
pub const LOG2_N_MIN: u8 = 4;
pub const LOG2_N_MAX: u8 = 62;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Task {
    Regression = 0,
    Binary = 1,
    Multiclass = 2,
    Multilabel = 3,
}

impl Task {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Regression),
            1 => Some(Self::Binary),
            2 => Some(Self::Multiclass),
            3 => Some(Self::Multilabel),
            _ => None,
        }
    }
    /// Whether header `K` is legal for this task.
    pub fn accepts_k(self, k: u8) -> bool {
        match self {
            Self::Regression => k == 0,
            Self::Binary => k == 2,
            Self::Multiclass => k >= 3,
            Self::Multilabel => k >= 1,
        }
    }
}

/// Nested level: A = S1-S3, B = +S4, C = +S5,S6, D = +S7,S8, E = +S9,S10.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
pub enum Level {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
    E = 4,
}

impl Level {
    pub const ALL: [Level; 5] = [Level::A, Level::B, Level::C, Level::D, Level::E];
    pub fn from_u8(v: u8) -> Option<Self> {
        Self::ALL.get(v as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|l| l.name() == s)
    }
    pub fn name(self) -> &'static str {
        ["A", "B", "C", "D", "E"][self as usize]
    }
    /// Number of sections on the wire.
    pub fn sections(self) -> usize {
        [3, 4, 6, 8, 10][self as usize]
    }
    /// Factors available to the sampler.
    pub fn rank(self) -> usize {
        [0, 4, 8, 8, 12][self as usize]
    }
    /// Knots per continuous column.
    pub fn knots(self) -> usize {
        [5, 5, 9, 17, 17][self as usize]
    }
}

/// Atom values of a discrete column, ascending, in unit space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AtomValues {
    /// `value_a = f16(base) + f16(step) * index_a`, indices strictly ascending.
    Lattice {
        base: u16,
        step: u16,
        index: Vec<u8>,
    },
    /// One IEEE binary16 value per atom, strictly ascending.
    Explicit(Vec<u16>),
}

impl AtomValues {
    pub fn len(&self) -> usize {
        match self {
            Self::Lattice { index, .. } => index.len(),
            Self::Explicit(v) => v.len(),
        }
    }
}

/// 2..=16 atoms; `prob` holds one 6-bit weight (1..=63) per atom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtomTable {
    pub values: AtomValues,
    pub prob: Vec<u8>,
}

impl AtomTable {
    /// Bytes of the S2 payload that follow the flag byte.
    pub fn payload_len(&self) -> usize {
        let a = self.prob.len();
        let probs = (6 * a).div_ceil(8);
        match &self.values {
            AtomValues::Lattice { .. } => 4 + a + probs,
            AtomValues::Explicit(_) => 2 * a + probs,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ColumnMarginal {
    /// Every FIT row holds the same value; decodes to unit 0 (the scale card restores the constant).
    Constant,
    /// `lo`/`hi`/`split` are S2, `mid9` is S5, `mid17` is S7 (zero while the level omits them).
    Continuous {
        /// 12-bit log codes of the distance from the origin down to knot 0 and up to knot 16.
        lo: u16,
        hi: u16,
        /// 8-bit positions of knots 4 and 12 inside their half.
        split: [u8; 2],
        mid9: [u8; 4],
        mid17: [u8; 8],
    },
    Discrete(AtomTable),
}

impl ColumnMarginal {
    pub fn is_continuous(&self) -> bool {
        matches!(self, Self::Continuous { .. })
    }
    pub fn payload_len(&self) -> usize {
        match self {
            Self::Constant => 0,
            Self::Continuous { .. } => 5,
            Self::Discrete(t) => t.payload_len(),
        }
    }
}

/// One S10 entry: columns `a`, `b` and one coefficient per output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairTerm {
    pub a: u16,
    pub b: u16,
    pub coef: Vec<i8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packet {
    pub task: Task,
    /// Header K: 0 regression, 2 binary, classes for multiclass, labels for multilabel.
    pub k: u8,
    pub p: u16,
    /// Highest level whose sections this value holds.
    pub level: Level,
    pub log2_n: u8,
    /// S1: `K-1` class codes (binary, multiclass; they sum to at most 255) or `K` label codes.
    pub prior: Vec<u8>,
    pub resid_sd: u8,
    pub temperature: u8,
    /// S2 (+ S5, S7 fields of continuous columns), one entry per column.
    pub columns: Vec<ColumnMarginal>,
    /// S3. Dense storage `outputs x P`; a clear `target_mask` bit means a zero weight.
    pub target_sparse: bool,
    pub target_mask: Vec<bool>,
    pub target_scale: Vec<u16>,
    pub target_intercept: Vec<u16>,
    pub target_weight: Vec<i8>,
    /// S4/S6/S9: `P x 12` loadings and `regime_means x 12` class factor means.
    pub loading: Vec<i8>,
    pub class_mean: Vec<i8>,
    /// S8. Dense storage `P x outputs x 4`; a clear `main_mask` bit means zeros.
    pub main_mask: Vec<bool>,
    pub main_effect: Vec<i8>,
    /// S10: exactly 16 entries at level E, empty below.
    pub pairs: Vec<PairTerm>,
}

impl Packet {
    pub fn cols(&self) -> usize {
        self.p as usize
    }
    /// Target outputs O.
    pub fn outputs(&self) -> usize {
        match self.task {
            Task::Regression | Task::Binary => 1,
            Task::Multiclass => (self.k as usize).saturating_sub(1),
            Task::Multilabel => self.k as usize,
        }
    }
    /// Class regimes that carry factor means (0 means a single regime).
    pub fn regime_means(&self) -> usize {
        match self.task {
            Task::Binary | Task::Multiclass => self.k as usize,
            Task::Regression | Task::Multilabel => 0,
        }
    }
    pub fn prior_len(&self) -> usize {
        match self.task {
            Task::Regression => 0,
            Task::Binary | Task::Multiclass => (self.k as usize).saturating_sub(1),
            Task::Multilabel => self.k as usize,
        }
    }
    pub fn continuous_cols(&self) -> usize {
        self.columns.iter().filter(|c| c.is_continuous()).count()
    }
    /// Bytes of one column index in S10.
    pub fn pair_index_len(&self) -> usize {
        if self.cols() <= 256 {
            1
        } else {
            2
        }
    }
    pub fn bitmap_len(&self) -> usize {
        self.cols().div_ceil(8)
    }

    /// An all-zero target, dependence and effect skeleton around the given marginals.
    pub fn skeleton(task: Task, k: u8, log2_n: u8, columns: Vec<ColumnMarginal>) -> Self {
        let p = columns.len();
        let mut out = Self {
            task,
            k,
            p: p as u16,
            level: Level::E,
            log2_n,
            prior: Vec::new(),
            resid_sd: 0,
            temperature: 0,
            columns,
            target_sparse: false,
            target_mask: vec![true; p],
            target_scale: Vec::new(),
            target_intercept: Vec::new(),
            target_weight: Vec::new(),
            loading: vec![0; p * MAX_RANK],
            class_mean: Vec::new(),
            main_mask: vec![true; p],
            main_effect: Vec::new(),
            pairs: Vec::new(),
        };
        let o = out.outputs();
        out.prior = vec![0; out.prior_len()];
        out.target_scale = vec![0; o];
        out.target_intercept = vec![0; o];
        out.target_weight = vec![0; o * p];
        out.class_mean = vec![0; out.regime_means() * MAX_RANK];
        out.main_effect = vec![0; p * o * MAIN_BINS];
        out.pairs = (0..PAIRS)
            .map(|_| PairTerm {
                a: 0,
                b: 0,
                coef: vec![0; o],
            })
            .collect();
        out
    }

    /// The same packet with every section above `level` cleared to its canonical empty form.
    pub fn at_level(&self, level: Level) -> Result<Self, StatError> {
        if level > self.level {
            return Err(StatError::Invalid(
                "requested level exceeds the packet level",
            ));
        }
        let mut out = self.clone();
        out.level = level;
        let rank = level.rank();
        for row in out.loading.chunks_mut(MAX_RANK) {
            row[rank..].fill(0);
        }
        for row in out.class_mean.chunks_mut(MAX_RANK) {
            row[rank..].fill(0);
        }
        for col in &mut out.columns {
            if let ColumnMarginal::Continuous { mid9, mid17, .. } = col {
                if level < Level::C {
                    *mid9 = [0; 4];
                }
                if level < Level::D {
                    *mid17 = [0; 8];
                }
            }
        }
        if level < Level::D {
            out.main_mask.fill(false);
            out.main_effect.fill(0);
        }
        if level < Level::E {
            out.pairs.clear();
        }
        Ok(out)
    }

    /// Structural invariants; `pack` refuses anything else, `unpack` never returns anything else.
    pub fn validate(&self) -> Result<(), StatError> {
        let (p, o) = (self.cols(), self.outputs());
        let bad = |why: &'static str| Err(StatError::Invalid(why));
        if p == 0 || self.columns.len() != p {
            return bad("column count does not match the header");
        }
        if !self.task.accepts_k(self.k) {
            return bad("class count is not valid for the task");
        }
        if !(LOG2_N_MIN..=LOG2_N_MAX).contains(&self.log2_n) {
            return bad("row-count exponent out of range");
        }
        if self.prior.len() != self.prior_len() {
            return bad("prior length");
        }
        let class_task = matches!(self.task, Task::Binary | Task::Multiclass);
        if class_task && self.prior.iter().map(|&c| c as u32).sum::<u32>() > 255 {
            return bad("class priors exceed the unit total");
        }
        if self.target_mask.len() != p
            || self.target_scale.len() != o
            || self.target_intercept.len() != o
            || self.target_weight.len() != o * p
        {
            return bad("target section shape");
        }
        if !self.target_sparse && self.target_mask.iter().any(|&m| !m) {
            return bad("dense target with a cleared column");
        }
        if self.loading.len() != p * MAX_RANK
            || self.class_mean.len() != self.regime_means() * MAX_RANK
        {
            return bad("dependence section shape");
        }
        if self.main_mask.len() != p || self.main_effect.len() != p * o * MAIN_BINS {
            return bad("main-effect section shape");
        }
        let pairs_ok = if self.level == Level::E {
            self.pairs.len() == PAIRS
        } else {
            self.pairs.is_empty()
        };
        if !pairs_ok
            || self
                .pairs
                .iter()
                .any(|t| t.coef.len() != o || t.a as usize >= p || t.b as usize >= p)
        {
            return bad("pair section shape");
        }
        for (j, on) in self.target_mask.iter().enumerate() {
            if !on && (0..o).any(|k| self.target_weight[k * p + j] != 0) {
                return bad("weight stored for a cleared target column");
            }
        }
        for (j, on) in self.main_mask.iter().enumerate() {
            let cell = &self.main_effect[j * o * MAIN_BINS..(j + 1) * o * MAIN_BINS];
            if !on && cell.iter().any(|&v| v != 0) {
                return bad("main effect stored for a cleared column");
            }
        }
        if *self != self.at_level(self.level)? {
            return bad("sections above the packet level are not empty");
        }
        self.columns.iter().try_for_each(validate_column)
    }
}

fn validate_column(col: &ColumnMarginal) -> Result<(), StatError> {
    let bad = |why: &'static str| Err(StatError::Invalid(why));
    match col {
        ColumnMarginal::Constant => Ok(()),
        ColumnMarginal::Continuous {
            lo,
            hi,
            mid9,
            mid17,
            ..
        } => {
            if *lo > HALF_SPAN_CODE_MAX || *hi > HALF_SPAN_CODE_MAX {
                return bad("half-span code exceeds 12 bits");
            }
            if mid9.iter().chain(mid17).any(|&c| c > CODE6_MAX) {
                return bad("knot code exceeds 6 bits");
            }
            Ok(())
        }
        ColumnMarginal::Discrete(t) => {
            let a = t.prob.len();
            if !(2..=MAX_ATOMS).contains(&a) || t.values.len() != a {
                return bad("atom count");
            }
            if t.prob.iter().any(|&c| c == 0 || c > CODE6_MAX) {
                return bad("atom weight outside 1..=63");
            }
            let ascending = match &t.values {
                AtomValues::Lattice { base, step, index } => {
                    let (base, step) = (f16_to_f64(*base), f16_to_f64(*step));
                    base.is_finite()
                        && step.is_finite()
                        && step > 0.0
                        && index.windows(2).all(|w| w[0] < w[1])
                }
                AtomValues::Explicit(v) => {
                    v.iter().all(|&h| f16_to_f64(h).is_finite())
                        && v.windows(2).all(|w| f16_to_f64(w[0]) < f16_to_f64(w[1]))
                }
            };
            if !ascending {
                return bad("atoms are not finite and strictly ascending");
            }
            Ok(())
        }
    }
}
