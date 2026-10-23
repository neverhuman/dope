use super::SignatureKind;
use crate::eval::Matrix;
use crate::stats::{mean, quantile_sorted};
use crate::{
    schema::{Kind, ModelData, Schema},
    target::Head,
    DopeError, Program, Result, Samples,
};
use std::collections::BTreeSet;

/// Original feature slots and physical row identities, with no headers/labels.
/// Construction never fits a transform on query rows. A group id can also be
/// a time-block id; folds are blocked, not forward-chaining time-series CV.
#[derive(Clone)]
pub struct LandmarkFrame {
    pub(crate) columns: Vec<Vec<f64>>,
    pub(crate) categories: Vec<Option<usize>>,
    pub(crate) y: Vec<f64>,
    pub(crate) ids: Vec<u64>,
    pub(crate) groups: Option<Vec<u64>>,
    pub(crate) head: Head,
    pub(crate) kind: SignatureKind,
}

impl LandmarkFrame {
    pub fn real(
        schema: &Schema,
        data: &ModelData,
        rows: &[usize],
        groups: Option<&[u64]>,
    ) -> Result<Self> {
        if data.x.len() != schema.features.len()
            || data.x.iter().any(|c| c.len() != data.y.len())
            || groups.is_some_and(|g| g.len() != data.y.len())
            || rows.iter().any(|&r| r >= data.y.len())
        {
            return Err(DopeError::Input("invalid landmark frame shape".into()));
        }
        let ids: Vec<u64> = rows.iter().map(|&r| r as u64).collect();
        if ids.iter().collect::<BTreeSet<_>>().len() != ids.len() {
            return Err(DopeError::Input("duplicate landmark row identity".into()));
        }
        let columns = schema
            .features
            .iter()
            .enumerate()
            .map(|(j, f)| {
                rows.iter()
                    .map(|&r| match &f.kind {
                        Kind::Numeric(s) if data.x[j][r].is_finite() => {
                            let t = s.lo + data.x[j][r] * (s.hi - s.lo);
                            if s.log {
                                t.exp_m1()
                            } else {
                                t
                            }
                        }
                        Kind::Categorical(c) if data.x[j][r].is_finite() => {
                            c.level_from_code(data.x[j][r]) as f64
                        }
                        _ => f64::NAN,
                    })
                    .collect()
            })
            .collect();
        let y = rows
            .iter()
            .map(|&r| match &schema.target.kind {
                Kind::Numeric(s) if data.y[r].is_finite() => {
                    let t = s.lo + data.y[r] * (s.hi - s.lo);
                    if s.log {
                        t.exp_m1()
                    } else {
                        t
                    }
                }
                _ => data.y[r],
            })
            .collect();
        let f = Self {
            columns,
            categories: schema
                .features
                .iter()
                .map(|f| match &f.kind {
                    Kind::Numeric(_) => None,
                    Kind::Categorical(c) => Some(c.levels.len()),
                })
                .collect(),
            y,
            ids,
            groups: groups.map(|g| rows.iter().map(|&r| g[r]).collect()),
            head: Program::head_for(schema),
            kind: SignatureKind::Real,
        };
        f.validate()?;
        Ok(f)
    }

    /// Input numerics in original units; categoricals use integer level indices.
    /// Row identities must be stable and unique within this fit role.
    pub fn from_columns(
        columns: Vec<Vec<f64>>,
        categories: Vec<Option<usize>>,
        y: Vec<f64>,
        head: Head,
        ids: Vec<u64>,
        groups: Option<Vec<u64>>,
    ) -> Result<Self> {
        let f = Self {
            columns,
            categories,
            y,
            head,
            ids,
            groups,
            kind: SignatureKind::Real,
        };
        f.validate()?;
        Ok(f)
    }

    pub fn synthetic(program: &Program, samples: &Samples) -> Result<Self> {
        let data = ModelData {
            x: samples.x.clone(),
            y: samples.y.clone(),
        };
        let mut f = Self::real(
            &program.schema,
            &data,
            &(0..data.rows()).collect::<Vec<_>>(),
            None,
        )?;
        f.kind = SignatureKind::Synthetic;
        Ok(f)
    }

    pub fn rows(&self) -> usize {
        self.y.len()
    }
    pub fn features(&self) -> usize {
        self.columns.len()
    }

    pub(crate) fn validate(&self) -> Result<()> {
        let n = self.y.len();
        let classes = match self.head {
            Head::Regression => 0,
            Head::Binary => 2,
            Head::Multiclass(k) => k,
        };
        if n < 20
            || self.columns.is_empty()
            || self.columns.len() != self.categories.len()
            || self
                .columns
                .iter()
                .any(|c| c.len() != n || c.iter().any(|v| v.is_infinite()))
            || self.ids.len() != n
            || self.ids.iter().collect::<BTreeSet<_>>().len() != n
            || self.groups.as_ref().is_some_and(|g| g.len() != n)
            || self.y.iter().any(|y| {
                !y.is_finite()
                    || (classes > 0 && (*y < 0.0 || y.fract() != 0.0 || *y >= classes as f64))
            })
            || classes > 256
            || matches!(self.head, Head::Multiclass(k) if k < 2)
        {
            return Err(DopeError::Input("invalid landmark frame".into()));
        }
        for (c, k) in self.columns.iter().zip(&self.categories) {
            if let Some(k) = k {
                if *k == 0
                    || *k > 257
                    || c.iter()
                        .any(|v| v.is_finite() && (*v < 0.0 || v.fract() != 0.0 || *v >= *k as f64))
                {
                    return Err(DopeError::Input(
                        "invalid landmark categorical codes".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

pub(crate) struct Encoded {
    pub x: Matrix,
    pub numeric: Vec<usize>,
    pub missing: Vec<usize>,
    pub groups: Vec<usize>,
}

enum Column {
    Numeric(f64, f64),
    Categorical(Vec<usize>),
}

/// Fold-local robust preprocessing. No target-dependent category ordering.
pub(crate) struct Representation {
    columns: Vec<Column>,
    missing: Vec<bool>,
}
impl Representation {
    pub fn fit(frame: &LandmarkFrame, rows: &[usize]) -> Self {
        let columns = frame
            .columns
            .iter()
            .zip(&frame.categories)
            .map(|(c, k)| {
                if let Some(k) = k {
                    let mut counts = vec![0usize; *k];
                    for &r in rows {
                        if c[r].is_finite() {
                            counts[c[r] as usize] += 1;
                        }
                    }
                    Column::Categorical((0..*k).filter(|&k| counts[k] >= 5).collect())
                } else {
                    let mut v: Vec<f64> = rows
                        .iter()
                        .map(|&r| c[r])
                        .filter(|v| v.is_finite())
                        .collect();
                    v.sort_by(f64::total_cmp);
                    if v.is_empty() {
                        return Column::Numeric(0.0, 1.0);
                    }
                    let median = quantile_sorted(&v, 0.5);
                    let mut scale = (quantile_sorted(&v, 0.75) - quantile_sorted(&v, 0.25)) / 1.349;
                    if scale <= 1e-12 {
                        let m = mean(&v);
                        scale = (v.iter().map(|v| (v - m).powi(2)).sum::<f64>() / v.len() as f64)
                            .sqrt();
                    }
                    if scale <= 1e-12 {
                        scale = 1.0;
                    }
                    Column::Numeric(median, scale)
                }
            })
            .collect();
        Self {
            columns,
            missing: frame
                .columns
                .iter()
                .map(|c| rows.iter().any(|&r| !c[r].is_finite()))
                .collect(),
        }
    }

    pub fn encode(&self, frame: &LandmarkFrame, rows: &[usize]) -> Result<Encoded> {
        let mut numeric = Vec::new();
        let mut missing = Vec::new();
        let mut groups = Vec::new();
        for (j, c) in self.columns.iter().enumerate() {
            match c {
                Column::Numeric(..) => {
                    numeric.push(groups.len());
                    groups.push(j);
                }
                Column::Categorical(levels) => {
                    groups.extend(std::iter::repeat_n(j, levels.len() + 1))
                }
            }
            if self.missing[j] {
                missing.push(groups.len());
                groups.push(j);
            }
        }
        if groups.len() > 512 {
            return Err(DopeError::Budget(
                "landmark representation exceeds 512 columns".into(),
            ));
        }
        let mut data = Vec::with_capacity(rows.len() * groups.len());
        for &r in rows {
            for (j, c) in self.columns.iter().enumerate() {
                let v = frame.columns[j][r];
                match c {
                    Column::Numeric(m, s) => data.push(if v.is_finite() {
                        ((v - m) / s).clamp(-5.0, 5.0)
                    } else {
                        0.0
                    }),
                    Column::Categorical(levels) => {
                        data.extend(levels.iter().map(|&l| (v == l as f64) as u8 as f64));
                        data.push((v.is_finite() && !levels.contains(&(v as usize))) as u8 as f64);
                    }
                }
                if self.missing[j] {
                    data.push((!v.is_finite()) as u8 as f64);
                }
            }
        }
        Ok(Encoded {
            x: Matrix {
                rows: rows.len(),
                cols: groups.len(),
                data,
            },
            numeric,
            missing,
            groups,
        })
    }
}
