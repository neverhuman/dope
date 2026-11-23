//! Pure-Rust FIT-only Rich Evidence Bank (`dope_core::evidence`, frozen by `contracts/evidence-layout-v1.json`).
//!
//! Pure function of `(FIT rows, schema, tier, seed)`:
//! - Single-threaded, zero global state, zero per-table caching, deterministic.
//! - Structurally FIT-only (`EvidenceInput` has no query/test fields -> query-poison-safe).
//! - Symmetric X-fingerprint row canonicalization guarantees exact row-permutation invariance
//!   and column-permutation equivariance.
//! - Enforces `k >= 10` support floor everywhere and never echoes raw values/headers/extrema.

pub mod closed_loop;
pub mod dependence_pairs;
pub mod geometry_privacy;
pub mod irls;
pub mod primitives;
pub mod rank_stats;
pub mod support_marginal;
pub mod target_helpers;
pub mod target_interactions;
pub mod witness;

use crate::error::{DopeError, Result};
use closed_loop::extract_closed_loop_t3;
use dependence_pairs::extract_dependence_t1;
use primitives::{
    order_by_fingerprint, row_fingerprints, set_mask_bit, tie_aware_normal_scores,
    value_consistent_bins, Folds,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use support_marginal::extract_column_t0;
use target_interactions::extract_target_t1;
use witness::extract_witness_t2;

pub const COL_DIM: usize = 128;
pub const EDGE_DIM: usize = 16;
pub const CLASS_DIM: usize = 16;
pub const CLASS_COL_DIM: usize = 4;
pub const GROUP_DIM: usize = 8;
pub const GLOB_DIM: usize = 160;
pub const MAX_CLASSES_EVIDENCE: usize = 8;
pub const MAX_COLS_EVIDENCE: usize = 256;
pub const LAYOUT_JSON: &[u8] = include_bytes!("../../../../contracts/evidence-layout-v1.json");

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EvidenceTier {
    T0 = 0,
    T1 = 1,
    T2 = 2,
    T3 = 3,
}

/// Privacy-safe column structural metadata (aligned 1:1 with `Nc2ColumnMetaV2` / `Dse8ColumnMeta`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColumnMeta {
    pub group_id: u16,
    /// 0=numeric_cont, 1=integer, 2=ordinal, 3=boolean, 4=nominal_onehot, 5=missing_ind
    pub group_kind: u8,
    pub group_width: u16,
    pub parent_col_idx: u16,
    /// 0=cont, 1=lattice, 2=atoms, 3=zero_inflated, 4=const
    pub support_class: u8,
    pub skew_guard_active: u8,
    pub lattice_q0_unit: f32,
    pub lattice_origin_unit: f32,
    pub has_lower_bound: u8,
    pub has_upper_bound: u8,
    pub log_pre_tail_asym: f32,
    pub log_unique_bucket: f32,
}

impl Default for ColumnMeta {
    fn default() -> Self {
        Self {
            group_id: 0,
            group_kind: 0,
            group_width: 1,
            parent_col_idx: u16::MAX,
            support_class: 0,
            skew_guard_active: 0,
            lattice_q0_unit: 0.0,
            lattice_origin_unit: 0.0,
            has_lower_bound: 0,
            has_upper_bound: 0,
            log_pre_tail_asym: 0.0,
            log_unique_bucket: 0.0,
        }
    }
}

/// Ephemeral candidate decoded packet probe supplied on the owner side for `M07` & Tier `T3`.
/// Never cached across tables or splits.
#[derive(Clone, Debug)]
pub struct PacketProbe {
    pub x_syn_unit: Vec<Vec<f32>>,
    pub packet_bytes: usize,
    pub probe_id: u64,
}

/// Strictly FIT-only input contract (impossible to supply query/test rows or labels).
pub struct EvidenceInput<'a> {
    pub x_unit: &'a [Vec<f32>],
    pub x_raw: Option<&'a [Vec<f64>]>,
    pub y: &'a [f64],
    pub task_id: u8,
    pub n_classes: usize,
    pub col_meta: Option<&'a [ColumnMeta]>,
    pub packet_probe: Option<&'a PacketProbe>,
    pub tier: EvidenceTier,
    pub target_free: bool,
    pub seed: u64,
}

#[derive(Clone, Debug)]
pub struct Evidence {
    pub p: usize,
    pub k_eff: usize,
    pub col: Vec<[f32; COL_DIM]>,
    pub col_mask: Vec<[u8; 16]>,
    pub edge: Vec<[f32; EDGE_DIM]>,
    pub edge_mask: Vec<u16>,
    pub class_tok: Vec<[f32; CLASS_DIM]>,
    pub class_col: Vec<[f32; CLASS_COL_DIM]>,
    pub group_tok: Vec<[f32; GROUP_DIM]>,
    pub glob: [f32; GLOB_DIM],
    pub glob_mask: [u8; 20],
    pub col_composite_signal: Vec<f64>,
    pub layout_sha256: [u8; 32],
}

pub fn evidence_layout_sha256() -> [u8; 32] {
    Sha256::digest(LAYOUT_JSON).into()
}

/// Stable identity of the FIT rows, one entry per `x_unit` row.
#[derive(Clone, Copy, Debug)]
pub struct RowIdentity<'a> {
    /// Distinct physical row ids (e.g. source-table row indices). They break ties between rows
    /// with equal X fingerprints, so the canonical order follows the physical rows.
    pub row_ids: &'a [u64],
    /// Duplicate-group ids: every row of a group goes to one cross-fitting fold. `None` groups
    /// rows by their X fingerprint.
    pub group_ids: Option<&'a [u64]>,
}

/// Pure-function FIT-only evidence extractor up to `input.tier`.
/// Zero global state, zero caching, single-threaded deterministic, query-poison-safe.
/// Rows tie-break by input position and folds are the historical `r % k` split; see
/// `extract_with_rows` for a result that is free of the input row order.
pub fn extract(input: &EvidenceInput<'_>) -> Result<Evidence> {
    extract_with_rows(input, None)
}

/// `extract` with stable row identity: the same evidence for any order of the same physical
/// rows, and duplicate-group cross-fitting folds. `rows: None` is exactly `extract`.
pub fn extract_with_rows(
    input: &EvidenceInput<'_>,
    rows: Option<RowIdentity<'_>>,
) -> Result<Evidence> {
    let p = input.x_unit.len();
    if p == 0 {
        return Err(DopeError::Input("evidence_empty_columns".into()));
    }
    if p > MAX_COLS_EVIDENCE {
        return Err(DopeError::Input("evidence_col_limit_exceeded".into()));
    }
    if input.task_id > 2 {
        return Err(DopeError::Input("evidence_invalid_task_id".into()));
    }
    if input.task_id == 2 && (input.n_classes < 2 || input.n_classes > MAX_CLASSES_EVIDENCE) {
        return Err(DopeError::Input("evidence_class_limit_exceeded".into()));
    }
    let n = input.x_unit[0].len();
    if n < 4 || input.x_unit.iter().any(|c| c.len() != n) {
        return Err(DopeError::Input("evidence_row_shape_mismatch".into()));
    }
    if !input.target_free && input.y.len() != n {
        return Err(DopeError::Input("evidence_target_len_mismatch".into()));
    }
    if let Some(raw) = input.x_raw {
        if raw.len() != p || raw.iter().any(|c| c.len() != n) {
            return Err(DopeError::Input("evidence_raw_shape_mismatch".into()));
        }
    }
    if let Some(meta) = input.col_meta {
        if meta.len() != p {
            return Err(DopeError::Input("evidence_meta_len_mismatch".into()));
        }
    }

    if let Some(id) = rows {
        let mut ids = id.row_ids.to_vec();
        ids.sort_unstable();
        if ids.len() != n
            || ids.windows(2).any(|w| w[0] == w[1])
            || id.group_ids.is_some_and(|g| g.len() != n)
        {
            return Err(DopeError::Input("evidence_row_identity_mismatch".into()));
        }
    }

    // Canonicalize FIT row order using column-permutation-invariant X fingerprint.
    let fp = row_fingerprints(input.x_unit);
    let order = order_by_fingerprint(&fp, rows.map(|id| id.row_ids));
    let folds = rows.map_or_else(Folds::default, |id| {
        let key = |r: usize| {
            id.group_ids
                .map_or(fp[r].0 ^ fp[r].1.rotate_left(29), |g| g[r])
        };
        Folds::grouped(&order.iter().map(|&r| key(r)).collect::<Vec<_>>())
    });
    let x_unit_c: Vec<Vec<f32>> = (0..p)
        .map(|j| order.iter().map(|&r| input.x_unit[j][r]).collect())
        .collect();
    let x_raw_c: Option<Vec<Vec<f64>>> = input.x_raw.map(|raw| {
        (0..p)
            .map(|j| order.iter().map(|&r| raw[j][r]).collect())
            .collect()
    });
    let y_c: Vec<f64> = if input.target_free {
        Vec::new()
    } else {
        order.iter().map(|&r| input.y[r]).collect()
    };

    let k_eff = match input.task_id {
        0 => 1,
        1 => 2,
        _ => input.n_classes.clamp(2, MAX_CLASSES_EVIDENCE),
    };

    let mut col = vec![[0.0f32; COL_DIM]; p];
    let mut col_mask = vec![[0u8; 16]; p];
    let mut edge = vec![[0.0f32; EDGE_DIM]; p * p];
    let mut edge_mask = vec![0u16; p * p];
    let mut class_tok = vec![[0.0f32; CLASS_DIM]; k_eff];
    let mut class_col = vec![[0.0f32; CLASS_COL_DIM]; k_eff * p];
    let mut glob = [0.0f32; GLOB_DIM];
    let mut glob_mask = [0u8; 20];

    let mut t0_summaries = Vec::with_capacity(p);
    let mut n_const = 0usize;
    let mut n_disc = 0usize;
    let mut sum_miss = 0.0;
    for j in 0..p {
        let raw_j = x_raw_c.as_ref().map(|r| r[j].as_slice());
        let meta_j = input.col_meta.map(|m| &m[j]);
        let s = extract_column_t0(
            &x_unit_c[j],
            raw_j,
            meta_j,
            input.packet_probe,
            j,
            &mut col[j],
            &mut col_mask[j],
        );
        if s.is_const {
            n_const += 1;
        }
        if s.is_disc {
            n_disc += 1;
        }
        sum_miss += s.miss_rate;
        t0_summaries.push(s);
    }

    let group_tok = build_group_tokens(p, input.col_meta, &t0_summaries);

    glob[3] = if input.target_free { 1.0 } else { 0.0 };
    glob[12] = ((n.clamp(16, 16384) as f64).log2() / 14.0) as f32;
    glob[13] = (p as f64 / 48.0).clamp(0.0, 2.0) as f32;
    glob[14] = ((p as f64 / n.max(1) as f64).log2().clamp(-12.0, 4.0) / 12.0) as f32;
    glob[54] = (sum_miss / p as f64) as f32;
    glob[55] = (n_const as f64 / p as f64) as f32;
    glob[56] = (n_disc as f64 / p as f64) as f32;
    for s in [3, 12, 13, 14, 54, 55, 56] {
        set_mask_bit(&mut glob_mask, s);
    }

    let mut composite_signal = vec![0.0f64; p];

    if input.tier >= EvidenceTier::T1 {
        let mut z_tie = Vec::with_capacity(p);
        let mut kappa = Vec::with_capacity(p);
        let mut bins16 = Vec::with_capacity(p);
        for col_u in x_unit_c.iter().take(p) {
            let xj: Vec<f64> = col_u.iter().map(|&v| v as f64).collect();
            let (z, k, _) = tie_aware_normal_scores(&xj);
            let (b, _) = value_consistent_bins(&xj, 16);
            z_tie.push(z);
            kappa.push(k);
            bins16.push(b);
        }

        let dep = extract_dependence_t1(
            &x_unit_c,
            &z_tie,
            &kappa,
            &bins16,
            &t0_summaries,
            input.col_meta,
            input.seed,
            &mut col,
            &mut col_mask,
            &mut edge,
            &mut edge_mask,
            &mut glob,
            &mut glob_mask,
        );

        if !input.target_free {
            let tgt = extract_target_t1(
                &x_unit_c,
                &folds,
                &z_tie,
                &bins16,
                &y_c,
                input.task_id,
                k_eff,
                &t0_summaries,
                &dep,
                &mut col,
                &mut col_mask,
                &mut edge,
                &mut edge_mask,
                &mut class_tok,
                &mut class_col,
                &mut glob,
                &mut glob_mask,
            );
            composite_signal = tgt.composite_signal.clone();

            if input.tier >= EvidenceTier::T2 {
                extract_witness_t2(
                    &bins16,
                    &z_tie,
                    &y_c,
                    input.task_id,
                    k_eff,
                    input.col_meta,
                    &tgt,
                    &mut col,
                    &mut col_mask,
                    &mut edge,
                    &mut edge_mask,
                    &mut glob,
                    &mut glob_mask,
                );
            }
        }
    }

    if input.tier >= EvidenceTier::T3 {
        if let Some(probe) = input.packet_probe {
            extract_closed_loop_t3(
                &x_unit_c,
                probe,
                input.seed,
                &mut col,
                &mut col_mask,
                &mut glob,
                &mut glob_mask,
            );
        }
    }

    sanitize_finite(
        &mut col,
        &mut edge,
        &mut class_tok,
        &mut class_col,
        &mut glob,
    );

    Ok(Evidence {
        p,
        k_eff,
        col,
        col_mask,
        edge,
        edge_mask,
        class_tok,
        class_col,
        group_tok,
        glob,
        glob_mask,
        col_composite_signal: composite_signal,
        layout_sha256: evidence_layout_sha256(),
    })
}

fn build_group_tokens(
    p: usize,
    meta: Option<&[ColumnMeta]>,
    t0: &[support_marginal::ColT0Summary],
) -> Vec<[f32; GROUP_DIM]> {
    let Some(ms) = meta else {
        return (0..p)
            .map(|j| {
                let mut g = [0.0f32; GROUP_DIM];
                g[0] = 1.0;
                g[6] = if t0[j].miss_rate > 0.0 { 1.0 } else { 0.0 };
                g[7] = (1.0 / p.max(1) as f32).clamp(0.0, 1.0);
                g
            })
            .collect();
    };
    let mut seen = std::collections::BTreeMap::<u16, (u8, u16, bool)>::new();
    for (j, m) in ms.iter().enumerate() {
        let entry = seen
            .entry(m.group_id)
            .or_insert((m.group_kind, m.group_width.max(1), false));
        if t0[j].miss_rate > 0.0 || m.group_kind == 5 {
            entry.2 = true;
        }
    }
    seen.values()
        .map(|&(kind, width, has_miss)| {
            let mut g = [0.0f32; GROUP_DIM];
            g[(kind as usize).min(4)] = 1.0;
            g[5] = ((width as f64).log2() / 4.0) as f32;
            g[6] = if has_miss { 1.0 } else { 0.0 };
            g[7] = (width as f32 / p.max(1) as f32).clamp(0.0, 1.0);
            g
        })
        .collect()
}

fn sanitize_finite(
    col: &mut [[f32; COL_DIM]],
    edge: &mut [[f32; EDGE_DIM]],
    cls: &mut [[f32; CLASS_DIM]],
    cc: &mut [[f32; CLASS_COL_DIM]],
    glob: &mut [f32; GLOB_DIM],
) {
    let clean = |v: &mut f32| {
        if !v.is_finite() {
            *v = 0.0;
        } else {
            *v = v.clamp(-4.0, 4.0);
        }
    };
    for r in col {
        for v in r.iter_mut() {
            clean(v);
        }
    }
    for r in edge {
        for v in r.iter_mut() {
            clean(v);
        }
    }
    for r in cls {
        for v in r.iter_mut() {
            clean(v);
        }
    }
    for r in cc {
        for v in r.iter_mut() {
            clean(v);
        }
    }
    for v in glob.iter_mut() {
        clean(v);
    }
}
