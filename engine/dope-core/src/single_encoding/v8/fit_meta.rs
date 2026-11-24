//! FIT-only column meta for DSE8, in the packet's own unit space.
//!
//! One function shared by every DSE8 fit path (the training episode stage in `dope-ffi`, the
//! `dope_v8_emit fit --fit-meta` KPI arm, the product fit), so the packets a model is trained
//! on and the packets it is scored and served on are built from the same meta.
//!
//! Unit-free fields (kinds, support class, bounds, tail asymmetry, unique bucket) decide the
//! scale card; the lattice step and phase are then refitted in that card's unit space (center,
//! scale), because DSE8 marginals use `lattice_q0_unit` directly as the lattice step. Columns
//! outside the probe card (only possible above `V8_MAX_COLS` columns) keep the meta of their own
//! robust unit space. Nothing here reads anything but the rows passed in, and no meta field
//! holds a source value or an extremum.

use super::scale_v8::{Dse8ColumnMeta, ScaleCardV8, V8_MAX_CLASSES, V8_MAX_COLS};
use crate::error::Result;
use crate::evidence::{
    extract_with_rows, ColumnMeta as EvColMeta, Evidence, EvidenceInput, EvidenceTier, RowIdentity,
};
use crate::marginal::column_meta::{fit_columns, ColumnDecl, Nc2ColumnMetaV2, UnitSpace};

impl From<&Nc2ColumnMetaV2> for Dse8ColumnMeta {
    fn from(m: &Nc2ColumnMetaV2) -> Self {
        Dse8ColumnMeta {
            group_id: m.group_id,
            group_kind: m.group_kind,
            group_width: m.group_width,
            parent_col_idx: m.parent_col_idx,
            support_class: m.support_class,
            skew_guard_active: m.skew_guard_active,
            lattice_q0_unit: m.lattice_q0_unit,
            lattice_origin_unit: m.lattice_origin_unit,
            has_lower_bound: m.has_lower_bound,
            has_upper_bound: m.has_upper_bound,
            log_pre_tail_asym: m.log_pre_tail_asym,
            log_unique_bucket: m.log_unique_bucket,
        }
    }
}

/// `Nc2ColumnMetaV2` of every raw column (`x_raw[col][row]`), with each card column's lattice
/// fields in the unit space of the probe card `fit_dse8_packet` builds from the same rows.
pub fn fit_unit_meta_v2(
    x_raw: &[Vec<f64>],
    y_raw: &[f64],
    task_id: u8,
    normalized_only: bool,
) -> Result<Vec<Nc2ColumnMetaV2>> {
    let mut meta = fit_columns(x_raw, None, None);
    let unit_free: Vec<Dse8ColumnMeta> = meta.iter().map(Dse8ColumnMeta::from).collect();
    let p_probe = V8_MAX_COLS.min(x_raw.len());
    let probe = ScaleCardV8::fit(
        x_raw,
        y_raw,
        task_id,
        p_probe,
        normalized_only,
        Some(&unit_free),
    )?;
    for (c, &j) in probe.col_indices.iter().enumerate() {
        let unit = UnitSpace {
            center: probe.center_active[c] as f64,
            scale: probe.scale_active[c] as f64,
        };
        meta[j] = Nc2ColumnMetaV2::fit(&x_raw[j], ColumnDecl::single(j), Some(unit));
    }
    Ok(meta)
}

/// `fit_unit_meta_v2` as the `Dse8FitOptions::col_meta` slice.
pub fn fit_unit_meta(
    x_raw: &[Vec<f64>],
    y_raw: &[f64],
    task_id: u8,
    normalized_only: bool,
) -> Result<Vec<Dse8ColumnMeta>> {
    Ok(fit_unit_meta_v2(x_raw, y_raw, task_id, normalized_only)?
        .iter()
        .map(Dse8ColumnMeta::from)
        .collect())
}

/// Shared FIT-only evidence extraction used by both serving (`fit_dse8_packet_with_encoder`)
/// and training (`dope-ffi`).
///
/// When `include_raw_companion` is `false` (historical checkpoints without `active-rank-v1`),
/// passes `x_raw: None` and `rows: None` to keep prior fit/decode behavior byte-identical.
/// When `include_raw_companion` is `true` (`active-rank-v1` serving and `dope-ffi`), passes
/// the active raw FIT columns and optional stable physical `fit_row_ids` sliced to `valid_rows`.
#[allow(clippy::too_many_arguments)]
pub fn extract_fit_evidence_v8(
    x_raw: &[Vec<f64>],
    x_unit_valid: &[Vec<f32>],
    y_sc: &[f64],
    valid_rows: &[usize],
    card: &ScaleCardV8,
    active_meta: Option<&[Dse8ColumnMeta]>,
    tier: EvidenceTier,
    seed: u64,
    fit_row_ids: Option<&[u64]>,
    include_raw_companion: bool,
) -> Result<Evidence> {
    let ev_meta: Option<Vec<EvColMeta>> =
        active_meta.map(|ms| ms.iter().map(EvColMeta::from).collect());
    let ev_raw: Option<Vec<Vec<f64>>> = include_raw_companion.then(|| {
        card.col_indices
            .iter()
            .map(|&j| valid_rows.iter().map(|&i| x_raw[j][i]).collect())
            .collect()
    });
    let ev_ids: Option<Vec<u64>> = if include_raw_companion {
        fit_row_ids.map(|ids| valid_rows.iter().map(|&i| ids[i]).collect())
    } else {
        None
    };
    let rows = ev_ids.as_deref().map(|row_ids| RowIdentity {
        row_ids,
        group_ids: None,
    });
    let n_classes = match card.task_id {
        0 => 1,
        1 => 2,
        _ => card.classes.len().clamp(2, V8_MAX_CLASSES),
    };
    extract_with_rows(
        &EvidenceInput {
            x_unit: x_unit_valid,
            x_raw: ev_raw.as_deref(),
            y: y_sc,
            task_id: card.task_id,
            n_classes,
            col_meta: ev_meta.as_deref(),
            packet_probe: None,
            tier,
            target_free: false,
            seed,
        },
        rows,
    )
}
