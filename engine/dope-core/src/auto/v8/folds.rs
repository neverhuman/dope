use super::{
    admit_neural_packet, decoded_v8_to_raw, raw_to_numeric_inputs, MIN_GAIN, TEST_CAP, TRAIN_CAP,
};
use crate::auto::auditors::{fit_predict, null_predictions, Featurizer, AUDITORS};
use crate::auto::check::{retention, CHECK_SYNTH_FACTOR};
use crate::auto::v7_schema::{encode_section, V7Schema};
use crate::compile::{fit as fit_f1, FitOptions};
use crate::error::{DopeError, Result};
use crate::eval::{prediction_loss, Matrix};
use crate::program::Program;
use crate::sample::{sample as sample_f1, SampleOptions};
use crate::schema::{infer, IntakeOptions, Schema, Task};
use crate::single_encoding::v8::{
    decode_dse8_rows, fit_dse8_packet, fit_dse8_packet_with_encoder, fit_unit_meta,
    unpack_dse8_packet, Dse8FitOptions, SingleEncV8Weights,
};
use crate::table::RawTable;

#[allow(clippy::too_many_arguments)]
pub(super) fn score_three_way(
    raw: &RawTable,
    target: usize,
    whole_schema: &Schema,
    train_idx: &[usize],
    test_idx: &[usize],
    f1_opts: &FitOptions,
    dse8_opts: &Dse8FitOptions<'_>,
    weights: Option<&SingleEncV8Weights>,
    dpa_payload_room: Option<usize>,
    seed: u64,
) -> Result<(Option<f64>, Option<f64>, Option<f64>)> {
    let train_tbl = raw.select_rows(train_idx);
    let test_tbl = raw.select_rows(test_idx);
    let intake = IntakeOptions {
        task: Some(whole_schema.task),
        ..Default::default()
    };
    let schema = infer(&train_tbl, target, &intake)?;
    let real = schema.to_model(&train_tbl)?;
    let featurizer = Featurizer::fit(&schema, &real);
    let head = Program::head_for(&schema);
    let (xr, yr) = featurizer.matrix(&real, TRAIN_CAP);
    let (xt, yt) = featurizer.matrix(&schema.to_model(&test_tbl)?, TEST_CAP);
    if yr.len() < 20 || yt.len() < 10 {
        return Ok((None, None, None));
    }
    // For classification, require >= 5 test rows per active class so a tiny 2-row minority
    // test fold cannot flip the Format 1 vs DSE8 choice on sampling noise!
    if schema.task == Task::Classification {
        let mut cnts = std::collections::BTreeMap::<i32, usize>::new();
        for &v in &yt {
            if v.is_finite() {
                *cnts.entry(v.round() as i32).or_default() += 1;
            }
        }
        if cnts.len() < 2 || cnts.values().any(|&c| c < 5) {
            return Ok((None, None, None));
        }
    }
    let n_syn = CHECK_SYNTH_FACTOR * train_tbl.rows();
    let as_mat = |tbl: Result<RawTable>| -> Option<(Matrix, Vec<f64>)> {
        let data = schema.to_model(&tbl.ok()?).ok()?;
        let (x, y) = featurizer.matrix(&data, TRAIN_CAP);
        (!y.is_empty()).then_some((x, y))
    };

    let syn_f1 = as_mat(crate::auto::guarded("f1_chk", || {
        let fitted = fit_f1(&train_tbl, target, f1_opts)?;
        let s = sample_f1(
            &fitted.program,
            &SampleOptions {
                rows: n_syn,
                seed: seed ^ 0xF101,
                ..Default::default()
            },
        )?;
        Ok(decoded_v8_to_raw(&schema, &s.x, &s.y))
    }));

    let (x_tr, y_tr, tr_tid, tr_row_ids) = raw_to_numeric_inputs(&train_tbl, target, &schema)?;
    let tr_meta = fit_unit_meta(&x_tr, &y_tr, tr_tid, dse8_opts.normalized_only).ok();
    let tr_dse8_opts = Dse8FitOptions {
        task_id: tr_tid,
        col_meta: tr_meta.as_deref(),
        fit_row_ids: weights
            .filter(|w| w.active_rank_v1)
            .map(|_| tr_row_ids.as_slice()),
        ..*dse8_opts
    };
    let syn_p0 = as_mat(crate::auto::guarded("p0_chk", || {
        let bytes = fit_dse8_packet(&x_tr, &y_tr, &tr_dse8_opts)?;
        let pkt = unpack_dse8_packet(&bytes)?;
        let dec = decode_dse8_rows(&pkt, None, n_syn, seed ^ 0xD5E8_0000)?;
        Ok(decoded_v8_to_raw(&schema, &dec.x, &dec.y))
    }));
    let syn_nn = weights.and_then(|w| {
        as_mat(crate::auto::guarded("nn_chk", || {
            let budget_bytes = if let Some(room) = dpa_payload_room {
                let section = encode_section(&V7Schema::from_schema(&schema), f1_opts.anonymize)?;
                room.checked_sub(section.len())
                    .ok_or_else(|| DopeError::Budget("auto_v8_fold_schema_exceeds_budget".into()))?
            } else {
                tr_dse8_opts.budget_bytes
            };
            let fold_opts = Dse8FitOptions {
                budget_bytes,
                ..tr_dse8_opts
            };
            let bytes = fit_dse8_packet_with_encoder(&x_tr, &y_tr, &fold_opts, w)?;
            let pkt = admit_neural_packet(&bytes, w.fingerprint())?;
            let dec = decode_dse8_rows(&pkt, Some(w), n_syn, seed ^ 0xD5E8_0001)?;
            Ok(decoded_v8_to_raw(&schema, &dec.x, &dec.y))
        }))
    });

    let l0 = prediction_loss(head, &null_predictions(head, &yr, yt.len()), &yt);
    let mut s_f1 = f64::INFINITY;
    let mut s_p0 = f64::INFINITY;
    let mut s_nn = f64::INFINITY;
    let mut any_usable = false;
    for name in AUDITORS {
        let s = seed ^ 0xA0D1_8888;
        let l_real = prediction_loss(head, &fit_predict(name, head, &xr, &yr, &xt, s), &yt);
        if !l_real.is_finite() || l0 - l_real <= MIN_GAIN {
            continue;
        }
        any_usable = true;
        let eval_syn = |syn: &Option<(Matrix, Vec<f64>)>| -> f64 {
            match syn {
                Some((xs, ys)) => {
                    let ls = prediction_loss(head, &fit_predict(name, head, xs, ys, &xt, s), &yt);
                    retention(l0, l_real, ls)
                }
                None => -1.0,
            }
        };
        s_f1 = s_f1.min(eval_syn(&syn_f1));
        s_p0 = s_p0.min(eval_syn(&syn_p0));
        if syn_nn.is_some() {
            s_nn = s_nn.min(eval_syn(&syn_nn));
        }
    }
    if !any_usable {
        return Ok((None, None, None));
    }
    Ok((Some(s_f1), Some(s_p0), syn_nn.map(|_| s_nn)))
}
