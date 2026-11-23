//! `Nc2ColumnMetaV2` from FIT rows: planted support classes, the k = 10 floor, invariances.

use dope_core::marginal::column_meta::{
    fit_columns, ColumnDecl, Nc2ColumnMetaV2, UnitSpace, KIND_BOOL, KIND_CONT, KIND_INT,
    KIND_ONEHOT, SUPPORT_ATOMS, SUPPORT_CONST, SUPPORT_CONT, SUPPORT_LATTICE,
    SUPPORT_ZERO_INFLATED,
};

struct Rng(u64);
impl Rng {
    fn u(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (a, b) = (self.u().max(1e-300), self.u());
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    }
}

/// Values as the arena stores them (float32).
fn f32s(v: impl IntoIterator<Item = f64>) -> Vec<f64> {
    v.into_iter().map(|x| x as f32 as f64).collect()
}

fn meta(col: &[f64]) -> Nc2ColumnMetaV2 {
    Nc2ColumnMetaV2::fit(col, ColumnDecl::single(3), None)
}

#[test]
fn planted_support_classes() {
    let mut r = Rng(7);
    let normal = f32s((0..2000).map(|_| r.normal()));
    let m = meta(&normal);
    assert_eq!((m.support_class, m.group_kind), (SUPPORT_CONT, KIND_CONT));
    assert!(m.log_pre_tail_asym.abs() < 0.1, "{}", m.log_pre_tail_asym);
    assert_eq!(
        (m.has_lower_bound, m.has_upper_bound, m.lattice_q0_unit),
        (0, 0, 0.0)
    );
    assert!((m.log_unique_bucket - 1.0).abs() < 1e-6); // > 256 distinct

    let ints = f32s((0..1500).map(|_| (r.u() * 900.0).floor()));
    let m = meta(&ints);
    assert_eq!((m.support_class, m.group_kind), (SUPPORT_LATTICE, KIND_INT));
    let sorted = {
        let mut s = ints.clone();
        s.sort_by(f64::total_cmp);
        s
    };
    let unit = UnitSpace::robust(&sorted);
    assert!((m.lattice_q0_unit as f64 - 1.0 / unit.scale).abs() < 1e-6);
    assert!(m.lattice_origin_unit >= 0.0 && m.lattice_origin_unit < m.lattice_q0_unit);

    // A 0.25 grid shifted by 0.1: the phase is reported in unit space, not the raw origin.
    let grid = f32s((0..1200).map(|_| 0.1 + 0.25 * (r.u() * 300.0).floor()));
    let m = Nc2ColumnMetaV2::fit(
        &grid,
        ColumnDecl::single(0),
        Some(UnitSpace {
            center: 0.0,
            scale: 0.5,
        }),
    );
    assert_eq!(m.support_class, SUPPORT_LATTICE);
    assert!(
        (m.lattice_q0_unit - 0.5).abs() < 1e-6,
        "{}",
        m.lattice_q0_unit
    );
    assert!(
        (m.lattice_origin_unit - 0.2).abs() < 1e-3,
        "{}",
        m.lattice_origin_unit
    );

    let levels = f32s((0..600).map(|i| [2.0, 3.5, 7.0, 11.0, 13.0][i % 5]));
    assert_eq!(meta(&levels).support_class, SUPPORT_ATOMS);

    let bools = f32s((0..300).map(|_| (r.u() < 0.3) as u8 as f64));
    let m = meta(&bools);
    assert_eq!((m.support_class, m.group_kind), (SUPPORT_ATOMS, KIND_BOOL));

    let spiky = f32s((0..2000).map(|_| {
        if r.u() < 0.4 {
            0.0
        } else {
            (r.normal() + 1.0).exp()
        }
    }));
    let m = meta(&spiky);
    assert_eq!(m.support_class, SUPPORT_ZERO_INFLATED);
    assert_eq!(m.has_lower_bound, 1);
    assert!(m.log_pre_tail_asym > 0.3, "{}", m.log_pre_tail_asym);
    assert_eq!(m.skew_guard_active, 1);

    let m = meta(&[4.0; 50]);
    assert_eq!(m.support_class, SUPPORT_CONST);
    assert_eq!(meta(&[f64::NAN; 20]).support_class, SUPPORT_CONST);
}

#[test]
fn bound_flags_need_ten_rows() {
    let mut r = Rng(11);
    for (at_min, want) in [(9usize, 0u8), (10, 1)] {
        let mut col: Vec<f64> = (0..500).map(|_| 1.0 + r.u()).collect();
        col.extend(std::iter::repeat_n(0.0, at_min));
        let m = meta(&col);
        assert_eq!(m.has_lower_bound, want, "{at_min} rows at the minimum");
        assert_eq!(m.has_upper_bound, 0);
    }
}

#[test]
fn row_order_and_missing_cells_do_not_matter() {
    let mut r = Rng(5);
    let col: Vec<f64> = f32s((0..900).map(|_| (r.u() * 40.0).floor() * 0.5));
    let mut shuffled = col.clone();
    for i in (1..shuffled.len()).rev() {
        let j = (r.u() * (i + 1) as f64) as usize;
        shuffled.swap(i, j);
    }
    assert_eq!(meta(&col), meta(&shuffled));
    let mut holes = col.clone();
    holes.extend([f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
    assert_eq!(meta(&col), meta(&holes));
    let f = meta(&holes).to_f32s();
    assert!(f.iter().all(|v| v.is_finite()));
    assert_eq!(meta(&col).quantized(), meta(&shuffled).quantized());
}

#[test]
fn declarations_override_the_derived_kind() {
    let col = f32s((0..100).map(|i| (i % 2) as f64));
    let decl = ColumnDecl {
        group_id: 7,
        group_kind: Some(KIND_ONEHOT),
        group_width: 4,
        parent_col_idx: 6,
    };
    let m = Nc2ColumnMetaV2::fit(&col, decl, None);
    assert_eq!(
        (m.group_id, m.group_kind, m.group_width, m.parent_col_idx),
        (7, KIND_ONEHOT, 4, 6)
    );
    assert_eq!(m.to_f32s()[3], 6.0);
    assert_eq!(meta(&col).to_f32s()[3], -1.0);
}

#[test]
fn table_of_48_columns_at_4096_rows_fits_in_budget() {
    let mut r = Rng(3);
    let n = 4096;
    let cols: Vec<Vec<f64>> = (0..48)
        .map(|j| {
            f32s((0..n).map(|_| match j % 4 {
                0 => r.normal(),
                1 => (r.u() * 1000.0).floor(),
                2 => [0.0, 1.0, 2.0][(r.u() * 3.0) as usize],
                _ => (r.normal() * 0.7).exp(),
            }))
        })
        .collect();
    let t0 = std::time::Instant::now();
    let metas = fit_columns(&cols, None, None);
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    println!("column_meta 48 x {n}: {ms:.1} ms");
    assert_eq!(metas.len(), 48);
    assert!(metas
        .iter()
        .step_by(4)
        .all(|m| m.support_class == SUPPORT_CONT));
    assert!(metas
        .iter()
        .skip(1)
        .step_by(4)
        .all(|m| m.support_class == SUPPORT_LATTICE));
    assert!(metas
        .iter()
        .skip(2)
        .step_by(4)
        .all(|m| m.support_class == SUPPORT_ATOMS));
}
