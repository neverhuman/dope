//! `fit --format auto` through the library API, on synthetic tables generated here.

use dope_core::auto::{
    fit_auto, parse_dpa1, wrap_dpa1, AutoOptions, Encoder, V7Decode, V7Decoder, V7Model,
    DPA1_HEADER_BYTES, NO_FINGERPRINT,
};
use dope_core::rng::Rng;
use dope_core::{decode, fit, sample, FitOptions, RawTable, SampleOptions};

fn model() -> V7Model {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/single-enc-v7/model.safetensors"
    );
    V7Model::from_safetensors(&std::fs::read(path).expect("repository V7 weights"), 4).unwrap()
}

fn table(headers: &[&str], rows: Vec<Vec<String>>) -> RawTable {
    let mut columns = vec![Vec::new(); headers.len()];
    for row in rows {
        for (c, v) in row.into_iter().enumerate() {
            columns[c].push(v);
        }
    }
    RawTable {
        headers: headers.iter().map(|h| h.to_string()).collect(),
        columns,
    }
}

/// `p` numeric features, target last; the target depends on the first five features.
fn wide(p: usize, n: usize, seed: u64) -> RawTable {
    let mut rng = Rng::new(seed);
    let names: Vec<String> = (0..p)
        .map(|j| format!("c{j}"))
        .chain(["target".into()])
        .collect();
    let rows = (0..n)
        .map(|_| {
            let x: Vec<f64> = (0..p).map(|_| rng.normal()).collect();
            let y = x[..5].iter().sum::<f64>() + 0.3 * rng.normal();
            x.iter().chain([&y]).map(|v| format!("{v:.4}")).collect()
        })
        .collect();
    table(&names.iter().map(String::as_str).collect::<Vec<_>>(), rows)
}

/// Distinctive column names and labels, so a report that quotes one is caught.
const MIXED_HEADERS: [&str; 4] = ["qq_age", "qq_city", "qq_label", "qq_spend"];
const MIXED_LABELS: [&str; 5] = ["vv_paris", "vv_lima", "vv_oslo", "vv_yes", "vv_no"];

/// Numeric, categorical and a string class label in the middle of the header.
fn mixed(n: usize, seed: u64) -> RawTable {
    let mut rng = Rng::new(seed);
    let rows = (0..n)
        .map(|_| {
            let age = 20.0 + (rng.uniform() * 50.0).floor();
            let city = rng.below(3);
            let spend = rng.normal();
            let z = 0.08 * (age - 45.0) + spend + if city == 2 { 1.0 } else { -0.5 };
            let label = if z + 0.5 * rng.normal() > 0.0 { 3 } else { 4 };
            vec![
                age.to_string(),
                MIXED_LABELS[city].into(),
                MIXED_LABELS[label].into(),
                format!("{spend:.3}"),
            ]
        })
        .collect();
    table(&MIXED_HEADERS, rows)
}

fn opts(budget: usize, seed: u64) -> AutoOptions {
    AutoOptions {
        budget,
        format1: FitOptions {
            seed,
            ..Default::default()
        },
        encoder: None,
    }
}

fn single_encoder(encoder: Encoder, seed: u64) -> AutoOptions {
    AutoOptions {
        encoder: Some(encoder),
        ..opts(2048, seed)
    }
}

#[test]
fn format1_fits_single_feature_tables_too_small_for_any_regime() {
    for classes in [false, true] {
        let mut rng = Rng::new(3);
        let rows = (0..40)
            .map(|_| {
                let a = rng.normal();
                let y = if classes {
                    (if a > 0.0 { "up" } else { "down" }).to_string()
                } else {
                    format!("{:.4}", 2.0 * a + rng.normal())
                };
                vec![format!("{a:.4}"), y]
            })
            .collect();
        let t = table(&["a", "y"], rows);
        let fitted = fit(
            &t,
            1,
            &FitOptions {
                budget: Some(2048),
                ..Default::default()
            },
        )
        .expect("one feature and forty rows fit");
        assert!(fitted.bytes.len() <= 2048);
        let program = decode(&fitted.bytes).unwrap();
        let s = sample(
            &program,
            &SampleOptions {
                rows: 25,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(s.y.len(), 25);
    }
}

#[test]
fn zero_feature_tables_are_refused_without_a_panic() {
    let only_target = table(
        &["y"],
        (0..120).map(|i| vec![format!("{}", i % 7)]).collect(),
    );
    let ids = table(
        &["id", "y"],
        (0..300)
            .map(|i| vec![format!("row-{i}"), format!("{}", i % 5)])
            .collect(),
    );
    let m = model();
    for t in [only_target, ids] {
        let target = t.width() - 1;
        let err = fit_auto(&t, target, &opts(2048, 0), &m)
            .err()
            .expect("refused");
        assert!(
            err.to_string().contains("no usable feature columns"),
            "{err}"
        );
    }
}

#[test]
fn wide_tables_choose_v7_within_the_budget_and_decode_target_last() {
    let t = wide(200, 300, 11);
    let m = model();
    let fitted = fit_auto(&t, 200, &opts(2048, 0), &m).unwrap();
    assert_eq!(fitted.encoder, Encoder::V7);
    assert!(fitted.bytes.len() <= 2048, "{} bytes", fitted.bytes.len());
    let r = &fitted.report;
    assert!(!r.format1.fitted && r.v7.fitted && !r.check.ran);
    assert!(r
        .format1
        .error
        .as_deref()
        .unwrap_or("")
        .starts_with("budget"));
    let dpa = parse_dpa1(&fitted.bytes).unwrap();
    assert_eq!(dpa.fingerprint, m.fingerprint());
    let decoder = V7Decoder::open(&dpa, &m).unwrap();
    let mut out = Vec::new();
    decoder
        .write_csv(V7Decode::Trained(&m), 64, 9, &mut out)
        .unwrap();
    let text = String::from_utf8(out).unwrap();
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    assert_eq!(header.len(), 201);
    assert_eq!(
        (header[0], header[199], header[200]),
        ("c0", "c199", "target")
    );
    let body: Vec<&str> = lines.collect();
    assert_eq!(body.len(), 64);
    for line in body {
        let cells: Vec<f64> = line.split(',').map(|c| c.parse().unwrap()).collect();
        assert_eq!(cells.len(), 201);
        assert!(cells.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn mixed_tables_run_the_internal_check_and_report_no_values() {
    let t = mixed(400, 5);
    let m = model();
    let fitted = fit_auto(&t, 2, &opts(2048, 1), &m).unwrap();
    let r = &fitted.report;
    assert!(r.format1.fitted && r.v7.fitted, "{r:?}");
    assert!(r.check.ran && r.check.groups >= 80, "{:?}", r.check);
    assert!(r.check.train_rows + r.check.test_rows == 400);
    let (f1, v7) = (r.check.score_format1.unwrap(), r.check.score_v7.unwrap());
    assert!((-1.0..=1.25).contains(&f1) && (-1.0..=1.25).contains(&v7));
    let expect = if v7 > f1 + 0.02 {
        Encoder::V7
    } else {
        Encoder::Format1
    };
    assert_eq!(fitted.encoder, expect);
    assert!(fitted.bytes.len() <= 2048);
    let json = serde_json::to_string(r).unwrap();
    for leaked in MIXED_HEADERS.iter().chain(&MIXED_LABELS) {
        assert!(!json.contains(leaked), "report leaks {leaked}: {json}");
    }
    let again = fit_auto(&t, 2, &opts(2048, 1), &m).unwrap();
    assert_eq!(fitted.bytes, again.bytes, "same seed, same bytes");
    // The chosen encoder alone writes the same file without the check.
    let alone = fit_auto(&t, 2, &single_encoder(fitted.encoder, 1), &m).unwrap();
    assert_eq!(alone.bytes, fitted.bytes);
    assert!(!alone.report.check.ran && alone.report.mode == fitted.encoder.name());
}

#[test]
fn a_requested_encoder_is_the_only_one_fitted() {
    let (t, m) = (mixed(300, 6), model());
    for e in [Encoder::V7, Encoder::Format1] {
        let fitted = fit_auto(&t, 2, &single_encoder(e, 2), &m).unwrap();
        let r = &fitted.report;
        assert_eq!((fitted.encoder, fitted.bytes[4]), (e, e.tag()));
        let (asked, other) = match e {
            Encoder::V7 => (&r.v7, &r.format1),
            Encoder::Format1 => (&r.format1, &r.v7),
        };
        assert!(asked.requested && asked.fitted && !other.requested && !other.fitted);
        assert!(!r.check.ran && r.check.note.contains("requested"));
    }
    // A requested encoder that cannot fit returns its own error.
    let small = AutoOptions {
        budget: 64,
        ..single_encoder(Encoder::Format1, 0)
    };
    let err = fit_auto(&t, 2, &small, &m).err().expect("64 bytes");
    let text = err.to_string();
    assert!(
        text.starts_with("budget") && !text.contains("no encoder"),
        "{text}"
    );
}

#[test]
fn v7_payloads_refuse_foreign_weights_and_truncation() {
    let t = wide(40, 200, 4);
    let m = model();
    let fitted = fit_auto(&t, 40, &single_encoder(Encoder::V7, 0), &m).unwrap();
    assert_eq!(fitted.encoder, Encoder::V7);
    let mut foreign = fitted.bytes.clone();
    foreign[DPA1_HEADER_BYTES - 1] ^= 0x5a;
    let err = V7Decoder::open(&parse_dpa1(&foreign).unwrap(), &m)
        .err()
        .unwrap();
    assert!(err.to_string().contains("fingerprint mismatch"), "{err}");
    // The zero-init decoder uses no weights, so it opens any intact V7 payload.
    let csv = |dpa: &[u8], how: V7Decode<'_>, seed: u64| {
        let decoder = match how {
            V7Decode::Trained(w) => V7Decoder::open(&parse_dpa1(dpa).unwrap(), w),
            V7Decode::ZeroInit => V7Decoder::open_weightless(&parse_dpa1(dpa).unwrap()),
        };
        let mut out = Vec::new();
        decoder.unwrap().write_csv(how, 50, seed, &mut out).unwrap();
        String::from_utf8(out).unwrap()
    };
    let zero = csv(&foreign, V7Decode::ZeroInit, 3);
    assert_eq!(zero, csv(&fitted.bytes, V7Decode::ZeroInit, 3));
    assert_ne!(zero, csv(&fitted.bytes, V7Decode::ZeroInit, 4));
    assert_ne!(zero, csv(&fitted.bytes, V7Decode::Trained(&m), 3));
    assert_eq!(zero.lines().count(), 51);
    assert!(zero.lines().skip(1).all(|l| l
        .split(',')
        .filter(|c| c.parse::<f64>().is_ok())
        .count()
        == 41));
    for cut in [
        DPA1_HEADER_BYTES + 1,
        DPA1_HEADER_BYTES + 40,
        fitted.bytes.len() - 1,
    ] {
        let short = &fitted.bytes[..cut];
        let opened = parse_dpa1(short).and_then(|d| V7Decoder::open(&d, &m));
        assert!(opened.is_err(), "accepted a payload cut to {cut} bytes");
    }
    let mut flipped = fitted.bytes.clone();
    let last = flipped.len() - 9;
    flipped[last] ^= 0x01;
    assert!(
        V7Decoder::open(&parse_dpa1(&flipped).unwrap(), &m).is_err(),
        "crc"
    );

    let f1 = fit(
        &mixed(300, 2),
        2,
        &FitOptions {
            budget: Some(2000),
            ..Default::default()
        },
    )
    .unwrap()
    .bytes;
    let file = wrap_dpa1(Encoder::Format1, NO_FINGERPRINT, &f1);
    let dpa = parse_dpa1(&file).unwrap();
    assert!(decode(dpa.payload).is_ok());
    assert!(
        V7Decoder::open(&dpa, &m).is_err(),
        "Format 1 payload opened as V7"
    );
    let short = parse_dpa1(&file[..file.len() - 3]).unwrap();
    assert!(
        decode(short.payload).is_err(),
        "truncated Format 1 payload decoded"
    );
}

#[test]
fn tiny_budgets_and_tiny_tables_fail_cleanly() {
    let m = model();
    let t = mixed(300, 8);
    for budget in [1, DPA1_HEADER_BYTES, DPA1_HEADER_BYTES + 1, 64] {
        assert!(
            fit_auto(&t, 2, &opts(budget, 0), &m).is_err(),
            "budget {budget}"
        );
    }
    let tiny = table(
        &["a", "y"],
        (0..6)
            .map(|i| vec![i.to_string(), (i * 2).to_string()])
            .collect(),
    );
    let fitted = fit_auto(&tiny, 1, &opts(2048, 0), &m).unwrap();
    assert_eq!(
        fitted.encoder,
        Encoder::V7,
        "Format 1 needs 20 rows, V7 four"
    );
    assert!(fitted
        .report
        .format1
        .error
        .as_deref()
        .unwrap()
        .starts_with("input"));
}
