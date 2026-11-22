//! End-to-end tests of the `dope_statpacket` command line beyond retention:
//! `inspect` shows structure only, wide and many-class tables meet the byte
//! budget through the documented fallback and still decode, `sample` is
//! row-local with the target attached, and `embed` separates tables.

mod statpacket_support;

use statpacket_support::eval::Table;
use statpacket_support::{
    cosine, dir, embed, fit, fixtures, multiclass_table, ok_json, read_mat, run, sample,
    spread_table, text, truncate, Rng, FIT_ROWS, LEVELS,
};

const BUDGET: usize = 2048;

/// Every number in a JSON value, depth first.
fn numbers(value: &serde_json::Value, out: &mut Vec<serde_json::Number>) {
    match value {
        serde_json::Value::Number(n) => out.push(n.clone()),
        serde_json::Value::Array(items) => items.iter().for_each(|v| numbers(v, out)),
        serde_json::Value::Object(map) => map.values().for_each(|v| numbers(v, out)),
        _ => {}
    }
}

#[test]
fn inspect_shows_the_header_and_the_ledger_and_no_data_value() {
    let d = dir("inspect");
    for f in fixtures() {
        // A planted value that would be easy to spot if anything echoed data.
        let mut table = f.fit.clone();
        table.x[5 * table.p] = 31_337.125;
        let (packet, card) = (
            d.join(format!("{}.dse", f.name)),
            d.join(format!("{}.json", f.name)),
        );
        fit(&table, f.task, f.classes, BUDGET, &packet, &card);
        let out = run(&["inspect", "--packet", text(&packet)]);
        assert!(out.status.success());
        let raw = String::from_utf8(out.stdout).unwrap();
        let report: serde_json::Value = serde_json::from_str(&raw).unwrap();
        // Structure: header fields, level, one ledger entry per section.
        let bytes = std::fs::read(&packet).unwrap();
        assert_eq!(report["header"]["magic"], "DSE1");
        assert_eq!(report["header"]["cols"], f.fit.p);
        assert_eq!(report["header"]["level"], "E");
        assert_eq!(report["header"]["classes"], f.classes);
        assert_eq!(report["ledger"]["sections"].as_array().unwrap().len(), 10);
        assert_eq!(report["ledger"]["total"], bytes.len());
        let sections: u64 = report["ledger"]["sections"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .sum();
        assert_eq!(sections + 28, bytes.len() as u64);
        // No float anywhere: every number is a non-negative integer (a count,
        // a byte length or a header field), and no digit string has a decimal point.
        let mut found = Vec::new();
        numbers(&report, &mut found);
        assert!(found.len() >= 20);
        assert!(found.iter().all(|n| n.is_u64()), "{raw}");
        let digits = |c: char| c.is_ascii_digit();
        assert!(!raw
            .as_bytes()
            .windows(3)
            .any(|w| digits(w[0] as char) && w[1] == b'.' && digits(w[2] as char)));
        assert!(!raw.contains("31337"));
        // `info` is the same command under its earlier name.
        let again = run(&["info", "--packet", text(&packet)]);
        assert_eq!(String::from_utf8(again.stdout).unwrap(), raw);
    }
}

/// A regression table of `p` columns: two latent factors, a sparse linear signal.
fn wide_regression(p: usize, rows: usize) -> Table {
    let mut rng = Rng(0x1de_0004 + p as u64);
    let mut table = Table {
        x: Vec::new(),
        y: Vec::new(),
        p,
    };
    for _ in 0..rows {
        let g = [rng.normal(), rng.normal()];
        let x: Vec<f64> = (0..p).map(|j| 0.5 * g[j % 2] + rng.normal()).collect();
        table.y.push(x[0] - x[7] + 0.5 * x[20] + 0.5 * rng.normal());
        table.x.extend(x);
    }
    table
}

#[test]
fn wide_and_many_class_tables_meet_the_budget_and_decode() {
    let d = dir("fallback");
    let cases: Vec<(&str, &str, usize, Table)> = vec![
        (
            "multiclass K=5 P=48",
            "multiclass",
            5,
            multiclass_table(48, FIT_ROWS, 0x4801),
        ),
        (
            "multiclass K=5 P=48, main effects",
            "multiclass",
            5,
            spread_table(48, FIT_ROWS, 0x4802),
        ),
        (
            "multiclass K=5 P=200",
            "multiclass",
            5,
            multiclass_table(200, FIT_ROWS, 0x2001),
        ),
        (
            "regression P=200",
            "regression",
            0,
            wide_regression(200, FIT_ROWS),
        ),
        (
            "regression P=320",
            "regression",
            0,
            wide_regression(320, FIT_ROWS),
        ),
    ];
    for (i, (name, task, classes, table)) in cases.iter().enumerate() {
        let (packet, card) = (d.join(format!("{i}.dse")), d.join(format!("{i}.json")));
        let report = fit(table, task, *classes, BUDGET, &packet, &card);
        let bytes = std::fs::read(&packet).unwrap();
        println!(
            "{name}: level {} via {}, {} bytes, target columns {}, main-effect columns {}",
            report["level"].as_str().unwrap(),
            report["budget_step"].as_str().unwrap(),
            bytes.len(),
            report["target_cols"],
            report["main_effect_cols"]
        );
        assert_eq!(report["status"], "ok", "{name}");
        assert!(bytes.len() <= BUDGET, "{name}: {}", bytes.len());
        assert_eq!(report["header"]["cols"], table.p);
        // The packet decodes: finite rows of the right shape, labels inside the classes.
        let rows = sample(&packet, &card, 1000, 0, 3, "sampled");
        assert_eq!((rows.rows(), rows.p), (1000, table.p));
        assert!(rows.x.iter().chain(&rows.y).all(|v| v.is_finite()));
        if *classes > 0 {
            assert!(rows
                .y
                .iter()
                .all(|&y| y.fract() == 0.0 && (y as usize) < *classes));
            let distinct = (0..*classes)
                .filter(|&c| rows.y.contains(&(c as f64)))
                .count();
            assert_eq!(distinct, *classes, "{name}");
        }
        // Every lower level is no larger and decodes too.
        let top = LEVELS
            .iter()
            .position(|l| *l == report["level"].as_str().unwrap())
            .unwrap();
        let mut last = bytes.len();
        for level in LEVELS[..top].iter().rev() {
            let cut = d.join(format!("{i}.{level}.dse"));
            truncate(&packet, level, &cut);
            let len = std::fs::read(&cut).unwrap().len();
            assert!(len <= last);
            last = len;
            assert_eq!(sample(&cut, &card, 50, 0, 3, "sampled").rows(), 50);
        }
    }
    // The documented fallbacks are the ones in use.
    let step = |i: usize| {
        let report = ok_json(&["inspect", "--packet", text(&d.join(format!("{i}.dse")))]);
        (
            report["header"]["level"].as_str().unwrap().to_string(),
            report["header"]["sparse_target"].as_bool().unwrap(),
        )
    };
    // K = 5, P = 200: dense level A is 2,050 bytes, so S3 keeps the leading columns only.
    assert_eq!(step(2), ("A".to_string(), true));
    // Regression: level A up to P = 287 dense, sparse above.
    assert_eq!(step(3), ("A".to_string(), false));
    assert_eq!(step(4), ("A".to_string(), true));
    // With 36 gated S8 columns the main-effects table is 1,968 bytes at level D;
    // a budget below that keeps level D with S8 on the leading columns only.
    let (packet, card) = (d.join("partial.dse"), d.join("partial.json"));
    let report = fit(&cases[1].3, "multiclass", 5, 1900, &packet, &card);
    assert_eq!(report["budget_step"], "PartialMainEffects");
    assert_eq!(
        (&report["level"], &report["status"]),
        (&"D".into(), &"ok".into())
    );
    let kept = report["main_effect_cols"].as_u64().unwrap();
    assert!((1..36).contains(&kept), "{kept}");
    assert!(report["packet_bytes"].as_u64().unwrap() <= 1900);
    assert_eq!(sample(&packet, &card, 200, 0, 3, "sampled").rows(), 200);
    println!("multiclass K=5 P=48, main effects, budget 1,900: level D with S8 on {kept} of 36 columns, {} bytes", report["packet_bytes"]);
    // A table that cannot fit is reported, never cut: P = 400 needs 2,434 bytes at least.
    let (packet, card) = (d.join("over.dse"), d.join("over.json"));
    let report = fit(
        &wide_regression(400, 64),
        "regression",
        0,
        BUDGET,
        &packet,
        &card,
    );
    assert_eq!(report["status"], "over_budget");
    assert!(report["packet_bytes"].as_u64().unwrap() > BUDGET as u64);
}

#[test]
fn sample_is_row_local_and_reproducible_with_the_target() {
    let d = dir("sample");
    for f in fixtures() {
        let (packet, card) = (
            d.join(format!("{}.dse", f.name)),
            d.join(format!("{}.json", f.name)),
        );
        fit(&f.fit, f.task, f.classes, BUDGET, &packet, &card);
        let whole = sample(&packet, &card, 1000, 0, 11, "sampled");
        let (head, tail) = (
            sample(&packet, &card, 500, 0, 11, "sampled"),
            sample(&packet, &card, 500, 500, 11, "sampled"),
        );
        let bits = |v: &[f64]| v.iter().map(|x| (*x as f32).to_bits()).collect::<Vec<_>>();
        let joined_x: Vec<f64> = head.x.iter().chain(&tail.x).copied().collect();
        let joined_y: Vec<f64> = head.y.iter().chain(&tail.y).copied().collect();
        assert_eq!(bits(&whole.x), bits(&joined_x), "{}", f.name);
        assert_eq!(bits(&whole.y), bits(&joined_y), "{}", f.name);
        let again = sample(&packet, &card, 1000, 0, 11, "sampled");
        assert_eq!(
            (bits(&whole.x), bits(&whole.y)),
            (bits(&again.x), bits(&again.y))
        );
        let other = sample(&packet, &card, 1000, 0, 12, "sampled");
        assert_ne!(bits(&whole.x), bits(&other.x));
        // Fitting again gives the same packet bytes and the same scale card.
        let (packet2, card2) = (d.join("again.dse"), d.join("again.json"));
        fit(&f.fit, f.task, f.classes, BUDGET, &packet2, &card2);
        assert_eq!(
            std::fs::read(&packet).unwrap(),
            std::fs::read(&packet2).unwrap()
        );
        assert_eq!(
            std::fs::read(&card).unwrap(),
            std::fs::read(&card2).unwrap()
        );
        // Raw units: integer and binary columns stay on their FIT levels.
        if f.name == "regression" {
            for row in whole.x.chunks(whole.p) {
                assert!(row[6] == 0.0 || row[6] == 1.0);
                assert_eq!(row[5], row[5].round());
                assert!((0.0..=4.0).contains(&row[7]) && row[7].fract() == 0.0);
            }
        }
        // Without a card the output is in unit space; an unknown label rule is refused.
        let unit = d.join("unit.mat");
        let report = ok_json(&[
            "sample",
            "--packet",
            text(&packet),
            "--rows",
            "20",
            "--out",
            text(&unit),
        ]);
        assert_eq!(report["units"], "unit");
        assert!(read_mat(&unit).x.iter().all(|v| v.abs() <= 6.0));
        let bad = run(&[
            "sample",
            "--packet",
            text(&packet),
            "--rows",
            "5",
            "--labels",
            "argmax",
            "--out",
            text(&d.join("bad.mat")),
        ]);
        assert_eq!(bad.status.code(), Some(2));
        assert_eq!(
            String::from_utf8(bad.stderr).unwrap().trim(),
            "dope_statpacket: stat packet: labels must be sampled or regime"
        );
    }
}

#[test]
fn embed_separates_tables_and_ignores_column_order() {
    let d = dir("embed");
    let all = fixtures();
    let mut halves = Vec::new();
    let mut report = Vec::new();
    for f in &all {
        let half = FIT_ROWS / 2;
        let mut codes = Vec::new();
        for (tag, table) in [
            ("a", f.fit.slice(0, half)),
            ("b", f.fit.slice(half, FIT_ROWS)),
        ] {
            let (packet, card) = (d.join(format!("{}.{tag}.dse", f.name)), d.join("card.json"));
            fit(&table, f.task, f.classes, BUDGET, &packet, &card);
            let (short, long) = embed(&packet);
            assert_eq!((short.len(), long.len()), (64, 128));
            assert_eq!(short[..], long[..64], "the 64-D embedding is the prefix");
            codes.push((short, long));
        }
        // The columns in reverse order: same embedding up to rounding of the refit.
        let order: Vec<usize> = (0..f.fit.p).rev().collect();
        let (packet, card) = (d.join(format!("{}.perm.dse", f.name)), d.join("card.json"));
        fit(
            &f.fit.slice(0, half).permuted(&order),
            f.task,
            f.classes,
            BUDGET,
            &packet,
            &card,
        );
        let permuted = embed(&packet).0;
        let moved = permuted
            .iter()
            .zip(&codes[0].0)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap();
        report.push(format!(
            "{}: halves cosine {:.4} (64-D) {:.4} (128-D); reversed columns: cosine {:.5}, largest code change {moved}",
            f.name,
            cosine(&codes[0].0, &codes[1].0),
            cosine(&codes[0].1, &codes[1].1),
            cosine(&permuted, &codes[0].0)
        ));
        assert!(
            cosine(&permuted, &codes[0].0) > 0.995 && moved <= 8,
            "{}",
            f.name
        );
        halves.push(codes);
    }
    for dim in [0usize, 1] {
        let pick = |c: &(Vec<i8>, Vec<i8>)| if dim == 0 { c.0.clone() } else { c.1.clone() };
        let same = halves
            .iter()
            .map(|c| cosine(&pick(&c[0]), &pick(&c[1])))
            .fold(f64::INFINITY, f64::min);
        let mut other = f64::NEG_INFINITY;
        for i in 0..halves.len() {
            for j in 0..halves.len() {
                if i != j {
                    other = other.max(cosine(&pick(&halves[i][0]), &pick(&halves[j][1])));
                }
            }
        }
        report.push(format!(
            "{}-D: lowest halves cosine {same:.4}, highest cosine between different tables {other:.4}",
            [64, 128][dim]
        ));
        assert!(same > other, "{same} {other}");
    }
    println!("{}", report.join("\n"));
}
