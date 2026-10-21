use super::*;

fn table(rows: usize) -> (Vec<String>, Vec<Vec<String>>) {
    let header = ["age", "city", "joined", "income", "spend", "member"]
        .map(String::from)
        .to_vec();
    let cities = ["Boise", "Provo", "Denver", "Ogden"];
    let body = (0..rows)
        .map(|row| {
            let unit = |salt: u64| {
                let mut x = (row as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ salt;
                x ^= x >> 31;
                x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
                (x >> 11) as f64 / (1u64 << 53) as f64
            };
            let age = 20 + (unit(1) * 50.0) as i64;
            let city = cities[(unit(2) * 4.0) as usize];
            let income = (unit(3) * 9.0).exp() * 10.0; // heavy right tail
            let lift = match city {
                "Denver" => 40.0,
                "Boise" => 10.0,
                _ => 0.0,
            };
            let spend = 0.002 * income + 1.5 * age as f64 + lift + 5.0 * unit(4);
            vec![
                age.to_string(),
                if row % 17 == 0 {
                    String::new()
                } else {
                    city.to_string()
                },
                format!("2024-{:02}-{:02}", 1 + row % 12, 1 + row % 28),
                format!("{income:.2}"),
                format!("{spend:.1}"),
                if spend > 120.0 {
                    "yes".into()
                } else {
                    "no".into()
                },
            ]
        })
        .collect();
    (header, body)
}

#[test]
fn plans_types_and_round_trips_codes() {
    let (header, rows) = table(400);
    let mut options = TableOptions::new("spend");
    options.drop.push("member".into());
    let planned = plan_table(&header, &rows, &options).unwrap();
    let kinds = planned
        .plan
        .columns
        .iter()
        .map(|c| c.encoding.kind())
        .collect::<Vec<_>>();
    assert_eq!(
        kinds,
        ["integer", "category", "date", "numeric_log", "numeric"]
    );
    assert_eq!(planned.plan.task, Task::Regression);
    assert_eq!(planned.dropped, vec![("member".to_string(), "requested")]);
    let Encoding::Category { levels } = &planned.plan.columns[1].encoding else {
        panic!()
    };
    // Ordered by mean spend: the missing level and low-lift cities first, Denver last.
    assert_eq!(levels.last().map(String::as_str), Some("Denver"));
    for (cell, column) in [("Denver", 1), ("2024-03-04", 2), ("37", 0)] {
        let encoding = &planned.plan.columns[column].encoding;
        assert_eq!(encoding.decode(encoding.encode(cell)), cell);
    }
    assert!(planned.plan.columns[3].encoding.encode("").is_nan());
}

#[test]
fn binary_targets_and_plans_survive_the_container() {
    let (header, rows) = table(300);
    let mut options = TableOptions::new("member");
    options.drop.push("spend".into());
    let planned = plan_table(&header, &rows, &options).unwrap();
    assert_eq!(planned.plan.task, Task::Binary);
    assert!(planned.target.iter().all(|v| *v == 0.0 || *v == 1.0));
    let kernel = crate::codec::encode_kernel(
        &crate::codec::decode_kernel(
            &crate::compiler::compile_kernel_from_arrays(
                &planned.features,
                &planned.target,
                planned.rows,
                planned.plan.feature_count(),
                planned.plan.task,
                &crate::compiler::CompileOptions {
                    seed: Some(7),
                    backend_id: Some("chow_liu".into()),
                    ..Default::default()
                },
            )
            .unwrap()
            .artifact,
        )
        .unwrap(),
    )
    .unwrap();
    let (artifact, accounting) = write_table_artifact(&planned.plan, &kernel);
    let (plan, _, read) = read_table_artifact(&artifact).unwrap();
    assert_eq!(plan, planned.plan);
    assert_eq!(read, accounting);
    assert_eq!(accounting.total_bytes, artifact.len());
    assert!(read_table_artifact(&artifact[..artifact.len() - 1]).is_err());
}

#[test]
fn fit_and_sample_return_the_source_schema() {
    let (header, rows) = table(500);
    let mut options = FitTableOptions::new("spend");
    options.native_target_profile =
        crate::compiler::NativeTargetProfile::parse("features12_steps256").ok();
    let (artifact, report) = fit_table(&header, &rows, &options).unwrap();
    assert_eq!(report.bytes.total_bytes, artifact.len());
    assert!(crate::codec::is_packed(
        &artifact[artifact.len() - report.bytes.kernel_bytes..]
    ));
    let (sampled_header, sampled) = sample_table(&artifact, 200, 3).unwrap();
    assert_eq!(sampled_header, header);
    assert_eq!(sampled.len(), 200);
    let cities = ["Boise", "Provo", "Denver", "Ogden", ""];
    for row in &sampled {
        assert!(
            row[0]
                .parse::<i64>()
                .is_ok_and(|age| (20..=70).contains(&age))
        );
        assert!(cities.contains(&row[1].as_str()) || row[1] == RARE_LEVEL);
        assert!(row[2].is_empty() || parse_date(&row[2]).is_some());
        assert!(row[5] == "yes" || row[5] == "no" || row[5].is_empty());
    }
    let mean = |column: usize, data: &[Vec<String>]| {
        let values = data
            .iter()
            .filter_map(|r| r[column].parse::<f64>().ok())
            .collect::<Vec<_>>();
        values.iter().sum::<f64>() / values.len() as f64
    };
    let (real, synthetic) = (mean(4, &rows), mean(4, &sampled));
    assert!(
        (real - synthetic).abs() < 0.2 * real,
        "{real} vs {synthetic}"
    );
    let inspection = inspect_table(&artifact).unwrap();
    assert_eq!(inspection["bytes"]["total_bytes"], artifact.len());
    assert!(!inspection.to_string().contains("Denver"));
}

#[test]
fn refuses_unusable_targets() {
    let (header, rows) = table(100);
    assert!(plan_table(&header, &rows, &TableOptions::new("missing")).is_err());
    assert!(plan_table(&header, &rows, &TableOptions::new("city")).is_err());
    let constant = rows
        .iter()
        .map(|r| {
            let mut r = r.clone();
            r[4] = "1".into();
            r
        })
        .collect::<Vec<_>>();
    assert!(plan_table(&header, &constant, &TableOptions::new("spend")).is_err());
}

#[test]
fn dates_round_trip_through_day_counts() {
    for cell in ["1970-01-01", "2000-02-29", "2024-12-31", "1969-07-20"] {
        let (days, time) = parse_date(cell).unwrap();
        assert_eq!(format_date(days, time), cell);
    }
    assert_eq!(parse_date("1970-01-02").unwrap().0, 1.0);
    let (days, time) = parse_date("2024-05-06T07:08:09").unwrap();
    assert!(time);
    assert_eq!(format_date(days, true), "2024-05-06T07:08:09");
    assert!(parse_date("2024-13-01").is_none() && parse_date("hello").is_none());
    // Calendar and clock bounds.
    assert!(parse_date("2024-02-31").is_none() && parse_date("2023-02-29").is_none());
    assert!(parse_date("2024-04-31").is_none() && parse_date("2024-01-01T24:00").is_none());
    // A time that rounds up to midnight moves to the next day.
    let (late, _) = parse_date("2024-05-06T23:59:59.8").unwrap();
    assert_eq!(format_date(late, true), "2024-05-07T00:00:00");
    // Date-only values round to the nearest day.
    assert_eq!(format_date(19_000.6, false), format_date(19_001.0, false));
}

#[test]
fn a_single_rare_level_keeps_its_label_and_numeric_labels_are_distinct_by_value() {
    let cells = ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b", "c"];
    let target = [0.0; 11];
    let Some(Encoding::Category { levels }) = plan_category(&cells, &target, 5) else {
        panic!("category plan expected");
    };
    assert!(levels.contains(&"c".to_string()) && !levels.contains(&RARE_LEVEL.to_string()));
    let labels = ["1", "0", "1.0", "0", "1"];
    let (task, encoding) = plan_target(&labels, TaskChoice::Auto, 2.0).unwrap();
    assert_eq!(task, Task::Binary);
    assert_eq!(encoding.encode("1.0"), encoding.encode("1"));
    assert_ne!(encoding.encode("0"), encoding.encode("1"));
    // Text labels are ordered by text, not by first appearance.
    let (_, first_yes) = plan_target(&["yes", "no", "yes"], TaskChoice::Auto, 2.0).unwrap();
    let (_, first_no) = plan_target(&["no", "yes", "yes"], TaskChoice::Auto, 2.0).unwrap();
    assert_eq!(first_yes, first_no);
    // Date bounds keep seconds.
    let late = Encoding::Date {
        time: true,
        low: 19_000.0,
        high: 19_000.0 + 1.0 / 86_400.0,
    };
    assert_eq!(late.encode("2022-01-08T00:00:01"), 1.0);
}

#[test]
fn packed_values_shrink_artifacts_and_keep_samples_identical() {
    let (header, rows) = table(400);
    let planned = plan_table(&header, &rows, &TableOptions::new("spend")).unwrap();
    let compile = |packed_codec: bool| {
        crate::compiler::compile_kernel_from_arrays(
            &planned.features,
            &planned.target,
            planned.rows,
            planned.plan.feature_count(),
            planned.plan.task,
            &crate::compiler::CompileOptions {
                seed: Some(11),
                backend_id: Some("compact_neural_residual_symbolic".into()),
                native_target_profile: crate::compiler::NativeTargetProfile::parse(
                    "features12_steps128",
                )
                .ok(),
                packed_codec,
                ..Default::default()
            },
        )
        .unwrap()
        .artifact
    };
    let (historical, packed) = (compile(false), compile(true));
    assert!(crate::codec::is_packed(&packed) && !crate::codec::is_packed(&historical));
    assert!(
        packed.len() * 3 < historical.len() * 2,
        "{} vs {}",
        packed.len(),
        historical.len()
    );
    let draw = |artifact: &[u8]| {
        let kernel = crate::codec::decode_kernel(artifact).unwrap();
        crate::sample::sample_kernel(
            &crate::codec::LoadedKernel::V3(Box::new(kernel)),
            crate::sample::SampleOptions {
                rows: 300,
                seed: Some(5),
            },
        )
        .unwrap()
    };
    let (left, right) = (draw(&historical), draw(&packed));
    assert_eq!(left.len(), right.len());
    assert!(
        left.iter()
            .zip(&right)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
}
