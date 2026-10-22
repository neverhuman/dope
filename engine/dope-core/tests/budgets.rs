//! Hard caps apply to the decoded complete generator, including forced plans.
use dope_core::{decode, fit, FitOptions, ModelSize, RawTable, Task};

fn table(classify: bool) -> RawTable {
    let mut csv = String::from("a,b,c,y\n");
    for i in 0..150 {
        let a = (i * 17 % 149) as f64 / 149.;
        let b = (i * 31 % 151) as f64 / 151.;
        let c = (i * 53 % 157) as f64 / 157.;
        let y = if classify {
            (i % 3).to_string()
        } else {
            format!("{}", a + b * c)
        };
        csv.push_str(&format!("{a},{b},{c},{y}\n"));
    }
    RawTable::from_reader(csv.as_bytes()).unwrap()
}

fn opts() -> FitOptions {
    FitOptions {
        audit: false,
        max_steps: 30,
        regimes: Some(1),
        dependence: Some("independent".into()),
        ..Default::default()
    }
}

#[test]
fn complete_generator_cap_selects_smaller_plans_and_roundtrips() {
    for classify in [false, true] {
        let fitted = fit(
            &table(classify),
            3,
            &FitOptions {
                budget: Some(4096),
                max_parameters: Some(120),
                task: Some(if classify {
                    Task::Classification
                } else {
                    Task::Regression
                }),
                ..opts()
            },
        )
        .unwrap();
        let p = decode(&fitted.bytes).unwrap();
        assert!(fitted.bytes.len() <= 4096);
        assert!(p.parameters().total <= 120);
        assert_eq!(p.parameters(), fitted.report.parameters);
        assert!(p.parameters().total > p.net.parameter_count());
    }
}

#[test]
fn presets_and_explicit_budget_precedence() {
    for (size, cap) in [
        (ModelSize::Small, 2048),
        (ModelSize::Medium, 4096),
        (ModelSize::Large, 16384),
    ] {
        assert_eq!(size.budget(), cap);
    }
    let fitted = fit(
        &table(false),
        3,
        &FitOptions {
            size: Some(ModelSize::Large),
            budget: Some(2048),
            plan: Some((3, 1, 0)),
            trees: false,
            ..opts()
        },
    )
    .unwrap();
    assert_eq!(fitted.report.budget, 2048);
    // This fixture has 2 linear coefficients, 9 marginal knots, 13 noise
    // scalars and 13 affine/missing/range slots. Zero-valued slots still count.
    assert_eq!(fitted.report.parameters.total, 37);
    assert_eq!(fitted.report.size, Some(ModelSize::Large));
    assert!(fitted.bytes.len() <= 2048);
    let fitted = fit(
        &table(false),
        3,
        &FitOptions {
            size: Some(ModelSize::Small),
            plan: Some((3, 1, 0)),
            trees: false,
            ..opts()
        },
    )
    .unwrap();
    assert_eq!(fitted.report.budget, 2048);
}

#[test]
fn impossible_caps_and_forced_architectures_never_escape_limits() {
    for options in [
        FitOptions {
            max_parameters: Some(1),
            ..opts()
        },
        FitOptions {
            budget: Some(1),
            plan: Some((3, 1, 0)),
            ..opts()
        },
        FitOptions {
            max_parameters: Some(120),
            plan: Some((9, 3, 16)),
            ..opts()
        },
        FitOptions {
            budget: Some(0),
            ..opts()
        },
        FitOptions {
            max_parameters: Some(0),
            ..opts()
        },
    ] {
        assert!(matches!(
            fit(&table(false), 3, &options),
            Err(dope_core::DopeError::Budget(_))
        ));
    }
}

#[test]
fn a_too_small_byte_budget_names_the_size_it_needs() {
    let options = FitOptions {
        budget: Some(64),
        ..opts()
    };
    let Err(err) = fit(&table(false), 3, &options) else {
        panic!("a 64-byte budget fitted");
    };
    assert!(err.to_string().contains("at least"), "{err}");
}
