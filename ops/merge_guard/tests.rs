//! Original scenario and native boundary proofs; no network or Python runtime.
mod feature_boundary;
mod legacy;
mod native;
mod workflow_events;

use super::*;
use serde_json::json;
use std::collections::VecDeque;

struct ScriptedApi {
    plan: VecDeque<Value>,
    observed: Vec<Value>,
}
impl ScriptedApi {
    fn new(plan: &Value) -> Self {
        Self {
            plan: array(plan).unwrap().iter().cloned().collect(),
            observed: Vec::new(),
        }
    }
    fn take(&mut self, method: &str, args: Value, optional: bool, paginate: bool) -> Result<Value> {
        let item = self.plan.pop_front().expect("unexpected API call");
        assert_eq!(item["method"], method);
        assert_eq!(item["args"], args);
        if method == "get" {
            assert_eq!(
                item["kwargs"]
                    .get("optional")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                optional
            );
            assert_eq!(
                item["kwargs"]
                    .get("paginate")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                paginate
            );
        }
        self.observed
            .push(json!({"endpoint":args[0],"method":"GET"}));
        if let Some(reason) = item["refused"].as_str() {
            assert_eq!(reason, "api_unavailable");
            Err(Refused("api_unavailable"))
        } else {
            Ok(item["value"].clone())
        }
    }
}
impl github::Api for ScriptedApi {
    fn get(&mut self, endpoint: &str, optional: bool, paginate: bool) -> Result<Value> {
        self.take("get", json!([endpoint]), optional, paginate)
    }
    fn listing(&mut self, endpoint: &str, key: Option<&str>) -> Result<Value> {
        let args = match key {
            Some(k) => json!([endpoint, k]),
            None => json!([endpoint]),
        };
        self.take("listing", args, false, false)
    }
    fn open_pulls(&mut self, root: &str) -> Result<Value> {
        self.take("open_pulls", json!([root]), false, false)
    }
    fn calls(&self) -> Value {
        json!(self.observed)
    }
}
struct GetPlanApi(ScriptedApi);
impl github::Api for GetPlanApi {
    fn get(&mut self, endpoint: &str, optional: bool, paginate: bool) -> Result<Value> {
        self.0.take("get", json!([endpoint]), optional, paginate)
    }
    fn calls(&self) -> Value {
        json!(self.0.observed)
    }
}
struct RecordedTransport(VecDeque<Value>);
impl github::Transport for RecordedTransport {
    fn run(&mut self, argv: &[String]) -> Result<github::Response> {
        let item = self.0.pop_front().expect("transport retried unexpectedly");
        assert_eq!(json!(argv), item["argv"]);
        assert_eq!(item["timeout"], 45);
        assert_eq!(argv.get(3).map(String::as_str), Some("GET"));
        Ok(github::Response {
            code: item["exit"].as_i64().unwrap() as i32,
            stdout: item["stdout"].as_str().unwrap().into(),
            stderr: String::new(),
        })
    }
}
fn parity(name: &str) {
    use github::Api;
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(format!("{name}.json"));
    let cases: Value = serde_json::from_slice(&std::fs::read(fixture).unwrap()).unwrap();
    assert!(!array(&cases).unwrap().is_empty());
    for (index, case) in array(&cases).unwrap().iter().enumerate() {
        let args = &case["args"];
        let kwargs = &case["kwargs"];
        let result = match text(&case["operation"]).unwrap() {
            "evaluate_preflight" => evidence::evaluate_preflight(&args[0], &args[1], &args[2])
                .map(controls::Policy::value),
            "evaluate_merge" => evidence::evaluate_merge(&args[0], &args[1]).map(|s| json!(s)),
            "inspect" | "inspect_controls" => {
                let mut api = ScriptedApi::new(&case["api"]);
                let actual = if case["operation"] == "inspect_controls" {
                    inspection::inspect_controls(
                        &mut api,
                        text(&args[0]).unwrap(),
                        text(&args[1]).unwrap(),
                    )
                } else {
                    let receipt = args.get(1).unwrap_or(&Value::Null);
                    let prior = kwargs.get("prior");
                    inspection::inspect(&mut api, &args[0], receipt, prior)
                };
                assert!(
                    api.plan.is_empty(),
                    "{name} subcase{index}: omitted API calls"
                );
                assert_eq!(actual["merge_authorized"], false);
                for (key, value) in case["expected"].as_object().unwrap() {
                    assert_eq!(&actual[key], value, "{name} subcase{index}, {key}");
                }
                continue;
            }
            "listing" | "open_pulls" => {
                let mut api = GetPlanApi(ScriptedApi::new(&case["api"]));
                let result = if case["operation"] == "listing" {
                    api.listing(text(&args[0]).unwrap(), args.get(1).and_then(Value::as_str))
                } else {
                    api.open_pulls(text(&args[0]).unwrap())
                };
                assert!(
                    api.0.plan.is_empty(),
                    "{name} subcase{index}: omitted pages"
                );
                result
            }
            "get" => {
                let transport =
                    RecordedTransport(array(&case["transport"]).unwrap().iter().cloned().collect());
                let mut api = github::GitHub::new("gh".into(), transport);
                let result = api.get(
                    text(&args[0]).unwrap(),
                    kwargs["optional"].as_bool().unwrap_or(false),
                    kwargs["paginate"].as_bool().unwrap_or(false),
                );
                assert!(api.transport.0.is_empty(), "{name} subcase{index}");
                result
            }
            other => panic!("unmapped operation {other}"),
        };
        if let Some(reason) = case["refused"].as_str() {
            assert_eq!(
                result.map_err(|e| e.0),
                Err(reason),
                "{name} subcase{index}"
            );
        } else {
            assert_eq!(
                result,
                Ok(case["expected"].clone()),
                "{name} subcase{index}"
            );
        }
    }
}

#[test]
fn invalid_dates_offsets_and_fractions_fail_closed() {
    assert_eq!(
        timestamp::parse("2024-02-29T03:00:00.1234569+03:00"),
        timestamp::parse("2024-02-29T00:00:00.123456Z")
    );
    assert!(
        timestamp::parse("2024-02-28T23:00:00-01:00").unwrap()
            < timestamp::parse("2024-02-29T00:00:00.000001Z").unwrap()
    );
    for bad in [
        "2023-02-29T00:00:00Z",
        "2024-04-31T00:00:00Z",
        "2024-01-01T00:00:00",
        "2024-01-01T24:00:00Z",
        "2024-01-01T00:00:60Z",
        "2024-01-01T00:00:00+24:00",
        "2024-01-01T00:00:00.Z",
        "éééé-01-01T00:00:00Z",
    ] {
        assert!(timestamp::parse(bad).is_err(), "{bad}");
    }
    assert!(timestamp::parse(&timestamp::now().unwrap()).is_ok());
}
#[test]
fn booleans_and_nonintegers_never_supply_numeric_identity() {
    for value in [
        json!(true),
        json!(false),
        json!(1.0),
        json!("1"),
        json!(-1),
        Value::Null,
    ] {
        assert!(integer(&value).is_err());
    }
}
#[test]
fn command_line_exclusive_inputs_and_required_source() {
    use clap::Parser;
    assert!(crate::Cli::try_parse_from(["guard", "--base", "b", "--controls-only"]).is_ok());
    for args in [
        vec!["guard", "--base", "b"],
        vec![
            "guard",
            "--base",
            "b",
            "--controls-only",
            "--verify-merge",
            "x",
        ],
        vec!["guard", "--base", "b", "--review-receipt", "x"],
    ] {
        assert!(crate::Cli::try_parse_from(args).is_err());
    }
    let cli = crate::Cli::try_parse_from([
        "guard",
        "--base",
        "b",
        "--pr",
        "1",
        "--source",
        "a",
        "--review-receipt",
        "a/path/that/does/not/exist",
    ])
    .unwrap();
    let result = crate::execute(cli);
    assert_eq!(result["reason"], "receipt_unavailable");
    assert_eq!(result["merge_authorized"], false);
}
