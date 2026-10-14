//! Advisory receipts for controls, preflight and actual merge observations.
use super::{
    Refused, Result, collect, controls, evidence, github::Api, integer, json::digest, require,
    text, timestamp, validate_pins,
};
use serde_json::{Value, json};

fn finish(result: &mut Value, outcome: Result<&str>, calls: Value) {
    match outcome {
        Ok(state) => result["state"] = json!(state),
        Err(Refused(reason)) => {
            result["reason"] = json!(reason);
            if reason == "control_unavailable" {
                result["state"] = json!(reason);
            }
        }
    }
    result["api_calls"] = calls;
}
fn initial() -> Result<Value> {
    Ok(
        json!({"version":1,"read_only":true,"merge_authorized":false,"state":"blocked","captured_at":timestamp::now()?}),
    )
}
fn clock_failure() -> Value {
    json!({"version":1,"read_only":true,"merge_authorized":false,"state":"blocked","reason":"invalid_timestamp","api_calls":[]})
}
pub fn inspect_controls(api: &mut impl Api, repo: &str, base: &str) -> Value {
    let mut result = match initial() {
        Ok(v) => v,
        Err(_) => return clock_failure(),
    };
    result["repo"] = json!(repo);
    result["base_sha"] = json!(base);
    let outcome = (|| {
        validate_pins(
            &json!({"repo":repo,"pr":1,"base_branch":"main","source_sha":base,"base_sha":base}),
        )?;
        let root = format!("repos/{repo}");
        result["operator"] = api.get("user", false, false)?;
        result["before_base"] =
            api.get(&format!("{root}/git/ref/heads/main"), false, false)?["object"]["sha"].clone();
        result["before_controls"] = collect::controls(api, &root, "main")?;
        result["after_controls"] = collect::controls(api, &root, "main")?;
        result["after_base"] =
            api.get(&format!("{root}/git/ref/heads/main"), false, false)?["object"]["sha"].clone();
        require(
            result["before_base"] == base && result["after_base"] == base,
            "base_moved",
        )?;
        require(
            result["before_controls"] == result["after_controls"],
            "controls_changed",
        )?;
        result["policy"] = controls::qualify(&result["after_controls"])?.value();
        Ok("server_policy_observed")
    })();
    finish(&mut result, outcome, api.calls());
    result
}
pub fn inspect(api: &mut impl Api, pins: &Value, receipt: &Value, prior: Option<&Value>) -> Value {
    let mut result = match initial() {
        Ok(v) => v,
        Err(_) => return clock_failure(),
    };
    result["pins"] = pins.clone();
    let outcome = (|| {
        validate_pins(pins)?;
        if let Some(prior) = prior {
            require(prior["pins"] == *pins, "receipt_identity_mismatch")?;
            evidence::evaluate_preflight(&prior["evidence"], pins, &prior["review_receipt"])?;
            let root = format!("repos/{}", text(&pins["repo"])?);
            let pr = integer(&pins["pr"])?;
            let branch = text(&pins["base_branch"])?;
            let before = collect::snapshot(api, &root, pr, branch)?;
            result["prior_receipt_sha256"] = json!(digest(prior)?);
            result["evidence"] = json!({"before":before,"reviews":api.listing(&format!("{root}/pulls/{pr}/reviews"),None)?});
            let evidence = &mut result["evidence"];
            require(
                evidence["before"]["pull"]["merged"].is_boolean(),
                "invalid_or_incomplete_evidence",
            )?;
            if evidence["before"]["pull"]["merged"] == true {
                let merge = &evidence["before"]["pull"]["merge_commit_sha"];
                require(super::valid_sha(merge), "merge_not_verified")?;
                let merge = text(merge)?.to_owned();
                evidence["merge_commit"] =
                    api.get(&format!("{root}/git/commits/{merge}"), false, false)?;
                evidence["receiving_runs"] = collect::runs(api, &root, &merge)?;
            }
            evidence["after"] = collect::snapshot(api, &root, pr, branch)?;
            evidence["reviews_after"] = api.listing(&format!("{root}/pulls/{pr}/reviews"), None)?;
            evidence["inspection_completed_at"] = json!(timestamp::now()?);
            evidence::evaluate_merge(evidence, prior)
        } else {
            result["review_receipt"] = receipt.clone();
            result["review_receipt_sha256"] = json!(digest(receipt)?);
            let collected = collect::preflight(api, pins)?;
            result["captured_at"] = collected["inspection_completed_at"].clone();
            result["evidence"] = collected;
            result["policy"] =
                evidence::evaluate_preflight(&result["evidence"], pins, receipt)?.value();
            Ok("ready_for_operator")
        }
    })();
    finish(&mut result, outcome, api.calls());
    result
}
