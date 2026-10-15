//! Visible server check and review enforcement; no exemption actors are admitted.
use super::{APP_ID, Refused, Result, array, check_set, require, strings};
use serde_json::{Value, json};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy)]
pub enum BasePolicy {
    Strict,
    Queue,
}
#[derive(Debug, Clone, Copy)]
pub struct Policy {
    pub base: BasePolicy,
}
impl Policy {
    pub fn queue(self) -> bool {
        matches!(self.base, BasePolicy::Queue)
    }
    pub fn value(self) -> Value {
        json!({"base_policy":if self.queue() {"queue"} else {"strict"},
            "required_checks":check_set(),"required_check_app_id":APP_ID,
            "independent_review_required":true,"stale_approval_dismissal":true,
            "last_push_approval_required":true,"bypass_policy":"none"})
    }
}
pub fn qualify(controls: &Value) -> Result<Policy> {
    qualify_inner(controls).map_err(|_| Refused("control_unavailable"))
}
fn qualify_inner(controls: &Value) -> Result<Policy> {
    let rules = array(&controls["rules"])?;
    let queue = rules.iter().any(|r| r["type"] == "merge_queue");
    let mut classic = false;
    let protection = &controls["protection"];
    if !protection.is_null() {
        let checks = &protection["required_status_checks"];
        let bindings = array(&checks["checks"])?;
        let bound: BTreeSet<_> = bindings
            .iter()
            .map(|b| super::text(&b["context"]).map(str::to_owned))
            .collect::<Result<_>>()?;
        let contexts = match checks.get("contexts") {
            None | Some(Value::Null) => BTreeSet::new(),
            Some(v) => strings(v)?,
        };
        require(
            bindings.len() == check_set().len()
                && bound == check_set()
                && bindings
                    .iter()
                    .all(|b| b["app_id"].as_u64() == Some(APP_ID))
                && (contexts.is_empty() || contexts == check_set()),
            "control_unavailable",
        )?;
        let review = &protection["required_pull_request_reviews"];
        let allowances = review.get("bypass_pull_request_allowances");
        let no_allowances = match allowances {
            None => true,
            Some(Value::Object(a)) => a.values().all(|v| v.as_array().is_some_and(Vec::is_empty)),
            _ => false,
        };
        classic = checks["strict"] == true
            && review["required_approving_review_count"]
                .as_u64()
                .is_some_and(|n| n >= 1)
            && review["dismiss_stale_reviews"] == true
            && review["require_last_push_approval"] == true
            && protection["enforce_admins"]["enabled"] == true
            && no_allowances
            && protection["allow_force_pushes"]["enabled"] == false
            && protection["allow_deletions"]["enabled"] == false;
    }
    let mut rule_qualified = false;
    if !rules.is_empty() {
        let mut all_contexts = BTreeSet::new();
        let mut all_strict = true;
        let mut reviews = false;
        for rule in rules {
            let id = super::integer(&rule["ruleset_id"])?;
            require(id > 0, "control_unavailable")?;
            let detail = &controls["rulesets"][id.to_string()];
            require(
                detail["id"].as_u64() == Some(id)
                    && detail["enforcement"] == "active"
                    && detail["bypass_actors"]
                        .as_array()
                        .is_some_and(Vec::is_empty),
                "control_unavailable",
            )?;
            let parameters = &rule["parameters"];
            if rule["type"] == "required_status_checks" {
                let bindings = array(&parameters["required_status_checks"])?;
                require(
                    bindings
                        .iter()
                        .all(|b| b["integration_id"].as_u64() == Some(APP_ID)),
                    "control_unavailable",
                )?;
                let names: BTreeSet<_> = bindings
                    .iter()
                    .map(|b| super::text(&b["context"]).map(str::to_owned))
                    .collect::<Result<_>>()?;
                require(
                    names.is_subset(&check_set()) && names.len() == bindings.len(),
                    "control_unavailable",
                )?;
                all_contexts.extend(names);
                all_strict &= parameters["strict_required_status_checks_policy"] == true;
            }
            if rule["type"] == "pull_request" {
                reviews |= parameters["required_approving_review_count"]
                    .as_u64()
                    .is_some_and(|n| n >= 1)
                    && parameters["dismiss_stale_reviews_on_push"] == true
                    && parameters["require_last_push_approval"] == true;
            }
        }
        rule_qualified = reviews && all_contexts == check_set() && (all_strict || queue);
    }
    require(rule_qualified || (classic && !queue), "control_unavailable")?;
    Ok(Policy {
        base: if queue {
            BasePolicy::Queue
        } else {
            BasePolicy::Strict
        },
    })
}
