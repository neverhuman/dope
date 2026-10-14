//! GET collectors and immutable before/after observations.
use super::{
    Refused, Result, array, evidence,
    github::{Api, quote},
    integer, require,
    runs::WORKFLOWS,
    text, timestamp,
};
use serde_json::{Value, json};

pub fn controls(api: &mut impl Api, root: &str, branch: &str) -> Result<Value> {
    (|| {
        let protection = api.get(
            &format!("{root}/branches/{}/protection", quote(branch)),
            true,
            false,
        )?;
        let rules = api.listing(&format!("{root}/rules/branches/{}", quote(branch)), None)?;
        let mut details = json!({});
        for rule in array(&rules)? {
            let id = integer(&rule["ruleset_id"])?;
            require(id > 0, "control_unavailable")?;
            let key = id.to_string();
            if details.get(&key).is_none() {
                details[&key] = api.get(
                    &format!("{root}/rulesets/{id}?includes_parents=true"),
                    false,
                    false,
                )?;
            }
        }
        Ok(json!({"protection":protection,"rules":rules,"rulesets":details}))
    })()
    .map_err(|_: Refused| Refused("control_unavailable"))
}
pub fn snapshot(api: &mut impl Api, root: &str, pr: u64, branch: &str) -> Result<Value> {
    let fetched = api.get(&format!("{root}/pulls/{pr}"), false, false)?;
    let mut pull = json!({});
    for key in [
        "number",
        "state",
        "draft",
        "head",
        "base",
        "user",
        "commits",
        "merged",
        "merged_at",
        "merge_commit_sha",
        "mergeable_state",
    ] {
        pull[key] = fetched
            .get(key)
            .ok_or(Refused("invalid_or_incomplete_evidence"))?
            .clone();
    }
    let base = api.get(
        &format!("{root}/git/ref/heads/{}", quote(branch)),
        false,
        false,
    )?;
    let base = base["object"]["sha"].clone();
    require(super::valid_sha(&base), "invalid_or_incomplete_evidence")?;
    let controls = controls(api, root, branch)?;
    let pulls = api.open_pulls(root)?;
    Ok(
        json!({"pull":pull,"base_sha":base,"controls":controls,"open_prs":evidence::normalize_open_pulls(&pulls)?}),
    )
}
fn base64(value: &str) -> Result<String> {
    let bytes: Vec<_> = value.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    require(bytes.len() % 4 == 0, "workflow_definition_invalid")?;
    let mut out = Vec::new();
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        let mut bits = 0_u32;
        let mut pad = 0;
        for (j, b) in chunk.iter().enumerate() {
            let value = match *b {
                b'A'..=b'Z' => b - b'A',
                b'a'..=b'z' => b - b'a' + 26,
                b'0'..=b'9' => b - b'0' + 52,
                b'+' => 62,
                b'/' => 63,
                b'=' => {
                    pad += 1;
                    0
                }
                _ => return Err(Refused("workflow_definition_invalid")),
            };
            require(
                pad == 0 || (i == bytes.len() / 4 - 1 && j >= 2 && *b == b'='),
                "workflow_definition_invalid",
            )?;
            bits = (bits << 6) | u32::from(value);
        }
        require(pad <= 2, "workflow_definition_invalid")?;
        out.push((bits >> 16) as u8);
        if pad < 2 {
            out.push((bits >> 8) as u8);
        }
        if pad == 0 {
            out.push(bits as u8);
        }
    }
    String::from_utf8(out).map_err(|_| Refused("workflow_definition_invalid"))
}
pub fn definitions(api: &mut impl Api, root: &str, source: &str) -> Result<Value> {
    let mut definitions = json!({});
    for (path, _) in WORKFLOWS {
        let item = api.get(
            &format!("{root}/contents/{path}?ref={source}"),
            false,
            false,
        )?;
        require(item["encoding"] == "base64", "workflow_definition_invalid")?;
        definitions[path] = json!(base64(text(&item["content"])?)?);
    }
    Ok(definitions)
}
pub fn runs(api: &mut impl Api, root: &str, source: &str) -> Result<Value> {
    let fetched = api.listing(
        &format!("{root}/actions/runs?head_sha={source}"),
        Some("workflow_runs"),
    )?;
    let mut collected = Vec::new();
    for run in array(&fetched)? {
        let path = run
            .get("path")
            .and_then(Value::as_str)
            .ok_or(Refused("invalid_or_incomplete_evidence"))?;
        if WORKFLOWS
            .iter()
            .any(|(p, _)| *p == path.split('@').next().unwrap_or(""))
        {
            let mut run = run.clone();
            let id = integer(&run["id"])?;
            run["jobs"] = api.listing(
                &format!("{root}/actions/runs/{id}/jobs?filter=all"),
                Some("jobs"),
            )?;
            collected.push(run);
        }
    }
    Ok(json!(collected))
}
pub fn preflight(api: &mut impl Api, pins: &Value) -> Result<Value> {
    let root = format!("repos/{}", text(&pins["repo"])?);
    let source = text(&pins["source_sha"])?;
    let pr = integer(&pins["pr"])?;
    let branch = text(&pins["base_branch"])?;
    let before = snapshot(api, &root, pr, branch)?;
    let operator = api.get("user", false, false)?;
    let commit = api.get(&format!("{root}/git/commits/{source}"), false, false)?;
    let permission = api.get(
        &format!("{root}/collaborators/{}/permission", super::REVIEWER),
        false,
        false,
    )?;
    let commits = api.listing(&format!("{root}/pulls/{pr}/commits"), None)?;
    let reviews = api.listing(&format!("{root}/pulls/{pr}/reviews"), None)?;
    let definitions = definitions(api, &root, source)?;
    let runs = runs(api, &root, source)?;
    require(
        reviews == api.listing(&format!("{root}/pulls/{pr}/reviews"), None)?,
        "inspection_changed",
    )?;
    require(
        runs == self::runs(api, &root, source)?,
        "inspection_changed",
    )?;
    let after = snapshot(api, &root, pr, branch)?;
    Ok(
        json!({"before":before,"operator":operator,"source_commit":commit,"reviewer_permission":permission,"commits":commits,
        "reviews":reviews,"workflow_definitions":definitions,"runs":runs,"after":after,"inspection_completed_at":timestamp::now()?}),
    )
}
