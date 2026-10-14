//! Permanent feature isolation proof; no fixed root-lock hash or online resolve.
use super::super::{
    Value,
    github::{ProcessTransport, Transport},
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_owned()
}
fn metadata(manifest: &Path) -> Value {
    let args = vec![
        "cargo".into(),
        "metadata".into(),
        "--offline".into(),
        "--locked".into(),
        "--manifest-path".into(),
        manifest.to_str().unwrap().into(),
        "--format-version".into(),
        "1".into(),
    ];
    let response = ProcessTransport::default()
        .run(&args)
        .expect("bounded offline metadata failed");
    assert_eq!(response.code, 0, "locked metadata failed");
    serde_json::from_str(&response.stdout).expect("invalid metadata JSON")
}
fn graphs() -> &'static (Value, Value) {
    static GRAPHS: OnceLock<(Value, Value)> = OnceLock::new();
    GRAPHS.get_or_init(|| {
        (
            metadata(&root().join("Cargo.toml")),
            metadata(&Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml")),
        )
    })
}
fn list(v: &Value) -> Result<&Vec<Value>, &'static str> {
    v.as_array().ok_or("invalid metadata")
}
fn ids(v: &Value) -> Result<BTreeSet<&str>, &'static str> {
    list(v)?
        .iter()
        .map(|x| x.as_str().ok_or("invalid metadata"))
        .collect()
}
fn feature_boundary(root: &Value, tool: &Value) -> Result<(), &'static str> {
    let root_members = ids(&root["workspace_members"])?;
    let tool_members = ids(&tool["workspace_members"])?;
    if tool_members.len() != 1
        || !root_members.is_disjoint(&tool_members)
        || root["workspace_root"] == tool["workspace_root"]
    {
        return Err("workspace membership violates isolation");
    }
    let root_packages = list(&root["packages"])?;
    let tool_packages = list(&tool["packages"])?;
    if root_packages
        .iter()
        .any(|p| tool_members.contains(p["id"].as_str().unwrap_or("")))
        || tool_packages
            .iter()
            .any(|p| root_members.contains(p["id"].as_str().unwrap_or("")))
    {
        return Err("cross-package dependency violates isolation");
    }
    for (graph, expected) in [(root, false), (tool, true)] {
        let package_ids: BTreeSet<_> = list(&graph["packages"])?
            .iter()
            .filter(|p| p["name"] == "serde_json")
            .map(|p| p["id"].as_str().ok_or("invalid metadata"))
            .collect::<Result<_, _>>()?;
        let nodes: Vec<_> = list(&graph["resolve"]["nodes"])?
            .iter()
            .filter(|n| package_ids.contains(n["id"].as_str().unwrap_or("")))
            .collect();
        if nodes.is_empty() {
            return Err("missing serde_json graph");
        }
        for node in nodes {
            let features = ids(&node["features"])?;
            if features.contains("float_roundtrip") != expected {
                return Err("serde_json feature crossed boundary");
            }
        }
    }
    Ok(())
}
type RegistryPin = (String, String, String, String);
fn registry_pins(lock: &str) -> Result<BTreeSet<RegistryPin>, &'static str> {
    let mut result = BTreeSet::new();
    for block in lock.split("[[package]]").skip(1) {
        let mut fields = BTreeMap::new();
        for line in block.lines() {
            if let Some((key, value)) = line.split_once(" = ") {
                if ["name", "version", "source", "checksum"].contains(&key) {
                    let parsed: String =
                        serde_json::from_str(value).map_err(|_| "invalid generated lock")?;
                    fields.insert(key, parsed);
                }
            }
        }
        if let Some(source) = fields.get("source") {
            if !source.starts_with("registry+") {
                return Err("non-registry dependency");
            }
            let value = |key| fields.get(key).cloned().ok_or("incomplete registry pin");
            result.insert((
                value("name")?,
                value("version")?,
                source.clone(),
                value("checksum")?,
            ));
        }
    }
    if result.is_empty() {
        return Err("empty lock");
    }
    Ok(result)
}
fn lock_boundary(
    root: &BTreeSet<RegistryPin>,
    tool: &BTreeSet<RegistryPin>,
) -> Result<(), &'static str> {
    if tool.is_subset(root) {
        Ok(())
    } else {
        Err("tool dependency differs from root-locked inventory")
    }
}
#[test]
fn current_feature_graphs_and_locked_versions_are_isolated() {
    let (root_graph, tool_graph) = graphs();
    assert_eq!(feature_boundary(root_graph, tool_graph), Ok(()));
    let root_lock = std::fs::read_to_string(root().join("Cargo.lock")).unwrap();
    let tool_lock =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock")).unwrap();
    assert_eq!(
        lock_boundary(
            &registry_pins(&root_lock).unwrap(),
            &registry_pins(&tool_lock).unwrap()
        ),
        Ok(())
    );
}
#[test]
fn feature_membership_and_lock_negative_cases_fail_closed() {
    let (root_graph, tool_graph) = graphs();
    let root_serde_id = root_graph["packages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["name"] == "serde_json")
        .unwrap()["id"]
        .clone();
    let mut leaked = root_graph.clone();
    let node = leaked["resolve"]["nodes"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|n| n["id"] == root_serde_id)
        .unwrap();
    node["features"]
        .as_array_mut()
        .unwrap()
        .push(Value::String("float_roundtrip".into()));
    assert!(feature_boundary(&leaked, tool_graph).is_err());
    let mut missing = tool_graph.clone();
    for node in missing["resolve"]["nodes"].as_array_mut().unwrap() {
        node["features"]
            .as_array_mut()
            .unwrap()
            .retain(|f| f != "float_roundtrip");
    }
    assert!(feature_boundary(root_graph, &missing).is_err());
    let mut member = root_graph.clone();
    member["workspace_members"]
        .as_array_mut()
        .unwrap()
        .push(tool_graph["workspace_members"][0].clone());
    assert!(feature_boundary(&member, tool_graph).is_err());
    let mut root_depends = root_graph.clone();
    let own_tool = tool_graph["packages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["id"] == tool_graph["workspace_members"][0])
        .unwrap()
        .clone();
    root_depends["packages"]
        .as_array_mut()
        .unwrap()
        .push(own_tool);
    assert!(feature_boundary(&root_depends, tool_graph).is_err());
    let mut tool_depends = tool_graph.clone();
    let own_root = root_graph["packages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["id"] == root_graph["workspace_members"][0])
        .unwrap()
        .clone();
    tool_depends["packages"]
        .as_array_mut()
        .unwrap()
        .push(own_root);
    assert!(feature_boundary(root_graph, &tool_depends).is_err());
    let root_lock =
        registry_pins(&std::fs::read_to_string(root().join("Cargo.lock")).unwrap()).unwrap();
    let tool_lock = registry_pins(
        &std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock")).unwrap(),
    )
    .unwrap();
    let mut mismatch = tool_lock.clone();
    let pin = mismatch.iter().next().unwrap().clone();
    mismatch.remove(&pin);
    mismatch.insert((pin.0, pin.1, pin.2, "wrong checksum".into()));
    assert!(lock_boundary(&root_lock, &mismatch).is_err());
    let mut future_root = root_lock.clone();
    future_root.insert((
        "future-authorized".into(),
        "1.0.0".into(),
        "registry+fixture".into(),
        "fixture checksum".into(),
    ));
    assert_eq!(lock_boundary(&future_root, &tool_lock), Ok(()));
}
