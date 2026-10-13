fn role_of(value: &str) -> Result<Role, BundleError> {
    match value {
        "dictionary" => Ok(Role::Dictionary),
        "evidence_authorization" => Ok(Role::Evidence),
        "materialized_rows" => Ok(Role::Rows),
        "runtime" => Ok(Role::Runtime),
        "sampling_payload" => Ok(Role::Payload),
        "shared_learned_state" => Ok(Role::Shared),
        "transform" => Ok(Role::Transform),
        _ => Err(err("unknown_role")),
    }
}

fn kind_of(value: &str) -> Result<Kind, BundleError> {
    match value {
        "dpk2" => Ok(Kind::Dpk2),
        "dpk3" => Ok(Kind::Dpk3),
        "opaque" => Ok(Kind::Opaque),
        _ => Err(err("unknown_kind")),
    }
}

fn sorted_identifiers(value: &Json) -> Result<Vec<String>, BundleError> {
    let Json::Array(values) = value else {
        return Err(err("schema_mismatch"));
    };
    let mut names = Vec::new();
    for value in values {
        let name = json_str(value)?;
        identifier(name)?;
        names.push(name.to_string());
    }
    let mut ordered = names.clone();
    ordered.sort();
    ordered.dedup();
    if ordered != names {
        return Err(err("schema_mismatch"));
    }
    Ok(names)
}

fn parse_component(value: &Json) -> Result<Component, BundleError> {
    let object = object(value)?;
    exact_keys(
        object,
        &[
            "bytes",
            "consumed_by_sampler",
            "depends_on",
            "executable",
            "handler",
            "id",
            "kind",
            "path",
            "required",
            "role",
            "sha256",
        ],
    )?;
    let id = json_str(&object["id"])?.to_string();
    identifier(&id)?;
    let path = json_str(&object["path"])?.to_string();
    component_path(&path)?;
    let digest = json_str(&object["sha256"])?.to_string();
    sha256_hex(&digest)?;
    if json_bool(&object["executable"])? {
        return Err(err("executable_rejected"));
    }
    if json_str(&object["handler"])? != "none" {
        return Err(err("unsupported_external_handler"));
    }
    Ok(Component {
        id,
        role: role_of(json_str(&object["role"])?)?,
        path,
        sha256: digest,
        bytes: json_u64(&object["bytes"], "schema_mismatch")?,
        kind: kind_of(json_str(&object["kind"])?)?,
        required: json_bool(&object["required"])?,
        consumed_by_sampler: json_bool(&object["consumed_by_sampler"])?,
        depends_on: sorted_identifiers(&object["depends_on"])?,
        measured_bytes: 0,
    })
}

fn closure(start: &str, edges: &BTreeMap<String, Vec<String>>) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    let mut stack = vec![start.to_string()];
    while let Some(node) = stack.pop() {
        if !seen.insert(node.clone()) {
            continue;
        }
        if let Some(children) = edges.get(&node) {
            stack.extend(children.iter().cloned());
        }
    }
    seen
}

fn check_graph(components: &[Component], handler: SamplingHandler) -> Result<(), BundleError> {
    let mut edges: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut lowered: BTreeMap<String, String> = BTreeMap::new();
    for component in components {
        if edges.contains_key(&component.id) {
            return Err(err("duplicate_component"));
        }
        let key = component.path.to_ascii_lowercase();
        if let Some(prior) = lowered.get(&key) {
            return Err(err(if prior != &component.path {
                "case_collision"
            } else {
                "duplicate_component"
            }));
        }
        lowered.insert(key, component.path.clone());
        edges.insert(component.id.clone(), Vec::new());
    }
    for component in components {
        for dependency in &component.depends_on {
            if !edges.contains_key(dependency) {
                return Err(err("unbound_dependency"));
            }
            if dependency == &component.id {
                return Err(err("cyclic_dependency"));
            }
            if let Some(children) = edges.get_mut(&component.id) {
                children.push(dependency.clone());
            }
        }
    }
    let mut state: BTreeMap<&str, u8> = edges.keys().map(|key| (key.as_str(), 0)).collect();
    fn visit<'a>(
        identity: &'a str,
        edges: &'a BTreeMap<String, Vec<String>>,
        state: &mut BTreeMap<&'a str, u8>,
    ) -> Result<(), BundleError> {
        state.insert(identity, 1);
        for child in &edges[identity] {
            match state[child.as_str()] {
                1 => return Err(err("cyclic_dependency")),
                0 => visit(child, edges, state)?,
                _ => {}
            }
        }
        state.insert(identity, 2);
        Ok(())
    }
    let identities: Vec<String> = edges.keys().cloned().collect();
    for identity in &identities {
        if state[identity.as_str()] == 0 {
            visit(identity, &edges, &mut state)?;
        }
    }
    let payloads: Vec<&Component> = components
        .iter()
        .filter(|component| component.role == Role::Payload)
        .collect();
    match handler {
        SamplingHandler::Native => {
            if payloads.len() != 1 || !payloads[0].required || !payloads[0].consumed_by_sampler {
                return Err(err("sampler_closure_mismatch"));
            }
            if !matches!(payloads[0].kind, Kind::Dpk2 | Kind::Dpk3) {
                return Err(err("sampler_closure_mismatch"));
            }
            if components
                .iter()
                .any(|component| component.consumed_by_sampler && component.id != payloads[0].id)
            {
                return Err(err("sampler_closure_mismatch"));
            }
            let reachable = closure(&payloads[0].id, &edges);
            if components
                .iter()
                .any(|component| component.required && !reachable.contains(&component.id))
            {
                return Err(err("unresolved_dependency"));
            }
        }
        SamplingHandler::Inventory => {
            if components
                .iter()
                .any(|component| component.consumed_by_sampler || component.kind != Kind::Opaque)
            {
                return Err(err("sampler_closure_mismatch"));
            }
            if !payloads.is_empty() {
                let mut reachable = BTreeSet::new();
                for payload in payloads {
                    reachable.extend(closure(&payload.id, &edges));
                }
                if components
                    .iter()
                    .any(|component| component.required && !reachable.contains(&component.id))
                {
                    return Err(err("unresolved_dependency"));
                }
            }
        }
    }
    Ok(())
}

fn role_total(components: &[Component], role: Role) -> Result<u64, BundleError> {
    components
        .iter()
        .filter(|component| component.role == role)
        .try_fold(0u64, |total, component| {
            checked_add(total, component.measured_bytes)
        })
}

fn account_inventory(
    components: &[Component],
    manifest_bytes: u64,
    amortization_count: u64,
    transport: TransportFraming,
) -> Result<SamplingAccounting, BundleError> {
    if !(1..=1_000_000).contains(&amortization_count) {
        return Err(err("invalid_amortization"));
    }
    let payload = role_total(components, Role::Payload)?;
    let shared = role_total(components, Role::Shared)?;
    let transform = role_total(components, Role::Transform)?;
    let dictionary = role_total(components, Role::Dictionary)?;
    let runtime = role_total(components, Role::Runtime)?;
    let evidence = role_total(components, Role::Evidence)?;
    let rows = role_total(components, Role::Rows)?;
    let learned = [payload, shared, transform, dictionary, manifest_bytes]
        .into_iter()
        .try_fold(0u64, checked_add)?;
    let shared_share = shared / amortization_count + u64::from(shared % amortization_count != 0);
    let amortized = [payload, transform, dictionary, manifest_bytes, shared_share]
        .into_iter()
        .try_fold(0u64, checked_add)?;
    let conditional = components
        .iter()
        .filter(|component| {
            component.required && matches!(component.role, Role::Evidence | Role::Rows)
        })
        .try_fold(0u64, |total, component| {
            checked_add(total, component.measured_bytes)
        })?;
    let charged = checked_add(checked_add(learned, runtime)?, conditional)?;
    let physical = components.iter().try_fold(0u64, |total, component| {
        checked_add(total, component.measured_bytes)
    })?;
    let complete_delivery_bytes = if transport == TransportFraming::Included {
        Some(checked_add(manifest_bytes, physical)?)
    } else {
        None
    };
    Ok(SamplingAccounting {
        sampling_payload_bytes: payload,
        shared_learned_state_bytes: shared,
        standalone_learned_state_bytes: learned,
        amortized_learned_state_bytes: amortized,
        runtime_package_bytes: runtime,
        evidence_authorization_bytes: evidence,
        materialized_rows_bytes: rows,
        transform_bytes: transform,
        dictionary_bytes: dictionary,
        manifest_bytes,
        charged_sampling_bytes: charged,
        amortization_count,
        transport_framing: transport,
        complete_delivery_bytes,
    })
}

fn parse_manifest(
    value: &Json,
    limits: &BundleLimits,
) -> Result<(SamplingHandler, u64, TransportFraming, Vec<Component>), BundleError> {
    let object = object(value)?;
    exact_keys(
        object,
        &[
            "amortization_count",
            "bundle_id",
            "capabilities",
            "components",
            "format",
            "sampling_handler",
            "transport_framing",
            "version",
        ],
    )?;
    if json_str(&object["format"])? != "dope.generator-bundle/v1"
        || json_u64(&object["version"], "unknown_version")? != 1
    {
        return Err(err("unknown_version"));
    }
    identifier(json_str(&object["bundle_id"])?)?;
    let handler = match json_str(&object["sampling_handler"])? {
        "inventory-only" => SamplingHandler::Inventory,
        "native-inline-dpk" => SamplingHandler::Native,
        _ => return Err(err("unknown_handler")),
    };
    let Json::Array(capabilities) = &object["capabilities"] else {
        return Err(err("schema_mismatch"));
    };
    let capability_names: Vec<&str> = capabilities
        .iter()
        .map(json_str)
        .collect::<Result<_, _>>()?;
    if capability_names
        .iter()
        .any(|name| !matches!(*name, "byte-inventory" | "native-dpk-sample"))
    {
        return Err(err("unknown_capability"));
    }
    let expected: &[&str] = if handler == SamplingHandler::Native {
        &["byte-inventory", "native-dpk-sample"]
    } else {
        &["byte-inventory"]
    };
    if capability_names != expected {
        return Err(err("sampler_closure_mismatch"));
    }
    let count = json_u64(&object["amortization_count"], "invalid_amortization")?;
    if !(1..=1_000_000).contains(&count) {
        return Err(err("invalid_amortization"));
    }
    let transport = match json_str(&object["transport_framing"])? {
        "unmeasured" => TransportFraming::Unmeasured,
        "included" => TransportFraming::Included,
        _ => return Err(err("schema_mismatch")),
    };
    let Json::Array(components) = &object["components"] else {
        return Err(err("schema_mismatch"));
    };
    if components.len() > limits.max_components {
        return Err(err("too_many_components"));
    }
    let parsed = components
        .iter()
        .map(parse_component)
        .collect::<Result<Vec<_>, _>>()?;
    let identities: Vec<&str> = parsed
        .iter()
        .map(|component| component.id.as_str())
        .collect();
    let mut ordered = identities.clone();
    ordered.sort_unstable();
    if identities != ordered {
        return Err(err("schema_mismatch"));
    }
    check_graph(&parsed, handler)?;
    Ok((handler, count, transport, parsed))
}

pub fn verify_bundle(
    root: &Path,
    expected_manifest_sha256: &str,
    limits: &BundleLimits,
) -> Result<VerifiedBundle, BundleError> {
    sha256_hex(expected_manifest_sha256).map_err(|_| err("manifest_digest_mismatch"))?;
    let manifest_bytes = read_relative(root, MANIFEST_NAME, limits.max_manifest_bytes as u64)
        .map_err(|error| {
            if error.code == "dependency_too_large" {
                err("manifest_too_large")
            } else {
                error
            }
        })?;
    if manifest_bytes.len() > limits.max_manifest_bytes {
        return Err(err("manifest_too_large"));
    }
    let manifest = parse_json_subset(&manifest_bytes)?;
    let canonical = canonical_subset(&manifest)?;
    if canonical != manifest_bytes {
        return Err(err("manifest_not_canonical"));
    }
    let digest = hex(&sha256(&canonical));
    if digest != expected_manifest_sha256 {
        return Err(err("manifest_digest_mismatch"));
    }
    let (handler, count, transport, mut components) = parse_manifest(&manifest, limits)?;
    let mut declared = BTreeSet::from([MANIFEST_NAME.to_string()]);
    for component in &components {
        declared.insert(component.path.clone());
    }
    let max_entries = limits.max_components.saturating_add(1);
    inventory(
        root,
        &declared,
        WalkBudget {
            max_entries,
            max_name_bytes: max_entries.saturating_mul(64),
        },
    )?;
    let mut payload = None;
    let mut features = None;
    let mut total = manifest_bytes.len() as u64;
    for component in &mut components {
        let bytes = read_relative(root, &component.path, limits.max_component_bytes as u64)
            .map_err(|error| {
                if error.code == "dependency_too_large" {
                    err("component_cap_exceeded")
                } else {
                    error
                }
            })?;
        if bytes.len() > limits.max_component_bytes {
            return Err(err("component_cap_exceeded"));
        }
        let file_digest = hex(&sha256(&bytes));
        if file_digest != component.sha256 || bytes.len() as u64 != component.bytes {
            return Err(err("dependency_identity_mismatch"));
        }
        component.measured_bytes = bytes.len() as u64;
        total = checked_add(total, component.measured_bytes)?;
        if component.role == Role::Payload && handler == SamplingHandler::Native {
            let summary = preflight_dpk(&bytes, limits)?;
            let expects_v3 = component.kind == Kind::Dpk3;
            if summary.v3 != expects_v3 {
                return Err(err("sampler_closure_mismatch"));
            }
            features = Some(summary.features);
            payload = Some(bytes);
        }
    }
    if total > limits.max_total_bytes as u64 {
        return Err(err("total_bytes_exceeded"));
    }
    let accounting = account_inventory(&components, manifest_bytes.len() as u64, count, transport)?;
    Ok(VerifiedBundle {
        accounting,
        payload,
        features,
        handler,
    })
}
