struct Fixture {
    root: PathBuf,
    receipt: Value,
}
fn node(
    root: &Path,
    nodes: &mut Vec<Value>,
    kind: &str,
    data: &[u8],
    requires: Vec<Value>,
) -> Value {
    let path = format!("{}.blob", nodes.len());
    fs::write(root.join(&path), data).unwrap();
    let key = hash(data);
    nodes.push(
        json!({"sha256":key, "path":path, "bytes":data.len(), "kind":kind, "requires":requires}),
    );
    key.into()
}
impl Fixture {
    fn new() -> Self {
        Self::with_role(false)
    }
    fn with_role(confirmation: bool) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
            "target/vnext/rust-evidence-fixtures/{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let mut nodes = vec![];
        let mut identity = Map::new();
        let s = schema(RUN_SCHEMA).unwrap();
        for field in s["$defs"]["EvaluationIdentity"]["properties"]
            .as_object()
            .unwrap()
            .keys()
        {
            if let Some(kind) = field.strip_suffix("_hash") {
                if !matches!(kind, "protocol" | "fit" | "artifact_inventory" | "split") {
                    identity.insert(
                        field.clone(),
                        node(&root, &mut nodes, kind, field.as_bytes(), vec![]),
                    );
                }
            }
        }
        for (field, value) in json!({"partition_role":"selection", "global_seed":7, "generator_fit_seed":11, "sample_seed":13, "learner_seed":17, "fit_rows":3, "requested_rows":4, "produced_rows":4, "consumed_rows":4, "query_rows":3, "context_rows":4, "ensemble_count":1, "precision":"float64", "determinism":"deterministic"}).as_object().unwrap() {
            identity.insert(field.clone(), value.clone());
        }
        if confirmation {
            identity.insert("partition_role".into(), "confirmation".into());
        }
        let split = json!({"format":"dope-split-manifest", "version":1, "data_hash":identity["data_hash"], "query_hash":identity["query_hash"], "partition_role":identity["partition_role"], "fit_rows":identity["fit_rows"], "query_rows":identity["query_rows"]});
        let split_key = node(
            &root,
            &mut nodes,
            "split",
            &canonical_json(&split).unwrap(),
            vec![
                identity["data_hash"].clone(),
                identity["query_hash"].clone(),
            ],
        );
        identity.insert("split_hash".into(), split_key);
        let mut protocol = parse_json(include_bytes!(
            "../../../production/behavior-contract-v1.json"
        ))
        .unwrap();
        let mut commitments = vec![];
        if confirmation {
            let policy = node(
                &root,
                &mut nodes,
                "policy",
                b"fixture-policy-without-real-authorization",
                vec![],
            );
            protocol["status"] = "frozen".into();
            protocol["protocol_frozen"] = true.into();
            protocol["confirmation_allowed"] = true.into();
            for v in protocol["freeze_requirements"]
                .as_object_mut()
                .unwrap()
                .values_mut()
            {
                *v = policy.clone();
            }
            for v in protocol["margins"].as_object_mut().unwrap().values_mut() {
                *v = json!(0.1);
            }
            for v in protocol["budgets"].as_object_mut().unwrap().values_mut() {
                *v = json!(1);
            }
            for (declared, actual) in [
                ("table_schema_hash", "schema_hash"),
                ("task_manifest_hash", "task_hash"),
                ("population_history_hash", "source_family_hash"),
            ] {
                protocol["workload"][declared] = identity[actual].clone();
                commitments.push(identity[actual].clone());
            }
            protocol["learners"]["held_out_families"] = json!(["held_out"]);
            protocol["disclosure"]["protected_unit"] = "row".into();
            protocol["disclosure"]["auxiliary_information_hash"] = policy.clone();
            protocol["disclosure"]["repeated_release_policy_hash"] = policy.clone();
            protocol["disclosure"]["privacy_route"] = "scoped".into();
            protocol["disclosure"]["legal_route"] = "scoped".into();
            protocol["disclosure"]["purpose_grants"] = json!(["customer_modeling"]);
            commitments.push(policy);
        }
        let protocol_bytes = if confirmation {
            canonical_json(&protocol).unwrap()
        } else {
            include_bytes!("../../../production/behavior-contract-v1.json").to_vec()
        };
        identity.insert(
            "protocol_hash".into(),
            node(&root, &mut nodes, "protocol", &protocol_bytes, commitments),
        );
        let state = node(&root, &mut nodes, "fit_state", b"fixture-fit-state", vec![]);
        let mut fit = json!({"format":"dope-generator-fit", "version":1, "state_hash":state});
        let mut fit_requires = vec![state];
        for field in FIT_FIELDS {
            fit[field] = identity[field].clone();
            if field.ends_with("_hash") {
                fit_requires.push(identity[field].clone());
            }
        }
        identity.insert(
            "fit_hash".into(),
            node(
                &root,
                &mut nodes,
                "fit",
                &canonical_json(&fit).unwrap(),
                fit_requires,
            ),
        );
        let artifact = node(
            &root,
            &mut nodes,
            "artifact",
            b"fixture-sampling-artifact",
            vec![],
        );
        let inventory =
            json!({"format":"dope-artifact-inventory", "version":1, "files":[artifact]});
        identity.insert(
            "artifact_inventory_hash".into(),
            node(
                &root,
                &mut nodes,
                "artifact_inventory",
                &canonical_json(&inventory).unwrap(),
                vec![artifact],
            ),
        );
        let paired = json!({"format":"dope-paired-losses", "version":1, "query_hash":identity["query_hash"], "metric_hash":identity["metric_hash"], "loss":"mse", "loss_space":"original_target_units", "loss_unit":"target_units_squared", "real":[0,1,2], "synthetic":[1,2,3], "null":[2,3,4]});
        let paired_key = node(
            &root,
            &mut nodes,
            "paired_evidence",
            &canonical_json(&paired).unwrap(),
            vec![
                identity["query_hash"].clone(),
                identity["metric_hash"].clone(),
            ],
        );
        let executor = node(&root, &mut nodes, "executor", b"fixture-executor", vec![]);
        let freeze_hash = if confirmation {
            let selection = node(
                &root,
                &mut nodes,
                "selection_receipt",
                b"fixture-selection-receipt",
                vec![],
            );
            let access = node(
                &root,
                &mut nodes,
                "access_ledger",
                b"fixture-access-ledger",
                vec![],
            );
            let mut freeze = json!({"format":"dope-freeze", "version":1, "selection_receipt_hash":selection, "executor_hash":executor, "access_ledger_hash":access, "cell_hash":scientific_cache_key(&EvaluationIdentity::new(identity.clone().into()).unwrap()).unwrap(), "parent_freeze_hash":null, "protocol_frozen":true, "confirmation_allowed":true});
            for field in [
                "protocol_hash",
                "artifact_inventory_hash",
                "configuration_hash",
                "split_hash",
                "evaluation_code_hash",
            ] {
                freeze[field] = identity[field].clone();
            }
            node(
                &root,
                &mut nodes,
                "freeze",
                &canonical_json(&freeze).unwrap(),
                vec![selection, access],
            )
        } else {
            Value::Null
        };
        let receipt = json!({"format":"dope-transfer-run", "version":1, "identity":identity, "executor_hash":executor, "freeze_hash":freeze_hash, "dependencies":nodes, "sampling_payload_bytes":b"fixture-sampling-artifact".len(), "paired_evidence_hash":paired_key, "loss":"mse", "loss_space":"original_target_units", "loss_unit":"target_units_squared", "losses":{"real":1.0,"synthetic":2.0,"null":3.0}, "execution":"complete", "attempt":{"attempt_id":"a".repeat(64),"parent_attempt_id":null,"exit_code":0,"wall_time_ms":1,"cpu_time_ms":0,"peak_memory_bytes":1}, "projection":"private", "release":"restricted_local", "transfer":"unmeasured", "empirical_privacy":"unmeasured", "formal_mechanism":"none", "legal_route":"unreviewed"});
        Self { root, receipt }
    }
    fn policy(&self) -> VerificationPolicy {
        VerificationPolicy {
            identity: EvaluationIdentity::new(self.receipt["identity"].clone()).unwrap(),
            executor_hash: text(&self.receipt["executor_hash"]).unwrap().into(),
            receipt_hash: digest(&self.receipt).unwrap(),
            freeze_hash: self.receipt["freeze_hash"].as_str().map(String::from),
        }
    }
    fn verify(&self) -> Result<VerifiedEvaluationReceipt> {
        verify_receipt(
            UnverifiedReceipt::from_bytes(&canonical_json(&self.receipt)?)?,
            &self.root,
            &self.policy(),
        )
    }
}
