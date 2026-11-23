# DOPE 3.0 independent verification

This handoff implements the independent verification lane of approved plan SHA-256 `b1563c1031adb936ac75af3e61b53c91995cf153c2b4b357aecc66cc419bcc6b`. The lane starts at receiving main `88b157779be680ab596a0ae36a9e0ff5e5fe3aba` (#399). The release sequence is explicit analytic DSE8 at 2,048 complete bytes, an equal-byte incremental gate for any learned compiler, and separate admission for automatic format selection.

The integration handoff is rebased onto `a3e9a9868c61a168a659c9324ad302be0feaecd4` (#400). Two files inherited from that commit exceeded the tracked source limit. The exact-marginal copula and its oracle tests now live in separate modules, and the existing training ring lives beside the episode factory. The rebased oracle and factory suites exercise the moved behavior.

- The [evidence crosswalk](evidence-crosswalk.md) maps the six proposals and preserves the existing landmark identities. Reconcile the draft aliases against the frozen extractor registry before binding checkpoints or receipts.
- The [published-method benchmark](published-method-benchmark.md) records the retained public paired replay and the next method roster.
- The pinned baseline inputs are `contracts/dope-kpi.json` SHA-256 `200ea0ed1ba23a25be89de2dac2a4101656119514e0d69b4ff86a98fa4857623`, `engine/Cargo.lock` SHA-256 `34b80b5558e93475ca0e76fed437ee50aee3eb0d094bd9493bdce8798e105a80` and the 2,755,388 B [released V7 checkpoint](../../../../models/single-enc-v7/manifest.json) SHA-256 `d013eb0cb08c9901c39e2bd65232c6bfbc41f68a613797971f7afb525b70cfac`.
- H2 milestone file SHA-256 `a00c6788f5bd78ceafac1eeffd00c627ec35de65fd688a31ed682a0f19ca4f8d` records analytic V7, round-108 and the later soup as different arms. The local round-108 full checkpoint verifies at `ca115356ff511a12be131c568c8022d6fcae83a41aad860b379b47f80b811501`. Its retained screening schedule and real reference verify at `9207d1575415e5f3f82f285c6473378885ead675b55ddcb89850755422d3909f` and `3e89330eb44c26b09ba6c266688bf34cff769ffc0f13e72eb814a9b1fb2ea31d`. The exact later soup still needs its full checkpoint digest, executable and run binding. [H2 milestones](../../../../H2/metrics.json)

The retained round-108 screen scored **−0.415029 TRG** against **−0.410784** for the retained `v72-step0` arm. The paired 4× difference was **−0.004245** [−0.045155, +0.039275] on 12 cells and 10 task-family units. Both arms used 782.5 B median. This reproduces the retained screening log and identifies the separate replay required for H2’s reported gain against `v73_step0`. The paired and audit receipts are under `target/v3-verification/h2-replay/`.

## Native receipt audit

```sh
cargo build --manifest-path engine/Cargo.toml --target-dir target --locked -p dope-arena --bin dope_kpi
target/debug/dope_kpi audit --run-dir RUN --receipt RECEIPT.json
```

The native ingress checks the schedule, real reference, ledger key coverage and per-row metadata. The audit checks their hashes, the family partition and bootstrap parameters against the receipt, then independently reconstructs 4× and 1× TRG denominators, per-task and paired points, and headline byte totals from the score rows. The output gives v2 estimates, `clip_low − R_B` failure-floor points and unclipped points under the v2 failure penalty. Send `--dev-tables` with the exact partition list for a DEV or confirmation receipt. The committed v2.0.0 receipts predate the explicit auditor roster and need `--legacy-v2` to replay their historical control count.

Pass `--artifact-root EMIT_DIR` to check retained `emit-cli` files. The audit reconciles emitted and scored lengths and verifies the size and SHA-256 of each packet plus its card when present. It reports a digest of the measured inventory for each arm. Package closure and any method-specific state outside that envelope require their own inventory.

For E0, include the four combinations in one audited receipt. Pass `--factorial PP RP PT RT` with the declared arm IDs for packet/packet, real/packet, packet/tree and real/tree in that order. The audit reports both main effects and the interaction on common cells, aggregating within family and task at both multipliers. Preserve the source-retaining oracle rows in evaluator storage. Percentile intervals, promotion guards, exact artifact inventories and protected custody remain separately bound release evidence.

The evaluator handoff in #400 provides the oracle arm IDs below. Use the same schedule and real reference throughout and store both files under `$run`. Run this block only in the declared evaluator environment after authorization.

```bash
oracle_arms=(oracle_packet oracle_real_x_packet_y oracle_packet_x_tree_y oracle_real_x_tree_y)
extra_arms=(oracle_packet_x_teacher_y oracle_exact_marginal_rank12 oracle_exact_marginal_fullrank twin_c)
primary_arms=()
for arm in "${oracle_arms[@]}" "${extra_arms[@]}"; do
  "$kpi_binary" score --arena "$arena" --schedule "$schedule" --realref "$run/realref.jsonl" \
    --emit-dir "$run/emit" --arm "$arm" --out "$run/score/$arm" --budget-bytes 2048 \
    --threads "$declared_workers"
  primary_arms+=(--arm "$arm")
done
"$kpi_binary" primary --run-dir "$run" --schedule "$schedule" "${primary_arms[@]}" \
  --out "$run/e0-primary.json"
"$kpi_binary" audit --run-dir "$run" --receipt "$run/e0-primary.json" \
  --factorial "${oracle_arms[@]}" > "$run/e0-audit.json"
```

## Wire and compatibility checks

```sh
cargo test --manifest-path engine/Cargo.toml --target-dir target --locked \
  -p dope-arena --test kpi_stat_cli --test dse8_wire_conformance
DOPE_V8_EMITTER=/absolute/path/to/dope_v8_emit DOPE_V8_EMITTER_SHA256=EXPECTED_SHA256 \
  cargo test --manifest-path engine/Cargo.toml --target-dir target --locked \
  -p dope-arena --test dse8_wire_conformance -- --ignored
```

The emitter test targets the current draft 24-byte header. It exercises the complete 2 KiB cap, 48/49 slots and 8/9 classes, deterministic fit, retained legal-base sections across truncation, first-row parity and malformed packets with independently recomputed CRCs. Rebind it to the frozen decoder identity, framing and exact packed prefix semantics at the integrator handoff. Full qualification also needs role-aware FIT/query poison tests, declared mixed-type legality, range and chunk identity, twelve training parity exports at `1e-5` and restricted product-package replay. The draft test uses an empty tool PATH and moves the FIT input from its original location. Run the packaged product separately on a runner with working recipient isolation.

## Candidate replay at `8f4b4b92`

The clean candidate commit was `8f4b4b92f8866be297409c55bcdb65d3706d6740`, tree `e17506f043daed2934d5cc17821234dc73a62d4b`. The locked build produced `dope_v8_emit` SHA-256 `71560b1f6cdebcefc7ef2a495025d658354cc0eeee8c02ee3ca337533900a6ae`. The nine native V8 tests passed. All twelve supplied training-reference cases also passed an independent rerun with the float threshold tightened to `1e-5` and the neural discrete agreement requirement raised to exact equality. The rerun used the same fixture, model, export and sample code with stricter assertions in a generated test copy.

| Independent probe | Result |
| --- | --- |
| Deterministic fit and sample prefix | The narrow binary fixture emitted 797 B. Fit repeated byte for byte and row zero matched between a single draw and the batch after moving the source. |
| Fresh CRC on malformed wire | The emitter refused all 11 variants, including duplicate and reordered sections, reserved and unknown flag bits, and a code tier with no code section. The analytic frame still needs its explicit decoder identity. |
| Wide multiclass at 2,048 B | The normalized 48-slot, eight-class fit returned `over_budget`. The planner can select a projected high-rank layout before trying smaller full-width layouts. Normalized fitting then refuses that projection. |
| Shape refusal above the admitted range | At a 4,096 B cap, normalized fitting produced a 3,199 B artifact for 49 slots and eight classes. The current normalized guard uses the shared 64-slot maximum. The 48-slot, nine-class input was refused. |
| Nested byte frontier | The 4,096 B fit emitted 3,157 B, including a 2,119 B legal base. Truncation refused 2,048, 1,024 and 512 B. Choose the shared legal base against the headline cap before pricing optional sections for larger rungs. |
| Physical row ordering | Reordering the same physical rows changed the evidence bytes when X fingerprints tied. The extractor sorts equal fingerprints by current input position. Stable physical row and group IDs must reach extraction and cross-fitting. |

The candidate exposes the arena emitter. The explicit product CLI and restricted package replay still need their integration handoff. Bubblewrap on this host could not set up a user namespace. The candidate stays on hold for analytic qualification. Build, parity and native logs are retained under `target/v3-verification/candidate/`, with `target/v3-verification/candidate-black-box.log` for the independent probes.

## Integration dependencies

| Owner | Handoff and next dependency |
| --- | --- |
| Evaluator | Receiving main `a3e9a9868c61a168a659c9324ad302be0feaecd4` (#400) includes the native oracle arms, FIT-only NC2 metadata and the episode-factory adapter. Verify the integrated scorer and bind the arena v2 role, lineage, power and promotion declarations to it. |
| Emitter | Bind the frozen wire, registry and analytic package to a source commit and binary digest. Include the native row-range interface, typed metadata declaration and twelve training references. |
| Integrator | Apply the verification lane changes after #399. The existing `engine/dope-arena/` owner and test routing covers the Rust verifier. Serialize any shared release registry and contract changes with the evaluator and emitter handoffs. |
| Owner and custodian | Authorize public pilot resources, freeze the ordered candidate and comparator recipes after DEV-SELECT, and return the access-bound protected receipts. Fleet campaigns remain paused. |

Run the exact integrated source through the repository engineering and packaging gates before publication. Release claims require the signed 2 KiB result, explicit guard coverage and the observed joint T90/D80 status at each measured point.
