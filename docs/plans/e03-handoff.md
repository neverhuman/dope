# E03 development pilot: parked handoff

Status on 22 October 2026: **parked by the primary human.** E03 is the only
canonical task left that agents could execute. E04, E05 and E06 need human
principals (`finish65-major-release.md`). v1.0.0 shipped as an uncertified
research release without them. This page records exactly where E03 stopped
and what resuming it requires, so no one has to rediscover it.

## What E03 runs

The plan's floor is twelve cells (`finish65-major-release.md:32`, `:202-226`):

- **Tasks:** one observational task (bikeshare, as the current plan binds it)
  and one adverse-mechanism task (synthetic XOR parity, 640 rows, seed 1729,
  target last).
- **Arms:** DOPE, Gaussian copula and compressed real rows, at byte caps of
  4096 and 32768.
- **Learners:** `kumo_tabular`, `tabicl`, `mitra`, `gradient_boosting` and
  `neural`, plus a constant-null reference, at scales n and 4n and in the
  standard and stratified-anchor contexts (`pilot_cell_plan.py`).
- **Stages:** each cell passes eight stages (prepare, compile, reload, sample,
  learners, drivers, receipt, report; `campaign_runner.py`). It ends in a
  private trace, a cost report and an independent acceptance.

## What is on main

- The pilot runner, the worker runtime, the stage journal and the physical
  operation guard.
- The native receipt bridge. The receipt stage makes one bounded
  `dope-kernel verify-pilot-receipts` call to the Rust `verify_receipt`
  (`research/transfer/pilot_native_receipt.py`,
  `rust/main/pilot_receipt_commands.rs`).
- `campaign_cli.py pilot` still refuses with
  `native_execution_admission_not_connected` by design. No product code calls
  `execute_pilot_cell` yet.

## Work left before the first cell

Each item is a small, separately reviewed PR. The person who reviews a PR must
not have written it. Effort estimates come from reading the code on
22 October.

| # | Work | Main files | Estimate |
|---|---|---|---|
| 1 | Explicit owner and test map entries for the bridge files and everything below | `agent/owner-map.json`, `agent/test-map.json` | 0.5 h |
| 2 | Trusted policy issuer and per-cell basis documents (`EXECUTOR_BINDING_INTERFACE`: `issue_policy(path, sha256) -> (bytes, sha256)`). The issuer recomputes every dependency hash, seed, registration and settled cost before it issues a policy. | `research/transfer/receipt_issuer.py` and tests | 1–1.5 d |
| 3 | Operator package: signed config with pins and expiry, learner registry, guardian, and the per-cell build order (cell document → input hash → worker authority → basis → bridge → runtime → capsule → plan → private evidence directory → signed admission → `execute_pilot_cell`). Ports the out-of-tree executor into reviewed source. | `research/transfer/e03_operator/` (each file under 500 lines) | 2 d |
| 4 | CLI wiring for `pilot` (`--operator-config`, `--evidence-dir`, `--public-key`, `--trusted-key-sha256`); `confirm` keeps refusing. Adds an admission signing helper and native binary pinning. | `campaign_cli.py`, `e03_operator/cell.py` | 1 d |
| 5 | Re-qualify all 24 learner role, task and context cases at the final source head. The guardian requires every qualification to match the run's commit and tree, and three pinned evidence folders no longer exist. This includes the null reference on the adverse task, whose old evidence read `row[-1]` from a target-first file. | `research/transfer/role_qualification.py` | 1 d |
| 6 | Optional: replace bikeshare with the 686-row concrete TRAIN export (manifest `fde08772…`, CC BY 4.0, target last). This is a plan change: it needs a hash-based split inside the TRAIN rows, new qualifications and new pricing. | `research/transfer/train_split.py` | 0.5–1 d |

### Native binary

Cargo hard-links `target/debug` and `target/release` outputs into `deps/`, and
`read_regular` refuses hard-link aliases. Production runs therefore build with
`cargo build --release --locked --bin dope-kernel`, then copy the binary into
the private run directory with `install -m0500`. The operator checks that the
copy has one link and pins its SHA-256 in the cell plan and in the bridge.

## Decisions only the primary human can make

1. **Signing key.** Every admission carries an Ed25519 signature
   (`pilot_runner.py:_verify_pilot_authorization`). The plan forbids an agent
   creating a signing identity silently. Proposed: the human generates the key
   on xbabe4 and keeps the private half there. Each run host holds only the
   public key. The human signs the operator config and the twelve cell
   admissions.
2. **Gate acceptors and reviewer.** Someone must accept the seven operation
   gates, and someone must perform the independent E03 acceptance.
3. **Budget.** The priced plan reserves 108.6 h serial, or 434 CPU-hours at
   4 CPUs per cell. The standing grant is 40 CPU-hours and 8 GPU-hours.
4. **Data.** Keep bikeshare, which is previously exposed public development
   data, or switch to the concrete TRAIN export (item 6).
5. **Byte accounting.** Control cells are charged their whole sampling
   closure, including the 29,432-byte native executable, so the 4 KiB control
   cells are expected to be over budget. Accept that as the recorded result,
   or rule on it in the protocol.
6. **Mitra licence record.** The Mitra checkpoints declare Apache-2.0 in
   `locks/other-auditors.json`, but no licence file was retained for
   `autogluon/mitra-regressor-2` at `246c7d31`. TabICL (BSD-3-Clause) and
   NVIDIA Kumo-Tabular (OpenMDW-1.1) were approved in SC-DEC-128. The registry
   entries in `registries/models.json` still read `unresolved`.

## Compute plan

- **Where the runtimes are:** only xbabe2 holds the runtimes, weights and
  data today. Mirroring them takes about 13.4 GB: the common runtime view, the
  Mitra and conventional runtimes, `w01-cert-venv`, the Kumo-Tabular large
  checkpoints and TabICL. Copy with `rsync -aH --links` to identical absolute
  paths, then verify with a SHA-256 manifest and a dry-run qualification
  import.
- **Which hosts can run:** the environment uses CUDA 13 (`torch` 2.13.0).
  xbabe2 and xbabe3 have driver 580. xbabe1 has driver 570, so it needs an
  upgrade before it can run Mitra. xbabe4 (macOS, no CUDA) serves only as the
  signing host.
- **Order:**
  1. Run one timing-probe cell on xbabe2 (DOPE, observational, 4096 bytes;
     reserved 10 h, likely 1–3 h).
  2. Re-price the remaining cells from the probe.
  3. Run the other eleven cells one per GPU host at a time.
- **Duration:** the worst case is about 55 h on two hosts.

## Where the evidence is

Captain packets under `xbabe2_kicker/captains-20261017/` cite these paths:

- E03 runtimes and W01 qualification environments: the main checkout's
  `target/` (`w01-*`, `e03-*`).
- Lane evidence: `~/dope-evidence/` on each host. `PATHMAP.json` there maps
  the old worktree and scratch paths to their new places.

Run `tools/claims/check.py` before any document reports E03 results. A pilot
result is development evidence only, never a sealed or certified result.
