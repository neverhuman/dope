### Addendum H: second looks at DOPE v3 against the four third-party generators

Written by Claude on 2026-11-24. Revision 4 applies Codex's text review of 06:39Z and adds part (b).

**Status and binding**
- Jepson approved parts (a) and (b) directly in Claude's session at about 06:42Z, in his own answer, not a relay.
- H binds only from its `PREREG_LOG.txt` line. That line records the logging time and the SHA-256 of this text, which is appended to `V3_SELECTION_PREREG.md` byte for byte as reviewed.
- Nothing under H is emitted or scored before that line exists.
- Until H is reported, the paper says the comparison rests on one look.

**Role**
- H is **descriptive**. It repeats the paper's comparison of DOPE v3 with SDV GaussianCopula, CTGAN and TVAE and arfpy ARF, which so far rests on one look (the report look, data seeds 5–8).
- It changes no arm, encoder, threshold or contract, and it ships nothing.
- **H has no role in Addendum G's G4 decision, in model selection or in release qualification.**
- **Part (a):** reused DEV splits, data seeds 9–12.
- **Part (b):** the G4 fresh look, DEV data seeds 13–16. It exists only if G4 runs.

## Part (a): data seeds 9–12

**Inputs (frozen; the run fails closed on any mismatch)**
- **Schedule:** `schedule-look-dev.jsonl`.
  - SHA-256 `beaa2a0fa3e2bcc44c37f23bf15adca0c2ff5b4adb5f212a198c99f3d23f71a2`.
  - 108 scheduled cells: 64 headline and 44 small-n, 27 per seed.
  - 22 DEV tables, data seeds 9–12 only.
- **Reference files:** `realref.jsonl` `5aa9d2d628409feb…` and `families.jsonl` `3fa7f21fe4b6ecea…`, with full digests recorded in `MANIFEST-h.json`.
- **Panel:** two digests, each checked against its own expected value.
  - The physical file `plan/baseline-v250.panel.json` has SHA-256 `d5bd12b0096238a11391e77fa9538123215e7cb54b23ff990f65ad5091bfbf84`.
  - The canonical panel digest carried by the schedule rows and receipts is `8d18e7837d6cc1bdf640f8e695045225a9c66d0d330ee4e835f8058cdcb7b29e`.
- **Other inputs:** `plan/ag-dev-tables.txt` (the frozen DEV list, `6fe72230…`) and contract `dope-kpi/v2` `200ea0ed1ba2…`.
- **Source:** all inputs come from xbabe2 `~/dope-data/kpi-runs/v3-final-look/` and are copied, never edited.
- **Checks before launch:**
  - the schedule digest;
  - the seed set {9, 10, 11, 12};
  - DEV membership of every scheduled table;
  - the realref, families, DEV list, panel (both digests) and contract digests.
- **Not used:** CONFIRM, HOLDOUT, SEALED and FMGAP data.
- **Known before writing (disclosed):**
  - These splits are not new for DOPE. Addendum F scored the released engine on them (`served-auto-v8`, scoring binary `58314f0a…`): TRG −0.183, Transfer Score 81.7, 41 informative cells in 10 families, one failed cell.
  - The later signed v3.0.0 asset `8723955b…` was packet-replayed separately.
  - We chose to run part (a) after seeing those numbers. No third-party generator has been scored on these splits.

**Arms**
- **DOPE v3:** `served-auto-v8`.
- **Controls:** `copy_a`, `bootstrap_a`, `indep_marginals`, `label_shuffle` and `twin_c`.
- **The six reused arms are checked before anything is scored.** Their score directories in `v3-final-look` are reused unchanged and not re-emitted.
  - Their primary rows, guards and guard summaries must equal those of the committed Addendum F `primary.json`, SHA-256 `892a8ee0442aec76a442eb7bc28fe6951c5187a54ccc92158ed5d7e440c44c73`.
  - After reduction, `primary-h.json` must repeat those six rows field for field.
- **Third party:** `sdv-gaussian-copula`, `sdv-ctgan`, `sdv-tvae` and `arfpy-arf`, using exactly the report-look recipes of the committed `MANIFEST.json` (`c44db0cc0178…`, full digest recorded):
  - xbabe0 arms file `~/dope-runtime-claude/baselines/arms-baselines.json` `1eeabd2975fd81f3e99afc6259265a6e7bc0d6ae460b06d23564b8db65e31ca3`;
  - `make_arms.py` `5bf45e1160e53052b611c914e77d71be3056704347039403a55bb68652b547da`;
  - wrapper `baselines.py` `b091e0ddd7dde26c…`;
  - venv `~/dope-runtime-claude/venv-baselines`: Python 3.12.3, sdv 2.2.0, ctgan 0.13.0, copulas 0.15.0, rdt 2.1.0, arfpy 0.1.1, torch 2.10.0 and scikit-learn 1.7.2, with its runtime-lock digest recorded;
  - library defaults, CPU, one thread inside the wrapper, no byte cap.
- Before launch, the four resolved arm specs must equal the report manifest's.
- Nothing is installed, upgraded or re-pinned. A change of any pinned digest voids the run.

**Tool: one `dope_kpi` build for every stage**
- `v3-final-look/bin/dope_kpi`, SHA-256 `ea855a89d4fad31ae1d197f3f2a1045a84daf98ec6b527a8e03c71b3653c2863`.
  - It is copied unchanged to an isolated path on xbabe0, `~/dope-data/kpi-runs/v3-h-s9-12/bin/dope_kpi`, and SHA-checked there before each stage.
  - That build scored `served-auto-v8` and the controls on these splits, so every arm on this look is scored and reduced by one build.
- **The report look's own driver is not used,** because it hardcodes `v3-bin/dope_kpi` (`c01fd919…`, a build that reduces differently).
  - Part (a) runs the bound command lines below directly, with the absolute `ea855a89` path.
  - The shared report binary and driver are not modified.

**Bound command lines (only the run directory, schedule path and binary path differ from the report look)**
- **emit:** `BIN emit-cli --arena ~/dope-data/flat/s3-v250 --schedule SCHED --out . --arm-file arms.json --arm ARM --threads T --timeout-s 14400 --resume`
  - The timeout of 14,400 s is the report look's, as its job logs `bl-fast`, `bl-ctgan` and `bl-tvae` record.
  - `T` per method is the report manifest's `wall.threads`: Gaussian copula 3, CTGAN 7, TVAE 4, ARF 3. The Gaussian copula and ARF run in sequence in one job, as in the report look.
  - **Retry rule:** if and only if the first pass reports a timeout, run one more pass of the same command for the same arm over the same schedule. That pass, like the report look's, resumes every cell whose rows are not all `ok`, so it can also reattempt non-timeout failures in that arm.
  - Both ledgers are kept, and there is no other retry of any kind.
- **score:** `BIN score --arena ~/dope-data/flat/s3-v250 --schedule SCHED --realref realref.jsonl --emit-dir . --arm ARM --out score/ARM --budget-bytes 1000000000 --privacy-rows 1024 --threads T`
- **guards:** `BIN guards --run-dir . --arm ARM --threads T`
- **Reduction, on xbabe2:**
  - `BIN primary` over the ten arms produces `primary-h.json`;
  - for each method, `BIN compare … --candidate served-auto-v8 --baseline <method>` produces `v3-vs-<method>-h.json`.
  - Both run with the explicit DEV flags `--panel dev --dev-tables plan/ag-dev-tables.txt`.
  - The checked argv reference for these two commands is the `receipts)` branch of `.agent/tools/v3-look/final_look.sh`. The `command` fields of `primary.json` and `compare-b.json` hold only the subcommand names, so they are not used as references.
- **`MANIFEST-h.json`:** before the first emission, it records the complete resolved argv of every stage, including the DEV flags and the chosen thread counts, together with every binary digest, input digest and package version.
  - `T` sets `dope_kpi`'s own worker count and changes no recipe.
- **Comparison with the report manifest (before launch).** The emit, score and guards argv in `MANIFEST-h.json` are compared token by token with the `commands` of the committed report `MANIFEST.json` (`c44db0cc…`). Only these differences are permitted:
  1. **Binary:** the isolated `ea855a89` path, where the manifest says `dope_kpi` (the report look ran `v3-bin/dope_kpi`, `c01fd919…`).
  2. **Paths:** the run directory and the schedule path `schedule-look-dev.jsonl`.
  3. **Timeout:** `--timeout-s 14400` where the manifest's `commands.emit` says `3600`.
     - The manifest string records the wrapper default.
     - The executed report jobs passed 14,400, as the xbabe0 job logs `bl-fast`, `bl-ctgan` and `bl-tvae` record (`emit … timeout_s=14400`).
     - This addendum is the authority for 14,400.
  4. **Threads:** `T` per method is the manifest's `methods[].wall.threads`.
- Every other token of the three commands must equal the manifest's. The four resolved arm specs must equal the report run's. The wrapper, arms-file, `make_arms.py` and venv-lock digests must equal the pins above, so every method runs the report manifest's `recipe`. Any other difference aborts the run before emission.

**Acceptance (validated, never repaired)**
- Every phase must exit 0, and the emit ledger and score outputs must cover exactly the 108 scheduled cells per method.
- A failed informative cell (error, timeout after the retry, single-class sample) is kept and scores −1 under the contract.
- A missing or malformed ledger or score file makes the run **void**. A void is disclosed with its reason, never repaired or dropped selectively.
- No cell, method or seed is re-run other than by the retry rule above.

**Family receipt and reading**
- **`family-h.json`** is materialized from the four singleton receipts and bound to their SHA-256s. It applies Holm (m = 4) over their `p_one_sided` fields.
  - Each pair's promotion outcome is re-evaluated with the adjusted p, other conditions and guards unchanged.
  - Adjusted outcomes may differ from the singleton outcomes, and both are reported.
- **Statistic:**
  - per method, the paired Transfer Score difference DOPE v3 − method at 4× in points (100 × ΔTRG);
  - family bootstrap with 10,000 draws, seed 20261123, 95 % percentile interval;
  - one-sided p for "v3 is better", Holm over the four;
  - the 1× difference disclosed beside it.
- **"Transfer advantage replicates"** for an SDV method iff, on that look, the 4× lower bound is above zero and the Holm (m = 4) p is below 0.05. Otherwise the paper says it did not replicate on that look.
- **ARF:** "DOPE is not shown to transfer better than ARF" stays unless every look clears that bar. Otherwise the paper says which looks did. Looks are never pooled.
- **Reporting:** whatever H shows is reported, as one more block of the paired table per look plus a results sentence. The ranking table and Figure 1 stay on the report look.

**Procedure and resources**
1. **xbabe0, behind `x0_gate2`** (which honours the TIMING.lock freeze), inside the JSR guard:
   - host-mean CPU ≤ 0.60, ≥ 16 GiB free, `nice -n 10`, no GPU;
   - about 3.8 CPU-hours;
   - run dir `~/dope-data/kpi-runs/v3-h-s9-12`, holding copies of the frozen inputs and the four arm specs;
   - steps: emit, score, guards.
2. **xbabe2:** a new run dir, `~/dope-data/kpi-runs/v3-h-s9-12`.
   - It holds copies of the six reused score directories and the frozen inputs, plus the four method score directories rsynced from xbabe0.
   - Steps: reduce, then the family receipt.
   - `v3-final-look` is never written.

## Part (b): the four methods on the G4 fresh look (data seeds 13–16)

- **Schedule:** the G4 schedule as frozen and logged at 2026-11-24T06:48:18Z, SHA-256 `534d3da82eda377892dac1c2712df0035980868de2b9d5f9eeda4acb2ee53062`, in xbabe2 `~/dope-data/kpi-runs/v3-g4-look`, unchanged.
  - Same panel, DEV list, contract and `dope_kpi` `ea855a89` as part (a).
  - Fresh for every arm: nothing has been emitted or scored on data seeds 13–16.
- **When:** part (b) runs if and only if Addendum G's G4 runs and produces its locked readout. Otherwise seeds 13–16 get no H scores and part (b) is reported as not run.
- **Order (the firewall to G4):**
  1. G4 is emitted, scored and reduced in `v3-g4-look` with its roster only.
  2. Its readout is frozen: the primary, compare and family receipts and the ship decision, with their SHA-256s logged in `PREREG_LOG.txt` and posted on the board.
  3. Only then does part (b) start.
- **Its own directories, never G4's:** xbabe2 and xbabe0 `~/dope-data/kpi-runs/v3-h-b-s13-16`.
  - No H arm is ever added to `v3-g4-look`'s `arms.json`, `primary.json` or any `compare-*.json`, and `final_look.sh` (its `EXTRA_ARMS`, `score` and `receipts` branches) is not used for H.
  - The directory holds digest-verified copies of:
    - the G4 schedule (`534d3da8…`);
    - G4's `realref.jsonl`, families, panel and DEV list;
    - the score directories of G4's served v3.0.0 arm and its five controls;
    - for the disclosure-only comparison, the G4 finalist's score directory.
  - Each copy is checked against the digests in G4's frozen readout before any H-b emission.
- **Arms:** the same four third-party recipes, pinned exactly as in part (a).
- **DOPE comparators:**
  - G4's served v3.0.0 arm, the signed release asset and the released encoder this paper describes;
  - descriptively, the G4 finalist.
- **Commands:** part (a)'s bound argv and its manifest comparison, with only the run directory and the schedule path changed:
  - emit with `--timeout-s 14400` and the same retry rule;
  - score with `--budget-bytes 1000000000`;
  - guards;
  - `T` per method as in part (a).
- **Manifests:** separate arm, score and receipt manifests, starting with `MANIFEST-h-b.json`.
- **Acceptance:** as in part (a). A missing or malformed H-b result voids part (b) only and has no effect on G4, whose readout is already frozen.
- **Receipts, all in `v3-h-b-s13-16`:**
  - `primary-h-b.json` over the served arm, the five controls and the four methods; the served and control rows must repeat G4's frozen primary rows field for field;
  - for each method, `v3-vs-<method>-h-b.json` with candidate served v3.0.0;
  - `family-h-b.json`, Holm (m = 4), read by the same rule as part (a);
  - finalist-versus-method compares, as disclosure only, without Holm or any claim.

## Public evidence (allowlist, both parts)

- **Committed** under `bench/real-world/results/scoreboard/v3.0.0/evidence/addendum-h/`, with `SHA256SUMS` and a README:
  - the primary receipts;
  - the `v3-vs-<method>-*.json` receipts;
  - the family receipts;
  - this addendum's logged text and `PREREG_LOG` line;
  - sanitized manifests (approved provenance, argv, digests and versions).
- **Stays in research custody, never committed:** private inputs, score directories, synthetic rows, packets, row-level seeds and indices, and worker logs.
