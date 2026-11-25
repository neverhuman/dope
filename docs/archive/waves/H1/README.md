# H1 — SingleEncV7 V7.1/V7.2 CPU-Fleet Pretraining Wave & DEV Head-Ablation Audit

- **Wave ID:** `H1` (`cpufleet-v71`, `cpufleet-v72`, `2026-11-22` – `2026-11-23`)
- **Status:** Closed / Retired (`2026-11-23T01:50Z` DEV selection complete; archived durably to private object storage).
- **Idea:** Pretrain the pure-Rust-compatible `SingleEncV7` column-Transformer (`d128l2`, `d160l3`, `d192l4`, `<8 MB` F32 SafeTensors, strict `<= 2,048 B` `DSE7` wire packets) across internal CPU workers (`0.0 GiB` DOPE GPU VRAM) with proper-scoring conditional NLL, RFF-TSTR, empirical-Bayes 4-quartile residuals `Q_nl`, and maximum-entropy data augmentations.
- **Pre-registered DEV Selection (`V71_SELECTION_PREREG.md`, `V72_SELECTION_PREREG.md`, Addenda B–E; 12 informative DEV cells, zero CONFIRM access):**
  1. **No V7.1 or V7.2 trained group beat step-0 beyond noise:** Best V7.1 group (`d160l3` `yh`, step `6000`) measured `+0.0004` vs step-0 (`full`: `+0.0000`); all 123 eligible V7.2 `full`/`yh` groups measured `-0.020` to `-0.072` vs step-0 (best V7.2 `xh` group: `-0.0015`).
  2. **Root cause 1 (Target-head calibration drift):** Zeroing target heads (`xh`) recovered V7.1 `d128l2` `k=12000` from `-0.048` to `-0.003` while zeroing feature heads (`yh`) remained `-0.052`; boosted-tree `dope_kpi` C2ST did not improve (`0.955` at step-0 vs `0.957–0.965`).
  3. **Root cause 2 (Training infrastructure):** Fleet-mirror audit revealed `64/65` V7 runs trained as isolated `world_size=1` seeds (no cross-node weight sharing) and revisited the 4,145 real training tables with replacement.
- **Outcome:** Under Addendum E tie-break, the provisional `v2.0.0` embedded checkpoint is V7.1 `d160l3` seed `s01` step `6000` (`delta_knots`/`delta_L` zeroed, SHA-256 `d013eb0c...`, DEV `+0.0004` = step-0 within noise). All findings directly motivated `H2` (V7.3 DiLoCo + zero-repeat + tightened target-head trust-region anchor + tree-visible realism loss).
- **Scope note:** Empirical privacy diagnostics are not formal differential privacy and not HIPAA de-identification.
