# H2 — SingleEncV7 V7.3 Cross-Host DiLoCo, Zero-Repeat Datasets & Tree-Visible Realism Wave

- **Wave ID:** `H2` (`ag-v73-diloco-d160l3`, `ag-v73-diloco-d192l4`, launched `2026-11-23T04:18Z`)
- **Status:** Active (pure-Rust `dope_diloco_hub` + multi-host CPU fleet on `xbabe1`/`xbabe3`/`xbabe4` coordinated with Claude's `cpufleet-w8` on `xbabe6`/`xbabe8`/`xbabe0`; `0.0 GiB` DOPE GPU VRAM).
- **Idea & Structural Fixes over H1:**
  1. **Cross-Host Shared Weights (`engine/dope-arena/src/bin/dope_diloco_hub/`):** Pure-Rust HTTP/1.1 SafeTensors outer-momentum DiLoCo server (`outer_lr=0.20`, `outer_momentum=0.70`, `staleness_alpha=0.15`, finite-value validation) unifying workers across hosts into one global model (`sync_mode=diloco_outer_momentum`, `sync_dir` & `sync_group` logged in `config.json`).
  2. **Zero-Repeat Datasets & Splits (`--never-repeat-datasets`):** Hash-partitioned disjoint shards across the 4,145 real S3 tables -> permanent `seen_datasets` retirement -> shard-salted infinite procedural synthetic priors (`0` dataset repeats and `0` split repeats verified via per-step `exposure.jsonl`).
  3. **Tree-Visible Joint Realism + Tightened Target-Head Trust-Region Anchor:** Differentiable 4-quartile 2-way copula & conditional target moment loss (`tree_visible_joint_loss`, `--tree-realism-weight 0.25`) paired with `--hard-tstr-v73` and a strengthened L2 residual trust region (`1.50*d_W^2 + 2.50*w2^2 + 0.50*w1^2 + 1.00*temp_sd^2`).
- **DEV-Only Gate 0 & Early Online Scorer Milestones (`schedule-lite-dev.jsonl`, zero CONFIRM access):**
  - Step-0 analytic baseline: `TRG = -0.3914`, boosted-tree `C2ST = 0.9553`.
  - Gate 0 unanchored `g0-httree-xh` (250 steps): `TRG = -0.3898` (`+0.0016` vs step-0; multiclass `+0.0151`) and `C2ST = 0.9533` (`-0.0020` better than step-0).
  - Anchored DiLoCo `ag-v73-diloco-d160l3` Round `108` (`full`): **`TRG = -0.3770` (`+0.0144` above step-0; binary `+0.0654`, regression `+0.0023`)**; `xh`: `TRG = -0.3899` (`+0.0015` above step-0).
- **Scope note:** All metrics in `H2/metrics.json` are DEV-only (`schedule-lite-dev.jsonl`) and live training telemetry (`train.jsonl`), never CONFIRM scoreboard claims. Empirical privacy diagnostics are not formal differential privacy and not HIPAA de-identification.
