# H2 — SingleEncV7 V7.3 / V7.3c Cross-Host DiLoCo, Zero-Repeat, Tree-Visible Realism & GPU Wave

- **Wave ID:** `H2` (`ag-v73-diloco-d160l3`, `ag-v73c-d160l3-fh-tree`, `ag-v73c-d160l3-cal-full`, launched `2026-11-23T04:18Z`)
- **Status:** Historical (superseded by DOPE v3.0.0 `DSE8`; pure-Rust `dope_diloco_hub` unified multi-host CPU workers and guarded NVIDIA GPU workers across internal compute nodes).
- **Idea & Structural Upgrades over H1:**
  1. **Cross-Host Shared Weights (`engine/dope-arena/src/bin/dope_diloco_hub/`):** Pure-Rust HTTP/1.1 SafeTensors outer-momentum DiLoCo server (`outer_lr=0.15..0.20`, `outer_momentum=0.70`, `staleness_alpha=0.15`, finite-value validation) unifying multi-host CPU & GPU workers (`sync_mode=diloco_outer_momentum`, `sync_dir` & `sync_group` logged in `config.json`).
  2. **Zero-Repeat Datasets & Splits (`--never-repeat-datasets`):** Hash-partitioned disjoint shards across the 4,145 real training tables -> permanent `seen_datasets` retirement -> calibrated infinite procedural synthetic priors (`0` dataset repeats and `0` split repeats verified via per-step `exposure.jsonl`, with 4× real-table pacing under `V7.3c`).
  3. **Tree-Visible Joint Realism + Binary-Calibrated Trust Region (`V7.3c`):** Differentiable 4-quartile 2-way copula & conditional target moment loss (`tree_visible_joint_loss`, up to `12` shifts on GPU) paired with `--hard-tstr-v73`, calibrated synthetic binary/multiclass priors (`logit_gain` + realistic label noise), and a binary-specific trust-region anchor preventing logit over-sharpening.
  4. **Real GPU Device Path & High-VRAM Pipeline:** Full `--device cuda|mps|auto` execution with TF32 Tensor Cores, `--gpu-mem-fraction`, GIL-free `spawn` multiprocess prefetch (`--prefetch-mode process`), async VRAM batch ring buffer (`--gpu-prefetch-depth`), richer GPU auditors (`--rff-dim 64 --swd-projs 64 --tree-shifts 12`), and `gpu_vram_alloc_mb`/`gpu_vram_reserved_mb` telemetry in `train.jsonl`.
- **DEV-Only `dope_kpi` Milestones (`schedule-lite-dev.jsonl`, zero CONFIRM access; bit-identical on both `main` PR #391 and `PR #390` emitters):**
  - Analytic step-0 baseline: `TRG = -0.3914`, boosted-tree `C2ST = 0.9553`.
  - Gate 0 `g0-httree-xh` (250 steps): `TRG = -0.3898` (`+0.0016` vs step-0; multiclass `+0.0151`) and `C2ST = 0.9533` (`-0.0020` better than step-0).
  - `ag-v73-diloco-d160l3-r000108-full` (`sha256: ca115356ff51...`): `TRG = -0.3770` (`+0.0144` vs step-0; binary `+0.0654`, regression `+0.0023`).
  - `ag-v73-diloco-d160l3-r001778-xh` (`sha256: 427222a4a37c...`, target heads zeroed): **`TRG = -0.3679` (`+0.0235` vs step-0; regression `+0.0179`, multiclass `+0.0527`)**.
  - `ag-v73-diloco-d160l3-r002036-a025`: **`TRG = -0.3669` (`+0.0245` vs step-0 — beats step-0 on all 3 tasks: binary `+0.0337`, regression `+0.0225`, multiclass `+0.0175`)**.
  - `ag-v73-diloco-d160l3-r002351-a025` (`sha256: f4daafe070b9...`): **`TRG = -0.3665` (`+0.0249` vs step-0; binary `+0.0608`, multiclass `+0.0240`)**.
  - `ag-v73c-d160l3-cal-full` GPU-boosted Round `7787` (`a025`, `sha256: 00cf0a73f783...`): **`TRG = -0.3768` (`+0.0146` vs step-0; regression `+0.0130`, multiclass `+0.0572`)**.
  - `ag-v73c-d192l4-gpu-fh` Round `31` (`sha256: cb1e3ab790b0...`, GPU-only `d192l4`): `C2ST = 0.9549` (`-0.0004` better than step-0 after 31 GPU rounds; multiclass `+0.0179`).
  - `ag-v73c-soup-3way-2351-2036-cal7787` (`sha256: 4ea569535ade...`): **`TRG = -0.3626` (`+0.0288` vs step-0 — beats step-0 on all 3 tasks: binary `+0.0630`, regression `+0.0073`, multiclass `+0.0162`, `C2ST = 0.9561`)**.
  - `ag-v73c-soup-2351-cal7787-4060` (`sha256: bebb4637927f...`, 40/60 early-DiLoCo + GPU `cal-full` soup): **`TRG = -0.3605` (`+0.0309` vs step-0 — beats step-0 on all 3 tasks: binary `+0.0550`, regression `+0.0158`, multiclass `+0.0219`)**.
- **Archive note:** Durable checkpoints and periodic global rounds are stored in private object storage (internal bucket paths and profile names redacted for public release).
- **Scope note:** All metrics in `metrics.json` are DEV-only (`schedule-lite-dev.jsonl`) and live training telemetry (`train.jsonl`), never CONFIRM scoreboard claims. Empirical privacy diagnostics are not formal differential privacy and not HIPAA de-identification.
