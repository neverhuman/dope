# v2.0.0 release checkpoint (decision record, 2026-11-23T01:49Z)

Rule: `V71_SELECTION_PREREG.md` + `V72_SELECTION_PREREG.md` addenda B–E (log: `PREREG_LOG.txt`).
DEV only (12 informative headline cells of the lite DEV schedule, `lite-v2rc-20261122`), same
emitter (`dope_v7_emit` built from PR #390 head 3af909ac) and `dope_kpi` builds for every arm.

- V7.1 selection over {full, xh, yh} × 2 archs × 5 steps (180 candidate arms + step 0; `dev/V71_HEADS_SELECTION.txt`,
  sha256 prefix 080e1368…): (d160l3 `yh`, k 6000), group mean ΔDEV **+0.0004** (full +0.0000).
- V7.2 selection over the eligible groups (base runs k 1500/3000/6000, d192l4 k 1500; 123 candidate arms + step 0;
  `dev/V72_DEV_SELECTION.txt`, 4c7e37ce…): (d128l2 `xh`, k 1500), group mean ΔDEV **−0.0015**.
  Every V7.2 full and `yh` group is below step 0 (−0.020 … −0.072).
- Addendum E: V7.1's group mean is higher (+0.0004 vs −0.0015) → **V7.1 d160l3 `yh` k 6000,
  release seed s01** (ΔDEV +0.0283).

Embedded file: `cpufleet-v71-d160l3-s01.step_006000_ema.yh.safetensors`, 2,755,388 bytes,
**sha256 d013eb0cb08c9901c39e2bd65232c6bfbc41f68a613797971f7afb525b70cfac** — a copy of
`cpufleet-v71-d160l3-s01/checkpoints/step_006000_ema.safetensors` (sha256 322ba652528b1be2…,
matches the run's ledger) with `delta_knots` and `delta_L` (weights and biases) set to zero;
loader and layout unchanged (`SingleEncV7Weights::from_safetensors_bytes`, n_heads 4).

Reading: no V7 group beats the step-0 packet decode on DEV by more than noise; the trained
target heads are what costs retention in V7.1/V7.2 (`xh` ≈ step 0), and realism (boosted-tree
C2ST) does not move. The CONFIRM look tests this checkpoint against step 0 as a secondary
comparison; nothing is claimed for it beyond what that look measures.
