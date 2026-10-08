# Baseline budget and timeout disposition addendum

Recorded 2026-10-08 after 75 Forest-Flow timeouts were observed. This is a
prospective disposition decision, not a claim of preregistration before those
outcomes. The original fit limit was already frozen on 2026-10-07 in round
`15048401c2966534e5d051aa31da3614a8a30555d163a4f59f38c3eba5f5cea3`.
This addendum leaves that round and its scientific inputs unchanged.

## Canonical comparison

Keep the existing **600-second fit limit**. The Forest-Flow continuation
contains 400 additional author-default fits: 100 matched lineages at fit seeds
23, 37, 53 and 71, with seed 11 reused separately. It performs no new tuning.
The existing research envelope uses one fit at a time, four CPU cores, one
learner thread, an 8-GiB memory quota, and a 15% free-RAM floor. It uses the
CPU Forest-Flow configuration, not the separate diffusion-mode GPU baseline.
CPU affinity, concurrency, RAM floor and storage admission remain as frozen.
A timeout is a recorded failure to finish under that resource budget, with
missing quality metrics. It does not show that a completed model has poor
quality. Pending or deadline-unstarted work is a scheduling cutoff, not a
method failure. Infrastructure failures retain their distinct reason.

Preserve every original fit, closure and job receipt, its configuration and
source identities, and its cost record. A timeout with no elapsed measurement
keeps measured elapsed time null; the 600-second allocation is a budget charge,
not an invented measurement. No partially written artifact is accepted as a
successful model. Publish the timeout roster and success denominator alongside
any descriptive result on the retained subset. Such a subset cannot establish
full-population or five-fit coverage.

Missing cells are not assigned a zero score, a favorable rank, or a DOPE win.
Dataset-paired comparisons state the shared usable lineage denominator and
apply the frozen missing/timeout rules. Native tuning uses each method's own
frozen objective; fidelity, utility, detection and privacy comparison metrics
do not select baseline configurations. Eight tuning trials and 12 hours per
method/dataset cell remain the ceiling, including failed and timed-out trials.
Default fits and fit-seed replications remain separately accounted. Equal
per-cell limits do not imply equal total research spending.

## Continuations and higher-budget sensitivity

There is **no automatic retry of these 75 timeouts** and no selective increase
of their canonical fit limit. An infrastructure repair may receive an admitted
retry at the same cap, under the same logical identity and a new physical
attempt identity; the original failure, elapsed measurement or allocation,
and accumulated trial/cell budget are retained. A retry does not erase a
timeout in the canonical budget-coverage ledger.

If a longer-budget sensitivity study is undertaken, freeze a separate protocol
before its first fit. Specify the eligible methods and lineage/seed matrix,
native objectives, exact larger cap, resources, stopping rules and analysis.
Offer the same revised per-cell opportunity to comparable methods; declare
any different CPU/GPU treatment. Report it as a separate budget track and
retain both tracks' costs and failures. Its results never replace the original
600-second comparison. This addendum neither launches nor admits that study.

## Evidence and interpretation

The completion checkpoint and all 75 timeout dispositions can be regenerated
from committed small receipt bytes:

```sh
python3 -B research/benchmark/publish_baseline_completion.py
```

Inputs and digests are in
`research/benchmark/results/baseline-completion-20261008-v1/inputs.lock.json`.
The live queue may advance beyond this frozen checkpoint. Its rate-based ETA
predicts dispositions, including timeouts, rather than successful fits or
metric completion. Evaluation allowances are planning assumptions, not
measured compute. Official tests remain sealed. MFS-v2, PTF-v1, release-safe
certification and superiority remain null.
