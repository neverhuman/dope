# Four-hour pilot status

Started 2026-09-30 13:13 UTC. Jepson approved the DOPE generator benchmark as the paper's primary comparison. BeyondArena is secondary. The real-data utility endpoint is null-normalized TSTR/TRTR retention on untouched real test rows; public test remains sealed during this pilot. The pilot cap is four hours, ending by 17:13 UTC.

Adult, California, and News are the fixed pilot datasets. The requested roster is DOPE, GaussianCopula, TabPC, TabKDE, TabSyn, TabDiff, and AIM, with fit seeds 11 and 23 and sample seeds 101 and 211. The queue must produce a receipt for every planned fit, including methods blocked by source, license, dependency, or resource checks. It must never score an unavailable method as a DOPE win.

At admission, xbabe1's RTX 4090 was idle. The RTX 3090s on xbabe2 and xbabe3 were occupied by other workloads and will not be preempted. The coordinator has 277 GB free at `/mnt/fast-scratch`; the prior 383 MB benchmark data directory has been copied to `/mnt/fast-scratch/dope-benchmark` and requires hash verification. No new pilot fit has started as of this record.

The method lock remains incomplete. DOPE and GaussianCopula are runnable with the existing adapters. TabPC, TabSyn, TabDiff, and AIM need executable adapter/dependency audits. TabKDE's pinned source lacks a recorded license grant. The pilot will measure what can run within four hours and report blocked cells and the resulting full-campaign feasibility bound before opening the public test set.

## Update 13:56 UTC

The frozen queue has 42 fit cells. All six DOPE fits completed with eight deterministic sample cells each. Thirty source-blocked cells are being recorded; GaussianCopula's six CPU jobs remain in progress. The first two Adult GaussianCopula fits have exceeded 30 minutes each without a fit receipt, demonstrating that the in-process Python alarm does not provide a reliable hard cap while upstream fitting runs in native code. The queue's outer `timeout` process is the hard cap.

Validation-only DOPE `n` cells, fit seed 11 and sampling seed 101, completed on all three datasets. CatBoost retention was 0.815 Adult, 0.724 California, and 0.578 News; these are descriptive validation estimates, not the final real-test endpoint or MFS-v2. Exact and near copy counts were zero in those cells. All three metric vectors explicitly retain `mfs_v2: null`.

Separate compatibility probes, excluded from the frozen comparison matrix, completed: TabPC News author configuration took 545.1 s to fit and produced a 555,381,583-byte artifact containing an 8,087,986-byte saved training tensor; AIM California with a fixed eight-bin domain took 63.1 s and produced a 14,424-byte artifact. The TabPC probe used installed Torch 2.7.0 rather than upstream's requested 2.9.1; its receipt correction states that recorded VRAM is sampling peak only. The AIM probe does not establish end-to-end differential privacy because the common projection was fitted on private training rows.

## Update 14:26 UTC

The first frozen queue pass finished with 6 DOPE successes, 2 Adult GaussianCopula outer timeouts at 2,700 s each, 4 California/News GaussianCopula infrastructure failures, and 30 blocked source cells. The four infrastructure failures came from the local subprocess importing the live adapter file instead of the frozen package. Their original failed receipts and logs are preserved. A repair queue now runs only these four cells with the frozen package as its working directory and writes attempt-2 receipts separately. The original queue summary's `measured_job_seconds` includes time spent waiting for host slots and must not be used as a compute-cost estimate; the reconciliation report uses runner fit/sample receipts and starts timing after slot admission for repair attempts.

The AIM California pilot-only adapter completed fit and all eight deterministic sample cells: 67.16 s fit, 14,554 B including the projection map. Its validation-only CatBoost, linear, and MLP TSTR retention was 0.698, 0.917, and 0.748 at `n`; MFS-v2 remains null because the full generator gate evidence is absent. The adapter is restricted to jobs marked `pilot_only`; it is not a final-comparison or formal-DP approval. Adult and News AIM follow-up fit attempts have 1,200 s in-worker hard caps. The first Adult attempt failed on xbabe3 because its Python environment lacked `cycler`; that receipt is retained, and a retry uses an isolated scratch dependency directory. News remains in progress.

The runner now executes adapters in child process groups and enforces fit/sample deadlines even if native code ignores a Python signal. The pilot reconciliation code and 24 benchmark tests pass. All bulk fits, samples, dependencies, and logs are under `/mnt/fast-scratch/dope-benchmark` (4.9 GB at this update); no full-campaign job or test evaluation has been admitted.

## Update 14:39 UTC

News AIM reached its 1,200 s fit deadline, was stopped by the child-process hard cap, and produced a timeout receipt with no sampling artifact. `/usr/bin/time` recorded a peak resident set of 18,902,724 KiB. The xbabe3 AIM Adult retry remains under its own cap.

The two source-locked reference compact methods, independent marginals and Chow–Liu, are now dispatched on xbabe1 as a **separate 12-fit supplemental matrix**. Their results will inform cost estimates for the protected compact panel but will not be added to the fixed seven-method pilot or treated as a public-test comparison. Their jobs use the same hash-locked training/validation inputs, two fit seeds, and eight sampling cells per fit. The xbabe1 benchmark path is an SSHFS mount of xbabe2 scratch; `df` on the exact benchmark directory confirmed it is not writing bulk data to xbabe1's root filesystem.

The AIM Adult retry also reached its 1,200 s child-process cap without a sampling artifact. Its peak resident set was 14,858,676 KiB. The earlier missing-dependency receipt and this timeout receipt are both preserved; neither is counted as a completed fit.

## Update 14:47 UTC

The 12-fit compact reference supplement completed with 12 successful fits and all 96 deterministic sample cells. Fit costs were 0.06–1.77 s for Chow–Liu and 0.06–0.52 s for independent marginals; sampling across eight cells took 6–185 s per fit depending on dataset and method. Charged artifact bytes were: Adult 19,100/27,896, California 4,296/11,783, and News 25,937/71,671 for marginals/Chow–Liu respectively. Only California independent marginals is within L3 among these six dataset–method combinations. These are cost and byte observations, not generator-gate passes. Validation-only `n` metrics for both methods are being completed.

## Update 15:02 UTC

Both California GaussianCopula repair jobs fitted in 54.3–54.5 s and produced 1,619,430-byte charged artifacts, far above L2 and L3. Each attempted four sample cells; all eight sample attempts reached the 600 s in-process sampling cap. The 2,700 s outer job cap then stopped both jobs, and their attempt-2 timeout receipts are preserved. The two News GaussianCopula repair jobs have now started on xbabe2. The reconciled fixed-matrix tally is currently 6 successful DOPE jobs, 4 Gaussian timeouts, 2 Gaussian jobs in progress, and 30 source-blocked cells. No Gaussian sample has entered validation scoring.

An independent xbabe1 rerun of the completed California Chow–Liu seed-11 host job returned all eight existing sample receipts with unchanged SHA-256 values for the fit and sample receipts. The check is saved as `compact-supplement/resume-check.json` on benchmark scratch. It demonstrates pilot host-job idempotence without opening test data.

## Update 15:27 UTC

Both News GaussianCopula repair fits have been active for about 25 minutes without a fit receipt. Each remains under the 2,700 s outer cap and 16-core affinity; the coordinator still reports more than 80 GiB memory available, so these pilot jobs are not pressuring other owners' workloads. The 21 completed runner artifact inventories available so far reconcile exactly with their charged byte receipts. The real-test admission check exited 2 with six missing or incomplete lock blockers, and the test partition remains sealed.

The frozen queue placed both simultaneous GaussianCopula seeds on affinity 0–15. This does not exceed either job's 16-core cap but creates contention and weakens standalone timing inference. The reusable launcher now assigns disjoint 16-core slots per host; the already-running frozen attempts are left unchanged and will be reported with this limitation.

## Closed 15:47 UTC

The four-hour maximum pilot closed in 2 h 34 min. Reconciled fixed-matrix status: 6 DOPE successes with 48 deterministic samples, 6 GaussianCopula outer timeouts, and 30 source-blocked cells. The failure ledger retains 40 blocked, failed, or timed-out attempts across original and repair queues. News GaussianCopula did fit on both seeds in 2,327–2,329 s, producing 11,720,459-byte artifacts, but no sample receipt before the 2,700 s job cap. Adult GaussianCopula timed out without a fit artifact; California fitted but every attempted sample timed out. None of these outcomes is scored as an MFS-v2 comparison.

The separate 12-fit compact reference supplement, one TabPC compatibility probe, and AIM pilot-only probes are accounted for in `PILOT_COST_REPORT.md`. The final artifact inventory audit reconciled all 23 completed runner fit artifacts. The public test remains unopened, `budget.lock.json` remains absent, and the admission check fails closed with six lock blockers. The decision is **no full campaign admission from this pilot** because the protected compact method set and evaluator are incomplete and the 14-day tuning and attack budget cannot be frozen from these measurements.
