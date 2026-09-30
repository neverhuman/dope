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
