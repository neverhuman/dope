# PATE-GAN original author-lab source audit

PATE-GAN remains `source_audit_pending`. No author code or runtime is executed;
no generator or campaign cell is admitted. Official tests stay sealed and
MFS-v2/PTF-v1/release/superiority remain null. No formal DP claim is made.

The [original author-lab source](https://github.com/vanderschaarlab/mlforhealthlabpub/tree/75beead341138094f89c1315ec3d722030d047cb/alg/pategan)
is distinct from the [later auditing project](https://github.com/spalabucr/pategan-audit)
previously listed in the central inventory. The selected original implementation
names Jinsung Yoon as code author and cites the
[ICLR paper](https://openreview.net/forum?id=S1zk9iRqF7). The pinned lab commit is
`75beead341138094f89c1315ec3d722030d047cb`; its root `LICENSE.md` grants BSD-3-Clause
unless an algorithm overrides it. The selected PATE-GAN files have no overriding
license notice. [pategan-source.audit.json](pategan-source.audit.json) binds the
license and six method README/requirements/Python files by Git blob and SHA256.
Full dependency/runtime rights closure remains pending. Citation: `jordon2019pate`.
The primary paper browser page presents a challenge; its PDF body hash is not
verified and no reported paper experiment is our reproduction.

## Released defaults and native selection

The executable defaults are 50 initialization iterations, batch64, one student
iteration, ten teachers, epsilon1, delta1e-5 and Laplace parameter1. The README's
example epsilon100/teacher100 is not the executable default. The CLI declares
teacher overrides as float while the generator uses `range(k)`; validate integer
parameter binding before execution. TensorFlow1.15 is declared.

The original experiment selects initialization outputs by logistic-regression
AUC on the real training table, maximized with a strict improvement rule, and
reports AUC/APR across eleven supervised model types. Campaign selection must instead use
grouped training-derived validation with the same author metric. Generator grid,
ties and that validation mapping are unfrozen. No verified native regression
objective is available: use defaults with tuning inapplicable unless source
applicability and an author regression objective are separately verified first.
Never substitute DOPE's shared KPI or rank cross-method native values.

## Generator, preprocessing and privacy gates

The original fit function returns only a generated training-size array. It does
not expose a reloadable generator artifact, while the experiment also returns the
real training table. A synthetic table or original rows cannot stand in for a
generator. A versioned adapter must expose generator parameters/configuration in
a safe row-free codec with full charged bytes, seeds, requested rows and replay.

The original credit-data CLI fits MinMaxScaler before splitting all input rows;
preprocessing and selection must be bound to training-derived partitions before
execution. Selection and final classifier evaluation round column `data_dim-1`
and use feature columns `:data_dim-1`; the mapping depends on the dataset path.
For the executable default `random` path, `data_dim=10` remains the requested
feature dimension while the generator appends a target, returning eleven columns.
Selection and evaluation therefore round feature index9, use features0-8, and
omit the appended target at index10. Only the `credit` branch resets `data_dim`
to the table width, making `data_dim-1` its last column. This is a static dimension
counterexample, established without executing author code or generating rows.
The original source is preserved. An adapter using the appended target for the
random default would change the audited native task; label mapping and
common-numeric regression applicability remain unverified.

Count all internal initializations and failed/timed-out cost, preserve the whole-
phase600-second and eight-trial/twelve-hour cell caps, and audit privacy composition
of preprocessing, repeated generators and private-real-data selection. The source
accountant loop alone establishes no formal DP guarantee. No mechanism/accountant
or actual epsilon-overshoot reproduction is claimed. Full runtime custody and
fresh owner/capacity admission remain required; no new host work is launched.
