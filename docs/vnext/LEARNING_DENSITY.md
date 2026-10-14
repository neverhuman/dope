# Learning per byte: frontier, `B*_tested`, and the unadopted QLD

N02 decision record for the metric semantics. It defines terms; it does not adopt a headline metric. Nothing here is measured, and no paper sentence rests on it. The implementation belongs to the transfer owner (`research/transfer/`); this file is the interface it must satisfy.

## Primary output: the tested-budget frontier

**Freeze first.** Before any artifact size or outcome is seen, the campaign freezes four things:
- the workload `W`: the scheduled tasks, each a table, target, learner, synthetic size and loss;
- the family weights `w_t`, which sum to one;
- the quality floor `q`;
- the budget grid `B_1 < … < B_k`, with its resolution stated.

Duplicating a task cannot add weight, because weights are fixed per source family before outcomes.

**Coverage at a budget.** At each grid budget `B_j`, every scheduled task is attempted under the cap `B_j`. A task counts toward `C(B_j)` only when:
- it has a complete artifact, meaning every byte that sampling reads is charged, including any decoder or checkpoint;
- that artifact is at most `B_j` bytes;
- every required cell is qualified.

`C(B_j)` is the sum of `w_t` over those tasks. Tasks that are unfitted, failed, unresolved or over the cap stay in the denominator as unqualified. A terminal failure never shrinks `W`.

The frontier is the list of points `(B_j, C(B_j))`, each with its uncertainty. It is reported as measured, without interpolation between grid points.

## `B*_tested(q)` and its states

`B*_tested(q)` is the smallest tested budget whose coverage reaches the floor. It has exactly one of these states:

| State | Condition | What is reported |
| --- | --- | --- |
| `measured` | `C(B_j) ≥ q` and `C(B_{j-1}) < q` for some `j > 1`, with `C` monotone over the grid | `B_j`, and that the minimum lies in `(B_{j-1}, B_j]` at grid resolution |
| `censored_below_grid` | `C(B_1) ≥ q` | "at most `B_1`"; no smaller value is claimed |
| `no_feasible_tested_artifact` | `C(B_k) < q` | "none up to `B_k`"; no extrapolation |
| `non_monotone` | `C` decreases somewhere on the grid | every measured point and the first crossing, flagged; no monotone summary |
| `pending` | the schedule is incomplete at any grid budget | nothing: `B*_tested` is refused |

No state supports a lower-bound, interpolation or information-theoretic-minimum claim without additional evidence.

## QLD: exploratory, not adopted

Qualified Learning Density would be qualified coverage per complete byte. It is defined only for **one deployed artifact on one fixed workload**:

`QLD(A, W) = C_A(W) / bytes(A)`

- `C_A(W)` is the verified qualified coverage of artifact `A` at the frozen floor.
- `bytes(A)` is its complete charged size.

Rules:
- It is **unmeasured** without verified qualified coverage and a fixed floor. It is **ineligible** when any sampling byte is uncharged.
- It is never computed across artifacts: no aggregate coverage over a median, and no success-only bytes.
- It is not comparable across workloads.
- It cannot authorize a release, rank methods, enter a table, or appear in a title or abstract.

QLD stays a candidate only if its interpretation survives the adversarial examples below. Until then the frontier and `B*_tested` are the outputs.

## Adversarial examples

| Case | Frontier and `B*_tested` | QLD |
| --- | --- | --- |
| Empty generator | Zero coverage at every budget, so `no_feasible_tested_artifact` | 0; it only looks small |
| Trivial (constant) target | Uninformative under the null rule. The frozen rule excludes it before outcomes and never after them | undefined; ineligible |
| Padded bytes | Only moves the artifact to a larger budget | lower; padding cannot help |
| Duplicated tasks | Weights fixed per family before outcomes, so no gain | unchanged |
| Hidden pretrained decoder | Decoder bytes are charged; an external checkpoint makes the artifact incomplete | ineligible |
| Partial coverage | Full denominator, so the missing tasks stay unqualified | coverage counts only qualified tasks |
| Repeated draws or seeds | One deployed artifact per run; best-of-k on test rows is forbidden, and seeds are reported as a distribution | per deployed artifact only |
| Learner overfit | Scored on held-out real rows; selection exposure is flagged | same |

## Relation to the historical paper

The historical paper (`docs/whitepaper/dope-mfs.tex`) reports neither QLD nor `B*_tested`. It reports retention beside charged bytes, and it says in the introduction that size alone shows nothing about learning.
