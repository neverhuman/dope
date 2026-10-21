# Comprehension check protocol, version 2

**Preregistered before collection; no reviewers have been recruited or scored.** This
protocol evaluates whether a fresh reader can distinguish the proposed TDL design from
the historical validation evidence. It does not validate any scientific result.

## Reviewers and procedure

Recruit five unaffiliated, technically literate reviewers who have not read this
repository, manuscript, or project discussion. Record each reviewer's eligibility and
conflict declaration without publishing identifying details. Each reviewer receives only
the first-page PDF image and these questions, has five minutes, and answers each in at most
two sentences. Do not show the answer key or other project material before answers are
locked. Preserve the page image hash, anonymized answers, scoring, and date in a separate
review receipt. If a reviewer is ineligible, replace them before opening any answers.

## Questions fixed before collection

| # | Question | Required elements for a correct answer |
|---|---|---|
| 1 | What does DOPE produce from a table? | A bounded sampling artifact that emits synthetic rows; it is tied to a declared task. |
| 2 | What is TDL intended to measure? | The smallest tested dataset-specific byte budget meeting transfer, driver, and hard-gate requirements; it is a proposal, not a measured result. |
| 3 | What evidence is currently measured? | Historical, selection-exposed v2 validation only; no TDL result or sealed-test result. |
| 4 | What does the byte quantity include, and what is separate? | Dataset-specific sampling bytes are the primary rate; runtime/decoder closure is separately reported. |
| 5 | What can the privacy evidence establish? | Only the scope of specified empirical attacks; it does not establish formal differential privacy or HIPAA de-identification. |

## Scoring and threshold

Two scorers who are not authors independently label each answer correct, partly correct,
or wrong against the required elements. Resolve disagreements by discussion before
unblinding the aggregate. A reviewer passes with at least four correct answers, including
Questions 2, 3, and 5, and no claim that TDL is measured, privacy is guaranteed, or
superiority has been established. The protocol passes only if at least four of five
reviewers pass. Report every answer category and the denominator; do not replace a failed
reader or change the key after collection. Any revision requires a new numbered protocol
and a fresh reviewer cohort.

## Collection status

No reviews, answers, scores, or comprehension outcomes are reported here. Version 1 at
[`COMPREHENSION_v1.md`](COMPREHENSION_v1.md) is a separate historical check and is not
pooled with this protocol.
