# Paper decisions

This file records the owner decisions that shape the paper and the one
decision that is still open. It never reads `test.csv` and never invents a
replacement value.

## Author byline (decided)

Jepson Taylor and Alton Alexander, NEVERHUMAN Research. The running header
is Taylor and Alexander. `docs/whitepaper/scripts/build_pdf.sh` writes the
anonymous PDFs by substituting that byline and `\markboth{Taylor and Alexander}`
only, then deletes the temporary tex. The method name DOPE is the generator
name in both PDFs.

| Build | File | Byline and running header |
| --- | --- | --- |
| Named | `docs/whitepaper/dope-mfs.pdf`, `docs/whitepaper/supplement.pdf` | Jepson Taylor and Alton Alexander, NEVERHUMAN Research |
| Anonymous | `docs/whitepaper/dope-mfs-anonymous.pdf`, `docs/whitepaper/supplement-anonymous.pdf` | Anonymous; same sentences and the same generated macros |

## Sealed official test (decided 2026-10-09: one run, at the end)

The owner authorized exactly one scoring run on the official test split,
after every method's configuration is frozen and hash-committed. The
procedure is fixed in the `sealed_test_v2` section of
`research/benchmark/review_fixes/predeclare_v2.json`:

1. Every train-only sample is generated before the freeze and listed in
   `sealed-v2-manifest.jsonl` (method, arm, dataset, fit seed, size, sample
   seed, artifact and CSV digests, shapes, and binary and runtime pins).
2. A seeded 5% regeneration audit runs per method; deterministic samplers
   must reproduce byte-identical files.
3. The manifest digest is committed, tagged `sealed-v2-freeze`, and
   hard-coded in the runner.
4. The run happens on xbabe3 only, with `DOPE_RF_UNSEAL=1` and an exclusive,
   fsynced registry marker. It checks every CSV digest and shape before any
   `test.csv` is opened and checks each `test.csv` against its recorded
   official digest. It scores with the pinned evaluator on CPU.
5. Sealed-test values become the paper's headline; validation values become
   development results. A sealed result weaker than validation is reported
   unchanged.

The v1 sealed run (`sealed_once.py`) can never execute: it refuses to start
once `predeclare_v2.json` exists.

### Still open

The resume rule after a crash during the sealed run. Proposed: strictly one
run; after a crash, resume only with the identical manifest digest and only
to fill missing cells, and print the number of sessions. The owner confirms
or replaces this rule at the freeze.

## What the sealed run recomputes

Every v2 macro emitted by `docs/whitepaper/scripts/review_v2_emit.py` from the
validation panel has a sealed counterpart: levels, Hodges-Lehmann contrasts,
W/T/L counts, Holm p-values, verdicts, and the strongest comparator. Byte
charges, fit receipts, catalog facts, and training monitors are unchanged by
the sealed run, because it scores existing samples and refits nothing.
