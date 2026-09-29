# Operations files

This directory owns CI commands, local parity, and the optional pre-push hook.
Keep workflows thin: call the same `ops/ci/*.sh` scripts from CI and
`scripts/ci-local.sh`. Add a tool pin to `ops/ci/lib.sh` before using a new
scanner. Run `bash scripts/ci-doctor.sh` and `actionlint` after workflow edits.
Never read sealed validation data in a CPU PR job.
