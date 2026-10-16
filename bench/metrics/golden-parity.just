# T-037: L0 imports/copies this recipe into the root Justfile and replaces the
# pending consumer line with its Rust all-fixtures gate. Fail until then, so a
# missing Rust test/filter cannot silently claim cross-language parity.
golden-parity:
    python3 -B -m research.transfer.tdl.golden --check
    python3 -c 'raise SystemExit("Rust golden consumer pending L0 wiring")'
