"""Read Rust source through a module's declared include! closure, not a fixed file name.

The fitter's constants moved when target_fitting/part_02.rs was split into
semantic modules. The paper reads whatever files rust/compiler/target_fitting.rs
declares with include!, in order, so the old and the split layouts give the same
constants. A missing or escaping include, a symlink, a missing constant, or two
different values for one constant all raise; nothing defaults.
"""

from __future__ import annotations

import re
from pathlib import Path

INCLUDE = re.compile(r'^\s*include!\("([^"\\]+)"\);\s*$', re.M)


def include_closure(root: Path, entry: str) -> tuple[tuple[str, str], ...]:
    """(relative path, text) for the entry file and every file it includes, depth first, in order."""
    root = Path(root).resolve()
    rust = root / "rust"
    seen: list[tuple[str, str]] = []

    def visit(path: Path) -> None:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing Rust source in the include closure: {path.relative_to(root)}")
        resolved = path.resolve()
        if rust not in resolved.parents:
            raise ValueError("Rust include escapes the rust/ tree")
        relative = resolved.relative_to(root).as_posix()
        if any(name == relative for name, _ in seen):
            raise ValueError(f"Rust include cycle or repeat: {relative}")
        text = resolved.read_text()
        seen.append((relative, text))
        for child in INCLUDE.findall(text):
            visit(resolved.parent / child)

    visit(root / entry)
    return tuple(seen)


def unique_match(pattern: str, closure) -> re.Match:
    """The one value a pattern takes across the closure. Absent or conflicting values raise."""
    compiled = re.compile(pattern)
    found = [(name, match) for name, text in closure for match in compiled.finditer(text)]
    if not found:
        raise ValueError(f"architecture source constant is missing: {pattern}")
    values = {match.groups() for _, match in found}
    if len(values) != 1:
        places = ", ".join(sorted({name for name, _ in found}))
        raise ValueError(f"architecture source constant is ambiguous in {places}: {pattern}")
    return found[0][1]


def closure_contains(closure, needle: str) -> bool:
    return any(needle in text for _, text in closure)
