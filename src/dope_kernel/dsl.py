from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

SUPPORTED_OPS = {
    "dk",
    "marginals",
    "beta",
    "zi_beta",
    "empq",
    "spike_grid",
    "ordinal",
    "bern",
    "dependence",
    "ind",
    "gauss_copula",
    "vine",
    "chow_liu",
    "factor_copula",
    "mixture",
    "manifold",
    "target",
    "linear_sparse",
    "gam",
    "poly_cross",
    "tree_piecewise",
    "rff_gp",
    "mlp_lowrank",
    "scm",
    "regime",
    "moe",
    "lift_logit",
    "residual",
    "homo",
    "hetero",
    "quantile",
    "censored",
    "zero_one_inflated",
    "bernoulli_cal",
    "SCM",
    "RFFGP",
    "TREEPW",
    "HETQ",
    "REGIME",
    "MOE",
    "CENS",
    "COUNT",
}

MACRO_EXPANSIONS = {
    "SCM": "scm",
    "RFFGP": "rff_gp",
    "TREEPW": "tree_piecewise",
    "HETQ": "quantile",
    "REGIME": "regime",
    "MOE": "moe",
    "CENS": "censored",
    "COUNT": "zero_one_inflated",
}

ATTR_ORDER = {
    "dk": ["v", "task", "n", "p", "seed"],
    "marginals": ["count"],
    "dependence": ["kind", "rank", "k", "r"],
    "target": ["kind"],
    "residual": ["kind"],
}


@dataclass(frozen=True)
class DSLNode:
    op: str
    attrs: dict[str, Any] = field(default_factory=dict)
    children: tuple["DSLNode", ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "attrs", dict(self.attrs))
        object.__setattr__(self, "children", tuple(self.children))

    def canonical(self) -> "DSLNode":
        attrs = {k: self.attrs[k] for k in canonical_attr_keys(self.op, self.attrs)}
        return DSLNode(self.op, attrs, tuple(child.canonical() for child in self.children))

    def to_sexpr(self) -> str:
        node = self.canonical()
        parts = [node.op]
        parts.extend(f"{key}={format_value(value)}" for key, value in node.attrs.items())
        parts.extend(child.to_sexpr() for child in node.children)
        return "(" + " ".join(parts) + ")"

    def gp_token_count(self) -> int:
        if self.op in MACRO_EXPANSIONS:
            return 1
        return 1 + sum(child.gp_token_count() for child in self.children)

    def expand_macros(self) -> "DSLNode":
        op = MACRO_EXPANSIONS.get(self.op, self.op)
        return DSLNode(op, self.attrs, tuple(child.expand_macros() for child in self.children))


def canonical_attr_keys(op: str, attrs: dict[str, Any]) -> list[str]:
    preferred = [key for key in ATTR_ORDER.get(op, []) if key in attrs]
    rest = sorted(key for key in attrs if key not in preferred)
    return preferred + rest


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.12g}"
        return "0" if text == "-0" else text
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(format_value(v) for v in value) + "]"
    text = str(value)
    if text and all(ch not in text for ch in " ()[]=\t\n\r\""):
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def parse_value(raw: str) -> Any:
    if raw == "true":
        return True
    if raw == "false":
        return False
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1]
        if not inner:
            return []
        return [parse_value(part) for part in split_list(inner)]
    try:
        if any(ch in raw for ch in ".eE"):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def split_list(inner: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quoted = False
    escaped = False
    for i, ch in enumerate(inner):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            quoted = not quoted
            continue
        if quoted:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(inner[start:i])
            start = i + 1
    parts.append(inner[start:])
    return parts


def tokenize(source: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(source):
        ch = source[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "()":
            tokens.append(ch)
            i += 1
            continue
        if ch == '"':
            j = i + 1
            escaped = False
            while j < len(source):
                if escaped:
                    escaped = False
                elif source[j] == "\\":
                    escaped = True
                elif source[j] == '"':
                    break
                j += 1
            if j >= len(source):
                raise ValueError("unterminated quoted string")
            tokens.append(source[i : j + 1])
            i = j + 1
            continue
        j = i
        depth = 0
        while j < len(source):
            if source[j] == "[":
                depth += 1
            elif source[j] == "]":
                depth -= 1
            elif depth == 0 and (source[j].isspace() or source[j] in "()"):
                break
            j += 1
        tokens.append(source[i:j])
        i = j
    return tokens


def parse_sexpr(source: str) -> DSLNode:
    tokens = tokenize(source)
    if not tokens:
        raise ValueError("empty S-expression")
    node, pos = _parse_node(tokens, 0)
    if pos != len(tokens):
        raise ValueError("trailing tokens after S-expression")
    return node.canonical()


def _parse_node(tokens: list[str], pos: int) -> tuple[DSLNode, int]:
    if pos >= len(tokens) or tokens[pos] != "(":
        raise ValueError("expected '('")
    pos += 1
    if pos >= len(tokens):
        raise ValueError("expected operator")
    op = tokens[pos]
    pos += 1
    attrs: dict[str, Any] = {}
    children: list[DSLNode] = []
    while pos < len(tokens) and tokens[pos] != ")":
        token = tokens[pos]
        if token == "(":
            child, pos = _parse_node(tokens, pos)
            children.append(child)
            continue
        if "=" not in token:
            raise ValueError(f"expected attr or child, got {token!r}")
        key, raw = token.split("=", 1)
        if not key:
            raise ValueError("empty attribute key")
        attrs[key] = parse_value(raw)
        pos += 1
    if pos >= len(tokens) or tokens[pos] != ")":
        raise ValueError("expected ')'")
    return DSLNode(op, attrs, tuple(children)).canonical(), pos + 1


def canonicalize(source: str) -> str:
    return parse_sexpr(source).to_sexpr()


def build_node(op: str, attrs: dict[str, Any] | None = None, children: Iterable[DSLNode] = ()) -> DSLNode:
    if op not in SUPPORTED_OPS:
        raise ValueError(f"unsupported DSL op: {op}")
    return DSLNode(op, attrs or {}, tuple(children)).canonical()
