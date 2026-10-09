"""Public device sentences and private-token checks for the manuscript.

Receipts may name a machine. The manuscript may not. An unknown machine is an
error, and the error text does not echo the private name.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

_PRIVATE_HOSTS = {
    "xbabe1": "fit-machine-1",
    "xbabe2": "fit-machine-2",
    "xbabe3": "fit-machine-3",
}
_PUBLIC_HOST = re.compile(r"^fit-machine-[123]$")
_COUNT_WORD = {1: "one", 2: "two", 3: "three"}
_BANNED = (
    "xbabe0",
    "xbabe1",
    "xbabe2",
    "xbabe3",
    "xbabe",
    "GPU host A",
    "GPU host B",
    "GPU host C",
    "/home/ubuntu",
    "/mnt/fast-scratch",
    ".agent",
    "worktree",
    "Lane W",
    "Lane B",
    "B-lane",
    "Codex",
    "neverhuman",
    "jepsontaylor",
    "production_evidence_unmeasured",
)
_PULL_REQUEST = re.compile(r"(?i)\b(?:PR|pull request)\s*#\s*\d+")
_PRIVATE_PATH = ("/home/ubuntu", "/mnt/fast-scratch", ".agent", "worktree", "xbabe")
_ALLOWED_REPO_URL = "https://github.com/neverhuman/dope"
_JOURNAL_AVAILABILITY_BEGIN = "% BEGIN JOURNAL AVAILABILITY\n"
_JOURNAL_AVAILABILITY_END = "% END JOURNAL AVAILABILITY\n"


def banned_hits(text):
    """Return banned tokens present in text. Longer host forms are listed first.

    Exactly one public repository URL is allowed. Any other ``neverhuman``,
    including a second copy of that URL, is still a hit.
    """
    if not isinstance(text, str):
        return []
    scanned = text
    if text.count(_ALLOWED_REPO_URL) == 1:
        scanned = text.replace(_ALLOWED_REPO_URL, "", 1)
    found = [token for token in _BANNED if token in scanned]
    if _PULL_REQUEST.search(scanned):
        found.append("pull request number")
    return found


def strip_journal_availability(src):
    """Drop the journal-only repository sentence. The Jankurai sentence stays."""
    if not isinstance(src, str):
        raise ValueError("journal availability markers are not unique")
    if src.count(_JOURNAL_AVAILABILITY_BEGIN) != 1 or src.count(_JOURNAL_AVAILABILITY_END) != 1:
        raise ValueError("journal availability markers are not unique")
    begin = src.index(_JOURNAL_AVAILABILITY_BEGIN)
    finish = src.index(_JOURNAL_AVAILABILITY_END)
    if finish < begin:
        raise ValueError("journal availability markers are out of order")
    return src[:begin] + src[finish + len(_JOURNAL_AVAILABILITY_END):]


def machine_label(host):
    """Map a known private host to a stable machine id.

    Already-public ids pass through. Any other name raises, and the message
    does not contain that name.
    """
    if not isinstance(host, str):
        raise ValueError("unknown compute host")
    if host in _PRIVATE_HOSTS:
        return _PRIVATE_HOSTS[host]
    if _PUBLIC_HOST.fullmatch(host):
        return host
    raise ValueError("unknown compute host")


def redact_fit_path(path):
    """Hash a filesystem path once. A value with no slash is already hashed."""
    if not isinstance(path, str) or not path:
        raise ValueError("fit path is missing")
    if "/" not in path:
        return path
    return hashlib.sha256(path.encode()).hexdigest()


def public_hardware(hosts, gpu_by_host):
    """One sentence a stranger can read. No machine name is included."""
    labels = []
    for host in hosts or []:
        labels.append(machine_label(host))
    if not labels:
        return "host not recorded"
    by_label = {}
    for host, name in (gpu_by_host or {}).items():
        by_label[machine_label(host)] = name
    ordered = sorted(set(labels))
    models = []
    for label in ordered:
        if label not in by_label:
            raise ValueError("GPU model is not recorded for every machine")
        models.append(by_label[label])
    if all(model is None or model == "" for model in models):
        return "GPU model not recorded"
    if any(not model for model in models):
        raise ValueError("GPU model is not recorded for every machine")
    groups = []
    index = {}
    for model in models:
        if model not in index:
            index[model] = len(groups)
            groups.append([model, 0])
        groups[index[model]][1] += 1
    if sum(count for _, count in groups) == 1:
        return groups[0][0]
    parts = []
    for model, count in groups:
        word = _COUNT_WORD.get(count)
        if word is None:
            raise ValueError("unexpected machine count")
        parts.append(f"{word} {model}")
    if len(parts) == 1:
        return parts[0] + " GPUs"
    return ", ".join(parts[:-1]) + " and " + parts[-1] + " GPUs"


def seal_cost_row(row):
    """Store the public sentence and drop machine keys from a cost row."""
    sentence = public_hardware(row.get("hosts") or [], row.get("gpu_by_host") or {})
    if banned_hits(sentence):
        raise ValueError("hardware sentence contains a private token")
    row["hardware"] = sentence
    row.pop("hosts", None)
    row.pop("gpu_by_host", None)
    row.pop("host_fits", None)
    notes = []
    for note in row.get("notes") or []:
        if banned_hits(str(note)):
            raise ValueError("compute note contains a private token")
        notes.append(note)
    row["notes"] = notes
    return row


def redact_host(value):
    if value is None:
        return None
    return machine_label(value)


def redact_bundle(bundle):
    """Replace private hosts and fit paths in the committed receipt bundle."""
    for record in bundle["density"]["dope"]:
        record["host"] = redact_host(record.get("host"))
        record["fit_path"] = redact_fit_path(record["fit_path"])
    for records in bundle["neural"].values():
        for record in records:
            record["host"] = redact_host(record.get("host"))
    for record in bundle["forest"]:
        record["host"] = redact_host(record.get("host"))
    return bundle


def _scrub_text(value):
    if value in _PRIVATE_HOSTS or (isinstance(value, str) and _PUBLIC_HOST.fullmatch(value)):
        return machine_label(value)
    if not isinstance(value, str) or value.startswith("redacted:"):
        return value
    if any(token in value for token in _PRIVATE_PATH):
        if "/" in value:
            digest = hashlib.sha256(value.encode()).hexdigest()[:16]
            return "redacted:" + digest
        rewritten = value
        for private, public in _PRIVATE_HOSTS.items():
            rewritten = rewritten.replace(private, public)
        if "xbabe" in rewritten or banned_hits(rewritten):
            raise ValueError("unredacted private token")
        return rewritten
    return value


def scrub_document(value):
    """Drop private hosts and local paths from a JSON-like document."""
    if isinstance(value, dict):
        scrubbed = {}
        for key, item in value.items():
            new_key = _scrub_text(key) if isinstance(key, str) else key
            if new_key in scrubbed:
                raise ValueError("redaction collided on a key")
            scrubbed[new_key] = scrub_document(item)
        return scrubbed
    if isinstance(value, list):
        return [scrub_document(item) for item in value]
    if isinstance(value, str):
        return _scrub_text(value)
    return value


def origin_counts(names):
    """Count lineage display names in a fixed family order.

    The order is feynman, then strogatz, then Friedman-style fri, then bng.
    Everything else is other. A name is not given two families.
    """
    counts = Counter({"feynman": 0, "strogatz": 0, "fri": 0, "bng": 0, "other": 0})
    for name in names:
        text = str(name).lower()
        if text.startswith("feynman"):
            bucket = "feynman"
        elif text.startswith("strogatz"):
            bucket = "strogatz"
        elif "_fri_" in text or text.startswith("fri"):
            bucket = "fri"
        elif "bng" in text:
            bucket = "bng"
        else:
            bucket = "other"
        counts[bucket] += 1
    return dict(counts)


def origin_sentence(counts):
    """One manuscript sentence. Counts come from the lineage ledger."""
    required = ("feynman", "strogatz", "fri", "bng", "other")
    if tuple(counts) != required:
        raise ValueError("origin counts are not in ledger order")
    return (
        f"Of these prepared lineages, {counts['feynman']} names begin with feynman, "
        "which the PMLB catalog describes as synthetic physics, "
        f"{counts['strogatz']} begin with strogatz, which that catalog describes as "
        "simulated dynamics, "
        f"{counts['fri']} contain the Friedman-style fri marker, "
        f"{counts['bng']} contain bng and are Bayesian-network generators, and "
        f"{counts['other']} are the remainder. This paper does not re-verify which "
        "of the remainder are observational measurements. Distinct lineage identifiers "
        "are not a claim of independent source families. Related names, including the "
        r"cpu\_small lineages, stay inside the one-row-per-lineage bootstrap. "
        "The resampling unit is the lineage, so related variants move together only "
        "by chance. A source-family interval is not published, because a verified "
        "parent map is not in the ledger."
    )
