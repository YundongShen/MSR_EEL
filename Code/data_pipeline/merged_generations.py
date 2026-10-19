"""Read candidate hunks from raw and compact generation records."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path

_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def read_jsonl(path: str | Path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc


def diff_hunks(patch: str, sample: int = 0) -> list[dict]:
    """Parse unified hunks, keeping file boundaries out of the preceding hunk."""
    out = []
    old_file = new_file = None
    lines = []
    remaining = [0, 0]

    def flush():
        if lines:
            filepath = new_file if old_file == "/dev/null" else old_file
            if not filepath:
                raise ValueError("A generated hunk has no file header")
            out.append(normalize_hunk({"file": filepath, "text": "".join(lines), "sample": sample}))
            lines.clear()

    for line in patch.splitlines(keepends=True):
        if lines and (any(remaining) or line.startswith("\\ No newline")):
            if line[:1] not in (" ", "+", "-", "\\"):
                raise ValueError("Incomplete generated hunk")
            lines.append(line)
            remaining[0] -= line[:1] in (" ", "-")
            remaining[1] -= line[:1] in (" ", "+")
            if min(remaining) < 0:
                raise ValueError("Generated hunk exceeds its declared line counts")
        elif line.startswith("diff --git "):
            flush()
            old_file = new_file = None
        elif line.startswith("--- "):
            flush()
            old_file = line[4:].strip().removeprefix("a/")
            new_file = None
        elif line.startswith("+++ "):
            new_file = line[4:].strip().removeprefix("b/")
        elif line.startswith("@@"):
            flush()
            header = _HEADER.match(line)
            if header is None:
                raise ValueError("Invalid generated hunk header")
            remaining[:] = [int(header.group(2) or 1), int(header.group(4) or 1)]
            lines.append(line)
    if any(remaining):
        raise ValueError("Incomplete generated hunk")
    flush()
    return out


def normalize_hunk(hunk: dict, default_sample: int = 0) -> dict:
    text = hunk.get("text", hunk.get("hunk_diff"))
    filepath = hunk.get("file", hunk.get("filepath"))
    if not isinstance(text, str) or not filepath:
        raise ValueError("A merged hunk needs file/text or filepath/hunk_diff")
    header = _HEADER.match(text)
    if header is None:
        raise ValueError(f"Invalid hunk header for {filepath}")
    old_start, old_len, new_start, new_len = [int(value) if value is not None else 1 for value in header.groups()]
    declared_start = hunk.get("old_start", hunk.get("old_start_line", old_start))
    if int(declared_start) != old_start:
        raise ValueError(f"Hunk start disagrees with its header for {filepath}")
    sample = hunk.get("sample", default_sample)
    if not isinstance(sample, int) or isinstance(sample, bool) or sample < 0:
        raise ValueError(f"Invalid sample index for {filepath}: {sample!r}")
    return {
        "file": filepath, "old_start": old_start, "old_len": old_len,
        "new_start": new_start, "new_len": new_len,
        "text": text.rstrip("\n") + "\n", "sample": sample,
    }


def hunk_identity(hunk: dict) -> tuple:
    # The new-side line offset can change when other hunks precede this edit.
    return hunk["file"], hunk["old_start"], tuple(hunk["text"].splitlines()[1:])


def export_hunk_key(hunk: dict) -> str:
    value = [hunk.get("file", hunk.get("filepath")),
             hunk.get("old_start", hunk.get("old_start_line")),
             hunk.get("text", hunk.get("hunk_diff")).rstrip("\n")]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def gold_distance(hunk: dict, reference: list[dict]) -> int:
    return min((abs(hunk["old_start"] - gold["old_start"])
                for gold in reference if hunk["file"] == gold["file"]), default=20)


def candidate_hunks(record: dict, default_sample: int = 0) -> list[dict]:
    """Union the first raw generation and later merged hunks, preserving sample tags."""
    raw = (record.get("generation") or {}).get("extracted_diff") or ""
    candidates = diff_hunks(raw, default_sample)
    raw_keys = {hunk_identity(h) for h in candidates}
    for entry in record.get("merged_hunks", []):
        hunk = normalize_hunk(entry, default_sample)
        if "sample" not in entry and "generation" in record and hunk_identity(hunk) not in raw_keys:
            hunk["sample"] = default_sample + 1
        candidates.append(hunk)
    seen = set()
    out = []
    for hunk in candidates:
        identity = hunk_identity(hunk)
        if identity not in seen:
            seen.add(identity)
            out.append(hunk)
    patch = (record.get("meta") or {}).get("gold_patch")
    if patch:
        reference = diff_hunks(patch)
        distances = [gold_distance(hunk, reference) for hunk in out]
        matched = sorted((distance, i) for i, distance in enumerate(distances) if distance < 20)
        keep = {i for _, i in matched[:len(reference)]}
        out = [hunk for i, hunk in enumerate(out) if i in keep or distances[i] >= 20]
    return out
