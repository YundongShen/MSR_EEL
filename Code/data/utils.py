"""Shared utilities for the data pipeline."""

from __future__ import annotations

import hashlib
import re
from typing import Sequence


def hash_signature(text: str) -> str:
    """Return a short SHA-256 hex digest of *text*."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def normalize_diff(diff: str) -> str:
    """Extract and clean a unified diff from raw LLM output.

    Handles markdown code fences (```diff ... ```) including unclosed ones,
    and strips metadata lines (index, diff --git) so that two
    functionally-identical diffs compare equal.
    """
    # Strip markdown code fences — handle both closed and unclosed (truncated).
    fence_match = re.search(r"```(?:diff|patch)?\s*\n(.*?)(?:```|$)", diff, re.DOTALL)
    if fence_match:
        diff = fence_match.group(1)

    kept: list[str] = []
    for line in diff.splitlines():
        if re.match(r"^(index |diff --git |\\ No newline|```)", line):
            continue
        kept.append(line)
    return "\n".join(kept) + "\n"


def diff_line_count(diff: str) -> tuple[int, int]:
    """Return (added, removed) line counts from a unified diff string."""
    added = sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff.splitlines() if l.startswith("-") and not l.startswith("---"))
    return added, removed


def tokenize_diff_hunks(diff: str) -> list[str]:
    """Split a diff into per-hunk strings.

    Each returned string starts with its ``@@ ... @@`` header.
    """
    hunks: list[str] = []
    current: list[str] = []
    for line in diff.splitlines(keepends=True):
        if line.startswith("@@") and current:
            hunks.append("".join(current))
            current = []
        current.append(line)
    if current:
        hunks.append("".join(current))
    return hunks


def extract_test_names(test_source: str) -> list[str]:
    """Return all function names starting with ``test_`` in *test_source*."""
    return re.findall(r"^def (test_\w+)", test_source, re.MULTILINE)


def build_test_signature(test_suite: dict[str, str]) -> str:
    """Concatenate sorted test-function names as a compact signature string.

    This string is used as the boundary anchor alongside the issue text.
    """
    names: list[str] = []
    for src in test_suite.values():
        names.extend(extract_test_names(src))
    return " ".join(sorted(set(names)))


def line_count_ratio(diff_a: str, diff_b: str, tolerance: float = 0.5) -> bool:
    """Return True if the line-count ratio of two diffs is within *tolerance*.

    Used during pair filtering to ensure size distribution is matched.
    """
    a_add, a_rem = diff_line_count(diff_a)
    b_add, b_rem = diff_line_count(diff_b)
    a_total = max(a_add + a_rem, 1)
    b_total = max(b_add + b_rem, 1)
    ratio = min(a_total, b_total) / max(a_total, b_total)
    return ratio >= (1 - tolerance)


def truncate(text: str, max_chars: int) -> str:
    """Truncate *text* to at most *max_chars* characters."""
    return text if len(text) <= max_chars else text[:max_chars - 3] + "..."


def flatten_codebase(
    files: dict[str, str],
    max_chars_per_file: int = 4000,
    max_total_chars: int | None = None,
) -> str:
    """Concatenate all files into a single string for LLM context."""
    parts: list[str] = []
    total = 0
    for path, src in files.items():
        snippet = f"# FILE: {path}\n{truncate(src, max_chars_per_file)}"
        if max_total_chars is not None and total + len(snippet) > max_total_chars:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n\n".join(parts)
