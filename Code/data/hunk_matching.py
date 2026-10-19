"""Match generated hunks against the reference patch and assign Tier 1 / 2 / 3.

The reference patch is used here and nowhere else in the candidate pipeline: it decides
which generated hunks are retained.

  - A generated hunk is retained (Tier 1/2) when some reference hunk in the same file
    starts within ``MATCH_TOLERANCE`` lines of it (position only, content is not compared).
  - Otherwise it is non-retained (Tier 3).
  - A retained hunk is Tier 1 when its file is statically linked to a fail-to-pass test,
    Tier 2 otherwise (heuristic in data.tier_labeler).
"""

from __future__ import annotations

import re

from .tier_labeler import tier_of_file

MATCH_TOLERANCE = 20    # lines between a generated hunk's start and a reference hunk's start


def parse_hunks(patch: str) -> list[dict]:
    """Split a unified diff into hunk records with file + start_line."""
    hunks: list[dict] = []
    current_file: str | None = None
    current_lines: list[str] = []
    old_start: int | None = None

    for raw in patch.splitlines(keepends=True):
        if raw.startswith("--- "):
            if current_lines and current_file and old_start is not None:
                hunks.append({
                    "filepath": current_file,
                    "old_start": old_start,
                    "lines": list(current_lines),
                })
            current_lines = []
            old_start = None
            m = re.match(r"^--- (?:a/)?(.+)", raw)
            current_file = m.group(1).strip() if m else None
            continue
        if raw.startswith("+++ "):
            if current_file == "/dev/null":
                m = re.match(r"^\+\+\+ (?:b/)?(.+)", raw)
                current_file = m.group(1).strip() if m else current_file
            continue
        if raw.startswith("@@"):
            if current_lines and current_file and old_start is not None:
                hunks.append({
                    "filepath": current_file,
                    "old_start": old_start,
                    "lines": list(current_lines),
                })
            current_lines = [raw]
            m = re.search(r"@@ -(\d+)", raw)
            old_start = int(m.group(1)) if m else 1
        elif current_lines is not None:
            current_lines.append(raw)

    if current_lines and current_file and old_start is not None:
        hunks.append({
            "filepath": current_file,
            "old_start": old_start,
            "lines": list(current_lines),
        })
    return hunks


def matches_reference(
    filepath: str,
    old_start: int,
    reference_hunks: list[dict],
    tolerance: int = MATCH_TOLERANCE,
) -> bool:
    """True if a reference hunk in the same file starts within ``tolerance`` lines."""
    return any(
        rh["filepath"] == filepath and abs(rh["old_start"] - old_start) < tolerance
        for rh in reference_hunks
    )


def extract_context(
    source_files: dict[str, str],
    filepath: str,
    old_start: int,
    hunk_lines: list[str],
    ctx: int = 5,
) -> tuple[list[str], list[str]]:
    """Extract ctx lines before and after the hunk from source_files."""
    if filepath not in source_files:
        return [], []
    src = source_files[filepath].splitlines()
    before_end = max(0, old_start - 1)
    before_start = max(0, before_end - ctx)
    context_before = src[before_start:before_end]

    old_count = sum(
        1 for l in hunk_lines[1:]
        if l.startswith(" ") or l.startswith("-")
    )
    after_start = old_start - 1 + old_count
    context_after = src[after_start: after_start + ctx]
    return context_before, context_after


def retained_tier(filepath: str, fail_to_pass_ids: list[str]) -> int:
    """Tier 1 or 2 of a retained hunk in ``filepath`` (file-level test-link heuristic)."""
    return tier_of_file(filepath, fail_to_pass_ids)
