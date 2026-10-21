"""Split retained hunks using static links to fail-to-pass tests.

A link exists when the modified file's parent-directory path is a substring of
at least one fail-to-pass test file's parent-directory path. Linked hunks are
Tier 1; unlinked hunks are Tier 2. Tier 2 does not imply an absence of test coverage.
"""

from __future__ import annotations

from pathlib import Path


def _package_dir(filepath: str) -> str:
    return str(Path(filepath).parent).replace("\\", "/")


def tier_of_file(filepath: str, fail_to_pass_ids: list[str]) -> int:
    """Return 1 for a static test link, otherwise 2."""
    if not filepath:
        return 2

    hunk_dir = _package_dir(filepath)
    linked = any(
        hunk_dir in _package_dir(tid.split("::", 1)[0])
        for tid in fail_to_pass_ids
    )
    return 1 if linked else 2
