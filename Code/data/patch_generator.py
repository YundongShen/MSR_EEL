"""Build source-code context around reference-patch hunks."""

from __future__ import annotations

import re

from .data_loader import DataSample

_CODE_BUDGET = (8192 - 800 - 2048) * 3
_CLAUDE_CODE_BUDGET = 100_000


def _extract_diff_file_paths(diff: str) -> list[str]:
    """Return the list of files modified by *diff* (strips a/ b/ prefixes)."""
    paths: list[str] = []
    for m in re.finditer(r"^--- (?:a/)?(.*)", diff, re.MULTILINE):
        p = m.group(1).strip()
        if p and p != "/dev/null":
            paths.append(p)
    return paths


def _hunk_start_lines(diff: str, filepath: str) -> list[int]:
    """Return the start line numbers of hunks for *filepath* in *diff*."""
    lines: list[int] = []
    in_file = False
    for line in diff.splitlines():
        if re.match(r"^--- (?:a/)?", line) and filepath in line:
            in_file = True
        elif line.startswith("--- "):
            in_file = False
        if in_file and line.startswith("@@"):
            m = re.search(r"@@ -(\d+)", line)
            if m:
                lines.append(int(m.group(1)))
    return lines


def _extract_window(src: str, center_lines: list[int], window: int = 200) -> str:
    """Extract up to 200 lines before and after each reference hunk start."""
    all_lines = src.splitlines(keepends=False)
    n = len(all_lines)
    include: set[int] = set()
    for cl in center_lines:
        start = max(0, cl - window - 1)
        end = min(n, cl + window)
        include.update(range(start, end))
    if not include:
        return ""
    segments: list[str] = []
    prev: int | None = None
    for i in sorted(include):
        if prev is None or i > prev + 1:
            segments.append(f"<!-- omitted: jump to line {i + 1} -->")
        segments.append(all_lines[i])
        prev = i
    return "\n".join(segments)


def _relevant_context(sample: DataSample, budget: int = _CODE_BUDGET) -> str:
    """Build a compact code context focused on files and hunks from the reference diff."""
    reference_files = _extract_diff_file_paths(sample.reference_diff)

    ordered: list[tuple[str, str]] = []
    seen: set[str] = set()
    for path in reference_files:
        if path in sample.old_codebase and path not in seen:
            src = sample.old_codebase[path]
            centers = _hunk_start_lines(sample.reference_diff, path)
            if not centers:
                continue
            src = _extract_window(src, centers)
            ordered.append((path, src))
            seen.add(path)

    parts: list[str] = []
    remaining = budget
    for path, src in ordered:
        header = f"# FILE: {path}\n"
        space = remaining - len(header) - 4
        if space <= 0:
            break
        snippet = src[:space]
        parts.append(header + snippet)
        remaining -= len(header) + len(snippet) + 2
        if remaining <= 0:
            break
    return "\n\n".join(parts)
