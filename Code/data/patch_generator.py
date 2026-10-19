"""LLM-based patch generation with controlled and unconstrained prompts."""

from __future__ import annotations

import difflib
import logging
import re

from .data_loader import DataSample
from .llm_client import LLMClient
from .utils import normalize_diff

log = logging.getLogger(__name__)

# SEARCH/REPLACE format prompts — avoids all unified-diff formatting pitfalls.
# The LLM outputs exact text blocks; we construct the diff with difflib.
_SR_INSTRUCTIONS = (
    "For each location you want to change, output a block in this exact format:\n\n"
    "<<<<<<< SEARCH\n"
    "FILE: path/to/file.py\n"
    "...exact lines to find, including surrounding context (5-10 lines)...\n"
    "=======\n"
    "...replacement lines (same indentation as original)...\n"
    ">>>>>>> REPLACE\n\n"
    "Rules:\n"
    "- The SEARCH content must appear verbatim in the file (copy/paste from the codebase above).\n"
    "- Preserve all whitespace and indentation exactly.\n"
    "- You may emit multiple SEARCH/REPLACE blocks for multiple locations or files.\n"
    "- Do NOT output a unified diff — output ONLY the SEARCH/REPLACE blocks above.\n"
    "- Do NOT include any prose or explanation.\n"
    "- Do NOT use '...' or any other placeholder/ellipsis in SEARCH blocks — every character must be the exact literal text from the file."
)

_CONTROLLED_SYSTEM = (
    "You are a minimal-change patch generator. "
    "Fix ONLY the exact defect described in the issue. "
    "Do NOT refactor, rename, or touch any code unrelated to the bug.\n\n"
    + _SR_INSTRUCTIONS
)

_UNCONSTRAINED_SYSTEM = (
    "You are a coding assistant. Fix the bug described in the issue.\n\n"
    "You may modify any relevant part of the codebase if needed to produce a robust fix.\n\n"
    "The final code should resolve the issue while keeping existing tests passing.\n\n"
    + _SR_INSTRUCTIONS
)

# chars budget for code context: 8192 token window, ~3.5 chars/token (Python is dense),
# minus ~800 tokens for system+issue+instructions, minus 2048 tokens for completion.
_CODE_BUDGET = (8192 - 800 - 2048) * 3  # ≈ 16 032 chars  (used for small-context models)

# Larger budget for long-context models (Claude Sonnet/Haiku: 200K token window).
# Leaves ~5K tokens for system prompt + issue + instructions + 4K for response.
# Allows entire relevant files to be shown without truncation for most repos.
_CLAUDE_CODE_BUDGET = 100_000  # ≈ 28K tokens of code context


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
    """Extract lines around *center_lines*, marking gaps with HTML-style comments.

    Gap markers use <!-- --> syntax so they are unambiguously NOT source code
    and will not be copied into SEARCH/REPLACE blocks by the LLM.
    Default window of 200 lines (400 total per hunk) means most files are shown
    in full; gaps only appear for very large files.
    """
    all_lines = src.splitlines(keepends=False)
    n = len(all_lines)
    include: set[int] = set()
    for cl in center_lines:
        start = max(0, cl - window - 1)
        end = min(n, cl + window)
        include.update(range(start, end))
    if not include:
        return src
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
            if centers and len(src) > budget // 2:
                src = _extract_window(src, centers)
            ordered.append((path, src))
            seen.add(path)
    for path, src in sample.old_codebase.items():
        if path not in seen:
            ordered.append((path, src))

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


# ---------------------------------------------------------------------------
# SEARCH/REPLACE parsing and application
# ---------------------------------------------------------------------------

_SR_BLOCK = re.compile(
    r"<<<<<<< SEARCH\s*\nFILE:\s*(.+?)\n(.*?)=======\n(.*?)>>>>>>> REPLACE",
    re.DOTALL,
)


def _parse_sr_blocks(text: str) -> list[tuple[str, str, str]]:
    """Parse SEARCH/REPLACE blocks → list of (filepath, old_text, new_text)."""
    return [
        (m.group(1).strip(), m.group(2), m.group(3))
        for m in _SR_BLOCK.finditer(text)
    ]


def _fuzzy_find(haystack: str, needle: str) -> int | None:
    """Return the start index of *needle* in *haystack*, with minor whitespace tolerance.

    First tries exact match, then tries after normalizing trailing spaces on each line.
    Returns None if not found.
    """
    if needle in haystack:
        return haystack.index(needle)
    # Normalize trailing whitespace on each line.
    def _strip_trailing(s: str) -> str:
        return "\n".join(line.rstrip() for line in s.splitlines())

    norm_hay = _strip_trailing(haystack)
    norm_need = _strip_trailing(needle)
    if norm_need in norm_hay:
        # Map back to original index (approximate — use line count)
        pre_lines = norm_hay[: norm_hay.index(norm_need)].count("\n")
        orig_lines = haystack.splitlines()
        if pre_lines < len(orig_lines):
            idx = sum(len(l) + 1 for l in orig_lines[:pre_lines])
            return idx
    return None


def _apply_sr_blocks(
    sample: DataSample, blocks: list[tuple[str, str, str]]
) -> str | None:
    """Apply SEARCH/REPLACE blocks to the old codebase and return a unified diff."""
    if not blocks:
        return None

    all_parts: list[str] = []

    for filepath, old_text, new_text in blocks:
        # Resolve the file path against known codebase paths.
        if filepath not in sample.old_codebase:
            candidates = [k for k in sample.old_codebase if k.endswith(filepath)]
            if len(candidates) == 1:
                filepath = candidates[0]
            else:
                log.debug("SR: cannot resolve file '%s' (candidates=%s)", filepath, candidates)
                return None

        original = sample.old_codebase[filepath]
        idx = _fuzzy_find(original, old_text)
        if idx is None:
            log.debug("SR: SEARCH block not found in '%s'", filepath)
            return None

        modified = original[:idx] + new_text + original[idx + len(old_text) :]

        diff_lines = list(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile=f"a/{filepath}",
                tofile=f"b/{filepath}",
            )
        )
        if diff_lines:
            chunk = "".join(diff_lines)
            if not chunk.endswith("\n"):
                chunk += "\n"
            all_parts.append(chunk)

    if not all_parts:
        log.debug("SR: blocks applied but produced no diff (no changes?)")
        return None

    return "".join(all_parts)


class PatchGenerator:
    """Generates controlled and unconstrained patches for a DataSample."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def _build_user_prompt(self, sample: DataSample) -> str:
        code_ctx = _relevant_context(sample)
        reference_files = _extract_diff_file_paths(sample.reference_diff)
        file_hint = (
            f"\nThe file(s) to modify are: {', '.join(reference_files)}\n"
            if reference_files else ""
        )
        return (
            f"## Issue\n{sample.issue_text}\n"
            f"{file_hint}\n"
            f"## Codebase\n{code_ctx}\n\n"
            "Fix the issue using SEARCH/REPLACE blocks as instructed."
        )

    def _generate(self, system: str, sample: DataSample) -> str | None:
        """Call LLM, parse SEARCH/REPLACE blocks, return unified diff or None."""
        user = self._build_user_prompt(sample)
        for attempt in range(2):
            raw = self._llm.complete(system, user)
            if raw is None:
                continue

            blocks = _parse_sr_blocks(raw)
            if blocks:
                diff = _apply_sr_blocks(sample, blocks)
                if diff:
                    log.debug("SR path succeeded (%d block(s), attempt %d)", len(blocks), attempt)
                    return diff
                log.debug("SR blocks parsed but application failed (attempt %d)", attempt)
            else:
                log.debug("No SR blocks found — trying unified diff fallback (attempt %d)", attempt)

            # Fallback: try to parse the response as a unified diff.
            diff = normalize_diff(raw)
            diff = self._repair_missing_header(diff, sample)
            diff = self._repair_placeholder_path(diff, sample)
            if re.search(r"^--- ", diff, re.MULTILINE):
                return diff

        log.debug("Both SR and diff fallback failed after 2 attempts")
        return None

    @staticmethod
    def _repair_missing_header(diff: str, sample: DataSample) -> str:
        """If the diff starts with @@ but lacks --- / +++ headers, add them."""
        if re.match(r"\s*@@", diff):
            reference_files = _extract_diff_file_paths(sample.reference_diff)
            if len(reference_files) == 1:
                path = reference_files[0]
                return f"--- a/{path}\n+++ b/{path}\n{diff}"
        return diff

    @staticmethod
    def _repair_placeholder_path(diff: str, sample: DataSample) -> str:
        """Replace literal 'path/to/file' placeholders with the actual file path."""
        if "path/to/file" not in diff:
            return diff
        reference_files = _extract_diff_file_paths(sample.reference_diff)
        if len(reference_files) != 1:
            return diff
        path = reference_files[0]
        return diff.replace("path/to/file", path)

    def generate_controlled(self, sample: DataSample) -> str | None:
        """Generate a minimal-change patch (used as the scope 'ruler')."""
        return self._generate(_CONTROLLED_SYSTEM, sample)

    def generate_unconstrained(self, sample: DataSample) -> str | None:
        """Generate an unconstrained patch (candidate negative)."""
        return self._generate(_UNCONSTRAINED_SYSTEM, sample)
