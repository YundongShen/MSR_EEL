"""Static test-link heuristic that splits retained hunks into Tier 1 and Tier 2.

Tier assignment (offline, no test execution required):

  Tier 1 — the hunk's file is likely covered by at least one fail-to-pass test,
            inferred from import statements in the test files and directory co-location.
  Tier 2 — no such inferred coverage link ("necessary but untested" changes).

Heuristic:
  A file F is Tier 1 if any fail-to-pass test file T satisfies:
    (a) T imports a module that is a prefix of F's dotted module name, OR
    (b) F and T share the same package directory (co-location).

The label is a property of the hunk's file, so it is applied to the generated retained
hunks when data_pipeline/build_dataset.py builds the dataset.
"""

from __future__ import annotations

import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _path_to_module(filepath: str) -> str:
    """Convert a/b/c.py → a.b.c (drop .py suffix)."""
    return Path(filepath).with_suffix("").as_posix().replace("/", ".")


def _extract_imports(source: str) -> set[str]:
    """Return all top-level module names referenced in import statements."""
    modules: set[str] = set()
    for m in re.finditer(
        r"^(?:from|import)\s+([\w.]+)", source, re.MULTILINE
    ):
        modules.add(m.group(1))
    return modules


def _package_dir(filepath: str) -> str:
    """Return the parent directory of filepath (normalised to forward slashes)."""
    return str(Path(filepath).parent).replace("\\", "/")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tier_of_file(
    filepath: str,
    fail_to_pass_ids: list[str],
    raw_test_sources: dict[str, str] | None = None,
) -> int:
    """Return 1 if ``filepath`` is linked to a fail-to-pass test, else 2.

    Parameters
    ----------
    filepath:
        Repo-relative path of the file the hunk modifies.
    fail_to_pass_ids:
        SWE-bench fail-to-pass test ids (``path/to/test_file.py::test_name``).
    raw_test_sources:
        Optional mapping test filepath → full source.  If provided, import-based
        coverage is checked in addition to directory co-location (more accurate).
    """
    if not filepath:
        return 2

    test_paths: set[str] = {tid.split("::")[0] for tid in fail_to_pass_ids}
    test_modules: set[str] = set()
    test_dirs: set[str] = set()
    for tp in test_paths:
        test_dirs.add(_package_dir(tp))
        if raw_test_sources and tp in raw_test_sources:
            test_modules |= _extract_imports(raw_test_sources[tp])

    hunk_module = _path_to_module(filepath)
    hunk_dir = _package_dir(filepath)

    covered = False

    # (a) import-based coverage
    if test_modules:
        covered = any(
            hunk_module == imp or hunk_module.startswith(imp + ".")
            for imp in test_modules
        )

    # (b) directory co-location
    if not covered:
        covered = any(
            hunk_dir in td or td.startswith(hunk_dir)
            for td in test_dirs
        )

    return 1 if covered else 2
