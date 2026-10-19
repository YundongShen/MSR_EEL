"""Generic data loading interface for Issue–PR pairs.

Concrete implementations can target GitHub, local JSON dumps, or mock data.
Each implementation returns :class:`DataSample` objects consumed by the rest
of the pipeline.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class DataSample:
    """One Issue–PR data point."""

    sample_id: str
    issue_text: str
    old_codebase: dict[str, str]     # filename → source
    reference_diff: str                  # unified-diff string
    test_suite: dict[str, str]        # test filename → source
    metadata: dict = field(default_factory=dict)

    def relevant_files(self) -> list[str]:
        """Files touched by the reference diff."""
        touched: list[str] = []
        for line in self.reference_diff.splitlines():
            if line.startswith("--- a/") or line.startswith("+++ b/"):
                fname = line.split("/", 1)[-1]
                touched.append(fname)
        return list(dict.fromkeys(touched))  # deduplicated, order-preserving


class DataLoader(ABC):
    """Abstract base for dataset loaders."""

    @abstractmethod
    def __iter__(self) -> Iterator[DataSample]:
        """Yield :class:`DataSample` objects one at a time."""

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of samples (may be approximate for streaming)."""

    def as_list(self) -> list[DataSample]:
        return list(self)


# ---------------------------------------------------------------------------
# JSON-backed loader (for pre-downloaded / cached datasets)
# ---------------------------------------------------------------------------

class JSONDataLoader(DataLoader):
    """Load samples from a JSONL file.

    Expected line format::

        {
            "id": "...",
            "issue": "...",
            "old_files": {"path/to/file.py": "..."},
            "diff": "...",
            "tests": {"tests/test_foo.py": "..."}
        }
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(self._path)
        self._lines: list[dict] = []
        with self._path.open() as fh:
            for raw in fh:
                raw = raw.strip()
                if raw:
                    self._lines.append(json.loads(raw))

    def __len__(self) -> int:
        return len(self._lines)

    def __iter__(self) -> Iterator[DataSample]:
        for row in self._lines:
            yield DataSample(
                sample_id=row["id"],
                issue_text=row["issue"],
                old_codebase=row.get("old_files", {}),
                reference_diff=row["diff"],
                test_suite=row.get("tests", {}),
                metadata=row.get("meta", {}),
            )


# ---------------------------------------------------------------------------
# GitHub loader (requires GITHUB_TOKEN in environment)
# ---------------------------------------------------------------------------

class GitHubDataLoader(DataLoader):
    """Fetch closed PRs that reference an issue from a GitHub repository.

    Only PRs whose bodies contain a ``Fixes #<N>`` or ``Closes #<N>`` link are
    included; the referenced issue text becomes ``issue_text``.

    Requires:
        pip install PyGithub

    Environment:
        GITHUB_TOKEN — personal access token with ``repo`` scope.
    """

    def __init__(
        self,
        repo_slug: str,
        max_samples: int = 500,
        cache_path: str | Path | None = None,
    ) -> None:
        self._repo_slug = repo_slug
        self._max_samples = max_samples
        self._cache_path = Path(cache_path) if cache_path else None
        self._samples: list[DataSample] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch(self) -> list[DataSample]:
        try:
            from github import Github  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("Install PyGithub: pip install PyGithub") from exc

        token = os.environ.get("GITHUB_TOKEN")
        g = Github(token)
        repo = g.get_repo(self._repo_slug)

        samples: list[DataSample] = []
        for pr in repo.get_pulls(state="closed", sort="updated", direction="desc"):
            if len(samples) >= self._max_samples:
                break
            issue_num = self._extract_issue_ref(pr.body or "")
            if issue_num is None:
                continue
            try:
                issue = repo.get_issue(issue_num)
            except Exception:
                continue

            files = {f.filename: (f.patch or "") for f in pr.get_files()}
            diff = "\n".join(
                f"--- a/{fn}\n+++ b/{fn}\n{patch}"
                for fn, patch in files.items()
            )
            samples.append(
                DataSample(
                    sample_id=f"{self._repo_slug}#PR{pr.number}",
                    issue_text=f"{issue.title}\n\n{issue.body or ''}",
                    old_codebase={},   # populated on demand; requires extra API calls
                    reference_diff=diff,
                    test_suite={},
                )
            )
        return samples

    @staticmethod
    def _extract_issue_ref(body: str) -> int | None:
        import re
        m = re.search(r"(?:fixes|closes|resolves)\s+#(\d+)", body, re.IGNORECASE)
        return int(m.group(1)) if m else None

    # ------------------------------------------------------------------
    # DataLoader interface
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._samples is not None:
            return
        if self._cache_path and self._cache_path.exists():
            loader = JSONDataLoader(self._cache_path)
            self._samples = loader.as_list()
        else:
            self._samples = self._fetch()
            if self._cache_path:
                self._write_cache()

    def _write_cache(self) -> None:
        assert self._samples is not None
        assert self._cache_path is not None
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("w") as fh:
            for s in self._samples:
                row = {
                    "id": s.sample_id,
                    "issue": s.issue_text,
                    "old_files": s.old_codebase,
                    "diff": s.reference_diff,
                    "tests": s.test_suite,
                }
                fh.write(json.dumps(row) + "\n")

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._samples)  # type: ignore[arg-type]

    def __iter__(self) -> Iterator[DataSample]:
        self._ensure_loaded()
        yield from self._samples  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Mock loader for unit-testing without network access
# ---------------------------------------------------------------------------

class MockDataLoader(DataLoader):
    """Deterministic in-memory loader for tests and CI."""

    _MOCK_ISSUE = (
        "BUG: `calculate_discount` returns wrong value for premium users.\n"
        "Expected: 20% off. Actual: 10% off."
    )
    _MOCK_OLD_CODE = {
        "billing/discounts.py": (
            "def calculate_discount(user, price):\n"
            "    if user.tier == 'premium':\n"
            "        return price * 0.10\n"
            "    return price\n"
        )
    }
    _MOCK_DIFF = (
        "--- a/billing/discounts.py\n"
        "+++ b/billing/discounts.py\n"
        "@@ -2,3 +2,3 @@\n"
        "     if user.tier == 'premium':\n"
        "-        return price * 0.10\n"
        "+        return price * 0.20\n"
    )
    _MOCK_TESTS = {
        "tests/test_discounts.py": (
            "def test_premium_discount():\n"
            "    from billing.discounts import calculate_discount\n"
            "    class U: tier = 'premium'\n"
            "    assert calculate_discount(U(), 100) == 80\n"
        )
    }

    def __init__(self, n: int = 10) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __iter__(self) -> Iterator[DataSample]:
        for i in range(self._n):
            yield DataSample(
                sample_id=f"mock-{i:04d}",
                issue_text=self._MOCK_ISSUE,
                old_codebase=dict(self._MOCK_OLD_CODE),
                reference_diff=self._MOCK_DIFF,
                test_suite=dict(self._MOCK_TESTS),
            )
