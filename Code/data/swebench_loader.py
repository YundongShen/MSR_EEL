"""DataLoader for locally downloaded SWE-bench Lite instances.

Expects the JSONL file produced by ``data_pipeline/download_swebench.py``.

Usage::

    loader = SWEBenchLoader("data/raw/swebench_instances.jsonl")
    for sample in loader:
        print(sample.sample_id, len(sample.old_codebase))
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .data_loader import DataLoader, DataSample


class SWEBenchLoader(DataLoader):
    """Load SWE-bench Lite instances from a local JSONL file.

    Each DataSample produced has:
    - ``old_codebase``: source files touched by the reference patch
    - ``test_suite``: test files referenced by FAIL_TO_PASS / PASS_TO_PASS
    - ``reference_diff``: reference patch from the SWE-bench instance
    - ``metadata``: includes ``repo_path``, ``fail_to_pass``, ``pass_to_pass``
                    needed by :class:`~data.sandbox.SandboxRunner`
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        max_samples: int | None = None,
    ) -> None:
        self._path = Path(jsonl_path)
        if not self._path.exists():
            raise FileNotFoundError(
                f"{self._path} not found. "
                "Run data_pipeline/download_swebench.py first."
            )
        self._max = max_samples
        self._data: list[DataSample] = []
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        with self._path.open() as fh:
            for i, line in enumerate(fh):
                if self._max and i >= self._max:
                    break
                row = json.loads(line)
                self._data.append(self._row_to_sample(row))
        self._loaded = True

    @staticmethod
    def _row_to_sample(row: dict) -> DataSample:
        return DataSample(
            sample_id=row["instance_id"],
            issue_text=row["problem_statement"],
            old_codebase=row["source_files"],
            reference_diff=row["patch"],
            test_suite=row["test_files"],
            metadata={
                "repo": row["repo"],
                "base_commit": row["base_commit"],
                "repo_path": row["repo_path"],
                "fail_to_pass": row["fail_to_pass"],
                "pass_to_pass": row["pass_to_pass"],
                "test_patch": row.get("test_patch", ""),
            },
        )

    def __len__(self) -> int:
        self._load()
        return len(self._data)

    def __iter__(self) -> Iterator[DataSample]:
        self._load()
        yield from self._data
