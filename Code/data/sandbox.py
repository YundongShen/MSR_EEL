"""Subprocess-based sandbox for applying diffs and collecting execution traces.

Writes a standalone pytest plugin (ebd_tracer_plugin.py) into the temp repo
and loads it via ``-p ebd_tracer_plugin``.  This avoids touching the repo's
existing conftest.py, which may have missing dependencies.

Safe to run on HPC nodes — no Docker required.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .execution_tracer import ExecutionTrace, FunctionEvent

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Standalone pytest plugin written into the temp repo
# Loaded via: pytest -p ebd_tracer_plugin
# Does NOT touch the repo's existing conftest.py.
# ---------------------------------------------------------------------------

_PLUGIN_CONTENT = '''\
import sys, os, json, collections, collections.abc

# Python 3.10+ compatibility: restore deprecated collections aliases.
for _name in ("Callable", "Mapping", "MutableMapping", "MutableSequence",
              "Sequence", "Set", "MutableSet", "Iterable", "Iterator",
              "Generator", "Coroutine", "AsyncGenerator", "AsyncIterable",
              "AsyncIterator", "Hashable", "Sized", "Container", "Awaitable"):
    if not hasattr(collections, _name):
        _obj = getattr(collections.abc, _name, None)
        if _obj is not None:
            setattr(collections, _name, _obj)

_ebd_events = []
_ebd_covered = {}
_ebd_filter = os.environ.get("EBD_FILTER_PREFIX", "")
_ebd_output = os.environ.get("EBD_TRACE_OUTPUT", "/tmp/ebd_trace.json")
_ebd_max = int(os.environ.get("EBD_MAX_EVENTS", "20000"))


def _ebd_trace(frame, event, arg):
    if len(_ebd_events) >= _ebd_max:
        return None
    fname = frame.f_code.co_filename
    if _ebd_filter and _ebd_filter not in fname:
        return _ebd_trace
    line = frame.f_lineno
    _ebd_covered.setdefault(fname, []).append(line)
    if event == "call":
        _ebd_events.append({"k": "call", "f": fname, "n": frame.f_code.co_name, "l": line})
    elif event == "return":
        try:
            r = repr(arg)[:80]
        except Exception:
            r = "<repr-error>"
        _ebd_events.append({"k": "return", "f": fname, "n": frame.f_code.co_name, "l": line, "r": r})
    elif event == "exception":
        try:
            r = arg[0].__name__ if arg[0] else ""
        except Exception:
            r = "<exc-error>"
        _ebd_events.append({"k": "exception", "f": fname, "n": frame.f_code.co_name, "l": line, "r": r})
    return _ebd_trace


def pytest_sessionstart(session):
    sys.settrace(_ebd_trace)


def pytest_sessionfinish(session, exitstatus):
    sys.settrace(None)
    data = {"events": _ebd_events,
            "covered": {k: list(set(v)) for k, v in _ebd_covered.items()}}
    with open(_ebd_output, "w") as fh:
        json.dump(data, fh)
'''

_PLUGIN_FILENAME = "ebd_tracer_plugin.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_patch(repo_path: Path, diff_str: str) -> bool:
    """Apply a unified diff to *repo_path* using the system ``patch`` command."""
    for strip in ("1", "0"):
        result = subprocess.run(
            ["patch", f"-p{strip}", "--batch", "--forward", "--fuzz=3",
             "--reject-file=-"],
            input=diff_str.encode(),
            cwd=repo_path,
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0:
            return True
        log.info("patch -p%s failed | stdout: %s | stderr: %s",
                 strip,
                 result.stdout.decode(errors="replace")[:200],
                 result.stderr.decode(errors="replace")[:150])
    log.info("diff_head: %s", diff_str[:300])
    return False


def _write_plugin(repo_path: Path) -> None:
    (repo_path / _PLUGIN_FILENAME).write_text(_PLUGIN_CONTENT)


def _inject_conftest(repo_path: Path) -> None:
    """Append our tracer to the repo's conftest.py (or create one).

    Preferred over ``-p`` flag to avoid a pytest 9 bug with plugin metadata.
    """
    conftest = repo_path / "conftest.py"
    existing = conftest.read_text(errors="replace") if conftest.exists() else ""
    if "EBD_TRACE_OUTPUT" not in existing:
        conftest.write_text(existing + "\n" + _PLUGIN_CONTENT)


def _parse_trace_json(path: Path, repo_root: Path | None = None) -> ExecutionTrace:
    data = json.loads(path.read_text())

    def _norm(fname: str) -> str:
        """Make path relative to repo_root so traces from different tmp dirs compare equal."""
        if repo_root is None:
            return fname
        try:
            return str(Path(fname).relative_to(repo_root))
        except ValueError:
            return fname  # outside repo (stdlib etc.) — keep absolute

    events = [
        FunctionEvent(
            kind=e["k"],
            filename=_norm(e["f"]),
            funcname=e["n"],
            lineno=e["l"],
            return_repr=e.get("r", ""),
        )
        for e in data.get("events", [])
    ]
    covered = {_norm(k): set(v) for k, v in data.get("covered", {}).items()}
    return ExecutionTrace(events=events, covered_lines=covered)


# ---------------------------------------------------------------------------
# SandboxRunner
# ---------------------------------------------------------------------------

class SandboxRunner:
    """Run test suites with execution trace collection in a subprocess sandbox.

    Parameters
    ----------
    timeout:
        Wall-clock seconds allowed for the pytest subprocess.
    filter_prefix:
        Only trace files whose path contains this string.
        Leave empty to trace everything (slow but complete).
    python_executable:
        Python to use inside the sandbox (default: current interpreter).
    """

    def __init__(
        self,
        timeout: int = 120,
        filter_prefix: str = "",
        python_executable: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.filter_prefix = filter_prefix
        self.python = python_executable or shutil.which("python3.12") or shutil.which("python3") or "python"

    def run_with_trace(
        self,
        repo_path: str | Path,
        test_ids: list[str],
        patch_str: str | None = None,
        prerequisite_patch: str | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> tuple[bool, ExecutionTrace]:
        """Run *test_ids* in a copy of *repo_path*, optionally after applying patches.

        Parameters
        ----------
        repo_path:
            Path to the cloned repository at base_commit.
        test_ids:
            Pytest node IDs to run, e.g. ``["tests/test_foo.py::test_bar"]``.
        patch_str:
            Unified diff to apply (the LLM patch being evaluated).  ``None`` = baseline.
        prerequisite_patch:
            Diff always applied BEFORE *patch_str* (e.g. SWE-bench test_patch that
            adds the FAIL_TO_PASS test functions to the repo).
        extra_env:
            Additional environment variables for the subprocess.

        Returns
        -------
        (passed, trace)
            *passed* is True iff pytest exits 0.
            *trace* is the collected execution trace.
        """
        repo_path = Path(repo_path)
        if not repo_path.is_dir():
            log.error("repo_path does not exist: %s", repo_path)
            return False, ExecutionTrace(error=f"repo_path not found: {repo_path}")

        with tempfile.TemporaryDirectory(prefix="ebd_sandbox_") as tmp_str:
            tmp = Path(tmp_str)
            work = tmp / "repo"

            # Copy repo to isolated work dir.
            shutil.copytree(repo_path, work, symlinks=True)

            # Apply prerequisite patch first (e.g. test_patch that adds test functions).
            if prerequisite_patch:
                if not apply_patch(work, prerequisite_patch):
                    log.warning("prerequisite patch application failed — continuing anyway")

            # Apply the main patch (LLM-generated fix being evaluated).
            if patch_str:
                if not apply_patch(work, patch_str):
                    return False, ExecutionTrace(error="patch application failed")

            # Inject tracer into conftest.py (avoids pytest 9 plugin-metadata bug).
            _inject_conftest(work)

            # Trace output file.
            trace_out = tmp / "trace.json"

            env = os.environ.copy()
            env["EBD_TRACE_OUTPUT"] = str(trace_out)
            env["EBD_FILTER_PREFIX"] = self.filter_prefix
            # Make the repo importable without pip install -e .
            existing_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(work) + (os.pathsep + existing_pp if existing_pp else "")
            if extra_env:
                env.update(extra_env)

            cmd = [
                self.python, "-m", "pytest",
                "--override-ini=addopts=",           # clear repo-level addopts (e.g. --doctest-rst)
                "--override-ini=filterwarnings=",    # clear repo-level warning filters
                "--tb=no", "--no-header", "-q",
                "--timeout=60",
                "-W", "ignore",
                *test_ids,
            ]

            try:
                result = subprocess.run(
                    cmd,
                    cwd=work,
                    env=env,
                    capture_output=True,
                    timeout=self.timeout,
                )
                passed = result.returncode == 0
                if not passed:
                    log.info("pytest rc=%d stdout=%r stderr=%r",
                             result.returncode,
                             result.stdout.decode(errors="replace")[:400],
                             result.stderr.decode(errors="replace")[:400])
            except subprocess.TimeoutExpired:
                log.warning("pytest timed out for %s", repo_path.name)
                passed = False
                result = None

            if trace_out.exists():
                try:
                    trace = _parse_trace_json(trace_out, repo_root=work)
                except Exception as exc:
                    trace = ExecutionTrace(error=f"trace parse error: {exc}")
            else:
                stderr = result.stderr.decode()[:300] if result else "timeout"
                trace = ExecutionTrace(error=f"no trace written — {stderr}")

        return passed, trace

    def baseline_trace(
        self,
        repo_path: str | Path,
        test_ids: list[str],
        prerequisite_patch: str | None = None,
    ) -> tuple[bool, ExecutionTrace]:
        """Run tests without any source patch to get the reference (failing) trace."""
        return self.run_with_trace(
            repo_path, test_ids,
            patch_str=None,
            prerequisite_patch=prerequisite_patch,
        )
