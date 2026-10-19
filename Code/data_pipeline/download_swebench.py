"""Download SWE-bench Lite and clone repos at their base commits.

Run once on a node with internet access:

    python data_pipeline/download_swebench.py --output data/raw/swebench_instances.jsonl

Each output line (JSONL):
{
  "instance_id": "django__django-11999",
  "repo": "django/django",
  "base_commit": "abc123",
  "problem_statement": "...",
  "patch": "--- a/... +++ b/...",
  "test_patch": "...",
  "fail_to_pass": ["tests/test_foo.py::test_bar"],
  "pass_to_pass": ["tests/test_foo.py::test_baz"],
  "repo_path": "data/repos/django__django-11999",
  "source_files": {"path/to/file.py": "content ..."},
  "test_files": {"tests/test_foo.py": "content ..."},
}
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GITHUB_BASE = "https://github.com/{repo}.git"
HF_DATASET = "princeton-nlp/SWE-bench_Lite"


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def clone_at_commit(repo_slug: str, commit: str, dest: Path) -> bool:
    """Clone a GitHub repo at a specific commit into dest. Returns success."""
    if dest.exists():
        log.info("Already cloned: %s", dest)
        return True

    url = GITHUB_BASE.format(repo=repo_slug)
    dest.mkdir(parents=True, exist_ok=True)

    # Minimal fetch: only the specific commit (no full history).
    cmds = [
        ["git", "init"],
        ["git", "remote", "add", "origin", url],
        ["git", "fetch", "--depth", "1", "origin", commit],
        ["git", "checkout", "FETCH_HEAD"],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, cwd=dest, capture_output=True, timeout=120)
        if result.returncode != 0:
            log.error("git cmd failed %s: %s", cmd, result.stderr.decode()[:300])
            shutil.rmtree(dest, ignore_errors=True)
            return False
    return True


def read_file_safe(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Patch parsing
# ---------------------------------------------------------------------------

def files_from_diff(diff: str) -> list[str]:
    """Extract file paths touched by a unified diff."""
    return re.findall(r"^--- a/(.+)$", diff, re.MULTILINE)


def test_files_from_ids(test_ids: list[str]) -> list[str]:
    """Extract unique file paths from pytest node IDs like 'tests/foo.py::bar'."""
    seen, paths = set(), []
    for tid in test_ids:
        p = tid.split("::")[0]
        if p.endswith(".py") and p not in seen:
            seen.add(p)
            paths.append(p)
    return paths


def resolve_test_ids(
    test_ids: list[str],
    repo_path: Path,
    test_patch: str = "",
) -> list[str]:
    """Ensure test IDs are in 'file::func' format.

    Resolution order for bare function names:
    1. Already 'file::func' → keep as-is.
    2. Function *added* by test_patch (+def line) → use that file.
    3. Function exists in a file *modified* by test_patch (+++ b/ header) → prefer that.
    4. Grep the repo, prefer test_patch files, else shortest match.
    """
    patch_files = re.findall(r"^\+\+\+ b/(.+)$", test_patch, re.MULTILINE)
    patch_test_files = [f for f in patch_files if "/test" in f or "test_" in f]

    patch_func_to_file: dict[str, str] = {}
    current_file = ""
    for line in test_patch.splitlines():
        m = re.match(r"^\+\+\+ b/(.+)$", line)
        if m:
            current_file = m.group(1)
        elif line.startswith("+") and re.match(r"^\+def (test_\w+)", line):
            func_name = re.match(r"^\+def (test_\w+)", line).group(1)
            patch_func_to_file[func_name] = current_file

    resolved = []
    for tid in test_ids:
        if "::" in tid:
            resolved.append(tid)
            continue

        # Priority 1: unittest dotted format "test_func (module.path.ClassName)"
        # Parse module path directly — no grep needed.
        m_unittest = re.match(r"^(\w+)\s+\(([^)]+)\)$", tid)
        if m_unittest:
            func = m_unittest.group(1)
            dotted = m_unittest.group(2)
            # Last component is the class name; the rest is the module path.
            module = dotted.rsplit(".", 1)[0] if "." in dotted else dotted
            rel_path = module.replace(".", "/") + ".py"
            for prefix in ("", "tests/"):
                candidate = repo_path / prefix / rel_path
                if candidate.exists():
                    resolved.append(f"{prefix}{rel_path}::{func}")
                    break
            else:
                resolved.append(f"{rel_path}::{func}")  # best guess
            continue

        func = tid

        # Priority 2: function added by test_patch.
        if func in patch_func_to_file:
            resolved.append(f"{patch_func_to_file[func]}::{func}")
            continue

        # Priority 3: function already exists in a file modified by test_patch.
        found_in_patch_file = None
        for pf in patch_test_files:
            full = repo_path / pf
            if full.exists():
                try:
                    if f"def {func}" in full.read_text(errors="replace"):
                        found_in_patch_file = pf
                        break
                except Exception:
                    pass
        if found_in_patch_file:
            resolved.append(f"{found_in_patch_file}::{func}")
            continue

        # Priority 4: unresolved — keep as bare name; sandbox will filter it out.
        resolved.append(func)
    return resolved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_instance(instance: dict, repos_dir: Path) -> dict | None:
    iid = instance["instance_id"]
    repo = instance["repo"]
    commit = instance["base_commit"]
    dest = repos_dir / iid

    if not clone_at_commit(repo, commit, dest):
        log.warning("Skipping %s — clone failed", iid)
        return None

    patch: str = instance.get("patch", "")
    test_patch: str = instance.get("test_patch", "")
    fail_to_pass: list[str] = json.loads(instance.get("FAIL_TO_PASS", "[]"))
    pass_to_pass: list[str] = json.loads(instance.get("PASS_TO_PASS", "[]"))

    # Collect source files touched by the reference patch.
    source_paths = files_from_diff(patch)
    # Collect test files from test IDs.
    test_paths = test_files_from_ids(fail_to_pass + pass_to_pass)
    # Also include test files modified by test_patch.
    test_paths += files_from_diff(test_patch)
    test_paths = list(dict.fromkeys(test_paths))  # dedup, preserve order

    # Resolve bare function names to file::func pytest node IDs.
    # Only resolve fail_to_pass — pass_to_pass is not used by the sandbox
    # and resolving it is prohibitively slow on large repos like django.
    fail_to_pass = resolve_test_ids(fail_to_pass, dest, test_patch=test_patch)

    # Re-derive test file paths after resolution.
    test_paths = test_files_from_ids(fail_to_pass)
    test_paths += files_from_diff(test_patch)
    test_paths = list(dict.fromkeys(test_paths))

    source_files: dict[str, str] = {}
    for rel in source_paths:
        content = read_file_safe(dest / rel)
        if content is not None:
            source_files[rel] = content

    test_files: dict[str, str] = {}
    for rel in test_paths:
        content = read_file_safe(dest / rel)
        if content is not None:
            test_files[rel] = content

    if not source_files:
        log.warning("No source files found for %s — skipping", iid)
        return None

    return {
        "instance_id": iid,
        "repo": repo,
        "base_commit": commit,
        "problem_statement": instance.get("problem_statement", ""),
        "patch": patch,
        "test_patch": test_patch,
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
        "repo_path": str(dest),
        "source_files": source_files,
        "test_files": test_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SWE-bench + clone repos")
    parser.add_argument("--output", default="data/raw/swebench_instances.jsonl")
    parser.add_argument("--repos-dir", default="data/repos")
    parser.add_argument("--max", type=int, default=None, help="Limit number of instances")
    parser.add_argument("--split", default="test", help="HuggingFace split name")
    parser.add_argument(
        "--dataset", default=HF_DATASET,
        help="HuggingFace dataset name, e.g. princeton-nlp/SWE-bench or princeton-nlp/SWE-bench_Lite",
    )
    parser.add_argument(
        "--repo-filter", default=None,
        help="Only process instances from this repo slug, e.g. 'sympy/sympy'",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError:
        sys.exit("Install HuggingFace datasets: pip install datasets")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    repos_dir = Path(args.repos_dir)
    repos_dir.mkdir(parents=True, exist_ok=True)

    # Resume: skip instance_ids already written to the output file.
    done_ids: set[str] = set()
    if out_path.exists():
        with out_path.open() as fh:
            for line in fh:
                try:
                    done_ids.add(json.loads(line)["instance_id"])
                except Exception:
                    pass
        if done_ids:
            log.info("Resuming: %d instances already in %s — skipping them", len(done_ids), out_path)

    log.info("Loading %s …", args.dataset)
    ds = load_dataset(args.dataset, split=args.split)

    # Optionally filter to a specific repo.
    if args.repo_filter:
        ds = [r for r in ds if r["repo"] == args.repo_filter]
        log.info("Filtered to %d instances from %s", len(ds), args.repo_filter)

    ok = skip = fail = 0
    with out_path.open("a") as fh:
        for i, instance in enumerate(ds):
            if args.max and ok >= args.max:
                break
            iid = instance["instance_id"]
            if iid in done_ids:
                skip += 1
                continue
            log.info("[%d/%d] %s  (ok=%d skip=%d fail=%d)", i + 1, len(ds), iid, ok, skip, fail)
            try:
                result = process_instance(instance, repos_dir)
            except Exception as exc:
                log.error("%s: unexpected error — %s", iid, exc)
                fail += 1
                continue
            if result:
                fh.write(json.dumps(result) + "\n")
                fh.flush()
                ok += 1
            else:
                fail += 1

    log.info("Done. OK=%d  SKIPPED=%d  FAILED=%d  → %s", ok, skip, fail, out_path)


if __name__ == "__main__":
    main()
