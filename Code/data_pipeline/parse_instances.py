"""
Parse SWE-bench JSONL instances into structured training data for Edit Entailment Learning.

Extracts the REQ / TEST / ORIG entities per instance (all from JSONL, no repo needed):
  REQ  - requirement text (problem_statement)
  TEST - fail-to-pass test function bodies, as they are AFTER the reference test patch is applied
         (most fail-to-pass tests are added by the test patch: they do not exist in the base commit)
  ORIG - original project code units (functions/classes from source_files)
  (HUNK entities are NOT stored here: candidate hunks are LLM-generated, see data_pipeline/build_dataset.py.
   The reference patch's own hunks are deliberately not written to the instances file.)

Output: data/processed/instances_lite.jsonl or instances_full.jsonl
        One line per instance, all entity types included.
Resumable: skips already-written instance_ids on restart.

Usage:
  # Experimental (Lite, 82 instances):
  python data_pipeline/parse_instances.py \
      --input data/raw/swebench_instances.jsonl \
      --output data/processed/instances_lite.jsonl

  # Later (Full, 2294 instances):
  python data_pipeline/parse_instances.py \
      --input data/raw/swebench_full_instances.jsonl \
      --output data/processed/instances_full.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Source code unit extraction
# ---------------------------------------------------------------------------

def _extract_code_units(filepath: str, source: str) -> list[dict]:
    """Extract top-level and class-level functions/classes via AST."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    src_lines = source.splitlines()
    units = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if not hasattr(node, "end_lineno"):
            continue
        start = node.lineno - 1
        end = node.end_lineno
        units.append({
            "filepath": filepath,
            "name": node.name,
            "kind": "class" if isinstance(node, ast.ClassDef) else "function",
            "start_line": node.lineno,
            "end_line": end,
            "code": "\n".join(src_lines[start:end]),
        })
    return units


# ---------------------------------------------------------------------------
# Test patch: the fail-to-pass tests are (mostly) added by it
# ---------------------------------------------------------------------------

_DIFF_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.MULTILINE)


def _split_file_diffs(patch: str) -> list[dict]:
    """Split a git diff into one record per file."""
    starts = [m.start() for m in _DIFF_HEADER.finditer(patch)] + [len(patch)]
    out = []
    for begin, end in zip(starts, starts[1:]):
        chunk = patch[begin:end]
        m = _DIFF_HEADER.match(chunk)
        out.append({
            "old": m.group(1), "new": m.group(2), "text": chunk,
            "added": bool(re.search(r"^--- /dev/null$", chunk, re.MULTILINE)),
            "deleted": bool(re.search(r"^\+\+\+ /dev/null$", chunk, re.MULTILINE)),
        })
    return out


def apply_test_patch(test_files: dict[str, str], test_patch: str) -> dict[str, str]:
    """Test files as they are after the reference test patch is applied.

    ``test_files`` holds the BASE version of the test files.  Files the patch adds need no base version; a
    file whose base version is not available, a renamed or a deleted file is left as it is.  A file diff that
    does not apply is skipped, so the base version is kept for that file.
    """
    post = dict(test_files)
    todo = [d for d in _split_file_diffs(test_patch or "")
            if not d["deleted"] and d["old"] == d["new"] and (d["added"] or d["new"] in test_files)]
    if not todo:
        return post
    with tempfile.TemporaryDirectory() as tmp:
        for d in todo:
            if not d["added"]:
                path = Path(tmp) / d["new"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(test_files[d["new"]].encode("utf-8"))
        for d in todo:
            done = subprocess.run(["git", "apply", "--whitespace=nowarn", "-"], input=d["text"].encode("utf-8"),
                                  cwd=tmp, capture_output=True)
            path = Path(tmp) / d["new"]
            if done.returncode == 0 and path.exists():
                post[d["new"]] = path.read_bytes().decode("utf-8", errors="replace")
    return post


# ---------------------------------------------------------------------------
# Test function extraction
# ---------------------------------------------------------------------------

def _split_ids(fail_to_pass_ids: list[str]) -> tuple[dict[str, set[str]], set[str]]:
    """Fail-to-pass ids -> ({test file path: function names}, {titles}).

    Most ids are ``path::function``.  Django ids may lack the ``tests/`` prefix in the path, and a Django test
    that has a docstring is named by the first line of that docstring (no path, no function name): a "title".
    """
    by_path: dict[str, set[str]] = {}
    titles: set[str] = set()
    for tid in fail_to_pass_ids:
        if "::" in tid:
            parts = tid.split("::")
            # Strip parametrize suffix e.g. test_foo[case0] → test_foo
            by_path.setdefault(parts[0], set()).add(parts[-1].split("[")[0])
        elif tid.strip():
            titles.add(tid.strip())
    return by_path, titles


def _names_for_file(by_path: dict[str, set[str]], filepath: str) -> set[str]:
    """Function names wanted in ``filepath``: an id path matches the file itself or a path suffix of it."""
    names: set[str] = set()
    for path, ns in by_path.items():
        candidates = {path}
        head, _, last = (path[:-3] if path.endswith(".py") else path).rpartition("/")
        if head and last[:1].isupper():
            candidates.add(head + ".py")     # 'pkg/tests/SomeClass.py' is really class SomeClass in 'pkg/tests.py'
        if any(filepath == c or filepath.endswith("/" + c) for c in candidates):
            names |= ns
    return names


def _extract_test_functions(
    test_files: dict[str, str],
    fail_to_pass_ids: list[str],
) -> list[dict]:
    """Extract test function bodies for the given fail_to_pass test IDs (by name, or by docstring title)."""
    by_path, titles = _split_ids(fail_to_pass_ids)

    results = []
    for filepath, source in test_files.items():
        names = _names_for_file(by_path, filepath)
        if not names and not titles:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        src_lines = source.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not hasattr(node, "end_lineno"):
                continue
            doc = ast.get_docstring(node) if titles else None
            if node.name not in names and not (doc and doc.strip().splitlines()[0].strip() in titles):
                continue
            start = node.lineno - 1
            end = node.end_lineno
            results.append({
                "filepath": filepath,
                "function_name": node.name,
                "code": "\n".join(src_lines[start:end]),
            })
    return results


def _extract_from_patch(
    test_patch: str,
    fail_to_pass_ids: list[str],
    already: list[dict],
) -> list[dict]:
    """Fallback for fail-to-pass tests whose test file is not available (e.g. too large to have been stored).

    A test the patch adds is written out in full in the patch's added lines, so its body can be read from the
    new side of the hunks (context + added lines).  Only used for (file, function) pairs not found otherwise.
    A test that the patch merely modifies is cut at the end of the hunk that shows it.
    """
    by_path, _ = _split_ids(fail_to_pass_ids)
    have = {(t["filepath"], t["function_name"]) for t in already}

    results = []
    for d in _split_file_diffs(test_patch or ""):
        names = {n for n in _names_for_file(by_path, d["new"]) if (d["new"], n) not in have}
        if not names or d["deleted"]:
            continue
        blocks, cur = [], None
        for line in d["text"].splitlines():
            if line.startswith("@@"):
                cur = []
                blocks.append(cur)
            elif cur is not None:
                if line.startswith(("+", " ")):
                    cur.append(line[1:])
                elif line == "":
                    cur.append("")
        for name in names:
            pattern = re.compile(r"^(\s*)(?:async\s+)?def\s+" + re.escape(name) + r"\b")
            for block in blocks:
                for k, line in enumerate(block):
                    m = pattern.match(line)
                    if not m:
                        continue
                    indent, end = len(m.group(1)), k + 1
                    while end < len(block) and (block[end].strip() == "" or len(block[end]) - len(block[end].lstrip()) > indent):
                        end += 1
                    while end > k + 1 and block[end - 1].strip() == "":
                        end -= 1
                    results.append({"filepath": d["new"], "function_name": name, "code": "\n".join(block[k:end])})
    return results


# ---------------------------------------------------------------------------
# Per-instance processing
# ---------------------------------------------------------------------------

def process_instance(inst: dict) -> dict:
    source_files: dict[str, str] = inst.get("source_files", {})
    test_files: dict[str, str] = inst.get("test_files", {})
    fail_to_pass: list[str] = inst.get("fail_to_pass", [])

    source_units = [
        unit
        for filepath, source in source_files.items()
        for unit in _extract_code_units(filepath, source)
    ]
    # Fail-to-pass tests are mostly ADDED by the test patch, so read the test files after applying it.
    test_functions = _extract_test_functions(apply_test_patch(test_files, inst.get("test_patch", "")), fail_to_pass)
    test_functions += _extract_from_patch(inst.get("test_patch", ""), fail_to_pass, test_functions)

    return {
        "instance_id": inst["instance_id"],
        "repo": inst["repo"],
        "requirement": inst["problem_statement"],
        "fail_to_pass_ids": fail_to_pass,
        "source_units": source_units,
        "test_functions": test_functions,
        "test_patch": inst.get("test_patch", ""),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: collect already-processed instance_ids
    done_ids: set[str] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["instance_id"])
                except Exception:
                    pass
        print(f"Resuming: {len(done_ids)} already done", flush=True)

    written = skipped = total = 0
    with open(args.input) as fin, open(out_path, "a") as fout:
        for lineno, raw in enumerate(fin, 1):
            if not raw.strip():
                continue
            try:
                inst = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"[WARN] skipping malformed line {lineno}: {exc}", flush=True)
                continue
            iid = inst["instance_id"]
            total += 1
            if iid in done_ids:
                skipped += 1
                continue
            record = process_instance(inst)
            fout.write(json.dumps(record) + "\n")
            fout.flush()
            written += 1
            print(
                f"[{written + skipped}/{total}] {iid}"
                f"  src_units={len(record['source_units'])}"
                f"  tests={len(record['test_functions'])}",
                flush=True,
            )

    print(f"Done. written={written} skipped={skipped} total={total}", flush=True)


if __name__ == "__main__":
    main()
