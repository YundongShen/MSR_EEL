"""Regression checks for parse_instances: the fail-to-pass test code comes from the test files AFTER the test patch.

Run from the project root (no GPU, no model):
    gpu_env312/bin/python tests/check_instances.py

Most fail-to-pass tests are added by the SWE-bench test patch, so they do not exist in the base commit.  Reading
them from the base test files left the TEST view empty for 55% of the issues.
"""

from __future__ import annotations

import difflib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_pipeline"))

import parse_instances as pi                                                      # noqa: E402

BASE_X = (
    "import unittest\n\n\nclass TestA(unittest.TestCase):\n"
    "    def test_old(self):\n        self.assertEqual(1, 1)\n"
)
POST_X = (
    "import unittest\n\n\nclass TestA(unittest.TestCase):\n"
    "    def test_old(self):\n        self.assertEqual(2, 2)\n"
    "\n    def test_new(self):\n        self.assertTrue(NEW_MARKER)\n"
)
ADDED = "def test_added_fn():\n    assert ADDED_MARKER\n"
BASE_Y = "def test_y_old():\n    assert Y_BASE_MARKER\n"


def _diff(path: str, old: str | None, new: str) -> str:
    a = old.splitlines(keepends=True) if old is not None else []
    body = "".join(difflib.unified_diff(a, new.splitlines(keepends=True),
                                       fromfile=f"a/{path}" if old is not None else "/dev/null", tofile=f"b/{path}"))
    head = f"diff --git a/{path} b/{path}\n" + ("" if old is not None else "new file mode 100644\n")
    return head + body


def _instance(f2p: list[str], test_patch: str, test_files: dict[str, str]) -> dict:
    return {"instance_id": "repo__x-1", "repo": "repo/x", "problem_statement": "fix it", "patch": "",
            "test_patch": test_patch, "fail_to_pass": f2p, "source_files": {}, "test_files": test_files}


def check_added_and_modified_tests() -> None:
    patch = _diff("tests/test_x.py", BASE_X, POST_X) + _diff("tests/test_added.py", None, ADDED)
    inst = _instance(["tests/test_x.py::test_new", "tests/test_x.py::test_old", "tests/test_added.py::test_added_fn"],
                     patch, {"tests/test_x.py": BASE_X})
    tf = {t["function_name"]: t["code"] for t in pi.process_instance(inst)["test_functions"]}
    assert set(tf) == {"test_new", "test_old", "test_added_fn"}, f"extracted: {sorted(tf)}"
    assert "NEW_MARKER" in tf["test_new"], "a test added to an existing class was not extracted"
    assert "ADDED_MARKER" in tf["test_added_fn"], "a test in a file added by the patch was not extracted"
    assert "2, 2" in tf["test_old"] and "1, 1" not in tf["test_old"], "a modified test must be the post-patch version"

    # before the fix: base files only -> the two new tests are missing and test_old is the stale version
    base_only = {t["function_name"]: t["code"] for t in pi._extract_test_functions({"tests/test_x.py": BASE_X}, inst["fail_to_pass"])}
    assert set(base_only) == {"test_old"} and "1, 1" in base_only["test_old"]


def check_unappliable_and_missing_base() -> None:
    bad = _diff("tests/test_y.py", "def other():\n    pass\n", "def other():\n    pass\n\ndef test_z():\n    pass\n")
    missing_base = _diff("tests/test_absent.py", "x = 1\n", "x = 2\n")
    inst = _instance(["tests/test_y.py::test_y_old"], bad + missing_base, {"tests/test_y.py": BASE_Y})
    post = pi.apply_test_patch(inst["test_files"], inst["test_patch"])
    assert post == {"tests/test_y.py": BASE_Y}, "a diff that does not apply must leave the base version, and a file without a base version is skipped"
    tf = pi.process_instance(inst)["test_functions"]
    assert [t["function_name"] for t in tf] == ["test_y_old"] and "Y_BASE_MARKER" in tf[0]["code"]


def check_no_test_found_stays_empty() -> None:
    inst = _instance(["tests/test_x.py::test_missing"], _diff("tests/test_x.py", BASE_X, POST_X), {"tests/test_x.py": BASE_X})
    assert pi.process_instance(inst)["test_functions"] == []
    assert "gold_hunks" not in pi.process_instance(inst), "instances must not carry the reference patch's own hunks"


def check_fallback_from_patch_when_file_missing() -> None:
    # the test file was too large to be stored: no base version, but the patch shows the added test in full
    big_base = "import unittest\n\n\nclass TestB(unittest.TestCase):\n    def test_keep(self):\n        pass\n"
    big_post = big_base + "\n    def test_from_patch(self):\n        x = 1\n        self.assertTrue(PATCH_MARKER)\n"
    inst = _instance(["tests/big/tests.py::test_from_patch", "tests/big/tests.py::test_absent_everywhere"],
                     _diff("tests/big/tests.py", big_base, big_post), {})
    tf = {t["function_name"]: t["code"] for t in pi.process_instance(inst)["test_functions"]}
    assert set(tf) == {"test_from_patch"}, f"extracted: {sorted(tf)}"
    code = tf["test_from_patch"]
    assert code.startswith("    def test_from_patch") and "PATCH_MARKER" in code and "x = 1" in code
    assert "test_keep" not in code, "the fallback must stop at the end of the function"

    # a test that is found in the (post-patch) file is not duplicated by the fallback
    inst2 = _instance(["tests/test_x.py::test_new"], _diff("tests/test_x.py", BASE_X, POST_X), {"tests/test_x.py": BASE_X})
    assert [t["function_name"] for t in pi.process_instance(inst2)["test_functions"]] == ["test_new"]


def check_django_style_ids() -> None:
    base = "import unittest\n\n\nclass MigrateTests(unittest.TestCase):\n    def test_keep(self):\n        pass\n"
    post = base + (
        "\n    def test_documented(self):\n"
        "        \"\"\"Options passed before settings are correctly handled.\n\n        More text.\n        \"\"\"\n"
        "        self.assertTrue(TITLE_MARKER)\n"
        "\n    def test_plain(self):\n        self.assertTrue(SUFFIX_MARKER)\n"
    )
    patch = _diff("tests/admin_scripts/tests.py", base, post)
    # (a) the id path lacks the leading 'tests/'; (b) a test named by its docstring title; both in the same issue
    inst = _instance(["admin_scripts/tests.py::test_plain", "Options passed before settings are correctly handled."],
                     patch, {"tests/admin_scripts/tests.py": base})
    tf = {t["function_name"]: t["code"] for t in pi.process_instance(inst)["test_functions"]}
    assert set(tf) == {"test_plain", "test_documented"}, f"extracted: {sorted(tf)}"
    assert "SUFFIX_MARKER" in tf["test_plain"] and "TITLE_MARKER" in tf["test_documented"]
    assert "test_keep" not in "".join(tf.values())
    # a path that is not a path suffix must not match ('ests/x.py' is not a suffix at a directory boundary)
    inst2 = _instance(["ests/admin_scripts/tests.py::test_plain"], patch, {"tests/admin_scripts/tests.py": base})
    assert pi.process_instance(inst2)["test_functions"] == []


def check_class_name_as_path() -> None:
    base = "import unittest\n\n\nclass SomeTests(unittest.TestCase):\n    def test_keep(self):\n        pass\n"
    post = base + "\n    def test_target(self):\n        self.assertTrue(CLASS_PATH_MARKER)\n"
    patch = _diff("tests/pkg/tests.py", base, post)
    # the id was built from 'pkg.tests.SomeTests': the class name ended up as the file name
    inst = _instance(["pkg/tests/SomeTests.py::test_target"], patch, {"tests/pkg/tests.py": base})
    tf = pi.process_instance(inst)["test_functions"]
    assert [t["function_name"] for t in tf] == ["test_target"] and "CLASS_PATH_MARKER" in tf[0]["code"]
    # a lower-case last component is a real file name: no reinterpretation
    inst2 = _instance(["pkg/tests/sometests.py::test_target"], patch, {"tests/pkg/tests.py": base})
    assert pi.process_instance(inst2)["test_functions"] == []


def check_real_instances_file() -> None:
    path = ROOT / "data/processed/instances_full.jsonl"
    if not path.exists():
        print("  (data/processed/instances_full.jsonl absent: skipped)")
        return
    import json
    n = empty = with_gold = 0
    for line in open(path):
        r = json.loads(line)
        n += 1
        empty += not r["test_functions"]
        with_gold += "gold_hunks" in r
    assert n == 2291, f"{n} instances"
    assert with_gold == 0, "the instances file carries the reference patch's own hunks: rebuild it with data_pipeline/parse_instances.py"
    assert empty / n < 0.02, f"TEST view empty for {empty}/{n} issues: the test code was not read from the post-test-patch files"
    print(f"  real file ok: {n} issues, TEST view empty for {empty}")


CHECKS = [check_added_and_modified_tests, check_unappliable_and_missing_base, check_no_test_found_stays_empty,
          check_fallback_from_patch_when_file_missing, check_django_style_ids, check_class_name_as_path,
          check_real_instances_file]

if __name__ == "__main__":
    failed = 0
    for fn in CHECKS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}\n{traceback.format_exc()}")
    sys.exit(1 if failed else 0)
