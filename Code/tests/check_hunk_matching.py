import sys
from pathlib import Path
from types import ModuleType

data = ModuleType("data")
data.__path__ = [str(Path(__file__).resolve().parent.parent / "data")]
sys.modules["data"] = data

from data.hunk_matching import parse_hunks


def test_two_files() -> None:
    patch = (
        "--- a/first.py\n"
        "+++ b/first.py\n"
        "@@ -2,1 +2,1 @@\n"
        "-old_first\n"
        "+new_first\n"
        "--- a/second.py\n"
        "+++ b/second.py\n"
        "@@ -7,1 +7,1 @@\n"
        "-old_second\n"
        "+new_second\n"
    )
    hunks = parse_hunks(patch)
    assert [h["filepath"] for h in hunks] == ["first.py", "second.py"]
    assert [h["old_start"] for h in hunks] == [2, 7]


if __name__ == "__main__":
    test_two_files()
