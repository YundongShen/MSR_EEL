"""Regression checks: training / evaluation only ever see LLM-generated hunks, never the reference patch's own.

Run from the project root (no pytest needed, CPU only):
    CUDA_VISIBLE_DEVICES="" gpu_env312/bin/python tests/check_hunk_source.py

What is checked
  1. static: no live script mentions ``gold_hunks`` except the loader that drops the field
  2. sentinel: an instance carrying a reference-patch hunk ("GOLD_SENTINEL") and an LLM-generated
     retained hunk ("GEN_SENTINEL") yields training pairs containing only the generated one
  3. regression of the old fallback (``hunks = llm_t12 if llm_t12 else gold_hunks``): an instance with a
     reference-patch hunk but NO retained hunk contributes no hunk pair
  4. hunk-file validation rejects reference-patch style files
  5. checkpoints without a valid data_provenance stamp are refused
  6. hunk_key is stable and content-sensitive
  7. (if present) the real data/cache files pass validation
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.entailment_dataset import EntailmentDataset                                   # noqa: E402
from data.instances import load_instances                                               # noqa: E402
from data.provenance import (                                                           # noqa: E402
    HUNK_SOURCE, PROTOCOL_VERSION, CheckpointProvenanceError, build_stamp, check_checkpoint,
    hunk_key, validate_generated_hunks,
)
from data.retained_hunks import load_retained_hunks, load_tier3_hunks                   # noqa: E402

GOLD, GEN = "GOLD_SENTINEL", "GEN_SENTINEL"
_SKIP_DIRS = {"old_v1", "archive_old", "recovered_from_mac", "gpu_env312", "__pycache__", ".git", "tests", "paper"}


def gen_hunk(text: str = GEN, tier: int = 1, **kw) -> dict:
    h = {"filepath": "a.py", "old_start_line": 1, "hunk_diff": f"@@ -1,1 +1,1 @@\n-x\n+{text}\n",
         "context_before": [], "context_after": [], "tier_label": tier, "sample": 0}
    h.update(kw)
    return h


def instance(iid: str) -> dict:
    return {
        "instance_id": iid, "repo": "repo", "requirement": "fix it", "fail_to_pass_ids": [],
        "gold_hunks": [{"filepath": "a.py", "old_start_line": 1, "context_before": [], "context_after": [],
                        "hunk_diff": f"@@ -1,1 +1,1 @@ def f\n-x\n+{GOLD}\n", "tier_label": None, "hunk_id": 0}],
        "source_units": [{"filepath": "a.py", "name": "f", "kind": "function", "start_line": 1, "end_line": 5,
                          "code": "def f():\n    pass"}],
        "test_functions": [{"filepath": "t.py", "function_name": "t", "code": "def t():\n    pass"}],
    }


def write_jsonl(path: Path, records: list[dict]) -> Path:
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return path


# ---------------------------------------------------------------------------
def check_static() -> None:
    offenders = []
    for path in ROOT.rglob("*"):
        if path.suffix not in (".py", ".slurm", ".sh") or set(path.relative_to(ROOT).parts) & _SKIP_DIRS:
            continue
        if path.name == "instances.py":          # the loader that drops the field
            continue
        for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if re.search(r"gold_hunks", line):
                offenders.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()[:80]}")
    assert not offenders, "live code mentions gold_hunks:\n  " + "\n  ".join(offenders)


def check_sentinel_and_fallback() -> None:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        inst_path = write_jsonl(d / "inst.jsonl", [instance("repo__x-1"), instance("repo__x-2")])
        # x-1 has a retained generated hunk; x-2 has none (only a reference-patch hunk in its record)
        ret_path = write_jsonl(d / "ret.jsonl", [{"instance_id": "repo__x-1", "llm_t12_hunks": [gen_hunk()]}])

        loaded = load_instances(inst_path)
        assert all("gold_hunks" not in i for i in loaded), "load_instances kept gold_hunks"

        ds = EntailmentDataset(inst_path, ret_path)
        hunk_texts = [p.text_b for p in ds if p.type_b == "HUNK"]
        assert hunk_texts, "no hunk pairs were built from the generated hunk"
        assert all(GEN in t for t in hunk_texts), "a HUNK text is not the generated hunk"
        assert not any(GOLD in p.text_a + p.text_b for p in ds), "reference-patch hunk text reached a training pair"

        by_inst = {p.instance_id for p in ds if p.pair_type in ("req_hunk", "orig_hunk")}
        assert by_inst == {"repo__x-1"}, f"instance without a retained hunk produced hunk pairs: {by_inst}"


def check_validation() -> None:
    ok = {"i": [gen_hunk()]}
    validate_generated_hunks(ok, (1, 2), "ok")                              # accepted
    validate_generated_hunks({"i": [gen_hunk(hunk_diff="@@ -1,1 +1,1 @@ def f\n-x\n+y\n")]}, (1, 2), "named")
    bad = {
        "no sample field (reference-patch record)": {"i": [{k: v for k, v in gen_hunk().items() if k != "sample"}]},
        "wrong tier": {"i": [gen_hunk(tier=3)]},
        "invalid diff header": {"i": [gen_hunk(hunk_diff="@@ invalid\n-x\n+y\n")]},
    }
    for name, recs in bad.items():
        try:
            validate_generated_hunks(recs, (1, 2), name)
        except ValueError:
            continue
        raise AssertionError(f"validation accepted: {name}")


def check_checkpoint_stamp() -> None:
    stamp = build_stamp("data/cache/llm_t12_hunks.jsonl", None, "data/processed/instances_full.jsonl",
                        ("req_hunk",), {"req_hunk": 1}, False, False)
    assert stamp["hunk_source"] == HUNK_SOURCE and stamp["protocol_version"] == PROTOCOL_VERSION
    check_checkpoint({"data_provenance": stamp}, "stamped.pt")              # accepted
    for name, ckpt in {"no stamp": {}, "wrong source": {"data_provenance": {**stamp, "hunk_source": "reference_patch"}},
                       "wrong version": {"data_provenance": {**stamp, "protocol_version": 1}}}.items():
        try:
            check_checkpoint(ckpt, name)
        except CheckpointProvenanceError:
            check_checkpoint(ckpt, name, allow_legacy=True)                 # explicit archival override works
            continue
        raise AssertionError(f"checkpoint accepted: {name}")


def check_hunk_key() -> None:
    a, b = gen_hunk(), gen_hunk()
    assert hunk_key(a) == hunk_key(b)
    assert hunk_key(a) != hunk_key(gen_hunk(text="OTHER"))
    assert hunk_key(a) != hunk_key(gen_hunk(old_start_line=2))


def check_real_files() -> None:
    ret, t3 = ROOT / "data/cache/llm_t12_hunks.jsonl", ROOT / "data/cache/tier3_hunks.jsonl"
    if not (ret.exists() and t3.exists()):
        print("  (real data files absent: skipped)")
        return
    r, t = load_retained_hunks(ret), load_tier3_hunks(t3)                   # validates on load
    assert r and t and all(h["hunk_key"] for hs in r.values() for h in hs)
    print(f"  real files ok: {sum(map(len, r.values()))} retained, {sum(map(len, t.values()))} Tier-3 hunks")


CHECKS = [check_static, check_sentinel_and_fallback, check_validation, check_checkpoint_stamp, check_hunk_key,
          check_real_files]

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
