"""Sample Haiku patches and merge unique hunks into one record per issue.

The first generation keeps its raw response. Later samples keep only new hunks
and a small sampling summary. Existing generation fields remain unchanged.
Both plain JSONL and JSONL.gz files are supported.

Run from Code:
  python data_pipeline/sample_haiku_merge.py --samples 1 --max 5
  python data_pipeline/sample_haiku_merge.py --samples 2 --run-id round-2

Repeating a run-id resumes that round without calling the API for completed samples.
Without a run-id, each invocation starts a new round. ANTHROPIC_API_KEY is required.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import gzip
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_pipeline.merged_generations import candidate_hunks, diff_hunks, gold_distance, hunk_identity, read_jsonl

log = logging.getLogger(__name__)


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def load_records(path: Path) -> dict[str, dict]:
    records = {}
    if path.exists():
        for record in read_jsonl(path):
            iid = record["instance_id"]
            if iid in records:
                raise ValueError(f"{path}: duplicate instance_id {iid}")
            records[iid] = record
    return records


def save_records(path: Path, records: dict[str, dict]) -> None:
    """Replace the file only after the complete new JSONL has been written."""
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as raw:
            if path.suffix == ".gz":
                with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=1) as fh:
                    for record in records.values():
                        fh.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
            else:
                for record in records.values():
                    raw.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
            raw.flush()
            os.fsync(raw.fileno())
        if path.exists():
            os.chmod(temporary, path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def next_sample(record: dict | None) -> int:
    if record is None:
        return 0
    indices = [h["sample"] for h in candidate_hunks(record)]
    indices.extend(r["sample"] for r in record.get("sampling", {}).get("runs", []))
    return max(indices, default=0) + 1


def merge_sample(record: dict | None, inst: dict, result: dict, *, sample: int,
                 run_id: str, model: str, temperature: float, max_tokens: int,
                 system_prompt: str, user_prompt: str) -> tuple[dict, int]:
    if record is None:
        record = {
            "instance_id": inst["instance_id"], "repo": inst.get("repo", ""),
            "created_at": now(), "source_files": inst["source_files"],
            "meta": {
                "base_commit": inst.get("base_commit", ""),
                "problem_statement": inst.get("problem_statement", ""),
                "gold_patch": inst.get("patch", ""),
                "fail_to_pass": inst.get("fail_to_pass", []),
            },
            "generation": {
                "model": model, "provider": "anthropic", "temperature": temperature,
                "max_tokens": max_tokens, "system_prompt": system_prompt,
                **{k: v for k, v in result.items() if k != "modified_files"},
            },
        }
        merged = []
    else:
        meta = record.get("meta", {})
        if (meta.get("base_commit") and inst.get("base_commit")
                and meta["base_commit"] != inst["base_commit"]):
            raise ValueError(f"{inst['instance_id']}: base_commit differs from the existing record")
        if meta.get("gold_patch") is not None and meta["gold_patch"] != inst.get("patch", ""):
            raise ValueError(f"{inst['instance_id']}: reference patch differs from the existing record")
        merged = candidate_hunks(record)
    seen = {hunk_identity(h) for h in merged}
    hunks = diff_hunks(result.get("extracted_diff") or "", sample)
    added = 0
    for hunk in hunks:
        identity = hunk_identity(hunk)
        if identity not in seen:
            seen.add(identity)
            merged.append(hunk)
            added += 1
    record.pop("export_order", None)
    record["merged_hunks"] = merged
    record["merged_hunks"] = candidate_hunks(record)
    summary = {
        "run_id": run_id, "sample": sample, "created_at": now(), "model": model,
        "temperature": temperature, "max_tokens": max_tokens,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "user_prompt_sha256": hashlib.sha256(user_prompt.encode()).hexdigest(),
        "response_sha256": hashlib.sha256(result.get("raw_response", "").encode()).hexdigest(),
        "n_hunks": len(hunks), "n_added": added,
    }
    record.setdefault("sampling", {}).setdefault("runs", []).append(summary)
    return record, added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT.parent / "Data/swebench/swebench_full_instances.jsonl.gz")
    parser.add_argument("--generations", type=Path, default=ROOT.parent / "Data/eel_dataset/haiku_generations.jsonl.gz")
    parser.add_argument("--samples", type=int, default=1, help="Samples per batch")
    parser.add_argument("--run-id", help="Reuse this name to resume a sampling round")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--code-budget", type=int, default=100000)
    parser.add_argument("--max", dest="max_instances", type=int)
    parser.add_argument("--repo-filter")
    parser.add_argument("--instance-id", action="append", help="Select specific issues; may be repeated")
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--splits-file", type=Path, default=ROOT.parent / "Data/eel_dataset/splits.json")
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--dry-run", action="store_true", help="Show pending API calls without sampling or writing")
    args = parser.parse_args()
    if args.samples < 1 or args.max_tokens < 1 or args.code_budget < 1:
        parser.error("samples, max-tokens and code-budget must be positive")
    if args.max_instances is not None and args.max_instances < 1:
        parser.error("max must be positive")
    if not 0 <= args.temperature <= 1 or args.sleep < 0:
        parser.error("temperature must be in [0, 1] and sleep must be nonnegative")
    if args.input.resolve() == args.generations.resolve():
        parser.error("input and generations must be different files")
    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        parser.error("ANTHROPIC_API_KEY is not set")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run_id = args.run_id or now()
    split_ids = None
    if args.split:
        split_ids = set(json.loads(args.splits_file.read_text())[args.split + "_ids"])
    selected_ids = set(args.instance_id) if args.instance_id else None
    if args.dry_run:
        run(args, load_records(args.generations), run_id, split_ids, selected_ids)
    else:
        args.generations.parent.mkdir(parents=True, exist_ok=True)
        with args.generations.with_name(args.generations.name + ".lock").open("a") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error("another sampler is updating this generations file")
            run(args, load_records(args.generations), run_id, split_ids, selected_ids)


def run(args, records: dict[str, dict], run_id: str, split_ids, selected_ids) -> None:
    llm = None
    if not args.dry_run:
        from data.llm_client import LLMClient
        from data_pipeline.generate_candidates import generate, _build_user_prompt, _PATCH_SYSTEM
        llm = LLMClient(provider="anthropic", model=args.model, temperature=args.temperature,
                        max_tokens=args.max_tokens, max_retries=2, retry_delay=5.0)
    processed = calls = 0
    seen_input = set()
    for inst in read_jsonl(args.input):
        iid = inst["instance_id"]
        if iid in seen_input:
            raise ValueError(f"Input contains duplicate instance_id {iid}")
        seen_input.add(iid)
        if (args.repo_filter and inst.get("repo") != args.repo_filter
                or split_ids is not None and iid not in split_ids
                or selected_ids is not None and iid not in selected_ids):
            continue
        old = records.get(iid)
        completed = sum(r.get("run_id") == run_id for r in (old or {}).get("sampling", {}).get("runs", []))
        pending = args.samples - completed % args.samples
        if args.max_instances is not None and processed >= args.max_instances:
            break
        if not inst.get("source_files") or not inst.get("patch"):
            raise ValueError(f"{iid}: input needs source_files and patch; use the raw SWE-bench file")
        if old is not None:
            meta = old.get("meta", {})
            if meta.get("base_commit") and inst.get("base_commit") and meta["base_commit"] != inst["base_commit"]:
                raise ValueError(f"{iid}: base_commit differs from the existing record")
            if meta.get("gold_patch") is not None and meta["gold_patch"] != inst["patch"]:
                raise ValueError(f"{iid}: reference patch differs from the existing record")
        reference = diff_hunks(inst["patch"])

        def enough():
            return sum(gold_distance(hunk, reference) < 20
                       for hunk in candidate_hunks(records.get(iid) or {})) >= len(reference)

        if enough():
            continue
        processed += 1
        if args.dry_run:
            log.info("%s: %d pending sample(s), next sample=%d", iid, pending, next_sample(old))
            calls += pending
            continue
        user_prompt = _build_user_prompt(inst, code_budget=args.code_budget)
        while not enough():
            for _ in range(pending):
                sample = next_sample(records.get(iid))
                result = generate(inst, llm, code_budget=args.code_budget)
                calls += 1
                if result is None or not diff_hunks(result.get("extracted_diff") or ""):
                    raise RuntimeError(f"{iid}: generation failed; rerun with --run-id {run_id} to resume")
                record, added = merge_sample(records.get(iid), inst, result, sample=sample,
                    run_id=run_id, model=args.model, temperature=args.temperature,
                    max_tokens=args.max_tokens, system_prompt=_PATCH_SYSTEM, user_prompt=user_prompt)
                records[iid] = record
                save_records(args.generations, records)
                log.info("%s sample=%d: added=%d merged=%d", iid, sample, added, len(record["merged_hunks"]))
                if args.sleep:
                    time.sleep(args.sleep)
            pending = args.samples
    log.info("%s: issues=%d calls=%d run-id=%s", "Dry run" if args.dry_run else "Done", processed, calls, run_id)


if __name__ == "__main__":
    main()
