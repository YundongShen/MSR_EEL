"""Generate candidate patches with an LLM (one raw generation per SWE-bench issue).

For each SWE-bench instance:
  1. Build the prompt: issue text + code context.  The context is centred on the hunks of
     the reference patch and the prompt names the reference patch's files ("The file(s) to
     modify are: ...").
  2. Call the LLM with the unconstrained prompt: "Fix the bug. Make any changes necessary."
  3. Read the proposed changes from the response.
  4. Append the full record (prompt inputs, raw response, extracted diff) to the generations
     file.  Nothing is labelled here.

Retained / non-retained (Tier 1/2/3) labels are derived from the generations by
data_pipeline/build_dataset.py, so the labelling can be redone without calling the API again.
For an independent extra sample of the same model, run again with a different
--generations file and pass all files to build_dataset.py.

Paper protocol: --model claude-haiku-4-5-20251001, temperature 0.7, max tokens 4096,
one prompt for every instance.

Usage:
  # Smoke test (5 sympy instances):
  export ANTHROPIC_API_KEY=sk-ant-...
  python data_pipeline/generate_candidates.py --max 5 --repo-filter sympy/sympy \\
      --generations /tmp/smoke_generations.jsonl

  # Full run via SLURM:
  sbatch slurm/data_pipeline/generate_candidates.slurm
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.hunk_matching import parse_hunks
from data.llm_client import LLMClient
from data.patch_generator import (
    _CLAUDE_CODE_BUDGET,
    _relevant_context,
    _extract_diff_file_paths,
)
from data.data_loader import DataSample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


_PATCH_SYSTEM = (
    "You are a coding assistant. Fix the bug described in the issue.\n\n"
    "You may modify any relevant part of the codebase if needed to produce a robust fix.\n\n"
    "The final code should resolve the issue while keeping existing tests passing.\n\n"
    "Keep your response focused on the code changes, preserving the original formatting "
    "and surrounding context."
)


# ---------------------------------------------------------------------------
# Per-instance generation
# ---------------------------------------------------------------------------

def _build_user_prompt(inst: dict, code_budget: int = _CLAUDE_CODE_BUDGET) -> str:
    """Build the user prompt with configurable code context budget."""
    source_files: dict[str, str] = inst.get("source_files", {})
    reference_patch: str = inst.get("patch", "")
    issue_text: str = inst.get("problem_statement", "")

    sample = DataSample(
        sample_id=inst.get("instance_id", ""),
        issue_text=issue_text,
        old_codebase=source_files,
        reference_diff=reference_patch,
        test_suite=inst.get("test_files", {}),
    )

    code_ctx = _relevant_context(sample, budget=code_budget)
    reference_files = _extract_diff_file_paths(reference_patch)
    file_hint = (
        f"\nThe file(s) to modify are: {', '.join(reference_files)}\n"
        if reference_files else ""
    )
    return (
        f"## Issue\n{issue_text}\n"
        f"{file_hint}\n"
        f"## Codebase\n{code_ctx}\n\n"
        "Fix the issue."
    )


def generate(inst: dict, llm: LLMClient, code_budget: int = _CLAUDE_CODE_BUDGET) -> dict | None:
    """Generate one candidate patch for one instance.

    Returns the generation fields, or None if the instance has no reference hunks or the
    LLM call fails.  The raw response is always kept, even when no patch can be extracted.
    """
    if not parse_hunks(inst.get("patch", "")):
        log.debug("%s: no reference hunks, skipping", inst.get("instance_id"))
        return None

    user = _build_user_prompt(inst, code_budget=code_budget)
    raw = llm.complete(_PATCH_SYSTEM, user)
    if raw is None:
        log.warning("%s: LLM returned None", inst.get("instance_id"))
        return None

    fenced = re.search(r"```(?:diff|patch)?[ \t]*\r?\n(.*?)```", raw, re.DOTALL)
    return {
        "raw_response": raw,
        "extracted_diff": fenced.group(1) if fenced else raw,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input", default="data/raw/swebench_full_instances.jsonl",
        help="Raw SWE-bench JSONL with source_files and patch fields",
    )
    parser.add_argument(
        "--generations", default="data/cache/haiku_generations.jsonl",
        help="Generations JSONL to append to (one record per instance; also the resume index)",
    )
    parser.add_argument(
        "--model", default="claude-haiku-4-5-20251001",
        help="LLM model ID",
    )
    parser.add_argument(
        "--max", type=int, default=None, dest="max_instances",
        help="Stop after this many instances (for smoke testing)",
    )
    parser.add_argument(
        "--repo-filter", default=None,
        help="Only process instances from this repo, e.g. sympy/sympy",
    )
    parser.add_argument(
        "--split", default=None, choices=["train", "val", "test"],
        help="Filter to instances in this split (reads data/processed/splits.json)",
    )
    parser.add_argument(
        "--splits-file", default="data/processed/splits.json",
        help="Path to splits.json (default: data/processed/splits.json)",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.3,
        help="Seconds to sleep between API calls (rate limiting)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096,
        help="Max tokens for LLM response (default 4096 to avoid cutoff)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default 0.7)",
    )
    parser.add_argument(
        "--provider", default="anthropic",
        help="LLM provider: anthropic or openai (for vLLM/Qwen/Gemini)",
    )
    parser.add_argument(
        "--api-base", default=None,
        help="API base URL for OpenAI-compatible endpoints (vLLM)",
    )
    parser.add_argument(
        "--code-budget", type=int, default=None,
        help="Max chars of code context per prompt (default: 100000 for cloud models). "
             "Set to ~12000 for 7B models with 8192-token context windows.",
    )
    args = parser.parse_args()

    out_path = Path(args.generations)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    split_ids: set[str] | None = None
    if args.split is not None:
        with open(args.splits_file) as fh:
            split_ids = set(json.load(fh)[f"{args.split}_ids"])
        log.info("Split filter: %s → %d instances", args.split, len(split_ids))

    # Resume: skip instances that already have a generation in the output file.
    done_ids: set[str] = set()
    if out_path.exists():
        with open(out_path) as fh:
            for line in fh:
                try:
                    done_ids.add(json.loads(line)["instance_id"])
                except Exception:
                    pass
        log.info("Resuming: %d already generated", len(done_ids))

    llm = LLMClient(
        provider=args.provider,
        model=args.model,
        api_base=args.api_base,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_retries=2,
        retry_delay=5.0,
    )

    code_budget = args.code_budget if args.code_budget is not None else _CLAUDE_CODE_BUDGET
    log.info("Model: %s  code budget: %d chars", args.model, code_budget)

    processed = skipped = 0

    with open(args.input) as fin, open(out_path, "a") as fout:
        for lineno, raw_line in enumerate(fin, 1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                inst = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed line %d: %s", lineno, exc)
                continue

            iid = inst.get("instance_id", f"line_{lineno}")

            if args.repo_filter and inst.get("repo") != args.repo_filter:
                continue
            if split_ids is not None and iid not in split_ids:
                continue
            if iid in done_ids:
                skipped += 1
                continue
            if args.max_instances and processed >= args.max_instances:
                break

            processed += 1
            log.info("[%d] %s", processed, iid)

            result = generate(inst, llm, code_budget=code_budget)
            if result is None:
                log.info("  → skipped (no reference hunks or LLM failure)")
                continue

            # meta.gold_patch is SWE-bench's own name for the reference patch.
            record = {
                "id": iid,
                "instance_id": iid,
                "source": "swebench",
                "repo": inst.get("repo", ""),
                "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                "source_files": inst.get("source_files", {}),
                "meta": {
                    "base_commit": inst.get("base_commit", ""),
                    "problem_statement": inst.get("problem_statement", ""),
                    "gold_patch": inst.get("patch", ""),
                    "fail_to_pass": inst.get("fail_to_pass", []),
                },
                "generation": {
                    "model": args.model,
                    "provider": args.provider,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "system_prompt": _PATCH_SYSTEM,
                    **result,
                },
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()
            log.info("  → generated")

            if args.sleep > 0:
                time.sleep(args.sleep)

    log.info("Done. processed=%d  skipped(resume)=%d  → %s", processed, skipped, out_path)
    log.info("Next: python data_pipeline/build_dataset.py --generations %s", out_path)


if __name__ == "__main__":
    main()
