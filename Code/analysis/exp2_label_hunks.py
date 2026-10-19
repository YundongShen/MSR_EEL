"""Exp2 — Label retained hunks by edit type using Gemini.

Retained hunks are the generated hunks matching the reference patch.
Each retained hunk gets one or more labels (multi-label):
  BEHAVIORAL  — change directly verifiable by tests; affects observable outputs
  SEMANTIC    — implements requirement logic, not directly test-verified
  STRUCTURAL  — maintains code consistency (signatures, imports, cross-file sync)

Output (data/processed/hunk_edit_types.jsonl):
  One record per hunk:
  {instance_id, hunk_key, hunk_id, filepath, labels: [...], raw_response}
  (hunk_key = content fingerprint; join on it, hunk_id changes when the dataset is rebuilt)

Usage:
    # Trial: 5 instances, print results to stdout
    python analysis/exp2_label_hunks.py --limit 5 --trial

    # Full run
    python analysis/exp2_label_hunks.py \\
        --api-key AIza... \\
        --output  data/processed/hunk_edit_types.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

from google import genai
from google.genai import types

from config import default_config
from data.instances import load_instances, load_split_ids
from data.retained_hunks import add_retained_args, apply_retained_args, retained_hunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

VALID_LABELS = {"BEHAVIORAL", "SEMANTIC", "STRUCTURAL"}

SYSTEM_PROMPT = """You are a code review assistant. Your task is to classify a code diff hunk by its edit type.

Definitions:
- BEHAVIORAL: The change fixes or implements a specific behavior that is directly verifiable by automated tests. This includes changes to return values, condition checks, algorithm logic, or output computation that a test could assert on.
- SEMANTIC: The change implements intent described in the requirement, but is NOT something an automated test would directly catch. Examples: updating docstrings, adjusting default values, adding validation logic that tests don't cover, fixing logic in an untested code path.
- STRUCTURAL: The change maintains code organizational consistency without being the primary behavior fix. Examples: updating a function signature to match callers, adding/removing imports, propagating a rename across files, adding a helper function.

Rules:
- Assign ALL labels that apply (can be one, two, or all three).
- Base your decision only on what the diff shows, not on what you assume tests cover.
- Respond with ONLY the applicable labels as a comma-separated list, e.g.: BEHAVIORAL or BEHAVIORAL, STRUCTURAL
"""

def _render_hunk(hunk: dict) -> str:
    before = "\n".join(hunk.get("context_before") or [])
    diff   = hunk.get("hunk_diff", "")
    after  = "\n".join(hunk.get("context_after") or [])
    parts  = [p for p in (before, diff, after) if p.strip()]
    return "\n".join(parts)


def _parse_labels(response_text: str) -> list[str]:
    """Extract valid labels from Gemini's response."""
    found = []
    upper = response_text.upper()
    for label in VALID_LABELS:
        if label in upper:
            found.append(label)
    return sorted(found)


def label_hunk(client, model_name: str, hunk_text: str, retries: int = 3) -> tuple[list[str], str]:
    """Call Gemini to label a single hunk. Returns (labels, raw_response)."""
    prompt = f"{SYSTEM_PROMPT}\n\nDiff hunk:\n```\n{hunk_text[:3000]}\n```"
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            raw = response.text.strip()
            labels = _parse_labels(raw)
            if labels:
                return labels, raw
            log.warning("No valid labels in response: %r — retrying", raw)
        except Exception as e:
            log.warning("Gemini API error (attempt %d): %s", attempt + 1, e)
            time.sleep(2 ** attempt)
    return ["SEMANTIC"], "[fallback]"  # safe default


def load_test_instances(instances_path: str, splits_path: str) -> list[dict]:
    return load_instances(instances_path, load_split_ids(splits_path, "test"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key",   default=os.environ.get("GEMINI_API_KEY"),
                        help="Gemini API key (default: $GEMINI_API_KEY)")
    parser.add_argument("--model",     default="gemini-2.5-flash")
    parser.add_argument("--instances", default="data/processed/instances_full.jsonl")
    parser.add_argument("--splits",    default="data/processed/splits.json")
    parser.add_argument("--output",    default="data/processed/hunk_edit_types.jsonl")
    parser.add_argument("--limit",     type=int, default=None,
                        help="Process only N instances (for trial runs)")
    parser.add_argument("--trial",     action="store_true",
                        help="Print results to stdout instead of writing file")
    parser.add_argument("--delay",     type=float, default=0.5,
                        help="Seconds between API calls")
    add_retained_args(parser)
    args = parser.parse_args()
    apply_retained_args(default_config, args)

    if not args.api_key:
        raise SystemExit("Set GEMINI_API_KEY or pass --api-key")
    client = genai.Client(api_key=args.api_key)
    log.info("Model: %s", args.model)
    log.info("Retained hunks: %s", default_config.data.llm_t12_path)

    instances = load_test_instances(args.instances, args.splits)
    log.info("Test instances: %d", len(instances))

    if args.limit:
        instances = instances[: args.limit]
        log.info("Trial mode: limiting to %d instances", args.limit)

    records: list[dict] = []
    total_hunks = 0
    label_counts: dict[str, int] = {k: 0 for k in VALID_LABELS}

    for i, inst in enumerate(instances):
        iid = inst["instance_id"]
        hunks = retained_hunks(inst, default_config)
        log.info("[%d/%d] %s  (%d hunks)", i + 1, len(instances), iid, len(hunks))

        for hunk in hunks:
            hunk_text = _render_hunk(hunk)
            if not hunk_text.strip():
                continue

            labels, raw = label_hunk(client, args.model, hunk_text)
            time.sleep(args.delay)

            record = {
                "instance_id": iid,
                "hunk_key":    hunk["hunk_key"],
                "hunk_id":     hunk.get("hunk_id", ""),
                "filepath":    hunk.get("filepath", ""),
                "tier_label":  hunk.get("tier_label"),
                "labels":      labels,
                "raw_response": raw,
            }
            records.append(record)
            total_hunks += 1

            for lbl in labels:
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

            log.info("  %s | %s → %s", hunk.get("filepath", "")[-40:], hunk.get("hunk_id", ""), labels)

    log.info("=== Summary ===")
    log.info("Total hunks labelled: %d", total_hunks)
    for lbl, cnt in label_counts.items():
        log.info("  %-12s  %d  (%.1f%%)", lbl, cnt, 100 * cnt / max(total_hunks, 1))

    if args.trial:
        print("\n=== TRIAL OUTPUT ===")
        for r in records:
            print(json.dumps(r, ensure_ascii=False))
    else:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info("Saved %d records → %s", len(records), out)


if __name__ == "__main__":
    main()
