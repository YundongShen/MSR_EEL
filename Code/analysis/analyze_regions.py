"""Region analysis: trace UMAP points back to hunks, classify edit types per region.

Proximity thresholds (derived from M2 similarity distributions):
  high_req  : sim_req  > 0.65
  high_test : sim_test > 0.40
  high_orig : sim_orig > 0.80

Four primary regions analysed:
  R1 (REQ+TEST)       : high_req + high_test           → core changes, behaviorally tested
  R2 (REQ only)       : high_req + ~high_test          → req-driven, test-invisible
  R3 (ORIG only)      : ~high_req + ~high_test + high_orig → structurally necessary (our focus)
  R4 (none)           : ~high_req + ~high_test + ~high_orig → low-necessity / scope creep

Usage:
    python analysis/analyze_regions.py --model claude-haiku-4-5-20251001 --n-sample 30
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
THR_REQ  = 0.65
THR_TEST = 0.40
THR_ORIG = 0.80


def assign_region(sreq: float, stest: float, sorig: float) -> str:
    hi_req  = sreq  > THR_REQ
    hi_test = stest > THR_TEST
    hi_orig = sorig > THR_ORIG
    if hi_req and hi_test:
        return "R1_req_test"
    if hi_req and not hi_test and hi_orig:
        return "R2_req_orig"
    if hi_req and not hi_test and not hi_orig:
        return "R3_req_only"
    if not hi_req and hi_orig:
        return "R4_orig_only"
    return "R5_none"


# ---------------------------------------------------------------------------
# LLM classification
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a software engineering expert classifying code diff hunks.
Given a requirement description and a diff hunk, classify the hunk into exactly one category:

  core_fix       - Directly implements or changes the logic described in the requirement.
  induced_update - Updates adjacent code to maintain consistency with a core change
                   (e.g. propagating a signature change, updating a related data structure,
                   adding/removing an import, adjusting a caller).
  defensive      - Adds error handling, null checks, or validation not explicitly required
                   by the stated requirement or its tests.
  other          - Does not fit the above categories.

Reply with ONLY the category name, nothing else."""


def classify_hunk(client, model: str, requirement: str, hunk_text: str) -> str:
    prompt = f"REQUIREMENT:\n{requirement[:800]}\n\nDIFF HUNK:\n{hunk_text[:800]}"
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=10,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        label = msg.content[0].text.strip().lower()
        if label not in ("core_fix", "induced_update", "defensive", "other"):
            return "other"
        return label
    except Exception as e:
        print(f"  [warn] API error: {e}")
        return "other"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz",      default="logs/umap_embs_m2.npz")
    parser.add_argument("--instances", default="data/processed/instances_full.jsonl")
    parser.add_argument("--model",    default="claude-haiku-4-5-20251001")
    parser.add_argument("--n-sample", type=int, default=30)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--out",      default="logs/region_analysis.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load requirement texts
    print("Loading instances …")
    req_map: dict[str, str] = {}
    with open(args.instances) as f:
        for line in f:
            inst = json.loads(line)
            req_map[inst["instance_id"]] = inst.get("requirement", "")

    # Load UMAP npz
    print("Loading UMAP embeddings …")
    d = np.load(args.npz, allow_pickle=True)
    n_hunks  = int(d["n_hunks"])
    tags     = list(d["tags"][:n_hunks])
    iids     = list(d["meta_iid"])
    texts    = list(d["meta_text"])
    sreqs    = d["meta_sreq"].tolist()
    stests   = d["meta_stest"].tolist()
    sorigs   = d["meta_sorig"].tolist()

    # Assign regions
    region_buckets: dict[str, list[dict]] = defaultdict(list)
    for i in range(n_hunks):
        region = assign_region(sreqs[i], stests[i], sorigs[i])
        region_buckets[region].append({
            "tier":   tags[i],
            "iid":    iids[i],
            "text":   texts[i],
            "sreq":   sreqs[i],
            "stest":  stests[i],
            "sorig":  sorigs[i],
        })

    print("\n=== Region sizes ===")
    for r, items in sorted(region_buckets.items()):
        tier_dist = Counter(x["tier"] for x in items)
        print(f"  {r}: n={len(items)}  tiers={dict(tier_dist)}")

    # Classify sampled hunks per region using LLM
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    except Exception as e:
        print(f"\n[error] Cannot initialise Anthropic client: {e}")
        print("Set ANTHROPIC_API_KEY and ensure anthropic is installed.")
        sys.exit(1)

    results: dict[str, dict] = {}

    for region, items in sorted(region_buckets.items()):
        print(f"\n=== Classifying {region} (n={len(items)}, sample={min(args.n_sample, len(items))}) ===")
        sample = rng.sample(items, min(args.n_sample, len(items)))
        labels = []
        for j, item in enumerate(sample):
            req_text = req_map.get(item["iid"], "")
            label = classify_hunk(client, args.model, req_text, item["text"])
            labels.append(label)
            print(f"  [{j+1}/{len(sample)}] tier={item['tier']}  label={label}")

        counts = Counter(labels)
        total  = len(labels)
        proportions = {k: round(v / total, 3) for k, v in counts.items()}
        results[region] = {
            "n_total":   len(items),
            "n_sample":  total,
            "tier_dist": dict(Counter(x["tier"] for x in items)),
            "label_counts":      dict(counts),
            "label_proportions": proportions,
            "examples": [
                {"iid": s["iid"], "tier": s["tier"],
                 "sreq": s["sreq"], "stest": s["stest"], "sorig": s["sorig"],
                 "label": lb, "text": s["text"][:400]}
                for s, lb in zip(sample[:5], labels[:5])
            ],
        }

    # Save
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.out}")

    # Summary table
    print("\n=== Summary ===")
    print(f"{'Region':<18} {'n':>5}  {'core_fix':>9} {'induced':>9} {'defensive':>10} {'other':>6}")
    for region, res in sorted(results.items()):
        p = res["label_proportions"]
        print(f"{region:<18} {res['n_total']:>5}  "
              f"{p.get('core_fix',0):>9.1%} "
              f"{p.get('induced_update',0):>9.1%} "
              f"{p.get('defensive',0):>10.1%} "
              f"{p.get('other',0):>6.1%}")


if __name__ == "__main__":
    main()
