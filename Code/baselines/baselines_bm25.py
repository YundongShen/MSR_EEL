"""BM25 lexical baseline for retention ranking.

Scores each candidate hunk as BM25(tokenize(REQ), tokenize(HUNK)).
Candidate set includes both retained hunks (T1/T2) and Tier-3 scope-creep
hunks, matching the evaluation setup used by evaluate.py.

No GPU needed — runs on login node or CPU-only allocation.

Usage:
    python baselines/baselines_bm25.py
    python baselines/baselines_bm25.py --instances data/processed/instances_full.jsonl \\
                              --tier3     data/cache/tier3_hunks.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from config import default_config
from data.retained_hunks import add_retained_args, apply_retained_args
from evaluate import run_baseline_retrieval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z_]\w*", text.lower())


def run_bm25_baseline(
    instances_path: str,
    tier3_path: str | None,
    all_instances: bool = False,
    repo_pool: bool = False,
) -> dict:
    cfg = default_config
    cfg.data.instances_path = instances_path
    cfg.eval.tier3_path = tier3_path or ""

    def score_fn(req: str, hunk_texts: list[str]) -> list[float]:
        bm25 = BM25Okapi([_tokenize(h) for h in hunk_texts])   # corpus = this instance's candidates
        return bm25.get_scores(_tokenize(req)).tolist()

    return run_baseline_retrieval(cfg, score_fn, "BM25", all_instances=all_instances, repo_pool=repo_pool)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BM25 lexical baseline")
    parser.add_argument(
        "--instances", default="data/processed/instances_full.jsonl",
    )
    parser.add_argument(
        "--tier3", default="data/cache/tier3_hunks.jsonl",
    )
    parser.add_argument("--all-instances", action="store_true")
    parser.add_argument("--repo-pool", action="store_true",
                        help="Add same-repo cross-issue retained hunks as tier-0 distractors (as evaluate.py --repo-pool)")
    parser.add_argument("--output", default=None)
    add_retained_args(parser)
    args = parser.parse_args()
    apply_retained_args(default_config, args)
    log.info("Retained-H: %s", default_config.data.llm_t12_path)

    results = run_bm25_baseline(
        instances_path=args.instances,
        tier3_path=args.tier3,
        all_instances=args.all_instances,
        repo_pool=args.repo_pool,
    )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(results, fh, indent=2)
        log.info("Results saved → %s", args.output)
