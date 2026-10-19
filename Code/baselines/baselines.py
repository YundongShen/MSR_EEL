"""Baseline scorers for Edit Entailment evaluation.

Three baselines for comparison against the trained EntailmentEncoder:

  random   — uniform random score, sanity check lower bound
  bm25     — lexical overlap between requirement and hunk (rank_bm25)
  untrained — EntailmentEncoder with random weights (no checkpoint loaded)

All baselines share the same evaluation loop from evaluate.py, producing
nDCG@k and Tier-2 Recall metrics for direct comparison.

NOTE: quick sanity baselines only.  Candidates are the retained hunks alone (no Tier 3, no T0 pool, no PSR),
so these numbers are NOT the paper protocol; the paper baselines are baselines/baselines_bm25.py / baselines/baselines_pretrained.py /
baselines/baselines_extra.py.

Usage:
    python baselines/baselines.py --baseline random
    python baselines/baselines.py --baseline bm25
    python baselines/baselines.py --baseline untrained --encoder microsoft/codebert-base
    python baselines/baselines.py --baseline all
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import math
import random
import re
from pathlib import Path

import numpy as np
import torch

from config import Config, default_config
from data.entailment_dataset import _render_hunk
from data.retained_hunks import add_retained_args, apply_retained_args, retained_hunks
from evaluate import (
    _dcg,
    _ndcg,
    load_test_instances,
    run_geometry,
)
from models.entailment_encoder import EntailmentEncoder

try:
    from rank_bm25 import BM25Okapi  # type: ignore
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenisation helper for BM25
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenisation (no external dependency)."""
    return re.findall(r"[a-zA-Z_]\w*", text.lower())


# ---------------------------------------------------------------------------
# Scoring functions  (instance → list of scores, one per candidate hunk)
# ---------------------------------------------------------------------------

def score_random(
    req: str,
    hunk_texts: list[str],
    **_,
) -> list[float]:
    return [random.random() for _ in hunk_texts]


def score_bm25(
    req: str,
    hunk_texts: list[str],
    **_,
) -> list[float]:
    if not _BM25_AVAILABLE:
        raise RuntimeError("rank_bm25 not installed. Run: pip install rank-bm25")
    corpus = [_tokenize(h) for h in hunk_texts]
    bm25 = BM25Okapi(corpus)
    query = _tokenize(req)
    return bm25.get_scores(query).tolist()


def score_untrained(
    req: str,
    hunk_texts: list[str],
    test_texts: list[str],
    orig_texts: list[str],
    encoder: EntailmentEncoder,
    device: torch.device,
    cfg: Config,
) -> list[float]:
    return encoder.entailment_score(
        hunk_texts=hunk_texts,
        req_texts=[req] * len(hunk_texts),
        test_texts=[test_texts] * len(hunk_texts),
        orig_texts=[orig_texts] * len(hunk_texts),
        device=device,
        alpha=cfg.eval.score_alpha,
        beta=cfg.eval.score_beta,
        gamma=cfg.eval.score_gamma,
    ).tolist()


# ---------------------------------------------------------------------------
# Shared evaluation loop
# ---------------------------------------------------------------------------

def run_baseline(
    name: str,
    cfg: Config,
    scorer,
    encoder: EntailmentEncoder | None = None,
    device: torch.device | None = None,
) -> None:
    log.info("=== Baseline: %s ===", name)
    instances = load_test_instances(cfg)
    log.info("Test instances: %d", len(instances))

    tier_rel = cfg.eval.tier_relevance
    k_values = list(cfg.eval.retrieval_k_values)

    ndcg_accum: dict[int, list[float]] = {k: [] for k in k_values}
    tier2_recall: list[float] = []
    tier_scores: dict[int, list[float]] = {1: [], 2: [], 3: []}

    for inst in instances:
        req        = inst.get("requirement", "")[:cfg.data.max_req_chars]
        test_texts = [tf["code"] for tf in inst.get("test_functions", []) if tf.get("code")]
        orig_texts = [u["code"]  for u  in inst.get("source_units", [])   if u.get("code")]

        candidates: list[tuple[str, int]] = []
        for hunk in retained_hunks(inst, cfg):
            text = _render_hunk(hunk)
            if text.strip():
                tier = hunk.get("tier_label") or 1
                candidates.append((text, tier))

        if len(candidates) < 2:
            continue

        hunk_texts = [c[0] for c in candidates]
        tiers      = [c[1] for c in candidates]

        scores = scorer(
            req=req,
            hunk_texts=hunk_texts,
            test_texts=test_texts,
            orig_texts=orig_texts,
            encoder=encoder,
            device=device,
            cfg=cfg,
        )

        for score, tier in zip(scores, tiers):
            tier_scores[tier].append(score)

        ranked     = sorted(zip(scores, tiers), key=lambda x: x[0], reverse=True)
        ranked_rel = [tier_rel.get(t, 0.0) for _, t in ranked]

        for k in k_values:
            ndcg_accum[k].append(_ndcg(ranked_rel, k))

        tier2_total = sum(1 for t in tiers if t == 2)
        if tier2_total > 0:
            tier2_in_top = sum(1 for _, t in ranked[:tier2_total] if t == 2)
            tier2_recall.append(tier2_in_top / tier2_total)

    # Report
    for tier in (1, 2, 3):
        s = tier_scores[tier]
        if s:
            log.info("  Tier %d  n=%d  mean=%.4f  std=%.4f", tier, len(s), np.mean(s), np.std(s))

    for k in k_values:
        vals = ndcg_accum[k]
        if vals:
            log.info("  nDCG@%2d  mean=%.4f  std=%.4f  n=%d", k, np.mean(vals), np.std(vals), len(vals))

    if tier2_recall:
        log.info(
            "  Tier-2 Recall  mean=%.4f  std=%.4f  n=%d",
            np.mean(tier2_recall), np.std(tier2_recall), len(tier2_recall),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: Config, baseline: str, encoder_name: str) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    log.info("Device: %s", device)

    # Untrained encoder (shared across baselines that need it)
    untrained_encoder = None
    if baseline in ("untrained", "all"):
        log.info("Loading untrained encoder (%s)...", encoder_name)
        untrained_encoder = EntailmentEncoder(
            model_name=encoder_name,
            projection_dim=cfg.model.projection_dim,
            dropout=0.0,
            max_length=cfg.model.max_length,
        ).to(device)
        untrained_encoder.eval()

    if baseline in ("random", "all"):
        run_baseline("random", cfg, score_random)

    if baseline in ("bm25", "all"):
        run_baseline("bm25", cfg, score_bm25)

    if baseline in ("untrained", "all"):
        run_baseline(
            f"untrained ({encoder_name})", cfg, score_untrained,
            encoder=untrained_encoder, device=device,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline scorers")
    parser.add_argument(
        "--baseline", choices=["random", "bm25", "untrained", "all"],
        default="all",
    )
    parser.add_argument("--instances", default=None)
    parser.add_argument(
        "--encoder", default="microsoft/codebert-base",
        help="Backbone for untrained baseline",
    )
    add_retained_args(parser)
    args = parser.parse_args()

    cfg = default_config
    apply_retained_args(cfg, args)
    log.info("Retained-H: %s", cfg.data.llm_t12_path)
    if args.instances:
        cfg.data.instances_path = args.instances

    main(cfg, args.baseline, args.encoder)
