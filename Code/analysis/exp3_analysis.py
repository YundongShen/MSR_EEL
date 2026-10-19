"""RQ3 mechanism diagnostics.

  Analysis A — Stratified Evaluation (Hard vs Easy)
    Split test instances by mean T3-REQ cosine similarity, measured with the
    *untrained* (M0) encoder to avoid circular dependency.

  Analysis B — ORIG Signal Alignment
    For every hunk with a matched ORIG unit, compute sim(hunk_emb, orig_emb)
    using M1 and M2 separately. Group by tier (retained=T1∪T2 vs scope=T3).
    Mann-Whitney U with rank-biserial effect size tests whether M2 encodes ORIG
    as a more discriminative signal than M1.

Usage:
    python analysis/exp3_analysis.py \\
        --m1 checkpoints/m1/best.pt \\
        --m2 checkpoints/m2/best.pt \\
        --instances data/processed/instances_full.jsonl \\
        --tier3     data/cache/tier3_hunks.jsonl \\
        --output    logs/exp3_analysis.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu

from config import Config, default_config
from data.entailment_dataset import _render_hunk, _units_for_hunk
from data.instances import load_instances, load_split_ids
from data.provenance import describe_hunk_files, load_checkpoint_checked
from data.retained_hunks import (
    add_retained_args, apply_retained_args, load_tier3_hunks, retained_hunks,
)
from models.entailment_encoder import EntailmentEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint / encoder helpers
# ---------------------------------------------------------------------------

def load_encoder(
    ckpt_path: str | None,
    cfg: Config,
    device: torch.device,
    allow_legacy: bool = False,
) -> EntailmentEncoder:
    encoder = EntailmentEncoder(
        model_name=cfg.model.encoder_name,
        projection_dim=cfg.model.projection_dim,
        dropout=0.0,
        max_length=cfg.model.max_length,
    ).to(device)

    if ckpt_path is not None:
        p = Path(ckpt_path)
        if p.exists():
            ckpt = load_checkpoint_checked(p, map_location=device, allow_legacy=allow_legacy)
            encoder.load_state_dict(ckpt["encoder_state"])
            log.info("Loaded checkpoint: %s", p)
        else:
            raise FileNotFoundError(f"Checkpoint not found: {p}")
    else:
        log.info("No checkpoint path — M0 (untrained backbone, no projection)")

    encoder.eval()
    return encoder


# ---------------------------------------------------------------------------
# Data loading (mirrors evaluate.py)
# ---------------------------------------------------------------------------

def load_test_instances(instances_path: str) -> list[dict]:
    """Test-split instances; the reference patch's own hunks are dropped on load."""
    return load_instances(instances_path, load_split_ids("data/processed/splits.json", "test"))


def load_tier3_lookup(tier3_path: str) -> dict[str, list[dict]]:
    """Validated LLM-generated Tier-3 hunks."""
    return load_tier3_hunks(tier3_path)


# ---------------------------------------------------------------------------
# nDCG helpers (identical to evaluate.py)
# ---------------------------------------------------------------------------

_TIER_REL = {1: 3.0, 2: 2.0, 3: 0.0}


def _dcg(relevances: list[float], k: int) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances[:k]))


def _ndcg(relevances: list[float], k: int) -> float:
    ideal = sorted(relevances, reverse=True)
    idcg = _dcg(ideal, k)
    return _dcg(relevances, k) / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Core retrieval metric (mirrors evaluate.py run_retrieval exactly)
# ---------------------------------------------------------------------------

def retrieval_metrics(
    instances: list[dict],
    encoder: EntailmentEncoder,
    device: torch.device,
    tier3_lookup: dict[str, list[dict]],
    cfg: Config,
    rng: random.Random | None = None,
) -> dict:
    """Compute nDCG@k, T2-Recall, PSR on a given instance list."""
    if rng is None:
        rng = random.Random(42)

    ndcg_vals:     list[float] = []
    t2_recall_vals: list[float] = []
    psr_vals:      list[float] = []

    for inst in instances:
        iid          = inst.get("instance_id", "")
        req          = inst.get("requirement", "")[:cfg.data.max_req_chars]
        test_texts   = [tf["code"] for tf in inst.get("test_functions", []) if tf.get("code")]
        source_units = inst.get("source_units", [])

        candidates: list[tuple[str, int, dict]] = []
        for hunk in retained_hunks(inst, cfg):
            text = _render_hunk(hunk)
            if text.strip():
                candidates.append((text, hunk.get("tier_label") or 1, hunk))
        for hunk in tier3_lookup.get(iid, []):
            text = _render_hunk(hunk)
            if text.strip():
                candidates.append((text, 3, hunk))

        if len(candidates) < 2:
            continue

        tiers       = [c[1] for c in candidates]
        has_tier3   = any(t == 3 for t in tiers)
        retained_total  = sum(1 for t in tiers if t in (1, 2))
        tier2_total = sum(1 for t in tiers if t == 2)

        if retained_total == 0 or not has_tier3:
            continue

        hunk_texts      = [c[0] for c in candidates]
        hunk_orig_texts = [
            [u["code"] for u in _units_for_hunk(c[2], source_units) if u.get("code")]
            for c in candidates
        ]

        scores = encoder.entailment_score(
            hunk_texts=hunk_texts,
            req_texts=[req] * len(hunk_texts),
            test_texts=[test_texts] * len(hunk_texts),
            orig_texts=hunk_orig_texts,
            device=device,
            alpha=cfg.eval.score_alpha,
            beta=cfg.eval.score_beta,
            gamma=cfg.eval.score_gamma,
            use_projection=True,
        ).tolist()

        tiebreaks  = [rng.random() for _ in scores]
        ranked     = sorted(zip(scores, tiers, tiebreaks), key=lambda x: (x[0], x[2]), reverse=True)
        ranked     = [(s, t) for s, t, _ in ranked]
        ranked_rel = [_TIER_REL.get(t, 0.0) for _, t in ranked]

        ndcg_vals.append(_ndcg(ranked_rel, retained_total))

        if tier2_total > 0:
            tier2_in_top = sum(1 for _, t in ranked[:retained_total] if t == 2)
            t2_recall_vals.append(tier2_in_top / tier2_total)

        retained_ranks = [i for i, (_, t) in enumerate(ranked) if t in (1, 2)]
        t3_ranks   = [i for i, (_, t) in enumerate(ranked) if t == 3]
        if retained_ranks and t3_ranks:
            psr_vals.append(1.0 if min(t3_ranks) > max(retained_ranks) else 0.0)

    return {
        "ndcg":         float(np.mean(ndcg_vals))      if ndcg_vals      else 0.0,
        "ndcg_std":     float(np.std(ndcg_vals))       if ndcg_vals      else 0.0,
        "t2_recall":    float(np.mean(t2_recall_vals)) if t2_recall_vals else 0.0,
        "t2_recall_std":float(np.std(t2_recall_vals))  if t2_recall_vals else 0.0,
        "psr":          float(np.mean(psr_vals))        if psr_vals       else 0.0,
        "n_instances":  len(ndcg_vals),
        "n_t2_instances": len(t2_recall_vals),
    }


# ---------------------------------------------------------------------------
# Analysis A — Stratified Evaluation
# ---------------------------------------------------------------------------

def analysis_a_stratified(
    instances: list[dict],
    tier3_lookup: dict[str, list[dict]],
    m1_encoder: EntailmentEncoder,
    m2_encoder: EntailmentEncoder,
    m0_encoder: EntailmentEncoder,
    device: torch.device,
    cfg: Config,
) -> dict:
    """Stratify test instances by T3-REQ sim (M0), evaluate M1 vs M2 per group."""
    log.info("=== Analysis A: Stratified Evaluation ===")

    # Compute mean T3-REQ cosine sim per instance using untrained M0 encoder
    instance_t3_sim: dict[str, float] = {}

    for inst in instances:
        iid     = inst.get("instance_id", "")
        t3_hunks = tier3_lookup.get(iid, [])
        if not t3_hunks:
            continue

        req        = inst.get("requirement", "")[:cfg.data.max_req_chars]
        hunk_texts = [_render_hunk(h) for h in t3_hunks]
        hunk_texts = [t for t in hunk_texts if t.strip()]
        if not hunk_texts:
            continue

        # use_projection=False → untrained backbone mean-pool (M0 geometry)
        r_emb  = m0_encoder.encode([req], "REQ", device, use_projection=False)   # (1, D)
        h_embs = m0_encoder.encode(hunk_texts, "HUNK", device, use_projection=False)  # (N, D)
        sims   = (h_embs * r_emb).sum(-1).tolist()   # L2-normalised → cosine
        instance_t3_sim[iid] = float(np.mean(sims))

    n_stratified = len(instance_t3_sim)
    if n_stratified < 4:
        log.warning("Too few instances with T3 hunks for stratification: %d", n_stratified)
        return {"error": "insufficient T3 instances", "n": n_stratified}

    sim_values = list(instance_t3_sim.values())
    median_sim = float(np.median(sim_values))
    log.info("T3-REQ sim (M0): median=%.4f  min=%.4f  max=%.4f  n=%d",
             median_sim, min(sim_values), max(sim_values), n_stratified)

    hard_ids = {iid for iid, s in instance_t3_sim.items() if s >= median_sim}
    easy_ids = {iid for iid, s in instance_t3_sim.items() if s <  median_sim}

    hard_instances = [inst for inst in instances if inst.get("instance_id") in hard_ids]
    easy_instances = [inst for inst in instances if inst.get("instance_id") in easy_ids]
    log.info("Hard group (T3≥median): %d  |  Easy group (T3<median): %d",
             len(hard_instances), len(easy_instances))

    results: dict = {
        "median_t3_req_sim": median_sim,
        "n_hard": len(hard_instances),
        "n_easy": len(easy_instances),
    }

    for group_name, group_instances in [("hard", hard_instances), ("easy", easy_instances)]:
        for model_name, encoder in [("m1", m1_encoder), ("m2", m2_encoder)]:
            log.info("Evaluating %s / %s (%d instances)...",
                     model_name, group_name, len(group_instances))
            metrics = retrieval_metrics(
                group_instances, encoder, device, tier3_lookup, cfg,
                rng=random.Random(42),
            )
            key = f"{group_name}_{model_name}"
            results[key] = metrics
            log.info(
                "  %s/%s: nDCG=%.4f  T2-Recall=%.4f  PSR=%.4f  n=%d",
                group_name, model_name,
                metrics["ndcg"], metrics["t2_recall"], metrics["psr"],
                metrics["n_instances"],
            )

    for group_name in ("hard", "easy"):
        m1_r = results[f"{group_name}_m1"]
        m2_r = results[f"{group_name}_m2"]
        delta = {
            "delta_ndcg":      m2_r["ndcg"]      - m1_r["ndcg"],
            "delta_t2_recall": m2_r["t2_recall"] - m1_r["t2_recall"],
            "delta_psr":       m2_r["psr"]       - m1_r["psr"],
        }
        results[f"{group_name}_delta"] = delta
        log.info(
            "  Δ(M2-M1) %s: nDCG=%.4f  T2-Recall=%.4f  PSR=%.4f",
            group_name, delta["delta_ndcg"], delta["delta_t2_recall"], delta["delta_psr"],
        )

    hard_d = results["hard_delta"]["delta_t2_recall"]
    easy_d = results["easy_delta"]["delta_t2_recall"]
    log.info("KEY: Hard Δ=%.4f  Easy Δ=%.4f  (expected Hard >> Easy)", hard_d, easy_d)

    return results


# ---------------------------------------------------------------------------
# Analysis B — ORIG Signal Alignment
# ---------------------------------------------------------------------------

def _rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial r = (2U - n1*n2) / (n1*n2); effect size for Mann-Whitney U.
    +1 means all group-1 scores exceed all group-2 scores; -1 means the opposite."""
    return (2.0 * u_stat - n1 * n2) / (n1 * n2)


def analysis_b_orig_alignment(
    instances: list[dict],
    tier3_lookup: dict[str, list[dict]],
    m1_encoder: EntailmentEncoder,
    m2_encoder: EntailmentEncoder,
    device: torch.device,
    cfg: Config,
) -> dict:
    """Per-hunk sim(hunk_emb, orig_emb) by tier; Mann-Whitney U for M1 vs M2."""
    log.info("=== Analysis B: ORIG Signal Alignment ===")

    # orig_sims[model][group] = list of sim(hunk, orig) scores
    orig_sims: dict[str, dict[str, list[float]]] = {
        "m1": {"retained": [], "t3": []},
        "m2": {"retained": [], "t3": []},
    }

    for inst in instances:
        iid          = inst.get("instance_id", "")
        source_units = inst.get("source_units", [])

        all_hunks: list[tuple[dict, int]] = [
            (h, h.get("tier_label") or 1) for h in retained_hunks(inst, cfg)
        ]
        for h in tier3_lookup.get(iid, []):
            all_hunks.append((h, 3))

        for hunk, tier in all_hunks:
            hunk_text = _render_hunk(hunk)
            if not hunk_text.strip():
                continue

            relevant_units = _units_for_hunk(hunk, source_units)
            orig_texts     = [u["code"] for u in relevant_units if u.get("code")]
            if not orig_texts:
                # Can't measure ORIG-sim without a matched source unit — skip
                continue

            group = "retained" if tier in (1, 2) else "t3"

            for model_name, encoder in [("m1", m1_encoder), ("m2", m2_encoder)]:
                h_emb  = encoder.encode([hunk_text], "HUNK", device, use_projection=True)  # (1, D)
                o_embs = encoder.encode(orig_texts,  "ORIG", device, use_projection=True)  # (N, D)
                o_mean = F.normalize(o_embs.mean(dim=0, keepdim=True), dim=-1)             # (1, D)
                sim    = (h_emb * o_mean).sum(-1).item()
                orig_sims[model_name][group].append(sim)

    results: dict = {}

    for model_name in ("m1", "m2"):
        retained_s = orig_sims[model_name]["retained"]
        t3_s   = orig_sims[model_name]["t3"]

        log.info(
            "%s ORIG-sim: Retained-H n=%d mean=%.4f std=%.4f | t3 n=%d mean=%.4f std=%.4f",
            model_name,
            len(retained_s), np.mean(retained_s) if retained_s else 0.0, np.std(retained_s) if retained_s else 0.0,
            len(t3_s),   np.mean(t3_s)   if t3_s   else 0.0, np.std(t3_s)   if t3_s   else 0.0,
        )

        if len(retained_s) >= 2 and len(t3_s) >= 2:
            u_stat, p = mannwhitneyu(retained_s, t3_s, alternative="greater")
            r         = _rank_biserial(u_stat, len(retained_s), len(t3_s))
            sig       = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            log.info(
                "%s Mann-Whitney U [Retained-H>t3]: U=%.0f  p=%.4e  r=%.4f  %s",
                model_name, u_stat, p, r, sig,
            )
            results[model_name] = {
                "n_retained":           len(retained_s),
                "n_t3":             len(t3_s),
                "mean_retained":        float(np.mean(retained_s)),
                "std_retained":         float(np.std(retained_s)),
                "mean_t3":          float(np.mean(t3_s)),
                "std_t3":           float(np.std(t3_s)),
                "mean_gap":         float(np.mean(retained_s) - np.mean(t3_s)),
                "u_stat":           float(u_stat),
                "p_value":          float(p),
                "rank_biserial_r":  float(r),
                "significant":      sig,
            }
        else:
            log.warning("%s: insufficient data (n_retained=%d, n_t3=%d)", model_name, len(retained_s), len(t3_s))
            results[model_name] = {}

    if results.get("m1") and results.get("m2"):
        delta_r   = results["m2"]["rank_biserial_r"] - results["m1"]["rank_biserial_r"]
        delta_gap = results["m2"]["mean_gap"]         - results["m1"]["mean_gap"]
        results["delta_rank_biserial_r"] = delta_r
        results["delta_mean_gap"]        = delta_gap
        direction = "stronger" if delta_r > 0 else "weaker"
        log.info(
            "KEY: M2 has %s ORIG alignment than M1  Δr=%.4f  Δmean_gap=%.4f",
            direction, delta_r, delta_gap,
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Exp3: Mechanism analysis for Edit Entailment")
    parser.add_argument("--m1",       required=True, help="M1 checkpoint path (e.g. checkpoints/m1/best.pt)")
    parser.add_argument("--m2",       required=True, help="M2 checkpoint path (e.g. checkpoints/m2/best.pt)")
    parser.add_argument("--instances", default="data/processed/instances_full.jsonl")
    parser.add_argument("--tier3",     default="data/cache/tier3_hunks.jsonl")
    parser.add_argument("--output",    default="logs/exp3_analysis.json")
    parser.add_argument("--encoder",   default="microsoft/unixcoder-base")
    parser.add_argument("--analysis",  choices=["a", "b", "all"], default="all",
                        help="a=stratified eval, b=ORIG alignment, all=both")
    add_retained_args(parser)
    parser.add_argument("--allow-legacy-checkpoint", action="store_true",
                        help="Load checkpoints without a data_provenance stamp (archival diagnostics only)")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    log.info("Device: %s", device)

    cfg = default_config
    apply_retained_args(cfg, args)
    log.info(describe_hunk_files(cfg.data.llm_t12_path, args.tier3))
    cfg.data.instances_path = args.instances
    cfg.eval.tier3_path          = args.tier3
    cfg.model.encoder_name       = args.encoder

    log.info("Loading %d test instances...", 0)
    instances    = load_test_instances(args.instances)
    log.info("Test instances: %d", len(instances))

    tier3_lookup = load_tier3_lookup(args.tier3)
    log.info("Tier3 lookup: %d instances with scope-creep hunks", len(tier3_lookup))

    log.info("Loading M1 encoder...")
    m1_enc = load_encoder(args.m1, cfg, device, allow_legacy=args.allow_legacy_checkpoint)
    log.info("Loading M2 encoder...")
    m2_enc = load_encoder(args.m2, cfg, device, allow_legacy=args.allow_legacy_checkpoint)

    output: dict = {
        "config": {
            "m1_checkpoint": args.m1,
            "m2_checkpoint": args.m2,
            "instances":     args.instances,
            "tier3":         args.tier3,
            "encoder":       args.encoder,
            "n_test_instances": len(instances),
        }
    }

    if args.analysis in ("a", "all"):
        log.info("Loading M0 (untrained) encoder for stratification...")
        m0_enc = load_encoder(None, cfg, device)
        output["analysis_a"] = analysis_a_stratified(
            instances, tier3_lookup, m1_enc, m2_enc, m0_enc, device, cfg,
        )

    if args.analysis in ("b", "all"):
        output["analysis_b"] = analysis_b_orig_alignment(
            instances, tier3_lookup, m1_enc, m2_enc, device, cfg,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    log.info("Results saved → %s", out_path)


if __name__ == "__main__":
    main()
