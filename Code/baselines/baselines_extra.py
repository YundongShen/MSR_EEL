"""Cross-Encoder and embedding baselines for retention ranking."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from config import default_config
from data.retained_hunks import add_retained_args, apply_retained_args
from evaluate import run_baseline_retrieval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MAX_LENGTH = 512


# ---------------------------------------------------------------------------
# Encoder wrappers
# ---------------------------------------------------------------------------

class CrossEncoderScorer:
    """Joint (REQ, HUNK) encoder — no bi-encoder, full cross-attention."""

    def __init__(self, model_name: str, device: torch.device) -> None:
        log.info("Loading cross-encoder: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def score(self, req: str, hunk_texts: list[str], batch_size: int = 16) -> list[float]:
        """Score (req, hunk) pairs. Returns raw logit as relevance score."""
        all_scores: list[float] = []
        for i in range(0, len(hunk_texts), batch_size):
            batch = hunk_texts[i : i + batch_size]
            enc = self.tokenizer(
                [req] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits  # (B, num_labels)
            # num_labels=1 → relevance regression score
            scores = logits[:, 0].cpu().tolist()
            all_scores.extend(scores)
        return all_scores


class BiEncoder:
    """Pretrained embedding scorer."""

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        trust_remote_code: bool = False,
    ) -> None:
        log.info("Loading bi-encoder: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        ).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        """Return L2-normalised embeddings, shape (N, D)."""
        all_embs: list[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**enc)

            # Handle both custom models (tensor output) and standard ModelOutput
            if isinstance(out, torch.Tensor):
                emb = out  # e.g. codet5p-110m-embedding returns embedding directly
            elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                emb = out.pooler_output
            else:
                mask = enc["attention_mask"].unsqueeze(-1).float()
                emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

            emb = F.normalize(emb, dim=-1)
            all_embs.append(emb.cpu())
        return torch.cat(all_embs, dim=0)


# ---------------------------------------------------------------------------
# Shared evaluation loop
# ---------------------------------------------------------------------------

def _eval_loop(
    instances_path: str,
    tier3_path: str | None,
    score_fn,          # callable(req, hunk_texts) -> list[float] over candidates
    model_label: str,
    repo_pool: bool = False,
) -> dict:
    cfg = default_config
    cfg.data.instances_path = instances_path
    cfg.eval.tier3_path = tier3_path or ""
    return run_baseline_retrieval(cfg, score_fn, model_label, repo_pool=repo_pool)


# ---------------------------------------------------------------------------
# Per-mode runners
# ---------------------------------------------------------------------------

def run_cross_encoder(
    model_name: str,
    instances_path: str,
    tier3_path: str | None,
    repo_pool: bool = False,
) -> dict[str, float]:
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer  = CrossEncoderScorer(model_name, device)

    def score_fn(req: str, hunk_texts: list[str]) -> list[float]:
        return scorer.score(req, hunk_texts)

    return _eval_loop(instances_path, tier3_path, score_fn, model_name, repo_pool=repo_pool)


def run_biencoder(
    model_name: str,
    instances_path: str,
    tier3_path: str | None,
    trust_remote_code: bool = False,
    repo_pool: bool = False,
) -> dict[str, float]:
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = BiEncoder(model_name, device, trust_remote_code=trust_remote_code)

    def score_fn(req: str, hunk_texts: list[str]) -> list[float]:
        req_emb   = encoder.encode([req])          # (1, D)
        hunk_embs = encoder.encode(hunk_texts)     # (N, D)
        return (hunk_embs @ req_emb.T).squeeze(-1).tolist()

    return _eval_loop(instances_path, tier3_path, score_fn, model_name, repo_pool=repo_pool)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "cross_encoder":  "checkpoints/cross_encoder",
    "codet5":         "Salesforce/codet5p-110m-embedding",
    "coderetriever":  "Lazyhope/unixcoder-nine-cpp",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extra baselines for Exp 1")
    parser.add_argument(
        "--mode", choices=["cross_encoder", "codet5", "coderetriever"], required=True,
        help="Which baseline to run",
    )
    parser.add_argument(
        "--model", default=None,
        help="HuggingFace model name (overrides per-mode default)",
    )
    parser.add_argument(
        "--instances", default="data/processed/instances_full.jsonl",
    )
    parser.add_argument(
        "--tier3", default="data/cache/tier3_hunks.jsonl",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to save JSON results",
    )
    parser.add_argument("--repo-pool", action="store_true",
                        help="Add same-repo cross-issue retained hunks as tier-0 distractors (as evaluate.py --repo-pool)")
    add_retained_args(parser)
    args = parser.parse_args()
    apply_retained_args(default_config, args)
    log.info("Retained-H: %s", default_config.data.llm_t12_path)

    model_name = args.model or _DEFAULTS[args.mode]
    log.info("Mode: %s  Model: %s", args.mode, model_name)

    if args.mode == "cross_encoder":
        results = run_cross_encoder(model_name, args.instances, args.tier3, repo_pool=args.repo_pool)

    elif args.mode == "codet5":
        results = run_biencoder(
            model_name, args.instances, args.tier3,
            trust_remote_code=True, repo_pool=args.repo_pool,
        )

    elif args.mode == "coderetriever":
        results = run_biencoder(
            model_name, args.instances, args.tier3,
            trust_remote_code=False, repo_pool=args.repo_pool,
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(results, fh, indent=2)
        log.info("Results saved → %s", args.output)
