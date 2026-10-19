"""Pretrained-encoder baseline for retention ranking.

Scores each candidate hunk as:
    score = cosine_similarity( mean_pool(encode(REQ)), mean_pool(encode(HUNK)) )

Works with any HuggingFace masked-LM encoder:
    microsoft/graphcodebert-base
    microsoft/codebert-base
    microsoft/unixcoder-base   (untrained, same backbone as EEL)

No fine-tuning — pure zero-shot transfer from pretraining.

Usage:
    python baselines/baselines_pretrained.py --model microsoft/graphcodebert-base
    python baselines/baselines_pretrained.py --model microsoft/codebert-base \\
        --instances data/processed/instances_full.jsonl \\
        --tier3     data/cache/tier3_hunks.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

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
# Encoder wrapper
# ---------------------------------------------------------------------------

class PretrainedEncoder:
    """Mean-pool HuggingFace encoder. Runs on GPU if available."""

    def __init__(self, model_name: str, device: torch.device) -> None:
        log.info("Loading tokenizer and model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        """Return L2-normalised mean-pooled embeddings, shape (N, D)."""
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
            # Mean pool over non-padding tokens
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = F.normalize(emb, dim=-1)
            all_embs.append(emb.cpu())
        return torch.cat(all_embs, dim=0)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_pretrained_baseline(
    model_name: str,
    instances_path: str,
    tier3_path: str | None,
    all_instances: bool = False,
    repo_pool: bool = False,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    encoder = PretrainedEncoder(model_name, device)

    cfg = default_config
    cfg.data.instances_path = instances_path
    cfg.eval.tier3_path = tier3_path or ""

    def score_fn(req: str, hunk_texts: list[str]) -> list[float]:
        req_emb   = encoder.encode([req])         # (1, D)
        hunk_embs = encoder.encode(hunk_texts)    # (N, D)
        return (hunk_embs @ req_emb.T).squeeze(-1).tolist()

    return run_baseline_retrieval(cfg, score_fn, model_name, all_instances=all_instances, repo_pool=repo_pool)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot pretrained encoder baseline")
    parser.add_argument(
        "--model", default="microsoft/graphcodebert-base",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--instances", default="data/processed/instances_full.jsonl",
        help="Path to instances JSONL",
    )
    parser.add_argument(
        "--tier3", default="data/cache/tier3_hunks.jsonl",
        help="Path to tier3_hunks.jsonl",
    )
    parser.add_argument(
        "--all-instances", action="store_true",
        help="Evaluate on all instances, not just test split",
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

    results = run_pretrained_baseline(
        model_name=args.model,
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
