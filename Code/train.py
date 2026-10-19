"""Training entry point for Edit Entailment Learning.

Trains a single EntailmentEncoder on four positive pair types:
    req_test  — (REQ, TEST)
    req_hunk  — (REQ, HUNK)
    orig_hunk — (ORIG, HUNK)
    req_orig  — (REQ, ORIG)

Each batch contains pairs from all active pair types.  The InfoNCE loss
treats in-batch cross-index combinations as negatives; no explicit negative
construction is needed.

Usage:
    python train.py
    python train.py --instances data/processed/instances_full.jsonl --tier3 data/cache/tier3_hunks.jsonl
    python train.py --resume checkpoints/epoch_003.pt
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import Config, default_config
from data.entailment_dataset import EntailmentDataset, EntailmentPair, _render_hunk
from data.provenance import build_stamp, check_checkpoint, describe_hunk_files, write_stamp_json
from data.retained_hunks import add_retained_args, apply_retained_args, load_tier3_hunks
from models.entailment_encoder import EntailmentEncoder
from models.info_nce import MultiPairInfoNCE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier-3 hard negatives
# ---------------------------------------------------------------------------

def load_tier3_for_training(path: str | Path) -> dict[str, list[str]]:
    """Load tier3_hunks.jsonl → {instance_id: [rendered_hunk_text, ...]} (validated LLM-generated hunks)."""
    result: dict[str, list[str]] = {}
    for iid, hunks in load_tier3_hunks(path).items():
        texts = [_render_hunk(h) for h in hunks]
        texts = [t for t in texts if t.strip()]
        if texts:
            result[iid] = texts
    return result


# ---------------------------------------------------------------------------
# Same-repo cross-issue hard negatives
# ---------------------------------------------------------------------------

def build_same_repo_hard_negs(dataset: EntailmentDataset) -> dict[str, list[str]]:
    """For each training instance, collect retained hunks from OTHER instances in the same repo.

    Returns {instance_id: [hunk_text, ...]} — sampled at training time as hard negs.
    Repo is inferred from instance_id by stripping the trailing issue number
    (e.g. 'django__django-12345' → repo='django__django').
    Only HUNK-side pairs (req_hunk, orig_hunk) are indexed to keep type consistent.
    """
    repo_index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    seen_per_iid: dict[str, set[str]] = defaultdict(set)

    for pair in dataset:
        if pair.pair_type not in ("req_hunk", "orig_hunk"):
            continue
        repo = pair.instance_id.rsplit("-", 1)[0]
        if pair.text_b not in seen_per_iid[pair.instance_id]:
            seen_per_iid[pair.instance_id].add(pair.text_b)
            repo_index[repo].append((pair.instance_id, pair.text_b))

    result: dict[str, list[str]] = {}
    for repo, iid_hunks in repo_index.items():
        for iid, _ in iid_hunks:
            others = [h for oid, h in iid_hunks if oid != iid]
            if others:
                result[iid] = others

    return result


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_fn(batch: list[EntailmentPair]) -> dict:
    """Group pairs by type; return dict of lists keyed by pair_type."""
    groups: dict[str, list[EntailmentPair]] = defaultdict(list)
    for pair in batch:
        groups[pair.pair_type].append(pair)
    return dict(groups)


def balanced_sampler(dataset: EntailmentDataset) -> WeightedRandomSampler:
    """Return a sampler that draws each pair_type with equal probability.

    Without this, req_orig (which can be 20× more frequent than req_hunk)
    would dominate every batch, starving the other three pair types of
    in-batch negatives.
    """
    from collections import Counter
    counts = Counter(p.pair_type for p in dataset)
    type_weight = {pt: 1.0 / cnt for pt, cnt in counts.items()}
    weights = [type_weight[p.pair_type] for p in dataset]
    return WeightedRandomSampler(weights, num_samples=250000, replacement=True)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def _warmup_cosine(warmup: int, total: int):
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step) / max(1, warmup)
        progress = float(step - warmup) / max(1, total - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


# ---------------------------------------------------------------------------
# One training step
# ---------------------------------------------------------------------------

def _train_step(
    encoder: EntailmentEncoder,
    loss_fn: MultiPairInfoNCE,
    batch_groups: dict,
    device: torch.device,
    tier3_by_instance: dict[str, list[str]] | None = None,
    same_repo_hunks: dict[str, list[str]] | None = None,
    tier_aware_loss: bool = False,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Encode all pair types and compute the combined InfoNCE loss.

    Hard negatives are per anchor (paper Eq. 2): an anchor is contrasted with the Tier-3 hunks of its OWN
    instance (M2) and/or with same-repo cross-issue hunks sampled for it (M3), never with those of the
    other anchors in the batch.  ``hard_masks[pair_type][i, j]`` marks hard negative j as anchor i's.
    """
    rng = rng or random
    embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    hard_negatives: dict[str, torch.Tensor] = {}
    hard_masks: dict[str, torch.Tensor] = {}
    tiers_for_loss: dict[str, list[int]] = {}

    for pair_type, pairs in batch_groups.items():
        texts_a = [p.text_a for p in pairs]
        texts_b = [p.text_b for p in pairs]
        type_a  = pairs[0].type_a
        type_b  = pairs[0].type_b

        enc_a = encoder.tokenize(texts_a, type_a, device)
        enc_b = encoder.tokenize(texts_b, type_b, device)

        emb_a = encoder(enc_a["input_ids"], enc_a["attention_mask"])
        emb_b = encoder(enc_b["input_ids"], enc_b["attention_mask"])

        embeddings[pair_type] = (emb_a, emb_b)

        if pair_type in ("req_hunk", "orig_hunk"):
            # Tier-aware loss weighting (M4 only — controlled by tier_aware_loss flag)
            if tier_aware_loss:
                tiers_for_loss[pair_type] = [p.tier for p in pairs]

            # Hard negatives per anchor: own-instance Tier-3 (M2) and/or same-repo cross-issue (M3)
            hard_index: dict[str, int] = {}            # text -> column in the hard-negative matrix
            hard_texts: list[str] = []
            owners: list[tuple[int, int]] = []         # (anchor row, hard-negative column)
            for i, p in enumerate(pairs):
                own = list((tier3_by_instance or {}).get(p.instance_id, []))
                pool = (same_repo_hunks or {}).get(p.instance_id, [])
                if pool:
                    own += rng.sample(pool, min(3, len(pool)))
                for t in own:
                    j = hard_index.get(t)
                    if j is None:
                        j = hard_index[t] = len(hard_texts)
                        hard_texts.append(t)
                    owners.append((i, j))
            if hard_texts:
                enc_hn = encoder.tokenize(hard_texts, "HUNK", device)
                hard_negatives[pair_type] = encoder(enc_hn["input_ids"], enc_hn["attention_mask"])
                mask = torch.zeros(len(pairs), len(hard_texts), dtype=torch.bool)
                for i, j in owners:
                    mask[i, j] = True
                hard_masks[pair_type] = mask.to(device)

    return loss_fn(embeddings, hard_negatives or None, tiers_for_loss or None, hard_masks or None)


# ---------------------------------------------------------------------------
# Epoch loops
# ---------------------------------------------------------------------------

def train_epoch(
    encoder: EntailmentEncoder,
    loss_fn: MultiPairInfoNCE,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
    cfg: Config,
    global_step: int,
    tier3_by_instance: dict[str, list[str]] | None = None,
    same_repo_hunks: dict[str, list[str]] | None = None,
    tier_aware_loss: bool = False,
) -> tuple[float, int]:
    encoder.train()
    loss_fn.train()
    total_loss = 0.0
    n_batches = 0

    for batch_groups in loader:
        total, per_type = _train_step(
            encoder, loss_fn, batch_groups, device,
            tier3_by_instance, same_repo_hunks, tier_aware_loss,
        )

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(loss_fn.parameters()),
            cfg.train.grad_clip,
        )
        optimizer.step()
        scheduler.step()

        total_loss += total.item()
        n_batches += 1
        global_step += 1

        if global_step % cfg.train.log_every == 0:
            lr  = scheduler.get_last_lr()[0]
            tau = loss_fn.info_nce.temperature.item()
            per = "  ".join(f"{k}={v:.3f}" for k, v in per_type.items())
            log.info(
                "step=%d  loss=%.4f  τ=%.3f  lr=%.2e  [%s]",
                global_step, total.item(), tau, lr, per,
            )

    return total_loss / max(1, n_batches), global_step


@torch.no_grad()
def validate(
    encoder: EntailmentEncoder,
    loss_fn: MultiPairInfoNCE,
    loader: DataLoader,
    device: torch.device,
    tier3_by_instance: dict[str, list[str]] | None = None,
    same_repo_hunks: dict[str, list[str]] | None = None,
    tier_aware_loss: bool = False,
    seed: int = 0,
) -> float:
    """Validation loss under the SAME objective the variant is trained with (same hard negatives, same
    weighting), so checkpoint selection measures what the variant learns.  ``same_repo_hunks`` must be
    built from the validation set.  Same-repo sampling uses a fixed seed: the loss is deterministic."""
    encoder.eval()
    loss_fn.eval()
    total_loss = 0.0
    n_batches = 0
    rng = random.Random(seed)

    for batch_groups in loader:
        total, _ = _train_step(
            encoder, loss_fn, batch_groups, device,
            tier3_by_instance, same_repo_hunks, tier_aware_loss, rng,
        )
        total_loss += total.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    encoder: EntailmentEncoder,
    loss_fn: MultiPairInfoNCE,
    optimizer: AdamW,
    epoch: int,
    val_loss: float,
    path: Path,
    stamp: dict,
) -> None:
    torch.save({
        "epoch": epoch,
        "val_loss": val_loss,
        "data_provenance": stamp,   # what this model was trained on; checked by every loader
        "encoder_state": encoder.state_dict(),
        "loss_fn_state": loss_fn.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)
    log.info("Saved checkpoint → %s  (val_loss=%.4f)", path, val_loss)


def load_checkpoint(
    path: Path,
    encoder: EntailmentEncoder,
    loss_fn: MultiPairInfoNCE,
    optimizer: AdamW,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu")
    check_checkpoint(ckpt, path)   # never resume from a checkpoint trained under another hunk protocol
    encoder.load_state_dict(ckpt["encoder_state"])
    loss_fn.load_state_dict(ckpt["loss_fn_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    epoch    = ckpt.get("epoch", 0)
    val_loss = ckpt.get("val_loss", float("inf"))
    log.info("Resumed from %s  (epoch=%d, val_loss=%.4f)", path, epoch, val_loss)
    return epoch, val_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: Config = default_config) -> None:
    # Ensure checkpoint directory exists (CLI may override the default path)
    Path(cfg.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Reproducibility
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    log.info("Device: %s", device)

    # ---- Dataset -----------------------------------------------------------
    instances_path = Path(cfg.data.instances_path)
    if not instances_path.exists():
        raise FileNotFoundError(
            f"{instances_path} not found.\n"
            "Run: python data_pipeline/parse_instances.py "
            "--input data/raw/swebench_full_instances.jsonl "
            "--output data/processed/instances_full.jsonl"
        )

    log.info("Loading dataset from %s", instances_path)
    log.info(describe_hunk_files(cfg.data.llm_t12_path, cfg.data.tier3_hard_neg_path or None))
    full_dataset = EntailmentDataset(
        instances_path=instances_path,
        llm_t12_path=cfg.data.llm_t12_path,
        pair_types=cfg.data.pair_types,
        max_req_chars=cfg.data.max_req_chars,
        seed=cfg.train.seed,
    )
    log.info("Dataset stats: %s", full_dataset.stats())

    train_set, val_set, _ = full_dataset.load_splits("data/processed/splits.json")
    log.info("Splits — train: %d  val: %d", len(train_set), len(val_set))

    # Fail loudly instead of training a hunk-less model (e.g. a wrong or empty retained-hunk file).
    train_pair_counts = train_set.stats()["by_pair_type"]
    for pt in ("req_hunk", "orig_hunk"):
        if pt in cfg.data.pair_types and train_pair_counts.get(pt, 0) == 0:
            raise RuntimeError(
                f"No '{pt}' training pairs were built from {cfg.data.llm_t12_path}: "
                "check the retained-hunk file and the instances file."
            )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.train.batch_size,
        sampler=balanced_sampler(train_set),  # equal probability per pair_type
        collate_fn=collate_fn,
        num_workers=min(cfg.train.num_workers, os.cpu_count() or 1),
        drop_last=True,   # avoid single-item batches that crash InfoNCE
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=min(cfg.train.num_workers, os.cpu_count() or 1),
        drop_last=False,
    )

    # ---- Model -------------------------------------------------------------
    encoder = EntailmentEncoder(
        model_name=cfg.model.encoder_name,
        projection_dim=cfg.model.projection_dim,
        dropout=cfg.model.dropout,
        max_length=cfg.model.max_length,
    ).to(device)

    loss_fn = MultiPairInfoNCE(
        temperature=cfg.train.temperature,
        learn_temperature=cfg.train.learn_temperature,
        pair_weights=cfg.train.pair_weights,
    ).to(device)

    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    log.info("Encoder parameters: %s", f"{n_params:,}")

    # ---- Optimiser ---------------------------------------------------------
    optimizer = AdamW(
        list(encoder.parameters()) + list(loss_fn.parameters()),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    total_steps = cfg.train.epochs * len(train_loader)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=_warmup_cosine(cfg.train.warmup_steps, total_steps),
    )

    # ---- Resume ------------------------------------------------------------
    start_epoch  = 0
    best_val_loss = float("inf")
    ckpt_dir = Path(cfg.train.checkpoint_dir)
    if cfg.train.resume_from and Path(cfg.train.resume_from).exists():
        start_epoch, best_val_loss = load_checkpoint(
            Path(cfg.train.resume_from), encoder, loss_fn, optimizer
        )

    global_step = start_epoch * len(train_loader)

    # ---- Hard negatives ----------------------------------------------------
    tier3_by_instance: dict[str, list[str]] = {}
    if cfg.data.tier3_hard_neg_path and Path(cfg.data.tier3_hard_neg_path).exists():
        tier3_by_instance = load_tier3_for_training(cfg.data.tier3_hard_neg_path)
        log.info(
            "Loaded Tier-3 hard negatives: %d instances (applied to req_hunk + orig_hunk)",
            len(tier3_by_instance),
        )

    same_repo_hunks: dict[str, list[str]] = {}
    if cfg.train.same_repo_hard_neg:
        same_repo_hunks = build_same_repo_hard_negs(train_set)
        log.info(
            "Built same-repo cross-issue hard-neg index: %d instances with cross-issue pool",
            len(same_repo_hunks),
        )

    same_repo_val: dict[str, list[str]] = build_same_repo_hard_negs(val_set) if cfg.train.same_repo_hard_neg else {}

    if cfg.train.tier_aware_loss:
        log.info("Tier-aware loss enabled: Tier-1 weight=1.0, Tier-2 weight=0.67")

    # Provenance stamp: stored in every checkpoint (data.provenance.check_checkpoint refuses checkpoints without it).
    stamp = build_stamp(
        retained_path=cfg.data.llm_t12_path,
        tier3_path=cfg.data.tier3_hard_neg_path or None,
        instances_path=instances_path,
        pair_types=cfg.data.pair_types,
        pair_counts=train_pair_counts,
        same_repo=cfg.train.same_repo_hard_neg,
        tier_aware_loss=cfg.train.tier_aware_loss,
    )
    write_stamp_json(stamp, ckpt_dir)
    log.info("Data provenance: %s", stamp)

    # ---- Training loop -----------------------------------------------------
    for epoch in range(start_epoch, cfg.train.epochs):
        log.info("=== Epoch %d / %d ===", epoch + 1, cfg.train.epochs)

        train_loss, global_step = train_epoch(
            encoder, loss_fn, train_loader, optimizer, scheduler,
            device, cfg, global_step,
            tier3_by_instance=tier3_by_instance or None,
            same_repo_hunks=same_repo_hunks or None,
            tier_aware_loss=cfg.train.tier_aware_loss,
        )
        val_loss = validate(
            encoder, loss_fn, val_loader, device,
            tier3_by_instance=tier3_by_instance or None,
            same_repo_hunks=same_repo_val or None,
            tier_aware_loss=cfg.train.tier_aware_loss,
        )

        log.info(
            "Epoch %d — train_loss=%.4f  val_loss=%.4f",
            epoch + 1, train_loss, val_loss,
        )

        save_checkpoint(
            encoder, loss_fn, optimizer, epoch + 1, val_loss,
            ckpt_dir / f"epoch_{epoch + 1:03d}.pt", stamp,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                encoder, loss_fn, optimizer, epoch + 1, val_loss,
                ckpt_dir / "best.pt", stamp,
            )

    log.info("Training complete. Best val_loss=%.4f", best_val_loss)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Edit Entailment encoder")
    parser.add_argument("--instances", default=None, help="Path to instances JSONL")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--encoder", default=None, help="HuggingFace model name")
    parser.add_argument(
        "--pair-types", nargs="+",
        choices=["req_test", "req_hunk", "orig_hunk", "req_orig"],
        default=None,
        help="Subset of pair types to train on (ablation). Default: all four.",
    )
    parser.add_argument("--checkpoint-dir", default=None, help="Override checkpoint directory")
    parser.add_argument("--tier3", default=None, help="Path to tier3_hunks.jsonl: own-instance Tier-3 hard negatives (M2)")
    add_retained_args(parser)
    parser.add_argument("--same-repo", action="store_true",
                        help="Same-repo cross-issue hard negatives (M3 = M2 + this flag, i.e. together with --tier3)")
    parser.add_argument("--tier-aware-loss", action="store_true", help="Tier-1/2 loss weighting (M4 = M3 + this flag, i.e. with --tier3 and --same-repo)")
    args = parser.parse_args()

    cfg = default_config
    if args.instances:
        cfg.data.instances_path = args.instances
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.lr:
        cfg.train.lr = args.lr
    if args.resume:
        cfg.train.resume_from = args.resume
    if args.encoder:
        cfg.model.encoder_name = args.encoder
    if args.pair_types:
        cfg.data.pair_types = tuple(args.pair_types)
    if args.checkpoint_dir:
        cfg.train.checkpoint_dir = args.checkpoint_dir
    if args.tier3:
        cfg.data.tier3_hard_neg_path = args.tier3
    apply_retained_args(cfg, args)
    if args.same_repo:
        cfg.train.same_repo_hard_neg = True
    if args.tier_aware_loss:
        cfg.train.tier_aware_loss = True

    main(cfg)
