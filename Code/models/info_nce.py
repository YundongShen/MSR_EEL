"""Symmetric InfoNCE loss for Edit Entailment Learning.

For a batch of (anchor, positive) embedding pairs, every other item in the
batch is an in-batch negative.  Loss is computed in both directions so that
neither side of the pair is privileged as the anchor.

Hard negatives (e.g. same-instance Tier-3 hunks) can be supplied via the
hard_negatives argument to MultiPairInfoNCE.forward().  They are appended to
the b-side negative pool for the a→b direction only; the b→a direction uses
only the in-batch pool so that hard-neg embeddings are never used as anchors.

A boolean hard_negative_mask (B, H) says which hard negatives belong to which anchor: anchor n is
contrasted only with ITS OWN hard negatives (paper Eq. 2: in-batch ∪ T3(x_n)), never with the hard
negatives of the other anchors in the batch.  Without a mask every anchor sees the whole pool.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    """Symmetric InfoNCE over a batch of embedding pairs.

    Parameters
    ----------
    temperature:
        Initial value of the learnable temperature scalar τ.
        Clamped to [0.01, 1.0] during forward to prevent collapse.
    learn_temperature:
        If True, τ is a learnable parameter (CLIP-style).
        If False, it is a fixed buffer.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learn_temperature: bool = True,
    ) -> None:
        super().__init__()
        log_tau = torch.tensor(temperature).log()
        if learn_temperature:
            self.log_temperature = nn.Parameter(log_tau)
        else:
            self.register_buffer("log_temperature", log_tau)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(0.01, 1.0)

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        hard_neg_embs: torch.Tensor | None = None,
        tiers: list[int] | None = None,
        hard_neg_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute symmetric InfoNCE loss with optional hard negatives and tier weights.

        Parameters
        ----------
        emb_a, emb_b:
            L2-normalised embeddings of shape (B, D).
            emb_a[i] and emb_b[i] are a positive pair;
            all cross-index combinations are in-batch negatives.
        hard_neg_embs:
            Optional (H, D) tensor of extra negatives appended to the b-side
            for the a→b direction.  b→a uses only the in-batch pool.
        hard_neg_mask:
            Optional (B, H) bool tensor; True where hard negative h is a negative of anchor i.
            None = every anchor sees every hard negative.
        tiers:
            Optional list of B tier labels (1 or 2).  Tier-1 pairs get
            weight=1.0; Tier-2 pairs get weight=0.67 (matching the 3:2
            relevance ratio used in nDCG evaluation).

        Returns
        -------
        torch.Tensor — scalar loss.
        """
        B = len(emb_a)
        labels = torch.arange(B, device=emb_a.device)

        # a→b: positive at column i; hard negs extend the negative pool
        logits_ab = emb_a @ emb_b.T / self.temperature              # (B, B)
        if hard_neg_embs is not None and hard_neg_embs.size(0) > 0:
            logits_hard = emb_a @ hard_neg_embs.T / self.temperature  # (B, H)
            if hard_neg_mask is not None:
                # -inf removes a hard negative from the softmax of every anchor it does not belong to
                logits_hard = logits_hard.masked_fill(~hard_neg_mask, float("-inf"))
            logits_ab = torch.cat([logits_ab, logits_hard], dim=1)  # (B, B+H)

        # b→a: symmetric, in-batch only (hard negs have no anchor on the a-side)
        logits_ba = emb_b @ emb_a.T / self.temperature              # (B, B)

        if tiers is not None:
            # Per-pair weighting: Tier-1 → 1.0, Tier-2 → 0.67 (3:2 relevance ratio)
            w = torch.tensor(
                [1.0 if t == 1 else 0.67 if t == 2 else 1.0 for t in tiers],
                device=emb_a.device, dtype=torch.float32,
            )
            loss_ab = (F.cross_entropy(logits_ab, labels, reduction="none") * w).mean()
            loss_ba = (F.cross_entropy(logits_ba, labels, reduction="none") * w).mean()
        else:
            loss_ab = F.cross_entropy(logits_ab, labels)
            loss_ba = F.cross_entropy(logits_ba, labels)

        return (loss_ab + loss_ba) * 0.5


class MultiPairInfoNCE(nn.Module):
    """InfoNCE across multiple positive pair types with a shared temperature.

    The four pair types in Edit Entailment Learning contribute independently:
        (REQ, TEST), (REQ, HUNK), (ORIG, HUNK), (REQ, ORIG)

    Each type contributes one InfoNCE term weighted by its pair_weight.
    Pair types absent from the current batch are skipped silently.
    """

    PAIR_TYPES = ("req_test", "req_hunk", "orig_hunk", "req_orig")

    def __init__(
        self,
        temperature: float = 0.07,
        learn_temperature: bool = True,
        pair_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.info_nce = InfoNCE(temperature, learn_temperature)
        self.pair_weights: dict[str, float] = pair_weights or {t: 1.0 for t in self.PAIR_TYPES}

    def forward(
        self,
        embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
        hard_negatives: dict[str, torch.Tensor] | None = None,
        tiers: dict[str, list[int]] | None = None,
        hard_negative_masks: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute weighted sum of InfoNCE losses over all present pair types.

        Parameters
        ----------
        embeddings:
            Mapping from pair_type → (emb_a, emb_b).
            Only pair types present in the dict are included.
        hard_negatives:
            Optional mapping from pair_type → (H, D) hard-negative embeddings.
            Applied to req_hunk and orig_hunk (Tier-3 + same-repo hunks).
        tiers:
            Optional mapping from pair_type → list of B tier labels.
            Applied to req_hunk and orig_hunk for Tier-aware loss weighting.
        hard_negative_masks:
            Optional mapping from pair_type → (B, H) bool mask of each anchor's own hard negatives.

        Returns
        -------
        total_loss : torch.Tensor — scalar.
        per_type   : dict[str, float] — detached per-type losses for logging.
        """
        total = torch.tensor(0.0, device=next(iter(embeddings.values()))[0].device)
        per_type: dict[str, float] = {}

        for pair_type, (emb_a, emb_b) in embeddings.items():
            if emb_a.size(0) < 2:
                # InfoNCE needs at least 2 samples to form a negative
                continue
            w = self.pair_weights.get(pair_type, 1.0)
            hard_neg  = hard_negatives.get(pair_type) if hard_negatives else None
            tier_list = tiers.get(pair_type) if tiers else None
            mask = hard_negative_masks.get(pair_type) if hard_negative_masks else None
            loss = self.info_nce(emb_a, emb_b, hard_neg, tier_list, mask)
            total = total + w * loss
            per_type[pair_type] = loss.item()

        return total, per_type
