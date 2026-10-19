"""Single encoder for all four entity types in Edit Entailment Learning.

Four entity types, each prepended with a dedicated type token:
  <REQ>  — requirement / issue text
  <TEST> — test function body
  <ORIG> — original project code unit (function or class)
  <HUNK> — diff hunk with surrounding context lines

All four types share the same backbone and projection head.  The type token
tells the model which "role" the text plays, so geometric distances in the
output space reflect entailment relationships rather than surface similarity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ENTITY_TYPES = ["REQ", "TEST", "ORIG", "HUNK"]
TYPE_TOKENS = {t: f"<{t}>" for t in ENTITY_TYPES}  # e.g. {"REQ": "<REQ>", ...}


class EntailmentEncoder(nn.Module):
    """UniXCoder-based encoder projecting all four entity types into one space.

    Parameters
    ----------
    model_name:
        HuggingFace checkpoint.  Default is UniXCoder; CodeBERT also works.
    projection_dim:
        Dimensionality of the shared embedding space.
    dropout:
        Applied inside the projection MLP.
    max_length:
        Token budget per input (type token + text).
    """

    def __init__(
        self,
        model_name: str = "microsoft/unixcoder-base",
        projection_dim: int = 256,
        dropout: float = 0.1,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Register entity-type tokens so the model treats them as atomic units
        self.tokenizer.add_tokens(list(TYPE_TOKENS.values()), special_tokens=True)

        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.resize_token_embeddings(len(self.tokenizer))

        hidden = self.backbone.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, projection_dim),
            nn.LayerNorm(projection_dim),
        )

    # ------------------------------------------------------------------
    # Text preparation
    # ------------------------------------------------------------------

    def _prepend_type(self, texts: list[str], entity_type: str) -> list[str]:
        prefix = TYPE_TOKENS[entity_type]
        return [f"{prefix} {t}" for t in texts]

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def tokenize(
        self,
        texts: list[str],
        entity_type: str,
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Tokenise a list of texts for the given entity type.

        Returns a dict with ``input_ids`` and ``attention_mask`` on *device*.
        """
        formatted = self._prepend_type(texts, entity_type)
        enc = self.tokenizer(
            formatted,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in enc.items()}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode pre-tokenised inputs → L2-normalised embeddings (B, D).

        Uses attention-masked mean pooling over the last hidden states,
        then projects through the MLP head.
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pool over non-padding positions
        mask = attention_mask.unsqueeze(-1).float()          # (B, L, 1)
        summed = (out.last_hidden_state * mask).sum(dim=1)   # (B, H)
        counts = mask.sum(dim=1).clamp(min=1e-9)             # (B, 1)
        pooled = summed / counts                             # (B, H)

        projected = self.projection(pooled)                  # (B, D)
        return F.normalize(projected, dim=-1)

    # ------------------------------------------------------------------
    # Convenience inference API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        entity_type: str,
        device: torch.device | str = "cpu",
        use_projection: bool = True,
    ) -> torch.Tensor:
        """Tokenise and encode in one call.

        Parameters
        ----------
        use_projection:
            If True (default), returns (B, projection_dim) L2-normalised embeddings
            from the full encoder + MLP head.
            If False, bypasses the projection MLP and returns the mean-pooled backbone
            hidden states L2-normalised — used for M0 (untrained UniXCoder baseline)
            so the "pretrained code geometry" claim is reproducible and architecture-neutral.
        """
        enc = self.tokenize(texts, entity_type, device)
        if use_projection:
            return self(enc["input_ids"], enc["attention_mask"])
        # Bypass projection: return L2-normalised mean-pooled backbone output
        out  = self.backbone(enc["input_ids"], attention_mask=enc["attention_mask"])
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return F.normalize(pooled, dim=-1)

    @torch.no_grad()
    def similarity_components(
        self,
        hunk_texts: list[str],
        req_texts: list[str],
        test_texts: list[list[str]],
        orig_texts: list[list[str]],
        device: torch.device | str = "cpu",
        use_projection: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-hunk similarity to each repository view: (sim_req, sim_test, sim_orig), each of shape (B,).

        sim_orig is the similarity to the mean-pooled ORIG unit embeddings, 0 when the hunk has no unit.
        Computing the three once lets the score weights (alpha, beta, gamma) be varied without re-encoding.
        """
        dev = torch.device(device)
        B = len(hunk_texts)
        up = use_projection  # shorthand

        h_emb = self.encode(hunk_texts, "HUNK", dev, up)   # (B, D)
        r_emb = self.encode(req_texts,  "REQ",  dev, up)   # (B, D)

        sim_test = h_emb.new_zeros(B)
        for i, tests in enumerate(test_texts):
            if tests:
                sim_test[i] = (self.encode(tests, "TEST", dev, up) @ h_emb[i]).max()

        # Mean-pool orig embeddings per hunk (caller passes only the relevant ORIG units
        # via _units_for_hunk — Issue 2 fix — so mean over a small set is appropriate)
        o_emb = torch.zeros(B, h_emb.size(-1), device=dev)
        for i, origs in enumerate(orig_texts):
            if origs:
                o_emb[i] = self.encode(origs, "ORIG", dev, up).mean(dim=0)

        sim_req  = (h_emb * r_emb).sum(dim=-1)
        sim_orig = (h_emb * F.normalize(o_emb, dim=-1)).sum(dim=-1)
        return sim_req, sim_test, sim_orig

    @torch.no_grad()
    def entailment_score(
        self,
        hunk_texts: list[str],
        req_texts: list[str],
        test_texts: list[list[str]],
        orig_texts: list[list[str]],
        device: torch.device | str = "cpu",
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.5,
        use_projection: bool = True,
    ) -> torch.Tensor:
        """Compute Edit Entailment Score for each hunk.

        Parameters
        ----------
        hunk_texts:
            List of B hunk strings.
        req_texts:
            List of B requirement strings (one per hunk).
        test_texts:
            List of B lists of test function strings.
        orig_texts:
            List of B lists of original code unit strings.

        Returns
        -------
        torch.Tensor of shape (B,) — scores in roughly [0, 2].
        """
        sim_req, sim_test, sim_orig = self.similarity_components(
            hunk_texts, req_texts, test_texts, orig_texts, device, use_projection,
        )
        return alpha * sim_req + beta * sim_test + gamma * sim_orig
