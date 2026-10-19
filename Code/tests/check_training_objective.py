"""Regression checks for the training objective and model selection.

Run from the project root (CPU only, no model download):
    CUDA_VISIBLE_DEVICES="" python tests/check_training_objective.py

  1. InfoNCE with a hard-negative mask equals the hand-computed loss over in-batch ∪ OWN hard negatives,
     and differs from the shared-pool loss (the old behaviour: every anchor saw every anchor's Tier-3 hunks)
  2. train._train_step builds the mask so that anchor i is masked to the Tier-3 hunks of ITS instance
     (and the same-repo hunks sampled for it), and nothing leaks between anchors / instances
  3. train.validate uses the variant's hard negatives (validation loss measures what the variant learns)
  4. M3 = M2 + same-repo negatives; M4 = M3 + tier weighting
"""

from __future__ import annotations

import math
import random
import re
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import train                                                     # noqa: E402
from data.entailment_dataset import EntailmentPair               # noqa: E402
from models.info_nce import InfoNCE                              # noqa: E402


def _norm(n: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g), dim=-1)


# ---------------------------------------------------------------------------
def check_masked_infonce() -> None:
    B, H, D = 3, 4, 8
    a, b, hard = _norm(B, D, 1), _norm(B, D, 2), _norm(H, D, 3)
    allowed = [[0], [1, 2], []]                                  # anchor 2 has no hard negative
    mask = torch.zeros(B, H, dtype=torch.bool)
    for i, cols in enumerate(allowed):
        mask[i, cols] = True

    nce = InfoNCE(temperature=0.1, learn_temperature=False)
    tau = nce.temperature.item()

    # hand computation: a->b over in-batch columns + own hard columns; b->a in-batch only
    ab = 0.0
    for i in range(B):
        logits = torch.cat([a[i] @ b.T, a[i] @ hard[allowed[i]].T]) / tau
        ab += F.cross_entropy(logits.unsqueeze(0), torch.tensor([i])).item()
    ab /= B
    ba = F.cross_entropy(b @ a.T / tau, torch.arange(B)).item()
    expected = 0.5 * (ab + ba)

    got = nce(a, b, hard, None, mask).item()
    assert math.isclose(got, expected, rel_tol=1e-5, abs_tol=1e-6), (got, expected)

    shared = nce(a, b, hard, None, None).item()                  # old behaviour: whole pool for every anchor
    assert not math.isclose(got, shared, rel_tol=1e-4), "mask had no effect"
    all_true = nce(a, b, hard, None, torch.ones(B, H, dtype=torch.bool)).item()
    assert math.isclose(all_true, shared, rel_tol=1e-6), "all-True mask must equal the shared pool"

    # a hard negative outside every mask row must not change the loss at all
    extra = torch.cat([hard, _norm(1, D, 9)])
    mask2 = torch.cat([mask, torch.zeros(B, 1, dtype=torch.bool)], dim=1)
    assert math.isclose(nce(a, b, extra, None, mask2).item(), got, rel_tol=1e-6), "unowned hard negative leaked in"


# ---------------------------------------------------------------------------
class _StubEncoder:
    """tokenize() keeps the texts; __call__ returns a deterministic embedding per text."""
    def tokenize(self, texts, entity_type, device):
        return {"input_ids": texts, "attention_mask": None}

    def __call__(self, input_ids, attention_mask):
        rows = [[math.sin((sum(map(ord, t)) % 97 + 1) * k) for k in range(1, 9)] for t in input_ids]
        return F.normalize(torch.tensor(rows), dim=-1)

    def eval(self):
        return self


class _StubLoss:
    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, embeddings, hard_negatives=None, tiers=None, hard_masks=None):
        self.calls.append((embeddings, hard_negatives, tiers, hard_masks))
        return torch.tensor(0.0), {}

    def eval(self):
        return self


def _pairs(instance_ids: list[str]) -> list[EntailmentPair]:
    return [EntailmentPair(text_a=f"req {i}", type_a="REQ", text_b=f"hunk {i}", type_b="HUNK",
                           pair_type="req_hunk", instance_id=iid, tier=1)
            for i, iid in enumerate(instance_ids)]


def check_mask_construction() -> None:
    dev = torch.device("cpu")
    batch = {"req_hunk": _pairs(["A", "A", "B", "C"])}          # C has no Tier-3 hunk
    t3 = {"A": ["t3-A1", "t3-A2"], "B": ["t3-B1"]}

    loss = _StubLoss()
    train._train_step(_StubEncoder(), loss, batch, dev, tier3_by_instance=t3)
    _, hard, _, masks = loss.calls[-1]
    m = masks["req_hunk"]
    assert hard["req_hunk"].shape[0] == 3, "hard negatives must be de-duplicated across anchors of one instance"
    expected = torch.tensor([[1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.bool)
    assert torch.equal(m, expected), f"anchor -> own Tier-3 mask wrong:\n{m}"

    # same-repo negatives (M3): sampled per anchor, never shared with the other anchors
    pool = {"B": [f"repo-{k}" for k in range(6)]}
    loss = _StubLoss()
    train._train_step(_StubEncoder(), loss, batch, dev, tier3_by_instance=None, same_repo_hunks=pool,
                      rng=random.Random(0))
    _, hard, _, masks = loss.calls[-1]
    m = masks["req_hunk"]
    assert m[2].sum().item() == 3 and m[[0, 1, 3]].sum().item() == 0, f"same-repo sample leaked:\n{m}"

    # M2 + M3 together: own T3 plus own sample, still nothing shared
    loss = _StubLoss()
    train._train_step(_StubEncoder(), loss, batch, dev, tier3_by_instance=t3, same_repo_hunks=pool,
                      rng=random.Random(0))
    m = loss.calls[-1][3]["req_hunk"]
    assert m[0].sum().item() == 2 and m[2].sum().item() == 1 + 3 and m[3].sum().item() == 0

    # no hard negatives configured (M1): no mask, no hard negatives
    loss = _StubLoss()
    train._train_step(_StubEncoder(), loss, batch, dev)
    assert loss.calls[-1][1] is None and loss.calls[-1][3] is None


def check_validation_uses_negatives() -> None:
    dev = torch.device("cpu")
    batch = {"req_hunk": _pairs(["A", "B"])}
    t3 = {"A": ["t3-A1"], "B": ["t3-B1"]}

    loss = _StubLoss()
    train.validate(_StubEncoder(), loss, [batch], dev, tier3_by_instance=t3)
    assert loss.calls[0][1] is not None and loss.calls[0][3] is not None, "validation loss ignored Tier-3 negatives"

    loss = _StubLoss()
    train.validate(_StubEncoder(), loss, [batch], dev)          # M1: nothing configured
    assert loss.calls[0][1] is None

    # deterministic same-repo sampling in validation
    pool = {"A": [f"r{k}" for k in range(9)], "B": [f"s{k}" for k in range(9)]}
    seen = []
    for _ in range(2):
        loss = _StubLoss()
        train.validate(_StubEncoder(), loss, [batch], dev, same_repo_hunks=pool)
        seen.append(loss.calls[0][3]["req_hunk"].clone())
    assert torch.equal(seen[0], seen[1]), "validation loss is not deterministic"


def check_recipes() -> None:
    def flags(name: str) -> set[str]:
        text = (ROOT / "slurm" / "training" / name).read_text()
        return set(re.findall(r"^\s+(--[a-z0-9-]+)", text, re.MULTILINE))
    assert "--tier3" in flags("m2.slurm") and "--same-repo" not in flags("m2.slurm")
    assert {"--tier3", "--same-repo"} <= flags("m3.slurm"), "M3 must be M2 + same-repo negatives"
    f4 = flags("m4.slurm")
    assert {"--tier3", "--same-repo", "--tier-aware-loss"} <= f4, "M4 must be M3 + tier weighting"
    f1 = flags("m1.slurm")
    assert not ({"--tier3", "--same-repo", "--tier-aware-loss"} & f1), "M1 must have no hard negatives"


CHECKS = [check_masked_infonce, check_mask_construction, check_validation_uses_negatives, check_recipes]

if __name__ == "__main__":
    failed = 0
    for fn in CHECKS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}\n{traceback.format_exc()}")
    sys.exit(1 if failed else 0)
