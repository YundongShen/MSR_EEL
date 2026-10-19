"""PyTorch Dataset for Edit Entailment Learning.

Loads parsed instances and the retained hunks built by data_pipeline/build_dataset.py, and
emits flat EntailmentPair objects representing the four positive pair types:

  req_test  — (REQ, TEST)  requirement ↔ fail-to-pass test function
  req_hunk  — (REQ, HUNK)  requirement ↔ retained hunk (generated hunk that matches
                           the reference patch)
  orig_hunk — (ORIG, HUNK) original code unit ↔ hunk that edits it
  req_orig  — (REQ, ORIG)  requirement ↔ source unit in a file the retained hunks modify

Negatives are handled implicitly by InfoNCE in-batch sampling; no explicit
negative construction is needed here.  Tier labels (1 / 2) come from the retained-hunk
records and are carried in each pair for weighted evaluation.  Tier-3 hunks are not
pairs; train.py injects them as extra negatives.

An instance without a retained hunk contributes no hunk pairs.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from torch.utils.data import Dataset

from .instances import load_instances
from .retained_hunks import load_retained_hunks


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class EntailmentPair:
    """One positive pair sample for contrastive training."""

    text_a: str          # anchor text
    type_a: str          # entity type: REQ | TEST | ORIG | HUNK
    text_b: str          # positive text
    type_b: str          # entity type: REQ | TEST | ORIG | HUNK
    pair_type: str       # req_test | req_hunk | orig_hunk | req_orig
    instance_id: str
    tier: int = 0        # tier of the HUNK entity (1 or 2, 0 if no hunk)


# ---------------------------------------------------------------------------
# Hunk text rendering
# ---------------------------------------------------------------------------

def _render_hunk(hunk: dict) -> str:
    """Combine context + diff into a single string for embedding."""
    before = "\n".join(hunk.get("context_before") or [])
    diff   = hunk.get("hunk_diff", "")
    after  = "\n".join(hunk.get("context_after") or [])
    parts  = [p for p in (before, diff, after) if p.strip()]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Source-unit ↔ hunk matching
# ---------------------------------------------------------------------------

def _units_for_hunk(hunk: dict, source_units: list[dict]) -> list[dict]:
    """Return source units whose line range overlaps the hunk's old start line."""
    filepath = hunk.get("filepath", "")
    start    = hunk.get("old_start_line", 0)

    # Infer approximate end from hunk context line count
    hunk_lines = hunk.get("hunk_diff", "").splitlines()
    old_count  = sum(1 for l in hunk_lines if l.startswith(" ") or l.startswith("-"))
    end        = start + max(old_count - 1, 0)

    matched = []
    for unit in source_units:
        if unit.get("filepath") != filepath:
            continue
        u_start = unit.get("start_line", 0)
        u_end   = unit.get("end_line", 0)
        # Overlap: hunk range [start, end] ∩ unit range [u_start, u_end]
        if u_start <= end and u_end >= start:
            matched.append(unit)
    return matched


# ---------------------------------------------------------------------------
# Files touched by the retained hunks
# ---------------------------------------------------------------------------

def _patched_filepaths(hunks: list[dict]) -> set[str]:
    return {h.get("filepath", "") for h in hunks if h.get("filepath")}


# ---------------------------------------------------------------------------
# Main Dataset
# ---------------------------------------------------------------------------

class EntailmentDataset(Dataset):
    """Flat dataset of EntailmentPair objects built from parsed SWE-bench instances.

    Parameters
    ----------
    instances_path:
        Path to the instances JSONL produced by ``data_pipeline/parse_instances.py``.
    pair_types:
        Which of the four pair types to include.  Defaults to all four.
    max_req_chars:
        Truncate requirement text to this many characters before creating pairs
        (the tokenizer applies a second truncation at token level).
    seed:
        RNG seed used when shuffling / splitting.
    """

    ALL_PAIR_TYPES = ("req_test", "req_hunk", "orig_hunk", "req_orig")

    def __init__(
        self,
        instances_path: str | Path,
        llm_t12_path: str | Path,
        pair_types: tuple[str, ...] = ALL_PAIR_TYPES,
        max_req_chars: int = 2000,
        seed: int = 42,
    ) -> None:
        self._rng = random.Random(seed)
        self._pairs: list[EntailmentPair] = []

        # Retained (T1/T2) hunks keyed by instance_id, built by data_pipeline/build_dataset.py.  The loader
        # validates the file: LLM-generated hunks only, never the reference patch's own hunks.
        llm_t12_lookup = load_retained_hunks(llm_t12_path)

        instances = self._load_instances(Path(instances_path))
        for inst in instances:
            self._pairs.extend(
                self._build_pairs(inst, pair_types, max_req_chars,
                                  llm_t12_lookup.get(inst["instance_id"]))
            )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _load_instances(path: Path) -> list[dict]:
        return load_instances(path)   # drops the reference patch's own hunks

    # ------------------------------------------------------------------
    # Pair construction
    # ------------------------------------------------------------------

    def _build_pairs(
        self,
        inst: dict,
        pair_types: tuple[str, ...],
        max_req_chars: int,
        llm_t12_hunks: list[dict] | None = None,
    ) -> Iterator[EntailmentPair]:
        iid       = inst["instance_id"]
        req       = inst.get("requirement", "")[:max_req_chars]
        units     = inst.get("source_units", [])
        tests     = inst.get("test_functions", [])

        hunks   = llm_t12_hunks or []
        patched = _patched_filepaths(hunks)

        # ---- (REQ, TEST) -----------------------------------------------
        if "req_test" in pair_types:
            for tf in tests:
                code = tf.get("code", "").strip()
                if code:
                    yield EntailmentPair(
                        text_a=req, type_a="REQ",
                        text_b=code, type_b="TEST",
                        pair_type="req_test",
                        instance_id=iid, tier=0,
                    )

        # ---- (REQ, HUNK) and (ORIG, HUNK) ------------------------------
        for hunk in hunks:
            hunk_text = _render_hunk(hunk)
            if not hunk_text.strip():
                continue
            tier = hunk.get("tier_label") or 1  # default to 1 if labeler missed it

            if "req_hunk" in pair_types:
                yield EntailmentPair(
                    text_a=req, type_a="REQ",
                    text_b=hunk_text, type_b="HUNK",
                    pair_type="req_hunk",
                    instance_id=iid, tier=tier,
                )

            if "orig_hunk" in pair_types:
                for unit in _units_for_hunk(hunk, units):
                    code = unit.get("code", "").strip()
                    if code:
                        yield EntailmentPair(
                            text_a=code, type_a="ORIG",
                            text_b=hunk_text, type_b="HUNK",
                            pair_type="orig_hunk",
                            instance_id=iid, tier=tier,
                        )

        # ---- (REQ, ORIG) -----------------------------------------------
        if "req_orig" in pair_types:
            for unit in units:
                if unit.get("filepath") in patched:
                    code = unit.get("code", "").strip()
                    if code:
                        yield EntailmentPair(
                            text_a=req, type_a="REQ",
                            text_b=code, type_b="ORIG",
                            pair_type="req_orig",
                            instance_id=iid, tier=0,
                        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> EntailmentPair:
        return self._pairs[idx]

    # ------------------------------------------------------------------
    # Split helpers
    # ------------------------------------------------------------------

    def load_splits(
        self,
        splits_path: str | Path,
    ) -> tuple["EntailmentDataset", "EntailmentDataset", "EntailmentDataset"]:
        """Load fixed train/val/test splits from splits.json.

        All training and evaluation must use this instead of split_by_repo()
        to guarantee identical splits across all runs and ablation variants.
        """
        import json as _json
        splits = _json.load(open(splits_path))
        return (
            self._subset(set(splits["train_ids"])),
            self._subset(set(splits["val_ids"])),
            self._subset(set(splits["test_ids"])),
        )

    def _subset(self, instance_ids: set[str]) -> "EntailmentDataset":
        obj = object.__new__(EntailmentDataset)
        obj._rng = self._rng
        obj._pairs = [p for p in self._pairs if p.instance_id in instance_ids]
        return obj

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        from collections import Counter
        pt_counts = Counter(p.pair_type for p in self._pairs)
        tier_counts = Counter(
            p.tier for p in self._pairs if p.type_b == "HUNK" or p.type_a == "HUNK"
        )
        return {
            "total_pairs": len(self._pairs),
            "by_pair_type": dict(pt_counts),
            "by_tier": dict(tier_counts),
            "unique_instances": len({p.instance_id for p in self._pairs}),
            "instances_with_req_hunk": len(
                {p.instance_id for p in self._pairs if p.pair_type == "req_hunk"}
            ),
        }
