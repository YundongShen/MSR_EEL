"""Candidate hunks of an instance: retained (Tier 1/2) and non-retained (Tier 3).

Retained-H are the LLM-generated hunks that match the reference patch, read from
``cfg.data.llm_t12_path`` (built by data_pipeline/build_dataset.py); Tier-3 hunks come from the
companion ``tier3_hunks.jsonl``.  An instance without a retained hunk has none.  Retained,
Tier-3 and distractor candidates all come from the same generation process and the same diff
renderer.  The reference patch's own hunks are never a candidate: both loaders validate the
file (data.provenance.validate_generated_hunks) and refuse anything that does not look like
build_dataset.py output.

Every hunk gets ``hunk_id`` (index in its instance list) and ``hunk_key`` (content fingerprint).
Join score files and labels on ``hunk_key``: the index changes whenever the dataset is rebuilt.
"""

from __future__ import annotations

import json
from pathlib import Path

from .provenance import hunk_key, validate_generated_hunks

_cache: dict[tuple[str, str], dict[str, list[dict]]] = {}


def _load(path: str | Path, field: str, allowed_tiers: tuple[int, ...]) -> dict[str, list[dict]]:
    key = (str(path), field)
    if key not in _cache:
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found. Build it with data_pipeline/build_dataset.py.")
        lookup: dict[str, list[dict]] = {}
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if field not in rec:
                    raise ValueError(f"{path}: record for {rec.get('instance_id')} has no '{field}' field")
                lookup[rec["instance_id"]] = rec[field]
        validate_generated_hunks(lookup, allowed_tiers, str(path))
        for hunks in lookup.values():
            for i, hunk in enumerate(hunks):
                hunk.setdefault("hunk_id", i)
                hunk.setdefault("hunk_key", hunk_key(hunk))
        _cache[key] = lookup
    return _cache[key]


def load_retained_hunks(path: str | Path) -> dict[str, list[dict]]:
    """Load llm_t12_hunks.jsonl → {instance_id: [hunk_dict, ...]} (validated, cached per path)."""
    return _load(path, "llm_t12_hunks", (1, 2))


def load_tier3_hunks(path: str | Path) -> dict[str, list[dict]]:
    """Load tier3_hunks.jsonl → {instance_id: [hunk_dict, ...]} (validated, cached per path)."""
    return _load(path, "tier3_hunks", (3,))


def retained_hunks(inst: dict, cfg) -> list[dict]:
    """Tier-1/2 hunks of ``inst``; each has ``tier_label``."""
    return load_retained_hunks(cfg.data.llm_t12_path).get(inst.get("instance_id", ""), [])


def add_retained_args(parser) -> None:
    parser.add_argument("--llm-t12", default=None, dest="llm_t12",
                        help="Path to llm_t12_hunks.jsonl (Retained-H, T1/T2; default: config)")


def apply_retained_args(cfg, args) -> None:
    if args.llm_t12:
        cfg.data.llm_t12_path = args.llm_t12
