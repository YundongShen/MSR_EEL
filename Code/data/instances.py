"""Load parsed instances without the reference patch's own hunks.

``instances_*.jsonl`` files written by older versions of data_pipeline/parse_instances.py carry a
``gold_hunks`` field: the hunks of the SWE-bench reference patch itself.  No training,
evaluation or analysis step may see it -- candidates are LLM-generated hunks only
(data/retained_hunks.py).  Every reader of the instances file therefore goes through
``load_instances``, which drops the field, so reference-patch hunks cannot reach a model even
if an old instances file is used.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_instances(path: str | Path, ids: set[str] | None = None) -> list[dict]:
    """Read an instances JSONL; keep only ``ids`` when given; never return ``gold_hunks``."""
    instances: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            inst = json.loads(line)
            inst.pop("gold_hunks", None)
            if ids is None or inst.get("instance_id") in ids:
                instances.append(inst)
    return instances


def load_split_ids(splits_path: str | Path, split: str) -> set[str]:
    """Instance ids of ``split`` (train | val | test) from the fixed splits file."""
    with open(splits_path) as fh:
        return set(json.load(fh)[f"{split}_ids"])
