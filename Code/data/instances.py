"""Load parsed issue, source, and test inputs."""

from __future__ import annotations

import json
from pathlib import Path


def load_instances(path: str | Path, ids: set[str] | None = None) -> list[dict]:
    """Read instance inputs, optionally restricted to ``ids``."""
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
