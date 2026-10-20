"""Candidate-file validation and checkpoint dataset metadata."""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import re
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

PROTOCOL_VERSION = 2
HUNK_SOURCE = "llm_generated_matched"

_REQUIRED_FIELDS = ("filepath", "old_start_line", "hunk_diff", "tier_label", "sample")
_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@(.*)$")


class CheckpointProvenanceError(RuntimeError):
    """The checkpoint metadata does not match the expected protocol."""


# ---------------------------------------------------------------------------
# Hunk identity and file validation
# ---------------------------------------------------------------------------

def hunk_key(hunk: dict) -> str:
    """Content fingerprint for joining hunk scores and analysis records."""
    raw = f"{hunk.get('filepath', '')}\x00{hunk.get('old_start_line', '')}\x00{hunk.get('hunk_diff', '')}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def validate_generated_hunks(
    by_instance: dict[str, list[dict]],
    allowed_tiers: tuple[int, ...],
    source: str,
) -> None:
    """Validate required hunk fields, tier labels, and diff headers."""
    for iid, hunks in by_instance.items():
        for h in hunks:
            missing = [k for k in _REQUIRED_FIELDS if k not in h]
            if missing:
                raise ValueError(
                    f"{source}: {iid} has a hunk without required fields {missing}. "
                    "Prepare the candidate file with data_pipeline/build_dataset.py."
                )
            if h["tier_label"] not in allowed_tiers:
                raise ValueError(f"{source}: {iid} has tier_label={h['tier_label']!r}, expected one of {allowed_tiers}")
            if not _HEADER.match(str(h["hunk_diff"]).split("\n", 1)[0]):
                raise ValueError(f"{source}: {iid} has a hunk_diff that is not a unified-diff hunk")


def file_digest(path: str | Path | None) -> str | None:
    """First 16 hex chars of the sha256 of a file, or None when there is no file."""
    if not path or not Path(path).exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def describe_hunk_files(retained_path: str | Path, tier3_path: str | Path | None) -> str:
    """Summarize candidate file paths and content digests."""
    parts = [f"retained={retained_path} sha256={file_digest(retained_path)}"]
    if tier3_path:
        parts.append(f"tier3={tier3_path} sha256={file_digest(tier3_path)}")
    return "Hunk files: " + "  ".join(parts)


# ---------------------------------------------------------------------------
# Checkpoint stamp
# ---------------------------------------------------------------------------

def _git_state() -> dict:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=30)
        return {"git_head": head.stdout.strip() or None, "git_dirty": bool(dirty.stdout.strip())}
    except Exception:
        return {"git_head": None, "git_dirty": None}


def build_stamp(
    retained_path: str | Path,
    tier3_path: str | Path | None,
    instances_path: str | Path,
    pair_types: tuple[str, ...],
    pair_counts: dict[str, int],
    same_repo: bool,
    tier_aware_loss: bool,
) -> dict:
    """Build the dataset and protocol metadata saved with a checkpoint."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "hunk_source": HUNK_SOURCE,
        "retained_file": str(retained_path),
        "retained_sha256": file_digest(retained_path),
        "tier3_file": str(tier3_path) if tier3_path else None,
        "tier3_sha256": file_digest(tier3_path),
        "instances_file": str(instances_path),
        "pair_types": list(pair_types),
        "pair_counts": dict(pair_counts),
        "same_repo": bool(same_repo),
        "tier_aware_loss": bool(tier_aware_loss),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        **_git_state(),
    }


def check_checkpoint(ckpt: dict, path: str | Path, allow_legacy: bool = False) -> dict | None:
    """Return the checkpoint's stamp, or raise ``CheckpointProvenanceError`` if it has none / a wrong one."""
    stamp = ckpt.get("data_provenance")
    problem = None
    if stamp is None:
        problem = "has no data_provenance stamp"
    elif stamp.get("hunk_source") != HUNK_SOURCE or stamp.get("protocol_version") != PROTOCOL_VERSION:
        problem = (f"declares hunk_source={stamp.get('hunk_source')!r}, "
                   f"protocol_version={stamp.get('protocol_version')!r}; expected "
                   f"{HUNK_SOURCE!r}, {PROTOCOL_VERSION}")
    if problem:
        if allow_legacy:
            log.warning("Checkpoint %s: %s", path, problem)
            return stamp
        raise CheckpointProvenanceError(
            f"Checkpoint {path} {problem}. Use a checkpoint with matching metadata, "
            "or pass --allow-legacy-checkpoint to bypass this metadata check."
        )
    log.info("Checkpoint %s: hunk_source=%s, retained=%s (sha256 %s), tier3=%s, pairs=%s, git=%s%s",
             path, stamp["hunk_source"], stamp["retained_file"], stamp["retained_sha256"],
             stamp["tier3_file"], stamp["pair_counts"], stamp.get("git_head"),
             "+dirty" if stamp.get("git_dirty") else "")
    return stamp


def load_checkpoint_checked(path: str | Path, map_location="cpu", allow_legacy: bool = False) -> dict:
    """Load a checkpoint and validate its dataset metadata."""
    import torch

    ckpt = torch.load(path, map_location=map_location)
    check_checkpoint(ckpt, path, allow_legacy=allow_legacy)
    return ckpt


def write_stamp_json(stamp: dict, directory: str | Path) -> None:
    """Human-readable copy next to the checkpoints (the stamp itself lives inside each .pt)."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    with open(Path(directory) / "train_provenance.json", "w") as fh:
        json.dump(stamp, fh, indent=2)
