"""Score weights (alpha, beta, gamma) selected on the VALIDATION split.

Paper §III-D: the weights of  s(h) = α·sim(h,q) + β·sim(h,t̄) + γ·sim(h,o(h))  are selected on the
validation split and held fixed for every reported result.  ``evaluate.py --select-weights`` does the
selection and writes ``score_weights.json`` next to the checkpoint; every later evaluation of that
checkpoint reads it (explicit ``--score-*`` flags override it, for the single-view / leave-one-view-out ablations).

The file records which checkpoint it was selected for; using it with another checkpoint is an error.
"""

from __future__ import annotations

import json
from pathlib import Path

WEIGHTS_FILE = "score_weights.json"
CRITERION = "mean nDCG@k on the validation split (ties: PSR, then T2-Recall)"
# alpha is fixed to 1 (the ranking is invariant to a global scale); beta and gamma range over these values.
# All weights stay positive: EEL scores a hunk against all three repository views.
GRID_VALUES = (0.25, 0.5, 1.0, 2.0)


def weight_grid() -> list[tuple[float, float, float]]:
    return [(1.0, b, g) for b in GRID_VALUES for g in GRID_VALUES]


def checkpoint_identity(ckpt: dict) -> dict:
    """Identity of a checkpoint that survives moving/copying the file (not its path or mtime)."""
    stamp = ckpt.get("data_provenance") or {}
    return {"created": stamp.get("created"), "epoch": ckpt.get("epoch"), "val_loss": ckpt.get("val_loss")}


def select_best(results: list[dict]) -> dict:
    """Best triple by (nDCG@k, PSR, T2-Recall); a metric that is undefined counts as -1."""
    def key(r: dict) -> tuple:
        return tuple(-1.0 if r[m] is None else r[m] for m in ("ndcg_k", "psr", "t2_recall"))
    return max(results, key=key)       # max() keeps the first of equal keys: grid order is the tie-break


def weights_path(ckpt_path: str | Path) -> Path:
    return Path(ckpt_path).parent / WEIGHTS_FILE


def save_selected_weights(ckpt_path: str | Path, identity: dict, best: dict, results: list[dict], meta: dict) -> Path:
    path = weights_path(ckpt_path)
    with open(path, "w") as fh:
        json.dump({
            "alpha": best["alpha"], "beta": best["beta"], "gamma": best["gamma"],
            "criterion": CRITERION,
            "selected_on": "val",
            "checkpoint": str(ckpt_path),
            "checkpoint_identity": identity,
            "best": best,
            "grid_results": results,
            **meta,
        }, fh, indent=2)
    return path


def load_selected_weights(ckpt_path: str | Path) -> dict | None:
    path = weights_path(ckpt_path)
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)
