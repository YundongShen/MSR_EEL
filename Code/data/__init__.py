"""Data pipeline for Edit Entailment Learning."""

from .data_loader import DataLoader, DataSample, GitHubDataLoader
from .tier_labeler import tier_of_file
from .utils import hash_signature, normalize_diff, tokenize_diff_hunks

def __getattr__(name):
    if name in ("EntailmentDataset", "EntailmentPair"):
        from . import entailment_dataset
        return getattr(entailment_dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # SWE-bench loading (kept for LLM pair generation pipeline)
    "DataLoader",
    "DataSample",
    "GitHubDataLoader",
    # New: Edit Entailment
    "EntailmentDataset",
    "EntailmentPair",
    "tier_of_file",
    # Utilities
    "hash_signature",
    "normalize_diff",
    "tokenize_diff_hunks",
]
