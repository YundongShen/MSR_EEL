"""Recompute macro-averaged ranking metrics from saved per-issue rankings."""

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parents[2] / "Data/Evaluation"
GAINS = {0: 0, 1: 3, 2: 2, 3: 0}
METRICS = ("ndcg_k", "t1_recall", "t2_recall", "psr")


def instance_metrics(ranked):
    tiers = [entry[1] for entry in ranked]
    retained = [i for i, tier in enumerate(tiers) if tier in (1, 2)]
    tier3 = [i for i, tier in enumerate(tiers) if tier == 3]
    if not retained or not tier3:
        return None
    k = len(retained)
    gains = [GAINS[tier] for tier in tiers]
    dcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(gains[:k]))
    ideal = sum(gain / math.log2(i + 2)
                for i, gain in enumerate(sorted(gains, reverse=True)[:k]))
    return {
        "ndcg_k": dcg / ideal,
        "t1_recall": tiers[:k].count(1) / tiers.count(1) if 1 in tiers else None,
        "t2_recall": tiers[:k].count(2) / tiers.count(2) if 2 in tiers else None,
        "psr": float(max(retained) < min(tier3)),
    }


def summarize(path):
    values = {name: [] for name in METRICS}
    for record in json.loads(path.read_text(encoding="utf-8")):
        result = instance_metrics(record["ranked"])
        if result is not None:
            for name, value in result.items():
                if value is not None:
                    values[name].append(value)
    label = path.relative_to(DATA_ROOT) if path.is_relative_to(DATA_ROOT) else path.name
    return {
        "file": str(label),
        **{name: statistics.fmean(items) if items else None for name, items in values.items()},
        "n_instances": len(values["ndcg_k"]),
        "n_t1": len(values["t1_recall"]),
        "n_t2": len(values["t2_recall"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[DATA_ROOT],
                        help="Per-issue JSON files or directories containing them")
    parser.add_argument("--output", type=Path, help="Save full-precision results as JSON")
    args = parser.parse_args()
    files = sorted({file.resolve() for path in args.paths
                    for file in (path.rglob("per_issue*.json") if path.is_dir() else [path])})
    if not files:
        parser.error("No per-issue result files found")
    results = [summarize(path) for path in files]
    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    writer.writerow(("File", "nDCG@k", "T1-Recall", "T2-Recall", "PSR", "Issues", "T1 issues", "T2 issues"))
    for result in results:
        writer.writerow([result["file"],
                         *(f"{result[name]:.3f}" if result[name] is not None else "--" for name in METRICS),
                         result["n_instances"], result["n_t1"], result["n_t2"]])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
