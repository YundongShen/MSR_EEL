"""One-time script to create fixed train/val/test splits.

Saves data/processed/splits.json with stable instance-level splits.
All training and evaluation scripts should read this file instead of
re-splitting at runtime.

Usage:
    python data_pipeline/make_splits.py \
        --input  data/processed/instances_full.jsonl \
        --output data/processed/splits.json \
        --val-frac 0.1 --test-frac 0.1 --seed 42
"""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     default="data/processed/instances_full.jsonl")
    parser.add_argument("--output",    default="data/processed/splits.json")
    parser.add_argument("--val-frac",  type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    ids = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                inst = json.loads(line)
                ids.append(inst["instance_id"])

    ids = sorted(set(ids))  # sorted for reproducibility regardless of PYTHONHASHSEED

    rng = random.Random(args.seed)
    rng.shuffle(ids)

    n = len(ids)
    n_val  = max(1, int(n * args.val_frac))
    n_test = max(1, int(n * args.test_frac))

    val_ids   = ids[:n_val]
    test_ids  = ids[n_val : n_val + n_test]
    train_ids = ids[n_val + n_test :]

    splits = {
        "train_ids": train_ids,
        "val_ids":   val_ids,
        "test_ids":  test_ids,
        "meta": {
            "seed":      args.seed,
            "val_frac":  args.val_frac,
            "test_frac": args.test_frac,
            "n_total":   n,
            "n_train":   len(train_ids),
            "n_val":     len(val_ids),
            "n_test":    len(test_ids),
        }
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Splits saved to {args.output}")
    print(f"  train: {len(train_ids)}  val: {len(val_ids)}  test: {len(test_ids)}")


if __name__ == "__main__":
    main()
