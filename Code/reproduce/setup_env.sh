#!/bin/bash
set -euo pipefail

CODE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
REPO_ROOT=$(cd -- "$CODE_ROOT/.." && pwd)
DATA_ROOT=${EEL_DATA_ROOT:-"$REPO_ROOT/Data"}
PYTHON_BIN=${EEL_PYTHON_BIN:-python3.12}
MODE=${1:-all}

if [[ $# -gt 1 || ! "$MODE" =~ ^(all|--data-only|--check)$ ]]; then
    echo "Usage: bash reproduce/setup_env.sh [--data-only|--check]" >&2
    exit 2
fi
command -v "$PYTHON_BIN" >/dev/null
cd "$CODE_ROOT"

"$PYTHON_BIN" - "$CODE_ROOT" "$DATA_ROOT" "$MODE" <<'PY'
import gzip
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11 or newer is required")
code, data, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
source = data / "eel_dataset"
names = ("splits.json", "eval_sets.json", "llm_t12_hunks.jsonl",
         "tier3_hunks.jsonl", "dataset_manifest.json", "instances_full.jsonl.gz")
for name in names:
    if not (source / name).is_file():
        raise SystemExit(f"Missing {source / name}; use Data from MSR_EEL_SUBMIT")

splits = json.loads((source / "splits.json").read_text())
split_ids = {s: set(splits[f"{s}_ids"]) for s in ("train", "val", "test")}
if any(len(splits[f"{s}_ids"]) != len(ids) for s, ids in split_ids.items()):
    raise SystemExit("Duplicate issue IDs in a split")
if tuple(len(split_ids[s]) for s in split_ids) != (1833, 229, 229):
    raise SystemExit("Unexpected train/val/test counts")
all_ids = set().union(*split_ids.values())
if len(all_ids) != 2291:
    raise SystemExit("Splits overlap or contain duplicate issue IDs")

def read_hunks(name, field, allowed):
    rows, counts = {}, Counter()
    with (source / name).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            iid, hunks = row["instance_id"], row[field]
            if iid in rows or iid not in all_ids:
                raise SystemExit(f"Duplicate or unknown issue in {name}: {iid}")
            for hunk in hunks:
                tier = hunk.get("tier_label")
                if tier not in allowed or not hunk.get("hunk_diff", "").strip():
                    raise SystemExit(f"Invalid hunk in {name}: {iid}")
                counts[tier] += 1
            rows[iid] = hunks
    return rows, counts

retained, rc = read_hunks("llm_t12_hunks.jsonl", "llm_t12_hunks", {1, 2})
tier3, tc = read_hunks("tier3_hunks.jsonl", "tier3_hunks", {3})
if (rc[1], rc[2], tc[3], sum(bool(h) for h in tier3.values())) != (3354, 5869, 1257, 743):
    raise SystemExit("Candidate counts do not match the published dataset")
eligible = {i for i in split_ids["test"] if retained.get(i) and tier3.get(i)}
t2 = {i for i in eligible if any(h["tier_label"] == 2 for h in retained[i])}
evaluation = json.loads((source / "eval_sets.json").read_text())["test"]
if (len(eligible), len(t2)) != (82, 47):
    raise SystemExit("Unexpected evaluation issue counts")
if eligible != set(evaluation["eval_instances"]) or t2 != set(evaluation["t2_recall_instances"]):
    raise SystemExit("Evaluation IDs do not match the candidates and fixed splits")
if sum(i.startswith("django__") and bool(h) and bool(retained.get(i)) for i, h in tier3.items()) != 204:
    raise SystemExit("Unexpected eligible Django issue count")

seen, django = set(), 0
with gzip.open(source / "instances_full.jsonl.gz", "rt") as handle:
    for line in handle:
        if not line.strip():
            continue
        iid = json.loads(line)["instance_id"]
        if iid in seen:
            raise SystemExit(f"Duplicate parsed issue: {iid}")
        seen.add(iid)
        django += iid.startswith("django__")
if seen != all_ids or django != 849:
    raise SystemExit("Parsed instances do not match the splits and repository counts")
print("Data checked: 2291 issues; T1=3354, T2=5869, T3=1257; evaluation=82, T2-recall=47")
if mode == "--check":
    raise SystemExit(0)

for directory in ("data/raw", "data/processed", "data/cache", "logs", "checkpoints"):
    (code / directory).mkdir(parents=True, exist_ok=True)
processed, cache = code / "data/processed", code / "data/cache"
for name in ("splits.json", "eval_sets.json"):
    shutil.copyfile(source / name, processed / name)
for name in ("llm_t12_hunks.jsonl", "tier3_hunks.jsonl", "dataset_manifest.json"):
    shutil.copyfile(source / name, cache / name)
generations = source / "haiku_generations.jsonl.gz"
if generations.is_file():
    shutil.copyfile(generations, cache / generations.name)
raw = data / "swebench/swebench_full_instances.jsonl.gz"
if raw.is_file():
    shutil.copyfile(raw, code / "data/raw" / raw.name)
with gzip.open(source / "instances_full.jsonl.gz", "rb") as src, \
     (processed / "instances_full.jsonl").open("wb") as full, \
     (processed / "instances_django_only.jsonl").open("wb") as dj, \
     (processed / "instances_non_django.jsonl").open("wb") as other, \
     (processed / "instances_sympy_only.jsonl").open("wb") as sympy, \
     (processed / "instances_non_sympy.jsonl").open("wb") as non_sympy:
    for line in src:
        full.write(line)
        if line.strip():
            iid = json.loads(line)["instance_id"]
            (dj if iid.startswith("django__") else other).write(line)
            (sympy if iid.startswith("sympy__") else non_sympy).write(line)
print("Runtime data prepared from", source)
PY

if [[ "$MODE" != all ]]; then
    exit 0
fi
"$PYTHON_BIN" -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url "${EEL_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
python -m pip install -r reproduce/requirements.txt
python -m pip check
