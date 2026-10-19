import argparse
import gzip
import json
from pathlib import Path


def read(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def has_text(hunk):
    parts = [
        "\n".join(hunk.get("context_before") or []),
        hunk.get("hunk_diff") or "",
        "\n".join(hunk.get("context_after") or []),
    ]
    return any(part.strip() for part in parts)


def retained_hunks(record):
    for field in ("llm_t12_hunks", "gold_hunks"):
        if field in record:
            return record[field] or []
    raise ValueError(f"Missing retained hunks: {record['instance_id']}")


def eligible_ids(instances, tier3_records):
    tier3 = {}
    for record in tier3_records:
        iid = record["instance_id"]
        if iid in tier3:
            raise ValueError(f"Duplicate Tier-3 issue: {iid}")
        tier3[iid] = record["tier3_hunks"]

    selected = []
    seen = set()
    for record in instances:
        iid = record["instance_id"]
        if iid in seen:
            raise ValueError(f"Duplicate issue: {iid}")
        seen.add(iid)
        has_retained = any(
            hunk.get("tier_label") in (None, 1, 2) and has_text(hunk)
            for hunk in retained_hunks(record)
        )
        has_tier3 = any(has_text(hunk) for hunk in tier3.get(iid, []) or [])
        if has_retained and has_tier3:
            selected.append(iid)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instances")
    parser.add_argument("tier3")
    parser.add_argument("output")
    args = parser.parse_args()
    selected = eligible_ids(read(args.instances), read(args.tier3))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
