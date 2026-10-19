"""Build the candidate dataset (retained + non-retained hunks) from raw LLM generations.

Input : raw generations, one record per SWE-bench issue (data_pipeline/generate_candidates.py),
        the parsed instances (fail-to-pass ids), source snapshots and the fixed split.
Output:
  data/cache/llm_t12_hunks.jsonl    generated hunks that match the reference patch (Tier 1/2)
    {"instance_id", "llm_t12_hunks": [{hunk_id, filepath, old_start_line, hunk_diff,
                                        context_before, context_after, tier_label, sample}]}
  data/cache/tier3_hunks.jsonl      generated hunks that do not match it (Tier 3)
    {"instance_id", "tier3_hunks":   [{... same fields, tier_label = 3}]}
  data/cache/dataset_manifest.json  protocol, repository counts and evaluation-set sizes

Every selected generated hunk goes to exactly one of the two files, so retained and non-retained
candidates share one generation process and one diff renderer.  Several generation files
(independent samples of the same model) may be given; hunks are united per instance and
de-duplicated on (file, start line, edit content).
Compact merged_hunks records and gzip-compressed JSONL inputs are also supported.

Usage:
    python data_pipeline/build_dataset.py
    python data_pipeline/build_dataset.py --generations data/cache/haiku_generations.jsonl \\
                                                  data/cache/haiku_generations_s1.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.hunk_matching import (
    MATCH_TOLERANCE,
    extract_context,
    matches_reference,
    parse_hunks,
    retained_tier,
)
from data_pipeline.merged_generations import candidate_hunks, export_hunk_key, hunk_identity, read_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def ordered_hunks(hunks: list[dict], keys: list[str], iid: str) -> list[dict]:
    lookup = {export_hunk_key(h): h for h in hunks}
    if len(lookup) != len(hunks) or len(set(keys)) != len(keys) or set(keys) != lookup.keys():
        raise ValueError(f"{iid}: export_order does not match generated candidates")
    return [lookup[key] for key in keys]


def dataset_statistics(order, references, retained, tier3, split_of, repos):
    fields = ("instances", "train", "val", "test", "gold_hunks",
              "T1", "T2", "T3_hunks", "T3_instances")
    per_repo = {}
    for iid in order:
        name = repos[iid].rsplit("/", 1)[-1]
        counts = per_repo.setdefault(name, dict.fromkeys(fields, 0))
        counts["instances"] += 1
        counts[split_of[iid]] += 1
        counts["gold_hunks"] += references[iid]
        counts["T1"] += sum(h["tier_label"] == 1 for h in retained.get(iid, []))
        counts["T2"] += sum(h["tier_label"] == 2 for h in retained.get(iid, []))
        counts["T3_hunks"] += len(tier3.get(iid, []))
        counts["T3_instances"] += bool(tier3.get(iid))
    per_repo = dict(sorted(per_repo.items(), key=lambda item: (item[1]["instances"], item[0])))
    total = {field: sum(counts[field] for counts in per_repo.values()) for field in fields}
    eligible = [iid for iid in order if split_of[iid] == "test"
                and retained.get(iid) and tier3.get(iid)]
    evaluation = {
        "test_eval_instances": len(eligible),
        "test_t2_recall_instances": sum(any(h["tier_label"] == 2 for h in retained[iid]) for iid in eligible),
    }
    return total, per_repo, evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations", nargs="+", default=["data/cache/haiku_generations.jsonl"],
                        help="Raw generation JSONL file(s); several = independent samples")
    parser.add_argument("--instances", default="data/processed/instances_full.jsonl")
    parser.add_argument("--source-instances", type=Path,
                        default=Path("data/raw/swebench_full_instances.jsonl.gz"))
    parser.add_argument("--splits", default="data/processed/splits.json")
    parser.add_argument("--retained-out", default="data/cache/llm_t12_hunks.jsonl")
    parser.add_argument("--tier3-out", default="data/cache/tier3_hunks.jsonl")
    parser.add_argument("--manifest", default="data/cache/dataset_manifest.json")
    parser.add_argument("--tolerance", type=int, default=MATCH_TOLERANCE,
                        help="Max line distance between a generated and a reference hunk start")
    args = parser.parse_args()

    with open(args.splits) as fh:
        sp = json.load(fh)
    split_of: dict[str, str] = {}
    for key, name in (("train_ids", "train"), ("val_ids", "val"), ("test_ids", "test")):
        for iid in sp[key]:
            split_of[iid] = name

    log.info("Loading fail-to-pass ids from %s", args.instances)
    f2p: dict[str, list[str]] = {}
    for rec in read_jsonl(args.instances):
        f2p[rec["instance_id"]] = rec.get("fail_to_pass_ids", [])

    sources = {}
    if args.source_instances.is_file():
        for rec in read_jsonl(args.source_instances):
            sources[rec["instance_id"]] = rec.get("source_files", {})

    retained: dict[str, list[dict]] = defaultdict(list)
    tier3: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple] = set()
    order: dict[str, None] = {}
    protocol: dict = {}
    outcome: Counter = Counter()
    n_generations = 0
    references, repos, layouts = {}, {}, {}

    for sample, path in enumerate(args.generations):
        log.info("Reading generations [%d] %s", sample, path)
        for rec in read_jsonl(path):
            iid = rec["instance_id"]
            gen = rec.get("generation") or {}
            n_generations += 1
            if not protocol:
                protocol = {
                    "model": gen.get("model"),
                    "provider": gen.get("provider"),
                    "temperature": gen.get("temperature"),
                    "max_tokens": gen.get("max_tokens"),
                    "system_prompt_sha256": hashlib.sha256(
                        (gen.get("system_prompt") or "").encode()).hexdigest()[:16],
                }
            order.setdefault(iid, None)
            if iid not in split_of:
                raise ValueError(f"{iid}: issue is absent from the splits")
            repos[iid] = rec.get("repo", iid.split("__", 1)[0])
            if "export_order" in rec:
                layouts[iid] = rec["export_order"]

            reference_hunks = parse_hunks(rec["meta"]["gold_patch"] or "")
            references[iid] = len(reference_hunks)
            generated_hunks = candidate_hunks(rec, default_sample=sample)
            if not reference_hunks or not generated_hunks:
                outcome["no reference hunks or no generated hunks"] += 1
                continue
            outcome["usable generation"] += 1

            source_files = rec.get("source_files") or sources.get(iid)
            if source_files is None:
                raise ValueError(f"{iid}: source_files missing; provide --source-instances")
            for h in generated_hunks:
                diff = h["text"]
                key = (iid, *hunk_identity(h))
                if key in seen:
                    continue
                seen.add(key)
                before, after = extract_context(source_files, h["file"], h["old_start"], diff.splitlines(keepends=True))
                out = {
                    "filepath": h["file"],
                    "old_start_line": h["old_start"],
                    "hunk_diff": diff,
                    "context_before": before,
                    "context_after": after,
                    "sample": h["sample"],
                }
                if matches_reference(h["file"], h["old_start"], reference_hunks, args.tolerance):
                    out["tier_label"] = retained_tier(h["file"], f2p.get(iid, []))
                    retained[iid].append(out)
                else:
                    out["tier_label"] = 3
                    tier3[iid].append(out)

    for iid, layout in layouts.items():
        for name, table in (("retained", retained), ("tier3", tier3)):
            hunks = ordered_hunks(table.get(iid, []), layout[name], iid)
            if hunks:
                table[iid] = hunks
    for iid, hunks in retained.items():
        id_last = set(layouts.get(iid, {}).get("retained_id_last", []))
        hunks[:] = [{**hunk, "hunk_id": index} if index in id_last
                    else {"hunk_id": index, **hunk} for index, hunk in enumerate(hunks)]

    for path, name, table in ((args.retained_out, "llm_t12_hunks", retained),
                              (args.tier3_out, "tier3_hunks", tier3)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            category = "retained" if name == "llm_t12_hunks" else "tier3"
            indices = {iid: i for i, iid in enumerate(order)}
            output_order = sorted(order, key=lambda iid: layouts.get(iid, {}).get(category + "_issue", indices[iid]))
            for iid in output_order:
                if iid in table:
                    fh.write(json.dumps({"instance_id": iid, name: table[iid]}, ensure_ascii=False) + "\n")
        log.info("Wrote %d instances → %s", len(table), path)

    # ---- integrity: every hunk is in exactly one tier file ---------------------------------
    for iid in retained.keys() & tier3.keys():
        keys_r = {(h["filepath"], h["old_start_line"], h["hunk_diff"]) for h in retained[iid]}
        keys_t = {(h["filepath"], h["old_start_line"], h["hunk_diff"]) for h in tier3[iid]}
        assert not keys_r & keys_t, f"{iid}: a hunk is both retained and non-retained"

    # ---- manifest ------------------------------------------------------------------------
    per_split: dict[str, Counter] = defaultdict(Counter)
    for iid in order:
        s = split_of.get(iid, "?")
        c = per_split[s]
        r, t = retained.get(iid, []), tier3.get(iid, [])
        c["instances"] += 1
        c["with_retained"] += bool(r)
        c["with_tier3"] += bool(t)
        c["eligible (retained and tier3)"] += bool(r and t)
        c["eligible with a Tier-2 hunk"] += bool(r and t and any(h["tier_label"] == 2 for h in r))
        c["tier1_hunks"] += sum(h["tier_label"] == 1 for h in r)
        c["tier2_hunks"] += sum(h["tier_label"] == 2 for h in r)
        c["tier3_hunks"] += len(t)

    total, per_repo, evaluation = dataset_statistics(order, references, retained, tier3, split_of, repos)
    generator = "Haiku" if "haiku" in (protocol.get("model") or "").lower() else "LLM"
    manifest = {
        "protocol": "T1/T2 = LLM-generated hunks matched to reference-patch hunks "
                    f"(same file, start line within {args.tolerance} lines of the gold hunk), "
                    "T1 if the file is statically linked to a fail-to-pass test, else T2; "
                    f"T3 = {generator}-generated hunks starting >= {args.tolerance} lines "
                    "from every gold hunk in the same file (or in another file)",
        "generator": f"{protocol.get('model')}, temperature {protocol.get('temperature')}, "
                     f"max_tokens {protocol.get('max_tokens')}",
        "total": total,
        "per_repo": per_repo,
        "eval": evaluation,
    }
    with open(args.manifest, "w") as fh:
        json.dump(manifest, fh, indent=2)
    log.info("Manifest → %s", args.manifest)
    for s, c in sorted(per_split.items()):
        log.info("  %-5s %s", s, dict(c))


if __name__ == "__main__":
    main()
