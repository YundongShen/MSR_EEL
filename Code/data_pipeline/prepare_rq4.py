import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.hunk_matching import retained_tier


DATASETS = {
    "sympy_1cdf6f2": ("instances_sympy.jsonl.gz", "tier3_hunks.jsonl"),
    "github_prs_python": ("instances.jsonl.gz", "tier3_mix10.jsonl.gz"),
    "github_prs_non_python": ("instances.jsonl.gz", "tier3_gen.jsonl.gz"),
    "crossllm_gemini": ("instances.jsonl.gz", "tier3_gen.jsonl.gz"),
    "crossllm_qwen7b": ("instances.jsonl.gz", "tier3_mix20.jsonl.gz"),
    "crossllm_deepseek7b": ("instances.jsonl.gz", "tier3_mix20.jsonl.gz"),
    "swebench_multilingual": ("instances.jsonl.gz", "tier3_gen.jsonl.gz"),
}


def read_records(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def write_record(fh, record):
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare(source, output, instances_name, tier3_name):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "instances.jsonl").open("w", encoding="utf-8") as instances, \
            (output / "llm_t12_hunks.jsonl").open("w", encoding="utf-8") as retained:
        for record in read_records(source / instances_name):
            hunks = record.pop("gold_hunks")
            for hunk in hunks:
                if hunk.get("tier_label") is None:
                    hunk["tier_label"] = retained_tier(hunk["filepath"], record.get("fail_to_pass_ids", []))
                hunk.setdefault("sample", None)
            write_record(instances, record)
            write_record(retained, {"instance_id": record["instance_id"], "llm_t12_hunks": hunks})
    with (output / "tier3_hunks.jsonl").open("w", encoding="utf-8") as tier3:
        for record in read_records(source / tier3_name):
            for hunk in record["tier3_hunks"]:
                hunk.setdefault("sample", None)
            write_record(tier3, record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parents[2] / "Data/Evaluation/generalization/evaluation_inputs")
    parser.add_argument("--output-dir", type=Path, default=Path("data/cache/rq4"))
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    args = parser.parse_args()
    names = DATASETS if args.dataset == "all" else [args.dataset]
    for name in names:
        prepare(args.input_dir / name, args.output_dir / name, *DATASETS[name])


if __name__ == "__main__":
    main()
