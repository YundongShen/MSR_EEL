# SWE-bench data preparation

Run `bash reproduce/setup_env.sh` from `Code/` to download SWE-bench Full
(`princeton-nlp/SWE-bench`, split `test`) and repository snapshots at their base
commits. The fixed issue splits select 2,291 instances.

Raw records are written to `Code/data/raw/swebench_full_instances.jsonl`;
parsed requirements, tests, and source units are written to
`Code/data/processed/instances_full.jsonl`.
