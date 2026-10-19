# SWE-bench instances

`swebench_full_instances.jsonl.gz` — 2,291 instances from the SWE-bench test split, one JSON object per line (gzip-compressed).

- Source: Hugging Face dataset `princeton-nlp/SWE-bench`, split `test` (2,294 instances), revision `e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6`, downloaded 2026-05-06.
- 2,291 of the 2,294 instances were processed successfully and are included here.
- Fields from SWE-bench: `instance_id`, `repo`, `base_commit`, `problem_statement`, `patch`, `test_patch`, `fail_to_pass`, `pass_to_pass`.
- Added fields: `source_files` and `test_files` (contents of the relevant source and test files at `base_commit`), `repo_path` (local clone location used during extraction).

Read with:

```python
import gzip, json
instances = [json.loads(line) for line in gzip.open("swebench_full_instances.jsonl.gz", "rt")]
```
