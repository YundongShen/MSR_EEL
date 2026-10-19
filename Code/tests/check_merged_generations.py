"""Offline checks for compact sampling and dataset construction."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
package = types.ModuleType("data")
package.__path__ = [str(ROOT / "data")]
sys.modules["data"] = package

from data_pipeline.merged_generations import candidate_hunks, diff_hunks, export_hunk_key, gold_distance, read_jsonl
from data_pipeline import sample_haiku_merge as sampler
from data_pipeline import build_dataset
from data.llm_client import LLMClient
from data_pipeline.generate_candidates import generate, _PATCH_SYSTEM
from data.provenance import validate_generated_hunks


def diff(start=1, value="fixed", new_start=None, file="a.py"):
    return f"--- a/{file}\n+++ b/{file}\n@@ -{start},1 +{new_start or start},1 @@\n-line{start - 1}\n+{value}\n"


def inst():
    return {
        "instance_id": "owner__repo-1", "repo": "owner/repo", "base_commit": "base",
        "patch": diff(), "problem_statement": "Fix line0",
        "source_files": {"a.py": "".join(f"line{i}\n" for i in range(60))},
    }


def result(patch_text, raw="first response"):
    return {"extracted_diff": patch_text, "raw_response": raw,
            "modified_files": {"a.py": "not retained"}}


def merge(record, data, generation, sample=0, run_id="round"):
    return sampler.merge_sample(record, data, generation, sample=sample, run_id=run_id,
        model="claude-haiku-4-5-20251001", temperature=0.7, max_tokens=4096,
        system_prompt="system", user_prompt="user")


class SamplingChecks(unittest.TestCase):
    def test_generated_patch_headers_are_preserved(self):
        for suffix in ("", " def example():"):
            with self.subTest(suffix=suffix):
                text = diff().replace(" @@\n", f" @@{suffix}\n") + diff(30, file="b.py")
                llm = types.SimpleNamespace(complete=lambda system, user: text)
                generated = generate(inst(), llm)
                self.assertEqual(generated, {"raw_response": text, "extracted_diff": text})
                record, count = merge(None, inst(), generated)
                self.assertEqual(count, 2)
                self.assertEqual([h["text"] for h in candidate_hunks(record)],
                                 [h["text"] for h in diff_hunks(text)])
                self.assertNotIn("sr_blocks_found", record["sampling"]["runs"][0])
                self.assertEqual(record["generation"]["raw_response"], text)

    def test_generated_patch_fences_and_api_failure(self):
        text = diff()
        for raw in (f"```diff\n{text}```", f"```patch\n{text}```", f"```\n{text}```"):
            generated = generate(inst(), types.SimpleNamespace(complete=lambda system, user: raw))
            self.assertEqual(generated["raw_response"], raw)
            self.assertEqual(generated["extracted_diff"], text)
        self.assertIsNone(generate(inst(), types.SimpleNamespace(complete=lambda system, user: None)))

    def test_non_patch_response_stops_sampling(self):
        with tempfile.TemporaryDirectory() as temp:
            data = inst()
            input_path, output = Path(temp) / "input.jsonl", Path(temp) / "generations.jsonl"
            sampler.save_records(input_path, {data["instance_id"]: data})
            args = ["sample_haiku_merge.py", "--input", str(input_path), "--generations", str(output),
                    "--run-id", "non-patch", "--sleep", "0"]
            with patch.object(LLMClient, "complete", return_value="Here is the updated code.") as complete, \
                    patch.dict(sampler.os.environ, {"ANTHROPIC_API_KEY": "test"}), \
                    patch.object(sys, "argv", args):
                with self.assertRaisesRegex(RuntimeError, "generation failed"):
                    sampler.main()
            complete.assert_called_once()

    def test_native_diff_validation(self):
        for suffix in ["", " def f():"]:
            hunk = {"filepath": "a.py", "old_start_line": 1, "sample": 0,
                    "hunk_diff": f"@@ -1,1 +1,1 @@{suffix}\n-x\n+y\n", "tier_label": 1}
            validate_generated_hunks({"i": [hunk]}, (1, 2), "generated")
            for invalid in [{k: v for k, v in hunk.items() if k != "sample"},
                            {**hunk, "tier_label": 3}, {**hunk, "hunk_diff": "@@ invalid\n-x\n+y\n"}]:
                with self.assertRaises(ValueError):
                    validate_generated_hunks({"i": [invalid]}, (1, 2), "invalid")

    def test_quota_keeps_closest_matches_and_unmatched_hunks(self):
        record = {"meta": {"gold_patch": diff(10)}, "generation": {
            "extracted_diff": diff(3) + diff(9) + diff(10) + diff(40)}}
        hunks = candidate_hunks(record)
        self.assertEqual([h["old_start"] for h in hunks], [10, 40])
        record["merged_hunks"] = hunks
        self.assertEqual([h["old_start"] for h in candidate_hunks(record)], [10, 40])

    def test_sampling_continues_until_quota_at_batch_end(self):
        with tempfile.TemporaryDirectory() as temp:
            data = inst()
            data["patch"] = diff(10)
            input_path = Path(temp) / "input.jsonl"
            output = Path(temp) / "generations.jsonl"
            sampler.save_records(input_path, {data["instance_id"]: data})
            responses = [
                diff(40, "extra"),
                diff(40, "extra"),
                diff(10),
                diff(14, "near"),
            ]
            requests = []

            def create(**kwargs):
                requests.append(kwargs)
                return types.SimpleNamespace(content=[types.SimpleNamespace(text=responses.pop(0))])

            fake = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
            args = ["sample_haiku_merge.py", "--input", str(input_path), "--generations", str(output),
                    "--samples", "2", "--run-id", "quota", "--sleep", "0"]
            with patch.object(LLMClient, "_ensure_client", lambda self: setattr(self, "_client", fake)), \
                    patch.dict(sampler.os.environ, {"ANTHROPIC_API_KEY": "test"}), patch.object(sys, "argv", args):
                sampler.main()
                sampler.main()
            self.assertEqual(len(requests), 4)
            record = sampler.load_records(output)[data["instance_id"]]
            hunks = candidate_hunks(record)
            reference = diff_hunks(data["patch"])
            self.assertEqual(sum(gold_distance(h, reference) < 20 for h in hunks), 1)
            self.assertEqual(sum(gold_distance(h, reference) >= 20 for h in hunks), 1)
            self.assertEqual(len(record["sampling"]["runs"]), 4)

    def test_dedup_preserves_first_response_and_sample_tags(self):
        data = inst()
        record, count = merge(None, data, result(diff()))
        self.assertEqual(count, 1)
        original = copy.deepcopy(record["generation"])
        record["export_order"] = {"retained": ["old"]}
        record, count = merge(record, data, result(diff(new_start=4) + diff(30), "later response"), 1)
        self.assertEqual(count, 1)
        self.assertEqual(record["generation"], original)
        self.assertNotIn("later response", json.dumps(record))
        self.assertNotIn("modified_files", record["generation"])
        self.assertNotIn("export_order", record)
        self.assertEqual([h["sample"] for h in candidate_hunks(record)], [0, 1])
        self.assertEqual(sampler.next_sample(record), 2)
        record, count = merge(record, data, result(diff(30), "duplicate"), 2)
        self.assertEqual(count, 0)
        self.assertEqual(len(record["merged_hunks"]), 2)
        self.assertEqual(sampler.next_sample(record), 3)

    def test_legacy_merged_hunks_preserve_raw_negatives(self):
        record = {"generation": {"extracted_diff": diff(30)},
                  "merged_hunks": [{"file": "a.py", "text": diff_hunks(diff())[0]["text"]}]}
        hunks = candidate_hunks(record)
        self.assertEqual([(h["old_start"], h["sample"]) for h in hunks], [(30, 0), (1, 1)])
        self.assertEqual(sampler.next_sample(record), 2)

    def test_file_boundaries_and_new_files(self):
        text = diff(file="a.py") + "diff --git a/b.py b/b.py\nindex 1..2 100644\n" + diff(file="b.py")
        hunks = diff_hunks(text)
        self.assertEqual([h["file"] for h in hunks], ["a.py", "b.py"])
        self.assertNotIn("diff --git", hunks[0]["text"])
        added = diff_hunks("--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,1 @@\n+hello\n")
        self.assertEqual((added[0]["file"], added[0]["old_len"]), ("new.py", 0))
        header_like = diff_hunks("--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n--- old divider\n+++ new divider\n")
        self.assertEqual(header_like[0]["file"], "a.py")
        self.assertIn("--- old divider", header_like[0]["text"])
        with self.assertRaises(ValueError):
            diff_hunks("--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,2 @@\n-a\n+b\n")

    def test_plain_gzip_and_atomic_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            for suffix in (".jsonl", ".jsonl.gz"):
                path = Path(temp) / ("records" + suffix)
                records = {"i": {"instance_id": "i", "value": "文本"}}
                sampler.save_records(path, records)
                self.assertEqual(sampler.load_records(path), records)
                before = path.read_bytes()
                with patch.object(sampler.os, "replace", side_effect=OSError("interrupted")):
                    with self.assertRaises(OSError):
                        sampler.save_records(path, {"i": {"instance_id": "i", "value": "new"}})
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(sorted(p.name for p in Path(temp).iterdir()),
                    sorted(["records.jsonl"] if suffix == ".jsonl" else ["records.jsonl", "records.jsonl.gz"]))

    def test_changed_reference_is_rejected(self):
        data = inst()
        record, _ = merge(None, data, result(diff()))
        data["patch"] = diff(value="different reference")
        with self.assertRaises(ValueError):
            merge(record, data, result(diff(30)), 1)

    def test_sampler_api_resume_and_dataset_build(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            data = inst()
            input_path, output = folder / "input.jsonl.gz", folder / "generations.jsonl.gz"
            sampler.save_records(input_path, {data["instance_id"]: data})
            requests = []
            responses = [
                diff(),
                diff(30, "extra"),
            ]

            def create(**kwargs):
                requests.append(kwargs)
                return types.SimpleNamespace(content=[types.SimpleNamespace(text=responses.pop(0))])

            fake = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
            args = ["sample_haiku_merge.py", "--input", str(input_path), "--generations", str(output),
                    "--samples", "2", "--run-id", "round-1", "--sleep", "0"]
            with patch.object(LLMClient, "_ensure_client", lambda self: setattr(self, "_client", fake)), \
                    patch.dict(sampler.os.environ, {"ANTHROPIC_API_KEY": "test"}), patch.object(sys, "argv", args):
                sampler.main()
                sampler.main()
            self.assertEqual(len(requests), 2)
            self.assertTrue(all(r["temperature"] == 0.7 and r["max_tokens"] == 4096 for r in requests))
            record = sampler.load_records(output)[data["instance_id"]]
            self.assertEqual([h["sample"] for h in candidate_hunks(record)], [0, 1])
            self.assertEqual(record["generation"]["raw_response"], diff())
            self.assertEqual(record["generation"]["system_prompt"], _PATCH_SYSTEM)
            self.assertTrue(all(r["system"] == _PATCH_SYSTEM for r in requests))
            self.assertEqual(len(record["sampling"]["runs"]), 2)

            splits = folder / "splits.json"
            splits.write_text(json.dumps({"train_ids": [data["instance_id"]], "val_ids": [], "test_ids": []}))
            instances = folder / "instances.jsonl.gz"
            sampler.save_records(instances, {data["instance_id"]: {"instance_id": data["instance_id"], "fail_to_pass_ids": []}})
            retained, tier3 = folder / "retained.jsonl", folder / "tier3.jsonl"
            args = ["build_dataset.py", "--generations", str(output), "--instances", str(instances),
                    "--splits", str(splits), "--retained-out", str(retained), "--tier3-out", str(tier3),
                    "--manifest", str(folder / "manifest.json")]
            with patch.object(sys, "argv", args):
                build_dataset.main()
            self.assertEqual(list(read_jsonl(retained))[0]["llm_t12_hunks"][0]["sample"], 0)
            self.assertEqual(list(read_jsonl(tier3))[0]["tier3_hunks"][0]["sample"], 1)
            before = output.read_bytes()
            with patch.object(sys, "argv", ["sample_haiku_merge.py", "--input", str(input_path),
                    "--generations", str(output), "--dry-run", "--max", "1"]):
                sampler.main()
            self.assertEqual(output.read_bytes(), before)


    def test_build_restores_source_context_and_export_order(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            data = inst()
            record = {
                "instance_id": data["instance_id"], "repo": data["repo"],
                "meta": {"gold_patch": diff(10) + diff(20)},
                "generation": {"extracted_diff": diff(10) + diff(20) + diff(45)},
            }
            hunks = candidate_hunks(record)
            record["export_order"] = {
                "retained": [export_hunk_key(hunks[1]), export_hunk_key(hunks[0])],
                "tier3": [export_hunk_key(hunks[2])],
                "retained_issue": 0, "tier3_issue": 0,
            }
            generations, sources = folder / "generations.jsonl.gz", folder / "sources.jsonl.gz"
            sampler.save_records(generations, {data["instance_id"]: record})
            sampler.save_records(sources, {data["instance_id"]: data})
            instances = folder / "instances.jsonl"
            sampler.save_records(instances, {data["instance_id"]: {
                "instance_id": data["instance_id"], "fail_to_pass_ids": []}})
            splits = folder / "splits.json"
            splits.write_text(json.dumps({"train_ids": [data["instance_id"]], "val_ids": [], "test_ids": []}))
            retained, tier3, manifest = (folder / name for name in ("retained.jsonl", "tier3.jsonl", "manifest.json"))
            args = ["build_dataset.py", "--generations", str(generations),
                    "--source-instances", str(sources), "--instances", str(instances),
                    "--splits", str(splits), "--retained-out", str(retained),
                    "--tier3-out", str(tier3), "--manifest", str(manifest)]
            with patch.object(sys, "argv", args):
                build_dataset.main()
            rows = list(read_jsonl(retained))[0]["llm_t12_hunks"]
            self.assertEqual([h["old_start_line"] for h in rows], [20, 10])
            self.assertEqual([h["hunk_id"] for h in rows], [0, 1])
            self.assertEqual(rows[0]["context_before"], [f"line{i}" for i in range(14, 19)])
            self.assertEqual(rows[0]["context_after"], [f"line{i}" for i in range(20, 25)])
            self.assertEqual([h["tier_label"] for h in rows], [2, 2])
            self.assertEqual(json.loads(manifest.read_text())["total"]["T3_hunks"], 1)
            broken = copy.deepcopy(record)
            broken["generation"]["extracted_diff"] = diff(10, "changed") + diff(20) + diff(45)
            sampler.save_records(generations, {data["instance_id"]: broken})
            with patch.object(sys, "argv", args), self.assertRaisesRegex(ValueError, "export_order"):
                build_dataset.main()
            sampler.save_records(generations, {data["instance_id"]: record})
            sources.unlink()
            with patch.object(sys, "argv", args), self.assertRaisesRegex(ValueError, "source_files missing"):
                build_dataset.main()


if __name__ == "__main__":
    unittest.main()
