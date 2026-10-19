# Learning Maintainer Preferences for Generated Code Edits from Repository Context

This repository contains the implementation, datasets, and experimental records for **Edit Entailment Learning (EEL)**. EEL is a multi-view contrastive method for ranking generated code edits by their correspondence to changes in a reference maintainer patch.

EEL maps requirements (REQ), failing tests (TEST), unmodified source units (ORIG), and candidate hunks (HUNK) into a shared representation space. Training aligns retained hunks with their repository context and uses unmatched hunks from the same issue as hard negatives. At inference, EEL combines REQ, TEST, and ORIG similarities to rank candidates within each issue. Reference patches provide training labels; ranking uses the candidate hunks and repository context.

## Repository structure

| Path | Contents |
|---|---|
| [`Code/`](Code/) | Model, training, evaluation, baselines, and analysis scripts |
| [`Code/data_pipeline/`](Code/data_pipeline/) | Candidate generation, hunk matching, and dataset construction |
| [`Code/reproduce/REPRODUCE.md`](Code/reproduce/REPRODUCE.md) | Environment setup and experiment commands |
| [`Data/eel_dataset/`](Data/eel_dataset/) | Parsed instances, generated candidates, and fixed data splits |
| [`Data/swebench/`](Data/swebench/) | SWE-bench source inputs |
| [`Data/Evaluation/`](Data/Evaluation/) | Training logs, evaluation results, and supporting analyses |

## Dataset and labels

The core dataset is based on SWE-bench Full and contains 2,291 issues from 12 Python repositories. Repeated Haiku sampling produces a deduplicated candidate set for each issue. Generated hunks matched to reference changes form the retained set; unmatched generated hunks form T3.

| Item | Count |
|---|---:|
| Training / validation / test issues | 1,833 / 229 / 229 |
| T1: retained, statically linked to a failing test | 3,354 |
| T2: retained, without a static test link | 5,869 |
| T3: unmatched generated hunks | 1,257 |
| Test issues containing both retained and T3 hunks | 82 |

`Retained-H` denotes generated T1 and T2 hunks. These candidates are stored in `llm_t12_hunks.jsonl`; T3 candidates are stored in `tier3_hunks.jsonl`. `splits.json` defines the fixed issue-level split. The parsed requirements, tests, and source units are stored in `instances_full.jsonl.gz`.

## Getting started

Use Python 3.11 or 3.12. From the repository root:

```bash
cd Code
bash reproduce/setup_env.sh
source env/bin/activate
export PYTHONHASHSEED=42
```

Setup installs the dependencies and prepares runtime data from the adjacent `Data` directory. Set `EEL_PYTHON_BIN` if the interpreter is not available as `python3.12`. To check the published data without installing dependencies, run `bash reproduce/setup_env.sh --check` from `Code/`.

Train the standard EEL configuration (M2):

```bash
python train.py \
  --instances data/processed/instances_full.jsonl \
  --llm-t12 data/cache/llm_t12_hunks.jsonl \
  --tier3 data/cache/tier3_hunks.jsonl \
  --pair-types req_test req_hunk orig_hunk req_orig \
  --epochs 10 \
  --checkpoint-dir checkpoints/m2
```

Training selects the checkpoint with the lowest validation loss. Evaluate that checkpoint with the reported scoring weights:

```bash
python evaluate.py \
  --checkpoint checkpoints/m2/best.pt \
  --exp retrieval \
  --instances data/processed/instances_full.jsonl \
  --llm-t12 data/cache/llm_t12_hunks.jsonl \
  --tier3 data/cache/tier3_hunks.jsonl \
  --repo-pool \
  --score-alpha 1.0 --score-beta 0.5 --score-gamma 0.5
```

## Evaluation and experiment records

Core ranking uses the 82 test issues containing at least one retained hunk and one T3 hunk. Each candidate set also includes up to 50 T0 distractors from other test issues in the same repository. All compared methods use the same candidate sets.

We report instance-level macro-averages. nDCG@k uses relevance values 3, 2, 0, and 0 for T1, T2, T3, and T0, with `k = |T1 ∪ T2|`. T1-Recall and T2-Recall measure recovery in the top k and are averaged over issues containing the respective tier. Perfect Separation Rate (PSR) measures whether every retained hunk ranks above every T3 hunk; it excludes T0.

| Experiment | Records under `Data/Evaluation/` |
|---|---|
| RQ1: retention ranking and baselines | [`baselines/`](Data/Evaluation/baselines/), [`training_variants/`](Data/Evaluation/training_variants/) |
| RQ2: contribution of repository views | [`view_ablation/`](Data/Evaluation/view_ablation/) |
| RQ3: negative constructions and mechanism diagnostics | [`training_variants/`](Data/Evaluation/training_variants/) |
| RQ4: transfer across repositories, datasets, generators, and languages | [`generalization/`](Data/Evaluation/generalization/) |
| Engineer judgments | [`Discussion/`](Data/Evaluation/Discussion/) |
| Generated–reference hunk similarity | [`hunk_similarity/`](Data/Evaluation/hunk_similarity/) |

The [reproduction guide](Code/reproduce/REPRODUCE.md) provides commands for the baselines, ablations, training variants, generalization experiments, and supporting analyses. M2 encoder parameters (`eel_m2_weights.pt`, stored under `encoder_state`) are available on [Zenodo](https://zenodo.org/records/23473861).

## Further use

We welcome researchers to apply EEL to other public datasets, construct additional hunk-level supervision from repository context, and investigate improvements to the method. Extending the available repositories, programming languages, and developer judgments can support further study of maintainer preferences for generated edits.
