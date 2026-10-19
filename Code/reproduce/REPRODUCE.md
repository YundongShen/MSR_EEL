# Reproduction

Commands below run from the `Code` directory of `MSR_EEL_SUBMIT` and use the
published data in the adjacent `Data` directory.

## Environment and data

From the root of the downloaded and extracted replication package:

```bash
cd Code
bash reproduce/setup_env.sh
source env/bin/activate
export PYTHONHASHSEED=42
```

The supported environment is Python 3.11 or 3.12 with the dependencies in
`reproduce/requirements.txt`. The default interpreter is `python3.12`; set
`EEL_PYTHON_BIN` to use another interpreter. Linux GPU setup uses PyTorch 2.5.1
CUDA 12.1 wheels. Set
`EEL_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu` for CPU-only setup.

`bash reproduce/setup_env.sh --check` checks the published data without writing
files or installing packages. `--data-only` prepares data without installing an
environment. For a separate Code checkout, set `EEL_DATA_ROOT` to the final
submission's `Data` directory. The setup checks:

| Item | Count |
|---|---:|
| Issues; train / validation / test | 2291; 1833 / 229 / 229 |
| Generated Retained-H; T1 / T2 | 9223; 3354 / 5869 |
| Generated T3 hunks / issues | 1257 / 743 |
| Core evaluation issues / T2-Recall issues | 82 / 47 |
| Django issues / eligible Django issues | 849 / 204 |

Setup decompresses the parsed instances into `data/processed/`, copies the fixed
splits and generated candidates, and creates Django and SymPy instance subsets
and their complements. Model candidates are read from `llm_t12_hunks.jsonl` and
`tier3_hunks.jsonl`; reference-patch hunks are excluded by the instance loader.

`Retained-H` denotes LLM-generated hunks matched to the reference patch
(T1/T2); `Gold-H` denotes the reference patch's own hunks. Code and JSON use
`retained` in identifiers, including `n_retained`, `mean_retained` and
`std_retained`. The metric field `ndcg_k` uses the Retained-H count for k.

For Slurm, prefix a command with `sbatch reproduce/job.sbatch`. Supply the
account, partition, GPU type and time limit required by your cluster as `sbatch`
options. Setup creates `logs/` before submission. The wrapper uses one GPU and
eight CPU cores. Training and encoder evaluation benefit from a GPU; BM25 and
data preparation can run on CPU.

## Protocol and training

Training uses seed 42, the fixed issue splits, UniXCoder, a 256-dimensional
projection, dropout 0.1, and at most 512 tokens per entity. Requirements are
truncated to 2000 characters. AdamW uses learning rate 2e-5, weight decay 1e-4,
200 warmup steps, cosine decay and gradient clipping at 1.0. Temperature starts
at 0.07, is learned and is clamped to [0.01, 1.0]. Batch size is 32; each epoch
draws 250000 pair-type-balanced samples. The four pair types are REQ–TEST,
REQ–HUNK, ORIG–HUNK and REQ–ORIG, with unit weights.

Ten epochs is the maximum training budget.

```bash
python train.py --instances data/processed/instances_full.jsonl --epochs 10 --checkpoint-dir checkpoints/m1 --pair-types req_test req_hunk orig_hunk req_orig
python train.py --instances data/processed/instances_full.jsonl --epochs 10 --checkpoint-dir checkpoints/m2 --pair-types req_test req_hunk orig_hunk req_orig --tier3 data/cache/tier3_hunks.jsonl
python train.py --instances data/processed/instances_full.jsonl --epochs 10 --checkpoint-dir checkpoints/m3 --pair-types req_test req_hunk orig_hunk req_orig --tier3 data/cache/tier3_hunks.jsonl --same-repo
```

M1 uses in-batch negatives. M2 adds same-issue T3 negatives. M3 retains M2's
negatives and adds up to three retained hunks from other issues in the same
repository per anchor. M0 uses the untrained backbone without the projection
for ranking.

```bash
python train.py --instances data/processed/instances_full.jsonl --epochs 10 --checkpoint-dir checkpoints/m2_abl_no_orig --pair-types req_test req_hunk --tier3 data/cache/tier3_hunks.jsonl
python train.py --instances data/processed/instances_full.jsonl --epochs 10 --checkpoint-dir checkpoints/m2_abl_no_test --pair-types req_hunk orig_hunk req_orig --tier3 data/cache/tier3_hunks.jsonl
python train.py --instances data/processed/instances_full.jsonl --epochs 10 --checkpoint-dir checkpoints/m2_abl_no_req --pair-types orig_hunk --tier3 data/cache/tier3_hunks.jsonl
python train.py --instances data/processed/instances_full.jsonl --epochs 10 --checkpoint-dir checkpoints/exp2_req_only --pair-types req_hunk --tier3 data/cache/tier3_hunks.jsonl
python train.py --instances data/processed/instances_full.jsonl --epochs 10 --checkpoint-dir checkpoints/exp2_test_only --pair-types req_hunk req_test --tier3 data/cache/tier3_hunks.jsonl
```

TEST-view retains REQ as a training anchor but scores with TEST alone. The
ORIG-only model and the no-REQ ablation both use the ORIG–HUNK relation.

## Core ranking and repository-view ablations

Define these Bash arrays once for the evaluation commands below:

```bash
A=(--instances data/processed/instances_full.jsonl --llm-t12 data/cache/llm_t12_hunks.jsonl --tier3 data/cache/tier3_hunks.jsonl --repo-pool)
W=(--score-alpha 1.0 --score-beta 0.5 --score-gamma 0.5)
```

The retention score is `alpha * sim(h, REQ) + beta * max_t sim(h, t) + gamma *
sim(h, o(h))`. TEST takes the maximum over individual tests. ORIG uses the
normalized mean of matched source-unit embeddings and is zero when no unit
matches. The reported weights are (1.0, 0.5, 0.5); the commands specify them
explicitly. The evaluator also supports validation-only weight selection with
`--select-weights`.

Core ranking uses the 82 eligible test issues. `--repo-pool` adds up to 50
same-repository retained hunks from other test issues as T0 candidates, with
sampling and tie breaks seeded at 42. All baseline commands use the same
candidate iterator. nDCG uses gains 3/2/0/0 for T1/T2/T3/T0 and
`k = |T1 union T2|`. T1-Recall is averaged over the 43 eligible issues with T1;
T2-Recall is averaged over the 47 eligible issues with T2;
PSR compares T1/T2 against T3 and excludes T0.

```bash
python evaluate.py --checkpoint none --no-projection --exp retrieval "${A[@]}" "${W[@]}"
python evaluate.py --checkpoint checkpoints/m1/best.pt --exp retrieval "${A[@]}" "${W[@]}"
python evaluate.py --checkpoint checkpoints/m2/best.pt --exp retrieval "${A[@]}" "${W[@]}"
python evaluate.py --checkpoint checkpoints/m3/best.pt --exp retrieval "${A[@]}" "${W[@]}"

python evaluate.py --checkpoint checkpoints/m2_abl_no_orig/best.pt --exp retrieval "${A[@]}" --score-alpha 1.0 --score-beta 0.5 --score-gamma 0.0
python evaluate.py --checkpoint checkpoints/m2_abl_no_test/best.pt --exp retrieval "${A[@]}" --score-alpha 1.0 --score-beta 0.0 --score-gamma 0.5
python evaluate.py --checkpoint checkpoints/exp2_req_only/best.pt --exp retrieval "${A[@]}" --score-alpha 1.0 --score-beta 0.0 --score-gamma 0.0
python evaluate.py --checkpoint checkpoints/exp2_test_only/best.pt --exp retrieval "${A[@]}" --score-alpha 0.0 --score-beta 1.0 --score-gamma 0.0
python evaluate.py --checkpoint checkpoints/m2_abl_no_req/best.pt --exp retrieval "${A[@]}" --score-alpha 0.0 --score-beta 0.0 --score-gamma 1.0
```

Training produces the checkpoints used by the evaluation commands. Checkpoint
weights are not included in the repository. The default loader checks the
generated-data provenance stamp; `--allow-legacy-checkpoint` permits loading
older checkpoints without this stamp.

## Baseline implementations

```bash
python baselines/cross_encoder_train.py --instances data/processed/instances_full.jsonl --llm-t12 data/cache/llm_t12_hunks.jsonl --tier3 data/cache/tier3_hunks.jsonl --splits data/processed/splits.json --out checkpoints/cross_encoder --cross-negatives 3 --epochs 3 --accum 8
python baselines/baselines_bm25.py "${A[@]}" --output logs/baseline_bm25.json
python baselines/baselines_pretrained.py --model microsoft/codebert-base "${A[@]}" --output logs/baseline_codebert.json
python baselines/baselines_pretrained.py --model microsoft/graphcodebert-base "${A[@]}" --output logs/baseline_graphcodebert.json
python baselines/baselines_pretrained.py --model microsoft/unixcoder-base "${A[@]}" --output logs/baseline_unixcoder_req_hunk.json
python baselines/baselines_extra.py --mode codet5 --model Salesforce/codet5p-110m-embedding "${A[@]}" --output logs/baseline_codet5.json
python baselines/baselines_extra.py --mode cross_encoder --model checkpoints/cross_encoder "${A[@]}" --output logs/baseline_cross_encoder.json
```

BM25 uses lexical matching. CodeBERT, GraphCodeBERT, and the UniXCoder baseline
score REQ–HUNK cosine similarity using normalized mean token embeddings. M0
uses the three-view retention score. CodeT5 uses the CodeT5+ 110M embedding checkpoint.

Cross-Encoder fine-tunes `microsoft/unixcoder-base` on training-split
requirement–hunk pairs and evaluates the checkpoint with the lowest validation
loss. The REQ-only command above trains the supervised requirement–hunk baseline.

## RQ3 diagnostics and Django figures

```bash
python analysis/exp3_analysis.py --m1 checkpoints/m1/best.pt --m2 checkpoints/m2/best.pt --instances data/processed/instances_full.jsonl --llm-t12 data/cache/llm_t12_hunks.jsonl --tier3 data/cache/tier3_hunks.jsonl --output logs/exp3_analysis.json --analysis all
python analysis/viz_django.py --checkpoint-m2 checkpoints/m2/best.pt --instances data/processed/instances_full.jsonl --llm-t12 data/cache/llm_t12_hunks.jsonl --tier3 data/cache/tier3_hunks.jsonl --out-prefix logs/django_viz --encode-only
python analysis/viz_django.py --out-prefix logs/django_viz --plot-only
python analysis/plot_django_regions.py --embs logs/django_viz_embs.npz --out-prefix logs/django --scatter-only
```

The mechanism diagnostics use the core test split. Semantic difficulty groups the 82 eligible
issues into 41/41 using M0's mean requirement–T3 cosine similarity. The ORIG
comparison includes test hunks with matched source units and excludes T0.

Analysis details and output records are listed in the
[analysis notes](../analysis/README.md#django-figures-and-region-statistics).

To classify Django regions:

```bash
python analysis/analyze_regions.py --npz logs/django_viz_data.npz --instances data/processed/instances_full.jsonl --out logs/django_region_analysis.json
```

Optional edit-type analysis labels generated retained hunks. Export scores with
`hunk_key` using the command below before running the analysis. Edit-type labels
are separate from the T1/T2 labels.

```bash
python evaluate.py --checkpoint checkpoints/m2/best.pt --exp geometry --instances data/processed/instances_full.jsonl --llm-t12 data/cache/llm_t12_hunks.jsonl --tier3 data/cache/tier3_hunks.jsonl "${W[@]}" --save-hunk-scores logs/exp2_hunk_scores_m2.jsonl
python analysis/exp2_label_hunks.py --instances data/processed/instances_full.jsonl --splits data/processed/splits.json --llm-t12 data/cache/llm_t12_hunks.jsonl
python analysis/exp2_analysis.py
```

## RQ4: generalization

RQ4 uses the files below under
`../Data/Evaluation/generalization/evaluation_inputs/`.

| Evaluation | Directory | Instances | T3 hunks |
|---|---|---|---|
| SymPy omitted from training | `sympy_1cdf6f2/` | `instances_sympy.jsonl.gz` | `tier3_hunks.jsonl` |
| GitHub PRs (Python) | `github_prs_python/` | `instances.jsonl.gz` | `tier3_mix10.jsonl.gz` |
| GitHub PRs (Non-Python) | `github_prs_non_python/` | `instances.jsonl.gz` | `tier3_gen.jsonl.gz` |
| Gemini-2.5-Flash-Lite | `crossllm_gemini/` | `instances.jsonl.gz` | `tier3_gen.jsonl.gz` |
| Qwen2.5-Coder-7B | `crossllm_qwen7b/` | `instances.jsonl.gz` | `tier3_mix20.jsonl.gz` |
| DeepSeek-Coder-7B | `crossllm_deepseek7b/` | `instances.jsonl.gz` | `tier3_mix20.jsonl.gz` |
| SWE-bench Multilingual | `swebench_multilingual/` | `instances.jsonl.gz` | `tier3_gen.jsonl.gz` |

Prepare the inputs and evaluate each dataset:

```bash
python data_pipeline/prepare_rq4.py --output-dir data/cache/rq4
RQ4=data/cache/rq4
W=(--score-alpha 1.0 --score-beta 0.5 --score-gamma 0.5)
python evaluate.py --exp retrieval --checkpoint checkpoints/exp4_lso/best.pt --instances "$RQ4/sympy_1cdf6f2/instances.jsonl" --llm-t12 "$RQ4/sympy_1cdf6f2/llm_t12_hunks.jsonl" --tier3 "$RQ4/sympy_1cdf6f2/tier3_hunks.jsonl" --all-instances --repo-pool "${W[@]}"
python evaluate.py --exp retrieval --checkpoint checkpoints/m2/best.pt --instances "$RQ4/github_prs_python/instances.jsonl" --llm-t12 "$RQ4/github_prs_python/llm_t12_hunks.jsonl" --tier3 "$RQ4/github_prs_python/tier3_hunks.jsonl" --all-instances --repo-pool "${W[@]}"
python evaluate.py --exp retrieval --checkpoint checkpoints/m2/best.pt --instances "$RQ4/github_prs_non_python/instances.jsonl" --llm-t12 "$RQ4/github_prs_non_python/llm_t12_hunks.jsonl" --tier3 "$RQ4/github_prs_non_python/tier3_hunks.jsonl" --all-instances --repo-pool "${W[@]}"
python evaluate.py --exp retrieval --checkpoint checkpoints/m2/best.pt --instances "$RQ4/crossllm_gemini/instances.jsonl" --llm-t12 "$RQ4/crossllm_gemini/llm_t12_hunks.jsonl" --tier3 "$RQ4/crossllm_gemini/tier3_hunks.jsonl" --all-instances --repo-pool "${W[@]}"
python evaluate.py --exp retrieval --checkpoint checkpoints/m2/best.pt --instances "$RQ4/crossllm_qwen7b/instances.jsonl" --llm-t12 "$RQ4/crossllm_qwen7b/llm_t12_hunks.jsonl" --tier3 "$RQ4/crossllm_qwen7b/tier3_hunks.jsonl" --all-instances --repo-pool "${W[@]}"
python evaluate.py --exp retrieval --checkpoint checkpoints/m2/best.pt --instances "$RQ4/crossllm_deepseek7b/instances.jsonl" --llm-t12 "$RQ4/crossllm_deepseek7b/llm_t12_hunks.jsonl" --tier3 "$RQ4/crossllm_deepseek7b/tier3_hunks.jsonl" --all-instances --repo-pool "${W[@]}"
python evaluate.py --exp retrieval --checkpoint checkpoints/m2/best.pt --instances "$RQ4/swebench_multilingual/instances.jsonl" --llm-t12 "$RQ4/swebench_multilingual/llm_t12_hunks.jsonl" --tier3 "$RQ4/swebench_multilingual/tier3_hunks.jsonl" --all-instances --repo-pool "${W[@]}"
```

## Candidate sampling and rebuilding

Rebuild the published candidate files from the archived generations and raw
source inputs:

```bash
python data_pipeline/build_dataset.py --generations data/cache/haiku_generations.jsonl.gz --source-instances data/raw/swebench_full_instances.jsonl.gz --instances data/processed/instances_full.jsonl --splits data/processed/splits.json --retained-out data/cache/rebuilt_llm_t12_hunks.jsonl --tier3-out data/cache/rebuilt_tier3_hunks.jsonl --manifest data/cache/rebuilt_dataset_manifest.json
```

Training and evaluation above use the published candidates without rebuilding
them. For a new sampling experiment, set `ANTHROPIC_API_KEY` and use a separate
generation output. Samples are batched until the number of matched generated
hunks reaches the reference-hunk count. Duplicate hunks are removed; later
samples are merged into the issue record. If matches exceed that count, the
matches with the smallest start-line distances to reference hunks in the same
file are retained. Reusing a run ID resumes the corresponding sampling round.

```bash
python data_pipeline/sample_haiku_merge.py --input data/raw/swebench_full_instances.jsonl.gz --generations data/cache/new_haiku_generations.jsonl.gz --model claude-haiku-4-5-20251001 --temperature 0.7 --max-tokens 4096 --samples 1 --run-id new-sampling --dry-run
python data_pipeline/sample_haiku_merge.py --input data/raw/swebench_full_instances.jsonl.gz --generations data/cache/new_haiku_generations.jsonl.gz --model claude-haiku-4-5-20251001 --temperature 0.7 --max-tokens 4096 --samples 1 --run-id new-sampling
python data_pipeline/build_dataset.py --generations data/cache/new_haiku_generations.jsonl.gz --source-instances data/raw/swebench_full_instances.jsonl.gz --instances data/processed/instances_full.jsonl --splits data/processed/splits.json --retained-out data/cache/new_llm_t12_hunks.jsonl --tier3-out data/cache/new_tier3_hunks.jsonl --manifest data/cache/new_dataset_manifest.json
```

`GEMINI_API_KEY` is needed for optional edit-type labeling, and
`ANTHROPIC_API_KEY` for optional region classification and sampling.

## Hunk similarity

This analysis compares changed lines using UniXCoder cosine similarity.
Pairing details are in the [analysis notes](../analysis/README.md#hunk-similarity).

Recompute the statistics from the submitted pairs and embeddings:

```bash
mkdir -p logs/hunk_similarity
python analysis/hunk_similarity/analyse_ux.py > logs/hunk_similarity/table.txt
```

To rebuild pairs and embeddings from the published candidates:

```bash
mkdir -p logs/hunk_similarity
python analysis/hunk_similarity/prep.py > logs/hunk_similarity/prep.log
python analysis/hunk_similarity/embed_ux.py --pairs logs/hunk_similarity/pairs.json
python analysis/hunk_similarity/analyse_ux.py --pairs logs/hunk_similarity/pairs.json --embeddings logs/hunk_similarity/emb_ux.npy > logs/hunk_similarity/table.txt
```

For GPU submission, replace the embedding command with
`sbatch analysis/hunk_similarity/run_embed_ux.slurm --pairs logs/hunk_similarity/pairs.json`.

## Submitted result records

| Paper experiment | Result directory under `../Data/Evaluation/` |
|---|---|
| RQ1 baseline JSONs | `baselines/` |
| EEL, training variants, and RQ3 diagnostics | `training_variants/` |
| RQ2 repository-view ablations | `view_ablation/` |
| RQ4 transfer | `generalization/` |
| Human evaluation | `Discussion/` |
| Hunk similarity | `hunk_similarity/` |

Evaluation commands write outputs to `logs/`. Training saves the checkpoint
with the lowest validation loss to `checkpoints/<variant>/best.pt`. Submitted
training logs and evaluation results are stored under `Data/Evaluation/`.

## Historical versions

### M4

M4 uses the M3 configuration with T1/T2 loss weights 1.0/0.67.

```bash
python train.py --instances data/processed/instances_full.jsonl --epochs 10 --checkpoint-dir checkpoints/m4 --pair-types req_test req_hunk orig_hunk req_orig --tier3 data/cache/tier3_hunks.jsonl --same-repo --tier-aware-loss
python evaluate.py --checkpoint checkpoints/m4/best.pt --exp retrieval "${A[@]}" "${W[@]}"
```

### Pooler-based Retrieval

Pooler-based Retrieval uses GraphCodeBERT's randomly initialized pooler without
task-specific training. The `coderetriever` CLI mode selects this pooler
configuration; the result JSON records the GraphCodeBERT model ID.

```bash
python baselines/baselines_extra.py --mode coderetriever --model microsoft/graphcodebert-base "${A[@]}" --output logs/baseline_pooler_retrieval.json
```

### Django-omitted comparison

The Django-omitted comparison is a historical version of the repository-shift
experiment. Its records are stored under
`Data/Evaluation/generalization/eval_logs/django_omitted.log` and
`Data/Evaluation/generalization/scores/per_issue_django_omitted.json`.
