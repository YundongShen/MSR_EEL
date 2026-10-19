"""Global configuration for Edit Entailment Learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    # Parsed instances (output of data_pipeline/parse_instances.py): the paper uses all 2,291 SWE-bench issues.
    # Read them with data.instances.load_instances, which drops the reference patch's own hunks.
    instances_path: str = "data/processed/instances_full.jsonl"

    # Tier-3 hunks used as same-instance hard negatives during training
    tier3_hard_neg_path: str = ""

    # Retained (Tier 1/2) hunks for training and every evaluation: LLM-generated hunks that
    # match a reference-patch hunk (built by data_pipeline/build_dataset.py).  Retained, Tier-3 and
    # distractor candidates all come from the same generation process.
    llm_t12_path: str = "data/cache/llm_t12_hunks.jsonl"

    cache_dir: str = "data/cache"

    # Which of the four pair types to include during training
    pair_types: tuple[str, ...] = (
        "req_test", "req_hunk", "orig_hunk", "req_orig"
    )

    # Splits (instance-level, not pair-level)
    val_frac:  float = 0.1
    test_frac: float = 0.1

    max_req_chars: int = 2000   # requirement truncation before tokenisation


@dataclass
class ModelConfig:
    # Backbone — set to "microsoft/codebert-base" for quick local tests
    encoder_name: str = "microsoft/unixcoder-base"
    projection_dim: int = 256
    dropout: float = 0.1
    max_length: int = 512       # token budget per entity


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 2e-5
    weight_decay: float = 1e-4
    epochs: int = 10
    warmup_steps: int = 200
    learn_temperature: bool = True     # learnable InfoNCE τ (CLIP-style)
    temperature: float = 0.07          # initial τ value
    grad_clip: float = 1.0
    log_every: int = 50
    eval_every: int = 500
    checkpoint_dir: str = "checkpoints"
    resume_from: str | None = None
    seed: int = 42
    num_workers: int = 4

    # Pair-type loss weights (uniform by default)
    pair_weights: dict[str, float] = field(default_factory=lambda: {
        "req_test":  1.0,
        "req_hunk":  1.0,
        "orig_hunk": 1.0,
        "req_orig":  1.0,
    })

    # Supervision level switches (M1→M4 spectrum)
    same_repo_hard_neg: bool = False   # M3 (= M2 + this): same-repo cross-issue retained hunks as extra negatives
    tier_aware_loss:    bool = False   # M4: Tier-1 weight=1.0, Tier-2 weight=0.67


@dataclass
class EvalConfig:
    checkpoint_path: str = "checkpoints/m2/best.pt"   # EEL = M2

    # Entailment score coefficients  α·sim(hunk,req) + β·sim(hunk,test) + γ·sim(hunk,orig)
    score_alpha: float = 1.0
    score_beta:  float = 0.5
    score_gamma: float = 0.5

    # nDCG relevance weights per tier
    tier_relevance: dict[int, float] = field(default_factory=lambda: {
        1: 3.0,   # Tier 1: retained hunk linked to a fail-to-pass test
        2: 2.0,   # Tier 2: retained hunk without such a link
        3: 0.0,   # Tier 3: scope creep
    })

    # Number of instances for geometric validation
    geometry_n_instances: int = -1   # -1 = all test instances

    # Retrieval top-k values
    retrieval_k_values: tuple[int, ...] = (5, 10, 20)

    # Optional: path to tier3_hunks.jsonl for mixing Tier-3 into evaluation
    tier3_path: str = ""


@dataclass
class LLMConfig:
    """Config for the LLM-based unconstrained patch generation pipeline."""
    provider: str = "openai"
    model: str = "Qwen2.5-Coder-32B-Instruct"
    api_base: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.2


@dataclass
class Config:
    data:  DataConfig  = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval:  EvalConfig  = field(default_factory=EvalConfig)
    llm:   LLMConfig   = field(default_factory=LLMConfig)

    def __post_init__(self) -> None:
        Path(self.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.data.cache_dir).mkdir(parents=True, exist_ok=True)


default_config = Config()
