"""Regression checks for the evaluation protocol.

Run from the project root (CPU only, no model download):
    CUDA_VISIBLE_DEVICES="" gpu_env312/bin/python tests/check_evaluation.py

  1. a T0 distractor is scored against the ORIG unit of ITS OWN instance (not against the evaluated issue's
     files, which always gave it ORIG = 0 and let the ORIG view separate T0 by file membership)
  2. scoring a grid of (alpha, beta, gamma) from one encoding gives the same metrics as running each triple alone
  3. the selected weights are stored per checkpoint, are applied on later runs, are not applied to another
     checkpoint, and explicit --score-* flags win (ablations)
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evaluate                                                       # noqa: E402
from config import default_config                                     # noqa: E402
from data.score_weights import (                                      # noqa: E402
    load_selected_weights, save_selected_weights, select_best, weight_grid, weights_path,
)


def _hunk(fp: str, start: int, text: str, tier: int) -> dict:
    return {"filepath": fp, "old_start_line": start, "hunk_diff": f"@@ -{start},1 +{start},1 @@\n-x\n+{text}\n",
            "context_before": [], "context_after": [], "tier_label": tier, "sample": 0}


def _instance(iid: str, fp: str, unit_code: str) -> dict:
    return {"instance_id": iid, "repo": "repo", "requirement": f"req {iid}",
            "source_units": [{"filepath": fp, "name": "f", "kind": "function", "start_line": 1, "end_line": 100,
                              "code": unit_code}],
            "test_functions": [{"filepath": "t.py", "function_name": "t", "code": f"def test_{iid}(): pass"}]}


def _write(path: Path, records: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return str(path)


class _Recorder:
    """Stub encoder: deterministic pseudo-random similarities per text; records the ORIG texts it was given."""
    def __init__(self):
        self.orig_by_hunk: dict[str, list[str]] = {}

    @staticmethod
    def _r(text: str, salt: str) -> float:
        return int(hashlib.md5((salt + text).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    def similarity_components(self, hunk_texts, req_texts, test_texts, orig_texts, device, use_projection=True):
        for h, o in zip(hunk_texts, orig_texts):
            self.orig_by_hunk[h] = list(o)
        n = len(hunk_texts)
        return (torch.tensor([self._r(h, "q") for h in hunk_texts]),
                torch.tensor([self._r(h, "t") for h in hunk_texts]),
                torch.tensor([self._r(h, "o") for h in hunk_texts]))


class _World:
    """A tiny split: instances, retained hunks, Tier-3 hunks, patched into evaluate for the duration of a check."""
    def __init__(self, tmp: Path, n_instances: int = 6, seed: int = 0):
        rng = random.Random(seed)
        self.instances, ret, t3 = [], [], []
        for k in range(n_instances):
            iid, fp = f"repo__x-{k}", f"f{k}.py"
            self.instances.append(_instance(iid, fp, f"UNIT_OF_{iid}"))
            ret.append({"instance_id": iid, "llm_t12_hunks": [_hunk(fp, 10 + 30 * j, f"ret{k}_{j}", 1 + (j % 2)) for j in range(2)]})
            t3.append({"instance_id": iid, "tier3_hunks": [_hunk(fp, 500 + 30 * j, f"t3_{k}_{j}", 3) for j in range(rng.randint(1, 3))]})
        self.cfg = copy.deepcopy(default_config)
        self.cfg.data.llm_t12_path = _write(tmp / f"ret_{seed}.jsonl", ret)
        self.cfg.eval.tier3_path = _write(tmp / f"t3_{seed}.jsonl", t3)

    def __enter__(self):
        self._orig = evaluate.load_test_instances
        evaluate.load_test_instances = lambda cfg, all_instances=False, split="test": self.instances
        return self

    def __exit__(self, *exc):
        evaluate.load_test_instances = self._orig


# ---------------------------------------------------------------------------
def check_distractor_orig_is_its_own() -> None:
    with tempfile.TemporaryDirectory() as d, _World(Path(d), n_instances=3) as w:
        enc = _Recorder()
        evaluate.run_retrieval(w.cfg, enc, torch.device("cpu"), repo_pool=True)
        seen = {}
        for h, origs in enc.orig_by_hunk.items():
            for k in range(3):
                if f"ret{k}_" in h:
                    seen.setdefault(k, set()).update(origs)
        # every retained hunk of instance k was scored (as a candidate of k, and as a distractor of the others)
        # and each time against instance k's own unit: never with an empty ORIG list, never with another issue's unit
        for k in range(3):
            assert seen[k] == {f"UNIT_OF_repo__x-{k}"}, f"instance {k}: ORIG texts were {seen[k]}"
        assert all(origs for h, origs in enc.orig_by_hunk.items() if "ret" in h), "a distractor got ORIG = []"


def check_grid_equals_single_runs() -> None:
    grid = [(1.0, 0.5, 0.5), (1.0, 0.25, 2.0), (1.0, 2.0, 0.25), (1.0, 1.0, 1.0)]
    with tempfile.TemporaryDirectory() as d, _World(Path(d), n_instances=8, seed=3) as w:
        together = evaluate.run_retrieval(w.cfg, _Recorder(), torch.device("cpu"), repo_pool=True, weight_grid=grid)
        assert len(together) == len(grid)
        for (a, b, g), res in zip(grid, together):
            cfg = copy.deepcopy(w.cfg)
            cfg.eval.score_alpha, cfg.eval.score_beta, cfg.eval.score_gamma = a, b, g
            alone = evaluate.run_retrieval(cfg, _Recorder(), torch.device("cpu"), repo_pool=True)[0]
            for m in ("ndcg_k", "t2_recall", "psr", "n_ndcg_k"):
                assert alone[m] == res[m], f"grid != single run for {(a, b, g)} {m}: {res[m]} vs {alone[m]}"
        assert len({r["ndcg_k"] for r in together}) > 1, "the stub produced identical metrics for all weights: test is vacuous"


def check_select_best_and_grid() -> None:
    g = weight_grid()
    assert len(g) == 16 and (1.0, 0.5, 0.5) in g and all(a > 0 and b > 0 and c > 0 for a, b, c in g)
    res = [{"alpha": 1, "beta": 1, "gamma": 1, "ndcg_k": 0.7, "psr": 0.5, "t2_recall": 0.9},
           {"alpha": 1, "beta": 2, "gamma": 1, "ndcg_k": 0.8, "psr": 0.4, "t2_recall": 0.1},
           {"alpha": 1, "beta": 3, "gamma": 1, "ndcg_k": 0.8, "psr": 0.6, "t2_recall": 0.1},
           {"alpha": 1, "beta": 4, "gamma": 1, "ndcg_k": None, "psr": None, "t2_recall": None}]
    assert select_best(res)["beta"] == 3, "ties on nDCG@k must be broken by PSR"


def check_weights_bound_to_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ckpt = d / "m2" / "best.pt"
        ckpt.parent.mkdir()
        ident = {"created": "2026-09-21T10:00:00", "epoch": 1, "val_loss": 4.2}
        best = {"alpha": 1.0, "beta": 2.0, "gamma": 0.25, "ndcg_k": 0.8, "psr": 0.5, "t2_recall": 0.5, "n_ndcg_k": 40}
        save_selected_weights(ckpt, ident, best, [best], {})
        assert weights_path(ckpt).exists() and load_selected_weights(ckpt)["beta"] == 2.0

        def run(identity, explicit=False, has_file=True):
            cfg = copy.deepcopy(default_config)
            cfg.eval.checkpoint_path = str(ckpt if has_file else d / "other" / "best.pt")
            src = evaluate.resolve_score_weights(cfg, SimpleNamespace(checkpoint_identity=identity), True, explicit)
            return cfg.eval, src

        ev, src = run(ident)
        assert (ev.score_alpha, ev.score_beta, ev.score_gamma) == (1.0, 2.0, 0.25) and "validation" in src
        try:
            run({**ident, "epoch": 2})
        except ValueError:
            pass
        else:
            raise AssertionError("weights selected for another checkpoint were applied")
        ev, src = run(ident, explicit=True)
        assert (ev.score_alpha, ev.score_beta, ev.score_gamma) == (1.0, 0.5, 0.5), "explicit flags must not be overridden"
        ev, src = run(ident, has_file=False)
        assert "NOT selected" in src, "a missing weights file must be reported"


CHECKS = [check_distractor_orig_is_its_own, check_grid_equals_single_runs, check_select_best_and_grid,
          check_weights_bound_to_checkpoint]

if __name__ == "__main__":
    failed = 0
    for fn in CHECKS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}\n{traceback.format_exc()}")
    sys.exit(1 if failed else 0)
