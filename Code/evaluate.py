"""Evaluation entry point for Edit Entailment Learning.

Geometry analysis and within-issue retention ranking.

Usage:
    python evaluate.py --exp geometry
    python evaluate.py --exp retrieval
    python evaluate.py --exp all
    python evaluate.py --exp retrieval --repo-pool --checkpoint checkpoints/m2/best.pt \
        --instances data/processed/instances_full.jsonl --tier3 data/cache/tier3_hunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import Config, default_config
from data.entailment_dataset import _render_hunk, _units_for_hunk
from data.instances import load_instances, load_split_ids
from data.provenance import describe_hunk_files, load_checkpoint_checked
from data.score_weights import (
    checkpoint_identity, load_selected_weights, save_selected_weights, select_best, weight_grid,
)
from data.retained_hunks import (
    add_retained_args, apply_retained_args, load_tier3_hunks, retained_hunks,
)
from models.entailment_encoder import EntailmentEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_encoder(
    cfg: Config,
    device: torch.device,
    use_projection: bool = True,
    allow_legacy: bool = False,
) -> EntailmentEncoder:
    encoder = EntailmentEncoder(
        model_name=cfg.model.encoder_name,
        projection_dim=cfg.model.projection_dim,
        dropout=0.0,
        max_length=cfg.model.max_length,
    ).to(device)

    ckpt_path = Path(cfg.eval.checkpoint_path)
    if ckpt_path.exists():
        # Refuses a checkpoint that was not trained on LLM-generated hunks (data.provenance).
        ckpt = load_checkpoint_checked(ckpt_path, map_location=device, allow_legacy=allow_legacy)
        encoder.load_state_dict(ckpt["encoder_state"])
        encoder.checkpoint_identity = checkpoint_identity(ckpt)   # ties score_weights.json to this checkpoint
        log.info("Loaded encoder from %s", ckpt_path)
    elif use_projection:
        raise FileNotFoundError(
            f"Checkpoint {ckpt_path} not found. Train one first, or pass --no-projection "
            "for the untrained M0 backbone."
        )
    else:
        log.info("M0: untrained backbone, no checkpoint")

    encoder.eval()
    return encoder


# ---------------------------------------------------------------------------
# Tier-3 hunk loading
# ---------------------------------------------------------------------------

def load_tier3_lookup(path: str) -> dict[str, list[dict]]:
    """Load tier3_hunks.jsonl → {instance_id: [hunk_dict, ...]} (validated LLM-generated hunks)."""
    return load_tier3_hunks(path)


# ---------------------------------------------------------------------------
# Instance loading
# ---------------------------------------------------------------------------

def load_test_instances(cfg: Config, all_instances: bool = False, split: str = "test") -> list[dict]:
    """Load the instances of a split (same split logic as train.py); ``split`` is "test" or "val".

    ``load_instances`` drops the reference patch's own hunks: candidates come from
    data.retained_hunks only.
    """
    ids = None if all_instances else load_split_ids("data/processed/splits.json", split)
    return load_instances(cfg.data.instances_path, ids)


def _load_all_instances(cfg: Config) -> list[dict]:
    """Load every instance from the instances file (no split filter)."""
    return load_instances(cfg.data.instances_path)


# ---------------------------------------------------------------------------
# Geometric Structure Validation
# ---------------------------------------------------------------------------

_TIER_COLORS = {
    "1":  "#2196F3",   # blue
    "2a": "#4CAF50",   # green  — untested, modifies existing code
    "2b": "#FF9800",   # orange — untested, new auxiliary code
    "3":  "#F44336",   # red
}
_TIER_LABELS = {
    "1":  "Tier 1 (test-linked)",
    "2a": "Tier 2a (untested, modifies existing)",
    "2b": "Tier 2b (untested, new code)",
    "3":  "Tier 3 (drift)",
}


def run_geometry(
    cfg: Config,
    encoder: EntailmentEncoder,
    device: torch.device,
    all_instances: bool = False,
    use_projection: bool = True,
    collect_umap: bool = False,
    model_label: str = "m4",
    save_scores_path: str | None = None,
) -> None:
    log.info("=== Experiment 1: Geometric Structure Validation ===")

    instances = load_test_instances(cfg, all_instances=all_instances)
    if cfg.eval.geometry_n_instances > 0:
        instances = instances[: cfg.eval.geometry_n_instances]
    log.info("Test instances: %d", len(instances))

    tier3_lookup = load_tier3_lookup(cfg.eval.tier3_path) if cfg.eval.tier3_path else {}
    if tier3_lookup:
        log.info("Tier-3 lookup loaded: %d instances with scope-creep hunks", len(tier3_lookup))

    # Full entailment scores by tier
    tier_scores: dict[str, list[float]] = {"1": [], "2a": [], "2b": [], "3": []}
    # ORIG-component scores for the Tier-2a vs Tier-2b internal-gradient test
    orig_sim_scores: dict[str, list[float]] = {"2a": [], "2b": []}
    # Hunk embeddings for UMAP (only populated when collect_umap=True)
    embs_for_umap: dict[str, list[np.ndarray]] = {"1": [], "2a": [], "2b": [], "3": []}
    # Entity embeddings for UMAP (REQ/TEST/ORIG, one point per instance)
    entity_embs_for_umap: dict[str, list[np.ndarray]] = {"REQ": [], "TEST": [], "ORIG": []}
    # Per-hunk metadata for traceability: (instance_id, hunk_text, sim_req, sim_test, sim_orig)
    hunk_meta_for_umap: dict[str, list[tuple]] = {"1": [], "2a": [], "2b": [], "3": []}
    # Per-hunk score records for --save-hunk-scores
    save_scores = save_scores_path is not None
    hunk_scores_records: list[dict] = []
    need_anchors = collect_umap or save_scores

    for inst in instances:
        req          = inst.get("requirement", "")[:cfg.data.max_req_chars]
        test_texts   = [tf["code"] for tf in inst.get("test_functions", []) if tf.get("code")]
        source_units = inst.get("source_units", [])

        # Carry base_tier explicitly: tier3_lookup hunks have no tier_label field
        all_hunks: list[tuple[dict, int]] = [
            (h, h.get("tier_label") or 1) for h in retained_hunks(inst, cfg)
        ]
        for h in tier3_lookup.get(inst.get("instance_id", ""), []):
            all_hunks.append((h, 3))

        # Pre-compute instance-level anchor embeddings (normalised) for per-hunk sims
        iid = inst.get("instance_id", "")
        r_norm_inst = t_embs_inst = o_norm_inst = None
        if need_anchors:
            r_emb_inst = encoder.encode([req], "REQ", device, use_projection)
            r_norm_inst = F.normalize(r_emb_inst, dim=-1)           # (1, D)
            if collect_umap:
                entity_embs_for_umap["REQ"].append(r_emb_inst.cpu().numpy()[0])

            if test_texts:
                t_embs_inst = encoder.encode(test_texts, "TEST", device, use_projection)
                if collect_umap:
                    entity_embs_for_umap["TEST"].append(t_embs_inst.mean(dim=0).cpu().numpy())

            orig_codes_inst = [u["code"] for u in source_units[:10] if u.get("code")]
            if orig_codes_inst:
                o_embs_inst = encoder.encode(orig_codes_inst, "ORIG", device, use_projection)
                o_norm_inst = F.normalize(o_embs_inst.mean(dim=0, keepdim=True), dim=-1)  # (1, D)
                if collect_umap:
                    entity_embs_for_umap["ORIG"].append(o_embs_inst.mean(dim=0).cpu().numpy())

        for hunk, base_tier in all_hunks:
            hunk_text = _render_hunk(hunk)
            if not hunk_text.strip():
                continue

            relevant_units  = _units_for_hunk(hunk, source_units)
            hunk_orig_texts = [u["code"] for u in relevant_units if u.get("code")]

            # Tier-2 → 2a (has relevant ORIG units) or 2b (no ORIG match → new code)
            if base_tier == 2:
                eff_tier = "2a" if hunk_orig_texts else "2b"
            else:
                eff_tier = str(base_tier)

            score = encoder.entailment_score(
                hunk_texts=[hunk_text],
                req_texts=[req],
                test_texts=[test_texts],
                orig_texts=[hunk_orig_texts],
                device=device,
                alpha=cfg.eval.score_alpha,
                beta=cfg.eval.score_beta,
                gamma=cfg.eval.score_gamma,
                use_projection=use_projection,
            ).item()
            tier_scores[eff_tier].append(score)

            # Hunk embedding (shared across UMAP + ORIG-component computation)
            h_emb = encoder.encode([hunk_text], "HUNK", device, use_projection)  # (1, D)
            h_norm = F.normalize(h_emb, dim=-1)

            # Per-hunk component similarities — computed when UMAP or score saving is active
            sim_req = sim_test = sim_orig = 0.0
            o_norm_h = None
            if need_anchors:
                if collect_umap:
                    embs_for_umap[eff_tier].append(h_emb.cpu().numpy()[0])

                sim_req  = (h_norm * r_norm_inst).sum(-1).item()
                sim_test = (t_embs_inst @ h_norm.T).max().item() if t_embs_inst is not None else 0.0
                # ORIG: hunk-specific only; no fallback to instance-level to keep
                # sim_orig consistent with entailment_score (which uses 0 when orig_texts=[]).
                has_orig = bool(hunk_orig_texts)
                if has_orig:
                    o_embs_h = encoder.encode(hunk_orig_texts, "ORIG", device, use_projection)
                    o_norm_h = F.normalize(o_embs_h.mean(dim=0, keepdim=True), dim=-1)
                    sim_orig = (h_norm * o_norm_h).sum(-1).item()

                if collect_umap:
                    hunk_meta_for_umap[eff_tier].append((
                        iid,
                        hunk_text[:600],   # truncated for storage
                        round(sim_req, 4),
                        round(sim_test, 4),
                        round(sim_orig, 4),
                    ))
                if save_scores:
                    # sim_orig_bg: similarity to instance-level mean ORIG (background).
                    # delta_orig = sim_orig(matched) - sim_orig_bg removes the systematic
                    # inflation from context-line text overlap: context lines are ~94%
                    # contained in the matched ORIG unit, pushing raw sim_orig near ceiling.
                    sim_orig_bg = (h_norm * o_norm_inst).sum(-1).item() if o_norm_inst is not None else 0.0
                    hunk_scores_records.append({
                        "instance_id": iid,
                        "hunk_id":     hunk.get("hunk_id", ""),
                        "hunk_key":    hunk.get("hunk_key", ""),   # join key; hunk_id changes when the dataset is rebuilt
                        "filepath":    hunk.get("filepath", ""),
                        "tier":        base_tier,
                        "eff_tier":    eff_tier,
                        "has_orig":    has_orig,
                        "score":       round(score, 6),
                        "sim_req":     round(sim_req, 6),
                        "sim_test":    round(sim_test, 6),
                        "sim_orig":    round(sim_orig, 6),
                        "sim_orig_bg": round(sim_orig_bg, 6),
                        "delta_orig":  round(sim_orig - sim_orig_bg, 6),
                    })

            # ORIG-component similarity: γ·sim(hunk, mean(relevant_ORIG))
            # Used to confirm the Tier-2a >> Tier-2b internal gradient.
            if base_tier == 2:
                if hunk_orig_texts:  # 2a: has relevant ORIG units
                    if need_anchors and o_norm_h is not None:
                        orig_sim = sim_orig  # already computed above
                    else:
                        o_embs = encoder.encode(hunk_orig_texts, "ORIG", device, use_projection)
                        o_norm = F.normalize(o_embs.mean(dim=0, keepdim=True), dim=-1)
                        orig_sim = (h_norm * o_norm).sum(-1).item()
                else:               # 2b: no ORIG match → similarity is 0 by construction
                    orig_sim = 0.0
                orig_sim_scores[eff_tier].append(orig_sim)

    # --- Summary statistics ---
    for key in ("1", "2a", "2b", "3"):
        scores = tier_scores[key]
        if scores:
            log.info(
                "%s  n=%d  mean=%.4f  std=%.4f  median=%.4f",
                key, len(scores), np.mean(scores), np.std(scores), np.median(scores),
            )
        else:
            log.info("%s  n=0  (no samples)", key)

    from scipy.stats import mannwhitneyu   # only the geometry experiment needs scipy

    # --- Mann-Whitney U: full entailment score ---
    for ta, tb in [("1", "2a"), ("1", "2b"), ("2a", "2b"), ("2a", "3"), ("2b", "3"), ("1", "3")]:
        sa, sb = tier_scores[ta], tier_scores[tb]
        if len(sa) >= 2 and len(sb) >= 2:
            stat, p = mannwhitneyu(sa, sb, alternative="greater")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            log.info(
                "Mann-Whitney U [score]  %s > %s:  U=%.0f  p=%.4e  %s",
                ta, tb, stat, p, sig,
            )

    # --- Mann-Whitney U: ORIG-component only (Tier-2 internal gradient) ---
    sa2a, sa2b = orig_sim_scores["2a"], orig_sim_scores["2b"]
    if len(sa2a) >= 2 and len(sa2b) >= 2:
        stat, p = mannwhitneyu(sa2a, sa2b, alternative="greater")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        log.info(
            "Mann-Whitney U [ORIG-sim]  2a > 2b:  U=%.0f  p=%.4e  %s  "
            "(n_2a=%d mean=%.4f | n_2b=%d mean=%.4f)",
            stat, p, sig,
            len(sa2a), np.mean(sa2a),
            len(sa2b), np.mean(sa2b),
        )
    else:
        log.info("Mann-Whitney U [ORIG-sim]  2a > 2b: skipped (n_2a=%d, n_2b=%d)",
                 len(sa2a), len(sa2b))

    _maybe_histogram(tier_scores)
    if collect_umap:
        _maybe_umap(embs_for_umap, entity_embs_for_umap,
                    hunk_meta=hunk_meta_for_umap, model_label=model_label)
    if save_scores and hunk_scores_records:
        out = Path(save_scores_path)
        out.parent.mkdir(exist_ok=True)
        with open(out, "w") as fh:
            for rec in hunk_scores_records:
                fh.write(json.dumps(rec) + "\n")
        log.info("Per-hunk scores saved → %s  (%d records)", out, len(hunk_scores_records))


def _maybe_histogram(tier_scores: dict[str, list[float]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        log.info("matplotlib not available — skipping histogram.")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    for key in ("1", "2a", "2b", "3"):
        scores = tier_scores.get(key, [])
        if scores:
            ax.hist(scores, bins=30, alpha=0.6, color=_TIER_COLORS[key], label=_TIER_LABELS[key])
    ax.set_xlabel("Edit Entailment Score")
    ax.set_ylabel("Count")
    ax.set_title("Score distribution by tier")
    ax.legend()
    out = Path("logs/geometry_hist.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    log.info("Histogram saved → %s", out)
    plt.close(fig)


def _maybe_umap(
    embs_for_umap: dict[str, list[np.ndarray]],
    entity_embs: dict[str, list[np.ndarray]],
    hunk_meta: dict[str, list[tuple]] | None = None,
    model_label: str = "m4",
) -> None:
    try:
        import umap as umap_lib          # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.colors as mcolors  # type: ignore
        from matplotlib.cm import ScalarMappable  # type: ignore
        from mpl_toolkits.mplot3d import Axes3D  # type: ignore  # noqa: F401
    except ImportError:
        log.info("umap-learn or matplotlib not available — skipping UMAP.")
        return

    # --- Collect hunk embeddings (256D) for silhouette + UMAP ---
    hunk_embs_256: list[np.ndarray] = []
    hunk_keys:     list[str] = []
    for key in ("1", "2a", "2b", "3"):
        for emb in embs_for_umap.get(key, []):
            hunk_embs_256.append(emb)
            hunk_keys.append(key)

    if len(hunk_embs_256) < 10:
        log.info("Too few hunk embeddings for UMAP (%d) — skipping.", len(hunk_embs_256))
        return

    # --- Silhouette score in original 256D space (cosine) ---
    sil: float | None = None
    try:
        from sklearn.metrics import silhouette_score  # type: ignore
        sil = silhouette_score(np.array(hunk_embs_256), hunk_keys, metric="cosine")
        log.info("Silhouette score (256D cosine, hunk tiers): %.4f", sil)
    except Exception as e:
        log.info("Silhouette score skipped: %s", e)

    # --- Build combined array: hunk + entity embeddings ---
    # Entity anchors: REQ (★), TEST (▲), ORIG (■) — one point per instance
    entity_cfg = {
        "REQ":  ("*", "#9C27B0", "REQ",  100),
        "TEST": ("^", "#00BCD4", "TEST", 60),
        "ORIG": ("s", "#795548", "ORIG", 60),
    }
    all_embs: list[np.ndarray] = list(hunk_embs_256)
    all_tags: list[str] = list(hunk_keys)
    for etype in ("REQ", "TEST", "ORIG"):
        for emb in entity_embs.get(etype, []):
            all_embs.append(emb)
            all_tags.append(etype)

    n_hunks = len(hunk_embs_256)
    log.info("Running 3D UMAP on %d points (%d hunks + %d entity anchors)...",
             len(all_embs), n_hunks, len(all_embs) - n_hunks)

    reducer = umap_lib.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(np.array(all_embs))  # (N, 3)

    # --- Gradient colormap: deep blue (T1, most necessary) → red (T3, scope creep) ---
    tier_cmap = mcolors.LinearSegmentedColormap.from_list(
        "tier_necessity", ["#2196F3", "#4CAF50", "#FF9800", "#F44336"], N=256,
    )
    tier_cval = {"1": 0.0, "2a": 1 / 3, "2b": 2 / 3, "3": 1.0}

    # --- Save raw coords for optional M0 vs M4 paired figure ---
    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)
    # Build flat metadata arrays ordered to match hunk_keys
    meta_iid, meta_text, meta_sreq, meta_stest, meta_sorig = [], [], [], [], []
    if hunk_meta:
        for key in ("1", "2a", "2b", "3"):
            for (iid, txt, sr, st, so) in hunk_meta.get(key, []):
                meta_iid.append(iid)
                meta_text.append(txt)
                meta_sreq.append(sr)
                meta_stest.append(st)
                meta_sorig.append(so)

    np.savez(
        out_dir / f"umap_embs_{model_label}.npz",
        coords=coords, tags=np.array(all_tags), n_hunks=n_hunks,
        sil=np.array([sil if sil is not None else float("nan")]),
        meta_iid=np.array(meta_iid, dtype=object),
        meta_text=np.array(meta_text, dtype=object),
        meta_sreq=np.array(meta_sreq, dtype=np.float32),
        meta_stest=np.array(meta_stest, dtype=np.float32),
        meta_sorig=np.array(meta_sorig, dtype=np.float32),
    )
    log.info("UMAP embeddings saved → logs/umap_embs_%s.npz", model_label)

    # --- Draw single-model 3D figure ---
    fig = plt.figure(figsize=(12, 9))
    ax  = fig.add_subplot(111, projection="3d")
    _draw_umap_ax(ax, coords, all_tags, n_hunks, entity_cfg, tier_cmap, tier_cval, sil,
                  title=f"4-Entity Embedding Space — {model_label.upper()} (UMAP-3D)")
    sm = ScalarMappable(cmap=tier_cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.12, aspect=20)
    cbar.set_ticks([0.0, 1 / 3, 2 / 3, 1.0])
    cbar.set_ticklabels(
        ["T1 (test-linked)", "T2a (untested, modifies)", "T2b (untested, new)", "T3 (scope creep)"],
        fontsize=7,
    )
    cbar.set_label("HUNK necessity →", fontsize=8)

    out = out_dir / f"umap_3d_{model_label}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    log.info("3D UMAP saved → %s", out)
    plt.close(fig)

    # --- Attempt paired M0/M4 comparison if both npz files exist ---
    _maybe_paired_umap(out_dir, entity_cfg, tier_cmap, tier_cval)


def _draw_umap_ax(
    ax,
    coords: np.ndarray,
    all_tags: list[str],
    n_hunks: int,
    entity_cfg: dict,
    tier_cmap,
    tier_cval: dict[str, float],
    sil: float | None,
    title: str = "",
) -> None:
    z_min = float(coords[:, 2].min()) - 0.5

    # HUNK points: gradient color by tier necessity
    for key in ("1", "2a", "2b", "3"):
        idx = [i for i, k in enumerate(all_tags[:n_hunks]) if k == key]
        if not idx:
            continue
        cx, cy, cz = coords[idx, 0], coords[idx, 1], coords[idx, 2]
        color = tier_cmap(tier_cval[key])
        ax.scatter(cx, cy, cz,
                   c=[color] * len(idx), marker="o", label=_TIER_LABELS[key],
                   alpha=0.7, s=12, depthshade=True, linewidths=0)
        ax.scatter(cx, cy, zs=z_min, zdir="z",
                   c=[color] * len(idx), marker="o", alpha=0.08, s=6, linewidths=0)

    # Entity anchors: REQ (★), TEST (▲), ORIG (■) — distinct marker per type
    for etype, (marker, color, label, sz) in entity_cfg.items():
        idx = [i for i, k in enumerate(all_tags) if k == etype]
        if not idx:
            continue
        cx, cy, cz = coords[idx, 0], coords[idx, 1], coords[idx, 2]
        ax.scatter(cx, cy, cz,
                   c=color, marker=marker, label=label,
                   alpha=0.85, s=sz, edgecolors="black", linewidths=0.4, depthshade=False)
        ax.scatter(cx, cy, zs=z_min, zdir="z",
                   c=color, marker=marker, alpha=0.15, s=sz // 2, linewidths=0)

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")
    ax.set_title(title)
    ax.legend(markerscale=2, loc="upper left", fontsize=8)

    # Silhouette score annotated beside the plot (in axes-fraction coordinates)
    if sil is not None:
        ax.text2D(0.01, 0.97, f"Sil = {sil:.3f}", transform=ax.transAxes,
                  fontsize=9, va="top",
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75))


def _maybe_paired_umap(
    out_dir: Path,
    entity_cfg: dict,
    tier_cmap,
    tier_cval: dict[str, float],
) -> None:
    m0_path = out_dir / "umap_embs_m0.npz"
    m4_path = out_dir / "umap_embs_m4.npz"
    if not (m0_path.exists() and m4_path.exists()):
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.colors as mcolors  # type: ignore
        from matplotlib.cm import ScalarMappable  # type: ignore
        from mpl_toolkits.mplot3d import Axes3D  # type: ignore  # noqa: F401
    except ImportError:
        return

    panels = []
    for panel_label, path in [("M0 (untrained UniXCoder)", m0_path), ("M4 (trained)", m4_path)]:
        d = np.load(path, allow_pickle=True)
        sil_val = float(d["sil"][0])
        panels.append((
            panel_label,
            d["coords"],
            list(d["tags"]),
            int(d["n_hunks"]),
            None if np.isnan(sil_val) else sil_val,
        ))

    fig = plt.figure(figsize=(22, 9))
    axes = []
    for col, (panel_label, coords, tags, n_hunks, sil) in enumerate(panels):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        _draw_umap_ax(ax, coords, tags, n_hunks, entity_cfg, tier_cmap, tier_cval, sil,
                      title=panel_label)
        axes.append(ax)

    sm = ScalarMappable(cmap=tier_cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.45, pad=0.04, aspect=25)
    cbar.set_ticks([0.0, 1 / 3, 2 / 3, 1.0])
    cbar.set_ticklabels(
        ["T1 (test-linked)", "T2a (untested, modifies)", "T2b (untested, new)", "T3 (scope creep)"],
        fontsize=8,
    )
    cbar.set_label("HUNK necessity →", fontsize=9)
    fig.suptitle("4-Entity Embedding Space: M0 vs M4 (UMAP-3D)", fontsize=13)

    out = out_dir / "umap_3d_paired.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    log.info("Paired M0/M4 UMAP saved → %s", out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# nDCG helpers
# ---------------------------------------------------------------------------

def _dcg(relevances: list[float], k: int) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances[:k]))


def _ndcg(relevances: list[float], k: int) -> float:
    ideal = sorted(relevances, reverse=True)
    idcg  = _dcg(ideal, k)
    return _dcg(relevances, k) / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Retrieval (nDCG@k)
# ---------------------------------------------------------------------------

def _instance_metrics(
    scores: list[float],
    tiers: list[int],
    tiebreaks: list[float],
    tier_rel: dict[int, float],
) -> tuple[float | None, float | None, float | None]:
    """(nDCG@k, T2-Recall, PSR) of one instance; None where the metric is undefined for it.

    k = |T1∪T2|.  All three need a Tier-3 candidate (otherwise there is no scope-creep pressure and
    the task is trivially easy); T2-Recall additionally needs a Tier-2 hunk.  ``tiebreaks`` is a random
    number per candidate that breaks score ties, so insertion order cannot favour any tier.
    """
    retained_total = sum(1 for t in tiers if t in (1, 2))
    tier2_total = sum(1 for t in tiers if t == 2)
    if not any(t == 3 for t in tiers):
        return None, None, None

    ranked = sorted(zip(scores, tiers, tiebreaks), key=lambda x: (x[0], x[2]), reverse=True)
    ranked = [(s, t) for s, t, _ in ranked]
    ranked_rel = [tier_rel.get(t, 0.0) for _, t in ranked]

    ndcg = _ndcg(ranked_rel, retained_total)

    # T2-Recall: top-retained_total slots — how many T2 hunks appear?
    t2 = None
    if tier2_total > 0:
        t2 = sum(1 for _, t in ranked[:retained_total] if t == 2) / tier2_total

    # Perfect-Sep: every T3 ranked strictly after every retained hunk
    psr = None
    retained_ranks = [i for i, (_, t) in enumerate(ranked) if t in (1, 2)]
    t3_ranks = [i for i, (_, t) in enumerate(ranked) if t == 3]
    if retained_ranks and t3_ranks:
        psr = 1.0 if min(t3_ranks) > max(retained_ranks) else 0.0
    return ndcg, t2, psr


def retrieval_candidates(
    cfg: Config,
    split: str = "test",
    all_instances: bool = False,
    repo_pool: bool = False,
    repo_pool_max: int = 50,
):
    """Candidate set of the retrieval experiment, shared by EEL (run_retrieval) and every baseline.

    Yields ``(inst, candidates, tiers, tiebreaks)`` for each instance of ``split`` that has a retained and a
    Tier-3 candidate.  ``candidates`` = [(text, tier, hunk_dict, orig_override|None)]: retained (T1/T2),
    Tier-3 and, with ``repo_pool``, up to ``repo_pool_max`` same-repo retained hunks of other instances of the
    split as tier-0 distractors.  ``tiebreaks`` = one random number per candidate that breaks score ties.
    All randomness comes from one ``random.Random(42)`` consumed in a fixed order, so every scorer sees the
    same distractors and the same tie-breaks.
    """
    instances = load_test_instances(cfg, all_instances=all_instances, split=split)
    log.info("%s instances: %d", split.capitalize(), len(instances))

    tier3_lookup = load_tier3_lookup(cfg.eval.tier3_path) if cfg.eval.tier3_path else {}
    if tier3_lookup:
        log.info("Tier-3 lookup: %d instances with scope-creep hunks", len(tier3_lookup))

    # Build repo-level distractor pool: same repo, different instance, retained hunks → tier 0
    # (same source as the T1/T2 candidates, so distractors are not distinguishable by diff format).
    # The pool comes from the instances of the evaluated split only: no leakage from training data.
    # Each distractor keeps the ORIG unit(s) of the instance it comes from: like every candidate it is scored
    # against the original code it modifies (o(h) in the paper), not against the evaluated issue's files.
    repo_hunk_pool: dict[str, list[tuple[str, str, dict, list[str]]]] = {}
    if repo_pool:
        log.info("Building repo-level distractor pool (%s split only) …", split)
        for inst in instances:  # already filtered to the split above
            iid  = inst.get("instance_id", "")
            repo = iid.split("__")[0]
            for hunk in retained_hunks(inst, cfg):
                text = _render_hunk(hunk)
                if text.strip():
                    own_orig = [u["code"] for u in _units_for_hunk(hunk, inst.get("source_units", [])) if u.get("code")]
                    repo_hunk_pool.setdefault(repo, []).append((iid, text, hunk, own_orig))
        log.info("Repo pool: %d repos, %d total hunks",
                 len(repo_hunk_pool), sum(len(v) for v in repo_hunk_pool.values()))

    rng = random.Random(42)
    for inst in instances:
        iid = inst.get("instance_id", "")

        # (text, tier, hunk_dict, orig_override|None)
        # orig_override=None  → the hunk belongs to this instance: use its source_units at score time
        # orig_override=list  → a distractor: the ORIG unit(s) of the instance it comes from
        candidates: list[tuple[str, int, dict, list[str] | None]] = []
        for hunk in retained_hunks(inst, cfg):
            text = _render_hunk(hunk)
            if text.strip():
                candidates.append((text, hunk.get("tier_label") or 1, hunk, None))
        for hunk in tier3_lookup.get(iid, []):
            text = _render_hunk(hunk)
            if text.strip():
                candidates.append((text, 3, hunk, None))

        # Same-repo cross-issue retained hunks as tier-0 distractors.  They get the ORIG unit of their OWN
        # instance, so the ORIG view cannot separate them from the real candidates by mere file membership
        # (a distractor scored against this issue's files would always get ORIG = 0).
        if repo_pool:
            repo = iid.split("__")[0]
            pool = [(t, h, o) for (eid, t, h, o) in repo_hunk_pool.get(repo, []) if eid != iid]
            if len(pool) > repo_pool_max:
                pool = rng.sample(pool, repo_pool_max)
            for text, hunk, own_orig in pool:
                candidates.append((text, 0, hunk, own_orig))

        if len(candidates) < 2:
            continue

        tiers      = [c[1] for c in candidates]
        retained_total = sum(1 for t in tiers if t in (1, 2))

        if retained_total == 0:
            continue

        # One random tiebreak per candidate, shared by all scorers / weight triples of this instance.
        tiebreaks = [rng.random() for _ in tiers]

        # Without a Tier-3 candidate no metric is defined (_instance_metrics), so the instance is not scored.
        # The tiebreaks above are drawn anyway: the random stream, hence every tie-break, is unchanged.
        if 3 not in tiers:
            continue

        yield inst, candidates, tiers, tiebreaks


def run_retrieval(
    cfg: Config,
    encoder: EntailmentEncoder,
    device: torch.device,
    all_instances: bool = False,
    use_projection: bool = True,
    repo_pool: bool = False,
    repo_pool_max: int = 50,
    split: str = "test",
    weight_grid: list[tuple[float, float, float]] | None = None,
) -> list[dict]:
    """Retrieval experiment.  Scores every candidate hunk of every instance of ``split`` and reports
    nDCG@k, T2-Recall and PSR.

    ``weight_grid`` = list of (alpha, beta, gamma): each candidate is encoded once and every weight triple
    is scored from the same similarities (used to select the weights on the validation split).  Default:
    the single triple in ``cfg.eval``.  Returns one dict of mean metrics per triple.
    """
    log.info("=== Experiment 2: Retrieval (%s split) ===", split)

    tier_rel = cfg.eval.tier_relevance
    weights = weight_grid or [(cfg.eval.score_alpha, cfg.eval.score_beta, cfg.eval.score_gamma)]

    # Per weight triple: nDCG@k (k = retained count), T2-Recall and PSR of every instance where defined
    acc = [{"ndcg_k": [], "t1_recall": [], "t2_recall": [], "psr": []} for _ in weights]

    for inst, candidates, tiers, tiebreaks in retrieval_candidates(
            cfg, split=split, all_instances=all_instances, repo_pool=repo_pool, repo_pool_max=repo_pool_max):
        req          = inst.get("requirement", "")[:cfg.data.max_req_chars]
        test_texts   = [tf["code"] for tf in inst.get("test_functions", []) if tf.get("code")]
        source_units = inst.get("source_units", [])

        hunk_texts = [c[0] for c in candidates]
        hunk_orig_texts = [
            c[3] if c[3] is not None
            else [u["code"] for u in _units_for_hunk(c[2], source_units) if u.get("code")]
            for c in candidates
        ]

        sim_req, sim_test, sim_orig = encoder.similarity_components(
            hunk_texts=hunk_texts,
            req_texts=[req] * len(hunk_texts),
            test_texts=[test_texts] * len(hunk_texts),
            orig_texts=hunk_orig_texts,
            device=device,
            use_projection=use_projection,
        )

        for (alpha, beta, gamma), a in zip(weights, acc):
            scores = (alpha * sim_req + beta * sim_test + gamma * sim_orig).tolist()
            ndcg, t2, psr = _instance_metrics(scores, tiers, tiebreaks, tier_rel)
            tier1_total = sum(t == 1 for t in tiers)
            if ndcg is not None and tier1_total:
                retained_total = sum(t in (1, 2) for t in tiers)
                ranked = sorted(zip(scores, tiers, tiebreaks), key=lambda x: (x[0], x[2]), reverse=True)
                a["t1_recall"].append(sum(t == 1 for _, t, _ in ranked[:retained_total]) / tier1_total)
            for key, value in (("ndcg_k", ndcg), ("t2_recall", t2), ("psr", psr)):
                if value is not None:
                    a[key].append(value)

    results = []
    for (alpha, beta, gamma), a in zip(weights, acc):
        results.append({
            "alpha": alpha, "beta": beta, "gamma": gamma,
            **{k: (float(np.mean(v)) if v else None) for k, v in a.items()},
            **{f"n_{k}": len(v) for k, v in a.items()},
        })

    if len(weights) == 1:
        a = acc[0]
        if a["t1_recall"]:
            log.info("T1-Recall(K=Retained-H)  mean=%.4f  std=%.4f  n=%d  (top-K = |T1∪T2|)",
                     np.mean(a["t1_recall"]), np.std(a["t1_recall"]), len(a["t1_recall"]))
        if a["ndcg_k"]:
            log.info("nDCG@k_retained  mean=%.4f  std=%.4f  n=%d  (k = |T1∪T2| per instance)",
                     np.mean(a["ndcg_k"]), np.std(a["ndcg_k"]), len(a["ndcg_k"]))
        if a["t2_recall"]:
            log.info("T2-Recall(K=Retained-H)  mean=%.4f  std=%.4f  n=%d  (top-K = |T1∪T2|)",
                     np.mean(a["t2_recall"]), np.std(a["t2_recall"]), len(a["t2_recall"]))
        if a["psr"]:
            log.info("Perfect-Sep  mean=%.4f  n=%d  (all T3 ranked after all Retained-H)",
                     np.mean(a["psr"]), len(a["psr"]))
    return results



def run_baseline_retrieval(
    cfg: Config,
    score_fn,
    label: str,
    split: str = "test",
    all_instances: bool = False,
    repo_pool: bool = False,
    repo_pool_max: int = 50,
) -> dict:
    """Retrieval metrics of a baseline scorer under exactly the EEL protocol (run_retrieval):
    same candidates (retained + Tier-3 + T0 distractors with ``repo_pool``), same random tie-breaks, same
    metric function.  ``score_fn(req, hunk_texts) -> list[float]``; ``req`` is truncated as for EEL.
    """
    log.info("=== Baseline %s: Retrieval (%s split, repo_pool=%s) ===", label, split, repo_pool)
    tier_rel = cfg.eval.tier_relevance
    acc: dict[str, list[float]] = {"ndcg_k": [], "t1_recall": [], "t2_recall": [], "psr": []}
    for n, (inst, candidates, tiers, tiebreaks) in enumerate(retrieval_candidates(
            cfg, split=split, all_instances=all_instances, repo_pool=repo_pool, repo_pool_max=repo_pool_max), 1):
        req = inst.get("requirement", "")[:cfg.data.max_req_chars]
        scores = [float(x) for x in score_fn(req, [c[0] for c in candidates])]
        ndcg, t2, psr = _instance_metrics(scores, tiers, tiebreaks, tier_rel)
        tier1_total = sum(t == 1 for t in tiers)
        if ndcg is not None and tier1_total:
            retained_total = sum(t in (1, 2) for t in tiers)
            ranked = sorted(zip(scores, tiers, tiebreaks), key=lambda x: (x[0], x[2]), reverse=True)
            acc["t1_recall"].append(sum(t == 1 for _, t, _ in ranked[:retained_total]) / tier1_total)
        for key, value in (("ndcg_k", ndcg), ("t2_recall", t2), ("psr", psr)):
            if value is not None:
                acc[key].append(value)
        if n % 10 == 0:
            log.info("  scored %d instances", n)

    results = {
        "model":       label,
        "split":       split,
        "repo_pool":   repo_pool,
        "n_instances": len(acc["ndcg_k"]),
        "ndcg_k":      float(np.mean(acc["ndcg_k"])) if acc["ndcg_k"] else None,
        "t1_recall":   float(np.mean(acc["t1_recall"])) if acc["t1_recall"] else None,
        "n_t1_recall": len(acc["t1_recall"]),
        "t2_recall":   float(np.mean(acc["t2_recall"])) if acc["t2_recall"] else None,
        "n_t2_recall": len(acc["t2_recall"]),
        "perfect_sep": float(np.mean(acc["psr"])) if acc["psr"] else None,
    }
    log.info("=== %s ===", label)
    log.info("  nDCG@k       mean=%s  n=%d", results["ndcg_k"], results["n_instances"])
    log.info("  T1-Recall    mean=%s  n=%d", results["t1_recall"], results["n_t1_recall"])
    log.info("  T2-Recall    mean=%s  n=%d", results["t2_recall"], results["n_t2_recall"])
    log.info("  Perfect-Sep  mean=%s  n=%d", results["perfect_sep"], len(acc["psr"]))
    return results

# ---------------------------------------------------------------------------
# Score weights (alpha, beta, gamma): selected on the validation split
# ---------------------------------------------------------------------------

def select_weights(
    cfg: Config,
    encoder: EntailmentEncoder,
    device: torch.device,
    use_projection: bool = True,
    repo_pool: bool = False,
) -> dict:
    """Pick (alpha, beta, gamma) on the VALIDATION split and store them next to the checkpoint.

    Every candidate is encoded once; all triples of ``data.score_weights.weight_grid()`` are scored from the
    same similarities.  The test split is not touched.  Later evaluations of this checkpoint read the file.
    """
    log.info("=== Selecting score weights on the validation split ===")
    grid = weight_grid()
    results = run_retrieval(cfg, encoder, device, use_projection=use_projection, repo_pool=repo_pool,
                            split="val", weight_grid=grid)
    best = select_best(results)
    if best["ndcg_k"] is None:
        raise RuntimeError("No validation instance has both a retained and a Tier-3 hunk; pass --tier3 "
                           "(path to tier3_hunks.jsonl) so there is something to rank.")
    for r in sorted(results, key=lambda r: -(r["ndcg_k"] or -1))[:5]:
        log.info("  alpha=%.2f beta=%.2f gamma=%.2f  nDCG@k=%.4f  PSR=%s  T2-Recall=%s", r["alpha"], r["beta"], r["gamma"],
                 r["ndcg_k"], "n/a" if r["psr"] is None else f"{r['psr']:.4f}",
                 "n/a" if r["t2_recall"] is None else f"{r['t2_recall']:.4f}")
    path = save_selected_weights(
        cfg.eval.checkpoint_path, getattr(encoder, "checkpoint_identity", None), best, results,
        {"repo_pool": repo_pool, "tier3_file": cfg.eval.tier3_path, "retained_file": cfg.data.llm_t12_path,
         "n_val_instances": best["n_ndcg_k"]},
    )
    log.info("Selected alpha=%.2f beta=%.2f gamma=%.2f (val nDCG@k=%.4f, n=%d) → %s",
             best["alpha"], best["beta"], best["gamma"], best["ndcg_k"], best["n_ndcg_k"], path)
    return best


def resolve_score_weights(cfg: Config, encoder: EntailmentEncoder, use_projection: bool, explicit: bool) -> str:
    """Set cfg.eval.score_* for this run and say where they come from.

    explicit (--score-* given): ablations, used as given.  Otherwise the weights selected on the validation
    split for THIS checkpoint (score_weights.json); a file that belongs to another checkpoint is an error.
    """
    if explicit:
        return "explicit --score-* flags"
    if not use_projection:
        return "config defaults (untrained M0 backbone)"
    sel = load_selected_weights(cfg.eval.checkpoint_path)
    if sel is None:
        log.warning("No score_weights.json beside %s: using the config defaults, which were NOT selected on the "
                    "validation split. Run `evaluate.py --select-weights` first (paper §III-D).", cfg.eval.checkpoint_path)
        return "config defaults, NOT selected on validation"
    if sel.get("checkpoint_identity") != getattr(encoder, "checkpoint_identity", None):
        raise ValueError(f"{cfg.eval.checkpoint_path}: score_weights.json was selected for another checkpoint "
                         f"({sel.get('checkpoint_identity')}); rerun `evaluate.py --select-weights`.")
    cfg.eval.score_alpha, cfg.eval.score_beta, cfg.eval.score_gamma = sel["alpha"], sel["beta"], sel["gamma"]
    return f"selected on the validation split ({sel['criterion']})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    cfg: Config,
    exp: str,
    all_instances: bool = False,
    use_projection: bool = True,
    collect_umap: bool = False,
    repo_pool: bool = False,
    save_scores_path: str | None = None,
    allow_legacy: bool = False,
    split: str = "test",
    select_only: bool = False,
    explicit_weights: bool = False,
) -> None:
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    log.info("Device: %s", device)
    log.info("use_projection: %s", use_projection)
    log.info(describe_hunk_files(cfg.data.llm_t12_path, cfg.eval.tier3_path or None))

    encoder = load_encoder(cfg, device, use_projection, allow_legacy=allow_legacy)

    if select_only:
        if not use_projection:
            raise SystemExit("--select-weights needs a trained checkpoint (not --no-projection)")
        select_weights(cfg, encoder, device, use_projection, repo_pool)
        return
    source = resolve_score_weights(cfg, encoder, use_projection, explicit_weights)
    log.info("Score weights: alpha=%.3g beta=%.3g gamma=%.3g  (%s)",
             cfg.eval.score_alpha, cfg.eval.score_beta, cfg.eval.score_gamma, source)

    # Derive a short label for UMAP file naming (m0 for untrained baseline).
    if not use_projection:
        model_label = "m0"
    else:
        ckpt = cfg.eval.checkpoint_path
        p = Path(ckpt)
        parent = p.parent.name
        # "checkpoints/m4/best.pt" → "m4"; legacy "checkpoints/best.pt" → "trained"
        model_label = parent if parent != "checkpoints" else "trained"

    if exp in ("geometry", "all"):
        run_geometry(
            cfg, encoder, device,
            all_instances=all_instances,
            use_projection=use_projection,
            collect_umap=collect_umap,
            model_label=model_label,
            save_scores_path=save_scores_path,
        )

    if exp in ("retrieval", "all"):
        run_retrieval(cfg, encoder, device,
                      all_instances=all_instances,
                      use_projection=use_projection,
                      repo_pool=repo_pool,
                      split=split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Edit Entailment encoder")
    parser.add_argument(
        "--exp", choices=["geometry", "retrieval", "all"],
        default="all", help="Which experiment to run",
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path")
    parser.add_argument("--instances", default=None, help="Instances JSONL path")
    parser.add_argument("--encoder", default=None, help="HuggingFace model name")
    parser.add_argument("--tier3", default=None, help="Path to tier3_hunks.jsonl (scope-creep candidates)")
    parser.add_argument("--all-instances", action="store_true", help="Evaluate on all instances, not just test split")
    parser.add_argument("--no-projection", action="store_true", help="Skip MLP projection head (M0 untrained baseline)")
    parser.add_argument("--umap", action="store_true", help="Collect hunk embeddings and save UMAP plot (requires umap-learn)")
    parser.add_argument("--score-alpha", type=float, default=None, help="Override score_alpha")
    parser.add_argument("--score-beta",  type=float, default=None, help="Override score_beta")
    parser.add_argument("--score-gamma", type=float, default=None, help="Override score_gamma")
    parser.add_argument("--repo-pool", action="store_true",
                        help="Add same-repo cross-issue retained hunks as tier-0 distractors")
    add_retained_args(parser)
    parser.add_argument("--split", choices=["val", "test"], default="test",
                        help="Split the retrieval experiment runs on (default: test; val is for diagnostics)")
    parser.add_argument("--select-weights", action="store_true",
                        help="Select alpha/beta/gamma on the validation split, write score_weights.json next to the "
                             "checkpoint and exit (uses the same candidates as --exp retrieval; pass --repo-pool and --tier3 as for the test run)")
    parser.add_argument("--allow-legacy-checkpoint", action="store_true",
                        help="Load a checkpoint without a data_provenance stamp (archival diagnostics only; such models may have been trained on the reference patch's own hunks)")
    parser.add_argument("--save-hunk-scores", default=None, metavar="PATH",
                        help="Save per-hunk (instance_id, hunk_id, hunk_key, tier, score, sim_req, sim_test, sim_orig) to JSONL")
    args = parser.parse_args()

    cfg = default_config
    apply_retained_args(cfg, args)
    if args.checkpoint:
        cfg.eval.checkpoint_path = args.checkpoint
    if args.instances:
        cfg.data.instances_path = args.instances
    if args.encoder:
        cfg.model.encoder_name = args.encoder
    if args.tier3:
        cfg.eval.tier3_path = args.tier3
    if args.score_alpha is not None:
        cfg.eval.score_alpha = args.score_alpha
    if args.score_beta is not None:
        cfg.eval.score_beta = args.score_beta
    if args.score_gamma is not None:
        cfg.eval.score_gamma = args.score_gamma

    use_projection = not args.no_projection
    main(
        cfg, args.exp,
        all_instances=args.all_instances,
        use_projection=use_projection,
        collect_umap=args.umap,
        repo_pool=args.repo_pool,
        save_scores_path=args.save_hunk_scores,
        allow_legacy=args.allow_legacy_checkpoint,
        split=args.split,
        select_only=args.select_weights,
        explicit_weights=any(w is not None for w in (args.score_alpha, args.score_beta, args.score_gamma)),
    )
