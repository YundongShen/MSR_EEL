"""Exp2: Edit-type stratified analysis of entailment score components.

Join per-hunk component scores (sim_req, sim_test, sim_orig) with Gemini-assigned
edit-type labels (BEHAVIORAL / SEMANTIC / STRUCTURAL) to answer:

  Which entailment signal component is relatively stronger for each edit type?
  BEHAVIORAL → sim_test higher than other types
  SEMANTIC   → sim_req  higher than other types
  STRUCTURAL → sim_orig higher than other types

Two analyses:
  A — Combination breakdown: stats for every label combo (pure + mixed)
  B — Cross-type comparison: for each component, Mann-Whitney across edit-type groups
      (pure single-label hunks only, to avoid label leakage)

Usage:
    python analysis/exp2_analysis.py \\
        --scores  logs/exp2_hunk_scores_m2.jsonl \\
        --labels  data/processed/hunk_edit_types.jsonl \\
        --output  logs/exp2_analysis.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EDIT_TYPES = ["BEHAVIORAL", "SEMANTIC", "STRUCTURAL"]
COMPONENTS = ["sim_req", "sim_test", "sim_orig"]
COMP_SHORT = {"sim_req": "REQ", "sim_test": "TEST", "sim_orig": "ORIG"}


def _key(rec: dict, path: str) -> tuple[str, str]:
    """Join key = (instance_id, hunk_key).  hunk_id is a list index and changes when the dataset is rebuilt,
    so files without hunk_key (made from an earlier dataset build) are refused rather than mis-joined."""
    if not rec.get("hunk_key"):
        raise ValueError(f"{path}: record without hunk_key; regenerate it with the current evaluate.py / "
                         "analysis/exp2_label_hunks.py (it was made from an earlier dataset build).")
    return (rec["instance_id"], rec["hunk_key"])


def load_scores(path: str) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[_key(rec, path)] = rec
    return out


def load_labels(path: str) -> dict[tuple, list[str]]:
    out: dict[tuple, list[str]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[_key(rec, path)] = rec.get("labels", [])
    return out


def _rank_biserial(u: float, n1: int, n2: int) -> float:
    return (2.0 * u - n1 * n2) / (n1 * n2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp2: edit-type component analysis")
    parser.add_argument("--scores",  default="logs/exp2_hunk_scores_m2.jsonl")
    parser.add_argument("--labels",  default="data/processed/hunk_edit_types.jsonl")
    parser.add_argument("--output",  default="logs/exp2_analysis.json")
    parser.add_argument("--plot",    default="logs/exp2_component_violin.png")
    args = parser.parse_args()

    scores = load_scores(args.scores)
    labels = load_labels(args.labels)
    log.info("Scores: %d hunks  |  Labels: %d hunks", len(scores), len(labels))

    joined: list[dict] = []
    for key, lbl_list in labels.items():
        if key not in scores:
            continue
        rec = scores[key]
        if rec["tier"] not in (1, 2):
            continue
        has_orig = rec.get("has_orig", rec["eff_tier"] != "2b")
        # delta_orig = sim_orig(matched) - sim_orig(background instance-level mean).
        # This removes the ~0.87 ceiling bias from context-line text overlap,
        # leaving only the specific "did this hunk modify this ORIG unit" signal.
        # T2b (no ORIG match): sim_orig=0, delta_orig = -sim_orig_bg (negative)
        delta_orig = rec.get("delta_orig", rec["sim_orig"])  # fallback for old data
        entry = {
            "instance_id": rec["instance_id"],
            "hunk_key":    rec["hunk_key"],
            "hunk_id":     rec["hunk_id"],
            "tier":        rec["tier"],
            "eff_tier":    rec["eff_tier"],
            "has_orig":    has_orig,
            "labels":      sorted(lbl_list),
            "combo":       "+".join(sorted(lbl_list)),
            "score":       rec["score"],
            "sim_req":     rec["sim_req"],
            "sim_test":    rec["sim_test"],
            "sim_orig":    rec["sim_orig"],
            "delta_orig":  delta_orig,
        }
        total = rec["sim_req"] + rec["sim_test"] + rec["sim_orig"]
        if total > 0:
            entry["rel_req"]  = rec["sim_req"]  / total
            entry["rel_test"] = rec["sim_test"] / total
            entry["rel_orig"] = rec["sim_orig"] / total
        else:
            entry["rel_req"] = entry["rel_test"] = entry["rel_orig"] = 1/3
        joined.append(entry)

    log.info("Joined Retained-H with labels: %d", len(joined))
    results: dict = {"n_joined": len(joined)}

    # -----------------------------------------------------------------------
    # Analysis A — Combination breakdown (every label combo)
    # -----------------------------------------------------------------------
    combo_groups: dict[str, list[dict]] = defaultdict(list)
    for rec in joined:
        combo_groups[rec["combo"]].append(rec)

    log.info("\n=== Analysis A: Label combination breakdown ===")
    log.info("%-34s  %5s  %7s  %7s  %7s  %7s  %7s  %7s",
             "Combo", "n", "sim_req", "sim_test", "sim_orig", "rel_req", "rel_test", "rel_orig")
    combo_stats: dict[str, dict] = {}
    for combo in sorted(combo_groups, key=lambda c: -len(combo_groups[c])):
        recs = combo_groups[combo]
        row: dict = {"n": len(recs)}
        parts_abs, parts_rel = [], []
        for comp, rel in [("sim_req","rel_req"), ("sim_test","rel_test"), ("sim_orig","rel_orig")]:
            vals = [r[comp] for r in recs]
            rvals = [r[rel] for r in recs]
            row[comp] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}
            row[rel]  = {"mean": round(float(np.mean(rvals)), 4)}
            parts_abs.append(f"{np.mean(vals):7.4f}")
            parts_rel.append(f"{np.mean(rvals):7.4f}")
        log.info("%-34s  %5d  %s  %s", combo, len(recs), "  ".join(parts_abs), "  ".join(parts_rel))
        combo_stats[combo] = row
    results["combo_stats"] = combo_stats

    # -----------------------------------------------------------------------
    # Analysis B — Cross-type comparison (pure single-label hunks only)
    # -----------------------------------------------------------------------
    pure: dict[str, list[dict]] = {
        et: [r for r in joined if r["combo"] == et] for et in EDIT_TYPES
    }
    log.info("\n=== Analysis B: Cross-type comparison (pure single-label hunks) ===")
    for et in EDIT_TYPES:
        n_total = len(pure[et])
        n_has_orig = sum(1 for r in pure[et] if r["has_orig"])
        log.info("  %s: n=%d  (has_orig=%d  no_orig=%d)", et, n_total, n_has_orig, n_total - n_has_orig)

    log.info("\n-- Absolute component means (pure hunks, all) --")
    log.info("%-12s  %8s  %8s  %8s  %7s", "EditType", "sim_req", "sim_test", "sim_orig", "n")
    for et in EDIT_TYPES:
        if not pure[et]:
            continue
        vals = {c: [r[c] for r in pure[et]] for c in COMPONENTS}
        log.info("%-12s  %8.4f  %8.4f  %8.4f  %7d",
                 et, np.mean(vals["sim_req"]), np.mean(vals["sim_test"]),
                 np.mean(vals["sim_orig"]), len(pure[et]))

    log.info("\n-- Absolute component means (pure hunks, has_orig=True only) --")
    log.info("%-12s  %8s  %8s  %8s  %7s", "EditType", "sim_req", "sim_test", "sim_orig", "n")
    for et in EDIT_TYPES:
        sub = [r for r in pure[et] if r["has_orig"]]
        if not sub:
            continue
        vals = {c: [r[c] for r in sub] for c in COMPONENTS}
        log.info("%-12s  %8.4f  %8.4f  %8.4f  %7d",
                 et, np.mean(vals["sim_req"]), np.mean(vals["sim_test"]),
                 np.mean(vals["sim_orig"]), len(sub))

    log.info("\n-- Relative component means (pure hunks, has_orig=True) --")
    log.info("%-12s  %8s  %8s  %8s", "EditType", "rel_req", "rel_test", "rel_orig")
    pure_rel_stats: dict[str, dict] = {}
    for et in EDIT_TYPES:
        sub = [r for r in pure[et] if r["has_orig"]]
        if not sub:
            continue
        rr = [r["rel_req"]  for r in sub]
        rt = [r["rel_test"] for r in sub]
        ro = [r["rel_orig"] for r in sub]
        log.info("%-12s  %8.4f  %8.4f  %8.4f  n=%d", et, np.mean(rr), np.mean(rt), np.mean(ro), len(sub))
        pure_rel_stats[et] = {
            "rel_req":  round(float(np.mean(rr)), 4),
            "rel_test": round(float(np.mean(rt)), 4),
            "rel_orig": round(float(np.mean(ro)), 4),
            "n": len(sub),
        }
    results["pure_relative_stats"] = pure_rel_stats

    # Cross-type Mann-Whitney U tests.
    # delta_orig = sim_orig(matched) - sim_orig(instance-level background).
    # This removes the ~0.87 ceiling from context-line text overlap and measures
    # only the specific ORIG-match signal.  All pure hunks included (not just has_orig=True)
    # so T2b (delta_orig<0) are included and push STRUCTURAL down.
    cross_tests = [
        # ORIG signal: BEHAVIORAL/SEMANTIC > STRUCTURAL (they modify existing code)
        ("BEHAVIORAL", "STRUCTURAL", "delta_orig", None,
         "BEHAVIORAL has stronger specific ORIG signal than STRUCTURAL"),
        ("SEMANTIC",   "STRUCTURAL", "delta_orig", None,
         "SEMANTIC has stronger specific ORIG signal than STRUCTURAL"),
        # TEST signal: SEMANTIC > BEHAVIORAL/STRUCTURAL
        ("SEMANTIC",   "BEHAVIORAL", "sim_test",  None,
         "SEMANTIC has higher TEST signal than BEHAVIORAL"),
        ("SEMANTIC",   "STRUCTURAL", "sim_test",  None,
         "SEMANTIC has higher TEST signal than STRUCTURAL"),
        # REQ signal: BEHAVIORAL > SEMANTIC
        ("BEHAVIORAL", "SEMANTIC",   "sim_req",   None,
         "BEHAVIORAL has higher REQ signal than SEMANTIC"),
    ]

    log.info("\n-- Cross-type Mann-Whitney U (pure hunks) --")
    log.info("  delta_orig = sim_orig(matched) - sim_orig(instance_bg)  [removes context-line ceiling]")
    cross_results: list[dict] = []
    for type_a, type_b, comp, filter_fn, desc in cross_tests:
        a_recs = pure[type_a] if filter_fn is None else [r for r in pure[type_a] if filter_fn(r)]
        b_recs = pure[type_b] if filter_fn is None else [r for r in pure[type_b] if filter_fn(r)]
        a_vals = [r[comp] for r in a_recs]
        b_vals = [r[comp] for r in b_recs]
        if len(a_vals) < 2 or len(b_vals) < 2:
            log.info("  SKIP (n_a=%d, n_b=%d): %s", len(a_vals), len(b_vals), desc)
            continue
        u, p = mannwhitneyu(a_vals, b_vals, alternative="greater")
        r = _rank_biserial(u, len(a_vals), len(b_vals))
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        a_mean = float(np.mean(a_vals))
        b_mean = float(np.mean(b_vals))
        log.info("  %s  mean_a=%.3f mean_b=%.3f  p=%.3e  r=%.3f  %s",
                 f"{COMP_SHORT.get(comp,comp)}({type_a}>{type_b})",
                 a_mean, b_mean, p, r, sig)
        cross_results.append({
            "hypothesis": desc,
            "type_a": type_a, "type_b": type_b, "component": comp,
            "mean_a": round(a_mean, 4), "mean_b": round(b_mean, 4),
            "n_a": len(a_vals), "n_b": len(b_vals),
            "u": float(u), "p": float(p), "r": round(r, 4), "sig": sig,
        })
    results["cross_type_tests"] = cross_results

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    log.info("\nResults saved → %s", out_path)

    if args.plot:
        _plot(pure, combo_groups, args.plot)


def _plot(
    pure: dict[str, list[dict]],
    combo_groups: dict[str, list[dict]],
    plot_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        log.info("matplotlib not available — skipping plot")
        return

    comp_colors = {"rel_req": "#2196F3", "rel_test": "#4CAF50", "rel_orig": "#FF9800"}
    comp_labels = {"rel_req": "REQ", "rel_test": "TEST", "rel_orig": "ORIG"}
    rel_comps = ["rel_req", "rel_test", "rel_orig"]

    # Left panel: relative component weight by pure edit type (violin)
    # Right panel: sim_orig mean by combo (bar)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: relative weights per pure type ---
    ax = axes[0]
    positions = np.arange(len(EDIT_TYPES))
    width = 0.22
    offsets = [-width, 0, width]
    for offset, rel_comp in zip(offsets, rel_comps):
        means = []
        errs = []
        for et in EDIT_TYPES:
            vals = [r[rel_comp] for r in pure[et]] if pure[et] else [0.0]
            means.append(np.mean(vals))
            errs.append(np.std(vals) / max(len(vals)**0.5, 1))
        ax.bar(positions + offset, means, width=width * 0.9,
               color=comp_colors[rel_comp], label=comp_labels[rel_comp],
               yerr=errs, capsize=3, alpha=0.85)
    ax.set_xticks(positions)
    ax.set_xticklabels(EDIT_TYPES)
    ax.set_ylabel("Relative component weight")
    ax.set_title("Component weights by pure edit type\n(normalized per hunk, single-label only)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # --- Right: sim_orig breakdown by combo ---
    ax2 = axes[1]
    sorted_combos = sorted(combo_groups, key=lambda c: -len(combo_groups[c]))
    combo_means = [np.mean([r["sim_orig"] for r in combo_groups[c]]) for c in sorted_combos]
    combo_ns    = [len(combo_groups[c]) for c in sorted_combos]
    colors = ["#FF9800" if "STRUCTURAL" in c else "#90A4AE" for c in sorted_combos]
    bars = ax2.barh(range(len(sorted_combos)), combo_means, color=colors, alpha=0.85)
    ax2.set_yticks(range(len(sorted_combos)))
    ax2.set_yticklabels([f"{c}  (n={n})" for c, n in zip(sorted_combos, combo_ns)], fontsize=9)
    ax2.set_xlabel("Mean sim_orig")
    ax2.set_title("sim_orig by label combination")
    ax2.grid(axis="x", alpha=0.3)
    ax2.invert_yaxis()

    fig.tight_layout()
    out = Path(plot_path)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    log.info("Plot saved → %s", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
