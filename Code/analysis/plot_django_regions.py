"""Generate combined figure:
  Left  — sim_req vs sim_orig scatter (T1/T2/T3 tiers, R1-R5 regions)
  Right — 2D UMAP of M2 embedding space (entity-type coloring)

Reads from logs/django_viz_embs.npz (produced by the encode phase).
No GPU needed — pure CPU.

Usage:
    python analysis/plot_django_regions.py [--embs logs/django_viz_embs.npz]
                                           [--out-prefix logs/django]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

# Scatter: two semantic groups — accepted boundary edits vs rejected scope creep
T_COLOR = {
    "1": "#0072B2",   # Wong blue  — accepted boundary edit
    "2": "#0072B2",   # Wong blue  — accepted boundary edit
    "3": "#D55E00",   # Wong vermilion — scope creep (colorblind-safe vs blue)
}
T_LABEL = {
    "retained":  "Retained-H",
    "creep": "Unaccepted hunk",
}

# UMAP: coloured by entity type
E_COLOR = {
    "ORIG": "#d4d4d4",  # light gray  — codebase background
    "REQ":  "#0072B2",  # Wong blue
    "TEST": "#009E73",  # Wong blue-green (teal)
    "HUNK": "#E69F00",  # Wong yellow-orange
}
E_LABEL = {
    "ORIG": "ORIG — codebase source units",
    "REQ":  "REQ — requirement",
    "TEST": "TEST — fail-to-pass test",
    "HUNK": "HUNK — edit candidate",
}

# ---------------------------------------------------------------------------
# Region helpers
# ---------------------------------------------------------------------------

THR_REQ  = 0.65
THR_TEST = 0.40
THR_ORIG = 0.80


def assign_region(sr: float, st: float, so: float) -> str:
    hi_req  = sr > THR_REQ
    hi_test = st > THR_TEST
    hi_orig = so > THR_ORIG
    if hi_req and hi_test:
        return "R1"
    if hi_req and not hi_test and hi_orig:
        return "R2"
    if hi_req and not hi_test and not hi_orig:
        return "R3"
    if not hi_req and hi_orig:
        return "R4"
    return "R5"


REGION_CENTER = {
    "R1": (0.80, 0.65),
    "R2": (0.78, 0.90),
    "R3": (0.75, 0.45),
    "R4": (0.25, 0.88),
    "R5": (0.00, -0.08),
}

REGION_DESC = {
    "R1": "R1: REQ+TEST\n(direct fix)",
    "R2": "R2: REQ+ORIG\n(structural match)",
    "R3": "R3: REQ only",
    "R4": "R4: ORIG only\n(structural necessity)",
    "R5": "R5: none\n(scope creep)",
}

# annotation offsets (dx, dy) in data coordinates
REGION_ANNOT = {
    "R1": ( 0.00, -0.18),
    "R2": ( 0.06,  0.06),
    "R3": (-0.22,  0.07),
    "R4": (-0.22,  0.06),
    "R5": ( 0.06,  0.08),
}


def pick_example(indices, sreqs, sorigs, target_sr, target_so):
    """Return index of the point in `indices` closest to (target_sr, target_so)."""
    best_i, best_d = None, float("inf")
    for i in indices:
        d = (sreqs[i] - target_sr) ** 2 + (sorigs[i] - target_so) ** 2
        if d < best_d:
            best_d, best_i = d, i
    return best_i


# ---------------------------------------------------------------------------
# KDE helper (for REQ and TEST only — small entity types)
# ---------------------------------------------------------------------------

def kde_region_2d(ax, pts2d, xlim, ylim, color,
                  alpha_fill=0.20, alpha_line=0.60, density_pct=70, bw=0.35):
    if len(pts2d) < 5:
        return
    xmin, xmax = xlim
    ymin, ymax = ylim
    xx, yy = np.mgrid[xmin:xmax:160j, ymin:ymax:160j]
    try:
        kde = gaussian_kde(pts2d.T, bw_method=bw)
        Z = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        thr = np.percentile(Z, density_pct)
        ax.contourf(xx, yy, Z, levels=[thr, Z.max()],
                    colors=[color], alpha=alpha_fill, zorder=2)
        ax.contour(xx, yy, Z, levels=[thr],
                   colors=[color], linewidths=1.2, alpha=alpha_line, zorder=3)
    except Exception as e:
        print(f"  [kde warn] {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embs",       default="logs/django_viz_embs.npz")
    parser.add_argument("--out-prefix",   default="logs/django")
    parser.add_argument("--scatter-only", action="store_true",
                        help="Skip UMAP, generate scatter figure only")
    args = parser.parse_args()

    d = np.load(args.embs, allow_pickle=True)
    tags   = list(d["tags2"])
    iids   = list(d["meta_iid"].tolist())
    sreqs  = d["meta_sreq"].tolist()
    stests = d["meta_stest"].tolist()
    sorigs = d["meta_sorig"].tolist()
    h2     = d["h2"]
    req2   = d["req2"]
    test2  = d["test2"]
    orig2  = d["orig2"]
    n      = len(tags)

    print(f"Loaded: {n} hunks, {len(req2)} REQ, {len(test2)} TEST, {len(orig2):,} ORIG")

    # Assign regions
    regions = [assign_region(sreqs[i], stests[i], sorigs[i]) for i in range(n)]
    from collections import Counter
    rc = Counter(regions)
    for rg, cnt in sorted(rc.items()):
        tier_d = Counter(tags[i] for i in range(n) if regions[i] == rg)
        print(f"  {rg}: n={cnt}  tiers={dict(tier_d)}")

    # One representative per region
    examples: dict[str, int] = {}
    for rg, (tsr, tso) in REGION_CENTER.items():
        cands = [i for i in range(n) if regions[i] == rg]
        if cands:
            examples[rg] = pick_example(cands, sreqs, sorigs, tsr, tso)

    # -----------------------------------------------------------------------
    # 2D UMAP (CPU — may take a while with large orig2)
    # -----------------------------------------------------------------------
    if args.scatter_only:
        umap_lib = None
    else:
        try:
            import umap as umap_lib
        except ImportError:
            print("umap-learn not installed — skipping UMAP panel")
            umap_lib = None

    if umap_lib is not None:
        # Fit on HUNK + ORIG(5k) so the manifold covers both text types.
        # REQ and TEST are transformed in (small counts, no distortion).
        rng_np = np.random.default_rng(42)
        orig2_sub = orig2[rng_np.choice(len(orig2), min(5000, len(orig2)), replace=False)]
        n_h2 = len(h2)
        fit_data = np.vstack([h2, orig2_sub])
        print(f"\nFitting 2D UMAP on HUNK+ORIG_5k ({len(fit_data):,} points) …")
        r2d = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=50, min_dist=0.4)
        fit_coords = r2d.fit_transform(fit_data)
        ch2_2d    = fit_coords[:n_h2]
        # Transform REQ, TEST, and full ORIG
        creq2_2d  = r2d.transform(req2)
        ctest2_2d = r2d.transform(test2)
        print(f"  Transforming {len(orig2):,} ORIG vectors …")
        corig2_2d = r2d.transform(orig2)
    else:
        ch2_2d = creq2_2d = ctest2_2d = corig2_2d = None

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------
    fig, ax_scat = plt.subplots(1, 1, figsize=(8, 6.5))
    ax_umap = None

    # ===========================
    # LEFT — sim_req vs sim_orig
    # ===========================

    # Quadrant shading
    ax_scat.axhspan(THR_ORIG, 1.06, color="#e8f5e9", alpha=0.45, zorder=0)
    ax_scat.axvspan(THR_REQ,  1.05, color="#e3f2fd", alpha=0.45, zorder=0)
    ax_scat.axvline(THR_REQ,  color="#999", lw=1.0, ls="--", zorder=1)
    ax_scat.axhline(THR_ORIG, color="#999", lw=1.0, ls="--", zorder=1)

    # Region labels drawn next to each star example (after examples dict is built)

    # Scatter — matplotlib cmap per tier so gradient is visually clear
    # Greens cmap for accepted edits, Reds for unaccepted.
    # sim_test mapped to [0.25, 1.0] so even the lowest value keeps some hue.
    stest_arr = np.array(stests)
    stest_lo, stest_hi = stest_arr.min(), stest_arr.max()
    TIER_CMAP = {"1": "Blues", "2": "Blues", "3": "Oranges"}

    for tier in ["3", "2", "1"]:   # T3 underneath, T2 on top
        idx = [i for i in range(n) if tags[i] == tier]
        if not idx:
            continue
        st_norm = 0.25 + 0.75 * (
            (np.array([stests[i] for i in idx]) - stest_lo) / (stest_hi - stest_lo + 1e-8)
        )
        ax_scat.scatter(
            [sreqs[i] for i in idx], [sorigs[i] for i in idx],
            c=st_norm, cmap=TIER_CMAP[tier], vmin=0, vmax=1,
            s=40, alpha=0.80,
            edgecolors=T_COLOR[tier], linewidths=0.4,
            zorder=3,
        )

    # Example stars + region label placed just above each star
    LABEL_OFFSET = {   # (dx, dy) in data coords
        "R1": ( 0.01,  0.03),
        "R2": ( 0.01,  0.03),
        "R3": ( 0.01,  0.03),
        "R4": (-0.06,  0.03),
        "R5": ( 0.01,  0.03),
    }
    for rg, idx in examples.items():
        col = T_COLOR.get(tags[idx], "#555")
        ax_scat.scatter(sreqs[idx], sorigs[idx],
                        s=300, c="white", marker="*",
                        edgecolors="#222222", linewidths=1.5, zorder=6)
        dx, dy = LABEL_OFFSET.get(rg, (0.02, 0.06))
        ax_scat.text(sreqs[idx] + dx, sorigs[idx] + dy, rg,
                     fontsize=8.5, color="#333", fontweight="bold",
                     style="italic", zorder=7)

    # Legend
    tier_handles = [
        mpatches.Patch(color=T_COLOR["2"], label=T_LABEL["retained"]),
        mpatches.Patch(color=T_COLOR["3"], label=T_LABEL["creep"]),
    ]
    gradient_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#08306b", markeredgecolor="#0072B2",
               markersize=7, label="high TEST sim (dark)"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#c6dbef", markeredgecolor="#0072B2",
               markersize=7, label="low TEST sim (pale)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="white",
               markeredgecolor="#222222", markeredgewidth=1.2,
               markersize=11, label="region example"),
    ]
    ax_scat.legend(handles=tier_handles + gradient_handles,
                   loc="lower left", fontsize=7.5, framealpha=0.92,
                   edgecolor="#ccc")

    ax_scat.set_xlabel("sim(hunk, REQ) — semantic alignment with requirement", fontsize=10)
    ax_scat.set_ylabel("sim(hunk, ORIG) — structural grounding in codebase", fontsize=10)
    ax_scat.set_title("Learned Edit Geometry (M2, Django)", fontsize=11)
    ax_scat.set_xlim(-0.40, 1.02)
    ax_scat.set_ylim(-0.50, 1.08)
    ax_scat.grid(True, alpha=0.20, lw=0.5)

    # ===========================
    # RIGHT — 2D UMAP
    # ORIG = hexbin background (codebase extent).
    # REQ / TEST = KDE contours (anchors).
    # HUNK = scatter coloured by tier (T2 retained vs T3 scope creep).
    # ===========================
    if ax_umap is not None and ch2_2d is not None:
        all2d = np.vstack([ch2_2d, creq2_2d, ctest2_2d, corig2_2d])
        xlim  = (all2d[:, 0].min() - 1.5, all2d[:, 0].max() + 1.5)
        ylim  = (all2d[:, 1].min() - 1.5, all2d[:, 1].max() + 1.5)

        # ORIG — tighter KDE showing only the dense core region
        kde_region_2d(ax_umap, corig2_2d, xlim, ylim, E_COLOR["ORIG"],
                      alpha_fill=0.35, alpha_line=0.55, density_pct=60, bw=0.6)

        # TEST, REQ — tighter KDE contours (anchors)
        kde_region_2d(ax_umap, ctest2_2d, xlim, ylim, E_COLOR["TEST"],
                      alpha_fill=0.28, alpha_line=0.70, density_pct=55, bw=0.45)
        kde_region_2d(ax_umap, creq2_2d,  xlim, ylim, E_COLOR["REQ"],
                      alpha_fill=0.35, alpha_line=0.78, density_pct=60, bw=0.50)

        # HUNK — vivid orange scatter (entity type, consistent with 3D)
        ax_umap.scatter(ch2_2d[:, 0], ch2_2d[:, 1],
                        c=E_COLOR["HUNK"], s=14, alpha=0.65,
                        edgecolors="none", zorder=4)

        # Legend
        umap_handles = [
            mpatches.Patch(color=E_COLOR["ORIG"],  label="ORIG — codebase"),
            mpatches.Patch(color=E_COLOR["TEST"],  label="TEST — fail-to-pass test"),
            mpatches.Patch(color=E_COLOR["REQ"],   label="REQ — requirement"),
            mpatches.Patch(color=E_COLOR["HUNK"],  label="HUNK — edit candidate"),
        ]
        ax_umap.legend(handles=umap_handles, fontsize=7.0, loc="best",
                       framealpha=0.92, edgecolor="#ccc")

        ax_umap.set_title(f"M2 Embedding Space (2D UMAP, {len(h2):,} hunks)", fontsize=11)
        ax_umap.set_xlabel("UMAP-1", fontsize=10)
        ax_umap.set_ylabel("UMAP-2", fontsize=10)
        ax_umap.set_xlim(xlim)
        ax_umap.set_ylim(ylim)
        ax_umap.grid(True, alpha=0.18, lw=0.5)

    plt.tight_layout()
    out_png = str(args.out_prefix) + "_combined.png"
    out_pdf = str(args.out_prefix) + "_combined.pdf"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"\nSaved → {out_png}")
    print(f"Saved → {out_pdf}")
    plt.close()

    # Also save scatter-only figure
    fig2, ax2 = plt.subplots(figsize=(8, 6.5))
    ax2.axhspan(THR_ORIG, 1.06, color="#e8f5e9", alpha=0.45, zorder=0)
    ax2.axvspan(THR_REQ,  1.05, color="#e3f2fd", alpha=0.45, zorder=0)
    ax2.axvline(THR_REQ,  color="#999", lw=1.0, ls="--", zorder=1)
    ax2.axhline(THR_ORIG, color="#999", lw=1.0, ls="--", zorder=1)
    # (region labels placed next to stars below)
    for tier in ["3", "2", "1"]:
        idx = [i for i in range(n) if tags[i] == tier]
        if not idx:
            continue
        st_norm = 0.25 + 0.75 * (
            (np.array([stests[i] for i in idx]) - stest_lo) / (stest_hi - stest_lo + 1e-8)
        )
        ax2.scatter([sreqs[i] for i in idx], [sorigs[i] for i in idx],
                    c=st_norm, cmap=TIER_CMAP[tier], vmin=0, vmax=1,
                    s=40, alpha=0.80, edgecolors=T_COLOR[tier], linewidths=0.4, zorder=3)
    for rg, idx in examples.items():
        col = T_COLOR.get(tags[idx], "#555")
        ax2.scatter(sreqs[idx], sorigs[idx], s=300, c="white", marker="*",
                    edgecolors="#222222", linewidths=1.5, zorder=6)
        dx, dy = LABEL_OFFSET.get(rg, (0.02, 0.06))
        ax2.text(sreqs[idx] + dx, sorigs[idx] + dy, rg,
                 fontsize=8.5, color="#333", fontweight="bold",
                 style="italic", zorder=7)
        dx, dy = REGION_ANNOT.get(rg, (0.05, 0.05))
        ax2.annotate(REGION_DESC[rg],
                     xy=(sreqs[idx], sorigs[idx]),
                     xytext=(sreqs[idx] + dx, sorigs[idx] + dy),
                     fontsize=7.5,
                     arrowprops=dict(arrowstyle="->", color="#333", lw=0.9),
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", lw=0.8),
                     color="#111", zorder=7)
    ax2.legend(handles=tier_handles + gradient_handles,
               loc="lower left", fontsize=7.5, framealpha=0.92, edgecolor="#ccc")

    ax2.set_xlabel("sim(hunk, REQ) — semantic alignment with requirement", fontsize=10)
    ax2.set_ylabel("sim(hunk, ORIG) — structural grounding in codebase", fontsize=10)
    ax2.set_title("Learned Edit Geometry (M2, Django)", fontsize=11)
    ax2.set_xlim(-0.40, 1.02)
    ax2.set_ylim(-0.50, 1.08)
    ax2.grid(True, alpha=0.20, lw=0.5)
    plt.tight_layout()
    out_sc = str(args.out_prefix) + "_region_scatter.png"
    plt.savefig(out_sc, dpi=200, bbox_inches="tight")
    print(f"Saved → {out_sc}")
    plt.close()


if __name__ == "__main__":
    main()
