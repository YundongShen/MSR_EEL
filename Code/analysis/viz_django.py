"""Generate side-by-side 3D UMAP comparing M0 (untrained) vs M2 (trained).

UMAP is fitted on HUNK embeddings only for each model independently.
ORIG background = ALL source units pooled across all instances (~17k vectors).
Tiers simplified to T1/T2/T3 (no 2a/2b split).

Usage:
    python analysis/viz_django.py \
        --checkpoint-m2 checkpoints/m2/best.pt \
        --instances      data/processed/instances_full.jsonl \
        --tier3          data/cache/tier3_hunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from config import default_config as cfg
from data.entailment_dataset import _render_hunk, _units_for_hunk
from data.instances import load_instances
from data.provenance import load_checkpoint_checked
from data.retained_hunks import add_retained_args, apply_retained_args, retained_hunks
from models.entailment_encoder import EntailmentEncoder

# 4 entity types only — no tier distinction in 3D
ENTITY_COLOR = {
    "ORIG": "#d4d4d4",   # light gray      — codebase background, recedes
    "REQ":  "#0072B2",   # Wong blue       — anchor
    "TEST": "#009E73",   # Wong teal       — anchor, distinct from REQ
    "HUNK": "#E69F00",   # Wong yellow-orange — foreground
}
ENTITY_LABEL = {
    "ORIG": "ORIG (codebase source units)",
    "REQ":  "REQ (requirement)",
    "TEST": "TEST (fail-to-pass test)",
    "HUNK": "HUNK (edit candidate)",
}


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def kde_region_2d(ax, pts2d, all_coords, color, alpha_fill=0.18, alpha_line=0.55,
                  density_pct=70, bw=0.35):
    from scipy.stats import gaussian_kde
    if len(pts2d) < 5:
        return
    margin = 1.5
    xmin, xmax = all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin
    ymin, ymax = all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin
    xx, yy = np.mgrid[xmin:xmax:180j, ymin:ymax:180j]
    try:
        kde = gaussian_kde(pts2d.T, bw_method=bw)
        Z = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        thr = np.percentile(Z, density_pct)
        ax.contourf(xx, yy, Z, levels=[thr, Z.max()],
                    colors=[color], alpha=alpha_fill, zorder=1)
        ax.contour(xx, yy, Z, levels=[thr],
                   colors=[color], linewidths=1.0, alpha=alpha_line, zorder=2)
    except Exception as e:
        print(f"  [kde warn] {e}")


# ---------------------------------------------------------------------------
# Encoding helper
# ---------------------------------------------------------------------------

def encode_all(encoder, instances, tier3_lookup, device):
    """Return hunk_embs, hunk_tags, hunk_meta, req_embs, test_embs, orig_embs."""
    hunk_embs, hunk_tags, hunk_meta = [], [], []
    req_embs, test_embs, orig_embs  = [], [], []

    with torch.no_grad():
        for inst in instances:
            iid          = inst.get("instance_id", "")
            req_text     = inst.get("requirement", "")[:cfg.data.max_req_chars]
            test_texts   = [tf["code"] for tf in inst.get("test_functions", []) if tf.get("code")]
            source_units = inst.get("source_units", [])

            # REQ — one per instance
            r_emb  = encoder.encode([req_text], "REQ", device, True)
            req_embs.append(r_emb.cpu().numpy()[0])

            # TEST — every test function individually
            if test_texts:
                t_embs = encoder.encode(test_texts, "TEST", device, True)
                test_embs.extend(t_embs.cpu().numpy().tolist())

            # ORIG — ALL source units for this instance (full project background)
            all_orig_codes = [u["code"] for u in source_units if u.get("code")]
            if all_orig_codes:
                # encode in batches of 16 to stay within GPU memory
                for i in range(0, len(all_orig_codes), 16):
                    batch = all_orig_codes[i:i+16]
                    o_batch = encoder.encode(batch, "ORIG", device, True)
                    orig_embs.extend(o_batch.cpu().numpy().tolist())

            # Hunks
            all_hunks = [(h, h.get("tier_label") or 1) for h in retained_hunks(inst, cfg)]
            for h in tier3_lookup.get(iid, []):
                all_hunks.append((h, 3))

            for hunk, base_tier in all_hunks:
                text = _render_hunk(hunk)
                if not text.strip():
                    continue

                # simplified tier: 1, 2, or 3 (no 2a/2b)
                tier = str(base_tier) if base_tier in (1, 3) else "2"

                h_emb  = encoder.encode([text], "HUNK", device, True)
                hunk_origs = [u["code"] for u in _units_for_hunk(hunk, source_units) if u.get("code")]
                sim_req, sim_test, sim_orig = encoder.similarity_components(
                    [text], [req_text], [test_texts], [hunk_origs], device, True,
                )

                hunk_embs.append(h_emb.cpu().numpy()[0])
                hunk_tags.append(tier)
                hunk_meta.append({
                    "iid": iid, "tier": tier, "text": text[:400],
                    "sreq": sim_req.item(),
                    "stest": sim_test.item(),
                    "sorig": sim_orig.item(),
                })

    print(f"  Hunks={len(hunk_embs)} REQ={len(req_embs)} "
          f"TEST={len(test_embs)} ORIG={len(orig_embs)}")
    return (np.array(hunk_embs), hunk_tags, hunk_meta,
            np.array(req_embs), np.array(test_embs), np.array(orig_embs))


# ---------------------------------------------------------------------------
# Plot one 3D axis
# ---------------------------------------------------------------------------

def plot_3d(ax3, coords_h, coords_req, coords_test, coords_orig, title,
            show_legend=True, title_fontsize=11, tick_labelsize=None):
    import matplotlib.patches as mpatches

    # ORIG — pale background, many points, very low alpha so it fades behind
    ax3.scatter(coords_orig[:, 0], coords_orig[:, 1], coords_orig[:, 2],
                c=ENTITY_COLOR["ORIG"], s=3, alpha=0.30,
                edgecolors="none", zorder=1)
    # TEST — soft green, medium size
    ax3.scatter(coords_test[:, 0], coords_test[:, 1], coords_test[:, 2],
                c=ENTITY_COLOR["TEST"], s=8, alpha=0.40,
                edgecolors="none", zorder=2)
    # REQ — soft blue, small count, slightly more visible
    ax3.scatter(coords_req[:, 0], coords_req[:, 1], coords_req[:, 2],
                c=ENTITY_COLOR["REQ"], s=14, alpha=0.55,
                edgecolors="none", zorder=3)
    # HUNK — warm red, foreground, clearly distinct
    ax3.scatter(coords_h[:, 0], coords_h[:, 1], coords_h[:, 2],
                c=ENTITY_COLOR["HUNK"], s=12, alpha=0.70,
                edgecolors="none", zorder=4)

    if show_legend:
        handles = [mpatches.Patch(color=ENTITY_COLOR[k], label=ENTITY_LABEL[k])
                   for k in ["ORIG", "TEST", "REQ", "HUNK"]]
        ax3.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.85)
    ax3.set_title(title, fontsize=title_fontsize, pad=8)
    if tick_labelsize is not None:
        ax3.tick_params(axis="both", which="major", labelsize=tick_labelsize)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-m2", default="checkpoints/m2/best.pt")
    parser.add_argument("--instances",     default="data/processed/instances_full.jsonl")
    parser.add_argument("--tier3",         default="data/cache/tier3_hunks.jsonl")
    parser.add_argument("--out-prefix",    default="logs/django_viz")
    parser.add_argument("--encode-only",   action="store_true",
                        help="Only encode and save embeddings, skip UMAP/plotting")
    parser.add_argument("--plot-only",     action="store_true",
                        help="Skip encoding, load saved embeddings and run UMAP/plotting")
    add_retained_args(parser)
    args = parser.parse_args()
    apply_retained_args(cfg, args)

    out = Path(args.out_prefix)
    emb_file = str(out) + "_embs.npz"

    # -----------------------------------------------------------------------
    # ENCODE phase (needs GPU)
    # -----------------------------------------------------------------------
    if not args.plot_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        tier3_lookup: dict[str, list[dict]] = {}
        with open(args.tier3) as f:
            for line in f:
                rec = json.loads(line)
                tier3_lookup[rec["instance_id"]] = rec.get("tier3_hunks", [])

        instances = [i for i in load_instances(args.instances) if i.get("instance_id", "").startswith("django")]
        print(f"Django instances: {len(instances)}")

        def make_encoder(checkpoint=None):
            enc = EntailmentEncoder(
                model_name=cfg.model.encoder_name,
                projection_dim=cfg.model.projection_dim,
                dropout=0.0,
                max_length=cfg.model.max_length,
            ).to(device)
            if checkpoint:
                ckpt = load_checkpoint_checked(checkpoint, map_location=device)   # refuses unstamped checkpoints
                enc.load_state_dict(ckpt["encoder_state"])
                print(f"Loaded checkpoint: {checkpoint}")
            else:
                print("M0: using untrained base encoder (no checkpoint)")
            enc.eval()
            return enc

        print("\n=== Encoding with M0 (untrained) ===")
        enc_m0 = make_encoder(checkpoint=None)
        (h0, tags0, meta0, req0, test0, orig0) = encode_all(
            enc_m0, instances, tier3_lookup, device)
        del enc_m0
        torch.cuda.empty_cache()

        print("\n=== Encoding with M2 ===")
        enc_m2 = make_encoder(checkpoint=args.checkpoint_m2)
        (h2, tags2, meta2, req2, test2, orig2) = encode_all(
            enc_m2, instances, tier3_lookup, device)
        del enc_m2
        torch.cuda.empty_cache()

        # Save all raw embeddings for the plot phase
        np.savez(
            emb_file,
            h0=h0, req0=req0, test0=test0, orig0=orig0,
            h2=h2, req2=req2, test2=test2, orig2=orig2,
            tags0=np.array(tags0), tags2=np.array(tags2),
            meta_iid   = np.array([m["iid"]   for m in meta2], dtype=object),
            meta_sreq  = np.array([m["sreq"]  for m in meta2], dtype=np.float32),
            meta_stest = np.array([m["stest"] for m in meta2], dtype=np.float32),
            meta_sorig = np.array([m["sorig"] for m in meta2], dtype=np.float32),
        )
        print(f"Embeddings saved → {emb_file}")

        # Also save M2 data.npz in standard format for downstream scripts
        n_hunks = len(h2)
        all_tags = (list(tags2) + ["REQ"]*len(req2) + ["TEST"]*len(test2) + ["ORIG"]*len(orig2))
        X_all = np.vstack([h2, req2, test2, orig2])
        meta_text = np.array([m["text"] for m in meta2], dtype=object)
        np.savez(
            str(out) + "_data.npz",
            X=X_all, tags=np.array(all_tags), n_hunks=n_hunks,
            meta_iid   = np.array([m["iid"]  for m in meta2], dtype=object),
            meta_text  = meta_text,
            meta_sreq  = np.array([m["sreq"]  for m in meta2], dtype=np.float32),
            meta_stest = np.array([m["stest"] for m in meta2], dtype=np.float32),
            meta_sorig = np.array([m["sorig"] for m in meta2], dtype=np.float32),
        )
        print(f"Data saved → {out}_data.npz")

        if args.encode_only:
            print("Encode-only mode — done.")
            return

    # -----------------------------------------------------------------------
    # PLOT phase (CPU, high memory)
    # -----------------------------------------------------------------------
    print(f"\nLoading embeddings from {emb_file} …")
    d = np.load(emb_file, allow_pickle=True)
    h0    = d["h0"];    req0  = d["req0"];  test0 = d["test0"]; orig0 = d["orig0"]
    h2    = d["h2"];    req2  = d["req2"];  test2 = d["test2"]; orig2 = d["orig2"]
    tags0 = list(d["tags0"]); tags2 = list(d["tags2"])
    print(f"  M0: hunks={len(h0)} req={len(req0)} test={len(test0)} orig={len(orig0)}")
    print(f"  M2: hunks={len(h2)} req={len(req2)} test={len(test2)} orig={len(orig2)}")

    try:
        import umap as umap_lib
    except ImportError:
        print("umap-learn not installed — skipping UMAP")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    umap_kw = dict(n_components=3, random_state=42, n_neighbors=50, min_dist=0.4)

    # Fit UMAP on M0 HUNK embeddings — HUNK vectors are homogeneous (all diff hunks),
    # so they produce a natural spherical manifold.  ORIG/REQ/TEST are then projected
    # into this hunk-space, showing how each entity type relates to the hunk distribution.
    print("\nFitting 3D UMAP on M0 hunk embeddings (natural reference space) …")
    r0 = umap_lib.UMAP(**umap_kw)
    ch0    = r0.fit_transform(h0)
    creq0  = r0.transform(req0)
    ctest0 = r0.transform(test0)
    print(f"  Transforming {len(orig0):,} ORIG vectors …")
    corig0 = r0.transform(orig0)

    # Project M2 entities into M0's coordinate system.
    # Retained HUNKs pulled toward anchors by training → cluster near REQ/TEST in M0 space.
    # T3 HUNKs pushed away from each instance's REQ → scatter differently from retained.
    print("Projecting M2 entities into M0 UMAP space …")
    ch2    = r0.transform(h2)
    # Anchors (REQ/TEST/ORIG) encode same text → use M0 coordinates as background
    creq2  = creq0
    ctest2 = ctest0
    corig2 = corig0

    # --- Side-by-side 3D plot ---
    # Original figure size/layout is untouched. The only changes from the
    # original are: no suptitle, no per-panel titles, and one shared legend
    # (bigger text) instead of two duplicated per-panel legends.
    n_hunks_str = f"{len(h0):,}"  # still used by the untouched single-M2 plot below

    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(18, 8))
    ax_m0 = fig.add_subplot(121, projection="3d")
    ax_m2 = fig.add_subplot(122, projection="3d")

    plot_3d(ax_m0, ch0, creq0, ctest0, corig0, "", show_legend=False)
    plot_3d(ax_m2, ch2, creq2, ctest2, corig2, "", show_legend=False)

    shared_handles = [
        Line2D([0], [0], marker="o", linestyle="", markeredgecolor="none",
               markerfacecolor=ENTITY_COLOR[k], markersize=10, label=k)
        for k in ["ORIG", "TEST", "REQ", "HUNK"]
    ]
    fig.legend(handles=shared_handles, labels=["ORIG", "TEST", "REQ", "HUNK"],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4,
               fontsize=18, markerscale=2.5,
               handletextpad=0.6, columnspacing=1.6,
               frameon=True, framealpha=0.85)

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(str(out) + "_3d_compare.png", dpi=600,
                bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"Saved → {out}_3d_compare.png")

    # --- Also save individual M2 3D for paper ---
    fig2 = plt.figure(figsize=(10, 8))
    ax2s = fig2.add_subplot(111, projection="3d")
    plot_3d(ax2s, ch2, creq2, ctest2, corig2,
            f"Django Edit Space — M2 (3D UMAP, {n_hunks_str} hunks)")
    plt.tight_layout()
    plt.savefig(str(out) + "_3d.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}_3d.png")

    # --- 2D UMAP for M2 (fit on hunks only) ---
    print("Fitting 2D UMAP for M2 hunk embeddings …")
    r2d = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=50, min_dist=0.4)
    ch2_2d    = r2d.fit_transform(h2)
    creq2_2d  = r2d.transform(req2)
    ctest2_2d = r2d.transform(test2)
    print(f"  Transforming {len(orig2):,} ORIG vectors for 2D …")
    corig2_2d = r2d.transform(orig2)

    import matplotlib.patches as mpatches

    all2d = np.vstack([ch2_2d, creq2_2d, ctest2_2d, corig2_2d])
    xlim2d = (all2d[:, 0].min() - 1.5, all2d[:, 0].max() + 1.5)
    ylim2d = (all2d[:, 1].min() - 1.5, all2d[:, 1].max() + 1.5)

    fig3, ax2d = plt.subplots(figsize=(10, 8))

    # ORIG — hexbin (preserves true density/extent, not normalised)
    ax2d.hexbin(corig2_2d[:, 0], corig2_2d[:, 1],
                gridsize=55, cmap="Greys", mincnt=1, alpha=0.55, zorder=1,
                extent=(xlim2d[0], xlim2d[1], ylim2d[0], ylim2d[1]))

    # TEST and REQ — KDE contours (small counts, concentrated)
    kde_region_2d(ax2d, ctest2_2d, all2d, ENTITY_COLOR["TEST"],
                  alpha_fill=0.22, alpha_line=0.65)
    kde_region_2d(ax2d, creq2_2d,  all2d, ENTITY_COLOR["REQ"],
                  alpha_fill=0.30, alpha_line=0.75)

    # HUNK — scatter, vivid orange
    ax2d.scatter(ch2_2d[:, 0], ch2_2d[:, 1],
                 c=ENTITY_COLOR["HUNK"], s=14, alpha=0.65,
                 edgecolors="none", zorder=4)

    handles = [
        mpatches.Patch(color="#aaa",               label=ENTITY_LABEL["ORIG"]),
        mpatches.Patch(color=ENTITY_COLOR["TEST"],  label=ENTITY_LABEL["TEST"]),
        mpatches.Patch(color=ENTITY_COLOR["REQ"],   label=ENTITY_LABEL["REQ"]),
        mpatches.Patch(color=ENTITY_COLOR["HUNK"],  label=ENTITY_LABEL["HUNK"]),
    ]
    ax2d.legend(handles=handles, fontsize=7, loc="best",
                framealpha=0.9, edgecolor="#ccc")
    ax2d.set_title(f"Django Edit Space — M2 (2D UMAP, {n_hunks_str} hunks)", fontsize=11)
    ax2d.set_xlabel("UMAP-1"); ax2d.set_ylabel("UMAP-2")
    ax2d.set_xlim(xlim2d); ax2d.set_ylim(ylim2d)
    plt.tight_layout()
    plt.savefig(str(out) + "_2d.png", dpi=200, bbox_inches="tight")
    plt.savefig(str(out) + "_2d.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}_2d.png")


if __name__ == "__main__":
    main()
