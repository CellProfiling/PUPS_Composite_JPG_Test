#!/usr/bin/env python3
"""STEP 9 — Panel A: PUPS predictions across input formats.

    # the 6 curated example tiles, PUPS's published configuration
    ./.venv/bin/python 9_Figure_Formats.py --pred TRACK/work/predictions \
        --variants TRACK/work/variants --out TRACK/figures --n 6

    # every condition, so nothing is pre-selected for you
    ./.venv/bin/python 9_Figure_Formats.py --pred TRACK/work/predictions \
        --variants TRACK/work/variants --out TRACK/figures --all-conditions

LAYOUT
------
Rows are tiles. Columns are:

    composite input | composite, default ckpt +0.19 (published) | composite, nothreshold ckpt
                    | single_jpg | single_tiff | real protein (TIFF)

Every prediction column states its own model and threshold in its header, because the figure
deliberately mixes checkpoints: the first two are a fixed reference pair, and the no-threshold
one uses the `nothreshold` CHECKPOINT, since feeding raw input to the thresholded model would be
a train/test mismatch. The two format columns follow `--checkpoint`, with the threshold implied
by the train/test-consistent pairing, so `--all-conditions` renders two figures that differ.
Read left to right: the input PUPS actually consumes, what it predicts from each format, the truth.

The last column is the same protein channel step 8 scores against, so the figure and the
tables cannot disagree.

WHICH TILES
-----------
Defaults to the tiles in --tiles if given, else the tiles carrying an `annotation` in
index.csv (the 6 hand-picked structures: cytokinetic bridge, plasma membrane, cell junctions,
mitochondria, nucleoli, microtubule ends), else the first --n tiles. Annotated tiles are
preferred because each was chosen to show a specific localisation, which is what makes a
format-induced difference legible rather than abstract.

OUTPUT: PDF + PNG + SVG, matching the companion project's figure conventions.
"""
import argparse
import csv
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl                 # noqa: E402
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

# Column order. The published configuration is `composite` with the threshold on.
# (kind, checkpoint, variant, threshold, header)
#
# The first two prediction columns are a FIXED reference pair, always drawn from the same two
# conditions so the effect of the 0.19 threshold on the composite is visible in every figure.
# The threshold-off one uses the `nothreshold` CHECKPOINT: a model trained WITH the threshold
# must be fed thresholded landmarks, so feeding it raw input is a train/test mismatch, not
# "PUPS without the threshold".
#
# The two FORMAT columns carry checkpoint None, meaning "follow --checkpoint", with the
# threshold implied by CONSISTENT_THR. That is what makes --all-conditions render two figures
# that actually differ. Every prediction column states its own model and threshold in the
# header, because this figure deliberately mixes checkpoints and a single global label would
# be wrong for at least one column.
CONSISTENT_THR = {"default": 1, "nothreshold": 0}


def _cond_label(ckpt, thr):
    return f"{ckpt} model\n{'+ 0.19 threshold' if thr else 'threshold off'}"


COLUMNS = [
    ("input", None, "composite", None, "composite input\n(what PUPS reads)"),
    ("pred", "default", "composite", 1,
     "composite\ndefault model\n+ 0.19 threshold\n(PUBLISHED)"),
    ("pred", "nothreshold", "composite", 0,
     "composite\nnothreshold model\nthreshold off"),
    ("pred", None, "single_jpg", None, "single-channel\nJPG\n{cond}"),
    ("pred", None, "single_tiff", None, "single-channel\nTIFF\n{cond}"),
    ("truth", None, "single_tiff", None, "real protein\n(ground truth)"),
]

NUCLEUS, MICROTUBULE, ER, PROTEIN = 0, 1, 2, 3

# The reference every prediction is scored against: the GROUND TRUTH, which is exactly the
# image drawn in this figure's rightmost column -- `target__single_tiff`, the 16-bit TIFF protein
# channel. Same constant as 8_Compute_Metrics.py's GROUND_TRUTH_VARIANT, so the annotations
# under each panel and the tables in step 8 are the same numbers.
#
# It is the TIFF protein channel for EVERY variant, never each variant's own protein channel.
# Grading composite JPG against the stack's own protein channel would score each arm against a
# different target and against one that carries the very compression artifact under test.
GROUND_TRUTH_VARIANT = "single_tiff"


def pearson(a, b):
    """Same definition as 8_Compute_Metrics.py, so figure and tables cannot disagree."""
    a, b = a.ravel(), b.ravel()
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def panel_scores(pred, truth):
    """(r, mse, max) for one panel. `pred` must already be clipped to [0,1], as in step 8."""
    return (pearson(pred, truth), float(np.mean((pred - truth) ** 2)), float(pred.max()))


def _norm(img):
    """imshow kwargs implementing SCALE. Empty dict lets imshow autoscale to the panel."""
    return {} if SCALE == "panel" else {"vmin": 0, "vmax": 1}


def load_selection(roi_file):
    """(Gene, ImagePrefix) pairs listed in a ROI file — used by --select-from."""
    import csv as _csv
    with open(roi_file, newline="") as fh:
        return {(r["Gene"].strip(), r["ImagePrefix"].strip()) for r in _csv.DictReader(fh)
                if (r.get("Skip") or "F").strip().upper() != "T"}


def pick_tiles(tiles, wanted, n, select_from=None):
    """Choose the figure rows, in preference order.

    1. --tiles              explicit tile_ids, in the given order
    2. --select-from FILE   only tiles listed in that ROI file (e.g. ROI_examples_Selected.txt)
    3. Example == T         the per-gene representative flagged in ROI_examples.txt
    4. annotated tiles      any tile with an Annotation
    5. first n tiles

    In cases 3-5 the result is DEDUPLICATED BY GENE. Without that, ROI.txt's ordering yields
    rows like "ABCE1, ABCE1, AFDN, AFDN, AFDN, AFDN" — six rows showing two genes, which
    hides the very variety the figure exists to show.
    """
    if wanted:
        by_id = {t["tile_id"]: t for t in tiles}
        missing = [w for w in wanted if w not in by_id]
        if missing:
            sys.exit(f"--tiles not found in index: {missing}")
        return [by_id[w] for w in wanted]

    if select_from:
        keep = load_selection(select_from)
        chosen = [t for t in tiles if (t["gene"], t["prefix"]) in keep]
        if not chosen:
            sys.exit(f"--select-from {select_from} matched no tiles in the index")
    else:
        chosen = ([t for t in tiles if (t.get("example") or "").upper() == "T"]
                  or [t for t in tiles if (t.get("annotation") or "").strip()]
                  or tiles)

    seen, picked = set(), []
    for t in chosen:
        if t["gene"] in seen:
            continue
        seen.add(t["gene"])
        picked.append(t)
        if len(picked) == n:
            break
    return picked


def _suffix(checkpoint):
    return ("   [WARNING: trained ~25% less than 'default' — epoch 0/step 2661 "
            "vs epoch 1/step 3548]" if checkpoint == "nothreshold" else "")


# Grayscale intensity LUT for the single-channel panels. PUPS's released viz_utils.py uses
# cmap="gray" -- the ONLY colormap anywhere in their codebase -- but the published Fig. 2c is
# clearly viridis, so that figure was not produced by the released code. We default to viridis
# so our panels sit beside theirs, and print the LUT as a reference bar. --cmap gray reproduces
# what their code actually does.
CMAP = "viridis"

# How each single-channel panel is mapped to the LUT.
#   "panel"    -> each panel stretched to its own min..max. Matches how their published figures
#                 look, and is the fair way to judge SPATIAL agreement.
#   "absolute" -> fixed 0..1 for every panel. Honest about amplitude, but the target column was
#                 rescale_intensity'd to span [0,1] in step 6 while predictions peak around
#                 0.3-0.5, so every prediction looks dim beside the truth and a low-amplitude
#                 arm looks worse than its spatial agreement warrants.
# Either way the printed `max` keeps the amplitude visible, and r is scale-invariant.
SCALE = "panel"


def composite_rgb(stack):
    """Reconstruct the displayed composite from the variant stack, for the input column.

    R = microtubule, G = protein, B = nucleus -- the same mapping PUPS reads out of
    blue_red_green.jpg. This is a visualisation only; nothing downstream uses it.
    """
    return np.clip(np.stack([stack[MICROTUBULE], stack[PROTEIN], stack[NUCLEUS]], -1), 0, 1)


def make_figure(tiles, pred_dir, var_dir, checkpoint, title_suffix="", page_note=""):
    n_rows, n_cols = len(tiles), len(COLUMNS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(1.85 * n_cols, 2.24 * n_rows + 1.0))
    axes = np.atleast_2d(axes)

    for r, t in enumerate(tiles):
        tile_id = t["tile_id"]
        P = np.load(pred_dir / f"{tile_id}.npz")
        V = np.load(var_dir / f"{tile_id}.npz", allow_pickle=True)

        truth_ref = P[f"target__{GROUND_TRUTH_VARIANT}"]

        for c, (kind, col_ckpt, variant, thr, header) in enumerate(COLUMNS):
            # A format column (col_ckpt None) follows --checkpoint, and its threshold is the
            # train/test-consistent partner of that checkpoint -- never the other pairing.
            if kind == "pred" and col_ckpt is None:
                col_ckpt = checkpoint
                thr = CONSISTENT_THR[col_ckpt]
                header = header.format(cond=_cond_label(col_ckpt, thr))
            ax = axes[r, c]
            score = None
            if kind == "input":
                ax.imshow(composite_rgb(V[variant]))
            elif kind == "truth":
                ax.imshow(P[f"target__{variant}"], cmap=CMAP, **_norm(P[f"target__{variant}"]))
            else:
                key = f"{col_ckpt}__{variant}__thr{thr}"
                if key not in P.files:
                    ax.text(.5, .5, "missing", ha="center", va="center", fontsize=7)
                    ax.set_facecolor("0.9")
                else:
                    # Same clipping rule as step 8, so figure and tables agree.
                    pred = np.clip(P[key], 0, 1)
                    ax.imshow(pred, cmap=CMAP, **_norm(pred))
                    score = panel_scores(pred, truth_ref)
            ax.set_xticks([]); ax.set_yticks([])
            # Per-panel score against the TIFF ground truth. Omitted under the input and the
            # ground-truth columns, where it would be meaningless or trivially r=1.
            if score is not None:
                rr, mm, pk = score
                # `max` is printed because SCALE="panel" hides amplitude, and because MSE is
                # amplitude-sensitive while r is not -- see the caption.
                # Two lines: one line of three numbers is wider than a 1.85" panel and
                # collides with the neighbouring column.
                ax.set_xlabel(f"r={rr:.3f}   MSE={mm:.4f}\nmax={pk:.2f}",
                              fontsize=5.8, labelpad=2, linespacing=1.35)
            for s in ax.spines.values():
                s.set_linewidth(0.4); s.set_color("0.6")
            if r == 0:
                ax.set_title(header, fontsize=7.5, pad=6)
            if c == 0:
                lbl = f"{t['gene']}\n{t['cell_line']}"
                if (t.get("annotation") or "").strip():
                    lbl += f"\n{t['annotation']}"
                ax.set_ylabel(lbl, fontsize=7, rotation=0, ha="right", va="center", labelpad=34)

    # No global "checkpoint:" claim — the first two prediction columns are a fixed reference
    # pair from two different checkpoints, so any single figure-wide label would be wrong for
    # one of them. Each column states its own model and threshold instead.
    fig.suptitle(
        f"PUPS predictions across HPA input formats — format columns: {checkpoint} model"
        f"{title_suffix}{page_note}\n"
        "identical tile, identical preprocessing; only the source file format differs — "
        "model and threshold are labelled per column",
        fontsize=9.5, y=0.995)
    # Bottom margin must hold the LUT bar and the score caption. Expressed in inches and
    # converted, so a 1-row page and a 12-row page both lay out correctly.
    bottom_in = 0.80
    bot = bottom_in / fig.get_figheight()
    fig.tight_layout(rect=(0, bot, 1, 0.965))

    # Reference LUT below the grid. Intensities are meaningless without it: every panel is
    # rescale_intensity'd to [0,1] per patch, so colour shows RELATIVE intensity within a
    # panel and is not comparable in absolute terms between panels.
    cax = fig.add_axes([0.38, bot * 0.52, 0.24, 0.012])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=CMAP),
                      cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
    cb.ax.tick_params(labelsize=6, length=2, pad=1)
    cb.outline.set_linewidth(0.4)
    scale_txt = ("each panel stretched to its own min..max, so colour is relative within a "
                 "panel and NOT comparable between panels"
                 if SCALE == "panel" else
                 "fixed 0..1 for every panel, so colour IS comparable between panels")
    fig.text(0.5, bot * 0.30,
             f"LUT: {CMAP}  ·  {scale_txt}  ·  the composite column is RGB, not this LUT\n"
             f"r / MSE / max beneath each prediction: vs the ground-truth column on the right "
             f"(16-bit TIFF protein channel), clipped to [0,1] — as 8_Compute_Metrics.py "
             f"tabulates.  r is scale-invariant; MSE and max are not.",
             ha="center", va="top", fontsize=6, color="0.35", linespacing=1.6)
    return fig


def build_figure(tiles, pred_dir, var_dir, checkpoint, out_stem, title_suffix=""):
    """Single-page figure in pdf/png/svg — the curated-selection case."""
    fig = make_figure(tiles, pred_dir, var_dir, checkpoint, title_suffix)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{out_stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_stem}.{{pdf,png,svg}}")


def build_paginated(tiles, pred_dir, var_dir, checkpoint, out_dir, per_page, title_suffix=""):
    """Every tile, split across pages.

    117 tiles in one figure would be ~230 inches tall and unreadable, so the output is a
    multi-page PDF (for reviewing everything in one file) plus one PNG per page (for quick
    viewing without a PDF reader). Tiles arrive sorted by gene, so a gene's fields of view sit
    on adjacent rows and can be compared directly.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    out_dir.mkdir(parents=True, exist_ok=True)
    pages = [tiles[i:i + per_page] for i in range(0, len(tiles), per_page)]
    pdf_path = out_dir / f"PanelA_{checkpoint}_all.pdf"

    with PdfPages(pdf_path) as pdf:
        for pi, page in enumerate(pages, 1):
            note = f"   [page {pi}/{len(pages)}]"
            fig = make_figure(page, pred_dir, var_dir, checkpoint, title_suffix, note)
            pdf.savefig(fig, bbox_inches="tight")
            fig.savefig(out_dir / f"PanelA_{checkpoint}_p{pi:02d}.png",
                        dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"    page {pi}/{len(pages)}  ({len(page)} tiles)")
    print(f"  wrote {pdf_path} ({len(pages)} pages) + per-page PNGs")


def main():
    global CMAP, SCALE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="e.g. TRACK/work/predictions")
    ap.add_argument("--variants", required=True, help="e.g. TRACK/work/variants")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--checkpoint", default="default", choices=["default", "nothreshold"])
    ap.add_argument("--all-conditions", action="store_true",
                    help="also render the nothreshold checkpoint")
    ap.add_argument("--tiles", nargs="*", help="explicit tile_ids, in the order you want them")
    ap.add_argument("--n", type=int, default=6, help="tile count when not using --tiles")
    ap.add_argument("--select-from", default=None,
                    help="ROI file whose tiles to use, e.g. Independent_Selection/ROI_examples_Selected.txt")
    ap.add_argument("--all-tiles", action="store_true",
                    help="render EVERY tile in the index, paginated, for review rather than "
                         "publication; bypasses the --n cap")
    ap.add_argument("--per-page", type=int, default=8,
                    help="rows per page when using --all-tiles")
    ap.add_argument("--scale", default=SCALE, choices=["panel", "absolute"],
                    help="panel: stretch each panel to its own range (default, matches their "
                         "published figures, fair for judging spatial agreement). absolute: "
                         "fixed 0..1, which shows amplitude but makes every prediction look "
                         "dim beside the rescaled ground truth.")
    ap.add_argument("--cmap", default=CMAP,
                    help=f"LUT for the single-channel panels (default {CMAP}, matching their "
                         f"published Fig. 2c). Use `gray` to match what PUPS's released "
                         f"viz_utils.py actually does.")
    args = ap.parse_args()

    CMAP = args.cmap
    SCALE = args.scale

    pred_dir, var_dir = pathlib.Path(args.pred), pathlib.Path(args.variants)
    if not (pred_dir / "index.csv").exists():
        sys.exit(f"{pred_dir}/index.csv not found — run 7_Run_PUPS_Inference.py first")

    all_tiles = list(csv.DictReader(open(pred_dir / "index.csv", newline="")))
    out = pathlib.Path(args.out)
    checkpoints = ["default", "nothreshold"] if args.all_conditions else [args.checkpoint]

    if args.all_tiles:
        # Sort by gene so a gene's fields of view land on adjacent rows.
        tiles = sorted(all_tiles, key=lambda t: (t["gene"], t["cell_line"], t["prefix"]))
        print(f"STEP 9 — Panel A over ALL {len(tiles)} tiles "
              f"({len({t['gene'] for t in tiles})} genes), {args.per_page}/page\n")
        for ck in checkpoints:
            print(f"  checkpoint: {ck}")
            build_paginated(tiles, pred_dir, var_dir, ck, out, args.per_page,
                            _suffix(ck))
    else:
        tiles = pick_tiles(all_tiles, args.tiles, args.n, args.select_from)
        # --n silently truncates. A 9-gene track run with the default 6 drops three genes
        # from the figure and says nothing, which is easy to miss in a long log.
        available = {t["gene"] for t in all_tiles}
        shown = {t["gene"] for t in tiles}
        if available - shown:
            print(f"  ⚠ --n {args.n} EXCLUDES {len(available - shown)} gene(s): "
                  f"{', '.join(sorted(available - shown))}")
            print(f"    pass --n {len(available)} to cover all of them.")
        print(f"STEP 9 — Panel A over {len(tiles)} tiles: "
              f"{', '.join(t['gene'] for t in tiles)}\n")
        for ck in checkpoints:
            build_figure(tiles, pred_dir, var_dir, ck, out / f"PanelA_{ck}", _suffix(ck))

    print("\nSTEP 9 COMPLETE")
    print("  NEXT: ./.venv/bin/python 10_Figure_Leakage.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
