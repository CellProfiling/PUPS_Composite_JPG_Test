#!/usr/bin/env python3
"""STEP 10 — Panel B: does PUPS follow the protein sequence, or the leakage in its input?

    ./.venv/bin/python 10_Figure_Leakage.py --pred TRACK/work/predictions \
        --variants TRACK/work/variants --embed TRACK/esm2_output --out TRACK/figures

Runs its OWN inference (step 7 scores only matched cell/sequence pairs; this deliberately
mismatches them), so it needs <track>/esm2_output/ too. A few minutes on CPU.

THE EXPERIMENT
--------------
This is Extended Data Fig. 5b's design, turned on the input format. Build a matrix:

    rows    = cells      (each stained for a DIFFERENT protein)
    columns = protein sequences fed to the model

Off-diagonal entries pair one cell's landmark images with another protein's sequence. Then:

  * if the model reads LEAKAGE in the landmark channels, every entry in a ROW looks the same —
    it reproduces whatever protein was actually stained in that cell, whatever sequence it was
    given. Rows constant.
  * if the model reads the SEQUENCE, every entry in a COLUMN looks the same. Columns constant.

The paper's own caption says the un-thresholded model "outputs images similar to the protein
that was stained in HPA for the particular cell, regardless of the input protein sequence".
That is the row-constant failure. We measure it PER INPUT FORMAT, which is the new part: if it
is weaker on single_tiff than on composite, the leakage being followed is an artifact of the
RGB composite rather than of the microscopy.

That claim only lands because HPA has no optical bleed-through to begin with — the channels are
acquired in three sequentials with separate excitation and emission bands — established by the
companion project, github.com/cellprofiling/HPA_BleedThrough_Exploration.

QUANTIFICATION (written to panelB_scores.csv)
---------------------------------------------
For each (format, checkpoint), with truth_i = the real protein image of cell i:

  matched_r    = mean_i  corr(P_ii, truth_i)          prediction when the sequence is correct
  mismatched_r = mean_i mean_{j!=i} corr(P_ij, truth_i)
                                                      prediction when the sequence is WRONG but
                                                      the cell is the same
  sequence_effect = matched_r - mismatched_r

`mismatched_r` is the leakage measure: high means the model reproduces the stained protein even
when told a different protein. `sequence_effect` near zero means the sequence is doing nothing.
Also reported: mean std across a row (should be ~0 under pure leakage) versus across a column.

⚠ CHECKPOINT CAVEAT: 'nothreshold' was trained ~25% less than 'default' (epoch 0/step 2661 vs
epoch 1/step 3548), so a default-vs-nothreshold gap is not attributable to the threshold alone.
The ACROSS-FORMAT comparison within one checkpoint is clean and is what the argument rests on.
"""
import argparse
import csv
import itertools
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
import torch                             # noqa: E402

# Match 9_Figure_Formats.py. PUPS's released code uses "gray"; their published
# figures are viridis. --cmap overrides.
CMAP = "viridis"

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "PUPS_Code"))

LANDMARK_SLICE = slice(0, 3)
PROTEIN_IDX = 3

# Same reference as 8_Compute_Metrics.py and 9_Figure_Formats.py: every variant is scored
# against the 16-bit TIFF protein channel, never against its own. Grading the composite arm
# against the composite's own green plane would score it against the artifact under test.
GROUND_TRUTH_VARIANT = "single_tiff"

STACKS = ["composite", "single_jpg", "single_tiff"]   # must match 6_Build_Input_Variants.py

# Same rule as 7_Run_PUPS_Inference.py:71. A model trained WITH the 0.19 threshold must be fed
# thresholded landmarks, so only these two pairings are train/test consistent:
#   --threshold 0 -> nothreshold checkpoint   (ED Fig. 5b, top panel)
#   --threshold 1 -> default checkpoint       (ED Fig. 5b, bottom panel; published setting)
CONSISTENT = {("nothreshold", 0), ("default", 1)}

CHECKPOINTS = {
    "default": "splice_isoform_dataset_cell_line_and_gene_split_full-epoch=01-val_combined_loss=0.18.ckpt",
    "nothreshold": "nothreshold.ckpt",
}


def load_selection(roi_file):
    """(Gene, ImagePrefix) pairs listed in a ROI file — used by --select-from."""
    import csv as _csv
    with open(roi_file, newline="") as fh:
        return {(r["Gene"].strip(), r["ImagePrefix"].strip()) for r in _csv.DictReader(fh)
                if (r.get("Skip") or "F").strip().upper() != "T"}


def pearson(a, b):
    a, b = a.ravel(), b.ravel()
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    global CMAP
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="e.g. TRACK/work/predictions")
    ap.add_argument("--variants", required=True, help="e.g. TRACK/work/variants")
    ap.add_argument("--embed", required=True, help="<track>/esm2_output from step 5")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--n", type=int, default=6, help="matrix size (n cells x n sequences)")
    ap.add_argument("--tiles", nargs="*", help="explicit tile_ids")
    ap.add_argument("--select-from", default=None,
                    help="ROI file whose tiles to use, e.g. Independent_Selection/ROI_examples_Selected.txt")
    ap.add_argument("--all-genes", action="store_true",
                    help="cover EVERY gene, in disjoint groups of --n")
    ap.add_argument("--threshold", type=int, default=1, choices=[0, 1],
                    help="apply the 0.19 landmark threshold (1 = PUPS published setting)")
    ap.add_argument("--all-conditions", action="store_true",
                    help="also render the train/test-inconsistent checkpoint/threshold pairings "
                         "(ablations; see CONSISTENT at the top of this file)")
    ap.add_argument("--cmap", default="viridis",
                    help="LUT for the panels (default viridis, as in their "
                         "published figures; `gray` matches their released code)")
    args = ap.parse_args()
    CMAP = args.cmap

    # Reuse step 7's loaders so the preprocessing cannot drift between steps.
    sys.path.insert(0, str(ROOT))
    import importlib
    step7 = importlib.import_module("7_Run_PUPS_Inference")
    from src.model.full_model import SubCellProtModel

    var_dir, pred_dir = pathlib.Path(args.variants), pathlib.Path(args.pred)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not (pred_dir / "index.csv").exists():
        sys.exit(f"{pred_dir}/index.csv not found — run 7_Run_PUPS_Inference.py first")

    tiles = list(csv.DictReader(open(pred_dir / "index.csv", newline="")))
    if args.tiles:
        by_id = {t["tile_id"]: t for t in tiles}
        tiles = [by_id[w] for w in args.tiles]
    else:
        if args.select_from:
            keep = load_selection(args.select_from)
            chosen = [t for t in tiles if (t["gene"], t["prefix"]) in keep]
            if not chosen:
                sys.exit(f"--select-from {args.select_from} matched no tiles in the index")
        else:
            chosen = ([t for t in tiles if (t.get("example") or "").upper() == "T"]
                      or [t for t in tiles if (t.get("annotation") or "").strip()]
                      or tiles)
        # One tile per gene: repeating a gene would put identical sequences in two columns.
        seen, picked = set(), []
        for t in chosen:
            if t["gene"] not in seen:
                seen.add(t["gene"]); picked.append(t)
            if not args.all_genes and len(picked) == args.n:
                break
        tiles = sorted(picked, key=lambda t: t["gene"]) if args.all_genes else picked
        # See the note in 9_Figure_Formats.py: --n truncates in silence.
        available = {c["gene"] for c in chosen}
        shown = {c["gene"] for c in tiles}
        if available - shown:
            print(f"  ⚠ --n {args.n} EXCLUDES {len(available - shown)} gene(s): "
                  f"{', '.join(sorted(available - shown))}")
            print(f"    pass --n {len(available)} for one matrix over all, or --all-genes "
                  f"for disjoint groups of {args.n}.")
    n = len(tiles)
    print(f"STEP 10 — Panel B, {n}x{n} cross-pairing matrix")
    print(f"  cells/sequences: {', '.join(t['gene'] for t in tiles)}")
    print(f"  threshold: {'0.19 (published)' if args.threshold else 'off'}\n")

    embeddings = step7.load_embeddings(args.embed)
    missing = [t["gene"] for t in tiles if t["gene"] not in embeddings]
    if missing:
        sys.exit(f"no embedding for {missing} — check <track>/esm2_output/sequences.json")

    stacks = {t["tile_id"]: np.load(var_dir / f"{t['tile_id']}.npz", allow_pickle=True)
              for t in tiles}
    variants = [k for k in next(iter(stacks.values())).files if k in STACKS]

    models = {}
    for name, fn in CHECKPOINTS.items():
        m = SubCellProtModel()
        ck = torch.load(ROOT / "PUPS_Code" / "checkpoints" / fn, map_location="cpu", weights_only=False)
        m.load_state_dict(ck["state_dict"]); m.eval()
        models[name] = m

    # Groups. A matrix costs n^2 forward passes and is unreadable much beyond ~8x8, so when
    # covering many genes we run several disjoint matrices of --n genes each and aggregate.
    # NOTE: mismatched pairs are then WITHIN a group only, which is the correct read -- every
    # off-diagonal entry is still a genuine cell/sequence mismatch. Group membership is
    # recorded so any group can be inspected on its own.
    groups = [tiles[i:i + args.n] for i in range(0, len(tiles), args.n)]
    groups = [g for g in groups if len(g) >= 2]        # a 1x1 matrix has no mismatched pair
    print(f"  {len(groups)} group(s) of up to {args.n} genes\n")

    scores = []
    for gi, group in enumerate(groups, 1):
        gn = len(group)
        gtag = f"g{gi:02d}" if len(groups) > 1 else ""
        for variant, (ck_name, model) in itertools.product(variants, models.items()):
            if not args.all_conditions and (ck_name, args.threshold) not in CONSISTENT:
                continue

            grid = np.empty((gn, gn), dtype=object)
            for i, ti in enumerate(group):
                arr = stacks[ti["tile_id"]][variant].astype(np.float32)
                if args.threshold:
                    arr = step7.apply_threshold(arr)
                stains = torch.from_numpy(arr[LANDMARK_SLICE])[None]
                for j, tj in enumerate(group):
                    esm2, x_len = embeddings[tj["gene"]]
                    with torch.no_grad():
                        img, _ = model.call_model(esm2, torch.tensor([x_len]), stains)
                    grid[i, j] = np.clip(img[0, 0].numpy(), 0, 1)

            truths = [stacks[t["tile_id"]][GROUND_TRUTH_VARIANT][PROTEIN_IDX] for t in group]
            matched = [pearson(grid[i, i], truths[i]) for i in range(gn)]
            mismatched = [pearson(grid[i, j], truths[i])
                          for i in range(gn) for j in range(gn) if i != j]

            # Normalised by mean amplitude, so formats with different output amplitude are
            # comparable. A dimmer arm would otherwise show a smaller within-row spread and
            # read as "more row-constant", i.e. more leakage, for no good reason.
            amp = float(np.mean([grid[i, j].mean() for i in range(gn) for j in range(gn)]))
            amp = amp if amp > 0 else np.nan
            row_std = float(np.mean([np.mean(np.std(np.stack(list(grid[i, :])), axis=0))
                                     for i in range(gn)])) / amp
            col_std = float(np.mean([np.mean(np.std(np.stack(list(grid[:, j])), axis=0))
                                     for j in range(gn)])) / amp

            scores.append({
                "variant": variant, "checkpoint": ck_name, "threshold": args.threshold,
                "group": gi, "n_genes": gn,
                "genes": ";".join(t["gene"] for t in group),
                "matched_r": np.nanmean(matched), "mismatched_r": np.nanmean(mismatched),
                "sequence_effect": np.nanmean(matched) - np.nanmean(mismatched),
                "within_row_cv": row_std, "within_col_cv": col_std,
            })
            stem = out / ("PanelB_" + "_".join(
                x for x in (variant, ck_name, f"thr{args.threshold}", gtag) if x))
            plot_matrix(grid, tiles=group, truths=truths, variant=variant,
                        checkpoint=ck_name, thr=args.threshold, stem=stem)

    df = pd.DataFrame(scores).round(5)
    df.to_csv(out / f"panelB_scores_thr{args.threshold}.csv", index=False)

    print("\n=== leakage quantification, aggregated over groups "
          "(gene-weighted mean) ===")
    agg = (df.groupby(["variant", "checkpoint", "threshold"])
             .apply(lambda g: pd.Series({
                 "n_genes": g.n_genes.sum(),
                 "matched_r": np.average(g.matched_r, weights=g.n_genes),
                 "mismatched_r": np.average(g.mismatched_r, weights=g.n_genes),
                 "sequence_effect": np.average(g.sequence_effect, weights=g.n_genes),
                 "within_row_cv": np.average(g.within_row_cv, weights=g.n_genes),
                 "within_col_cv": np.average(g.within_col_cv, weights=g.n_genes),
             }), include_groups=False)
             .round(5))
    agg.to_csv(out / f"panelB_aggregate_thr{args.threshold}.csv")
    print(agg.to_string())
    if len(groups) > 1:
        print(f"\n=== per group ({len(groups)} groups) ===")
        print(df[["variant", "checkpoint", "group", "matched_r", "mismatched_r",
                  "sequence_effect"]].to_string(index=False))
    print("\n  mismatched_r high  -> the prediction tracks the CELL, not the sequence (leakage)")
    print("  sequence_effect ~0 -> the protein sequence is doing nothing")
    print("  within_row_cv ~0  -> rows constant = the row-constant failure of ED Fig. 5b")
    print("\n  Compare ACROSS FORMATS within one checkpoint; the two checkpoints differ in")
    print("  training length, so that axis is confounded (see module docstring).")
    print(f"\n  figures + panelB_scores.csv -> {out}")
    print("\nSTEP 10 COMPLETE")
    print("  NEXT: ./.venv/bin/python 11_Ablate_Channels.py --variants TRACK/work/variants "
          "--embed TRACK/esm2_output --out TRACK/work/ablation")

    return 0


def plot_matrix(grid, tiles, truths, variant, checkpoint, thr, stem):
    """Matrix plus a leading column of ground truth, so each row can be read against it."""
    n = len(tiles)
    fig, axes = plt.subplots(n, n + 1, figsize=(1.55 * (n + 1), 1.6 * n + 1.1))
    for i in range(n):
        axes[i, 0].imshow(truths[i], cmap=CMAP, vmin=0, vmax=1)
        axes[i, 0].set_ylabel(f"{tiles[i]['gene']}\ncell", fontsize=7,
                              rotation=0, ha="right", va="center", labelpad=22)
        if i == 0:
            axes[i, 0].set_title("real protein\nin this cell", fontsize=7)
        for j in range(n):
            ax = axes[i, j + 1]
            ax.imshow(grid[i, j], cmap=CMAP, vmin=0, vmax=1)
            if i == 0:
                ax.set_title(f"seq:\n{tiles[j]['gene']}", fontsize=7)
            if i == j:      # matched pair: the only cell that is not a deliberate mismatch
                for s in ax.spines.values():
                    s.set_color("tab:green"); s.set_linewidth(1.8)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        f"Panel B — same cell, different protein sequence   [{variant}, {checkpoint}, "
        f"threshold={'0.19' if thr else 'off'}]\n"
        "rows constant => model follows leakage in the input;  columns constant => follows the "
        "sequence.  green = matched pair",
        fontsize=9, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.{{pdf,png,svg}}")


if __name__ == "__main__":
    sys.exit(main())
