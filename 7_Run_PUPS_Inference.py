#!/usr/bin/env python3
"""STEP 7 — Run PUPS on every input variant, with and without the 0.19 threshold, on both checkpoints.

    ./.venv/bin/python 7_Run_PUPS_Inference.py \
        --variants TRACK/work/variants --embed TRACK/esm2_output \
        --out TRACK/work/predictions

REQUIRES <track>/esm2_output/ — produced by 5_Make_ESM2_Embeddings.py --track <TRACK>.
Everything else is local. CPU only; ~73 ms/sample, so the full grid is a couple of minutes.

THE GRID
--------
    3 input stacks (composite, single_jpg, single_tiff)
  x 2 train/test-consistent checkpoint/threshold pairs (see CONSISTENT below)
  = 6 conditions per tile.  --all-conditions adds the 6 inconsistent pairings.

`composite` with the threshold ON is PUPS's published configuration. Naming
conditions "<checkpoint>__<variant>__thr{0,1}" keeps every combination addressable rather than
privileging one; the figure picks from the full set at writing time.

THE 0.19 THRESHOLD
------------------
Methods: "All landmark stain channels are additionally filtered for low-intensity pixels, with
a threshold of 0.19 to remove bleed-through from the targeting protein channel due to the
overlapping spectra (Extended Data Fig. 5)."

So it applies to the LANDMARK channels only (nucleus, microtubule, ER) and zeroes pixels below
0.19. It is applied AFTER the per-patch rescale_intensity, since the value is on the [0,1]
scale that rescaling produces. The protein channel is the target and is never thresholded.

⚠ THE CHECKPOINTS ARE NOT TRAINED EQUALLY LONG
   default      epoch 1, global_step 3548
   nothreshold  epoch 0, global_step 2661   (~25% less)
   Any default-vs-nothreshold difference therefore conflates the threshold with training
   duration. The within-checkpoint, across-format comparison is unaffected and should carry
   the argument. Step 7 repeats this warning where it matters.

ONLY CHANNELS 0-2 GO IN. Channel 3 (protein) is the prediction target and is copied to the
output for step 8 to score against. Feeding it to the model would leak the answer.
"""
import argparse
import csv
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "PUPS_Code"))

MAX_SEQ_LEN = 2000          # PUPS esm2_utils.MAX_SEQ_LEN
EMB_DIM = 1280
THRESHOLD = 0.19            # PUPS Methods
LANDMARK_SLICE = slice(0, 3)    # nucleus, microtubule, ER
STACKS = ["composite", "single_jpg", "single_tiff"]   # must match 6_Build_Input_Variants.py
PROTEIN_IDX = 3

CHECKPOINTS = {
    "default": "splice_isoform_dataset_cell_line_and_gene_split_full-epoch=01-val_combined_loss=0.18.ckpt",
    "nothreshold": "nothreshold.ckpt",
}

# The train/test-consistent pairings, and the only ones any figure uses:
#   nothreshold checkpoint <-> threshold off    (ED Fig. 5b, top panel)
#   default checkpoint     <-> threshold 0.19   (ED Fig. 5b, bottom panel; published setting)
# The other two combinations feed a model landmarks it was not trained on. They are ablations,
# not "PUPS with/without the threshold", and are computed only with --all-conditions.
CONSISTENT = {("nothreshold", 0), ("default", 1)}

def load_embeddings(embed_dir):
    """gene -> (1, 2000, 1280) float32 tensor, plus gene -> x_len.

    Step 5 stores representations UNPADDED as (L+2, 1280) to avoid writing ~350 MB of
    zeros. Padding to MAX_SEQ_LEN happens here. The BOS/EOS rows are included, matching PUPS:
    its get_esm2_representation passes the untrimmed aa_rep to pad_collate, so x_len == L + 2.

    The 4-D shape is required: LightAttentionNN.forward does
    x.reshape(x.shape[0], x.shape[2], x.shape[3]), so a singleton axis must sit at dim 1.
    PUPS's own pad_collate emits 3-D, so something in their pipeline unsqueezes; we do it here.
    """
    embed_dir = pathlib.Path(embed_dir)
    rep_path = embed_dir / "esm2_representations.npz"
    if not rep_path.exists():
        sys.exit(f"embeddings not found: {rep_path}\n"
                 f"  Run 5_Make_ESM2_Embeddings.py --track <TRACK> first.")
    reps = np.load(rep_path)
    lens = np.load(embed_dir / "esm2_lengths.npz")

    out = {}
    for gene in reps.files:
        r = reps[gene]
        assert r.ndim == 2 and r.shape[1] == EMB_DIM, f"{gene}: expected (L,1280), got {r.shape}"
        assert r.shape[0] <= MAX_SEQ_LEN, f"{gene}: {r.shape[0]} rows exceeds MAX_SEQ_LEN"
        padded = np.zeros((MAX_SEQ_LEN, EMB_DIM), dtype=np.float32)
        padded[:r.shape[0]] = r
        out[gene] = (torch.from_numpy(padded)[None, None],      # (1, 1, 2000, 1280)
                     float(lens[gene]))
    return out


def apply_threshold(stack):
    """Zero landmark pixels below 0.19. Protein channel (target) untouched."""
    out = stack.copy()
    lm = out[LANDMARK_SLICE]
    lm[lm < THRESHOLD] = 0.0
    out[LANDMARK_SLICE] = lm
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", required=True, help="output dir of step 6, e.g. TRACK/work/variants")
    ap.add_argument("--embed", required=True, help="<track>/esm2_output from step 5")
    ap.add_argument("--out", required=True, help="e.g. TRACK/work/predictions")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--all-conditions", action="store_true",
                    help="also compute the two train/test-inconsistent pairings (ablations)")

    args = ap.parse_args()

    from src.model.full_model import SubCellProtModel

    var_dir = pathlib.Path(args.variants)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = var_dir / "index.csv"
    if not index_path.exists():
        sys.exit(f"{index_path} not found — run 6_Build_Input_Variants.py first")

    tiles = list(csv.DictReader(open(index_path, newline="")))
    if args.limit:
        tiles = tiles[:args.limit]

    print("STEP 7 — PUPS inference across all conditions")
    embeddings = load_embeddings(args.embed)
    print(f"  embeddings: {len(embeddings)} genes from {args.embed}")

    missing_genes = sorted({t["gene"] for t in tiles} - set(embeddings))
    if missing_genes:
        print(f"  WARNING: no embedding for {len(missing_genes)} gene(s): {missing_genes}")
        print( "    Those tiles will be skipped. Check <track>/esm2_output/sequences.json —")
        print( "    a gene is absent if Ensembl had no protein-coding transcript for it.")
        tiles = [t for t in tiles if t["gene"] in embeddings]
        if not tiles:
            sys.exit("no tiles left after dropping those without an embedding — check "
                     "<track>/sequences.json and <track>/esm2_output/.")

    models = {}
    for name, fn in CHECKPOINTS.items():
        p = ROOT / "PUPS_Code" / "checkpoints" / fn
        if not p.exists():
            sys.exit(f"checkpoint missing: {p} — see PIPELINE.md step 0 (git clone)")
        m = SubCellProtModel()
        ck = torch.load(p, map_location="cpu", weights_only=False)
        miss, unexp = m.load_state_dict(ck["state_dict"], strict=False)
        assert not miss and not unexp, f"{name}: unclean load ({len(miss)}/{len(unexp)})"
        m.eval()
        models[name] = m
        print(f"  checkpoint '{name}': epoch {ck.get('epoch')}, step {ck.get('global_step')}")
    print("  NOTE: the two checkpoints differ in training length — see the module docstring.\n")

    variant_names = None
    t_start = time.perf_counter()
    n_cond = 0

    for i, t in enumerate(tiles, 1):
        tile_id = t["tile_id"]
        npz = np.load(var_dir / f"{tile_id}.npz", allow_pickle=True)
        if variant_names is None:
            variant_names = [k for k in npz.files if k in STACKS]
            print(f"  variants found: {variant_names}")
        esm2, x_len = embeddings[t["gene"]]
        lens = torch.tensor([x_len], dtype=torch.float32)

        results = {}
        for vname in variant_names:
            stack = npz[vname].astype(np.float32)
            # Target is identical across threshold settings; store once per variant.
            results[f"target__{vname}"] = stack[PROTEIN_IDX]
            for thr in (0, 1):
                arr = apply_threshold(stack) if thr else stack
                stains = torch.from_numpy(arr[LANDMARK_SLICE])[None]   # (1, 3, 128, 128)
                for ck_name, model in models.items():
                    if not args.all_conditions and (ck_name, thr) not in CONSISTENT:
                        continue
                    with torch.no_grad():
                        img, lab = model.call_model(esm2, lens, stains)

                    key = f"{ck_name}__{vname}__thr{thr}"
                    results[key] = img[0, 0].numpy().astype(np.float32)
                    results[f"label__{key}"] = lab[0].numpy().astype(np.float32)
                    n_cond += 1

        np.savez_compressed(out_dir / f"{tile_id}.npz", **results)
        if i % 10 == 0 or i == len(tiles):
            print(f"  [{i}/{len(tiles)}] {tile_id}")

    # Carry the index forward so downstream steps need only one directory.
    with open(out_dir / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tiles[0].keys()))
        w.writeheader()
        w.writerows(tiles)

    dt = time.perf_counter() - t_start
    print(f"\n  {len(tiles)} tiles x {n_cond // max(len(tiles), 1)} conditions "
          f"= {n_cond} forward passes in {dt:.0f} s")
    print(f"  wrote -> {out_dir}")
    print("\nSTEP 7 COMPLETE")
    print("  NEXT: ./.venv/bin/python 8_Compute_Metrics.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
