#!/usr/bin/env python3
"""STEP 11 — why does the same field of view give a dimmer prediction from a JPG than a TIFF?

    ./.venv/bin/python 11_Ablate_Channels.py --variants Fig_2C/work/variants \
        --embed Fig_2C/esm2_output --out Fig_2C/work/ablation
    ./.venv/bin/python 11_Ablate_Channels.py ... --blocks --images-root Fig_2C/Images \
        --roi Fig_2C/ROI_examples.txt

THE QUESTION THIS ANSWERS
-------------------------
`single_jpg` and `single_tiff` are the same fields of view differing only in bit depth and JPEG
compression. After step 6 they have the SAME mean intensity to within 2% and correlate at
r = 0.992-0.9996, and they look identical in a preview. Yet PUPS predicts from `single_jpg`
at roughly a third of the amplitude. That looks like a preprocessing bug, so it needs an
explanation that is measured rather than argued.

The hypothesis this tests is that they differ in SUPPORT rather than in total signal:

    the mean is the same; the fraction of pixels carrying ANY signal is not.

JPEG quantises 8x8 DCT blocks, so dark blocks go uniformly to exact zero. PUPS's `//4`
downsample averages 4x4 neighbourhoods: a TIFF's scattered small values survive that averaging,
an all-zero JPEG block cannot. `--blocks` measures this directly on the source files.

WHAT IT REPORTS
---------------
1. SUPPORT vs OUTPUT. Per tile: the single_jpg/single_tiff ratio of input mean, of input support (fraction of
   nonzero pixels in the landmark channels), and of output amplitude. Reported together so the
   two candidate explanations separate: if amplitude followed total signal, the output ratio
   would sit near the mean ratio.

2. CHANNEL SWAP ABLATION. Feed single_tiff with one channel replaced by single_jpg's, to localise which
   landmark is responsible. Established 2026-09-06 on Fig_2C: nucleus and microtubule each
   cause a large drop and compound; swapping ER moves the output by under 1%, i.e.
   **the model effectively ignores its third landmark channel** — the one PUPS calls
   `mitochondria_stain` while HPA's XML says ER.

3. --blocks: the fraction of k x k blocks that are entirely zero, in the raw JPG and the raw
   TIFF, which is the mechanism behind (1).

ACTIONABLE: this is a diagnostic, not part of the headline result. `single_jpg` is OUR
construction — PUPS never reads single-channel JPGs, it reads the composite. Comparing it against
`single_tiff` is a controlled probe that separates compositing from 8-bit quantisation. The claim
that bears on the paper is `composite` vs `single_tiff`.
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "PUPS_Code"))

MAX_SEQ_LEN = 2000
EMB_DIM = 1280
THRESHOLD = 0.19
CHANNELS = ["nucleus", "microtubule", "er", "protein"]
LANDMARKS = CHANNELS[:3]
LANDMARK_SLICE = slice(0, 3)
CHECKPOINTS = {
    "default": "splice_isoform_dataset_cell_line_and_gene_split_full-epoch=01-val_combined_loss=0.18.ckpt",
    "nothreshold": "nothreshold.ckpt",
}
BASE, PROBE = "single_tiff", "single_jpg"     # baseline, and the arm being explained


def load_model(name):
    import torch
    from src.model.full_model import SubCellProtModel
    ck = ROOT / "PUPS_Code" / "checkpoints" / CHECKPOINTS[name]
    if not ck.exists():
        sys.exit(f"missing checkpoint {ck.name}\n  run 0_Get_Checkpoints.py first.")
    m = SubCellProtModel()
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    m.load_state_dict(sd["state_dict"])
    m.eval()
    return m, sd.get("epoch"), sd.get("global_step")


def load_embeddings(embed_dir):
    """Same padding contract as step 7: unpadded (L+2, 1280) on disk -> (1, 1, 2000, 1280)."""
    import torch
    d = pathlib.Path(embed_dir)
    reps, lens = np.load(d / "esm2_representations.npz"), np.load(d / "esm2_lengths.npz")
    out = {}
    for g in reps.files:
        pad = np.zeros((MAX_SEQ_LEN, EMB_DIM), np.float32)
        r = reps[g][:MAX_SEQ_LEN]
        pad[:r.shape[0]] = r
        out[g] = (torch.from_numpy(pad)[None, None], float(lens[g]))
    return out


def predict(model, emb, gene, stack3, threshold):
    """One forward pass on a 3-channel landmark stack. Identical to step 7's path."""
    import torch
    if gene not in emb:
        return None
    esm2, ln = emb[gene]
    s = np.ascontiguousarray(stack3, dtype=np.float32)
    if threshold:
        s = s.copy()
        s[s < THRESHOLD] = 0.0
    with torch.no_grad():
        img, _ = model.call_model(esm2, torch.tensor([ln]), torch.from_numpy(s)[None])
    return np.clip(img[0, 0].numpy(), 0.0, 1.0)


def support(a):
    """Fraction of pixels carrying any signal. The quantity that turns out to matter."""
    return float((a > 1e-6).mean())


def block_zero_fraction(a, k):
    """Fraction of k x k blocks that are ENTIRELY zero — isolated zeros vs flat dead blocks."""
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    bl = a[:h, :w].reshape(h // k, k, w // k, k).transpose(0, 2, 1, 3).reshape(-1, k * k)
    return float((bl.max(1) == 0).mean())


def blocks_report(images_root, roi_file, out_dir, channel="blue"):
    """The mechanism: where the zeros are in the raw files, before any of our preprocessing."""
    from PIL import Image
    import tifffile
    LUT = {"blue": 2, "red": 0, "green": 1}
    rows = []
    with open(roi_file, newline="") as fh:
        entries = [r for r in csv.DictReader(fh)
                   if (r.get("Skip") or "F").strip().upper() != "T"]
    print(f"\n=== raw-file block analysis, {channel} channel, {len(entries)} field(s) of view ===")
    print(f"  {'block':>7s} {'JPG all-zero':>13s} {'TIFF all-zero':>14s} {'gap':>9s}")
    acc = {}
    for r in entries:
        stem = (pathlib.Path(images_root) / r["FolderPath"].strip().replace("\\", "/")
                / r["ImagePrefix"].strip())
        jp, tp = f"{stem}_{channel}.jpg", f"{stem}_{channel}.tif"
        if not (pathlib.Path(jp).exists() and pathlib.Path(tp).exists()):
            continue
        j = np.array(Image.open(jp))[..., LUT[channel]].astype(np.float32) / 255.0
        with tifffile.TiffFile(tp) as tf:
            arr = tf.pages[0].asarray()
        t = arr.astype(np.float32) / (65535.0 if arr.dtype == np.uint16 else 255.0)
        for k in (1, 2, 4, 8, 16):
            acc.setdefault(k, []).append((block_zero_fraction(j, k),
                                          block_zero_fraction(t, k)))
    for k in sorted(acc):
        m = np.array(acc[k]).mean(0) * 100
        print(f"  {k:>4d}x{k:<2d} {m[0]:12.2f}% {m[1]:13.2f}% {m[0]-m[1]:+8.2f}pp")
        rows.append({"block": f"{k}x{k}", "jpg_all_zero_pct": round(m[0], 3),
                     "tiff_all_zero_pct": round(m[1], 3), "gap_pp": round(m[0] - m[1], 3)})
    print("\n  all-zero fraction by block size, JPG vs TIFF, same field of view.")
    print("  PUPS's `//4` downsample averages 4x4 neighbourhoods, so the 4x4 row is the one")
    print("  that reaches the model input.")
    if rows:
        with open(pathlib.Path(out_dir) / "block_zeros.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", required=True, help="<track>/work/variants from step 6")
    ap.add_argument("--embed", required=True, help="<track>/esm2_output from step 5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default="default", choices=sorted(CHECKPOINTS))
    ap.add_argument("--threshold", type=int, default=1, choices=[0, 1],
                    help="apply PUPS's 0.19 landmark threshold to the input. Default 1, which "
                         "pairs with the `default` checkpoint — the published setting and the "
                         "only train/test-consistent pairing for that checkpoint. Use 0 with "
                         "--checkpoint nothreshold.")

    ap.add_argument("--limit", type=int, default=None, help="first N tiles only")
    ap.add_argument("--blocks", action="store_true",
                    help="also run the raw-file block analysis (needs --images-root and --roi)")
    ap.add_argument("--images-root", default=None)
    ap.add_argument("--roi", default=None)
    args = ap.parse_args()

    var = pathlib.Path(args.variants)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tiles = sorted(var.glob("*.npz"))

    # index.csv is the authoritative tile -> gene mapping. Parsing the gene out of the filename
    # breaks for any symbol containing an underscore, and does so silently.
    gene_by_tile = {}
    if (var / "index.csv").exists():
        with open(var / "index.csv", newline="") as fh:
            gene_by_tile = {r["tile_id"]: r["gene"] for r in csv.DictReader(fh)}

    if args.limit:
        tiles = tiles[:args.limit]
    if not tiles:
        sys.exit(f"no .npz tiles in {var}")

    model, epoch, step = load_model(args.checkpoint)
    emb = load_embeddings(args.embed)
    print(f"STEP 11 — channel ablation on {len(tiles)} tile(s)")
    print(f"  checkpoint '{args.checkpoint}': epoch {epoch}, step {step}")
    print(f"  threshold : {'on (0.19)' if args.threshold else 'off'}")
    print(f"  baseline {BASE}  vs probe {PROBE}\n")

    rows, ratios = [], []
    for p in tiles:
        Z = np.load(p, allow_pickle=True)
        if BASE not in Z.files or PROBE not in Z.files:
            print(f"  {p.stem}: missing {BASE} or {PROBE}, skipped"); continue
        gene = gene_by_tile.get(p.stem) or p.stem.split("_")[0]
        base, probe = Z[BASE][LANDMARK_SLICE], Z[PROBE][LANDMARK_SLICE]

        conds = {BASE: base, PROBE: probe}
        for i, c in enumerate(LANDMARKS):                 # one channel swapped in
            s = base.copy(); s[i] = probe[i]
            conds[f"{BASE}_with_{c}_from_{PROBE}"] = s

        preds = {}
        for name, s in conds.items():
            a = predict(model, emb, gene, s, args.threshold)
            if a is None:
                print(f"  {p.stem}: no embedding for {gene}, skipped"); break
            preds[name] = a
            rows.append({"tile_id": p.stem, "gene": gene, "condition": name,
                         "out_mean": round(float(a.mean()), 6),
                         "out_max": round(float(a.max()), 6),
                         "in_support": round(support(s), 6),
                         "in_mean": round(float(s.mean()), 6)})
        if len(preds) != len(conds):
            continue

        ob, op = preds[BASE].mean(), preds[PROBE].mean()
        ratios.append((p.stem, probe.mean() / base.mean(),
                       support(probe) / support(base), op / ob if ob else np.nan))
        print(f"  {p.stem}")
        print(f"    {'baseline ' + BASE:44s} mean={ob:.4f} max={preds[BASE].max():.4f}")
        for c in LANDMARKS:
            k = f"{BASE}_with_{c}_from_{PROBE}"
            d = (preds[k].mean() - ob) / ob * 100 if ob else np.nan
            print(f"    {BASE} but {c:11s} <- {PROBE:14s} mean={preds[k].mean():.4f} "
                  f"({d:+6.1f}%)")
        print(f"    {'all three (' + PROBE + ')':44s} mean={op:.4f} max={preds[PROBE].max():.4f}")

    if not rows:
        sys.exit("no tiles were scored — check that the variants directory and the embeddings "
                 "cover the same genes.")
    with open(out / "ablation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    
    if ratios:
        print(f"\n=== does output amplitude track input MEAN, or input SUPPORT? ===")
        print(f"  {'tile':38s} {'mean ratio':>11s} {'support ratio':>14s} {'OUTPUT ratio':>13s}")
        for tid, mr, sr, orr in ratios:
            print(f"  {tid[:38]:38s} {mr:11.3f} {sr:14.3f} {orr:13.3f}")
        a = np.array([[m, s, o] for _, m, s, o in ratios])
        print(f"  {'MEAN':38s} {a[:,0].mean():11.3f} {a[:,1].mean():14.3f} {a[:,2].mean():13.3f}")
        print("\n  mean ratio    = single_jpg / single_tiff, input channel mean")
        print("  support ratio = same ratio for the fraction of nonzero pixels")
        print("  OUTPUT ratio  = same ratio for the prediction mean")

    print(f"\n  per-condition rows -> {out / 'ablation.csv'}")

    if args.blocks:
        if not (args.images_root and args.roi):
            sys.exit("--blocks needs --images-root and --roi")
        blocks_report(args.images_root, args.roi, out)

    print("\nSTEP 11 COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
