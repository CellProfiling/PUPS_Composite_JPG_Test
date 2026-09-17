#!/usr/bin/env python3
"""STEP 8 — Score every condition and write the result tables.

    ./.venv/bin/python 8_Compute_Metrics.py \
        --pred TRACK/work/predictions --variants TRACK/work/variants --out TRACK/work/metrics

Produces one long-format CSV of per-tile results plus aggregated summaries, for EVERY
condition. Nothing is pre-filtered: the figure scripts and the final write-up choose from the
complete set.

WHAT IS SCORED AGAINST WHAT
---------------------------
Ground truth is the protein channel of single_tiff — the least-processed version of the same tile.
Every prediction, whatever variant produced it, is scored against that one reference, so the
comparison across variants is like-for-like.

  MSE       the paper's own metric, for direct comparability. They report 0.00705 / 0.00960
            for holdouts 1 / 2 against a random baseline of 0.408 / 0.412.
  Pearson r scale-invariant, so it survives the per-patch rescaling
  SSIM      structural agreement, less dominated by bright pixels than MSE
  intranuc  intra-nuclear proportion — the fraction of predicted protein signal inside the
            nuclear mask. This is the paper's Fig. 3a readout, and it is the one that shows
            whether the format changes a BIOLOGICAL CONCLUSION rather than just a loss number.

CLIPPING RULE — decided here, applied everywhere
------------------------------------------------
PUPS has no output activation bounding its predictions, while targets are rescaled to [0,1]
(observed range on synthetic input: [0.38, 1.64]). Predictions are therefore CLIPPED to [0,1]
before every metric. Uniform across all conditions, so it cannot favour one variant. The raw
unclipped range is recorded per row as pred_min / pred_max so the effect stays auditable.

STRATIFY BY tif_dtype — this is not optional
--------------------------------------------
In `Independent_Selection` — 120 ROI rows, of which 117 are live once the three marked `Skip=T`
are dropped — 97 tiles have uint16 TIFFs (~62,800 distinct levels) and 20 have uint8 TIFFs
(256 levels); size predicts dtype exactly. On the uint8 tiles the TIFF variant carries NO bit-depth advantage
over the JPGs — only freedom from compositing. So the two strata separate "8-bit quantisation"
from "composite mixing", and pooling them would blur exactly the distinction we are making.
Summaries are emitted overall AND per stratum.

INPUT FIDELITY (diagnostic)
---------------------------
Also scores each variant's INPUT channels against single_tiff's, before any model runs. Without
this a downstream difference cannot be attributed to a cause: it shows how far each format's
nucleus/microtubule/ER actually deviate, per channel.
"""
import argparse
import csv
import pathlib
import re
import sys

import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.metrics import structural_similarity

CHANNELS = ["nucleus", "microtubule", "er", "protein"]
GROUND_TRUTH_VARIANT = "single_tiff"
STACKS = ["composite", "single_jpg", "single_tiff"]   # must match 6_Build_Input_Variants.py

# Condition keys written by step 7 look like "default__single_jpg__thr1". The npz also holds
# "label__default__single_jpg__thr1" (the 29-class head) and "target__<stack>", which must NOT
# match: a label row is shape (29,) and would blow up against a (128,128) target.
# `[a-z]+(?:_[a-z]+)*` forbids CONSECUTIVE underscores, which is what separates the two —
# a plain `[a-z_]+` variant group happily swallows "default__single_jpg" out of a label key.
COND_RE = re.compile(r"^(?P<ckpt>[a-z]+)__(?P<variant>[a-z]+(?:_[a-z]+)*)__thr(?P<thr>[01])$")

def pearson(a, b):
    a, b = a.ravel(), b.ravel()
    if a.std() == 0 or b.std() == 0:
        return np.nan            # a constant image has no correlation defined
    return float(np.corrcoef(a, b)[0, 1])


def nuclear_mask(nucleus):
    """Otsu mask of the nucleus channel, as PUPS does for its intra-nuclear proportion.

    Returns None when Otsu cannot split the histogram (e.g. a near-empty crop), so the caller
    records NaN rather than a fabricated number.
    """
    try:
        if nucleus.std() == 0:
            return None
        return nucleus > threshold_otsu(nucleus)
    except ValueError:
        return None


def intranuclear_proportion(img, mask):
    """Fraction of total signal inside the nucleus. PUPS Fig. 3a readout."""
    if mask is None or mask.sum() == 0:
        return np.nan
    total = float(img.sum())
    return float(img[mask].sum() / total) if total > 0 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="e.g. TRACK/work/predictions")
    ap.add_argument("--variants", required=True, help="e.g. TRACK/work/variants")
    ap.add_argument("--out", required=True, help="e.g. TRACK/work/metrics")

    args = ap.parse_args()

    pred_dir, var_dir = pathlib.Path(args.pred), pathlib.Path(args.variants)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not (pred_dir / "index.csv").exists():
        sys.exit(f"{pred_dir}/index.csv not found — run 7_Run_PUPS_Inference.py first")

    tiles = list(csv.DictReader(open(pred_dir / "index.csv", newline="")))
    print(f"STEP 8 — scoring {len(tiles)} tiles\n")

    pred_rows, input_rows = [], []

    for t in tiles:
        tile_id = t["tile_id"]
        P = np.load(pred_dir / f"{tile_id}.npz")
        V = np.load(var_dir / f"{tile_id}.npz", allow_pickle=True)

        truth = P[f"target__{GROUND_TRUTH_VARIANT}"]
        mask = nuclear_mask(V[GROUND_TRUTH_VARIANT][CHANNELS.index("nucleus")])
        truth_intranuc = intranuclear_proportion(truth, mask)

        meta = {"tile_id": tile_id, "gene": t["gene"], "cell_line": t["cell_line"],
                "annotation": t.get("annotation", ""),
                "full_size": t.get("full_size", ""), "tif_dtype": t.get("tif_dtype", "")}

        # ---- input fidelity: how far is each variant's INPUT from the TIFF input?
        #
        # Whole-patch Pearson r is NOT sufficient here: it runs 0.99+ for every variant because
        # bright structure dominates, while the artifact lives in the low-intensity range. The
        # discriminating measure is the RESIDUAL MAP, the same device the companion project
        # uses (github.com/cellprofiling/HPA_BleedThrough_Exploration, its Figure 2):
        #
        #     residual = rank(variant_channel) - rank(tiff_channel)
        #     leak_r   = corr(residual, tiff_PROTEIN_channel)
        #
        # If splitting the composite pushes protein signal into the nucleus or microtubule
        # channels, that excess is SHAPED LIKE THE PROTEIN STAIN, so leak_r is positive.
        # The ER column is NOT a compositing test: `composite` takes ER from the single-channel
        # yellow.jpg, exactly as PUPS does, so nothing in that channel passed through the
        # composite. Read the nucleus and microtubule columns only.
        # The residual is rank-normalised first, so that a monotone 8-bit highlight-compression
        # difference between single-channel formats does NOT register as leakage; only
        # protein-shaped structure does. leak_r is an internal diagnostic and is not a reported
        # figure in the manuscript.
        truth_protein = V[GROUND_TRUTH_VARIANT][CHANNELS.index("protein")]
        for vname in [k for k in V.files if k in STACKS]:
            for ci, ch in enumerate(CHANNELS):
                a, b = V[vname][ci], V[GROUND_TRUTH_VARIANT][ci]
                residual = a - b
                # leak_r is computed on a RANK-NORMALISED residual so that a monotone 8-bit
                # tone-mapping difference (highlight compression) cannot masquerade as leakage;
                # only protein-shaped excess survives rank-matching. rmse and residual_mean_abs
                # stay on the raw residual, where they are the honest quantities.
                ra = pd.Series(a.ravel()).rank().to_numpy().reshape(a.shape)
                rb = pd.Series(b.ravel()).rank().to_numpy().reshape(b.shape)
                input_rows.append({
                    **meta, "variant": vname, "channel": ch,
                    "pearson_r": pearson(a, b),
                    "rmse": float(np.sqrt(np.mean(residual ** 2))),
                    "ssim": float(structural_similarity(a, b, data_range=1.0)),
                    "leak_r": pearson(ra - rb, truth_protein),
                    "residual_mean_abs": float(np.abs(residual).mean()),
                })

        # ---- prediction quality per condition
        for key in P.files:
            m = COND_RE.match(key)
            if not m:
                continue                       # target__* and label__* rows
            raw = P[key]
            pred = np.clip(raw, 0.0, 1.0)      # see CLIPPING RULE
            pred_rows.append({
                **meta,
                "checkpoint": m["ckpt"], "variant": m["variant"], "threshold": int(m["thr"]),
                "condition": key,
                "mse": float(np.mean((pred - truth) ** 2)),
                "pearson_r": pearson(pred, truth),
                "ssim": float(structural_similarity(pred, truth, data_range=1.0)),
                "intranuclear_pred": intranuclear_proportion(pred, mask),
                "intranuclear_truth": truth_intranuc,
                "pred_min_raw": float(raw.min()), "pred_max_raw": float(raw.max()),
                "clipped_frac": float(np.mean((raw < 0) | (raw > 1))),
            })

    df = pd.DataFrame(pred_rows)
    di = pd.DataFrame(input_rows)
    # Derive before writing, so per_tile_predictions.csv is self-contained and any downstream
    # filtering (e.g. to a matched subset) does not have to recompute it.
    df["intranuclear_abs_err"] = (df.intranuclear_pred - df.intranuclear_truth).abs()
    df.to_csv(out / "per_tile_predictions.csv", index=False)
    di.to_csv(out / "per_tile_input_fidelity.csv", index=False)
    print(f"  per-tile rows: {len(df)} predictions, {len(di)} input-fidelity")

    # intra-nuclear ERROR (derived above) is the biologically meaningful summary,
    # not the raw proportion.
    metrics = ["mse", "pearson_r", "ssim", "intranuclear_abs_err", "clipped_frac"]
    group = ["checkpoint", "variant", "threshold"]

    summary = df.groupby(group)[metrics].agg(["mean", "std", "median"]).round(6)
    summary.to_csv(out / "summary_overall.csv")

    strat = df.groupby(group + ["tif_dtype"])[metrics].agg(["mean", "median", "count"]).round(6)
    strat.to_csv(out / "summary_by_tif_dtype.csv")

    di.groupby(["variant", "channel"])[["pearson_r", "rmse", "ssim", "leak_r",
                                        "residual_mean_abs"]] \
      .agg(["mean", "median"]).round(6).to_csv(out / "summary_input_fidelity.csv")
    di.groupby(["variant", "channel", "tif_dtype"])[["leak_r", "rmse"]] \
      .agg(["mean", "count"]).round(6).to_csv(out / "summary_leakage_by_tif_dtype.csv")

    print(f"\n=== prediction quality vs {GROUND_TRUTH_VARIANT} protein channel "
          f"(lower MSE better, higher r/SSIM better) ===")
    print(df.groupby(group)[["mse", "pearson_r", "ssim", "intranuclear_abs_err"]]
            .mean().round(5).to_string())

    # Only worth printing where it separates anything: on a single-dtype track the stratified
    # table is the table above with an extra column. The CSV is written either way.
    if df.tif_dtype.nunique() > 1:
        print("\n=== stratified by TIFF bit depth ===")
        print(df.groupby(group + ["tif_dtype"])[["mse", "pearson_r"]]
                .agg(["mean", "count"]).round(5).to_string())
    else:
        print(f"\n  every tile is {df.tif_dtype.iloc[0]} — no bit-depth stratum to separate "
              f"(summary_by_tif_dtype.csv written anyway)")

    # One line, not a table: r ~ 0.99 everywhere is the POINT, and it takes one line to say.
    lo, hi = di[di.variant != GROUND_TRUTH_VARIANT].pearson_r.agg(["min", "max"])
    print(f"\n  input fidelity vs {GROUND_TRUTH_VARIANT}: Pearson r = {lo:.3f}-{hi:.3f} over every "
          f"tile, stack and channel — bright structure dominates, so r cannot discriminate "
          f"formats.\n  (per-channel means: summary_input_fidelity.csv; use leak_r below instead)")

    print("\n=== LEAKAGE: corr(rank-residual vs TIFF, true protein channel) — diagnostic ===")
    print(di.pivot_table(index="variant", columns="channel",
                         values="leak_r", aggfunc="mean").round(5).to_string())
    print("  rank-normalised, so monotone 8-bit tone-mapping differences do NOT count as leakage")
    print("  >0 in a LANDMARK channel = protein-shaped EXCESS (compositing mixing protein in)")
    print("  <0 = protein-shaped DEFICIT (compression zeroing dim protein-correlated pixels)")
    print("  the protein column is a control: it is never a model input and should sit near 0")

    frac = df.clipped_frac.mean()
    print(f"\n  mean fraction of predicted pixels outside [0,1] before clipping: {frac:.4f}")
    if frac > 0.05:
        print("    ACTIONABLE: >5% clipped. Report this; it means the model is routinely "
              "predicting out of range and MSE is partly a clipping artefact.")

    print(f"\n  tables -> {out}")
    print("\nSTEP 8 COMPLETE")
    print("  NEXT: ./.venv/bin/python 9_Figure_Formats.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
