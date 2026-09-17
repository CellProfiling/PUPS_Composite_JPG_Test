#!/usr/bin/env python3
"""STEP 6 — Build the four input variants for every ROI tile.

Reads the ROI coordinates from the companion HPA bleed-through project, extracts the same
tiles from the original image files in four different source formats, and applies PUPS's
exact preprocessing to all of them identically.

    # first pass: the 6 hand-picked example tiles
    ./.venv/bin/python 6_Build_Input_Variants.py \
        --images-root Independent_Selection/Images --roi Independent_Selection/ROI_examples_Selected.txt \
        --out Independent_Selection/work/variants_selected

    # full set: all 120 tiles
    ./.venv/bin/python 6_Build_Input_Variants.py \
        --images-root Independent_Selection/Images --roi Independent_Selection/ROI.txt \
        --out Independent_Selection/work/variants

Needs no GPU, no model, and no ESM-2 embeddings. Runs in seconds.

WHY WE RE-EXTRACT INSTEAD OF REUSING Crops/*.png
------------------------------------------------
The companion project's `Crops/*.png` are DISPLAY RENDERS (640x640, 8-bit, adjustments baked
in -- the _adj / _raw / _unadj suffixes). Feeding those to PUPS would stack another processing
layer on top of the very artifact being measured, and the adjusted ones are not linear in the
original intensities. So we reuse only the ROI COORDINATES and re-extract from the original
.tif / .jpg. The tiles are therefore spatially identical to the published bleed-through figure.

THE VARIANTS  (channel order in every saved stack: nucleus, microtubule, ER, protein)
------------------------------------------------------------------------------------------
  composite      RGB split of <fov>_blue_red_green.jpg  + yellow.jpg   <- what PUPS reads
  single_jpg     <fov>_{blue,red,green}.jpg (LUT channel) + yellow.jpg
  single_tiff    <fov>_{blue,red,green,yellow}.tif  (16-bit)

The 0.19 threshold is not a stack. Step 7 applies it to `composite` at load time; that
combination is PUPS's published configuration.

ONLY THE FIRST THREE CHANNELS ARE MODEL INPUT. The protein (green) channel is the prediction
TARGET and is saved alongside purely as ground truth. Never feed channel 3 to the model.

PREPROCESSING -- matches PUPS exactly; order matters
----------------------------------------------------
PUPS downsamples the WHOLE 2048x2048 image and only then cuts the patch
(get_image_and_resize -> the crop loop in find_centers_and_crop). Cropping first and resizing
afterwards is NOT equivalent: it aligns the 4x4 averaging grid to the crop origin rather than
the image origin, shifting the result by a sub-pixel amount unless Roi_X/Roi_Y are multiples
of 4. So, per channel:

    normalise to float [0,1]
    skimage.transform.resize(img, (512, 512))        # //4, skimage defaults, as PUPS
    crop 128x128 at row = Roi_Y//4 + 16, col = Roi_X//4 + 16
    skimage.exposure.rescale_intensity(patch, out_range=(0,1))   # PUPS process_patch

Geometry: PUPS's U-Net is hard-locked to 128x128 (nn_unet.py: image_input_dim=128,
assert image_dim == 128). Because the resize happens first, one model input corresponds to a
512x512 region at full resolution. The ROI tiles are 640x640, so nothing needs enlarging --
we take the central 512x512, hence the "+16" offset (640/2/4 - 64 = 16).

TWO FILE-FORMAT TRAPS, both handled below
-----------------------------------------
  1. The .tif files carry OME metadata referencing a multi-file series that does not exist
     ('image--X00--Y00--Z00--C00.ome.tif'). tifffile.imread() raises ValueError and prints
     "Missing data are zeroed". We read pages[0] directly. ACTIONABLE: never switch this to
     a plain imread() -- a future tifffile version could zero-fill silently instead of raising.
  2. HPA's "single-channel" JPGs are 3-component RGB with a colour LUT applied, NOT greyscale.
     Measured on 776_F2_1: blue.jpg has per-channel max [19, 14, 255]; red.jpg [255, 40, 49];
     green.jpg [104, 239, 101]; yellow.jpg [255, 255, 167]. So we take the LUT channel
     (blue->B, red->R, green->G) rather than a luminance conversion, which would scale blue by
     0.114 and crush it ~9x. Yellow is a two-channel LUT with R ~= G (mean |R-G| = 1.3), so we
     average R and G.
"""
import argparse
import csv
import pathlib
import sys

import numpy as np
import tifffile
from PIL import Image, ImageOps
from skimage.exposure import rescale_intensity
from skimage.transform import resize

# HPA fields of view are NOT all the same size. Measured across Independent_Selection/ROI.txt:
#   2048x2048 -> 97 tiles      1728x1728 -> 20 tiles      (its 117 live rows of 120)
# PUPS is size-agnostic (get_image_and_resize does `ti // 4` on the actual image shape), so we
# are too: the full size is read per file and never hard-coded.
PATCH = 128            # PUPS model input (nn_unet.py image_input_dim, assert image_dim == 128)
ROI_SIZE = 640         # 2_Square_Selector.py roi_size

# Crop offset in DOWNSAMPLED space, independent of field-of-view size:
#   ROI centre  = roi//4 + (640/2)/4 = roi//4 + 80
#   crop origin = centre - 128/2     = roi//4 + 16
OFFSET = ROI_SIZE // 2 // 4 - PATCH // 2      # = 16

CHANNEL_ORDER = ["nucleus", "microtubule", "er", "protein"]

STACKS = ["composite", "single_jpg", "single_tiff"]
VARIANTS = STACKS

Image.MAX_IMAGE_PIXELS = None      # these are 2048x2048; silence the decompression-bomb guard


# --------------------------------------------------------------------------- loaders

TIF_MAX = {np.uint8: 255.0, np.uint16: 65535.0}


def load_tif(path):
    """Single-channel TIFF -> float [0,1]. Accepts uint8 and uint16.

    Read pages[0] rather than imread(): these files declare an OME multi-file series that is
    not present, so imread() fails. See "FILE-FORMAT TRAPS" in the module docstring.

    BIT DEPTH IS NOT UNIFORM across HPA. Measured over all available ROI tiles:
        2048x2048 -> uint16   (97 tiles, ~62,800 distinct levels)
        1728x1728 -> uint8    (20 tiles, 256 levels)
    Size predicts dtype exactly — two acquisition generations. This matters for interpretation:
    on the 8-bit tiles the TIFF variant carries NO bit-depth advantage over the JPGs, only the
    absence of compositing artifacts. That makes them a natural control separating the two
    effects, so step 8 should stratify by `tif_dtype` (recorded in index.csv).
    """
    with tifffile.TiffFile(path) as tf:
        arr = tf.pages[0].asarray()
    assert arr.ndim == 2 and arr.shape[0] == arr.shape[1], f"{path}: expected square 2-D, got {arr.shape}"
    scale = TIF_MAX.get(arr.dtype.type)
    assert scale, f"{path}: unexpected TIFF dtype {arr.dtype} (expected uint8 or uint16)"
    return arr.astype(np.float32) / scale


def tif_dtype(stem):
    """dtype of this field of view's TIFFs. All four channels must agree."""
    dts = {}
    for c in ("blue", "red", "green", "yellow"):
        with tifffile.TiffFile(f"{stem}_{c}.tif") as tf:
            dts[c] = str(tf.pages[0].dtype)
    assert len(set(dts.values())) == 1, f"TIFF dtypes differ across channels: {dts}"
    return dts["blue"]


def load_lut_channel(path, colour):
    """HPA 'single-channel' JPG -> float [0,1], taking the LUT channel (not luminance)."""
    arr = np.array(Image.open(path))
    assert arr.ndim == 3 and arr.shape[2] == 3 and arr.shape[0] == arr.shape[1], \
        f"{path}: expected square RGB, got {arr.shape}"
    arr = arr.astype(np.float32) / 255.0
    idx = {"red": 0, "green": 1, "blue": 2}
    if colour == "yellow":
        return arr[:, :, [0, 1]].mean(axis=2)      # yellow LUT: R ~= G
    return arr[:, :, idx[colour]]


def load_composite_rgb(path):
    """RGB composite JPG -> (N, N, 3) float [0,1]."""
    arr = np.array(Image.open(path))
    assert arr.ndim == 3 and arr.shape[2] == 3 and arr.shape[0] == arr.shape[1], \
        f"{path}: expected square RGB, got {arr.shape}"
    return arr.astype(np.float32) / 255.0


def load_pups_grayscale(path):
    """PIL luminance conversion -- exactly what PUPS applies to yellow.jpg.

    Kept so the run can report how much it differs from load_lut_channel(). For the yellow LUT
    it is a near-uniform 0.886x scaling, which the per-patch rescale_intensity removes, so the
    two are equivalent in the final tensors. Reported, not assumed -- see --report-yellow-check.
    """
    return np.array(ImageOps.grayscale(Image.open(path))).astype(np.float32) / 255.0


# --------------------------------------------------------------------------- preprocessing

def downsample(img):
    """//4 with skimage defaults, exactly as PUPS's get_image_and_resize does.

    Output size is derived from the input, so 2048 -> 512 and 1728 -> 432 both work. A trailing
    channel axis is preserved by skimage when output_shape has fewer dimensions.
    """
    out = tuple(s // 4 for s in img.shape[:2])
    return resize(img, out)          # anti_aliasing defaults on for downsampling, as in PUPS


def crop_and_rescale(img_down, roi_x, roi_y):
    """Cut the 128x128 patch at the ROI centre and rescale it to [0,1] (PUPS process_patch)."""
    down = img_down.shape[0]
    r0 = roi_y // 4 + OFFSET
    c0 = roi_x // 4 + OFFSET
    # Holds by construction: 2_Square_Selector clamps the 640 ROI inside the image, so
    # roi <= N-640 and roi//4 + 16 + 128 <= N//4. Asserted rather than assumed.
    assert 0 <= r0 and r0 + PATCH <= down, f"row {r0}..{r0 + PATCH} outside {down}"
    assert 0 <= c0 and c0 + PATCH <= down, f"col {c0}..{c0 + PATCH} outside {down}"
    patch = img_down[r0:r0 + PATCH, c0:c0 + PATCH]
    return rescale_intensity(patch.astype(np.float32), out_range=(0, 1))


def build_variant(variant, stem, roi_x, roi_y):
    """-> (4, 128, 128) float32 in CHANNEL_ORDER: nucleus, microtubule, ER, protein."""
    if variant == "composite":
        # PUPS full_lightfield: resize the whole RGB composite, then split.
        # R -> microtubules, G -> antibody/protein, B -> nuclei  (get_image_and_resize).
        rgb = downsample(load_composite_rgb(f"{stem}_blue_red_green.jpg"))
        planes = {"microtubule": rgb[:, :, 0], "protein": rgb[:, :, 1], "nucleus": rgb[:, :, 2],
                  "er": downsample(load_pups_grayscale(f"{stem}_yellow.jpg"))}
    elif variant == "single_jpg":
        planes = {"nucleus": downsample(load_lut_channel(f"{stem}_blue.jpg", "blue")),
                  "microtubule": downsample(load_lut_channel(f"{stem}_red.jpg", "red")),
                  "er": downsample(load_lut_channel(f"{stem}_yellow.jpg", "yellow")),
                  "protein": downsample(load_lut_channel(f"{stem}_green.jpg", "green"))}
    elif variant == "single_tiff":
        planes = {"nucleus": downsample(load_tif(f"{stem}_blue.tif")),
                  "microtubule": downsample(load_tif(f"{stem}_red.tif")),
                  "er": downsample(load_tif(f"{stem}_yellow.tif")),
                  "protein": downsample(load_tif(f"{stem}_green.tif"))}
    else:
        raise ValueError(variant)

    # All four channels of one variant must come from the same field of view geometry.
    shapes = {c: planes[c].shape[:2] for c in CHANNEL_ORDER}
    assert len(set(shapes.values())) == 1, f"{variant}: channel size mismatch {shapes}"

    return np.stack([crop_and_rescale(planes[c], roi_x, roi_y) for c in CHANNEL_ORDER])


def source_size(stem):
    """Full-resolution size of this field of view, for the run report."""
    with Image.open(f"{stem}_blue_red_green.jpg") as im:
        return im.size[0]

def check_geometry(stem):
    """Every format of one field of view must have the SAME full-resolution size.

    The crop origin is computed in each variant's own downsampled frame, so a composite and a
    TIFF of different sizes would crop different physical regions at different scales while
    passing every other assertion in this file. Returns the common size.
    """
    shapes = {}
    for name, path in (("composite", f"{stem}_blue_red_green.jpg"),
                       ("single_jpg", f"{stem}_blue.jpg")):
        with Image.open(path) as im:
            shapes[name] = (im.size[1], im.size[0])
    with tifffile.TiffFile(f"{stem}_blue.tif") as tf:
        shapes["single_tiff"] = tuple(tf.pages[0].shape[:2])
    assert len(set(shapes.values())) == 1, f"format geometry differs: {shapes}"
    return shapes["composite"][0]

# --------------------------------------------------------------------------- ROI table

def read_rois(path):
    """Parse ROI.txt / ROI_examples*.txt. Skips rows flagged Skip == T."""
    rows, skipped = [], 0
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if (r.get("Skip") or "F").strip().upper() == "T":
                skipped += 1
                continue
            rows.append({
                "gene": r["Gene"].strip(),
                "antibody": r["Antibody"].strip(),
                "cell_line": r["CellLine"].strip(),
                "prefix": r["ImagePrefix"].strip(),
                # FolderPath uses Windows backslashes in the source file.
                "folder": r["FolderPath"].strip().replace("\\", "/"),
                "roi_x": int(r["Roi_X"]),
                "roi_y": int(r["Roi_Y"]),
                "annotation": (r.get("Annotation") or "").strip(),
                # ROI_examples.txt marks one representative tile per gene with Example=T.
                # ROI.txt has neither column, so both default to empty.
                "example": (r.get("Example") or "").strip().upper(),
            })
    return rows, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-root", required=True,
                    help="track Images folder, e.g. Independent_Selection/Images")
    ap.add_argument("--roi", required=True,
                    help="ROI table, e.g. Independent_Selection/ROI_examples.txt")
    ap.add_argument("--out", required=True, help="e.g. TRACK/work/variants")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N tiles")
    ap.add_argument("--no-previews", action="store_true", help="skip preview PNGs")
    args = ap.parse_args()

    root = pathlib.Path(args.images_root)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        sys.exit(f"--images-root not found: {root.resolve()}")

    rows, skipped = read_rois(args.roi)
    if args.limit:
        rows = rows[:args.limit]
    print(f"STEP 6 — building input variants")
    print(f"  ROI table : {args.roi}")
    print(f"  tiles     : {len(rows)} ({skipped} skipped via Skip=T)")
    print(f"  stacks    : {', '.join(STACKS)}   (the 0.19 threshold is applied in step 7)")
    print(f"  geometry  : NxN -> resize (N//4)^2 -> crop {PATCH}^2 at ROI centre (+{OFFSET})"
          f"   [N is per-image: 2048 or 1728]")
    print(f"  output    : {out}\n")

    import collections
    index, failures, missing, yellow_checked = [], [], [], False
    sizes = collections.Counter()

    for i, r in enumerate(rows, 1):
        tile_id = f"{r['gene']}_{r['antibody']}_{r['cell_line']}_{r['prefix']}"
        stem = str(root / r["folder"] / r["prefix"])

        # Missing source images are a data-completeness issue, not a code failure -- ROI.txt
        # can reference fields of view that were never downloaded. Reported separately so a
        # genuine bug is never buried among them.
        if not pathlib.Path(f"{stem}_blue_red_green.jpg").exists():
            missing.append(f"{tile_id}  ({r['folder']}/{r['prefix']})")
            print(f"  [{i}/{len(rows)}] {tile_id}  SKIPPED: source images not present")
            continue

        try:
            n_full, dt = check_geometry(stem), tif_dtype(stem)
            sizes[(n_full, dt)] += 1
            stacks = {v: build_variant(v, stem, r["roi_x"], r["roi_y"]) for v in VARIANTS}
        except (AssertionError, FileNotFoundError, OSError) as e:
            failures.append(f"{tile_id}: {type(e).__name__}: {e}")
            print(f"  [{i}/{len(rows)}] {tile_id}  FAILED: {e}")
            continue

        # One-off sanity report: is PUPS's luminance yellow equivalent to our LUT yellow
        # after per-patch rescaling? Reported on the first tile so the claim is evidenced.
        if not yellow_checked:
            a = crop_and_rescale(downsample(load_pups_grayscale(f"{stem}_yellow.jpg")),
                                 r["roi_x"], r["roi_y"])
            b = crop_and_rescale(downsample(load_lut_channel(f"{stem}_yellow.jpg", "yellow")),
                                 r["roi_x"], r["roi_y"])
            print(f"  yellow-extraction check on {tile_id}:")
            print(f"    PUPS luminance vs LUT-mean, after rescale_intensity:"
                  f" max|diff|={np.abs(a - b).max():.5f}  corr={np.corrcoef(a.ravel(), b.ravel())[0, 1]:.6f}")
            print(f"    (near-identical => the two extractions are interchangeable here)\n")
            yellow_checked = True

        np.savez_compressed(out / f"{tile_id}.npz",
                            channel_order=np.array(CHANNEL_ORDER),
                            **stacks)
        index.append({"tile_id": tile_id, **r, "full_size": n_full, "tif_dtype": dt})
        print(f"  [{i}/{len(rows)}] {tile_id}"
              f"  roi=({r['roi_x']},{r['roi_y']})"
              + (f"  [{r['annotation']}]" if r["annotation"] else ""))

        if not args.no_previews:
            save_preview(out / "previews" / f"{tile_id}.png", stacks, tile_id, r)

    with open(out / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index[0].keys()) if index else ["tile_id"])
        w.writeheader()
        w.writerows(index)

    print(f"\n  wrote {len(index)} tiles -> {out}")
    print(f"  index -> {out / 'index.csv'}")
    print("  source formats: " + ", ".join(
        f"{s}x{s} {d}: {n}" for (s, d), n in sorted(sizes.items(), reverse=True)))
    if len({d for _, d in sizes}) > 1:
        print("    NOTE: mixed TIFF bit depth -- stratify by tif_dtype in step 8 "
              "(8-bit TIFFs give no bit-depth advantage, only no compositing)")
    if not args.no_previews:
        print(f"  previews -> {out / 'previews'}  (EYEBALL THESE before running step 7)")

    if missing:
        print(f"\n  {len(missing)} tile(s) skipped — source images not downloaded:")
        for m in missing:
            print(f"    {m}")
        print("    ACTIONABLE: this is NOT expected on a track whose 4b gate passed. Run"
              "\n    4b_Verify_Track_Images.py --track <TRACK>: it names every missing file and"
              "\n    exits non-zero, whereas this step skips and still exits 0.")

    if failures:
        print(f"\n  {len(failures)} FAILURE(S) — these are bugs or corrupt files, investigate:")
        for f in failures:
            print(f"    {f}")
        return 1

    print("\nSTEP 6 COMPLETE")
    print("  NEXT: 5_Make_ESM2_Embeddings.py if not done, then 7_Run_PUPS_Inference.py")
    return 0


def save_preview(path, stacks, tile_id, meta):
    """Grid: rows = variants, cols = channels. For visual QC, not for analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(stacks), 4, figsize=(9, 2.4 * len(stacks)))
    axes = np.atleast_2d(axes)
    for ri, v in enumerate(stacks):
        for ci, ch in enumerate(CHANNEL_ORDER):
            ax = axes[ri, ci]
            ax.imshow(stacks[v][ci], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(ch + ("\n(TARGET)" if ch == "protein" else "\n(input)"), fontsize=8)
            if ci == 0:
                ax.set_ylabel(v, fontsize=8)
    fig.suptitle(f"{tile_id}   {meta['annotation']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
