#!/usr/bin/env python3
"""STEP 1 — Verify both PUPS checkpoints load and run offline on CPU.

A validation gate, not an analysis. Run it after step 0 and after any environment change.
If this fails, nothing downstream can be trusted.

    ./.venv/bin/python 1_Verify_PUPS_Model.py

Checks, in order:
  1. the vendored model instantiates with defaults alone (no dataset objects, no MongoDB)
  2. each checkpoint loads with ZERO missing and ZERO unexpected keys
  3. a synthetic forward pass returns the documented shapes and finite values
  4. changing the protein embedding changes the output -- i.e. the sequence branch actually
     reaches the decoder. Panel B measures exactly this quantity, so if it were zero the whole
     experiment would be meaningless.
  5. throughput, so the run-time estimates in PIPELINE.md stay honest

ACTIONABLE: exits non-zero on any failure. Wire it into a Makefile or CI if this grows.
"""
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "PUPS_Code"))

BATCH, SEQ, EMB, IMG = 2, 2000, 1280, 128

# Landmark stains only: nucleus, microtubule, ER. Inpainting_Model defaults to
# image_input_channels=3. The protein (green) channel is the PREDICTION TARGET and must
# never be fed to the model -- doing so would leak the answer into the input.
LANDMARKS = 3
N_CLASSES = 29

CHECKPOINTS = {
    "default (0.19 threshold, all paper results)":
        "splice_isoform_dataset_cell_line_and_gene_split_full-epoch=01-val_combined_loss=0.18.ckpt",
    "nothreshold (ablation, Extended Data Fig. 5b top panel)":
        "nothreshold.ckpt",
}


def synthetic_batch(seed):
    """Shapes are read from the PUPS source, not guessed:
      - LightAttentionNN.forward does x.reshape(x.shape[0], x.shape[2], x.shape[3]), so the
        embedding must be 4-D with a singleton at dim 1. NOTE their own pad_collate emits 3-D
        (B, 2000, 1280), so something must unsqueeze in between -- we do it explicitly.
      - x_lens builds the attention mask and is PUPS's x_len == L + 2 (BOS + EOS included).
    """
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(BATCH, 1, SEQ, EMB, generator=g),
            torch.tensor([812.0, 1503.0]),
            torch.rand(BATCH, LANDMARKS, IMG, IMG, generator=g))


def check_one(label, filename, failures):
    from src.model.full_model import SubCellProtModel

    print(f"\n{'-' * 72}\n{label}\n  {filename}")
    path = ROOT / "PUPS_Code" / "checkpoints" / filename
    if not path.exists():
        failures.append(f"{label}: checkpoint missing -- see PIPELINE.md step 0 (git clone)")
        print("  MISSING -- run step 0")
        return

    model = SubCellProtModel()
    n_param = sum(p.numel() for p in model.parameters())

    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)

    print(f"  parameters: {n_param:,}   state_dict entries: {len(sd)}")
    print(f"  lightning {ck.get('pytorch-lightning_version', '?')}  "
          f"epoch {ck.get('epoch', '?')}  global_step {ck.get('global_step', '?')}")
    if missing or unexpected:
        failures.append(f"{label}: {len(missing)} missing, {len(unexpected)} unexpected keys")
        print(f"  NOT A CLEAN LOAD: missing={list(missing)[:3]} unexpected={list(unexpected)[:3]}")
    else:
        print("  clean load: 0 missing, 0 unexpected")

    model.eval()
    esm2, lens, stains = synthetic_batch(0)
    with torch.no_grad():
        img, lab = model.call_model(esm2, lens, stains)

    ok_img = tuple(img.shape) == (BATCH, 1, IMG, IMG)
    ok_lab = tuple(lab.shape) == (BATCH, N_CLASSES)
    ok_fin = bool(torch.isfinite(img).all() and torch.isfinite(lab).all())
    print(f"  forward: image {tuple(img.shape)} {'ok' if ok_img else 'UNEXPECTED'}"
          f"   labels {tuple(lab.shape)} {'ok' if ok_lab else 'UNEXPECTED'}")
    print(f"  image value range [{img.min():.4f}, {img.max():.4f}]")
    if img.min() < 0 or img.max() > 1:
        # Not an error: there is no output activation clamping to [0,1], while targets ARE
        # rescaled to [0,1]. Step 8 must apply one clipping rule to every variant alike.
        print("    note: outside [0,1] -- no output clamp in the model; step 8 decides clipping")
    for cond, msg in ((ok_img, "image shape"), (ok_lab, "label shape"), (ok_fin, "finite values")):
        if not cond:
            failures.append(f"{label}: {msg}")

    # Does the sequence actually influence the prediction?
    esm2b, _, _ = synthetic_batch(1)
    with torch.no_grad():
        img_b, _ = model.call_model(esm2b, lens, stains)
    delta = (img - img_b).abs().mean().item()
    print(f"  sequence sensitivity (mean |delta|, same image, different embedding): {delta:.6f}")
    if delta <= 0:
        failures.append(f"{label}: output does not depend on the protein embedding")

    # Throughput
    with torch.no_grad():
        model.call_model(esm2, lens, stains)          # warm-up
        t0 = time.perf_counter()
        for _ in range(3):
            model.call_model(esm2, lens, stains)
    per = (time.perf_counter() - t0) / 3 / BATCH
    # Step 7's whole workload for the largest track: 3 stacks x 2 train/test-consistent
    # checkpoint/threshold pairs = 6 forward passes per tile (see CONSISTENT in step 7).
    print(f"  throughput: {per * 1000:.0f} ms/sample on {torch.get_num_threads()} threads"
          f"  ->  120 tiles x 6 conditions = {120 * 6 * per:.0f} s")


def main():
    print("=" * 72)
    print("STEP 1 — verifying PUPS runs offline on CPU")
    print("=" * 72)
    if not (ROOT / "PUPS_Code" / "src" / "model" / "full_model.py").exists():
        sys.exit("PUPS_Code/src/model missing -- see PIPELINE.md step 0 (git clone) first")

    failures = []
    for label, fn in CHECKPOINTS.items():
        check_one(label, fn, failures)

    print("\n" + "=" * 72)
    if failures:
        print("STEP 1 FAILED")
        for f in failures:
            print(f"  - {f}")
        print("=" * 72)
        return 1
    print("STEP 1 PASSED — both checkpoints load and run, no MongoDB, no GPU")
    print("  NEXT: ./.venv/bin/python 2_Resolve_Source_List.py --track <TRACK>")

    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
