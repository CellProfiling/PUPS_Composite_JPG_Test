#!/usr/bin/env python3
"""STEP 0 — materialise the two published checkpoints we actually use.

    ./.venv/bin/python 0_Get_Checkpoints.py            # the two we need
    ./.venv/bin/python 0_Get_Checkpoints.py --all      # all 23, ~1.6 GB

The clone's 23 checkpoints are 1.6 GB and we use two of them, so they are not kept on disk.
This restores them on demand. It tries the clone's own object store first (offline, instant)
and falls back to downloading from GitHub.

WHICH TWO, AND WHY
------------------
  default      splice_isoform_dataset_cell_line_and_gene_split_full-epoch=01-val_combined_loss=0.18.ckpt
               The repo README states: "All results in the paper are based on the model
               parameters in [this file]". epoch 1, global_step 3548.
  nothreshold  nothreshold.ckpt
               The ablation trained WITHOUT the 0.19 landmark intensity threshold — the model
               in the top panel of Extended Data Fig. 5b. epoch 0, global_step 2661.

⚠ They are NOT trained equally long (3548 vs 2661 steps, ~25% apart), so a
default-vs-nothreshold difference conflates the threshold with training duration. Compare
across INPUT FORMATS within one checkpoint; that contrast is clean.
"""
import argparse
import pathlib
import subprocess
import sys
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
CKPT_DIR = ROOT / "PUPS_Code" / "checkpoints"
RAW = ("https://raw.githubusercontent.com/uhlerlab/PUPS/"
       "e0354b9e6374c1f6676fcbc9d9c15ca6c7fccd6d/checkpoints")

NEEDED = [
    "splice_isoform_dataset_cell_line_and_gene_split_full-epoch=01-val_combined_loss=0.18.ckpt",
    "nothreshold.ckpt",
]


def from_git(names):
    """Restore from the clone's object store — offline and instant."""
    try:
        subprocess.run(["git", "-C", str(ROOT / "PUPS_Code"), "checkout", "--",
                        *[f"checkpoints/{n}" for n in names]],
                       check=True, capture_output=True, text=True)
        return True
    except Exception as e:
        print(f"  git restore unavailable ({e}); falling back to download")
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="all 23 checkpoints (~1.6 GB)")
    args = ap.parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if args.all:
        out = subprocess.run(["git", "-C", str(ROOT / "PUPS_Code"), "ls-files", "checkpoints/"],
                             capture_output=True, text=True)
        names = [pathlib.Path(l).name for l in out.stdout.split() if l.endswith(".ckpt")]
    else:
        names = NEEDED

    missing = [n for n in names if not (CKPT_DIR / n).exists()]
    print(f"STEP 0 — {len(names)} checkpoint(s) requested, {len(missing)} missing\n")
    if not missing:
        for n in names:
            print(f"  present  {n}  ({(CKPT_DIR/n).stat().st_size/1e6:.0f} MB)")
        print("\nnothing to do.")
        return 0

    if not from_git(missing):
        for n in missing:
            dest = CKPT_DIR / n
            print(f"  downloading {n} ...", end="", flush=True)
            try:
                urllib.request.urlretrieve(f"{RAW}/{urllib.parse.quote(n)}", dest)
                print(f" {dest.stat().st_size/1e6:.0f} MB")
            except Exception as e:
                print(f" FAILED: {e}")
                return 1

    ok = True
    for n in names:
        p = CKPT_DIR / n
        if p.exists():
            print(f"  ready    {n}  ({p.stat().st_size/1e6:.0f} MB)")
        else:
            print(f"  MISSING  {n}")
            ok = False
    print("\nNEXT: ./.venv/bin/python 1_Verify_PUPS_Model.py" if ok else "\nincomplete")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
