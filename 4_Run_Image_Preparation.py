#!/usr/bin/env python3
"""STEP 4 — run one of the Image_Preparation tools inside a track folder.

    ./.venv/bin/python 4_Run_Image_Preparation.py --track Fig_2C --tool download
    ./.venv/bin/python 4_Run_Image_Preparation.py --track Fig_2C --tool roi
    ./.venv/bin/python 4_Run_Image_Preparation.py --track Fig_2C --tool examples
    ./.venv/bin/python 4_Run_Image_Preparation.py --track Fig_Ext5B --tool matrix

WHERE THESE TOOLS COME FROM
---------------------------
`Image_Preparation/` is a **git clone of the companion project**, not our code:

    git clone https://github.com/cellprofiling/HPA_BleedThrough_Exploration.git Image_Preparation

pinned at commit `6225801e46409f9b8ab9e9274656c9dd25a5c4ad`. We do not modify it — check with
`git -C Image_Preparation status`. Verified 2026-09-03 to be identical to the working copies the
companion project's own analysis was run from, apart from line endings (those are CRLF, upstream
is LF; all seven scripts match byte-for-byte after normalising).

ONE TOOL RUNS FROM A MODIFIED COPY
----------------------------------
`download` runs from `Image_Preparation_Modified/1_HPA_Image_Download_BatchDownload.py`, a
marked fork that falls back to archived HPA releases when the current one has dropped an
antibody's immunofluorescence images. Everything else runs from the pristine clone. The fork is
mechanically audited on every run: `verify_fork()` applies
`Image_Preparation_Modified/0001-archived-release-fallback.patch` to the clone's file and
requires the result to equal our copy byte-for-byte, so the script and its stated diff can
never drift apart. See `Image_Preparation_Modified/README.md`.

WHY A RUNNER INSTEAD OF CALLING THEM DIRECTLY
---------------------------------------------
Every one of those tools resolves its input and output paths from `Path(__file__).parent`:

    script_dir  = Path(__file__).parent
    output_dir  = script_dir / protein_name / antibody_id / cell_line
    roi_file    = script_dir / "ROI.txt"

They are therefore not relocatable: run from Image_Preparation/ they would write there. Rather
than edit settled code, this runner places the chosen tool INSIDE the track folder, verifies it
is byte-identical to the Image_Preparation original, runs it there, and leaves it in place so
the track folder shows exactly which code produced its contents.

TOOLS
-----
  download  1_HPA_Image_Download_BatchDownload.py  reads Organelle_Example_GenesAndCellLines.txt
                                                   (written by step 2), fetches into <track>/
  roi       2_Square_Selector.py                   tkinter GUI; writes <track>/ROI.txt
  examples  3_Best_Example_Selector.py             tkinter GUI; ROI.txt -> ROI_examples.txt
  matrix    7_Matrix_Figure_Generator_4.py         the published bleed-through comparison figure
  matrix1/2/3                                      the other three generator variants

ACTIONABLE: `download` writes gene folders directly under <track>/, because that is what their
script does. Step 6 expects them under <track>/Images/, so this runner moves them there
afterwards and says so. Nothing else reorganises files behind your back.

WHY THE OTHER TOOLS RUN INSIDE <track>/Images/
----------------------------------------------
`2_Square_Selector.py` derives the Gene / Antibody / CellLine columns from folder DEPTH:

    rel_path = folder_path.relative_to(script_dir)      # expects Gene/Antibody/CellLine
    gene, antibody, cell_line = rel_path.parts[:3]

Run from <track>/ — after `download` has moved the genes into Images/ — that relative path is
`Images/GENE/ANTIBODY/CELLLINE`, so every row comes out shifted by one: `Gene` is literally
"Images", `Antibody` holds the gene, and `FolderPath` gains an `Images/` prefix that step 6
then resolves to <track>/Images/Images/... . `3_Best_Example_Selector.py` inherits the shift and
looks for `Images/anno.txt`, so every annotation becomes "N/A" and all 14 fields of view are
grouped under one pseudo-gene, giving one Example=T for the whole track instead of one per gene.

So the roi / examples / matrix tools are placed and run in <track>/Images/, where the depth
assumption holds. The ROI tables are shuttled: copied down into Images/ before the run, moved
back up to <track>/ afterwards, because <track>/ROI*.txt is where they belong and what step 6
reads. Those tools are still byte-identical to the clone — only their working directory differs.
"""
import argparse
import filecmp
import pathlib
import shutil
import subprocess
import sys
import tempfile

TOOLS = {
    "download": "1_HPA_Image_Download_BatchDownload.py",
    "roi":      "2_Square_Selector.py",
    "examples": "3_Best_Example_Selector.py",
    "matrix1":  "4_Matrix_Figure_Generator.py",
    "matrix2":  "5_Matrix_Figure_Generator_2.py",
    "matrix3":  "6_Matrix_Figure_Generator_3.py",
    "matrix":   "7_Matrix_Figure_Generator_4.py",
}
ROOT = pathlib.Path(__file__).resolve().parent
PREP = ROOT / "Image_Preparation"

# Tools we run from a MODIFIED local copy instead of the pristine clone, mapped to the patch
# that documents the fork. `Image_Preparation_Modified/` holds both. For each of these the
# runner asserts that applying the patch to the clone's file reproduces our copy byte-for-byte,
# so the fork can never drift from its stated diff. Everything else comes from the clone and is
# asserted byte-identical to it. See Image_Preparation_Modified/README.md.
MODIFIED = ROOT / "Image_Preparation_Modified"
FORKED = {
    "1_HPA_Image_Download_BatchDownload.py": "0001-archived-release-fallback.patch",
}

# ROI tables live at <track>/ but the tools that read or write them must run one level down,
# in <track>/Images/ (see the module docstring). These are copied down and moved back up.
SHUTTLED = ["ROI.txt", "ROI_examples.txt", "ROI_examples_Selected.txt"]


def merge_into(src, dst):
    """Move `src` to `dst`, MERGING if `dst` already exists. Returns the number of files kept.

    `shutil.move(src, dst)` on an existing directory moves src INSIDE it, producing
    `Images/N4BP2/N4BP2/HPA042607/...` — silently, and the glob that counts fields of view
    cannot see the extra level, so the run reports fewer images than it downloaded and step 6
    skips those tiles. That is exactly what happens for a gene whose folder is already partly
    present, which is the normal case when a track carries some images in the repository and
    re-downloads the rest.

    So the tree is merged one leaf at a time. A file already at the destination WINS: the
    committed copies are the irreplaceable ones (see PIPELINE.md section 6, step 4b), and a
    re-download must never overwrite them.
    """
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return 0
    kept = 0
    for s in sorted(src.rglob("*")):
        if s.is_dir():
            continue
        d = dst / s.relative_to(src)
        d.parent.mkdir(parents=True, exist_ok=True)
        if d.exists():
            kept += 1                      # keep what was already there
            s.unlink()
        else:
            shutil.move(str(s), str(d))
    shutil.rmtree(src, ignore_errors=True)
    return kept


def verify_fork(name):
    """Assert that our modified copy of `name` is exactly the clone's file plus its patch.

    This is the same kind of check as `git -C PUPS_Code diff -- src/`: the fork is allowed, but
    it must be the diff it claims to be. Applying the patch to a pristine copy of the clone's
    file has to reproduce ours byte-for-byte, or the run stops. That way editing either the
    script or the patch without the other is caught immediately, instead of the repository
    quietly carrying an undocumented change to third-party code.

    Returns a one-line provenance string for the run log.
    """
    ours, patch = MODIFIED / name, MODIFIED / FORKED[name]
    for p in (ours, patch, PREP / name):
        if not p.exists():
            sys.exit(f"missing: {p}")
    with tempfile.TemporaryDirectory() as tmp:
        staged = pathlib.Path(tmp) / name
        shutil.copy2(PREP / name, staged)
        # `patch(1)` is absent from many minimal server images, so fall back to `git apply`,
        # which is guaranteed present -- both upstream repositories are obtained by git clone.
        attempts = [(["patch", "-s", "-p1", name], {"stdin": open(patch)}),
                    (["git", "apply", "-p1", "--unsafe-paths", str(patch)], {})]
        r = None
        for cmd, kw in attempts:
            try:
                r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, **kw)
            except FileNotFoundError:
                continue
            if r.returncode == 0:
                break
        if r is None:
            sys.exit("neither `patch` nor `git` is available; one is needed to verify "
                     f"Image_Preparation_Modified/{name} against its patch.")
        if r.returncode != 0:
            sys.exit(f"{FORKED[name]} does not apply to Image_Preparation/{name}:\n"
                     f"{r.stdout}{r.stderr}"
                     f"  The clone may have been re-pinned. Regenerate the patch with:\n"
                     f"    diff -u Image_Preparation/{name} Image_Preparation_Modified/{name}")
        if not filecmp.cmp(staged, ours, shallow=False):
            sys.exit(f"Image_Preparation_Modified/{name} is NOT the clone's file plus "
                     f"{FORKED[name]}.\n"
                     f"  One of the two was edited without the other. Regenerate the patch:\n"
                     f"    diff -u Image_Preparation/{name} Image_Preparation_Modified/{name} "
                     f"> Image_Preparation_Modified/{FORKED[name]}")
    n_hunks = sum(1 for line in open(patch) if line.startswith("@@"))
    return (f"MODIFIED copy from Image_Preparation_Modified/ — verified == clone + "
            f"{FORKED[name]} ({n_hunks} hunks)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", required=True)
    ap.add_argument("--tool", required=True, choices=sorted(TOOLS))
    ap.add_argument("--stdin", default=None,
                    help="text piped to the tool, e.g. --stdin 1 for the matrix generators' "
                         "'which ROI file?' prompt")
    args = ap.parse_args()

    track = ROOT / args.track
    if not track.is_dir():
        sys.exit(f"track folder not found: {track}")
    name = TOOLS[args.tool]
    if name in FORKED:
        src = MODIFIED / name
        origin = verify_fork(name)
    else:
        src = PREP / name
        origin = "byte-identical to Image_Preparation/"
    if not src.exists():
        sys.exit(f"missing tool: {src}")

    # `download` writes gene folders next to itself and is moved afterwards; everything else
    # reads the Gene/Antibody/CellLine hierarchy and must sit directly above it.
    if args.tool == "download":
        workdir = track
    else:
        workdir = track / "Images"
        if not workdir.is_dir():
            sys.exit(f"no images yet: {workdir}\n  run --tool download first.")
    rel = workdir.relative_to(ROOT)
    dst = workdir / name

    if dst.exists() and filecmp.cmp(src, dst, shallow=False):
        print(f"  {name} already present in {rel}/ and byte-identical")
    else:
        shutil.copy2(src, dst)
        print(f"  copied {name} -> {rel}/  ({origin})")
    assert filecmp.cmp(src, dst, shallow=False), "copy is not byte-identical"

    # Shuttle the ROI tables down so the tool finds them where it looks (script_dir).
    shuttled = []
    if workdir != track:
        for fn in SHUTTLED:
            up = track / fn
            if up.exists():
                shutil.copy2(up, workdir / fn)
                shuttled.append(fn)
        if shuttled:
            print(f"  copied {', '.join(shuttled)} down into {rel}/ for the tool to read")

    print(f"\nrunning {rel}/{name}\n" + "-" * 70)
    proc = subprocess.run([sys.executable, str(dst)], cwd=str(workdir),
                          input=(args.stdin + "\n") if args.stdin else None, text=True)
    print("-" * 70)

    # Move results back up, whatever the exit status, so a partial selection is never lost.
    if workdir != track:
        moved_up = []
        for fn in SHUTTLED:
            down = workdir / fn
            if down.exists():
                shutil.move(str(down), str(track / fn))
                moved_up.append(fn)
        for p in sorted(workdir.glob("HPA_comparison_matrix*")):
            figures = track / "figures"
            figures.mkdir(exist_ok=True)
            shutil.move(str(p), str(figures / p.name))
            moved_up.append(f"figures/{p.name}")
        if moved_up:
            print(f"  moved back up to {args.track}/: {', '.join(moved_up)}")

    if proc.returncode != 0:
        print(f"tool exited {proc.returncode}")
        return proc.returncode

    if args.tool == "download":
        # Their script writes GENE/ANTIBODY/CELLLINE straight into script_dir. Step 6 wants
        # them under Images/. Move, and report exactly what moved.
        images = track / "Images"
        images.mkdir(exist_ok=True)
        moved = []
        for p in sorted(track.iterdir()):
            if p.is_dir() and p.name not in {"Images", "work", "figures", "esm2_output",
                                             "Crops", "__pycache__"} \
               and any(p.glob("*/*/*_blue_red_green.jpg")):
                moved.append(f"{p.name}{'' if merge_into(p, images / p.name) == 0 else ' (merged)'}")
        print(f"\nmoved {len(moved)} gene folder(s) into {args.track}/Images/: "
              f"{', '.join(moved) if moved else '(none)'}")
        n = len(list(images.glob('*/*/*/*_blue.tif')))
        print(f"fields of view now present: {n}")
        print("\nACTIONABLE: check the per-gene count against your source_list.csv. The fork now "
              "\n  tallies skipped entries and exits non-zero (the unmodified script printed an "
              "\n  error and still finished with status 0), but its bare `except` clauses are "
              "\n  unchanged, so a partial gene can still slip through. Re-run this step to "
              "\n  resume, then run 4b_Verify_Track_Images.py — that is the real gate, because it "
              "\n  checks every field of view named in ROI.txt rather than just counting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
