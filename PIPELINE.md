# Reproducing PUPS on different HPA image formats

An end-to-end, reproducible pipeline that runs the published **PUPS** model
([Zhang et al., *Nature Methods* 22:1265–1275, 2025](https://doi.org/10.1038/s41592-025-02696-1);
code at [uhlerlab/PUPS](https://github.com/uhlerlab/PUPS)) on the *same* fields of view supplied
in different image formats, and measures how much the format changes the prediction.

**The question.** PUPS obtains its HPA images only as 8-bit JPGs, recovering three of its four
channels by splitting a single lossy RGB composite. JPG-compositing introduces artificial inter-channel
correlations that look like spectral bleed-through but are a compression artifact — HPA acquires
in three sequential acquisitions, so there is no optical bleed-through to begin with — established by the
companion project,
[HPA_BleedThrough_Exploration](https://github.com/cellprofiling/HPA_BleedThrough_Exploration).
This pipeline quantifies the consequence.

The PUPS model and the HPA image tooling are both **pinned git clones**. The model carries a one-line
diff; the image tooling carries none, and the single tool we needed to change is forked into
`Image_Preparation_Modified/` where every changed region is tagged and verified against a patch
on each run (§2). Every external URL we fetch is logged with a checksum, and the
proteoform-resolution chain is validated against the authors' own published database
screenshots.

---

## Quickstart — one complete track, start to finish

Everything below is copy-pasteable. It clones the two upstream repositories, builds the
environment, and reproduces one full track. Budget **~20 GB disk** and roughly **40–60
minutes** on a first run — the track itself takes about 23 minutes ([measured](#6-run-order)),
and the rest is setup, most of it the one-time 2.6 GB ESM-2 download and the image downloads.

```bash
# ── setup (once) ──────────────────────────────────────────────────────────────
git clone <this-repo> && cd PUPS_Examples

git clone https://github.com/uhlerlab/PUPS.git PUPS_Code
git -C PUPS_Code checkout e0354b9e6374c1f6676fcbc9d9c15ca6c7fccd6d
git -C PUPS_Code apply "$(pwd)/patches/0001-comment-out-unused-tensorflow-import.patch"

git clone https://github.com/cellprofiling/HPA_BleedThrough_Exploration.git Image_Preparation
git -C Image_Preparation checkout 6225801e46409f9b8ab9e9274656c9dd25a5c4ad

python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python 0_Get_Checkpoints.py

# ── gates: both must pass before anything else ────────────────────────────────
./.venv/bin/python 1_Verify_PUPS_Model.py
./.venv/bin/python 2_Resolve_Source_List.py --self-test

# ── one track, end to end.  Fig_2C is the cheapest and cleanest to start with ──
T=Fig_2C            # or Fig_Ext5B, or Independent_Selection

./.venv/bin/python 2_Resolve_Source_List.py    --track $T
./.venv/bin/python 3_Write_Provenance.py       --track $T
./.venv/bin/python 4_Run_Image_Preparation.py  --track $T --tool download
./.venv/bin/python 4b_Verify_Track_Images.py   --track $T      # must end "nothing missing"

#   NOTE: do NOT run --tool roi / --tool examples. Every track ships committed ROI
#   tables; the selectors would overwrite them, and Fig_Ext5B's were matched by eye
#   to the published panels and cannot be recovered automatically.

./.venv/bin/python 5_Make_ESM2_Embeddings.py --track $T --faithful
./.venv/bin/python 6_Build_Input_Variants.py --images-root $T/Images \
    --roi $T/ROI_examples.txt --out $T/work/variants
./.venv/bin/python 7_Run_PUPS_Inference.py   --variants $T/work/variants \
    --embed $T/esm2_output --out $T/work/predictions
./.venv/bin/python 8_Compute_Metrics.py      --pred $T/work/predictions \
    --variants $T/work/variants --out $T/work/metrics

#   --n caps how many genes the figures cover and DEFAULTS TO 6. Fig_2C has exactly 6,
#   so the default is right above — but passing too small an --n silently drops genes:
#       Fig_Ext5B             --n 9
#       Independent_Selection --n 34 for step 9, and --all-genes for step 10
#   Both scripts warn when they exclude a gene, so watch for a line starting with a warning sign.
./.venv/bin/python 9_Figure_Formats.py       --pred $T/work/predictions \
    --variants $T/work/variants --out $T/figures --n 6
    # Fig_2C: 6 genes. See the note above.

#   Same panel over EVERY tile, paginated, for review rather than publication: one tile per
#   row instead of one per gene, so nothing is hidden behind the --n cap. run_track.sh always
#   does this; skip it only if you do not intend to inspect the run.
./.venv/bin/python 9_Figure_Formats.py       --pred $T/work/predictions \
    --variants $T/work/variants --out $T/figures --all-tiles

#   Panel B must run TWICE — see section 9, "The word threshold means two different things"
./.venv/bin/python 10_Figure_Leakage.py --pred $T/work/predictions \
    --variants $T/work/variants --embed $T/esm2_output --out $T/figures --threshold 0
./.venv/bin/python 10_Figure_Leakage.py --pred $T/work/predictions \
    --variants $T/work/variants --embed $T/esm2_output --out $T/figures --threshold 1

./.venv/bin/python 11_Ablate_Channels.py --variants $T/work/variants \
    --embed $T/esm2_output --out $T/work/ablation \
    --blocks --images-root $T/Images --roi $T/ROI_examples.txt
#   Step 12, not a repeat of step 4. 4_Run_Image_Preparation.py is the single audited entry
#   point for the companion project's tools: --tool download ran their downloader, --tool
#   matrix runs their 7_Matrix_Figure_Generator_4.py to build the HPA comparison matrix.
#   `--stdin 1` answers that tool's interactive prompt.
./.venv/bin/python 4_Run_Image_Preparation.py --track $T --tool matrix --stdin 1
```

Results land in `$T/figures/` and `$T/work/`; **[what a run produces](#what-a-run-produces)**
lists every file and which step wrote it.

**[EXAMPLE_RUN.md](EXAMPLE_RUN.md)** is a transcript of these commands run on a machine that
had none of this installed — the console output of every step, with a note on which lines to
check. Compare your own output against it to tell "working" from "ran without erroring".

---

## Contents

1. [What gets compared](#1-what-gets-compared)
2. [Change overview — what we changed in their code, and what we did not](#2-change-overview)
3. [Setup from scratch](#3-setup-from-scratch)
4. [Repository layout](#4-repository-layout)
5. [How proteoforms are resolved — the subtle part](#5-how-proteoforms-are-resolved)
6. [Run order](#6-run-order)
7. [Verification](#7-verification)
8. [Licensing and what this repository contains](#8-licensing-and-what-this-repository-contains)
9. [Caveats — read before quoting any number](#9-caveats--read-before-quoting-any-number)

*Also:* [Quickstart](#quickstart--one-complete-track-start-to-finish) ·
[What a run produces](#what-a-run-produces) (in §6) ·
[EXAMPLE_RUN.md](EXAMPLE_RUN.md) — annotated console output of a complete run

---

## 1. What gets compared

Each field of view is built into a 4-channel stack `[nucleus, microtubule, ER, protein]` three
ways. Only the source file format differs; the preprocessing is identical.

| Stack | Nucleus / Microtubule / Protein | ER |
|---|---|---|
| **`composite`** | RGB split of `<fov>_blue_red_green.jpg` — what PUPS reads | `<fov>_yellow.jpg` |
| **`single_jpg`** | `<fov>_{blue,red,green}.jpg` (LUT channel) | `<fov>_yellow.jpg` |
| **`single_tiff`** | `<fov>_{blue,red,green}.tif` (16-bit) | `<fov>_yellow.tif` |

The 0.19 threshold is PUPS's own mitigation (Methods: *"All landmark stain channels are
additionally filtered for low-intensity pixels, with a threshold of 0.19"*). It is a
preprocessing toggle, not a fourth stack: step 6 writes three stacks (composite, single-channel
JPG, TIFF) and step 7 applies the threshold at load time. That gives 3 stacks × 2 train/test-
consistent checkpoint/threshold pairs = 6 conditions per tile (§9).

**Only 3 channels are model input** — nucleus, microtubule, ER. The protein channel is the
prediction *target* and must never be fed in.

### Three datasets ("tracks")

| Track | What it is | Why |
|---|---|---|
| `Fig_Ext5B` | the 9 proteoform/cell-line pairs from their Extended Data Fig. 5b | their own bleed-through figure, run on the same cells |
| `Fig_2C` | the 6 pairs from their Fig. 2c | *"six randomly selected cells… all proteins and cell lines held out from model training"* |
| `Independent_Selection` | 120 hand-picked ROIs over 34 genes | chosen independently of the paper, larger *n* (provenance below) |

Track sizes as committed: `Fig_Ext5B` 28 fields of view / 9 genes / 9 selected examples,
`Fig_2C` 14 / 6 / 6, `Independent_Selection` 120 / 34 / 34. Every ROI row has a matching
`source_list.csv` entry.

`Independent_Selection`'s crop coordinates come from the companion
[HPA_BleedThrough_Exploration](https://github.com/cellprofiling/HPA_BleedThrough_Exploration)
project's own ROI table (121 fields of view over 35 genes, selected for that project without
reference to PUPS). One row is removed here: `KATNBL1/1878_H7_32`, whose images were never
downloaded, leaving **120 ROIs over 34 genes**.

Every track is **self-contained**: its `source_list.csv` drives the image download, and its
`ROI*.txt` tables are committed, so a fresh clone on any machine reproduces the same crops.

---

## 2. Change overview

Two pieces of third-party code are touched, both mechanically auditable, and they are touched
for different reasons:

| What | Change | Audit |
|---|---|---|
| `PUPS_Code/` — **the model under test** | **one line**, a dead `import tensorflow` commented out | `git -C PUPS_Code diff -- src/` |
| `Image_Preparation/` — the HPA image tooling | **none**; one of its eight scripts is forked into `Image_Preparation_Modified/` instead | `git -C Image_Preparation status` must be empty; the fork is verified on every run to equal clone + patch |

The distinction matters for the argument this repository makes. **Nothing about PUPS's
behaviour is altered**, which is key to this investigation. The forked tool only *fetches
images*; it does not touch the model, the inputs it receives, or any metric.

### The fork of the HPA downloader

`Image_Preparation_Modified/1_HPA_Image_Download_BatchDownload.py` adds a fallback to archived
HPA releases, because the original reads only the current release and so loses a withdrawn
antibody's fields of view in silence (see §6, step 4b). It is deliberately **not** proposed
upstream: the companion project browses whatever HPA currently publishes, whereas here a *named*
field of view either exists or a figure row is wrong.

Every changed region carries a `CHANGED-BY-US (n/7)` comment and the file opens with a header
naming the origin commit. `4_Run_Image_Preparation.py --tool download` audits it before running
it — applying `0001-archived-release-fallback.patch` to the clone's file must reproduce ours
byte-for-byte, or the run stops with instructions. Editing one without the other cannot slip
through. `Image_Preparation_Modified/README.md` explains every change and records the
end-to-end test.

```bash
grep -n CHANGED-BY-US Image_Preparation_Modified/*.py            # the quick tour
diff -u Image_Preparation/1_HPA_Image_Download_BatchDownload.py \
        Image_Preparation_Modified/1_HPA_Image_Download_BatchDownload.py   # the full diff
```

### The one-line change to PUPS

`PUPS_Code/` is a real git clone pinned at
`e0354b9e6374c1f6676fcbc9d9c15ca6c7fccd6d` (2025-05-14), so this is mechanically checkable:

```bash
git -C PUPS_Code diff -- src/     # the code audit
git -C PUPS_Code rev-parse HEAD   # the pinned commit
```

It is applied from a tracked patch file, so it is version-controlled here rather than described
in prose:

```bash
git -C PUPS_Code apply "$(pwd)/patches/0001-comment-out-unused-tensorflow-import.patch"
```

Verified to apply cleanly to a pristine checkout of the pinned commit.

```diff
--- a/src/model/full_model.py
+++ b/src/model/full_model.py
 # NOTE: Need to import tensorflow before pytorch lightning else protocol buffer runtime library miscongruencies
-import tensorflow as tf
+# import tensorflow as tf  # CHANGED-BY-US: unused (no `tf.` in this file); installing
+#   TensorFlow core-dumps alongside torch — the very protobuf clash their comment warns about.
```

We first tried *installing* TensorFlow so their code could run untouched; `tensorflow-cpu`
alongside torch **core-dumped**. Commenting out a provably dead import was the smaller change.
Their model then loads and runs unmodified: `SubCellProtModel()` builds (5,798,400 parameters),
both published checkpoints load with **0 missing / 0 unexpected keys**, inference runs on CPU.

The clone ships 23 checkpoints, 1.6 GB, of which this project uses two.
`0_Get_Checkpoints.py` materialises just those two from the clone's own object store, offline,
and does nothing if they are already present — so it is safe to delete the other 21 to reclaim
the space. Use `git diff -- src/` for the code audit, since deleting them makes `git status`
noisy.

### Their code we use as-is

`src/model/*` (the model) and `checkpoints/*` (published weights). `CLASSES` is imported from
their `src/dataset/dataset.py` unmodified.

### Their code we do not use, and why

| Component | Reason |
|---|---|
| `src/dataset/download_data.py` | requires their unreleased MongoDB. Its logic is *copied* (see §5), but its HPA scraper's locator (`soup.find("span", string="Matching transcripts")`) no longer matches HPA's current page markup |
| `src/utils/esm2_utils.py` | cannot run as published — `get_esm2_representation()` calls `np.array()` while numpy is never imported. Reimplemented with its settings copied line by line |
| `train.py`, all notebooks | we evaluate published checkpoints and never retrain |

### Reimplemented, with settings copied verbatim

`5_Make_ESM2_Embeddings.py` replaces `get_esm2_representation()`: `esm2_t33_650M_UR50D`
(**fair-esm**, not the HuggingFace port), `repr_layers=[33]`, per-residue, **BOS/EOS retained**
so `x_len == L + 2`, `MAX_SEQ_LEN=2000`, 1280-dim, zero-padded.

**Run step 5 with `--faithful` whenever the machine has the memory**. It is the recommended
mode and reproduces PUPS's order of operations exactly.

Their code feeds the *whole* sequence to ESM-2 and truncates the representation afterwards.
Because ESM-2 attention is O(L²) and the longest proteoforms are not always computable on a
laptop, **our default** truncates the sequence to `MAX_SEQ_LEN` *before* the model call. That is
a deviation from their order, and `--faithful` restores it.

The output shape is identical either way — `min(L+2, 2000) × 1280`, since `rep[:MAX_SEQ_LEN]`
runs in both modes. What differs is what attention saw: with `--faithful`, the embeddings of the
first 2000 residues were computed in the context of the entire protein, which is what PUPS did.

Only sequences over 2000 aa are affected at all, so the cost is small and bounded:

| Track | Sequences | Over 2000 aa |
|---|---|---|
| `Fig_2C` | 6 | 1 — ALMS1 4126 |
| `Independent_Selection` | 34 | 2 — MKI67 3256, SON 2426 |
| `Fig_Ext5B` | 9 | none |

The longest sequence costs roughly `(L/2000)²` the peak of a 2000-length pass — about 4× for
ALMS1, transient under `torch.no_grad()`. If it does run out of memory, drop the flag for that
track and say so; the finding does not depend on it.

Two things to keep straight. **Use the same mode for every track**, or you are left with
embeddings produced two ways and a footnote to explain for no benefit — `metadata.json` records
`faithful_mode` and `sequences_truncated_before_esm2` per run, so this is auditable either way.
And **the choice cannot bias the format comparison**: one embedding feeds all three stacks, so
any change shifts every arm together.

Step 5 has **no resume logic** — it recomputes every sequence and overwrites its three output
files, so re-running with a different flag needs no cleanup.

**Validated against the authors' own data.** `2_Resolve_Source_List.py --self-test` reproduces
the proteoform shown in their MongoDB screenshots (`PUPS_Code/mongo.001_transcribed.txt`):
`TSPAN6-201` / `ENSP00000362111` → 245 aa, prefix identical to their stored `sequence`, and
`L + 2 = 247` exactly matching their stored `length`. All four assertions pass.

---

## 3. Setup from scratch

**Requirements**: This code was tested on Linux Ubuntu, Python 3.10+, **~20 GB free disk** (see
the budget at the end of this section), network access to `github.com`, `rest.ensembl.org`,
`*.proteinatlas.org` and `dl.fbaipublicfiles.com`. Steps 4c/4d need a display and **tkinter**
(`sudo apt-get install python3-tk` on Debian/Ubuntu).

This pipeline is **glue around two other repositories**, both cloned rather than copied, so
neither is vendored and both keep their own history and licence.

```bash
# 1. this repository
git clone <this-repo> && cd PUPS_Examples

# 2. the model under test — uhlerlab/PUPS, pinned
git clone https://github.com/uhlerlab/PUPS.git PUPS_Code
git -C PUPS_Code checkout e0354b9e6374c1f6676fcbc9d9c15ca6c7fccd6d
git -C PUPS_Code apply "$(pwd)/patches/0001-comment-out-unused-tensorflow-import.patch"

# 3. the HPA image tooling — the companion bleed-through project, pinned
git clone https://github.com/cellprofiling/HPA_BleedThrough_Exploration.git Image_Preparation
git -C Image_Preparation checkout 6225801e46409f9b8ab9e9274656c9dd25a5c4ad
#    used UNMODIFIED: `git -C Image_Preparation status` must stay clean.
#    One of its eight scripts is forked into Image_Preparation_Modified/ instead (see §2);
#    that fork is committed here, so there is nothing to apply.

# 4. python environment  (tested with Python 3.10.12)
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# 5. the two checkpoints we use (~140 MB, from PUPS_Code's own object store)
./.venv/bin/python 0_Get_Checkpoints.py

# 6. GATES — both must pass before anything else
./.venv/bin/python 1_Verify_PUPS_Model.py
./.venv/bin/python 2_Resolve_Source_List.py --self-test
```

### The three repositories

| Folder | Upstream | Pinned at | Our changes |
|---|---|---|---|
| *(this repo)* | — | — | the pipeline itself |
| `PUPS_Code/` | [uhlerlab/PUPS](https://github.com/uhlerlab/PUPS) | `e0354b9e` | **one line** (§2) |
| `Image_Preparation/` | [cellprofiling/HPA_BleedThrough_Exploration](https://github.com/cellprofiling/HPA_BleedThrough_Exploration) | `6225801e` | **none** — one tool forked into `Image_Preparation_Modified/` (§2) |

`Image_Preparation/` provides eight scripts: the batch HPA downloader, a single-protein
downloader we do not use, the two ROI-selection GUIs, and four bleed-through
comparison-matrix generators. `4_Run_Image_Preparation.py` exposes seven of them and invokes
them without ever editing the clone — see §6.

Those scripts resolve their paths from `Path(__file__).parent`, so they cannot write to a
different directory. The runner therefore copies the chosen tool into the track folder and runs
it there, so the track ends up showing exactly which code produced its contents. What it asserts
before running depends on the tool:

| Tool | Source | Assertion |
|---|---|---|
| `download` | `Image_Preparation_Modified/` | the fork equals the clone's file **plus its patch**, byte-for-byte (§2) |
| `roi`, `examples`, `matrix*` | `Image_Preparation/` | **byte-identical** to the clone |

Either way the clone stays pristine and the run log states which of the two it used.

**Which directory each tool runs in matters.** `2_Square_Selector.py` derives its Gene /
Antibody / CellLine columns from folder *depth* (`folder.relative_to(script_dir).parts[:3]`), so
it must sit directly above the `GENE/ANTIBODY/CELLLINE` hierarchy. `download` writes gene folders
next to itself and the runner then moves them into `Images/`; every other tool is therefore
placed and run inside `<track>/Images/`, with the ROI tables copied down before the run and
moved back up to `<track>/` after it. Run from `<track>/` instead and the selectors emit rows
labelled `Gene = "Images"` with an extra `Images/` path prefix, every annotation becomes `N/A`,
and all fields of view collapse into one pseudo-gene — silently, with a zero exit status. If you
add a tool to the runner, put it in the same working directory.

ESM-2 weights (2.6 GB) download on first use of step 5, into `.cache/torch/` inside the
project — no dependence on a home-directory cache.

**Disk budget**, measured on this machine 2026-09-05: `PUPS_Code` 2.0 GB (1.6 GB of that is the
23 checkpoints, restorable on demand) · ESM-2 weights 2.6 GB · venv 6.1 GB (a CUDA-build torch,
even on CPU) · `Image_Preparation` 0.5 MB · images 915 MB for `Fig_Ext5B`, 469 MB for `Fig_2C`,
~4 GB for the 34-gene `Independent_Selection` set · intermediates ~0.5 GB per track. Budget **~20 GB** to
run all three tracks comfortably.

---

## 4. Repository layout

```
PUPS_Examples/
├── README.md                the GitHub landing page; points here
├── PIPELINE.md              this file
├── EXAMPLE_RUN.md           annotated console output of a complete run, for comparison
├── LICENSE                  MIT, matching the companion project
├── media/                   figures used by the documentation (CC BY-SA, see its README)
├── patches/                 our one-line change to PUPS, as an appliable patch
├── requirements.txt         pinned packages (tested with Python 3.10.12)
├── run_track.sh             one track end to end, unattended — for `nohup` on a server
├── 0_ … 11_*.py             the pipeline, in order (4b is the image-completeness gate)
│
├── PUPS_Code/               CLONE of uhlerlab/PUPS @ e0354b9e — one line changed
├── Image_Preparation/       CLONE of cellprofiling/HPA_BleedThrough_Exploration
│                            @ 6225801e — unmodified
├── Image_Preparation_Modified/  our marked fork of ONE of its eight scripts, plus the patch
│                            that audits it (§2)
│
├── Fig_Ext5B/               ┐
├── Fig_2C/                  ├─ one folder per track
└── Independent_Selection/   ┘
```

Each track holds **inputs only** in a clean checkout:

| File | Meaning |
|---|---|
| `source_list.csv` | the target table — hand-entered from the paper figure, or the ROI list |
| `ROI.txt` | 640×640 crop coordinates, one per field of view |
| `ROI_examples.txt` | as above plus `Annotation` and `Example` columns; `Example=T` selects one FOV per gene |
| `Images/<GENE>/<ANTIBODY>/<CELLLINE>/` | downloaded HPA images |

`Images/` is generated by step 4a and not committed — with **one exception, which is committed**:
`Fig_Ext5B/Images/N4BP2/HPA036770/U2OS/` (22 files, ~72 MB). HPA has withdrawn that antibody's
immunofluorescence data and no longer serves its TIFFs at all, so a clean clone could not rebuild
that row; carrying it is what keeps ED Fig 5b reproducible 9-of-9. Full story in the step 4b note
in §6.

Everything else is generated and safe to delete: `targets_resolved.tsv`, `sequences.json`,
`Organelle_Example_GenesAndCellLines.txt`, `esm2_output/`, `work/`, `figures/`, `PROVENANCE.*`,
`Crops/`.

`Independent_Selection` additionally ships `ROI_examples_Selected.txt` (6 curated fields of view) for
small figures. Its `source_list.csv` is the original `Note,Link,CellLine` download list the
dataset was built from, so step 2 parses the ENSG straight out of the HPA URL.

---

## 5. How proteoforms are resolved

This is the part most likely to go silently wrong, so it is worth understanding before running
anything.

PUPS keys its dataset on **HPA proteoform labels** like `TSPAN6-201`, and stores an **ENSP**
(Ensembl protein id) per label. From their MongoDB screenshot:

```
_id                        : "TSPAN6-201"
splice_isoform_ensemble_id : "ENSP00000362111"
sequence                   : "MASPSRRLQTKPVITCFKSVLLIYTFIFWITGVILLAVGIWGK…"
esm2_representation.length : 247                    (= 245 aa + BOS + EOS)
```

That label → ENSP mapping exists in exactly **one** place: the "Matching transcripts" block on
HPA's `/antibody` page, which is what their `antibody_scrape()` reads.

```python
url = f"https://{hpa_version}.proteinatlas.org/{gene}-{gene_name}/antibody"
# ...  Matching transcripts   ALMS1-204 - ENSP00000478155 [100%]
"splice_isoform_ensemble_id": re.search(r"(ENSP\d+)", splice_variant.text).group(0)
```

**Two plausible shortcuts silently return the wrong protein:**

- **Ensembl transcript names drift.** `C1QL3-202` meant `ENSP00000480149` (213 aa, since
  retired); today the same label means `ENSP00000520824` (287 aa).
- **The gene XML has no proteoforms.** Its ENSPs come from AlphaFold `<chain>` entries. No HPA
  XML in any version (checked current and v18–v23) contains the `-2xx` labels.

### Two modes, auto-selected from the table's schema

| Mode | Used for | Rule |
|---|---|---|
| `label` | `Fig_Ext5B`, `Fig_2C` — the paper names the proteoform | scan HPA **v22 → v16** for the label; when versions disagree take the newest and print the disagreement |
| `gene` | `Independent_Selection` — we chose the genes ourselves | take the **longest** matching transcript on HPA v22 |

`--mode` overrides the auto-detection; an `ensp_override` column in `source_list.csv` always
wins for a single row.

**Why `gene` mode takes the longest.** There is no published label to honor, so the only
question is which proteoform to embed. A rule is necessary because the number of matching
transcripts ranges from 1 (ABCE1) to 30 (CANX).

Taking the *lowest-numbered* label instead, on the assumption that `-201` is canonical, is
wrong for **~9% of genes** and badly so — measured across the 34-gene set:

```
AP2B1   -201  181 aa   vs  -211  951 aa    (5x truncated)
TAF15   -205  223 aa   vs  -215  592 aa
PTPN20  -201  339 aa   vs  -202  420 aa
```

Handing PUPS a 181 aa fragment when the protein is 951 aa is not a defensible representation of
"the protein", so **length decides**, with the lowest label as tie-break. Critically, **the
choice cannot bias the comparison** either way: the same embedding enters all three stacks, so
it shifts every arm together. All alternatives are recorded in
`all_matching_transcripts`, and `ensp_override` forces a specific one.

### Worked examples (established 2026-09-03)

| Label | Resolves to | Via |
|---|---|---|
| `ALMS1-204` | `ENSP00000478155`, 4126 aa | HPA v19–v22. v16–v18 used old-style `ALMS1-001`; v23 renumbered it `-203` |
| `C1QL3-202` | `ENSP00000480149`, 213 aa | HPA v22; sequence from the `feb2023` Ensembl archive (retired id) |
| `DUSP27-203` | `ENSP00000404874`, 1158 aa | HPA v19/v20 under the pre-rename symbol (now `STYXL2`) |
| `EIF4G1-202` | `ENSP00000316879`, 1599 aa | v19–v22; v16/v17 map it to `ENSP00000338020` — disagreement reported |

---

## 6. Run order

Replace `TRACK` with `Fig_Ext5B`, `Fig_2C` or `Independent_Selection`.

Times are **measured**, on an 8-thread CPU server with the ESM-2 weights already cached, for two
tracks: `Fig_Ext5B` (28 fields of view, 9 genes) and `Independent_Selection` (117 tiles, 34
genes). `run_track.sh` stamps every step, so a run reports its own.

| # | Command | Needs | `Fig_Ext5B` | `Independent_Selection` |
|---|---|---|---|---|
| 2 | `2_Resolve_Source_List.py --track TRACK` | network | 6 min | 10 min |
| 3 | `3_Write_Provenance.py --track TRACK` | network | 3 min | 8 min |
| 4a | `4_Run_Image_Preparation.py --track TRACK --tool download` | network | 10 min | 43 min |
| 4b | `4b_Verify_Track_Images.py --track TRACK` | network | seconds | seconds |
| — | `--tool roi` / `--tool examples` | **GUI** | **NEW TRACKS ONLY.** Every shipped track has committed ROI tables; running these overwrites them, and `Fig_Ext5B`'s were matched by eye to the published panels and cannot be regenerated automatically. | |
| 5 | `5_Make_ESM2_Embeddings.py --track TRACK --faithful` | 2.6 GB once | 30 s | 90 s |
| 6 | `6_Build_Input_Variants.py --images-root TRACK/Images --roi TRACK/ROI_examples.txt --out TRACK/work/variants` | — | 50 s | 3 min |
| 7 | `7_Run_PUPS_Inference.py --variants TRACK/work/variants --embed TRACK/esm2_output --out TRACK/work/predictions` | — | 12 s | 35 s |
| 8 | `8_Compute_Metrics.py --pred TRACK/work/predictions --variants TRACK/work/variants --out TRACK/work/metrics` | — | 4 s | 12 s |
| 9 | `9_Figure_Formats.py --pred TRACK/work/predictions --variants TRACK/work/variants --out TRACK/figures` | — | 7 s | 34 s |
| 9b | `9_Figure_Formats.py … --all-tiles` | — | 10 s | 41 s |
| 10a | `10_Figure_Leakage.py --pred TRACK/work/predictions --variants TRACK/work/variants --embed TRACK/esm2_output --out TRACK/figures --threshold 0` | — | 45 s | 105 s |
| 10b | same, `--threshold 1` | — | 45 s | 108 s |
| 11 | `11_Ablate_Channels.py --variants TRACK/work/variants --embed TRACK/esm2_output --out TRACK/work/ablation` | — | 20 s | 64 s |
| 12 | `4_Run_Image_Preparation.py --track TRACK --tool matrix --stdin 1` | — | 37 s | 20 s |

**End to end: 23 min / ~1 GB for `Fig_Ext5B`, 73 min / ~4 GB for `Independent_Selection`.**
Steps 2–4a are 19 of those 23 and 61 of those 73. Everything runs on CPU; inference measured
73 ms per forward pass on a workstation, 42–58 ms on a server.

Three numbers above need a qualifier:

- **4b** is seconds when nothing is missing, minutes when it walks the archived release hosts.
- **Step 5** excludes the one-time 2.6 GB ESM-2 download.
- **Step 4a resumes**, so a second run over an already-downloaded track is much faster.

Step 5 is written with `--faithful`, the recommended mode (§2). Drop the flag only if the
machine runs out of memory on a long sequence; the run records which mode was used.

**Step 10 must be run twice.** `--threshold 0` and `--threshold 1` are not alternatives —
reproducing Extended Data Fig. 5b needs both, paired across the checkpoint/input diagonal. §9
explains why, and it is the easiest thing in this pipeline to get wrong.

Steps 6 and 7 take `--limit N` for a quick smoke test on the first N tiles. Step 6 takes
`--no-previews` to skip the preview PNGs. Step 5's `--max-len` overrides `MAX_SEQ_LEN=2000`;
leave it alone — 2000 is PUPS's value and changing it makes the embeddings incomparable to
theirs.

### Unattended: `run_track.sh`

```bash
nohup ./run_track.sh Independent_Selection > /dev/null 2>&1 &
tail -f Independent_Selection/run_*.log
```

Runs steps 2 → 12 for one track, including a second step-9 pass with --all-tiles that paginates
every tile into a multi-page PDF, logging to `<track>/run_<timestamp>.log` and to the terminal.
It derives `--n` from the track's own ROI table so the figures cannot silently drop genes,
retries **only** the download step (the one that fails for transient reasons) and stops on any
other failure rather than carrying a partial result forward. It never runs the ROI selectors.

Panel B needs its own size, because it is a gene × gene matrix rather than a list of rows: up
to 9 genes it renders as one matrix, and above that `--all-genes` splits into disjoint groups
whose scores are aggregated as a gene-weighted mean. The script prints both numbers on its
`GENES :` line, so which happened is always visible.

Every step is stamped with the wall-clock time and the seconds elapsed since the run started,
so a run reports its own per-step cost rather than relying on the estimates in the table above
— which is where those numbers came from. `Fig_Ext5B` took 23 minutes this way.

**[EXAMPLE_RUN.md](EXAMPLE_RUN.md) shows this script's console output end to end**, alongside
the step-by-step Quickstart run, so you can compare against a run that is known to have
worked.

### Per-track notes

**Steps 4c/4d are only for a new track.** `Fig_Ext5B`, `Fig_2C` and `Independent_Selection`
ship hand-made ROI tables; re-running the selectors destroys them. `Fig_Ext5B`'s `Example=T`
rows were matched by eye to the published panels, which is what lets those figures sit beside
theirs.

**`Fig_2C` is the exception: its `Example=T` flags are deliberately representative, not matched
by eye.** Fig 2c shows "six randomly selected cells", so there is no published panel to line up
against — say so in the methods. The track holds 14 fields of view over 6 genes, with AMPD3
contributing 4 (two antibodies × 2), so step 4d is also where one antibody per gene gets chosen;
for AMPD3 it selected `HPA047408`. All 6 are set.

**After any download, check the per-gene field-of-view count.** The original downloader has no
retry around the HPA XML fetch, and its bare `except` prints an error then continues — so it
could **exit 0 having silently skipped genes** — transient proxy or HTTP errors can cost you most of
a track. The fork now tallies skipped entries, lists them and **exits non-zero**; re-running
resumes, and step 4a prints the per-gene counts. Their bare `except` clauses are unchanged and
deliberately out of scope, so a *partial* gene can still slip through — which is what step 4b is
for.

**Step 4b is not optional, and it is the reason a fresh machine does not silently produce a
smaller figure.** The original downloader reads `https://www.proteinatlas.org/<ENSG>.xml` — the
current release only, with **no fallback to an older one** — and takes its image URLs from that
single document. When HPA withdraws an antibody's immunofluorescence data the fields of view
simply are not in the XML it read: no warning, exit 0. Step 6 then reports them as "source
images not present" and also exits 0.

Two independent things now cover that, at different granularities:

- **Step 4a** runs the fork (§2), which spots any antibody the current release lists with no
  ICC/IF images at all and walks `v23` → `v15` for exactly those, per *antibody*, from the
  download list.
- **Step 4b** checks every field of view named in the ROI table for the nine files step 6 opens,
  walks the archived hosts for anything absent, and **exits non-zero** on whatever is still
  missing — per *field of view*, against `ROI.txt`.

The overlap is intentional. 4b is the one that actually fails the pipeline on a named missing
tile, and it also catches images downloaded before the fork existed.

*Known case, and the limit of what the archive can return.* In HPA v25 the N4BP2 antibody
`HPA036770` is still listed but carries zero ICC/IF images; v19–v23 likewise. Only **v18** still
lists its U2OS fields of view `410_D12_1` and `410_D12_2` — the ED Fig 5b N4BP2-201 (Holdout 2)
row, with `410_D12_2` as the selected example. From v18, step 4b recovers all ten JPGs
byte-for-byte, but `<prefix>_<channel>.tif.gz` answers **HTTP 400** on every release host and on
`images.proteinatlas.org`: for a withdrawn antibody HPA keeps the JPGs and drops the TIFFs.

Because the 16-bit TIFF arm is the control the whole comparison rests on, those two fields of
view **cannot be reconstituted from public sources**. Their TIFFs came from an internal
CellProfiling archive, not from proteinatlas.org — they carry their original 2010 acquisition
timestamps and `*_image.ome.xml` sidecars. They are therefore **the one exception to `Images/`
being generated**: that folder is committed (~72 MB, see `.gitignore`) so ED Fig 5b reproduces
9-of-9 from a clean clone. This is a property of HPA's release policy, not of the pipeline.

Verified on this exact case: step 4a's fork reports
`v18: recovered 2 image URL(s) for HPA036770` and produces three antibody folders where the
unmodified script produces two.

### Figure outputs

**`9_Figure_Formats.py` — Panel A: predictions across the three input formats**, one row per
field of view, columns `composite input | composite +0.19 (published) | composite, no threshold |
single_jpg | single_tiff | ground truth`.

- Beneath every prediction panel: **`r` and `MSE` against the ground-truth column** (the 16-bit
  TIFF protein channel), predictions clipped to [0,1] — the same definitions step 8 tabulates,
  computed by shared code so the figure and the tables cannot disagree.
- The **LUT is printed as a reference bar**. Default `--cmap viridis`, matching their published
  figures; `--cmap gray` matches what their released `viz_utils.py` actually does.
- `--scale` controls the intensity mapping and it changes what you can conclude:
  **`panel`** (default) stretches each panel to its own range — matches how their figures look
  and is the fair way to judge *spatial* agreement; **`absolute`** fixes 0–1 for every panel,
  which shows amplitude honestly but makes every prediction look dim beside the rescaled ground
  truth. The printed `max` keeps amplitude visible in either mode. See §9.
- Selection: default is the `Example=T` tiles, one per gene. `--select-from FILE` takes the
  tiles listed in another ROI table, `--tiles ID...` names them explicitly, `--n` caps the count.
- `--all-tiles` paginates **every** tile into a multi-page PDF plus one PNG per page,
  `--per-page` sets the page size. `--checkpoint` picks which model, `--all-conditions` emits
  every condition rather than the curated column set.

**`10_Figure_Leakage.py` — Panel B: does PUPS follow the sequence, or the image?** An
*n* × *n* cross-pairing matrix, rows = cells, columns = protein sequences, reproducing the design
of their Extended Data Fig. 5b. It runs its **own** inference, because steps 7–8 only score
matched cell/sequence pairs.

Read it as: **rows constant ⇒ the model follows leakage in the input; columns constant ⇒ it
follows the sequence.** Quantified in `panelB_scores*.csv` and printed per condition:

| Column | Meaning |
|---|---|
| `matched_r` | prediction vs truth when the sequence is the correct one |
| `mismatched_r` | same cell, *wrong* sequence — **the leakage measure**; high means the model reproduces the stained protein regardless of input sequence |
| `sequence_effect` | `matched_r − mismatched_r`; ≈ 0 means the sequence is doing nothing |
| `within_row_cv` | ≈ 0 is the row-constant failure of ED Fig. 5b |
| `within_col_cv` | large relative to `within_row_cv` ⇒ output tracks the cell |

One figure per `variant × checkpoint`, so the format comparison is the set of matrices at a
fixed checkpoint. `--n` sets the matrix size (default 6); `--all-genes` covers every gene in
disjoint groups of `--n`, with mismatched pairs formed **within** a group. `--select-from` and
`--tiles` choose tiles as in Panel A; `--cmap` as above.

⚠ `sequence_effect` is a small difference between two large numbers. At n = 6 it is unstable —
see §9.

**`11_Ablate_Channels.py` — supporting work.** Figures do not depend on it and only need
steps 6–10. The purpose of step 11 is to explore why the same field of view predicts dimmer
from a JPG so that the format effect cannot be dismissed as a preprocessing error. It
measures rather than argues:

- **support vs output** — per tile, the `single_jpg`/`single_tiff` ratio of input mean, input
  support (fraction of nonzero pixels) and output amplitude. If amplitude followed total signal,
  the output ratio would sit near the mean ratio. Writes `ablation.csv`.
- **channel swap** — feeds `single_tiff` with one landmark replaced by `single_jpg`'s, to
  localise the cause.
- `--blocks --images-root … --roi …` adds the raw-file analysis: what fraction of *k*×*k* blocks
  is entirely zero in the JPG versus the TIFF, which is the mechanism. Writes `block_zeros.csv`.

`--checkpoint` and `--threshold` as elsewhere. The default is `--threshold 1`, PUPS's published
configuration; pass `--threshold 0` to see the support effect without the threshold. See §9.

- Step 12 — the published bleed-through comparison matrix, from the companion project's
  generator, on the same crops.

---

### What a run produces

| Path | From | Contents |
|---|---|---|
| `targets_resolved.tsv`, `sequences.json` | 2 | proteoform → ENSP → sequence, with how each was chosen |
| `PROVENANCE.tsv`, `PROVENANCE.md` | 3 | every URL fetched, with status and fingerprint |
| `Images/<GENE>/<ANTIBODY>/<CELLLINE>/` | 4a | the HPA files, 10 per field of view |
| `esm2_output/` | 5 | `esm2_representations.npz`, `esm2_lengths.npz`, `metadata.json` |
| `work/variants/` | 6 | one `.npz` per tile (`composite`, `single_jpg`, `single_tiff` stacks + `channel_order`), `index.csv`, `previews/*.png` |
| `work/predictions/` | 7 | one `.npz` per tile, 6 conditions named `<ckpt>__<variant>__thr{0,1}` = 3 stacks x the 2 train/test-consistent pairs. `--all-conditions` adds the other 6 |
| `work/metrics/` | 8 | 2 per-tile tables + 4 summaries (see above) |
| `figures/PanelA_<ckpt>.{pdf,png,svg}` | 9 | predictions across formats, with per-panel `r`/`MSE`/`max` |
| `figures/PanelB_<variant>_<ckpt>_thr<n>.{pdf,png,svg}` | 10 | cross-pairing matrices — one per stack, so 3 per threshold and 6 in total |
| `figures/panelB_scores_thr<n>.csv`, `panelB_aggregate_thr<n>.csv` | 10 | the leakage quantification |
| `work/ablation/ablation.csv`, `block_zeros.csv` | 11 | the support/ER diagnostics |
| `HPA_comparison_matrix_4*.{pdf,svg,png}` | 12 | the companion project's published comparison matrix, on the same crops |

---

## 7. Verification

```bash
git -C PUPS_Code diff -- src/                                # our footprint in the model repo
git -C Image_Preparation status --porcelain                  # must be EMPTY — we never edit it
diff -u Image_Preparation{,_Modified}/1_HPA_Image_Download_BatchDownload.py  # our forked tool
grep -n CHANGED-BY-US Image_Preparation_Modified/*.py        # every changed region, tagged
./.venv/bin/python 1_Verify_PUPS_Model.py                    # checkpoints load and run
./.venv/bin/python 2_Resolve_Source_List.py --self-test      # chain matches their MongoDB
./.venv/bin/python 3_Write_Provenance.py --track TRACK --verify   # upstream drift
./.venv/bin/python 4b_Verify_Track_Images.py --track TRACK --check-only  # no missing images
```

`3_Write_Provenance.py` writes `PROVENANCE.tsv` and `PROVENANCE.md` listing **every external
URL** with its HTTP status and a sha256 of the response, so any input can be re-checked by hand.
`--verify` re-fetches and reports changed fingerprints, which catches HPA relabelling or Ensembl
retiring an id.

Its `listed_by_HPA` column checks the ENSP against HPA's **"Matching transcripts"** block on
the `/antibody` page, across releases — the same route step 2 uses, and the only place a
proteoform label maps to an ENSP. A `NO` there is a genuine problem worth stopping for.

It also records label drift: HPA renumbers proteoform labels between releases, so an ENSP can be
correct while its label has changed. `DUSP27-203` reports *"HPA now labels it STYXL2-203 — same
ENSP, label renumbered"*. Both the old and current names belong in the write-up.

⚠ The `hpa_gene_xml` and `hpa_subcellular_page` rows are still logged for provenance, but their
ENSPs are AlphaFold `<chain>` entries and are **not** a proteoform list. Do not use them to
judge whether an ENSP is legitimate — an earlier version of this step did, and reported
`ALMS1-204` as unlisted when HPA has published that ENSP continuously since v18.

---

## 8. Licensing and what this repository contains

**This pipeline is MIT-licensed** (`LICENSE`), matching the companion
[HPA_BleedThrough_Exploration](https://github.com/cellprofiling/HPA_BleedThrough_Exploration)
project it builds on.

Neither upstream repository is redistributed wholesale — both are cloned during setup (§3), so
each keeps its own licence. One **single file** is redistributed, as a modification:

| Repository | Licence | How it is obtained |
|---|---|---|
| this pipeline | MIT | you are reading it |
| `Image_Preparation/` | MIT | `git clone`, unmodified |
| `Image_Preparation_Modified/` | MIT (derivative of the above) | committed here — one modified file, see below |
| `PUPS_Code/` | **GPL-3.0** | `git clone` + a tracked one-line patch |
| hosted HPA images | **CC BY-SA 3.0** | committed here — see below |

`Image_Preparation_Modified/1_HPA_Image_Download_BatchDownload.py` is a **modified copy** of an
MIT-licensed file from the companion project. MIT permits that, and the obligations are met: the
file's header names the origin repository and commit, this project carries the same MIT licence
(`LICENSE`, with CellProfiling's copyright), and the modification is both marked in-line
(`CHANGED-BY-US`) and reproducible as a patch. It is not a fork of that project but just one of its
eight scripts, changed for a reason specific to this pipeline, and deliberately not pushed back (§2).

⚠ **A note on the GPL.** PUPS is GPL-3.0. This repository does not contain or redistribute any
of it: you clone it yourself, and our single change ships as a patch rather than as modified
source. Our scripts do `import` from that clone at runtime, so if you intend to redistribute a
*combined* artifact (a container image, a bundled release), take your own view on whether the
GPL applies to that combination. Distributing this pipeline on its own does not raise the
question.

⚠ **A note on the hosted HPA images**. The field-of-view images carried in this
repository are © the Human Protein Atlas, licensed CC BY-SA 3.0
(https://www.proteinatlas.org/about/licence). Because CC BY-SA is share-alike, these
images keep that licence and are not covered by the MIT licence above; they are
redistributed here under the same CC BY-SA 3.0 terms, with attribution. We host them only
because HPA no longer serves these particular fields of view, and reproducibility depends
on them. Per-image attribution is in that folder's README (Fig_Ext5B > Images > N4BP2).

What a fresh clone contains: the pipeline scripts and `run_track.sh`, this file, `README.md`,
`EXAMPLE_RUN.md`, `LICENSE`, `requirements.txt`, `media/`, the PUPS patch, the three files of
`Image_Preparation_Modified/`, each track's `source_list.csv` and hand-made `ROI*.txt` tables,
and the one carried field-of-view folder with its CC BY-SA attribution (§4).
`git ls-files | wc -l` is the authoritative count.

Everything else (the two upstream clones, all other images, every generated artefact, the
virtualenv and the model caches) is regenerated by the pipeline.

---

## 9. Caveats — read before quoting any number

Properties of the data, the model and the published figures that change how results must be
interpreted. None of these are bugs in this pipeline; all of them will bite you if ignored.

### The word "threshold" means two different things

This is the easiest thing to get wrong, because both are called the 0.19 threshold.

| | What it is | Selected by |
|---|---|---|
| **trained** with/without it | two separately trained models | the **checkpoint** — `default` vs `nothreshold.ckpt` |
| **applied to the input** at inference | preprocessing before the forward pass | our `--threshold 0/1` flag — `composite` + `--threshold 1` is PUPS's published configuration |

**Extended Data Fig. 5b's axis is the first one.** Its top panel is the model *trained* without
thresholding; `nothreshold.ckpt` ships in their repo because it is that panel's model.

A model trained on thresholded landmarks must be *fed* thresholded landmarks, so only two of the
four combinations are meaningful:

| | input `--threshold 0` | input `--threshold 1` |
|---|---|---|
| `nothreshold` ckpt | ✅ **their ED Fig. 5b top panel** | ✗ train/test mismatch |
| `default` ckpt | ✗ train/test mismatch | ✅ **their ED Fig. 5b bottom panel** |

So reproducing their figure means running step 10 **twice** and pairing across the diagonal:

```bash
./.venv/bin/python 10_Figure_Leakage.py ... --threshold 0   # -> PanelB_*_nothreshold_thr0  (top)
./.venv/bin/python 10_Figure_Leakage.py ... --threshold 1   # -> PanelB_*_default_thr1      (bottom)
```

The off-diagonal cells are still written, and are still interesting as ablations, but they are
**not** what the paper shows. Do not compare `default_thr0` against `nothreshold_thr0` and call
it their figure.

### Small tile counts do not replicate

Panel B's `sequence_effect` is a small difference between two large numbers and is unstable at
small *n*; an early 6-gene result in this project did not survive going to 34 genes.

`Fig_2C` (6 genes) and the `Example=T` selections exist to make **figures**. Quote statistics
from `Independent_Selection` with `--all-genes`, and state *n* wherever a number appears.

### Whole-image correlation cannot discriminate between the formats

Pearson *r* between a variant's channels and the TIFF's comes out around 0.99 for every format,
because bright structure dominates the correlation while any compositing or compression artifact
lives in the low-intensity range. Quoting that number would suggest the formats are equivalent.

The discriminating measure is the residual map, computed by step 8:

```
leak_r = corr(rank(variant_channel) − rank(tiff_channel), tiff_protein_channel)
```

Positive `leak_r` in a **landmark** channel means that channel carries protein-shaped excess
signal.

**Compare the right cells.** `microtubule` and `nucleus` are the channels `composite` recovers by
splitting the RGB file and `single_jpg` reads from separate files, so those two isolate
compositing. The `er` and `protein` columns do **not**: both take ER from the same `yellow.jpg`, so a
difference there is JPEG artifact rather than compositing, and the protein channel is never a
model input.

### An 8-bit and a 16-bit copy of the same field of view can predict very differently

`single_jpg` and `single_tiff` are the same fields of view differing only in
bit depth and JPEG compression. After step 6 their means agree closely, they correlate at ~0.99,
and they are indistinguishable in a preview — yet the predictions can differ substantially.

That difference is worth understanding before drawing any conclusion from a format comparison.
`11_Ablate_Channels.py` (§6) exists to test whether it is a normalisation error: it reports the
`single_jpg`/`single_tiff` ratio of input mean, input support and output amplitude per tile, a
per-channel swap that localises which landmark is responsible, and — with `--blocks` — the
mechanism in the raw files, where JPEG's 8×8 DCT quantisation sends whole dark blocks to exact
zero while a TIFF keeps isolated zeros among small nonzero values. PUPS's `//4` downsample
recovers signal from the latter and nothing from the former.

**Run step 11 before attributing a JPG-vs-TIFF difference to bit depth as such.**

⚠ **`single_jpg` is our construction, not something PUPS does.** PUPS reads the composite.
Comparing `single_jpg` against `single_tiff` is a controlled probe separating compositing from
8-bit quantisation; the comparison that bears on the paper is **`composite` vs `single_tiff`**.

### Intensities are per-panel relative, in every figure

Every panel is `rescale_intensity`'d to [0,1] independently, so colour shows relative intensity
*within* a panel and is not comparable between panels. The figures print their LUT as a
reference bar and say so; their published figures do not.

### Their dataset-building code cannot be re-run today

`src/dataset/download_data.py` requires an unreleased MongoDB, and its HPA scraper's locator no
longer matches the live site's markup. §5 reimplements the parts needed, from their code.
