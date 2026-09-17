#!/usr/bin/env python3
"""STEP 5 — protein sequences -> PUPS-compatible ESM-2 representations.

    ./.venv/bin/python 5_Make_ESM2_Embeddings.py --track Fig_Ext5B --faithful   # RECOMMENDED
    ./.venv/bin/python 5_Make_ESM2_Embeddings.py --track Fig_Ext5B              # if memory-tight

Reads <track>/sequences.json (written by step 2) and writes <track>/esm2_output/.
No network needed after the one-time ESM-2 weight download.

MATCHES PUPS/src/utils/esm2_utils.py EXACTLY
--------------------------------------------
    model            esm.pretrained.esm2_t33_650M_UR50D()   (fair-esm, NOT the HF port)
    repr_layers      [33]
    representation   per-residue, results["representations"][33]
    BOS/EOS          KEPT. Their get_esm2_representation passes the UNTRIMMED aa_rep to
                     pad_collate, so the stored tensor includes the BOS row at index 0 and
                     the EOS row at index L+1, and x_len == L + 2. (Their mean_rep trims
                     these, but mean_rep is discarded.)
    MAX_SEQ_LEN      2000, ESM2_EMBEDDING_LEN 1280
    padding          zeros to 2000 rows

We store representations UNPADDED as (L+2, 1280) plus lengths, and pad to 2000 at load time
in step 7. Padding here would write mostly-zeros; unpadded reconstructs exactly.

WHY THIS IS NOT PUPS's OWN FUNCTION
-----------------------------------
Their get_esm2_representation() cannot run as published: src/utils/esm2_utils.py calls
np.array() without importing numpy. It also loads the 650M model
at module import time, and hard-codes device='cpu'. So the logic is reimplemented here rather
than imported — the settings above are copied from their file line by line.

ONE DELIBERATE DEVIATION, AND HOW TO AVOID IT
---------------------------------------------
By default, sequences longer than --max-len are truncated BEFORE the model call. Their code
feeds the whole sequence and truncates the representation afterwards; ESM-2 attention is O(L^2),
so their order is not always computable on a small machine.

    USE --faithful WHENEVER MEMORY ALLOWS. It reproduces their order exactly and is the
    recommended mode (PIPELINE.md section 2).

The output shape is the same either way -- min(L+2, 2000) x 1280, because `rep[:MAX_SEQ_LEN]`
runs in both modes. What differs is what attention saw: with --faithful the first 2000 residues
were embedded in the context of the whole protein. Only sequences over 2000 aa are affected at
all (Fig_2C: ALMS1 4126; Independent_Selection: MKI67 3256, SON 2426; Fig_Ext5B: none), and the longest
costs about (L/2000)^2 the peak of a 2000-length pass, transient under torch.no_grad().

Use the SAME mode for every track. metadata.json records `faithful_mode` and
`sequences_truncated_before_esm2` per run, so which was used is always checkable. The choice
cannot bias the format comparison: one embedding feeds all three stacks, so it shifts every arm
together.

NO RESUME: every sequence is recomputed and the three output files are overwritten, so
re-running with a different flag needs no cleanup.
"""
import argparse
import json
import os
import pathlib
import sys

import numpy as np

MAX_SEQ_LEN = 2000          # PUPS esm2_utils.MAX_SEQ_LEN
EMB_DIM = 1280              # PUPS esm2_utils.ESM2_EMBEDDING_LEN
REPR_LAYER = 33

# Keep the 2.6 GB weight cache inside the project, so runs do not depend on the state of a
# home-directory cache. Must be set before `import esm`.
os.environ.setdefault("TORCH_HOME",
                      str(pathlib.Path(__file__).resolve().parent / ".cache" / "torch"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", required=True,
                    help="track folder, e.g. Fig_Ext5B / Fig_2C / Independent_Selection")
    ap.add_argument("--max-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--faithful", action="store_true",
                    help="RECOMMENDED. Feed full sequences to ESM-2 and truncate the "
                         "representation afterwards, as PUPS does. Costs ~(L/2000)^2 peak "
                         "memory on sequences over 2000 aa; omit only if that runs out.")
    args = ap.parse_args()

    track = pathlib.Path(args.track)
    seq_path = track / "sequences.json"
    if not seq_path.exists():
        sys.exit(f"{seq_path} not found — run 2_Resolve_Source_List.py --track {track} first")
    out = track / "esm2_output"
    out.mkdir(parents=True, exist_ok=True)

    seqs = json.loads(seq_path.read_text())
    print(f"STEP 5 — ESM-2 for {len(seqs)} sequence(s) in {track}")
    print(f"  TORCH_HOME = {os.environ['TORCH_HOME']}\n")

    import torch
    import esm
    print("loading esm2_t33_650M_UR50D (2.6 GB on first run) ...", flush=True)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    bc = alphabet.get_batch_converter()

    reps, lengths, truncated = {}, {}, []
    for i, (key, d) in enumerate(sorted(seqs.items()), 1):
        seq = d["sequence"]
        if not args.faithful and len(seq) > args.max_len:
            truncated.append({"gene": key, "original_length": len(seq),
                              "fed_to_esm2": args.max_len})
            seq = seq[:args.max_len]
        _, _, toks = bc([(key, seq)])
        n_unk = int((toks == alphabet.unk_idx).sum())
        with torch.no_grad():
            o = model(toks, repr_layers=[REPR_LAYER], return_contacts=False)
        rep = o["representations"][REPR_LAYER][0].cpu().numpy().astype(np.float32)
        rep = rep[:MAX_SEQ_LEN]              # PUPS truncates the REPRESENTATION at 2000
        reps[key] = rep
        lengths[key] = rep.shape[0]          # == min(L+2, 2000): PUPS's x_len
        note = "  (truncated)" if any(t["gene"] == key for t in truncated) else ""
        print(f"[{i}/{len(seqs)}] {key:12s} {d.get('transcript_name', '?'):16s} "
              f"{len(seq):>5} aa -> {rep.shape}  UNK={n_unk}{note}")
        if n_unk:
            print(f"    WARNING {n_unk} unknown token(s) — is this really a protein sequence?")

    np.savez_compressed(out / "esm2_representations.npz", **reps)
    np.savez(out / "esm2_lengths.npz", **{k: np.int64(v) for k, v in lengths.items()})
    (out / "metadata.json").write_text(json.dumps({
        "model": "esm2_t33_650M_UR50D", "repr_layer": REPR_LAYER,
        "embedding_dim": EMB_DIM, "max_seq_len": MAX_SEQ_LEN,
        "bos_eos_included": True,
        "storage": "unpadded (L+2, 1280) float32; pad with zeros to 2000 at load",
        "faithful_mode": bool(args.faithful),
        "sequences_truncated_before_esm2": truncated,
        "n_genes": len(reps),
        "source": str(seq_path),
    }, indent=2))

    mb = sum(r.nbytes for r in reps.values()) / 1e6
    print(f"\ndone: {len(reps)} representations, {mb:.0f} MB -> {out}")
    print(f"  truncated before ESM-2: {len(truncated)}")
    print(f"\nNEXT: ./.venv/bin/python 6_Build_Input_Variants.py "
          f"--images-root {track}/Images --roi {track}/ROI_examples.txt "
          f"--out {track}/work/variants")
    return 0


if __name__ == "__main__":
    sys.exit(main())
