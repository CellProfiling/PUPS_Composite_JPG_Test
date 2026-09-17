#!/usr/bin/env bash
# Run one track end to end, unattended. Intended for `nohup`/`setsid` on a server.
#
#     ./run_track.sh Independent_Selection
#     nohup ./run_track.sh Independent_Selection > /dev/null 2>&1 &
#
# Everything is logged to <track>/run_<timestamp>.log AND echoed to the terminal, so an
# interactive run looks normal and a detached run leaves a complete record.
#
# WHY A SCRIPT RATHER THAN PASTING THE COMMANDS
# ---------------------------------------------
#   * --n differs per track and passing it wrong silently drops genes from the figures, so it
#     is derived here from the track's own ROI table instead of being remembered.
#   * the download step is the only one that fails for transient reasons (proxy 5xx, HPA
#     timeouts), so it is retried; every other step is deterministic and a failure there is
#     real, so the run STOPS rather than carrying a partial result into the next step.
#   * step 4b is a hard gate: if a named field of view is missing, nothing downstream is worth
#     computing.
#
# It never runs the ROI selectors (steps 4c/4d): every track ships committed crop tables and
# the selectors would overwrite them.
set -uo pipefail

# Stdout is a pipe into tee, not a terminal, so Python block-buffers it while bash's own echoes
# go straight through. The two streams then interleave wrongly in the log: a tool's output can
# appear BEFORE the line announcing that the tool is being run. Unbuffered costs nothing here.
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")" || exit 1
T="${1:-}"

# Logging is set up BEFORE any validation. Under `nohup ... > /dev/null` a guard that fired
# early used to leave no trace at all: the job exited, the log was never created, and the only
# symptom was `tail: no such file`. The log therefore lands in the track when there is one and
# at the repository root when there is not, so a bad argument is still recorded somewhere.
STAMP=$(date +%Y%m%d_%H%M%S)
if [ -n "$T" ] && [ -d "$T" ]; then LOG="$T/run_$STAMP.log"; else LOG="./run_$STAMP.log"; fi
exec > >(tee -a "$LOG") 2>&1

[ -n "$T" ] || { echo "usage: $0 <track>   e.g. $0 Independent_Selection"; exit 2; }
if [ ! -d "$T" ]; then
    echo "no such track: $T"
    echo "  tracks present here: $(ls -d */ 2>/dev/null | tr -d / | tr '\n' ' ')"
    exit 2
fi
PY=./.venv/bin/python
[ -x "$PY" ] || { echo "no venv at $PY — see PIPELINE.md section 3"; exit 2; }

# ONE run per track. Two concurrent runs write the same targets_resolved.tsv, sequences.json,
# variants/ and predictions/ — the second silently corrupts the first. Easy to do by accident:
# start it with nohup, see nothing (the log takes a moment to appear), and start it again.
LOCK="$T/.run_track.lock"
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "ALREADY RUNNING for $T as PID $(cat "$LOCK") — refusing to start a second run."
    echo "  watch it:  tail -f $T/run_*.log"
    echo "  stop it:   kill $(cat "$LOCK")"
    exit 3
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

started=$(date +%s)
say() { printf '\n═══ %s  [%s, +%ds]\n' "$1" "$(date +%H:%M:%S)" "$(( $(date +%s) - started ))"; }
die() { printf '\n✗ FAILED at %s — see %s\n' "$1" "$LOG"; exit 1; }
run() { local lbl="$1"; say "$lbl"; shift; "$@" || die "$lbl"; }

# How many genes this track has, so --n never silently truncates the figures.
NGENES=$($PY - "$T" <<'PYEOF'
import csv, sys, pathlib
p = pathlib.Path(sys.argv[1]) / "ROI_examples.txt"
if not p.exists():
    p = pathlib.Path(sys.argv[1]) / "ROI.txt"
rows = [r for r in csv.DictReader(open(p, newline=""))
        if (r.get("Skip") or "F").strip().upper() != "T"]
print(len({r["Gene"].strip() for r in rows}))
PYEOF
)
# Panel B is a gene x gene matrix, so unlike Panel A its size cannot just follow the gene
# count: n genes cost n^2 forward passes and the figure stops being readable past roughly
# 9x9. Up to 9 genes it renders as ONE matrix, which is what mirrors their ED Fig. 5b; above
# that --all-genes splits into disjoint groups and the printed table is a gene-weighted mean
# over them. Without this, a 9-gene track silently came out as a 6x6 plus a 3x3.
if [ "$NGENES" -le 9 ]; then PANELB_N="$NGENES"; else PANELB_N=6; fi

echo "TRACK   : $T"
echo "GENES   : $NGENES   (Panel A --n $NGENES, Panel B matrix --n $PANELB_N)"
echo "LOG     : $LOG"
echo "STARTED : $(date)"

run "step 2 — resolve proteoforms"      $PY 2_Resolve_Source_List.py --track "$T"
# Step 3 is the one step whose non-zero exit is INFORMATIVE rather than fatal: it returns 1
# when an ENSP is not listed by HPA, which is a finding to read in PROVENANCE.md, not a reason
# to abandon the run. `run` would call die() here and abort — so it is invoked directly.
say "step 3 — provenance"
if ! $PY 3_Write_Provenance.py --track "$T"; then
    echo "  ⚠ step 3 exited non-zero: an ENSP is not listed by HPA. Continuing —"
    echo "    read $T/PROVENANCE.md before quoting anything for the affected target(s)."
fi

# --- download, retried: the only step that fails for transient reasons -----------------------
for attempt in 1 2 3 4; do
  say "step 4a — download images (attempt $attempt)"
  $PY 4_Run_Image_Preparation.py --track "$T" --tool download && break
  echo "  attempt $attempt returned non-zero; it resumes, so retrying after 60 s"
  sleep 60
done

run "step 4b — image completeness GATE" $PY 4b_Verify_Track_Images.py --track "$T"

run "step 5 — ESM-2 embeddings"         $PY 5_Make_ESM2_Embeddings.py --track "$T" --faithful
run "step 6 — input variants"           $PY 6_Build_Input_Variants.py \
      --images-root "$T/Images" --roi "$T/ROI_examples.txt" --out "$T/work/variants"
run "step 7 — PUPS inference"           $PY 7_Run_PUPS_Inference.py \
      --variants "$T/work/variants" --embed "$T/esm2_output" --out "$T/work/predictions"
run "step 8 — metrics"                  $PY 8_Compute_Metrics.py \
      --pred "$T/work/predictions" --variants "$T/work/variants" --out "$T/work/metrics"

run "step 9 — Panel A (all $NGENES genes)" $PY 9_Figure_Formats.py \
      --pred "$T/work/predictions" --variants "$T/work/variants" --out "$T/figures" --n "$NGENES"
run "step 9b — Panel A, every tile"        $PY 9_Figure_Formats.py \
      --pred "$T/work/predictions" --variants "$T/work/variants" --out "$T/figures" --all-tiles

# Panel B twice: --threshold 0/1 are NOT alternatives. Each renders only its train/test-consistent
# checkpoint (CONSISTENT in 10_Figure_Leakage.py), so thr0 gives ED Fig. 5b's top panel
# (nothreshold) and thr1 its bottom panel (default). PIPELINE.md §9.
for THR in 0 1; do
  run "step 10 — Panel B, all genes, threshold $THR" $PY 10_Figure_Leakage.py \
        --pred "$T/work/predictions" --variants "$T/work/variants" --embed "$T/esm2_output" \
        --out "$T/figures" --threshold "$THR" --all-genes --n "$PANELB_N"
done

run "step 11 — channel ablation"        $PY 11_Ablate_Channels.py \
      --variants "$T/work/variants" --embed "$T/esm2_output" --out "$T/work/ablation" \
      --blocks --images-root "$T/Images" --roi "$T/ROI_examples.txt"
run "step 12 — comparison matrix"       $PY 4_Run_Image_Preparation.py \
      --track "$T" --tool matrix --stdin 1

say "DONE"
printf 'total %d min\n' $(( ($(date +%s) - started) / 60 ))
echo
echo "outputs:"
du -sh "$T"/work "$T"/figures "$T"/esm2_output 2>/dev/null
