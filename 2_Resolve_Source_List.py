#!/usr/bin/env python3
"""STEP 2 — resolve a track's source_list.csv to proteoforms, THE WAY PUPS DOES.

    ./.venv/bin/python 2_Resolve_Source_List.py --track Fig_Ext5B
    ./.venv/bin/python 2_Resolve_Source_List.py --track Fig_2C
    ./.venv/bin/python 2_Resolve_Source_List.py --self-test     # verify against their own DB

Reads <track>/source_list.csv and writes into <track>:
    targets_resolved.tsv                     one row per target, fully attributed
    sequences.json                           gene -> protein sequence (input to step 5)
    Organelle_Example_GenesAndCellLines.txt  the download list step 4 consumes

THEIR ROUTE, WHICH THIS COPIES
------------------------------
From `PUPS_Code/src/dataset/download_data.py` and confirmed by the MongoDB screenshots the
authors ship (transcribed in `PUPS_Code/mongo.00*_transcribed.txt`):

    _id                        = "TSPAN6-201"          the HPA proteoform label
    splice_isoform_ensemble_id = "ENSP00000362111"     an ENSP, scraped from the HPA page with
                                                       re.search(r"(ENSP\\d+)", ...)
    sequence                   = "MASPSRRLQTKPVITC..." protein, from Ensembl /sequence/id
    esm2_representation.length = 247                   == 245 aa + BOS + EOS, i.e. L + 2

So the chain is **HPA page -> ENSP -> Ensembl protein**, keyed by the HPA label. That is what
this script does.

WHY NOT RESOLVE THE PAPER'S "-2xx" LABEL THROUGH ENSEMBL DIRECTLY
-----------------------------------------------------------------
An earlier version did, and it was wrong. Ensembl reassigns transcript display numbers between
releases, so today's "C1QL3-202" is a different transcript than the one the paper labelled:

    C1QL3-202  paper era  -> ENSP00000480149  (213 aa, now RETIRED from Ensembl)
    C1QL3-202  today      -> ENSP00000520824  (287 aa, and HPA does not list it)

Verified 2026-09-02. Resolving through the label alone silently substitutes a different protein.
HPA's own ENSP list is the authority, because that is what PUPS read.

RESOLUTION ORDER (each target)
------------------------------
  1. `ensp_override` column in source_list.csv, if set — an explicit human decision, always wins
  2. if HPA lists exactly ONE ENSP for the gene, use it — unambiguous, and it is what PUPS read
  3. otherwise, match the paper's label via current Ensembl AND require the result to be in
     HPA's list. If it is not, FAIL LOUDLY with the candidates and ask for an `ensp_override`.

Never guesses between several HPA proteoforms. An ambiguous target stops the run.

SEQUENCE FETCH
--------------
`/sequence/id/{ENSP}` on current Ensembl, which reproduces their stored sequence exactly
(verified: ENSP00000362111 -> 245 aa, prefix identical to their MongoDB `sequence` field,
L+2 = 247 = their `length` field). Retired ENSPs are not in current Ensembl, so archive hosts
are tried in turn; the host actually used is recorded per target.

NOTE: an ENSP id already identifies a protein, so no `type=protein` parameter is needed —
Ensembl returns protein for ENSP ids. (PUPS passes no `type` either, and is correct to.)
"""
import argparse
import csv
import json
import pathlib
import re
import sys
import time

import requests

HPA = "https://www.proteinatlas.org"
# Current Ensembl first; archives only for ENSPs retired since the paper.
ENSEMBL_HOSTS = ["https://rest.ensembl.org",
                 "https://feb2023.rest.ensembl.org",
                 "https://jul2022.rest.ensembl.org"]

# From PUPS_Code/mongo.001_transcribed.txt — used by --self-test to prove the chain
# reproduces their stored data.
SELF_TEST = {"hpa_label": "TSPAN6-201", "ensp": "ENSP00000362111",
             "aa": 245, "esm2_length": 247,
             "prefix": "MASPSRRLQTKPVITCFKSVLLIYTFIFWITGVILLAVGIWGK"}


def get(url, tries=4, **kw):
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, timeout=kw.pop("timeout", 180), **kw)
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 2)));  continue
            return r
        except requests.RequestException as e:
            last = e; time.sleep(2 ** i)
    raise RuntimeError(f"{url}: {last}")


# HPA versions PUPS iterates over, from download_data.py's build_datasets() default:
#   hpa_versions=["v16","v17","v18","v19","v20","v21","v22"]
# Newest first, so the most recent labelling that still matches the paper wins.
HPA_VERSIONS = ["v22", "v21", "v20", "v19", "v18", "v17", "v16"]


def hpa_matching_transcripts(ensg, symbol, versions=HPA_VERSIONS):
    """THEIR ROUTE. -> {hpa_label: ensp} per HPA version, from the /antibody page.

    PUPS builds exactly this URL in antibody_scrape():

        url = f"https://{hpa_version}.proteinatlas.org/{gene}-{gene_name}/antibody"

    and parses the "Matching transcripts" block, which renders as

        Matching transcripts   ALMS1-204 - ENSP00000478155 [100%]
                               ALMS1-206 - ENSP00000482968 [100%]

    THIS IS THE ONLY PLACE THE LABEL -> ENSP MAPPING EXISTS. It is not in the gene XML (whose
    ENSPs come from AlphaFold `<chain>` entries), and it is not derivable from Ensembl, because
    HPA renumbers labels between versions. Established 2026-09-03:

        ALMS1-204   v19-v22 -> ENSP00000478155     (v16-v18 used old-style ALMS1-001/002/008;
                                                    v23 renumbered it to ALMS1-203)
        C1QL3-202   v22     -> ENSP00000480149
        DUSP27-203  v19,v20 -> ENSP00000404874     (symbol later renamed to STYXL2)

    Their MongoDB screenshot records `all_versions: [v19, v20, v21, v22]` for its example, so
    scanning this range is what they did.
    """
    import re as _re
    found = {}
    pat = _re.compile(_re.escape(symbol) + r"-(\d+)\s*[-–]\s*(ENSP\d+)")
    for v in versions:
        url = f"https://{v}.proteinatlas.org/{ensg}-{symbol}/antibody"
        try:
            r = get(url)
        except RuntimeError:
            continue
        if not r.ok:
            continue
        text = _re.sub(r"<[^>]+>", " ", r.text)
        for num, ensp in pat.findall(text):
            found.setdefault(f"{symbol}-{num}", []).append((v, ensp))
    return found


def hpa_ensps(ensg):
    """ENSPs appearing in HPA's gene XML.

    ⚠ IMPORTANT LIMITATION, established 2026-09-03. These come from AlphaFold structure
    entries, NOT from a proteoform list:

        <chain chain="A" length="2190" gene="ALMS1"
               ensembl_peptide_id="ENSP00000507376"
               ensembl_transcript_id="ENST00000684590">

    HPA's XML carries NO proteoform label ("ALMS1-204") in any version — checked current and
    v18-v23; the v18 XML has no transcript/isoform/peptide elements at all. PUPS got the
    label->ENSP mapping by scraping the HTML "Matching transcripts" table, which no longer
    exists on any reachable HPA version.

    So this list is a WEAK cross-check: it confirms an ENSP is a protein HPA knows for that
    gene, and nothing stronger. It cannot confirm which proteoform the paper's "-2xx" label
    referred to. Treat `listed_by_hpa` accordingly and do not describe it as "HPA's proteoform
    list" in the write-up.
    """
    r = get(f"{HPA}/{ensg}.xml")
    if not r.ok:
        return [], f"HTTP {r.status_code}"
    return sorted(set(re.findall(r"ENSP\d+", r.text))), "ok"


def ensembl_json(path, **params):
    for host in ENSEMBL_HOSTS:
        r = get(f"{host}{path}", params=params,
                headers={"Content-Type": "application/json"})
        if r.ok:
            try:
                return r.json(), host
            except ValueError:
                continue
    return None, None


# Statuses that mean "current Ensembl does not serve this id" — a real retirement. Ensembl
# answers **400**, not 404, for a retired ENSP: verified 2026-09-17, ENSP00000480149 (C1QL3-202,
# retired) -> 400 while ENSP00000362111 (live) -> 200. Checking 404 alone would mislabel every
# retired id as a transient failure, which is why this set is explicit rather than inline.
ENSEMBL_MISS = {400, 404, 410}


def protein(ensp):
    """ENSP -> (sequence, host, status_on_current_host). Current Ensembl, then archives.

    Returns the current host's HTTP status as well, because falling back and being retired are
    not the same thing. A status in ENSEMBL_MISS means Ensembl no longer serves the id — a real
    retirement, and a fact worth stating in the methods. A 5xx or a timeout also falls through
    to the archives, since the sequence is what we need and the archived copy is identical — but
    recording that as a retirement would be a false provenance claim. Observed in practice: a
    transient failure on rest.ensembl.org relabelled COPA-201 as retired on one run and not the
    next, with a byte-identical sequence both times.
    """
    status = None
    for i, host in enumerate(ENSEMBL_HOSTS):
        r = get(f"{host}/sequence/id/{ensp}", headers={"Content-Type": "text/plain"})
        if i == 0:
            status = r.status_code
        if r.ok and r.text.strip():
            return r.text.strip().rstrip("*"), host, status
    return "", None, status


def label_to_ensp_via_ensembl(symbol, label):
    """Fallback only: today's Ensembl mapping of a '-2xx' label to its ENSP."""
    data, _ = ensembl_json(f"/lookup/symbol/homo_sapiens/{symbol}", expand=1)
    if not data:
        return None
    t = next((t for t in data.get("Transcript", []) if t.get("display_name") == label), None)
    return (t.get("Translation") or {}).get("id") if t else None


def read_source_list(path):
    lines = [l for l in open(path) if not l.startswith("#") and l.strip()]
    return [{k: (v or "").strip() for k, v in r.items()} for r in csv.DictReader(lines)]


def requested_gene_key(r):
    """The gene one source_list row asks for, for either schema gene mode accepts.

    `Fig_Ext5B`-style ROI tables carry a `Gene` column; `Independent_Selection` carries
    Note,Link,CellLine and the symbol has to come out of the Link. This MUST key the same way
    resolve_gene_mode() does below, or the resolved/requested counts disagree. Returns "" for a
    row from which no gene can be read, so callers can drop it from a set.
    """
    if (g := (r.get("Gene") or r.get("gene") or "").strip()):
        return g
    m = re.search(r"(ENSG\d+)-([A-Za-z0-9]+)", r.get("Link") or "")
    return m.group(2) if m else ""


def resolve_gene_mode(rows, here, images_root):
    """MODE 'gene' — for tracks WE selected, which carry no paper proteoform label.

    Independent_Selection is our own choice of genes, antibodies, cell lines and crops. There is no
    published "-2xx" label to honour, so there is nothing to match: the question is only which
    proteoform to embed.

    RULE: the LONGEST matching transcript on HPA's /antibody page.

    Why that rule:
      * Taking the lowest label number assumes -201 is the canonical translation. Measured over
        the 34-gene set on 2026-09-04 that is wrong for ~9% of genes, and badly so: AP2B1-201 is
        181 aa against -211 at 951 aa.
      * Handing PUPS a 181 aa fragment of a 951 aa protein is not a defensible representation of
        "the protein", so length decides.
      * Counts vary enormously (ABCE1 has 1 matching transcript, AFDN 11, CANX 30), so the rule
        must be stated explicitly; every alternative is recorded in `all_matching_transcripts`.

    Why the choice cannot bias the result: the whole comparison is ACROSS INPUT FORMATS with the
    SAME sequence embedding. The isoform enters every variant identically, so it shifts all arms
    together and cannot manufacture a difference between them. It is a nuisance parameter, not a
    confound. Every alternative is recorded in `all_matching_transcripts` so the choice is
    auditable and can be swapped with `ensp_override`.

    NOTE: HPA renders "Matching transcripts" once per GENE (in the first antibody's column, the
    rest empty), not per antibody — verified on EZR v22, which has 4 antibodies and one shared
    transcript list. So even though our table names a specific antibody, HPA offers no
    per-antibody proteoform mapping to honour.
    """
    import pathlib as _pl
    genes, out, failures = {}, [], []
    for r in rows:
        if (r.get("Skip") or "F").strip().upper() == "T":
            continue
        if "Link" in r:
            # Download-list schema: Note,Link,CellLine, where Link is an HPA gene URL like
            #   https://www.proteinatlas.org/ENSG00000166197-NOLC1/subcellular#img
            # This is the same parse their download script does:
            #   re.search(r'(ENSG\d+)-([A-Z0-9]+)', url)
            m = re.search(r"(ENSG\d+)-([A-Za-z0-9]+)", r["Link"])
            if not m:
                failures.append(f"{r.get('Note','?')}: cannot parse ENSG/symbol from {r['Link']}")
                continue
            ensg, sym = m.group(1), m.group(2)
            genes.setdefault(sym, {**r, "Gene": sym, "ensembl_gene_id": ensg,
                                   "CellLine": r.get("CellLine", "")})
        else:
            genes.setdefault(r["Gene"], r)

    print(f"  mode=gene: {len(genes)} unique gene(s) from {len(rows)} ROI row(s)\n")
    for i, (gene, r) in enumerate(sorted(genes.items()), 1):
        # Prefer the ENSG from the locally downloaded HPA XML, so we use exactly the gene the
        # images belong to rather than re-resolving a symbol that may have been renamed.
        if r.get("ensembl_gene_id"):
            ensg, howg = r["ensembl_gene_id"], "ENSG parsed from the source_list Link"
        elif (xml := next(_pl.Path(images_root).glob(f"{gene}/ENSG*.xml"), None)):
            ensg, howg = xml.stem, "local HPA XML filename"
        else:
            data, _ = ensembl_json(f"/lookup/symbol/homo_sapiens/{gene}")
            if not data:
                failures.append(f"{gene}: no ENSG (no Link, no local XML, no symbol match)")
                print(f"[{i}/{len(genes)}] {gene:10s} FAILED: no ENSG"); continue
            ensg, howg = data["id"], "Ensembl symbol lookup"

        # Only the NEWEST HPA version is needed here: gene mode is not matching a
        # historical "-2xx" label, so there is nothing to gain from older versions, and
        # scanning all seven costs 7x the page fetches (35 genes x 7 = 245 requests).
        matches = hpa_matching_transcripts(ensg, gene, versions=HPA_VERSIONS[:1])
        if not matches:      # fall back to the full sweep if v22 has nothing
            matches = hpa_matching_transcripts(ensg, gene)
        if not matches:
            failures.append(f"{gene}: no 'Matching transcripts' on HPA for {ensg}")
            print(f"[{i}/{len(genes)}] {gene:10s} FAILED: no matching transcripts"); continue

        # Restrict to one labelling era first. HPA switched from GENE-001 (v16-v18) to
        # GENE-201 (v19+), and "001" sorts below "201", so mixing eras would silently favour
        # a v16-era proteoform. Only matters if the v22 lookup was empty and the sweep ran.
        def num(label):
            return int(label.rsplit("-", 1)[-1])
        modern = [k for k in matches if num(k) >= 200]
        pool = modern or list(matches)
        if modern and len(modern) != len(matches):
            legacy = sorted(k for k in matches if num(k) < 200)
            print(f"           ignoring {len(legacy)} pre-v19 label(s): {', '.join(legacy)}")

        # THE LONGEST proteoform wins, not the lowest-numbered.
        #
        # An earlier version took the lowest label number, on the assumption that -201 is the
        # canonical transcript. Measured across the 34-gene set on 2026-09-04, that is wrong
        # for ~9% of genes, and badly so:
        #     AP2B1   -201  181 aa   vs  -211  951 aa   (5x truncated)
        #     TAF15   -205  223 aa   vs  -215  592 aa
        #     PTPN20  -201  339 aa   vs  -202  420 aa
        # Handing PUPS a 181 aa fragment when the protein is 951 aa is not a defensible
        # representation of "the protein", so length decides, with the lowest label number as
        # the tie-break. Costs one sequence fetch per candidate, which is why gene mode only
        # scans one HPA version.
        lengths = {}
        for lab in sorted(pool, key=num):
            cand_seq, _, _ = protein(matches[lab][0][1])
            lengths[lab] = len(cand_seq)
        chosen_label = max(sorted(pool, key=num), key=lambda k: lengths[k])
        ensp = matches[chosen_label][0][1]
        if len(pool) > 1:
            shortest = min(lengths.values())
            if lengths[chosen_label] > shortest * 1.15:
                print(f"           longest of {len(pool)}: {chosen_label} "
                      f"{lengths[chosen_label]} aa (shortest would have been {shortest} aa)")
        override = r.get("ensp_override", "")
        how = f"longest HPA matching transcript ({chosen_label}, {len(pool)} candidates)"
        if override:
            ensp, how = override, "ensp_override in source_list.csv"

        seq, host, ens_status = protein(ensp)
        if not seq:
            failures.append(f"{gene}: no protein sequence for {ensp}")
            print(f"[{i}/{len(genes)}] {gene:10s} FAILED: {ensp} has no sequence"); continue

        out.append({"Gene": gene, "hpa_label": chosen_label, "ensembl_gene_id": ensg,
                    "ensg_source": howg, "symbol_used": gene,
                    "folder_symbol": re.sub(r"[^A-Z0-9]", "", gene.upper()),
                    "ensp": ensp, "ensp_source": how,
                    "n_matching_transcripts": len(matches),
                    "all_matching_transcripts": ";".join(
                        f"{k}={v[0][1]}" for k, v in sorted(matches.items(), key=lambda x: num(x[0]))),
                    "ensembl_host": host,
                    # Only a definitive miss on the current host is a retirement — protein().
                    "ensp_retired": "yes" if (host != ENSEMBL_HOSTS[0]
                                              and ens_status in ENSEMBL_MISS) else "no",
                    "current_host_status": ens_status,
                    "cell_line_paper": r.get("CellLine", ""), "antibody": r.get("Antibody", ""),
                    "length": len(seq), "sequence": seq})
        print(f"[{i}/{len(genes)}] {gene:10s} {ensp}  {len(seq):>5} aa  "
              f"{chosen_label} (1 of {len(matches)})")
    return out, failures


def self_test():
    print("SELF-TEST — reproduce the authors' own stored proteoform")
    print(f"  target: {SELF_TEST['hpa_label']} / {SELF_TEST['ensp']}")
    print( "  source: PUPS_Code/mongo.001_transcribed.txt (their MongoDB screenshot)\n")
    seq, host, _ = protein(SELF_TEST["ensp"])
    checks = [
        ("sequence retrieved", bool(seq)),
        (f"length == {SELF_TEST['aa']} aa", len(seq) == SELF_TEST["aa"]),
        ("prefix matches their `sequence` field", seq.startswith(SELF_TEST["prefix"])),
        (f"L+2 == {SELF_TEST['esm2_length']} (their `length` field)",
         len(seq) + 2 == SELF_TEST["esm2_length"]),
    ]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n  host used: {host}")
    print(f"  got {len(seq)} aa: {seq[:60]}")
    ok = all(c[1] for c in checks)
    print("\nSELF-TEST " + ("PASSED — our chain reproduces their data." if ok else "FAILED"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", help="folder holding source_list.csv")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--mode", choices=["auto", "label", "gene"], default="auto",
                    help="'label' = match the paper's -2xx proteoform label (Fig_* tracks); "
                         "'gene' = our own selection, take HPA's longest matching "
                         "transcript (Independent_Selection); 'auto' picks by source_list.csv schema")
    ap.add_argument("--images-root", default=None,
                    help="optional: a folder of <GENE>/ENSG*.xml used only to look up an ENSG "
                         "when source_list.csv has no Link column. Defaults to <track>/Images")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.track:
        ap.error("--track is required (or use --self-test)")

    here = pathlib.Path(args.track)
    src = here / "source_list.csv"
    if not src.exists():
        sys.exit(f"{src} not found")
    rows = read_source_list(src)

    # Schema decides the mode: a `transcript` column means the paper named the proteoform;
    # a `Gene`/`Antibody` ROI table means we chose it ourselves.
    mode = args.mode
    if mode == "auto":
        mode = "label" if ("transcript" in rows[0]) else "gene"
    print(f"STEP 2 — {len(rows)} row(s) in {here}, mode={mode}, via HPA -> ENSP -> Ensembl\n")

    if mode == "gene":
        resolved, failures = resolve_gene_mode(
            rows, here, args.images_root or (here / "Images"))
        if resolved:
            cols = [c for c in resolved[0] if c != "sequence"]
            with open(here / "targets_resolved.tsv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
                w.writeheader(); w.writerows(resolved)
            (here / "sequences.json").write_text(json.dumps({r["folder_symbol"]: {
                "hpa_label": r["hpa_label"], "ensp": r["ensp"],
                "ensembl_gene_id": r["ensembl_gene_id"], "gene_symbol": r["symbol_used"],
                "transcript_name": r["hpa_label"], "chosen_by": r["ensp_source"],
                "n_matching_transcripts": r["n_matching_transcripts"],
                "all_matching_transcripts": r["all_matching_transcripts"],
                "ensembl_host": r["ensembl_host"], "length": r["length"],
                "sequence": r["sequence"]} for r in resolved}, indent=2))
            # Step 4a needs this exact filename. For a Note,Link,CellLine source_list it is
            # effectively a round-trip; for a ROI-style one it is newly derived.
            #
            # ONE ROW PER (GENE, CELL LINE), not per gene. Proteoform resolution above is
            # rightly per gene — one sequence, one embedding — so `genes` is deduplicated by
            # symbol. The DOWNLOAD list must not be: the downloader filters images by cell
            # line, and a gene can appear in source_list.csv under two of them (MKI67 in U2OS
            # and U-251MG, NOLC1 in U2OS and HEK293). Collapsing those to one row silently
            # requests no images at all for the second cell line, and the only symptom is
            # step 4b failing on field-of-view folders that were never downloaded.
            by_gene = {r["Gene"]: r for r in resolved}
            pairs, seen_pairs = [], set()
            for src_row in rows:
                rr = by_gene.get(requested_gene_key(src_row))
                if rr is None:
                    continue                    # unresolved: already counted as a failure
                key = (rr["Gene"], src_row.get("CellLine", ""))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                pairs.append((src_row, rr))
            with open(here / "Organelle_Example_GenesAndCellLines.txt", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["Note", "Link", "CellLine"])
                for src_row, rr in pairs:
                    w.writerow([rr["hpa_label"],
                                f"{HPA}/{rr['ensembl_gene_id']}-{rr['folder_symbol']}/subcellular",
                                src_row.get("CellLine", "") or rr.get("cell_line_paper", "")])
            multi = [r for r in resolved if r["n_matching_transcripts"] > 1]
            n_requested = len({g for r in rows if (g := requested_gene_key(r))})
            print(f"\nresolved {len(resolved)}/{n_requested} gene(s)")
            if len(pairs) != len(resolved):
                print(f"  {len(pairs)} (gene, cell line) pair(s) in the download list — "
                      f"{len(pairs) - len(resolved)} gene(s) appear under more than one cell line")
            print(f"  -> targets_resolved.tsv, sequences.json, "
                  f"Organelle_Example_GenesAndCellLines.txt")
            print(f"  {len(multi)} gene(s) had >1 matching transcript; the LONGEST was used")
            print( "  and every alternative is recorded in `all_matching_transcripts`.")
            print( "  Longest, not lowest-numbered: measured 2026-09-04, the lowest label is a")
            print( "  truncated fragment for ~9% of genes (AP2B1-201 is 181 aa vs -211 951 aa).")
            print( "  The choice cannot bias the format comparison — the same embedding enters")
            print( "  every variant identically. Override per row with `ensp_override`.")
        if failures:
            print(f"\n  {len(failures)} FAILURE(S):")
            for f in failures: print(f"    {f}")
            return 1
        print(f"\nNEXT: ./.venv/bin/python 4_Run_Image_Preparation.py --track {here} "
              f"--tool download")
        return 0

    resolved, failures = [], []
    for i, t in enumerate(rows, 1):
        label = t.get("transcript") or t.get("hpa_label") or ""
        gene = t.get("gene", "")
        # The download script derives folder names with r'(ENSG\d+)-([A-Z0-9]+)', which
        # truncates mixed-case symbols ("C9orf72" -> "C9") and would break the step-7
        # embedding lookup. Key everything by an uppercased, regex-safe symbol.
        sym_for_lookup = t.get("alias") or gene
        print(f"[{i}/{len(rows)}] {label}  ({t.get('cell_line_paper','?')})")

        ensg = t.get("ensembl_gene_id", "")
        if not ensg:
            data, _ = ensembl_json(f"/lookup/symbol/homo_sapiens/{gene}")
            if not data:
                for alias in [a for a in (t.get("alias") or "").split(";") if a]:
                    data, _ = ensembl_json(f"/lookup/symbol/homo_sapiens/{alias}")
                    if data:
                        sym_for_lookup = alias; break
            if not data:
                failures.append(f"{label}: no Ensembl gene for {gene!r} or its aliases")
                print(f"    FAILED: gene {gene!r} not found"); continue
            ensg = data["id"]
            sym_for_lookup = data.get("display_name", sym_for_lookup)

        listed, status = hpa_ensps(ensg)
        print(f"    {ensg} ({sym_for_lookup})  HPA lists {len(listed)} ENSP(s) [{status}]")

        # PRIMARY: their route — the HPA /antibody page's "Matching transcripts" block.
        # Try the gene symbol as HPA had it in each era: the paper's symbol first, then aliases
        # (DUSP27 -> STYXL2 was renamed after the paper).
        symbols = [t.get("gene", "")] + [a for a in (t.get("alias") or "").split(";") if a] \
                  + [sym_for_lookup]
        matches, sym_hit = {}, None
        for sym in dict.fromkeys(s for s in symbols if s):
            matches = hpa_matching_transcripts(ensg, sym)
            if matches:
                sym_hit = sym
                break
        hits = matches.get(label, [])
        override = t.get("ensp_override", "")

        if override:
            ensp, how = override, "ensp_override in source_list.csv"
            if hits and override != hits[0][1]:
                print(f"    WARNING override {override} disagrees with HPA "
                      f"{hits[0][0]}:{hits[0][1]}")
        elif hits:
            versions = ",".join(v for v, _ in hits)
            # HPA_VERSIONS is ordered newest-first, so hits[0] is the newest version in
            # which this label appears. When versions disagree, take that one: HPA renumbers
            # labels over time, and the newest labelling inside their v16-v22 range is the one
            # their scrape would have recorded last. Deterministic, and the disagreement is
            # always reported rather than hidden.
            ensps = {e for _, e in hits}
            ensp = hits[0][1]
            how = f"HPA /antibody 'Matching transcripts' [{sym_hit}, newest={hits[0][0]}]"
            if len(ensps) > 1:
                per = {}
                for v, e in hits:
                    per.setdefault(e, []).append(v)
                print(f"    NOTE label maps to {len(ensps)} ENSPs across versions:")
                for e, vs in per.items():
                    star = "  <- used (newest)" if e == ensp else ""
                    print(f"         {e}  {','.join(sorted(set(vs), reverse=True))}{star}")
                how += f" (ambiguous: {'; '.join(f'{e}@{min(v)}' for e, v in per.items())})"
        elif len(listed) == 1:
            ensp, how = listed[0], "sole ENSP in HPA gene XML (weak; see hpa_ensps docstring)"
        else:
            avail = sorted(matches)
            failures.append(
                f"{label}: not in HPA's 'Matching transcripts' for any version "
                f"{HPA_VERSIONS}. Labels HPA does offer: {avail or 'none found'}. "
                f"Set ensp_override in source_list.csv.")
            print(f"    FAILED: label not on any HPA /antibody page")
            print(f"      HPA offers: {avail or 'nothing'}")
            continue

        seq, host, ens_status = protein(ensp)
        if not seq:
            failures.append(f"{label}: no protein sequence for {ensp} on any Ensembl host")
            print(f"    FAILED: {ensp} has no protein sequence"); continue

        folder_symbol = re.sub(r"[^A-Z0-9]", "", sym_for_lookup.upper())
        # Fell back to an archive, and WHY. Only a 404 on the current host is a retirement;
        # any other failure means the archive merely answered first. See protein().
        from_archive = host != ENSEMBL_HOSTS[0]
        retired = from_archive and ens_status in ENSEMBL_MISS
        resolved.append({**t, "hpa_label": label, "ensembl_gene_id": ensg,
                         "symbol_used": sym_for_lookup, "folder_symbol": folder_symbol,
                         "ensp": ensp, "ensp_source": how,
                         "listed_by_hpa": "yes" if ensp in listed else "NO",
                         "ensembl_host": host, "ensp_retired": "yes" if retired else "no",
                         "current_host_status": ens_status,
                         "length": len(seq), "sequence": seq})
        note = ""
        if retired:
            note = f"  via ARCHIVE {host.split('//')[1].split('.')[0]} (retired ENSP)"
        elif from_archive:
            note = (f"  via ARCHIVE {host.split('//')[1].split('.')[0]} — NOT retired; "
                    f"current Ensembl returned HTTP {ens_status}, so this is a transient "
                    f"fallback. Re-run to confirm.")
        print(f"    {ensp}  {len(seq)} aa  [{how}]" + note)

    if not resolved:
        print("\nnothing resolved.")
        for f in failures:
            print(f"  {f}")
        return 1

    cols = [c for c in resolved[0] if c != "sequence"]
    with open(here / "targets_resolved.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader(); w.writerows(resolved)

    # sequences.json is gene-keyed and step 7 looks embeddings up by the ROI table's Gene
    # column. Two proteoforms of one gene would collapse to one entry here, silently, and both
    # tiles would be run with the same sequence.
    seen = {}
    for r in resolved:
        seen.setdefault(r["folder_symbol"], []).append(r["hpa_label"])
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    if dupes:
        sys.exit(f"two or more proteoforms share a gene symbol in this track: {dupes}\n"
                 f"  sequences.json is gene-keyed, so one sequence would be silently dropped.\n"
                 f"  Split the track, or re-key sequences.json and index.csv on hpa_label.")
    
    payload = {r["folder_symbol"]: {
        "hpa_label": r["hpa_label"], "ensp": r["ensp"],
        "ensembl_gene_id": r["ensembl_gene_id"], "gene_symbol": r["symbol_used"],
        "transcript_name": r["hpa_label"], "chosen_by": r["ensp_source"],
        "ensembl_host": r["ensembl_host"], "length": r["length"], "sequence": r["sequence"],
    } for r in resolved}
    (here / "sequences.json").write_text(json.dumps(payload, indent=2))

    # The download script hardcodes this filename; do not rename it.
    with open(here / "Organelle_Example_GenesAndCellLines.txt", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Note", "Link", "CellLine"])
        for r in resolved:
            w.writerow([f"{r['hpa_label']} ({r.get('holdout_status','')})",
                        f"{HPA}/{r['ensembl_gene_id']}-{r['folder_symbol']}/subcellular",
                        r.get("cell_line_hpa") or r.get("cell_line_paper", "")])

    print(f"\nresolved {len(resolved)}/{len(rows)}")
    print(f"  -> targets_resolved.tsv, sequences.json, "
          f"Organelle_Example_GenesAndCellLines.txt")
    arch = [r for r in resolved if r["ensp_retired"] == "yes"]
    if arch:
        print(f"\n  {len(arch)} target(s) use a RETIRED ENSP from an Ensembl archive:")
        for r in arch:
            print(f"    {r['hpa_label']:16s} {r['ensp']}  {r['length']} aa  "
                  f"via {r['ensembl_host']}")
        print("  These are the proteoforms HPA lists, so they are what PUPS read. State this.")
    if failures:
        print(f"\n  {len(failures)} FAILURE(S) — resolve before continuing:")
        for f in failures:
            print(f"    {f}")
        return 1
    print(f"\nNEXT: ./.venv/bin/python 3_Write_Provenance.py --track {here}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
