#!/usr/bin/env python3
"""STEP 3 — emit an auditable provenance table for a track.

    ./.venv/bin/python 3_Write_Provenance.py --track Fig_Ext5B
    ./.venv/bin/python 3_Write_Provenance.py --track Fig_2C --verify

WHY THIS EXISTS
---------------
The proteoform-resolution chain (PIPELINE.md section 5) is the part of this pipeline most
likely to go silently wrong: HPA renumbers proteoform labels between releases and Ensembl
retires protein ids, so the same label can mean different proteins depending on when you look.
A confident-looking result built on the wrong protein is worse than no result.

So every external fetch this project depends on is listed with its exact URL and a fingerprint
of what came back, and can be checked by hand instead of taken on trust.

It writes, into <dir>:

    PROVENANCE.tsv   one row per (target, resource) with the exact URL, HTTP status, and a
                     short fingerprint of what came back
    PROVENANCE.md    the same, grouped per target and readable

Every row is CHECKABLE: paste the URL into a browser or curl and compare the fingerprint.

WHAT IT RECORDS
---------------
Per protein target:
  * HPA subcellular page  — the page PUPS scrapes for (label, ENSP) pairs
  * HPA gene XML          — what our download script reads for antibodies and image URLs
  * Ensembl protein seq   — for the ENSP actually used, with the exact host (archive hosts are
                            spelled out, because retired ENSPs only exist in older releases)
  * the ENSPs HPA lists   — so a mismatch against the one we used is visible immediately

Per image field of view (from ROI_examples.txt, if present):
  * the images.proteinatlas.org base URL, from which every channel file is derived

--verify re-fetches and flags any row whose fingerprint changed since the last run, which
catches silent upstream drift (HPA reassigns proteoform labels; Ensembl retires ENSPs).
"""
import argparse
import csv
import datetime
import hashlib
import json
import pathlib
import re
import sys

import requests

HPA = "https://www.proteinatlas.org"
ENSEMBL_HOSTS = ["https://rest.ensembl.org", "https://feb2023.rest.ensembl.org",
                 "https://jul2022.rest.ensembl.org"]


def fingerprint(text, n=12):
    return hashlib.sha256(text.encode()).hexdigest()[:n]


def get(url, **kw):
    try:
        r = requests.get(url, timeout=kw.pop("timeout", 120), **kw)
        return r.status_code, (r.text if r.ok else "")
    except Exception as e:
        return f"ERR:{type(e).__name__}", ""


def protein_seq(ensp):
    """Try current Ensembl, then archives. Retired ENSPs only exist in older releases."""
    for host in ENSEMBL_HOSTS:
        url = f"{host}/sequence/id/{ensp}"
        status, body = get(url, headers={"Content-Type": "text/plain"}, timeout=60)
        if status == 200 and body.strip():
            return url, status, body.strip()
    return f"{ENSEMBL_HOSTS[0]}/sequence/id/{ensp}", status, ""


def hpa_matching_transcripts(ensg, symbol):
    """Reuse step 2's implementation rather than keeping a second copy of the scraper.

    The module name starts with a digit, so it cannot be a normal import. Importing it runs no
    side effects — 2_Resolve_Source_List.py guards its entry point.
    """
    global _STEP2
    if _STEP2 is None:
        import importlib.util
        path = pathlib.Path(__file__).resolve().parent / "2_Resolve_Source_List.py"
        if not path.exists():
            sys.exit(f"missing {path.name}; step 3's HPA cross-check needs its scraper.")
        spec = importlib.util.spec_from_file_location("resolve_source_list", path)
        _STEP2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_STEP2)
    return _STEP2.hpa_matching_transcripts(ensg, symbol)


_STEP2 = None


def col(row, *names):
    """First of `names` present in `row`, else exit naming what was looked for.

    targets_resolved.tsv is written by step 2, so a rename there used to surface here as a bare
    KeyError from deep in the loop. This makes the mismatch self-describing and lists the columns
    that ARE present, which is what you need to fix it.
    """
    for n in names:
        if row.get(n):
            return row[n]
    sys.exit(f"targets_resolved.tsv has none of {names}.\n"
             f"  columns present: {', '.join(row.keys())}\n"
             f"  It was probably written by a different version of 2_Resolve_Source_List.py — "
             f"re-run that step for this track.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # --track is the name every other step uses; --dir kept as an alias so older
    # command lines keep working.
    ap.add_argument("--track", "--dir", dest="dir", required=True,
                    help="track folder holding targets_resolved.tsv")
    ap.add_argument("--verify", action="store_true",
                    help="compare against the existing PROVENANCE.tsv and report drift")
    args = ap.parse_args()

    here = pathlib.Path(args.dir)
    resolved = here / "targets_resolved.tsv"
    if not resolved.exists():
        sys.exit(f"{resolved} not found — run 2_Resolve_Source_List.py --track <TRACK> first")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    rows = []
    targets = list(csv.DictReader(open(resolved), delimiter="\t"))
    print(f"STEP 3 — provenance for {len(targets)} target(s) in {here}\n")

    for t in targets:
        # Column names must track 2_Resolve_Source_List.py's writer. `transcript`/`protein_id`
        # were the pre-2026-09-03 names, from before the HPA /antibody route replaced the
        # Ensembl-name shortcut; both are accepted so an older tsv still reads.
        label = col(t, "hpa_label", "transcript")
        ensg = col(t, "ensembl_gene_id")
        ensp = col(t, "ensp", "protein_id")
        # TWO symbol columns, and they are not interchangeable. `folder_symbol` is uppercased
        # with punctuation stripped, for image folder names; `symbol_used` is HPA's actual
        # spelling. The "Matching transcripts" regex is CASE-SENSITIVE, so scraping with
        # folder_symbol finds nothing for any mixed-case gene: C9orf72 has 10 labels,
        # "C9ORF72" has 0, which reads as "ENSP not listed by HPA".
        sym = col(t, "symbol_used", "folder_symbol")
        print(f"  {label}")

        page = f"{HPA}/{ensg}-{sym}/subcellular"
        st, body = get(page)
        page_ensps = sorted(set(re.findall(r"ENSP\d+", body))) if body else []
        rows.append({"target": label, "resource": "hpa_subcellular_page", "url": page,
                     "http": st, "detail": f"ENSPs listed: {','.join(page_ensps) or 'none'}",
                     "fingerprint": fingerprint(body) if body else "-", "fetched": stamp})

        xml = f"{HPA}/{ensg}.xml"
        st, body = get(xml)
        xml_ensps = sorted(set(re.findall(r"ENSP\d+", body))) if body else []
        rows.append({"target": label, "resource": "hpa_gene_xml", "url": xml, "http": st,
                     "detail": f"ENSPs listed: {','.join(xml_ensps) or 'none'}",
                     "fingerprint": fingerprint(body) if body else "-", "fetched": stamp})

        # THE AUTHORITATIVE CHECK: the /antibody page's "Matching transcripts" block, across
        # HPA versions, which is the only place a proteoform label -> ENSP mapping exists and is
        # the exact route step 2 (and PUPS) uses. This replaces an earlier check against the
        # gene XML and subcellular page, which produced false alarms for two compounding
        # reasons: those ENSPs are AlphaFold <chain> entries rather than proteoforms, and only
        # the CURRENT release was consulted while HPA renumbers labels between versions.
        # ALMS1-204/ENSP00000478155 failed that old check while being listed in every release
        # from v18 on -- as ALMS1-002, then -204, and -203 today.
        matches = hpa_matching_transcripts(ensg, sym)
        where = sorted({(lab, v) for lab, pairs in matches.items()
                        for v, e in pairs if e == ensp})
        agree = bool(where)
        labels_now = sorted({lab for lab, _ in where})
        drift = agree and label not in labels_now
        ab_url = f"https://<version>.proteinatlas.org/{ensg}-{sym}/antibody"
        rows.append({"target": label, "resource": "hpa_matching_transcripts", "url": ab_url,
                     "http": "-" if not matches else 200,
                     "detail": (f"{len(matches)} label(s) across versions; {ensp} listed as "
                                + (", ".join(f"{lab}@{v}" for lab, v in where) or "NOWHERE")),
                     "fingerprint": fingerprint(";".join(f"{l}@{v}" for l, v in where)) if where
                                    else "-",
                     "fetched": stamp})

        url, st, seq = protein_seq(ensp)
        rows.append({"target": label, "resource": "ensembl_protein_used", "url": url,
                     "http": st,
                     "detail": (f"{len(seq)} aa; first40={seq[:40]}; "
                                f"ENSP={ensp}; listed_by_HPA={'YES' if agree else 'NO'}"
                                + (f"; LABEL DRIFT: HPA now calls it {'/'.join(labels_now)}"
                                   if drift else "")),
                     "fingerprint": fingerprint(seq) if seq else "-", "fetched": stamp})
        flag = ""
        if not agree:
            flag = "   <-- ENSP NOT LISTED BY HPA, see PROVENANCE.md"
        elif drift:
            flag = f"   [HPA now labels it {'/'.join(labels_now)} — same ENSP, label renumbered]"
        print(f"      {ensp}  {len(seq)} aa  via {url.split('//')[1].split('.')[0]}{flag}")

    # image base URLs, so every channel file we read is traceable
    roi = here / "ROI_examples.txt"
    if roi.exists():
        for r in csv.DictReader(open(roi, newline="")):
            if (r.get("Example") or "").strip().upper() != "T":
                continue
            rows.append({"target": f"{r['Gene']} / {r['Antibody']} / {r['CellLine']}",
                         "resource": "hpa_image_base",
                         "url": f"https://images.proteinatlas.org/<id>/{r['ImagePrefix']}",
                         "http": "-", "fetched": stamp,
                         "detail": "channel files are <base>_{blue,red,green,yellow}.{jpg,tif}"
                                   " and <base>_blue_red_green[_yellow].jpg",
                         "fingerprint": "-"})

    cols = ["target", "resource", "url", "http", "detail", "fingerprint", "fetched"]
    out_tsv = here / "PROVENANCE.tsv"

    if args.verify and out_tsv.exists():
        old = {(r["target"], r["resource"]): r for r in
               csv.DictReader(open(out_tsv), delimiter="\t")}
        drift = [r for r in rows
                 if (r["target"], r["resource"]) in old
                 and r["fingerprint"] not in ("-",)
                 and old[(r["target"], r["resource"])]["fingerprint"] != r["fingerprint"]]
        print(f"\n=== --verify: {len(drift)} row(s) changed since the last run ===")
        for r in drift:
            print(f"  {r['target']} / {r['resource']}\n      {r['url']}")
        if not drift:
            print("  none — every external resource returned the same content.")

    with open(out_tsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    md = [f"# Provenance — {here.name}", "",
          f"Generated by `3_Write_Provenance.py` on {stamp}. **Every URL below is checkable"
          " by hand.** Re-run with `--verify` to detect upstream drift.", "",
          "`listed_by_HPA=NO` means the ENSP we used is not among those HPA publishes for that"
          " gene — a version-drift mismatch that must be resolved before the target is used.",
          ""]
    for t in targets:
        # Same column contract as the loop above. There is no transcript-ID column: step 2
        # resolves an HPA proteoform LABEL straight to an ENSP, never touching ENST ids, which
        # is the whole point of the /antibody route. The old `transcript_id` field never existed.
        label = col(t, "hpa_label", "transcript")
        md += [f"## {label}  ({t.get('cell_line_paper', '?')})", "",
               f"- gene `{col(t, 'ensembl_gene_id')}` symbol `{col(t, 'symbol_used')}`"
               f" proteoform `{label}` protein `{col(t, 'ensp', 'protein_id')}`",
               f"- chosen by: {t.get('ensp_source', '?')}",
               f"- holdout status: {t.get('holdout_status', '?')}", ""]
        for r in [r for r in rows if r["target"] == label]:
            md += [f"- **{r['resource']}** → <{r['url']}> (HTTP {r['http']})",
                   f"  - {r['detail']}", f"  - sha256[:12] `{r['fingerprint']}`"]
        md += [""]
    img = [r for r in rows if r["resource"] == "hpa_image_base"]
    if img:
        md += ["## Image fields of view (Example=T)", ""]
        md += [f"- `{r['target']}` → `{r['url']}`" for r in img] + [""]
    (here / "PROVENANCE.md").write_text("\n".join(md))

    mism = [r for r in rows if "listed_by_HPA=NO" in r["detail"]]
    print(f"\n  wrote {out_tsv} ({len(rows)} rows) and {here/'PROVENANCE.md'}")
    if mism:
        print(f"\n  {len(mism)} TARGET(S) WITH AN ENSP HPA DOES NOT LIST:")
        for r in mism:
            print(f"    {r['target']}  — {r['detail']}")
        print("  Resolve these before using them; see PIPELINE.md section 5.")
        return 1
    print("  all ENSPs used are listed by HPA.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
