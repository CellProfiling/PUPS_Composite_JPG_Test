#!/usr/bin/env python3
"""STEP 4b — gate: is every field of view named in ROI.txt actually on disk?

    ./.venv/bin/python 4b_Verify_Track_Images.py --track Fig_Ext5B
    ./.venv/bin/python 4b_Verify_Track_Images.py --track Fig_Ext5B --check-only
    ./.venv/bin/python 4b_Verify_Track_Images.py --track Independent_Selection --roi Independent_Selection/ROI_examples_Selected.txt

WHY THIS STEP EXISTS
--------------------
Step 4a (`--tool download`, their `1_HPA_Image_Download_BatchDownload.py`) fetches
`https://www.proteinatlas.org/<ENSG>.xml` — the CURRENT release, with **no fallback to older
releases** — and takes its image URLs from that one document. So when HPA withdraws an
antibody's immunofluorescence data, their downloader does not fail: the antibody simply is not
in the XML it read, so the fields of view are invisible to it. It prints no warning about them
and exits 0. Step 6 then skips those tiles ("source images not present") and also returns 0.
A track can therefore lose tiles in silence, end to end.

Confirmed live example. In HPA v25, N4BP2 antibody `HPA036770` is still listed but carries
**zero** ICC/IF images; v19-v23 likewise; only **v18** still lists its U2OS fields of view
`410_D12_1` and `410_D12_2`. Those two tiles are our ED Fig 5b N4BP2-201 (Holdout 2) row, and
`410_D12_2` is the selected example. Without this gate, Fig_Ext5B quietly becomes 26 tiles of
28 on a fresh machine, missing exactly one of the nine published pairs.

WHAT THIS STEP DOES
-------------------
1. Reads the track's ROI table and lists the 9 files step 6 opens for each field of view.
2. Reports every one that is absent.
3. For the absent ones, walks the ARCHIVED release hosts newest-to-oldest
   (`v24.proteinatlas.org` ... `v15.proteinatlas.org`), which is the fallback their downloader
   lacks, and fetches whatever is still served. Cell-line spelling drifted between releases
   (`U2OS` was `U-2 OS` up to v20, `Rh30` was `RH-30`), so matching is done on a normalised
   form. Downloads go through THEIR `download_file()` / `decompress_gz()`, imported from the
   Image_Preparation clone, so recovered bytes are fetched exactly as step 4a fetches them.
4. **Exits non-zero** listing anything still missing, so the pipeline stops here instead of
   producing a quietly short figure.

ACTIONABLE — the limit of what the archive can give back. For a withdrawn antibody HPA keeps
serving the JPGs (`images.proteinatlas.org/<num>/<prefix>_*.jpg` still returns 200) but NOT the
TIFFs: `<prefix>_*.tif.gz` returns HTTP 400. Since the 16-bit TIFF arm is the point of the
comparison, such a field of view cannot be reconstituted from public sources. If this script
reports missing `.tif` files it will say so explicitly — those must be carried in the
repository or supplied from an internal archive, and PIPELINE.md must say which.
"""
import argparse
import csv
import importlib.util
import pathlib
import re
import sys
import time

import requests
from lxml import etree

ROOT = pathlib.Path(__file__).resolve().parent
PREP = ROOT / "Image_Preparation"
DOWNLOADER = PREP / "1_HPA_Image_Download_BatchDownload.py"

# Archived release hosts, newest first. `www` is the current release and is what step 4a
# already tried, so it is not repeated here. v24/v25 do not exist as separate hosts (the
# newest release is served from `www`), and unknown subdomains do not refuse the connection
# through a proxy — they accept it and hang — so the read timeout is short and a host that
# fails once is not tried again for the rest of the run.
HPA_ARCHIVE = [f"v{n}" for n in range(23, 14, -1)]
XML_TIMEOUT = (10, 30)          # (connect, read); a dead host costs 30 s once, not per tile
_XML_CACHE = {}                 # (host, ensg) -> parsed root or None
_DEAD_HOSTS = set()

# The nine files 6_Build_Input_Variants.py opens per field of view (see its VARIANTS table).
# `_blue_red_green_yellow.jpg` is downloaded by step 4a but read by no later step, so it is
# reported as optional rather than gating.
REQUIRED = ["_blue_red_green.jpg",
            "_blue.jpg", "_red.jpg", "_green.jpg", "_yellow.jpg",
            "_blue.tif", "_red.tif", "_green.tif", "_yellow.tif"]
OPTIONAL = ["_blue_red_green_yellow.jpg"]


def their_downloader():
    """Import the Image_Preparation downloader for `download_file`, `decompress_gz` and
    `clean_image_prefix`. It has an `if __name__ == "__main__"` guard, so importing it runs
    nothing. Using their functions keeps recovered bytes identical to step 4a's."""
    if not DOWNLOADER.exists():
        sys.exit(f"missing: {DOWNLOADER}\n  clone it first — see PIPELINE.md section 3.")
    spec = importlib.util.spec_from_file_location("hpa_dl", DOWNLOADER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def norm_cl(name):
    """Cell-line names are not stable across HPA releases: 'U2OS' was 'U-2 OS' up to v20,
    'U-251MG' was 'U-251 MG', 'Rh30' was 'RH-30'. Compare on alphanumerics only."""
    return re.sub(r"[^A-Z0-9]", "", (name or "").upper())


def read_rois(path):
    """Same parsing as step 6, so the two agree on which tiles are expected. Skip == T rows
    are checked but never gate — they were deliberately excluded."""
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "gene": r["Gene"].strip(),
                "antibody": r["Antibody"].strip(),
                "cell_line": r["CellLine"].strip(),
                "prefix": r["ImagePrefix"].strip(),
                "folder": r["FolderPath"].strip().replace("\\", "/"),
                "annotation": (r.get("Annotation") or "").strip(),
                "example": (r.get("Example") or "").strip().upper() == "T",
                "skip": (r.get("Skip") or "F").strip().upper() == "T",
            })
    return rows


def ensg_for(gene, images_root, resolved):
    """ENSG, preferring the XML step 4a already wrote next to the images so we interrogate
    exactly the gene entry the download used."""
    xml = next(pathlib.Path(images_root).glob(f"{gene}/ENSG*.xml"), None)
    if xml:
        return xml.stem, "local HPA XML written by step 4a"
    if gene in resolved:
        return resolved[gene], "targets_resolved.tsv (step 2)"
    r = requests.get(f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene}",
                     headers={"Content-Type": "application/json"}, timeout=60)
    if r.ok and r.json().get("id", "").startswith("ENSG"):
        return r.json()["id"], "Ensembl symbol lookup"
    return None, "unresolved"


def archived_xml(host, ensg, urls_seen):
    """Parsed release XML, memoised per (host, ENSG). Returns (root, note); root is None on
    any transport or parse problem, so one dead host cannot abort the walk."""
    key = (host, ensg)
    if key in _XML_CACHE:
        return _XML_CACHE[key], "cached"
    if host in _DEAD_HOSTS:
        return None, "host did not answer earlier"
    url = f"https://{host}.proteinatlas.org/{ensg}.xml"
    urls_seen.append(url)
    try:
        r = requests.get(url, timeout=XML_TIMEOUT)
        if r.status_code != 200:
            _XML_CACHE[key] = None
            return None, f"HTTP {r.status_code}"
        root = etree.fromstring(r.content)
    except Exception as e:
        _DEAD_HOSTS.add(host)
        return None, type(e).__name__
    _XML_CACHE[key] = root
    return root, f"{len(r.content) // 1024} KiB"


def archived_image_urls(host, ensg, antibody, cell_line, urls_seen):
    """Image URLs an archived release lists for this antibody + cell line."""
    root, note = archived_xml(host, ensg, urls_seen)
    if root is None:
        return [], note
    out = []
    for ab in root.xpath(".//antibody"):
        if ab.get("id") != antibody:
            continue
        for data in ab.xpath(".//cellExpression[@technology='ICC/IF']/subAssay/data"):
            cl = "".join(data.xpath("./cellLine/text()"))
            if norm_cl(cl) != norm_cl(cell_line):
                continue
            out += [u.strip() for u in
                    data.xpath("./assayImage/image[@imageType='sampleImage']/imageUrl/text()")]
    return out, f"{len(out)} image(s)"


def candidate_bases(image_url, clean_image_prefix):
    """Base URLs (no channel suffix) to try, from an archived `_blue_red_green.jpg` URL.

    Archived releases serve their own host path, e.g.
        http://v18.proteinatlas.org/images/36770/410_D12_1_blue_red_green.jpg
    while the current release uses the version-independent image host,
        http://images.proteinatlas.org/36770/410_D12_1_blue_red_green.jpg
    Both are tried: the second usually still works for withdrawn antibodies, and it is the
    form step 4a would have used, which keeps provenance comparable.
    """
    stem = re.sub(r"_blue_red_green(_yellow)?\.jpg$", "", image_url)
    stem = stem.replace("http://", "https://")
    head, _, name = stem.rpartition("/")
    name = clean_image_prefix(name)
    bases = [f"{head}/{name}"]
    num = head.rstrip("/").rpartition("/")[2]          # numeric image directory, e.g. 36770
    if num.isdigit():
        alt = f"https://images.proteinatlas.org/{num}/{name}"
        if alt not in bases:
            bases.append(alt)
    return name, bases


def probe(url):
    """HTTP status for a URL without pulling the body. HEAD is not answered consistently by
    the image host, so a streamed GET is opened and closed immediately."""
    try:
        r = requests.get(url, stream=True, timeout=(10, 60))
        code = r.status_code
        r.close()
        return code
    except Exception:
        return None


def try_fetch(dl, base, folder, prefix, suffixes, tries=3):
    """Fetch the still-missing suffixes from one base URL.

    Returns (recovered, gone). `gone` are the suffixes the server answered with a definitive
    4xx: for a withdrawn antibody `<prefix>_<channel>.tif.gz` redirects to
    `www.proteinatlas.org/download_file.php` which replies 400, and no amount of retrying or
    trying an older release host changes that. They are recorded as permanently unavailable so
    the walk does not hammer every archived host with the same doomed request.

    Their `download_file` has timeout=30, no retry, and returns a bare False for every kind of
    failure (a logged defect), so the status is probed first and their function is called only
    for a 200; transport errors alone are retried.
    """
    got, gone = [], []
    for suf in list(suffixes):
        dest = folder / f"{prefix}{suf}"
        url = f"{base}{suf}.gz" if suf.endswith(".tif") else f"{base}{suf}"
        code = probe(url)
        if code is not None and 400 <= code < 500:
            gone.append(suf)
            continue
        ok = False
        for t in range(tries):
            if suf.endswith(".tif"):
                gz = folder / f"{prefix}{suf}.gz"
                if dl.download_file(url, gz, skip_if_exists=False):
                    dl.decompress_gz(gz)
                    ok = dest.exists()
                if gz.exists() and not ok:
                    gz.unlink()
            else:
                if dl.download_file(url, dest, skip_if_exists=False):
                    ok = dest.exists() and dest.stat().st_size > 0
                if dest.exists() and dest.stat().st_size == 0:
                    dest.unlink(); ok = False
            if ok:
                break
            time.sleep(1.5 * (t + 1))
        if ok:
            got.append(suf)
    return got, gone


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", required=True)
    ap.add_argument("--roi", default=None, help="default <track>/ROI.txt")
    ap.add_argument("--check-only", action="store_true",
                    help="report gaps, do not contact the archived releases")
    args = ap.parse_args()

    track = ROOT / args.track
    if not track.is_dir():
        sys.exit(f"track folder not found: {track}")
    roi = pathlib.Path(args.roi) if args.roi else track / "ROI.txt"
    if not roi.exists():
        sys.exit(f"ROI table not found: {roi}")
    images = track / "Images"
    if not images.is_dir():
        sys.exit(f"no images yet: {images}\n  run step 4a first "
                 f"(4_Run_Image_Preparation.py --track {args.track} --tool download)")

    rows = read_rois(roi)
    print(f"STEP 4b — image completeness gate")
    print(f"  track : {args.track}")
    print(f"  roi   : {roi}  ({len(rows)} row(s), "
          f"{sum(r['skip'] for r in rows)} flagged Skip=T)")
    print(f"  images: {images}\n")

    # ---- 1. what is missing -------------------------------------------------------------
    gaps = []
    for r in rows:
        folder = images / r["folder"]
        need = [s for s in REQUIRED if not (folder / f"{r['prefix']}{s}").exists()]
        opt = [s for s in OPTIONAL if not (folder / f"{r['prefix']}{s}").exists()]
        if need or opt:
            gaps.append({**r, "folder_path": folder, "need": need, "opt": opt})

    complete = len(rows) - len([g for g in gaps if g["need"]])
    print(f"  {complete}/{len(rows)} field(s) of view have all {len(REQUIRED)} required files")
    if not any(g["need"] for g in gaps):
        for g in gaps:
            print(f"    note: {g['folder']}/{g['prefix']} lacks only optional "
                  f"{', '.join(g['opt'])}")
        print("\n  nothing missing. NEXT: 5_Make_ESM2_Embeddings.py")
        return 0

    for g in gaps:
        if g["need"]:
            tag = "  [SELECTED EXAMPLE]" if g["example"] else ("  [Skip=T]" if g["skip"] else "")
            print(f"    MISSING {g['folder']}/{g['prefix']}: "
                  f"{len(g['need'])} file(s) — {', '.join(g['need'])}{tag}")
            if g["annotation"]:
                print(f"            {g['annotation']}")

    if args.check_only:
        print("\n  --check-only: not contacting the archived releases.")
        return 1

    # ---- 2. try the archived releases ---------------------------------------------------
    dl = their_downloader()
    resolved = {}
    tsv = track / "targets_resolved.tsv"
    if tsv.exists():
        with open(tsv, newline="") as fh:
            for t in csv.DictReader(fh, delimiter="\t"):
                # Key on the same symbol the ROI table and the image folders use. In `label`
                # mode step 2 writes no `Gene` column at all, so indexing it raised KeyError —
                # and only on a track that actually HAS a gap, because this block is skipped
                # when nothing is missing.
                key = (t.get("folder_symbol") or t.get("symbol_used") or t.get("Gene") or "")
                if t.get("ensembl_gene_id") and key:
                    resolved[key.strip()] = t["ensembl_gene_id"]

    print("\n  attempting recovery from archived HPA releases "
          f"({HPA_ARCHIVE[0]} -> {HPA_ARCHIVE[-1]})")
    print("  ACTIONABLE: every URL tried is appended to the provenance log below; step 3"
          "\n    (--verify) will fold it into PROVENANCE.tsv.\n")

    log, unrecovered = [], []
    for g in [g for g in gaps if g["need"]]:
        label = f"{g['folder']}/{g['prefix']}"
        ensg, how = ensg_for(g["gene"], images, resolved)
        print(f"  {label}")
        print(f"    ENSG {ensg or '??'}  ({how})")
        if not ensg:
            unrecovered.append((g, g["need"])); continue

        still, withdrawn = list(g["need"]), set()
        for host in HPA_ARCHIVE:
            urls_seen = []
            found, note = archived_image_urls(host, ensg, g["antibody"],
                                              g["cell_line"], urls_seen)
            log += [(label, u, note) for u in urls_seen]
            if not found:
                print(f"    {host:>4}  {note}")
                continue
            hit = None
            for u in found:
                name, bases = candidate_bases(u, dl.clean_image_prefix)
                if name == g["prefix"]:
                    hit = (name, bases); break
            if not hit:
                print(f"    {host:>4}  lists {len(found)} image(s) but not {g['prefix']}")
                continue
            name, bases = hit
            todo = [s for s in still if s not in withdrawn]
            if not todo:
                print(f"    {host:>4}  lists {g['prefix']}, but every remaining file already"
                      f" answered 4xx on every base URL — not retrying")
                break
            print(f"    {host:>4}  lists {g['prefix']}  ->  trying {len(bases)} base URL(s)")
            # A suffix counts as withdrawn only when EVERY base URL answered 4xx for it, so a
            # file that moved between the release host and the image host is still found.
            gone_all = set(todo)
            for base in bases:
                todo = [s for s in still if s not in withdrawn]
                if not todo:
                    break
                got, gone = try_fetch(dl, base, g["folder_path"], g["prefix"], todo)
                if got:
                    print(f"          recovered from {base}: {', '.join(got)}")
                    still = [s for s in still if s not in got]
                if gone:
                    print(f"          withdrawn (HTTP 4xx) at {base}: {', '.join(gone)}")
                log.append((label, base + "_*",
                            f"recovered {len(got)}, withdrawn {len(gone)}"))
                gone_all &= set(gone)
            withdrawn |= gone_all
            if not still:
                break
        if still:
            unrecovered.append((g, still))
            print(f"    STILL MISSING: {', '.join(still)}")

    # ---- 3. provenance for whatever the archive gave us ---------------------------------
    if log:
        out = track / "PROVENANCE_archive_recovery.tsv"
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["field_of_view", "url", "result"])
            w.writerows(log)
        print(f"\n  {len(log)} URL(s) logged -> {out}")

    # ---- 4. verdict ---------------------------------------------------------------------
    if not unrecovered:
        print("\n  all gaps closed from the archived releases. "
              "NEXT: 5_Make_ESM2_Embeddings.py")
        return 0

    print(f"\n  {len(unrecovered)} field(s) of view CANNOT be rebuilt from public HPA:")
    tif_only = True
    for g, still in unrecovered:
        kind = "TIFF only" if all(s.endswith(".tif") for s in still) else "JPG and/or TIFF"
        tif_only &= all(s.endswith(".tif") for s in still)
        print(f"    {g['folder']}/{g['prefix']}  ({kind}): {', '.join(still)}")
        if g["example"]:
            print(f"      ^ this is a SELECTED EXAMPLE — a figure row depends on it")
    if tif_only:
        print("\n  ACTIONABLE: HPA still serves the JPGs for withdrawn antibodies but returns"
              "\n    HTTP 400 for `<prefix>_<channel>.tif.gz`. The 16-bit TIFF arm is the"
              "\n    control the whole comparison rests on, so these fields of view have to be"
              "\n    carried in the repository or restored from an internal archive; they will"
              "\n    not come back from proteinatlas.org. State the origin in PIPELINE.md.")
    print("\n  step 6 would SKIP these tiles and still exit 0 — that is why this gate exists.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
