# `Image_Preparation_Modified/` — our marked fork of one companion-project tool

```
1_HPA_Image_Download_BatchDownload.py   forked from the clone; falls back to archived HPA releases
0001-archived-release-fallback.patch    the audit: clone's file + this patch == the file above
README.md                               this file
```

`Image_Preparation/` is a pristine clone of
[cellprofiling/HPA_BleedThrough_Exploration](https://github.com/cellprofiling/HPA_BleedThrough_Exploration)
at `6225801e` and is **never edited** — `git -C Image_Preparation status --porcelain` must stay
empty. Exactly one of its eight scripts needs a change for this project, so that one is forked
here rather than upstream.

**This is deliberately not an upstream change.** The need is specific to reproducing PUPS's
published figures, where a *named* field of view either exists or the figure row is wrong. The
companion project browses whatever HPA currently publishes, so an archived-release fallback
would be noise there and would make its behaviour harder to predict. Do not sync this file back.

`4_Run_Image_Preparation.py --tool download` runs the fork instead of the clone's copy, and
audits it first: `verify_fork()` applies the patch to a pristine copy of the clone's file and
requires the result to equal ours **byte-for-byte**, exiting with instructions otherwise. Edit
the script without regenerating the patch and the next run stops. Regenerate with:

```bash
diff -u Image_Preparation/1_HPA_Image_Download_BatchDownload.py \
        Image_Preparation_Modified/1_HPA_Image_Download_BatchDownload.py \
  > Image_Preparation_Modified/0001-archived-release-fallback.patch
```

Every changed region in the script carries a `CHANGED-BY-US (n/7)` comment, and the file opens
with a header block naming the origin commit and summarising the seven sites. `grep -n
CHANGED-BY-US` is the quick tour; this README is the detailed one.

Step `4b_Verify_Track_Images.py` still runs after the download. It overlaps with the fork on
purpose: the fork recovers per *antibody* from the download list, while 4b checks per *field of
view* against `ROI.txt` and is what actually fails the pipeline when a named tile is absent —
including for images downloaded before this fork existed.

---

## The problem

`download_xml()` fetches `https://www.proteinatlas.org/<ENSG>.xml` — the current release, with no
fallback — and `parse_xml_for_images_with_antibody()` takes every image URL from that one
document. When HPA withdraws an antibody's immunofluorescence data, that antibody is either
absent from the XML or present with zero `<imageUrl>` entries. The loop then finds nothing for
it, prints no line naming it, and the run **exits 0**. There is no notion of an *expected* set of
fields of view, so nothing can detect the gap.

Confirmed 2026-09-05: **N4BP2 antibody `HPA036770`** is listed in v25 with zero ICC/IF images,
and likewise in v19–v23. Only **v18** still lists its U2OS fields of view `410_D12_1` and
`410_D12_2`.

Refetching from v18 recovers all twelve JPGs **byte-identical** (sha256) to copies downloaded
when they were still current. The TIFFs do not come back: `<prefix>_<channel>.tif.gz` redirects
to `www.proteinatlas.org/download_file.php`, which answers **HTTP 400** on every release host
and on `images.proteinatlas.org`. HPA keeps serving the lossy JPGs for a withdrawn antibody and
drops the 16-bit originals.

---

## The changes, in order

### 1. `HPA_ARCHIVE_VERSIONS`, `ARCHIVE_WALK_ALL_ANTIBODIES`, `HTTP_TIMEOUT` (new constants)

`HPA_ARCHIVE_VERSIONS = ['v23' … 'v15']`, newest first. `v24`/`v25` do not exist as separate
hosts — the newest release is served from `www`.

`ARCHIVE_WALK_ALL_ANTIBODIES = False`. Controls how eagerly the walk fires; see change 6.

`HTTP_TIMEOUT = (10, 60)`. `download_xml()` previously called `requests.get()` with **no
timeout**; an unknown `proteinatlas.org` subdomain behind a proxy does not refuse the connection,
it accepts and hangs, so a bad host could stall the batch indefinitely.

### 2. `normalize_cell_line(name)` (new function)

`re.sub(r'[^A-Z0-9]', '', name.upper())`. Cell-line spellings are not stable between releases:
`U2OS` was `U-2 OS` up to v20, `U-251MG` was `U-251 MG`, `Rh30` was `RH-30`. Without this, a
download list written against the current release silently matches nothing on an archived one.

### 3. `download_xml(ensembl_id, protein_name, hpa_version=None)`

New optional third argument. `None` keeps the existing behaviour exactly (`www`,
`<ENSG>.xml`); a version string fetches `https://<version>.proteinatlas.org/<ENSG>.xml` and
caches it as `<ENSG>.<version>.xml`, so an archived document never shadows the current one.
Callers that pass two arguments are unaffected.

### 4. `parse_xml_for_images_with_antibody()` — match the cell line in Python

The old XPath embedded the name as a literal equality test:

```python
xpath_query = f".//cellExpression[@technology='ICC/IF']/subAssay/data[cellLine='{cell_line}']/assayImage/image[@imageType='sampleImage']/imageUrl"
```

Now the `data` elements are iterated and compared through `normalize_cell_line()`. Same result on
the current release; works on archived ones too. *(It also removes an unescaped f-string
interpolation into an XPath expression.)*

### 5. `list_antibodies_missing_if()` (new) and the fallback in `process_protein()`

**The trigger has to be per antibody, not per gene.** N4BP2 keeps two live antibodies in U2OS
(`HPA042607`, `HPA072549`) alongside the withdrawn `HPA036770`, so `antibody_images` is non-empty
and a `if not antibody_images:` check — the obvious formulation, and the one tried first —
never fires on the very case it is for. It only covers a gene losing *all* its images.

So `list_antibodies_missing_if(xml_path)` returns the antibody ids the XML lists with **no ICC/IF
sample images in any cell line**, which is the withdrawal signature. `process_protein()` then
walks `HPA_ARCHIVE_VERSIONS` newest-first for exactly those, merging recovered URLs into
`antibody_images` per antibody and stopping as soon as none are outstanding.

That default trigger is deliberately narrow so the walk stays cheap: an antibody that was simply
never imaged in the requested cell line still has ICC/IF images for *other* cell lines, so it is
not reported and does not start a walk that could never succeed. Set
`ARCHIVE_WALK_ALL_ANTIBODIES = True` to also chase a per-cell-line withdrawal, at the cost of up
to `len(HPA_ARCHIVE_VERSIONS)` extra XML fetches for every gene that has an antibody unimaged in
that cell line.

`requests.exceptions.RequestException` and `etree.XMLSyntaxError` are caught **per host** (some
archived hosts return non-XML) so one dead release cannot abort the walk. Anything still
outstanding at the end is named, rather than passed over.

On a hit it prints the release and warns that the TIFFs will very likely 400 — the honest signal,
since a JPG-only recovery is not equivalent for anything that needs bit depth.

*Side effect:* archived XMLs are cached next to the current-release one as
`<ENSG>.<version>.xml`, so a re-run is instant. For the N4BP2 test that is 6 extra files, ~5 MB.

### 6. `extract_base_url()` — canonicalise archived URLs

An archived release serves its own host path, `http://v18.proteinatlas.org/images/36770/…`,
whereas the current release uses the version-independent image host,
`http://images.proteinatlas.org/42607/…`. The archived form is rewritten to the latter, and
`http://` upgraded to `https://`. Both forms serve the files today; using the canonical one keeps
a single code path and does not go stale when a release host is retired.

### 7. `process_protein()` returns a bool, `main()` tallies and `sys.exit()`s

This is the part that matters as much as the fallback. Previously every skipped entry printed a
line and the batch still finished with status 0, so a pipeline step could lose whole genes and
look like a clean run. Now `process_protein()` returns `False` when it finds no images or raises,
`main()` collects the failures, prints them as a list at the end, and returns 1; the
`__main__` guard propagates it via `sys.exit(main())` (needs the added `import sys`).

Scope of the exit status, precisely: it flags **entries skipped entirely**. Individual files that
could not be fetched — including the permanently-gone TIFFs of a withdrawn antibody — are still
reported per file and still leave the status at 0, because for a withdrawn antibody they are
never coming back and failing the batch on them would be wrong. Checking a download against an
*expected* list of fields of view is a different job; in this project
`4b_Verify_Track_Images.py` does it.

---

## Verified behaviour

Run from a clean directory with a one-line download list —
`N4BP2-201 (Holdout 2),https://www.proteinatlas.org/ENSG00000078177-N4BP2/subcellular,U2OS`:

```
  Found 2 images for antibody HPA042607 in U2OS
  Found 2 images for antibody HPA072549 in U2OS
  Listed with no U2OS ICC/IF images in the current release: HPA036770
  Checking archived releases for them
    v18: recovered 2 image URL(s) for HPA036770
    NOTE: HPA keeps serving the JPGs for a withdrawn antibody but returns HTTP 400 for
          <prefix>_<channel>.tif.gz, so expect the 16-bit TIFFs to be unavailable ...
```

Result: 3 antibody folders instead of 2. `HPA036770` yields 12 JPGs and 0 TIFFs; the two live
antibodies yield 12 JPGs and 8 TIFFs each. All 12 recovered JPGs are **byte-identical (sha256)**
to copies downloaded while the antibody was still current. No `.tif.gz` partials left behind.
The walk consulted v23→v18 and stopped there; 6 archived XMLs cached.

Without the patch the same run produces 2 antibody folders and exits 0.

---

## Deliberately not changed

- **`download_file()`'s `timeout=30` and lack of retry.** It also returns a bare `False` for
  every failure mode, so a permanent 400 is indistinguishable from a transient timeout. Worth
  fixing, but it is a wider change to that function's contract and independent of this one.
- **The 13 bare `except:` clauses.** Same reasoning — separate cleanup.
- **`2_Square_Selector.py`'s depth-derived label columns.** A real defect (running it one level
  above the images root silently writes `Gene = "Images"` and shifts every column), but a
  different file and a different fix.
