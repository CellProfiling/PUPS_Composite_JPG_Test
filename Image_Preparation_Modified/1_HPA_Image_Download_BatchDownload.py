# %%

#!/usr/bin/env python3
"""
HPA Batch Image Downloader
Downloads images from the Human Protein Atlas for multiple proteins and cell lines from a file.
Now with file existence checking and automatic retry for missing files.
"""

# =============================================================================================
# MODIFIED COPY -- NOT the companion project's file. Do not sync this back upstream.
# =============================================================================================
# Original: cellprofiling/HPA_BleedThrough_Exploration @ 6225801e46409f9b8ab9e9274656c9dd25a5c4ad
#           1_HPA_Image_Download_BatchDownload.py   (MIT)
# Pristine copy kept alongside at ../Image_Preparation/ , which is never edited.
#
# WHY A LOCAL FORK RATHER THAN AN UPSTREAM CHANGE
# -----------------------------------------------
# This need is specific to reproducing PUPS's published figures, where a named field of view
# either exists or the figure row is wrong. The companion project browses whatever HPA
# currently publishes, so an archived-release fallback would be noise there. Keeping the change
# here keeps that project's behaviour unsurprising.
#
# WHAT WAS CHANGED, AND WHY
# -------------------------
# `download_xml()` fetched https://www.proteinatlas.org/<ENSG>.xml -- the current release, with
# no fallback -- and `parse_xml_for_images_with_antibody()` took every image URL from that one
# document. When HPA withdraws an antibody's immunofluorescence data, the antibody is either
# absent or present with zero <imageUrl> entries, so the loop finds nothing for it, prints no
# line naming it, and the run EXITS 0. There is no notion of an expected set of fields of view,
# so nothing can detect the gap.
#
# Confirmed 2026-09-05: N4BP2 antibody HPA036770 is listed in v25 with zero ICC/IF images, and
# likewise in v19-v23. Only v18 still lists its U2OS fields of view 410_D12_1 and 410_D12_2 --
# the ED Fig 5b N4BP2-201 (Holdout 2) row.
#
# Every change below is tagged `CHANGED-BY-US`. Eleven edits in seven places:
#   1. HPA_ARCHIVE_VERSIONS / ARCHIVE_WALK_ALL_ANTIBODIES / HTTP_TIMEOUT   new constants
#   2. normalize_cell_line()                                              new
#   3. download_xml(..., hpa_version=None)                                new optional argument
#   4. parse_xml_for_images_with_antibody()                               tolerant cell-line match
#   5. list_antibodies_missing_if()                                       new
#   6. extract_base_url()                                                 canonicalise archived URLs
#   7. process_protein() / main()                                         the walk, and a real
#                                                                         exit status
# `0001-archived-release-fallback.patch` in this folder is the mechanical audit: applying it to
# the pristine clone's file reproduces this file byte-for-byte, which
# `4_Run_Image_Preparation.py` asserts on every run. README.md explains each hunk.
#
# NOT changed, deliberately: download_file()'s timeout=30 / no retry / bare-False return, and
# the 13 bare `except:` clauses. Both are real defects, both are out of scope here.
# =============================================================================================

import os
import re
import sys
import requests
import gzip
import shutil
import csv
from pathlib import Path
from lxml import etree
from urllib.parse import urlparse


# CHANGED-BY-US (1/7): new module constants.
# HPA archived release hosts, newest first. `www` is the current release; these are consulted
# only when the current release lists no ICC/IF images for an antibody, which happens when HPA
# withdraws the immunofluorescence data for it. v24/v25 do not exist as separate hosts.
HPA_ARCHIVE_VERSIONS = ['v23', 'v22', 'v21', 'v20', 'v19', 'v18', 'v17', 'v16', 'v15']

# By default the archived releases are consulted only for antibodies that the current release
# lists with no ICC/IF images at all -- the signature of withdrawn immunofluorescence data. Set
# this to True to consult them for any antibody with no images in the requested cell line, which
# recovers a per-cell-line withdrawal too but costs up to len(HPA_ARCHIVE_VERSIONS) extra XML
# fetches for every gene that has an antibody never imaged in that cell line.
ARCHIVE_WALK_ALL_ANTIBODIES = False

# Unknown subdomains do not refuse the connection behind a proxy, they accept it and hang, so
# every HTTP call needs an explicit timeout. (connect, read) seconds.
HTTP_TIMEOUT = (10, 60)


# CHANGED-BY-US (2/7): new helper. Cell line spellings are not stable between HPA
#   releases, so a download list written against the current one matches nothing on an
#   archived one unless the comparison is normalised.
def normalize_cell_line(name):
    """
    Normalise a cell line name for comparison across HPA releases.

    Cell line spellings are not stable between releases: 'U2OS' was 'U-2 OS' up to v20,
    'U-251MG' was 'U-251 MG', 'Rh30' was 'RH-30'. Comparing on alphanumerics only makes a
    download list written against the current release work against archived releases too.

    Args:
        name: Cell line name as written anywhere (list file or XML)

    Returns:
        str: Uppercase alphanumerics only (e.g., 'U-2 OS' -> 'U2OS')
    """
    return re.sub(r'[^A-Z0-9]', '', (name or '').upper())


def extract_protein_info(url):
    """
    Extract Ensembl ID and protein name from HPA URL.
    
    Args:
        url: HPA URL like https://www.proteinatlas.org/ENSG00000117519-CNN3/subcellular
    
    Returns:
        tuple: (ensembl_id, protein_name)
    """
    # Match pattern like ENSG00000117519-CNN3
    pattern = r'(ENSG\d+)-([A-Z0-9]+)'
    match = re.search(pattern, url)
    
    if not match:
        raise ValueError(f"Could not extract protein info from URL: {url}")
    
    ensembl_id = match.group(1)
    protein_name = match.group(2)
    
    return ensembl_id, protein_name


def create_output_folders(protein_name, antibody_id, cell_line):
    """
    Create nested folder structure: GeneName/AntibodyName/CellLine
    
    Args:
        protein_name: Name of the protein
        antibody_id: Antibody identifier (e.g., HPA051237)
        cell_line: Name of the cell line
    
    Returns:
        Path: Path to the cell line folder
    """
    script_dir = Path(__file__).parent
    # Sanitize cell line name for folder creation
    safe_cell_line = sanitize_folder_name(cell_line)
    output_dir = script_dir / protein_name / antibody_id / safe_cell_line
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir

def sanitize_folder_name(name):
    """
    Sanitize folder name by replacing problematic characters.
    
    Args:
        name: Original folder name
    
    Returns:
        str: Sanitized folder name safe for file systems
    """
    # Replace forward slash with underscore
    sanitized = name.replace('/', '_')
    # Replace backslash with underscore
    sanitized = sanitized.replace('\\', '_')
    # Replace other potentially problematic characters
    sanitized = sanitized.replace(':', '_')
    sanitized = sanitized.replace('*', '_')
    sanitized = sanitized.replace('?', '_')
    sanitized = sanitized.replace('"', '_')
    sanitized = sanitized.replace('<', '_')
    sanitized = sanitized.replace('>', '_')
    sanitized = sanitized.replace('|', '_')
    return sanitized

def clean_image_prefix(filename):
    """
    Clean image prefix by removing hash suffixes.
    Keeps only the standard pattern: Number_Well_Number
    
    Args:
        filename: Original filename (e.g., "1878_H7_8_cr5bb224b655898")
    
    Returns:
        str: Cleaned filename (e.g., "1878_H7_8")
    """
    # Pattern: Number_Letter+Number_Number (e.g., 1878_H7_8)
    # This removes anything after the third underscore-separated part
    parts = filename.split('_')
    
    # If we have more than 3 parts, it likely has a hash suffix
    if len(parts) >= 3:
        # Take first 3 parts: Number_Well_Number
        cleaned = '_'.join(parts[:3])
        
        # Verify it matches expected pattern (optional validation)
        # Pattern: digits_letters+digits_digits
        import re
        if re.match(r'^\d+_[A-Z]+\d+_\d+$', cleaned):
            return cleaned
        else:
            # If pattern doesn't match, return original
            return filename
    
    # If less than 3 parts, return as-is
    return filename

def create_annotation_file(protein_name, note):
    """
    Create anno.txt file in the gene folder with the note.
    
    Args:
        protein_name: Name of the protein
        note: Annotation text to write
    """
    script_dir = Path(__file__).parent
    protein_dir = script_dir / protein_name
    protein_dir.mkdir(parents=True, exist_ok=True)
    
    anno_path = protein_dir / "anno.txt"
    
    # Only create if doesn't exist
    if not anno_path.exists():
        with open(anno_path, 'w', encoding='utf-8') as f:
            f.write(note)
        print(f"Annotation saved to: {anno_path}")
    else:
        print(f"Annotation already exists: {anno_path}")


# CHANGED-BY-US (3/7): `hpa_version` added (None == the original behaviour exactly), and
#   the request given a timeout -- it had none, and an unknown proteinatlas.org subdomain
#   behind a proxy accepts the connection and hangs rather than refusing it.
def download_xml(ensembl_id, protein_name, hpa_version=None):
    """
    Download the protein XML file from HPA.

    Args:
        ensembl_id: Ensembl gene identifier
        protein_name: Name of the protein (for folder structure)
        hpa_version: Archived release host such as 'v18', or None for the current release.
                     Archived XMLs are cached under their own filename so they never shadow
                     the current-release one.

    Returns:
        Path: Path to the downloaded XML file
    """
    host = 'www' if hpa_version is None else hpa_version
    xml_url = f"https://{host}.proteinatlas.org/{ensembl_id}.xml"

    # Save XML at the protein level (Gene name folder)
    script_dir = Path(__file__).parent
    protein_dir = script_dir / protein_name
    protein_dir.mkdir(parents=True, exist_ok=True)
    xml_path = protein_dir / (f"{ensembl_id}.xml" if hpa_version is None
                              else f"{ensembl_id}.{hpa_version}.xml")

    # Check if XML already exists
    if xml_path.exists():
        print(f"XML already exists: {xml_path}")
        return xml_path

    print(f"Downloading XML from: {xml_url}")
    response = requests.get(xml_url, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    
    with open(xml_path, 'wb') as f:
        f.write(response.content)
    
    print(f"XML saved to: {xml_path}")
    return xml_path


def parse_xml_for_images_with_antibody(xml_path, cell_line):
    """
    Parse XML to extract image URLs and antibody IDs for the specified cell line.
    
    Args:
        xml_path: Path to the XML file
        cell_line: Cell line to search for (e.g., 'U2OS')
    
    Returns:
        dict: Dictionary mapping antibody IDs to lists of image URLs
              e.g., {'HPA051237': ['url1', 'url2']}
    """
    tree = etree.parse(str(xml_path))
    root = tree.getroot()
    
    # Define namespace if present
    namespaces = root.nsmap
    if None in namespaces:
        namespaces['default'] = namespaces[None]
        del namespaces[None]
    
    antibody_images = {}
    
    # Find all antibody elements
    antibody_elements = root.xpath('.//antibody', namespaces=namespaces if namespaces else None)
    
    # CHANGED-BY-US (4/7): the cell line was matched by literal XPath equality,
    #   `data[cellLine='{cell_line}']`. Now the data elements are iterated and compared
    #   through normalize_cell_line(), so archived releases match too. Same result on the
    #   current release. Also removes an unescaped f-string interpolation into XPath.
    wanted = normalize_cell_line(cell_line)

    for antibody in antibody_elements:
        antibody_id = antibody.get('id')

        # The cell line is matched in Python rather than with `data[cellLine='...']` so that a
        # release which spells it differently ('U-2 OS' for 'U2OS') still matches. See
        # normalize_cell_line().
        image_urls = []
        for data in antibody.xpath(".//cellExpression[@technology='ICC/IF']/subAssay/data",
                                   namespaces=namespaces if namespaces else None):
            names = [t.strip() for t in data.xpath('./cellLine/text()')]
            if not any(normalize_cell_line(n) == wanted for n in names):
                continue
            image_urls += [elem.text.strip() for elem in data.xpath(
                "./assayImage/image[@imageType='sampleImage']/imageUrl")]

        if image_urls:
            antibody_images[antibody_id] = image_urls
            print(f"  Found {len(image_urls)} images for antibody {antibody_id} in {cell_line}")
    
    if not antibody_images:
        print(f"  Warning: No images found for cell line '{cell_line}'")
    else:
        total_images = sum(len(urls) for urls in antibody_images.values())
        print(f"  Total: Found {total_images} images across {len(antibody_images)} antibodies for {cell_line}")
    
    return antibody_images


# CHANGED-BY-US (5/7): new helper -- the trigger for the walk below. It must be per
#   ANTIBODY, not per gene: N4BP2 keeps two live U2OS antibodies alongside the withdrawn
#   one, so a `if not antibody_images:` test never fires on the case it is meant for.
def list_antibodies_missing_if(xml_path):
    """
    Antibody ids the XML lists with NO ICC/IF sample images in any cell line.

    That is the signature of withdrawn immunofluorescence data: HPA keeps the antibody in the
    record but its images are gone, so parse_xml_for_images_with_antibody() cannot see them and
    the run would report success without them.

    Restricting the archived-release walk to these keeps it cheap. An antibody that simply was
    never imaged in the requested cell line still has ICC/IF images for other cell lines, so it
    is not reported here and does not trigger a walk that could never succeed. Set
    ARCHIVE_WALK_ALL_ANTIBODIES to widen this at the cost of up to len(HPA_ARCHIVE_VERSIONS)
    extra XML fetches per gene.

    Args:
        xml_path: Path to the XML file

    Returns:
        list: Antibody ids in document order (e.g., ['HPA036770'])
    """
    root = etree.parse(str(xml_path)).getroot()
    out = []
    for ab in root.xpath('.//antibody'):
        if not ab.get('id'):
            continue
        images = ab.xpath(".//cellExpression[@technology='ICC/IF']"
                          "//image[@imageType='sampleImage']/imageUrl")
        if ARCHIVE_WALK_ALL_ANTIBODIES or not images:
            out.append(ab.get('id'))
    return out


def extract_base_url(image_url):
    """
    Extract base URL from image URL by removing the file extension.
    
    Args:
        image_url: Full image URL (e.g., https://images.proteinatlas.org/9849/1001_C9_1_blue_red_green.jpg)
    
    Returns:
        str: Base URL without extension (e.g., https://images.proteinatlas.org/9849/1001_C9_1)
    """
    # Remove _blue_red_green.jpg or similar suffix
    base = re.sub(r'_blue_red_green\.jpg$', '', image_url)

    # CHANGED-BY-US (6/7): rewrite an archived release's own host path to the
    #   version-independent image host the current release already uses.
    # An archived release serves its own host path, e.g.
    #     http://v18.proteinatlas.org/images/36770/410_D12_1
    # Rewrite it to the version-independent image host used by the current release, so there is
    # one code path and the URL does not go stale when that release host is retired:
    #     https://images.proteinatlas.org/36770/410_D12_1
    base = re.sub(r'^https?://v\d+\.proteinatlas\.org/images/',
                  'https://images.proteinatlas.org/', base)
    return base.replace('http://', 'https://')


def download_file(url, output_path, skip_if_exists=True):
    """
    Download a file from URL to output path.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        skip_if_exists: If True, skip download if file already exists
    
    Returns:
        bool: True if successful or already exists, False otherwise
    """
    # Check if file already exists
    if skip_if_exists and output_path.exists():
        return True
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        return True
    except requests.exceptions.RequestException as e:
        print(f"    Failed to download {url}: {e}")
        return False


def decompress_gz(gz_path):
    """
    Decompress a .gz file.
    
    Args:
        gz_path: Path to the .gz file
    
    Returns:
        Path: Path to the decompressed file
    """
    output_path = Path(str(gz_path).replace('.gz', ''))
    
    # Check if decompressed file already exists
    if output_path.exists():
        # Remove the .gz file if decompressed version exists
        if gz_path.exists():
            gz_path.unlink()
        return output_path
    
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Remove the .gz file after decompression
    gz_path.unlink()
    
    return output_path


def download_image_channels(base_url, output_dir, image_index, skip_existing=True):
    """
    Download all channel images (JPG and TIF) for a single image.
    
    Args:
        base_url: Base URL without extension
        output_dir: Directory to save files
        image_index: Index of the image (for numbering)
        skip_existing: If True, skip files that already exist
    
    Returns:
        dict: Dictionary with success status for each file type
    """
    channels = ['blue', 'red', 'green', 'yellow']
    base_filename = Path(base_url).name
    
    # Clean the filename to remove hash suffixes
    cleaned_filename = clean_image_prefix(base_filename)
    
    # If filename has hash suffix, use cleaned version for URLs
    if cleaned_filename != base_filename:
        print(f"  Image {image_index}: Detected hash suffix in {base_filename}")
        print(f"    Using cleaned filename: {cleaned_filename}")
        
        # Reconstruct base URL with cleaned filename
        base_url_parts = base_url.rsplit('/', 1)
        if len(base_url_parts) == 2:
            base_url = f"{base_url_parts[0]}/{cleaned_filename}"
        
        # Use cleaned filename for saving
        base_filename = cleaned_filename
    
    # Skip images that start with 'si' (these are not standard images but only siRNA images)
    if base_filename.lower().startswith('si'):
        print(f"  Skipping image {image_index}: {base_filename} (starts with 'si')")
        return {'skipped': True, 'reason': 'si_prefix'}
    
    print(f"  Downloading image {image_index}: {base_filename}")
    
    download_status = {}
    
    # Download composite JPG (3 channels)
    composite_url = f"{base_url}_blue_red_green.jpg"
    composite_path = output_dir / f"{base_filename}_blue_red_green.jpg"
    if skip_existing and composite_path.exists():
        print(f"    Skipping existing: {composite_path.name}")
    else:
        download_file(composite_url, composite_path, skip_if_exists=False)
    download_status['composite_3ch'] = composite_path.exists()

    # Download composite JPG (4 channels)
    composite_url = f"{base_url}_blue_red_green_yellow.jpg"
    composite_path = output_dir / f"{base_filename}_blue_red_green_yellow.jpg"
    if skip_existing and composite_path.exists():
        print(f"    Skipping existing: {composite_path.name}")
    else:
        download_file(composite_url, composite_path, skip_if_exists=False)
    download_status['composite_4ch'] = composite_path.exists()
    
    # Download individual channel JPGs
    for channel in channels:
        jpg_url = f"{base_url}_{channel}.jpg"
        jpg_path = output_dir / f"{base_filename}_{channel}.jpg"
        if skip_existing and jpg_path.exists():
            print(f"    Skipping existing: {jpg_path.name}")
        else:
            download_file(jpg_url, jpg_path, skip_if_exists=False)
        download_status[f'{channel}_jpg'] = jpg_path.exists()
    
    # Download and decompress individual channel TIFs
    for channel in channels:
        tif_path = output_dir / f"{base_filename}_{channel}.tif"
        
        # Check if TIF already exists
        if tif_path.exists():
            print(f"    Skipping existing: {tif_path.name}")
            download_status[f'{channel}_tif'] = True
            continue
        
        # Try to download TIF
        tif_url = f"{base_url}_{channel}.tif.gz"
        tif_gz_path = output_dir / f"{base_filename}_{channel}.tif.gz"
        
        if download_file(tif_url, tif_gz_path, skip_if_exists=False):
            decompress_gz(tif_gz_path)
            download_status[f'{channel}_tif'] = True
        else:
            download_status[f'{channel}_tif'] = False
    
    return download_status


def retry_missing_tifs(base_url, output_dir, base_filename, missing_channels):
    """
    Retry downloading missing TIF files with alternative methods.
    
    Args:
        base_url: Base URL without extension
        output_dir: Directory to save files
        base_filename: Base filename (already cleaned in download_image_channels)
        missing_channels: List of channel names to retry
    
    Returns:
        dict: Dictionary with success status for retried channels
    """
    print(f"  Retrying missing TIFs for: {base_filename}")
    retry_status = {}
    
    
    for channel in missing_channels:
        tif_path = output_dir / f"{base_filename}_{channel}.tif"
        
        if tif_path.exists():
            retry_status[channel] = True
            continue
        
        # Try .gz URL (base_url is already cleaned)
        tif_gz_url = f"{base_url}_{channel}.tif.gz"
        print(f"    Trying .gz URL: {channel}")
        
        tif_gz_path = output_dir / f"{base_filename}_{channel}.tif.gz"
        if download_file(tif_gz_url, tif_gz_path, skip_if_exists=False):
            decompress_gz(tif_gz_path)
            retry_status[channel] = tif_path.exists()
            if retry_status[channel]:
                print(f"    ✓ Successfully downloaded {channel}.tif")
                continue
        
        # Try direct TIF URL
        tif_url = f"{base_url}_{channel}.tif"
        print(f"    Trying direct TIF: {channel}")
        if download_file(tif_url, tif_path, skip_if_exists=False):
            retry_status[channel] = True
            print(f"    ✓ Successfully downloaded {channel}.tif")
            continue
        
        retry_status[channel] = False
        print(f"    ✗ Could not download {channel}.tif")
    
    return retry_status


def process_protein(note, url, cell_line, protein_num, total_proteins):
    """
    Process a single protein entry.
    
    Args:
        note: Annotation note for the protein
        url: HPA URL
        cell_line: Cell line to download
        protein_num: Current protein number
        total_proteins: Total number of proteins to process
    """
    print("\n" + "=" * 70)
    print(f"Processing {protein_num}/{total_proteins}")
    print("=" * 70)
    
    try:
        # Extract protein information
        ensembl_id, protein_name = extract_protein_info(url)
        print(f"Protein: {protein_name}")
        print(f"Ensembl ID: {ensembl_id}")
        print(f"Cell Line: {cell_line}")
        print(f"Note: {note}")
        
        # Create annotation file
        create_annotation_file(protein_name, note)
        
        # Download XML (saved at protein level)
        xml_path = download_xml(ensembl_id, protein_name)

        # Parse XML for image URLs organized by antibody
        antibody_images = parse_xml_for_images_with_antibody(xml_path, cell_line)

        # CHANGED-BY-US (7/7): the fallback itself, plus process_protein() now returning a
        #   bool so main() can tally skipped entries and exit non-zero. Previously a
        #   skipped entry printed a line and the batch still finished with status 0.
        # An antibody that the current release still LISTS but has no ICC/IF images for is the
        # signature of withdrawn immunofluorescence data: the images are gone from this XML, so
        # nothing here can see them and the run would report success without them. Consult the
        # archived releases for exactly those antibodies, newest release first.
        #
        # Note the test is per antibody, not per gene: a gene can keep several live antibodies
        # while one of them is withdrawn, which is the common case and would be missed by a
        # `if not antibody_images` check.
        pending = [a for a in list_antibodies_missing_if(xml_path)
                   if a not in antibody_images]
        if pending:
            print(f"  Listed with no {cell_line} ICC/IF images in the current release: "
                  f"{', '.join(pending)}")
            print(f"  Checking archived releases for them")
            for hpa_version in HPA_ARCHIVE_VERSIONS:
                if not pending:
                    break
                try:
                    archived_xml = download_xml(ensembl_id, protein_name,
                                                hpa_version=hpa_version)
                    archived = parse_xml_for_images_with_antibody(archived_xml, cell_line)
                except requests.exceptions.RequestException as e:
                    print(f"    {hpa_version}: {e}")
                    continue
                except etree.XMLSyntaxError as e:
                    print(f"    {hpa_version}: unparseable response ({e})")
                    continue
                recovered = [a for a in pending if a in archived]
                for antibody_id in recovered:
                    antibody_images[antibody_id] = archived[antibody_id]
                    print(f"    {hpa_version}: recovered {len(archived[antibody_id])} "
                          f"image URL(s) for {antibody_id}")
                if recovered:
                    print(f"    NOTE: HPA keeps serving the JPGs for a withdrawn antibody but "
                          f"returns HTTP 400 for <prefix>_<channel>.tif.gz, so expect the "
                          f"16-bit TIFFs to be unavailable for these fields of view.")
                pending = [a for a in pending if a not in recovered]
            if pending:
                print(f"  Not found in any archived release: {', '.join(pending)} "
                      f"(most likely never imaged in {cell_line})")

        if not antibody_images:
            print(f"No images found for {protein_name} in {cell_line} in the current release "
                  f"or any of {len(HPA_ARCHIVE_VERSIONS)} archived releases. Skipping.")
            return False
        
        # Download all images organized by antibody
        for antibody_id, image_urls in antibody_images.items():
            print(f"\n  Processing Antibody: {antibody_id}")
            
            # Create output folder for this antibody
            output_dir = create_output_folders(protein_name, antibody_id, cell_line)
            print(f"  Output directory: {output_dir}")
            
            # Download all images for this antibody
            for idx, image_url in enumerate(image_urls, 1):
                base_url = extract_base_url(image_url)
                base_filename = Path(base_url).name
                
                # Download all channels
                download_status = download_image_channels(base_url, output_dir, idx, skip_existing=True)

                # Safety check: ensure download_status is a dictionary
                if download_status is None:
                    print(f"    Warning: download_image_channels returned None for {base_filename}")
                    download_status = {'error': True}
                    continue

                # Skip retry logic if image was skipped (e.g., si* images, which are not standard HPA images)
                if download_status.get('skipped', False):
                    continue

                # Check for missing TIFs
                channels = ['blue', 'red', 'green', 'yellow']
                missing_tifs = [ch for ch in channels if not download_status.get(f'{ch}_tif', False)]

                # Retry missing TIFs
                if missing_tifs:
                    print(f"  Found {len(missing_tifs)} missing TIF(s): {', '.join(missing_tifs)}")
                    retry_status = retry_missing_tifs(base_url, output_dir, base_filename, missing_tifs)
                    
                    # Update download status
                    for ch in missing_tifs:
                        if retry_status.get(ch, False):
                            print(f"    ✓ Successfully downloaded {ch}.tif on retry")
                        else:
                            print(f"    ✗ Failed to download {ch}.tif after retry")
        
        print(f"\n✓ Completed {protein_name}")
        return True

    except Exception as e:
        print(f"\n✗ Error processing {url}: {e}")
        import traceback
        traceback.print_exc()
        return False


def read_download_list(file_path):
    """
    Read the Organelle_Example_GenesAndCellLines.txt file.
    
    Args:
        file_path: Path to the file
    
    Returns:
        list: List of tuples (note, link, cell_line)
    """
    entries = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            note = row['Note'].strip()
            link = row['Link'].strip()
            cell_line = row['CellLine'].strip()
            entries.append((note, link, cell_line))
    
    return entries


def main():
    """
    Main function to orchestrate the batch download process.
    """
    print("=" * 70)
    print("HPA Batch Image Downloader")
    print("=" * 70)
    
    # Path to the download list file
    script_dir = Path(__file__).parent
    download_list_path = script_dir / "Organelle_Example_GenesAndCellLines.txt"
    
    if not download_list_path.exists():
        print(f"\nError: Could not find {download_list_path}")
        print("Please ensure Organelle_Example_GenesAndCellLines.txt is in the same directory as this script.")
        return 1
    
    try:
        # Read the download list
        print(f"\nReading download list from: {download_list_path}")
        entries = read_download_list(download_list_path)
        print(f"Found {len(entries)} entries to process")
        
        # Process each entry. Failures are collected rather than ignored: previously every
        # skipped entry printed a line and the run still finished with status 0, so a batch
        # could lose whole genes without anything downstream noticing.
        failed = []
        for idx, (note, link, cell_line) in enumerate(entries, 1):
            if not process_protein(note, link, cell_line, idx, len(entries)):
                failed.append((note, cell_line))

        print("\n" + "=" * 70)
        if failed:
            print(f"DOWNLOADS FINISHED WITH {len(failed)} OF {len(entries)} ENTRIES SKIPPED")
            for note, cell_line in failed:
                print(f"  - {note}  ({cell_line})")
            print(f"Files saved to: {script_dir}")
            print("=" * 70)
            return 1
        print("ALL DOWNLOADS COMPLETE!")
        print(f"Files saved to: {script_dir}")
        print("=" * 70)
        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Propagate the status so a batch that skipped entries fails a pipeline step instead of
    # looking like a clean run.
    sys.exit(main())