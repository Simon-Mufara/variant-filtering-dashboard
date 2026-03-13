"""VCF input validation — called before parsing to surface clear errors."""
from __future__ import annotations
import re
from typing import Tuple

MAX_FILE_SIZE_MB = 10240
_VCF_HEADER_RE = re.compile(r"^##fileformat=VCFv", re.IGNORECASE)
_CHROM_HEADER_RE = re.compile(r"^#CHROM", re.IGNORECASE)


def validate_vcf(vcf_file) -> Tuple[bool, str]:
    """Validate an uploaded or path-based VCF file before full parsing.

    Returns:
        (True, "")            — file looks valid, proceed
        (False, error_msg)    — describe the problem to the user
    """
    try:
        # ── 1. File size ──────────────────────────────────────────────────────
        if hasattr(vcf_file, "size"):
            size_mb = vcf_file.size / (1024 ** 2)
            if size_mb > MAX_FILE_SIZE_MB:
                return False, (
                    f"File is {size_mb:.0f} MB — maximum allowed is {MAX_FILE_SIZE_MB} MB. "
                    "Split the VCF by chromosome for large datasets."
                )

        # ── 2. Read first 64 KB to inspect headers ────────────────────────────
        raw = _peek(vcf_file, 65_536)
        lines = raw.splitlines()

        if not lines:
            return False, "File appears to be empty."

        first_non_empty = next((ln for ln in lines if ln.strip()), "")
        if not first_non_empty.startswith("##"):
            return False, (
                "File does not start with VCF meta-information lines (##). "
                "Make sure this is a valid VCF file."
            )

        has_fileformat = any(_VCF_HEADER_RE.match(ln) for ln in lines[:20])
        if not has_fileformat:
            return False, (
                "Missing '##fileformat=VCFv4.x' header. "
                "The file may not be a valid VCF."
            )

        has_chrom = any(_CHROM_HEADER_RE.match(ln) for ln in lines)
        if not has_chrom:
            return False, (
                "No '#CHROM' column header found. "
                "The VCF is missing its column-header line."
            )

        # ── 3. Spot-check first data line ─────────────────────────────────────
        for line in lines:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                parts = line.split()
            if len(parts) < 5:
                return False, (
                    f"First data line has only {len(parts)} columns "
                    "(expected ≥8 for a valid VCF)."
                )
            break

        return True, ""

    except Exception as exc:
        return False, f"Could not read file: {exc}"


# ── helpers ───────────────────────────────────────────────────────────────────

def _peek(vcf_file, n_bytes: int) -> str:
    """Read the first n_bytes (decompressed) without consuming the stream."""
    import gzip
    import io as _io

    if hasattr(vcf_file, "read"):
        raw = vcf_file.read(n_bytes)
        if hasattr(vcf_file, "seek"):
            vcf_file.seek(0)

        # Detect gzip by magic bytes — decompress using streaming GzipFile
        # (gzip.decompress() requires the *complete* stream; GzipFile handles partial)
        if isinstance(raw, bytes) and raw[:2] == b'\x1f\x8b':
            try:
                # Read the full compressed stream if seekable
                if hasattr(vcf_file, "read") and hasattr(vcf_file, "seek"):
                    compressed = vcf_file.read()
                    vcf_file.seek(0)
                else:
                    compressed = raw
                with gzip.GzipFile(fileobj=_io.BytesIO(compressed)) as gz:
                    return gz.read(n_bytes).decode("utf-8", errors="replace")
            except (OSError, EOFError):
                pass

        return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw

    path = str(vcf_file)
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            return f.read(n_bytes)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read(n_bytes)
